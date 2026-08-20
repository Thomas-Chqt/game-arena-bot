// amoeba_train: improves a network by playing it against itself, forever.
//
//   amoeba_train [weights.safetensors]
//
// With no argument it takes the first .safetensors file in the working directory,
// and starts from random weights if there are none. Whenever a generation beats
// the one before it, the new weights are written back over that same file - which
// is the file amoeba_bot reads, so a bot running alongside picks the improvement
// up at its next game.
//
// One generation is three steps:
//
//   1. play GAMES games of the current best against itself, into a replay buffer
//   2. train a candidate on batches drawn from the buffer
//   3. play the candidate against the current best, and promote only if it wins
//
// Step 3 is the point. A training loss can fall while the player gets worse, and
// AlphaZero bugs produce clean loss curves, so nothing is promoted on a curve -
// only on games won.
//
// The defaults are meant for a real overnight run, not a smoke test. To check
// the wiring quickly instead:
//
//   BLOCKS=2 WIDTH=64 HEADS=4 GAMES=8 SIMULATIONS=50 STEPS=100 GATE_GAMES=8
//
// Environment, all optional:
//   network     BLOCKS WIDTH HEADS          (only read when starting from scratch)
//   generating  SEED GAMES SIMULATIONS LEAVES SAMPLING_PLIES NOISE
//   training    STEPS BATCH RATE DECAY BUFFER
//   gating      GATE_GAMES GATE GATE_SIMULATIONS
//
// STEPS and GATE_GAMES are both caps rather than costs: training stops when the
// held-out loss stops improving, and the gate stops as soon as the verdict is
// settled. Both run to the cap only when the extra work is actually buying
// something.

#include "network.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <print>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace bot
{

namespace
{

// std::println leaves stdout block-buffered whenever it is not a terminal, so a
// redirected run shows nothing at all until the buffer fills - which for short
// progress lines means not until the process exits.
template <typename... Args>
void report(std::format_string<Args...> format, Args&&... args)
{
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
}

int envInt(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr ? fallback : std::stoi(text);
}

float envFloat(const char* name, float fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr ? fallback : std::stof(text);
}

// The defaults are sized for an overnight run on one Mac, not for a smoke test.
// A generation at these settings is expected to take on the order of an hour;
// if it takes much longer, GAMES is the knob to turn down first, because it is
// the only one that trades directly against how many generations a night holds.
struct Settings
{
    uint64_t seed = static_cast<uint64_t>(envInt("SEED", 20260819));

    // Read only when there is no checkpoint to load - otherwise the shape comes
    // out of the file. ~1.2M parameters, which is the size the encoder and the
    // relative-position bias were designed around.
    NetworkShape shape = {.blocks = envInt("BLOCKS", 6),
                          .width  = envInt("WIDTH", 128),
                          .heads  = envInt("HEADS", 8)};

    // 400 simulations is half of AlphaZero's 800, which is the usual trade on one
    // machine: the policy target is a distribution over visits, and doubling the
    // visits sharpens it far less than doubling the games broadens it.
    // 64 leaves per forward pass rather than 16: the GPU is the bottleneck and it
    // costs 0.16 ms/position at 64 against ~0.21 at 16, almost all of it fixed
    // dispatch overhead. The cost is coarser search - only six rounds of "look,
    // learn, redirect" per move at 400 simulations - which virtual loss covers
    // but does not make free.
    Config search = {.simulations = envInt("SIMULATIONS", 400),
                     .batchSize   = envInt("LEAVES", 64)};

    // 200 games is roughly 20k positions. Fewer than that and each generation's
    // training set is mostly the previous generation's, so the gate compares two
    // networks that saw nearly the same data and rejects almost everything.
    int   games         = envInt("GAMES", 200);
    int   samplingPlies = envInt("SAMPLING_PLIES", 20);
    float noise         = envFloat("NOISE", 0.25f);

    int   steps     = envInt("STEPS", 1000);
    int   batchSize = envInt("BATCH", 256);
    float rate      = envFloat("RATE", 1e-3f);
    float decay     = envFloat("DECAY", 1e-4f);

    // About the last ten generations. Older positions came from networks several
    // generations weaker and hold the current one back; keeping none of them at
    // all makes each generation overfit the games it just played. ~440 MB.
    size_t buffer = static_cast<size_t>(envInt("BUFFER", 200000));

    // 200 games puts the gate's error bar at +/-3.5%, which is what it takes for
    // a 55% result to mean anything - at 40 games the bar is +/-8% and promotion
    // is close to a coin flip. It costs as much as the self-play it judges, and
    // that is the price of the only honest signal in the system.
    // 200 is the cap, not the usual cost: the gate stops as soon as the verdict is
    // settled, which for a clearly better candidate is about twenty games. The cap
    // only gets spent when the two networks are genuinely close.
    int   gateGames = envInt("GATE_GAMES", 200);
    float gate      = envFloat("GATE", 0.55f);

    // Ranking two networks needs far less search than generating a training
    // target does: the visit counts are thrown away here, only the result counts.
    int gateSimulations = envInt("GATE_SIMULATIONS", 200);
};

// ---------------------------------------------------------------------------
// Playing games
// ---------------------------------------------------------------------------

struct Sample
{
    amoeba::Board board;
    VisitCounts   visits;
    float         outcome = 0.0f;
    int           game    = 0;   // unique across generations, so whole games can be held out
};

// Proportional to visits early, argmax afterwards. Without the early sampling
// every game walks the same opening and the training set is far narrower than
// its position count suggests.
uint16_t chooseMove(const VisitCounts& counts, int ply, int samplingPlies, std::mt19937_64& rng)
{
    if (ply >= samplingPlies)
        return bestMove(counts);

    uint64_t total = 0;
    for (const uint32_t count : counts) {
        total += count;
    }

    uint64_t remaining = std::uniform_int_distribution<uint64_t>{0, total - 1}(rng);
    for (uint16_t id = 0; id < amoeba::kNumMoveIds; ++id)
    {
        if (counts[id] > remaining)
            return id;
        remaining -= counts[id];
    }
    return bestMove(counts);
}

// `white` and `black` are the same search in self-play and different ones in the
// gate. The samples come back with the game's result already written into them.
std::vector<Sample> playGame(Search& white, Search& black, int samplingPlies, std::mt19937_64& rng)
{
    std::vector<Sample>   samples;
    amoeba::Board         board = amoeba::startPosition();
    std::vector<uint64_t> history{board.hash};

    while (board.state == amoeba::State::Ongoing)
    {
        Search&           search = board.whiteToMove ? white : black;
        const VisitCounts counts = search.run(board, history);
        samples.push_back({board, counts, 0.0f, 0});

        const uint16_t chosen = chooseMove(counts, board.ply, samplingPlies, rng);
        board = amoeba::apply(board, amoeba::Move::fromId(chosen), history);
        history.push_back(board.hash);
    }

    for (Sample& sample : samples) {
        sample.outcome = outcomeFor(board.state, sample.board.whiteToMove);
    }
    return samples;
}

// Games share nothing, so one slot per game means each thread writes where nobody
// else does and no lock is needed. Seeding from the game index rather than the
// thread keeps a whole run reproducible from SEED however the threads interleave.
//
// Concurrent MLX evaluation was tested and behaved - 24 searches across 8 threads
// agreed with the single-threaded result - but that is evidence, not a guarantee.
// If self-play ever misbehaves in a way that smells like a race, try one thread.
// `playOne` returns false once further games cannot change the answer - the gate
// uses that to quit early. Games already in flight still finish, so the count
// that comes back is not necessarily where it stopped asking.
template <typename PlayOne>
int acrossGames(int games, const char* label, PlayOne&& playOne)
{
    const unsigned    threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int>  nextGame{0};
    std::atomic<int>  completed{0};
    std::atomic<bool> stop{false};
    const auto        start = std::chrono::steady_clock::now();

    const auto worker = [&] {
        for (;;)
        {
            const int game = nextGame.fetch_add(1);
            if (game >= games || stop.load())
                return;

            if (!playOne(game))
                stop = true;

            const int done = completed.fetch_add(1) + 1;
            if (done == 1 || done % 25 == 0 || done == games)
            {
                const double elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                report("[{}] {}/{} games, {:.0f}s elapsed, {:.0f}s left", label, done, games, elapsed,
                       elapsed / done * (games - done));
            }
        }
    };

    {
        std::vector<std::jthread> pool;
        pool.reserve(threads);
        for (unsigned i = 0; i < threads; ++i) {
            pool.emplace_back(worker);
        }
    }
    return completed.load();
}

std::vector<Sample> selfPlay(const Network& best, const Settings& settings, uint64_t seed)
{
    Config search  = settings.search;
    search.rootNoise = settings.noise;

    report("[selfplay] {} games at {} simulations, {} leaves, noise {:.2f}", settings.games,
           search.simulations, search.batchSize, search.rootNoise);

    std::vector<std::vector<Sample>> perGame(static_cast<size_t>(settings.games));

    acrossGames(settings.games, "selfplay", [&](int game) {
        const uint64_t gameSeed = seed + static_cast<uint64_t>(game);

        NetworkEvaluator evaluator{best};
        Config           config = search;
        config.noiseSeed        = gameSeed;
        Search          tree{evaluator, config};
        std::mt19937_64 rng{seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(game) + 1))};

        std::vector<Sample> played = playGame(tree, tree, settings.samplingPlies, rng);
        for (Sample& sample : played) {
            sample.game = game;
        }
        perGame[static_cast<size_t>(game)] = std::move(played);
        return true;   // self-play always plays every game it was asked for
    });

    std::vector<Sample> samples;
    int whiteWins = 0, blackWins = 0, draws = 0;
    for (const std::vector<Sample>& game : perGame)
    {
        samples.insert(samples.end(), game.begin(), game.end());

        // The last sample's mover lost, drew, or was adjudicated against; read the
        // result off it rather than threading the final Board out of playGame.
        const float last = game.back().outcome;
        if (last == 0.0f) ++draws;
        else if (game.back().board.whiteToMove == (last > 0.0f)) ++whiteWins;
        else ++blackWins;
    }

    report("[selfplay] {} positions from {} games: {} White, {} Black, {} drawn", samples.size(),
           settings.games, whiteWins, blackWins, draws);
    return samples;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

Batch batchFrom(const std::vector<Sample>& samples, std::span<const size_t> picks)
{
    std::vector<const amoeba::Board*> boards;
    std::vector<VisitCounts>          visits;
    std::vector<float>                outcomes;
    boards.reserve(picks.size());
    visits.reserve(picks.size());
    outcomes.reserve(picks.size());

    for (const size_t i : picks)
    {
        boards.push_back(&samples[i].board);
        visits.push_back(samples[i].visits);
        outcomes.push_back(samples[i].outcome);
    }
    return makeBatch(boards, visits, outcomes);
}

// Holds out whole games. Positions within a game are near-copies of each other and
// every one carries the same outcome label, so splitting by position leaves a
// held-out position's own game in the training set and the value head can score
// well by recognising the game rather than by reading the board. The symptom is a
// held-out loss that improves as you generate *fewer* games.
struct Split
{
    std::vector<size_t> training;
    std::vector<size_t> validation;
};

Split splitByGame(const std::vector<Sample>& samples, std::mt19937_64& rng)
{
    std::set<int> ids;
    for (const Sample& sample : samples) {
        ids.insert(sample.game);
    }

    std::vector<int> order(ids.begin(), ids.end());
    std::ranges::shuffle(order, rng);
    const size_t        heldOutCount = std::max<size_t>(1, order.size() / 10);
    const std::set<int> heldOut(order.begin(), order.begin() + static_cast<long>(heldOutCount));

    Split split;
    for (size_t i = 0; i < samples.size(); ++i) {
        (heldOut.contains(samples[i].game) ? split.validation : split.training).push_back(i);
    }
    return split;
}

// Leaves `net` holding the weights from the best step, not the last. Held-out loss
// turns back up well before training ends, so keeping the final parameters ships a
// network measurably worse than one already in hand.
void train(Network& net, const std::vector<Sample>& samples, const Settings& settings, uint64_t seed)
{
    std::mt19937_64 rng{seed};
    const Split     split = splitByGame(samples, rng);
    if (split.training.empty() || split.validation.empty())
        throw std::runtime_error("not enough games to hold any out");

    const Batch validation = batchFrom(
        samples, std::span{split.validation}.subspan(0, std::min<size_t>(1024, split.validation.size())));

    std::vector<mlx::core::array> params = net.parameters();
    std::vector<int>              argnums(params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    report("[train] {} parameters, {} positions, {} held out, {} steps at batch {}",
           net.parameterCount(), split.training.size(), split.validation.size(), settings.steps,
           settings.batchSize);
    report("{:>7}  {:>9}  {:>9}  {:>9}  {:>9}", "step", "policy", "value", "held.pol", "held.val");

    // Checked four times more often than it is printed, because the optimum moves
    // as the replay buffer fills: at 22k positions it lands around step 100, and
    // at the buffer's full 200k the same step count is barely one pass over the
    // data and the optimum is far later. Stopping on patience rather than on a
    // fixed count is what lets one STEPS serve both.
    constexpr int kCheckEvery = 25;
    constexpr int kPatience   = 8;

    Adam                          adam{params};
    std::vector<size_t>           picks(static_cast<size_t>(settings.batchSize));
    std::vector<mlx::core::array> best = params;
    float                         bestHeld = std::numeric_limits<float>::max();
    int                           bestStep = 0;
    int                           stale    = 0;

    for (int step = 1; step <= settings.steps; ++step)
    {
        for (size_t& pick : picks) {
            pick = split.training[std::uniform_int_distribution<size_t>{0, split.training.size() - 1}(rng)];
        }
        const Batch batch = batchFrom(samples, picks);

        const auto lossFn = [&](const std::vector<mlx::core::array>& p) {
            return loss(p, net.shape(), batch, settings.decay);
        };
        auto [values, gradients] = mlx::core::value_and_grad(lossFn, argnums)(params);

        params = adam.step(params, gradients, settings.rate);
        mlx::core::eval(params);

        if (step % kCheckEvery != 0 && step != 1)
            continue;

        const std::vector<mlx::core::array> held = loss(params, net.shape(), validation, 0.0f);
        mlx::core::eval(values);
        mlx::core::eval(held);

        // Judged on policy and value together: one orders the moves, the other
        // evaluates the leaves, and the search needs both.
        const float combined = held[1].item<float>() + held[2].item<float>();
        if (combined < bestHeld)
        {
            bestHeld = combined;
            bestStep = step;
            best     = params;
            stale    = 0;
        }
        else
        {
            ++stale;
        }

        if (step % 100 == 0 || step == 1)
            report("{:>7}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}", step, values[1].item<float>(),
                   values[2].item<float>(), held[1].item<float>(), held[2].item<float>());

        if (stale >= kPatience)
        {
            report("[train] held-out loss has not improved in {} checks, stopping at step {}",
                   kPatience, step);
            break;
        }
    }

    report("[train] best held-out loss {:.4f} at step {} of {}, keeping those weights", bestHeld,
           bestStep, settings.steps);
    net.replaceParameters(best);
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

// Standard error of a proportion. Quoted with every gate result so a score is
// not read as more precise than it is: 3.5 points at 200 games, 8 at 40.
double standardError(double score, int games)
{
    return std::sqrt(std::max(score * (1.0 - score), 0.01) / games);
}

// Whether more games could still change the verdict. A candidate that is clearly
// better says so in about twenty games, and one that is genuinely within a point
// or two of the champion would need thousands - neither is worth playing two
// hundred for. Three sigma rather than two because the question is asked after
// every game, and repeated testing at two sigma promotes noise.
bool settled(int played, double score, float threshold)
{
    constexpr int kMinimumGames = 20;
    return played >= kMinimumGames
        && std::abs(score - static_cast<double>(threshold)) > 3.0 * standardError(score, played);
}

// The candidate's score, draws counted as half. Colours alternate so the
// first-move advantage cancels rather than being handed to whoever is listed
// first, and the early sampling in chooseMove is what makes the games differ -
// two argmax players would replay one game GATE_GAMES times.
//
// Ranks with fewer simulations than self-play used, because only the result is
// read here - the visit counts that needed the deeper search are discarded.
double gate(const Network& candidate, const Network& champion, const Settings& settings, uint64_t seed)
{
    Config search      = settings.search;
    search.simulations = settings.gateSimulations;
    search.rootNoise   = 0.0f;   // competition keeps the network's own opinion

    std::atomic<int> wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> losses{0};

    acrossGames(settings.gateGames, "gate", [&](int game) {
        const uint64_t gameSeed = seed + static_cast<uint64_t>(game);

        NetworkEvaluator candidateEval{candidate};
        NetworkEvaluator championEval{champion};
        Search           candidateSearch{candidateEval, search};
        Search           championSearch{championEval, search};

        const bool candidateIsWhite = game % 2 == 0;
        Search&    white = candidateIsWhite ? candidateSearch : championSearch;
        Search&    black = candidateIsWhite ? championSearch : candidateSearch;

        std::mt19937_64           rng{gameSeed};
        const std::vector<Sample> played = playGame(white, black, settings.samplingPlies, rng);

        // The final sample is from the point of view of the side that lost or drew.
        const Sample& last     = played.back();
        const bool    whiteWon = last.outcome < 0.0f ? !last.board.whiteToMove : last.board.whiteToMove;

        if (last.outcome == 0.0f)
            draws.fetch_add(1);
        else if (whiteWon == candidateIsWhite)
            wins.fetch_add(1);
        else
            losses.fetch_add(1);

        const int    total = wins.load() + draws.load() + losses.load();
        const double score = (wins.load() + 0.5 * draws.load()) / total;
        return !settled(total, score, settings.gate);
    });

    const int    total = wins.load() + draws.load() + losses.load();
    const double score = (wins.load() + 0.5 * draws.load()) / total;

    report("[gate] candidate {:.1f}% +/- {:.1f}% ({}-{}-{}) over {} games at {} simulations",
           100.0 * score, 100.0 * standardError(score, total), wins.load(), draws.load(),
           losses.load(), total, settings.gateSimulations);
    return score;
}

// ---------------------------------------------------------------------------

std::filesystem::path firstCheckpointHere()
{
    std::vector<std::filesystem::path> found;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator("."))
        if (entry.path().extension() == ".safetensors")
            found.push_back(entry.path());

    std::ranges::sort(found);
    return found.empty() ? std::filesystem::path{"amoeba.safetensors"} : found.front();
}

} // namespace

} // namespace bot

int main(int argc, char** argv)
{
    const bot::Settings         settings;
    const std::filesystem::path weights =
        argc > 1 ? std::filesystem::path{argv[1]} : bot::firstCheckpointHere();

    // Generation 0 is random weights, written straight away so that amoeba_bot has
    // something to load while the first generation is still being played.
    std::unique_ptr<bot::Network> best;
    if (std::filesystem::exists(weights))
    {
        best = std::make_unique<bot::Network>(weights);
        bot::report("[train] resuming from {}: {} blocks, width {}, {} heads, {} parameters",
                    weights.string(), best->shape().blocks, best->shape().width, best->shape().heads,
                    best->parameterCount());
    }
    else
    {
        best = std::make_unique<bot::Network>(settings.shape, settings.seed);
        best->save(weights);
        bot::report("[train] {} did not exist: started from random weights, {} parameters, saved",
                    weights.string(), best->parameterCount());
    }

    std::vector<bot::Sample> replay;
    int                      gameIdBase = 0;

    for (int generation = 1;; ++generation)
    {
        bot::report("");
        bot::report("======== generation {} ========", generation);

        std::vector<bot::Sample> fresh = bot::selfPlay(*best, settings, settings.seed + static_cast<uint64_t>(generation) * 1000);

        // Game ids have to stay unique across generations, or splitByGame will hold
        // out a game from this generation and train on a different one with the same
        // id from the last.
        for (bot::Sample& sample : fresh) {
            sample.game += gameIdBase;
        }
        gameIdBase += settings.games;

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()),
                      std::make_move_iterator(fresh.end()));
        if (replay.size() > settings.buffer)
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(replay.size() - settings.buffer));
        bot::report("[train] replay buffer holds {} positions", replay.size());

        // The candidate starts from the best weights rather than from scratch: each
        // generation is meant to refine, not to relearn the game.
        bot::Network candidate = *best;
        bot::train(candidate, replay, settings, settings.seed + static_cast<uint64_t>(generation));

        const double score =
            bot::gate(candidate, *best, settings, settings.seed + 7777 + static_cast<uint64_t>(generation));

        if (score < settings.gate)
        {
            bot::report("[train] generation {} rejected at {:.1f}%, keeping the previous weights",
                        generation, 100.0 * score);
            continue;
        }

        *best = candidate;
        best->save(weights);
        bot::report("[train] generation {} PROMOTED at {:.1f}%, wrote {}", generation, 100.0 * score,
                    weights.string());
    }
}
