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
//   BLOCKS=2 WIDTH=64 HEADS=4 GAMES=8 CONCURRENT=8 SIMULATIONS=50 STEPS=100
//   GATE_GAMES=8 GATE_CONCURRENT=8
//
// Environment, all optional:
//   network     BLOCKS WIDTH HEADS          (only read when starting from scratch)
//   generating  SEED GAMES CONCURRENT SIMULATIONS LEAVES SAMPLING_PLIES NOISE
//   training    STEPS BATCH RATE DECAY BUFFER
//   gating      GATE_GAMES GATE_CONCURRENT GATE GATE_SIMULATIONS
//
// STEPS and GATE_GAMES are both caps rather than costs: training stops when the
// held-out loss stops improving, and the gate stops as soon as the verdict is
// settled. Both run to the cap only when the extra work is actually buying
// something.

#include "network.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
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

// MLX hands freed device buffers to a pool rather than back to the system, and the
// pool's default limit is the memory limit - on Metal, 1.5x the device's
// recommended working set, so most of the machine. That makes the split worth
// printing rather than guessing at:
//
//   live    what MLX is actually holding for arrays that exist
//   cached  handed back by MLX but not released to the system
//   process the OS's high-water mark, which includes everything above plus the
//           replay buffer, the trees and every other allocation we make
//
// A large cache beside a small live figure is shape churn, not a leak: the field's
// batch shrinks by one every time a game ends, and each distinct batch size is a
// differently shaped set of intermediates for the pool to keep. A large live figure
// is something genuinely retained, and then the fault is ours.
void reportMemory(const char* phase)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);   // ru_maxrss is bytes on macOS, kilobytes on Linux

    report("[memory] after {}: {:.2f} GiB live, {:.2f} GiB cached, {:.2f} GiB MLX peak, "
           "{:.2f} GiB process high-water",
           phase, static_cast<double>(mlx::core::get_active_memory()) / kGiB,
           static_cast<double>(mlx::core::get_cache_memory()) / kGiB,
           static_cast<double>(mlx::core::get_peak_memory()) / kGiB,
           static_cast<double>(usage.ru_maxrss) / kGiB);
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
    // visits sharpens it far less than doubling the games broadens it. It counts
    // simulations through the root, so a re-rooted tree arrives with a good share
    // of them already spent.
    // One leaf per search, because the batch comes from the other games in flight
    // and there is nothing to gain from guessing at a second leaf before hearing
    // about the first.
    Config search = {.simulations = envInt("SIMULATIONS", 400),
                     .batchSize   = envInt("LEAVES", 1)};

    // 512 games is roughly 60k positions. Fewer than that and each generation's
    // training set is mostly the previous generation's, so the gate compares two
    // networks that saw nearly the same data and rejects almost everything.
    //
    // Above CONCURRENT, so a slot takes on another game as its own ends rather than
    // going idle. What a game costs does not depend on when it is played - every one
    // of the 512 builds its own tree from nothing and gets its own cheap endgame - so
    // this buys games rather than adding overhead, and it keeps the batch full for
    // the bulk of the run instead of only until the first game ends.
    int   games         = envInt("GAMES", 512);
    int   samplingPlies = envInt("SAMPLING_PLIES", 20);
    float noise         = envFloat("NOISE", 0.25f);

    // How many games are in flight at once, which is also the batch the network
    // sees: one position per game per round. The evaluator costs 0.155 ms/position
    // at 256 against 3.84 ms on its own, so the field is what buys the throughput.
    //
    // Measure before trusting 256: a field of 64 came out at 1.82x the old code and
    // a field of 256 at parity with it, which the evaluator's own numbers say should
    // be impossible. See "Batching across games" in CLAUDE.md.
    int concurrent = envInt("CONCURRENT", 256);

    int   steps     = envInt("STEPS", 1000);
    int   batchSize = envInt("BATCH", 256);
    float rate      = envFloat("RATE", 1e-3f);
    float decay     = envFloat("DECAY", 1e-4f);

    // About the last five generations. Older positions came from networks several
    // generations weaker and hold the current one back; keeping none of them at
    // all makes each generation overfit the games it just played. ~650 MB.
    size_t buffer = static_cast<size_t>(envInt("BUFFER", 300000));

    // 200 games puts the gate's error bar at +/-3.5%, which is what it takes for
    // a 55% result to mean anything - at 40 games the bar is +/-8% and promotion
    // is close to a coin flip. It costs as much as the self-play it judges, and
    // that is the price of the only honest signal in the system.
    // 200 is the cap, not the usual cost: the gate stops as soon as the verdict is
    // settled, which for a clearly better candidate is about twenty games. The cap
    // only gets spent when the two networks are genuinely close.
    int   gateGames = envInt("GATE_GAMES", 200);
    float gate      = envFloat("GATE", 0.55f);

    // A quarter of self-play's field, because a gate that stops early throws away
    // whatever is still in flight. settled() needs twenty games in, and twenty of a
    // field of 64 land well before the other 44 - which are then abandoned.
    int gateConcurrent = envInt("GATE_CONCURRENT", 64);

    // Ranking two networks needs far less search than generating a training
    // target does: the visit counts are thrown away here, only the result counts.
    int gateSimulations = envInt("GATE_SIMULATIONS", 200);
};

// ---------------------------------------------------------------------------
// Playing games, hundreds of them at once
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

// Which network takes which colour, as indices into the field's network list.
struct Pairing
{
    int white;
    int black;
};

// One game in flight.
struct Playing
{
    amoeba::Board         board;
    std::vector<uint64_t> history;
    std::vector<Sample>   samples;
    std::mt19937_64       rng;
    Pairing               pairing{0, 0};
    int                   id     = 0;
    bool                  active = false;

    // Reported when the game ends. `samples` is one per move, so the move count is
    // already there; the evaluations have to be counted as they are absorbed.
    std::chrono::steady_clock::time_point started;
    long long                             evaluations = 0;

    // One tree per network, not per colour. Self-play has a single network and so
    // a single tree, which it re-roots after every ply instead of every other one -
    // that is where most of the reuse comes from, since the tree keeps what both
    // sides found. The gate has two, because a tree's statistics are worth exactly
    // what the network that produced them is.
    std::vector<Search> trees;

    // Set fresh each round: the boards this game cannot go on without, which
    // network owes it the answers, and where they sit in that network's batch.
    std::span<const amoeba::Board* const> pending;
    int                                   network = 0;
    size_t                                offset  = 0;
};

// Plays a whole field of games at once, one simulation each per round, so that a
// single network call answers every game in flight. The batch is the size of the
// field rather than the size of whatever leaves one search could guess at, and
// nothing is stale: each descent sees the statistics its own tree ended the last
// round with, so no virtual loss is needed and none is applied.
class Field
{
public:
    Field(std::span<Evaluator* const> networks, Config search, int samplingPlies)
        : m_networks(networks), m_search(search), m_samplingPlies(samplingPlies)
    {
    }

    // Plays `games` games with `slots` of them in flight, refilling a slot as its
    // game ends so the batch stays full until the work runs out. `pairingFor(game)`
    // says who plays which colour; `finished` is handed each game as it ends, under
    // a lock, so its tally needs no synchronisation of its own. `finished` returning
    // false stops the field there and then - the games still in flight are
    // abandoned, so the tally is exactly what it was when it said stop.
    void play(int games, int slots, uint64_t seed, const char* label,
              const std::function<Pairing(int)>&                     pairingFor,
              const std::function<bool(int, std::vector<Sample>&&)>& finished);

private:
    void begin(Playing&, int id, Pairing, uint64_t seed) const;
    void takeToPending(Playing&) const;

    std::span<Evaluator* const> m_networks;
    Config                      m_search;
    int                         m_samplingPlies;
};

void Field::begin(Playing& game, int id, Pairing pairing, uint64_t seed) const
{
    game.id          = id;
    game.board       = amoeba::startPosition();
    game.pairing     = pairing;
    game.active      = true;
    game.started     = std::chrono::steady_clock::now();
    game.evaluations = 0;
    game.history.assign(1, game.board.hash);
    game.samples.clear();
    game.rng.seed(seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(id) + 1)));

    // Everything random about a game comes from its id, never from which slot or
    // thread happened to pick it up, so a whole run stays reproducible from SEED
    // however the field interleaves.
    game.trees.clear();
    for (size_t i = 0; i < m_networks.size(); ++i)
    {
        Config config    = m_search;
        config.noiseSeed = seed + static_cast<uint64_t>(id);
        game.trees.emplace_back(config);
    }
    for (Search& tree : game.trees)
        tree.restart(game.board, game.history);
}

// Plays as far as it can without an evaluation, leaving the boards it is waiting
// on in `pending` - or leaving it empty, which means the game is over.
void Field::takeToPending(Playing& game) const
{
    while (game.board.state == amoeba::State::Ongoing)
    {
        const int network = game.board.whiteToMove ? game.pairing.white : game.pairing.black;
        Search&   tree    = game.trees[static_cast<size_t>(network)];

        const std::span<const amoeba::Board* const> pending = tree.pendingLeaves();
        if (!pending.empty())
        {
            game.network = network;
            game.pending = pending;
            return;
        }

        const VisitCounts counts = tree.visits();
        game.samples.push_back({game.board, counts, 0.0f, game.id});

        const uint16_t chosen = chooseMove(counts, game.board.ply, m_samplingPlies, game.rng);
        game.board = amoeba::apply(game.board, amoeba::Move::fromId(chosen), game.history);
        game.history.push_back(game.board.hash);

        // Every tree follows the game, not just the one that was searching: a tree
        // can only keep a subtree while its root is where the game is.
        for (Search& follower : game.trees)
            follower.advance(chosen, game.board, game.history);
    }
    game.pending = {};
}

void Field::play(int games, int slots, uint64_t seed, const char* label,
                 const std::function<Pairing(int)>&                     pairingFor,
                 const std::function<bool(int, std::vector<Sample>&&)>& finished)
{
    if (games <= 0)
        return;

    std::vector<Playing> field(static_cast<size_t>(std::max(1, std::min(games, slots))));
    std::vector<std::vector<const amoeba::Board*>> boards(m_networks.size());
    std::vector<std::vector<Evaluation>>           answers(m_networks.size());

    std::mutex        gatekeeper;
    std::atomic<bool> wanted{true};
    int               completed = 0;
    int               started   = 0;

    report("[{}] {} games, {} in flight at {} simulations, {} leaves per search", label, games,
           field.size(), m_search.simulations, m_search.batchSize);

    for (Playing& game : field)
    {
        begin(game, started, pairingFor(started), seed);
        ++started;
    }

    // Hands a finished game over and takes on the next one, under one lock: it is
    // where the caller keeps its tally and decides whether any more games are worth
    // playing, and where the one line a game gets is printed. Once per game rather
    // than once per round, so the lock is never contended for long.
    //
    // The line reports the game's own numbers because the field's do not mean much:
    // the games are never on the same move, so there is no shared round to report.
    // A game's moves and evaluations together say what its tree reuse was worth -
    // 400 simulations a move, minus whatever it inherited.
    const auto handOver = [&](Playing& game) {
        for (Sample& sample : game.samples) {
            sample.outcome = outcomeFor(game.board.state, sample.board.whiteToMove);
        }

        const std::lock_guard guard{gatekeeper};

        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - game.started).count();
        const size_t moves = game.samples.size();

        if (!finished(game.id, std::move(game.samples)))
            wanted = false;
        ++completed;

        report("[{}] game {}: {:.0f}s, {} moves, {} network calls -- {} games left", label, game.id,
               seconds, moves, game.evaluations, games - completed);

        if (!wanted || started >= games)
        {
            game.active = false;
            return;
        }
        begin(game, started, pairingFor(started), seed);
        ++started;
    };

    const std::function<void(size_t)> stepSlot = [&](size_t slot) {
        Playing& game = field[slot];
        if (!game.active)
            return;

        if (!game.pending.empty())
        {
            game.evaluations += static_cast<long long>(game.pending.size());
            game.trees[static_cast<size_t>(game.network)].absorb(
                std::span{answers[static_cast<size_t>(game.network)]}.subspan(game.offset, game.pending.size()));
        }

        takeToPending(game);
        while (game.active && game.pending.empty())
        {
            handOver(game);
            if (game.active)
                takeToPending(game);
        }
    };

    for (;;)
    {
        ThreadPool::global().forEach(field.size(), stepSlot);
        if (!wanted)
            return;

        size_t total = 0;
        for (std::vector<const amoeba::Board*>& group : boards) {
            group.clear();
        }
        for (Playing& game : field)
        {
            if (!game.active || game.pending.empty())
                continue;

            std::vector<const amoeba::Board*>& group = boards[static_cast<size_t>(game.network)];
            game.offset = group.size();
            group.insert(group.end(), game.pending.begin(), game.pending.end());
            total += game.pending.size();
        }
        if (total == 0)
            return;

        // One call per network, from one thread. MLX is dispatch-bound at this size,
        // so a single stream of large batches beats several threads each pushing a
        // small one, and it is the whole reason the games are driven in lockstep.
        for (size_t n = 0; n < m_networks.size(); ++n)
        {
            if (boards[n].empty())
                continue;
            answers[n].resize(boards[n].size());
            m_networks[n]->evaluate(boards[n], answers[n]);
        }
    }
}

std::vector<Sample> selfPlay(const Network& best, const Settings& settings, uint64_t seed)
{
    Config search    = settings.search;
    search.rootNoise = settings.noise;

    NetworkEvaluator evaluator{best};
    Evaluator* const networks[]{&evaluator};

    std::vector<Sample> samples;
    int whiteWins = 0, blackWins = 0, draws = 0;

    Field field{networks, search, settings.samplingPlies};
    field.play(settings.games, settings.concurrent, seed, "selfplay",
               [](int) { return Pairing{0, 0}; },
               [&](int, std::vector<Sample>&& played) {
                   // The last sample's mover lost, drew, or was adjudicated against;
                   // read the result off it rather than threading the final Board out.
                   const float last = played.back().outcome;
                   if (last == 0.0f) ++draws;
                   else if (played.back().board.whiteToMove == (last > 0.0f)) ++whiteWins;
                   else ++blackWins;

                   samples.insert(samples.end(), std::make_move_iterator(played.begin()),
                                  std::make_move_iterator(played.end()));
                   return true;   // self-play always plays every game it was asked for
               });

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

    NetworkEvaluator candidateEval{candidate};
    NetworkEvaluator championEval{champion};
    Evaluator* const networks[]{&candidateEval, &championEval};

    int wins = 0, draws = 0, losses = 0;

    // A smaller field than self-play, because everything still in flight when the
    // verdict settles is work thrown away: settled() is consulted on each game that
    // comes in, and the twentieth of 64 lands long before the rest.
    Field field{networks, search, settings.samplingPlies};
    field.play(settings.gateGames, settings.gateConcurrent, seed, "gate",
               [](int game) { return game % 2 == 0 ? Pairing{0, 1} : Pairing{1, 0}; },
               [&](int game, std::vector<Sample>&& played) {
                   // The final sample is from the point of view of the side that lost or drew.
                   const Sample& last     = played.back();
                   const bool    whiteWon = last.outcome < 0.0f ? !last.board.whiteToMove : last.board.whiteToMove;

                   if (last.outcome == 0.0f)
                       ++draws;
                   else if (whiteWon == (game % 2 == 0))
                       ++wins;
                   else
                       ++losses;

                   const int    total = wins + draws + losses;
                   const double score = (wins + 0.5 * draws) / total;
                   return !settled(total, score, settings.gate);
               });

    const int    total = wins + draws + losses;
    const double score = (wins + 0.5 * draws) / total;

    report("[gate] candidate {:.1f}% +/- {:.1f}% ({}-{}-{}) over {} games at {} simulations",
           100.0 * score, 100.0 * standardError(score, total), wins, draws, losses, total,
           settings.gateSimulations);
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

        bot::reportMemory("self-play");

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()),
                      std::make_move_iterator(fresh.end()));
        if (replay.size() > settings.buffer)
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(replay.size() - settings.buffer));
        bot::report("[train] replay buffer holds {} positions", replay.size());

        // The candidate starts from the best weights rather than from scratch: each
        // generation is meant to refine, not to relearn the game.
        bot::Network candidate = *best;
        bot::train(candidate, replay, settings, settings.seed + static_cast<uint64_t>(generation));
        bot::reportMemory("training");

        const double score =
            bot::gate(candidate, *best, settings, settings.seed + 7777 + static_cast<uint64_t>(generation));
        bot::reportMemory("the gate");

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
