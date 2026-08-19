// Proof of concept for the AlphaZero loop, and the two things needed to judge it.
//
// This is a working skeleton, not the finished program. It exists so the real entry
// point can be built on something that has been run end to end rather than on a
// design document. Everything it calls - the engine, the encoder, the search, the
// network, the loss, Adam - is already tested; what is new here is only the wiring.
//
// MODE=selfplay   the loop. Generation 0 is random weights. Each generation plays
//                 itself, pushes the games into a replay buffer, trains a candidate,
//                 and promotes it only if it beats the current best in a match.
//
// MODE=train      imitation learning from rollout MCTS instead of from itself. A
//                 fixed teacher, so the network is the only moving part. Useful as a
//                 control: it reaches the teacher's strength and stops.
//
// MODE=match      load two players and count wins. random | rollout | network.
//
// Environment, all optional and listed with the mode that reads it:
//   common      SEED BLOCKS WIDTH HEADS SIMULATIONS LEAVES
//   generating  GAMES SAMPLING_PLIES NOISE
//   training    STEPS BATCH RATE DECAY
//   selfplay    GENERATIONS BUFFER GATE_GAMES GATE OUT_DIR
//   train       OUT
//   match       CHECKPOINT PLAYER OPPONENT
//
// Known gaps, left deliberately for whoever builds the real thing:
//   - Games are never written to disk, so nothing can be re-trained without being
//     re-generated.
//   - Adam's moment estimates are not checkpointed, so a generation cannot be
//     resumed part way through.
//   - The gate needs several hundred games to mean anything. At the default 20 its
//     error bar is +/-11%, which cannot separate 55% from a coin flip.
//   - Self-play batches leaves within one game. Batching across many concurrent
//     games is the real throughput win and is a larger change.

#include "training.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <print>
#include <random>
#include <set>
#include <span>
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

std::string envText(const char* name, const char* fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr ? fallback : text;
}

// ---------------------------------------------------------------------------
// Generating games
// ---------------------------------------------------------------------------

struct Sample
{
    amoeba::Board board;
    VisitCounts visits;
    float outcome = 0.0f;
    int game = 0;   // unique across generations, so whole games can be held out
};

// Proportional to visits early, argmax afterwards. Without the early sampling every
// game walks the same opening and the training set is far narrower than its position
// count suggests.
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

std::vector<Sample> playGame(Search& search, int samplingPlies, std::mt19937_64& rng)
{
    std::vector<Sample> samples;
    amoeba::Board board = amoeba::startPosition();
    std::vector<uint64_t> history{board.hash};

    while (board.state == amoeba::State::Ongoing)
    {
        const VisitCounts counts = search.run(board, history);
        samples.push_back({board, counts, 0.0f, 0});

        const uint16_t chosen = chooseMove(counts, board.ply, samplingPlies, rng);
        board = amoeba::apply(board, amoeba::Move::fromId(chosen), history);
        history.push_back(board.hash);
    }

    // Who actually won, from the point of view of whoever was to move at that
    // position - not from White's. Inverting this trains a bot that prefers losing
    // while every loss curve stays perfectly healthy.
    for (Sample& sample : samples)
    {
        sample.outcome = board.state == amoeba::State::Draw ? 0.0f
                       : (board.state == amoeba::State::WhiteWins) == sample.board.whiteToMove ? 1.0f : -1.0f;
    }
    return samples;
}

// A fresh evaluator per game: Search owns the vectors it reuses and RolloutEvaluator
// owns an rng, so neither can be shared across threads.
using EvaluatorFactory = std::function<std::unique_ptr<Evaluator>(uint64_t)>;

// Games share nothing, so one slot per game means each thread writes where nobody
// else does and no lock is needed on the results. Seeding from the game index rather
// than the thread keeps a whole run reproducible from SEED however the threads
// interleave, and keeps samples in game order.
//
// Concurrent MLX evaluation was tested and behaved - 24 searches across 8 threads
// agreed with the single-threaded result - but that is evidence, not a guarantee. If
// self-play ever misbehaves in a way that smells like a race, try one thread first.
std::vector<Sample> generate(const EvaluatorFactory& makeEvaluator, Config search, int games,
                             int samplingPlies, uint64_t seed, const char* label)
{
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    report("[{}] {} games at {} simulations, {} leaves, noise {:.2f}, across {} threads", label, games,
           search.simulations, search.batchSize, search.rootNoise, threads);

    std::vector<std::vector<Sample>> perGame(static_cast<size_t>(games));
    std::atomic<int> nextGame{0};
    std::atomic<int> completed{0};
    const auto start = std::chrono::steady_clock::now();

    const auto worker = [&] {
        while (true)
        {
            const int game = nextGame.fetch_add(1);
            if (game >= games)
                return;

            const uint64_t gameSeed = seed + static_cast<uint64_t>(game);
            const std::unique_ptr<Evaluator> evaluator = makeEvaluator(gameSeed);

            Config config = search;
            config.noiseSeed = gameSeed;
            Search tree{*evaluator, config};
            std::mt19937_64 rng{seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(game) + 1))};

            std::vector<Sample> played = playGame(tree, samplingPlies, rng);
            for (Sample& sample : played) {
                sample.game = game;
            }
            perGame[static_cast<size_t>(game)] = std::move(played);

            const int done = completed.fetch_add(1) + 1;
            if (done == 1 || done % 25 == 0 || done == games)
            {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

    report("[{}] {} positions from {} games: {} White, {} Black, {} drawn", label, samples.size(), games,
           whiteWins, blackWins, draws);
    return samples;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

Batch batchFrom(const std::vector<Sample>& samples, std::span<const size_t> picks)
{
    std::vector<const amoeba::Board*> boards;
    std::vector<VisitCounts> visits;
    std::vector<float> outcomes;
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
// held-out position's own game in the training set and the value head can score well
// by recognising the game rather than by reading the board. The symptom is a
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
    const size_t heldOutCount = std::max<size_t>(1, order.size() / 10);
    const std::set<int> heldOut(order.begin(), order.begin() + static_cast<long>(heldOutCount));

    Split split;
    for (size_t i = 0; i < samples.size(); ++i) {
        (heldOut.contains(samples[i].game) ? split.validation : split.training).push_back(i);
    }
    return split;
}

struct TrainConfig
{
    int steps = 600;
    int batchSize = 256;
    float rate = 1e-3f;
    float decay = 1e-4f;
};

// Leaves `net` holding the weights from the best step, not the last. Held-out loss
// turns back up well before training ends, so keeping the final parameters ships a
// network measurably worse than one already in hand.
void train(Network& net, const std::vector<Sample>& samples, TrainConfig config, uint64_t seed)
{
    std::mt19937_64 rng{seed};
    const Split split = splitByGame(samples, rng);
    if (split.training.empty() || split.validation.empty())
        throw std::runtime_error("not enough games to hold any out");

    const Batch validation =
        batchFrom(samples, std::span{split.validation}.subspan(0, std::min<size_t>(1024, split.validation.size())));

    std::vector<mlx::core::array> params = net.parameters();
    std::vector<int> argnums(params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    report("[train] {} parameters, {} positions, {} held out, {} steps at batch {}", net.parameterCount(),
           split.training.size(), split.validation.size(), config.steps, config.batchSize);
    report("{:>7}  {:>9}  {:>9}  {:>9}  {:>9}", "step", "policy", "value", "held.pol", "held.val");

    Adam adam{params};
    std::vector<size_t> picks(static_cast<size_t>(config.batchSize));
    std::vector<mlx::core::array> best = params;
    float bestHeld = std::numeric_limits<float>::max();
    int bestStep = 0;

    for (int step = 1; step <= config.steps; ++step)
    {
        for (size_t& pick : picks) {
            pick = split.training[std::uniform_int_distribution<size_t>{0, split.training.size() - 1}(rng)];
        }
        const Batch batch = batchFrom(samples, picks);

        const auto lossFn = [&](const std::vector<mlx::core::array>& p) { return loss(p, net.shape(), batch, config.decay); };
        auto [values, gradients] = mlx::core::value_and_grad(lossFn, argnums)(params);

        params = adam.step(params, gradients, config.rate);
        mlx::core::eval(params);

        if (step % 100 == 0 || step == 1)
        {
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
                best = params;
            }
            report("{:>7}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}", step, values[1].item<float>(),
                   values[2].item<float>(), held[1].item<float>(), held[2].item<float>());
        }
    }

    report("[train] best held-out loss {:.4f} at step {} of {}, keeping those weights", bestHeld, bestStep,
           config.steps);
    net.replaceParameters(best);
}

// ---------------------------------------------------------------------------
// Matches
// ---------------------------------------------------------------------------

struct Player
{
    virtual ~Player() = default;
    virtual uint16_t pick(const amoeba::Board& board, std::span<const uint64_t> history) = 0;
};

class RandomPlayer final : public Player
{
public:
    explicit RandomPlayer(uint64_t seed) : m_rng(seed) {}

    uint16_t pick(const amoeba::Board& board, std::span<const uint64_t>) override
    {
        return randomLegalMove(board, m_rng);
    }

private:
    std::mt19937_64 m_rng;
};

class SearchPlayer final : public Player
{
public:
    SearchPlayer(Evaluator& evaluator, Config config) : m_search(evaluator, config) {}

    uint16_t pick(const amoeba::Board& board, std::span<const uint64_t> history) override
    {
        return bestMove(m_search.run(board, history));
    }

private:
    Search m_search;
};

// A side of a match. `net` is read only when kind is "network", which is what lets a
// candidate play the current best without either being a special case.
struct Side
{
    std::string kind;
    const Network* net = nullptr;
};

std::unique_ptr<Player> makePlayer(const Side& side, Config search, uint64_t seed,
                                   std::vector<std::unique_ptr<Evaluator>>& owned)
{
    if (side.kind == "random")
        return std::make_unique<RandomPlayer>(seed);

    if (side.kind == "rollout")
    {
        owned.push_back(std::make_unique<RolloutEvaluator>(seed));
        search.batchSize = 1;   // a rollout gains nothing from batching
    }
    else if (side.kind == "network")
    {
        if (side.net == nullptr)
            throw std::runtime_error("a network side needs a network");
        owned.push_back(std::make_unique<NetworkEvaluator>(*side.net));
    }
    else
    {
        throw std::runtime_error(std::format("a side must be random, rollout or network, not {}", side.kind));
    }

    // Competition takes the argmax, so no root noise here whatever generation used.
    search.rootNoise = 0.0f;
    return std::make_unique<SearchPlayer>(*owned.back(), search);
}

// +1 if White won, -1 if Black, 0 drawn.
int playMatch(Player& white, Player& black, int& plies)
{
    amoeba::Board board = amoeba::startPosition();
    std::vector<uint64_t> history{board.hash};

    while (board.state == amoeba::State::Ongoing)
    {
        Player& mover = board.whiteToMove ? white : black;
        board = amoeba::apply(board, amoeba::Move::fromId(mover.pick(board, history)), history);
        history.push_back(board.hash);
        ++plies;
    }
    return board.state == amoeba::State::Draw ? 0 : board.state == amoeba::State::WhiteWins ? 1 : -1;
}

// The score for `first`, draws counted as half. Colours alternate so the first-move
// advantage cancels rather than being handed to whoever is listed first.
double runSeries(const Side& first, const Side& second, Config search, int games, uint64_t seed,
                 bool verbose)
{
    int wins = 0, losses = 0, draws = 0, plies = 0;

    for (int game = 0; game < games; ++game)
    {
        std::vector<std::unique_ptr<Evaluator>> owned;
        const std::unique_ptr<Player> a = makePlayer(first, search, seed + static_cast<uint64_t>(game), owned);
        const std::unique_ptr<Player> b = makePlayer(second, search, seed + 0xabcdef + static_cast<uint64_t>(game), owned);

        const bool firstIsWhite = game % 2 == 0;
        const int result = firstIsWhite ? playMatch(*a, *b, plies) : playMatch(*b, *a, plies);
        const int forFirst = firstIsWhite ? result : -result;

        if (forFirst > 0) ++wins;
        else if (forFirst < 0) ++losses;
        else ++draws;

        if (verbose)
            report("[match] {}/{}: {}-{}-{}", game + 1, games, wins, losses, draws);
    }

    const double score = (wins + 0.5 * draws) / games;
    // Standard error of a proportion, so a result is not read as more precise than
    // it is. At 20 games this is about 11 points.
    const double error = std::sqrt(score * (1.0 - score) / games);
    report("[match] {} vs {}: {:.1f}% +/- {:.1f}% ({}-{}-{}) over {} games, {} plies average", first.kind,
           second.kind, 100.0 * score, 100.0 * error, wins, losses, draws, games, plies / games);
    return score;
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

struct Common
{
    uint64_t seed;
    NetworkShape shape;
    Config search;
    int games;
    int samplingPlies;
    TrainConfig training;
};

Common readCommon()
{
    Common common;
    common.seed = static_cast<uint64_t>(envInt("SEED", 20260819));
    common.shape = {.blocks = envInt("BLOCKS", 2), .width = envInt("WIDTH", 64), .heads = envInt("HEADS", 4)};
    common.search = {.simulations = envInt("SIMULATIONS", 100), .batchSize = envInt("LEAVES", 16)};
    common.games = envInt("GAMES", 30);
    common.samplingPlies = envInt("SAMPLING_PLIES", 20);
    common.training = {.steps = envInt("STEPS", 300), .batchSize = envInt("BATCH", 128),
                       .rate = envFloat("RATE", 1e-3f), .decay = envFloat("DECAY", 1e-4f)};
    return common;
}

// The loop. Each generation plays the current best against itself, trains a
// candidate on everything in the buffer, and keeps the candidate only if it wins.
//
// The gate is what makes this honest. A training loss can fall while the player gets
// worse, so nothing is promoted on a loss curve - only on games won.
int runSelfPlay(const Common& common)
{
    const int generations = envInt("GENERATIONS", 3);
    const size_t buffer = static_cast<size_t>(envInt("BUFFER", 50000));
    const int gateGames = envInt("GATE_GAMES", 20);
    const float gate = envFloat("GATE", 0.55f);
    const float noise = envFloat("NOISE", 0.25f);
    const std::filesystem::path outDir = envText("OUT_DIR", "checkpoints");

    std::filesystem::create_directories(outDir);

    Network best{common.shape, common.seed};
    best.save(outDir / "gen0.safetensors");
    report("[selfplay] generation 0 is random weights, {} parameters, saved to {}", best.parameterCount(),
           (outDir / "gen0.safetensors").string());

    std::vector<Sample> replay;
    int gameIdBase = 0;
    int promoted = 0;

    for (int generation = 1; generation <= generations; ++generation)
    {
        report("");
        report("======== generation {} of {} ========", generation, generations);

        Config search = common.search;
        search.rootNoise = noise;

        const EvaluatorFactory factory = [&best](uint64_t) { return std::make_unique<NetworkEvaluator>(best); };
        std::vector<Sample> fresh = generate(factory, search, common.games, common.samplingPlies,
                                             common.seed + static_cast<uint64_t>(generation) * 1000, "selfplay");

        // Game ids have to stay unique across generations or splitByGame will hold
        // out a game from this generation and train on a different one with the same
        // id from the last.
        for (Sample& sample : fresh) {
            sample.game += gameIdBase;
        }
        gameIdBase += common.games;

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()), std::make_move_iterator(fresh.end()));
        if (replay.size() > buffer)
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(replay.size() - buffer));
        report("[selfplay] replay buffer holds {} positions", replay.size());

        // The candidate starts from the best weights rather than from scratch: each
        // generation is meant to refine, not to relearn the game.
        Network candidate = best;
        train(candidate, replay, common.training, common.seed + static_cast<uint64_t>(generation));

        const double score = runSeries({"network", &candidate}, {"network", &best}, common.search, gateGames,
                                       common.seed + 7777 + static_cast<uint64_t>(generation), false);

        const std::filesystem::path path = outDir / std::format("gen{}.safetensors", generation);
        if (score >= gate)
        {
            best = candidate;
            best.save(path);
            ++promoted;
            report("[selfplay] generation {} PROMOTED at {:.1f}% and saved to {}", generation, 100.0 * score,
                   path.string());
        }
        else
        {
            report("[selfplay] generation {} rejected at {:.1f}%, keeping the previous best", generation,
                   100.0 * score);
        }
    }

    report("");
    report("[selfplay] {} of {} generations promoted", promoted, generations);
    report("[selfplay] a {}-game gate has an error bar of about {:.0f} points, so at this scale promotion is",
           gateGames, 100.0 * std::sqrt(0.25 / gateGames));
    report("[selfplay] close to a coin flip - raise GATE_GAMES into the hundreds before trusting it");
    return 0;
}

// Imitation learning from rollout MCTS. A fixed teacher, so the network is the only
// moving part - which is why this is worth having beside the loop: if self-play
// misbehaves, this isolates whether the network or the loop is at fault.
int runTrain(const Common& common)
{
    const std::filesystem::path out = envText("OUT", "gen1.safetensors");

    Config search = common.search;
    search.batchSize = 1;
    const EvaluatorFactory factory = [](uint64_t seed) { return std::make_unique<RolloutEvaluator>(seed); };

    const std::vector<Sample> samples =
        generate(factory, search, common.games, common.samplingPlies, common.seed, "generate");

    Network net{common.shape, common.seed};
    train(net, samples, common.training, common.seed);
    net.save(out);
    report("[train] wrote {}", out.string());
    return 0;
}

int runMatch(const Common& common)
{
    const std::filesystem::path checkpoint = envText("CHECKPOINT", "gen1.safetensors");
    const std::string playerKind = envText("PLAYER", "network");
    const std::string opponentKind = envText("OPPONENT", "random");

    // Only load a checkpoint if a side actually needs one.
    std::unique_ptr<Network> net;
    if (playerKind == "network" || opponentKind == "network")
    {
        net = std::make_unique<Network>(checkpoint);
        report("[match] loaded {}: {} blocks, width {}, {} heads, {} parameters", checkpoint.string(),
               net->shape().blocks, net->shape().width, net->shape().heads, net->parameterCount());
    }

    runSeries({playerKind, net.get()}, {opponentKind, net.get()}, common.search, envInt("GAMES", 20),
              common.seed, true);
    return 0;
}

} // namespace

} // namespace bot

int main()
{
    const std::string mode = bot::envText("MODE", "selfplay");
    const bot::Common common = bot::readCommon();

    bot::report("[config] mode {}, seed {}, {} blocks / width {} / {} heads", mode, common.seed,
                common.shape.blocks, common.shape.width, common.shape.heads);

    if (mode == "selfplay")
        return bot::runSelfPlay(common);
    if (mode == "train")
        return bot::runTrain(common);
    if (mode == "match")
        return bot::runMatch(common);

    bot::report("MODE must be selfplay, train or match, not {}", mode);
    return EXIT_FAILURE;
}
