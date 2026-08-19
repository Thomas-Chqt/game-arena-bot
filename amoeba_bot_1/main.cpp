// Trains the network by imitating rollout MCTS.
//
// This is deliberately not self-play. The teacher is RolloutEvaluator, which never
// changes, so the games are generated once and the only moving part is the network.
// Self-play is the same code with NetworkEvaluator passed to the search instead and
// the whole thing wrapped in a loop - worth doing second, because when a
// bootstrapping loop fails, a broken network, bad data and an unstable loop all
// look identical.
//
// The policy target is the teacher's opinion and inherits its ceiling. The value
// target is who actually won, which is ground truth however weak the teacher is.
//
// Environment, all optional:
//   GAMES SIMULATIONS SAMPLING_PLIES SEED     generation
//   BLOCKS WIDTH HEADS                        network size
//   STEPS BATCH RATE DECAY                    training
//   OUT                                       where the checkpoint goes

#include "training.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace bot
{

namespace
{

// std::println leaves stdout block-buffered whenever it is not a terminal, so a
// redirected or piped run shows nothing at all until the buffer fills - which for
// short progress lines means not until the process exits. Generation takes minutes
// and watching it is the whole point.
template <typename... Args>
void report(std::format_string<Args...> format, Args&&... args)
{
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
}

struct Sample
{
    amoeba::Board board;
    VisitCounts visits;
    float outcome = 0.0f;
};

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

// Proportional to visits early, argmax afterwards. Without the early sampling the
// teacher walks nearly the same opening every game, and the training set is far
// narrower than its position count suggests.
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
        samples.push_back({board, counts, 0.0f});

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

// Each sample holds a whole Board and a whole 444-entry visit array, so roughly
// 2.2 KB. Fine for the ~85k positions a first run wants; a real replay buffer
// would store the visits sparsely.
//
// Games share nothing, so this is as parallel as work gets. One slot per game
// means each thread writes where nobody else does and no lock is needed on the
// results; seeding from the game index rather than the thread keeps a whole run
// reproducible from SEED however the threads happen to interleave.
std::vector<Sample> generate(int games, int simulations, int samplingPlies, uint64_t seed)
{
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    report("[generate] {} games at {} simulations across {} threads", games, simulations, threads);

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

            // Search owns the node and edge vectors it reuses and RolloutEvaluator
            // owns an rng, so neither can be shared across threads. Rebuilding both
            // per game costs a handful of allocations against a game that takes about
            // a second.
            RolloutEvaluator evaluator{seed + static_cast<uint64_t>(game)};
            Search search{evaluator, {.simulations = simulations}};
            std::mt19937_64 rng{seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(game) + 1))};

            perGame[static_cast<size_t>(game)] = playGame(search, samplingPlies, rng);

            const int done = completed.fetch_add(1) + 1;
            if (done == 1 || done % 25 == 0 || done == games)
            {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                report("[generate] {}/{} games, {:.0f}s elapsed, {:.0f}s left",
                       done, games, elapsed, elapsed / done * (games - done));
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
        // result back off it rather than threading the final Board out of playGame.
        const float last = game.back().outcome;
        if (last == 0.0f) ++draws;
        else if (game.back().board.whiteToMove == (last > 0.0f)) ++whiteWins;
        else ++blackWins;
    }

    report("[generate] {} positions from {} games: {} White, {} Black, {} drawn",
           samples.size(), games, whiteWins, blackWins, draws);
    return samples;
}

} // namespace

} // namespace bot

int main()
{
    const int games = bot::envInt("GAMES", 200);
    const int simulations = bot::envInt("SIMULATIONS", 200);
    const int samplingPlies = bot::envInt("SAMPLING_PLIES", 20);
    const uint64_t seed = static_cast<uint64_t>(bot::envInt("SEED", 20260819));

    const bot::NetworkShape shape{
        .blocks = bot::envInt("BLOCKS", 2),
        .width = bot::envInt("WIDTH", 64),
        .heads = bot::envInt("HEADS", 4)
    };

    const int steps = bot::envInt("STEPS", 2000);
    const int batchSize = bot::envInt("BATCH", 256);
    const float rate = bot::envFloat("RATE", 1e-3f);
    const float decay = bot::envFloat("DECAY", 1e-4f);
    const std::filesystem::path out = bot::envText("OUT", "gen1.safetensors");

    bot::report("[config] sampling the first {} plies, seed {}", samplingPlies, seed);
    bot::report("[config] {} blocks, width {}, {} heads | {} steps, batch {}, rate {}, decay {}",
                shape.blocks, shape.width, shape.heads, steps, batchSize, rate, decay);
    bot::report("[config] checkpoint goes to {}", out.string());

    std::vector<bot::Sample> samples = bot::generate(games, simulations, samplingPlies, seed);

    // Held out by game, not by position: positions from one game are near-copies of
    // each other, so splitting inside a game would let the network see almost the
    // answer and the validation loss would flatter it.
    std::mt19937_64 rng{seed};
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    const size_t validationSize = order.size() / 10;
    std::ranges::shuffle(order, rng);
    const std::span<const size_t> validation{order.data(), validationSize};
    const std::span<const size_t> training{order.data() + validationSize, order.size() - validationSize};

    const bot::Batch validationBatch = bot::batchFrom(samples, validation.subspan(0, std::min<size_t>(1024, validationSize)));

    bot::Network net{shape, seed};
    std::vector<mlx::core::array> params = net.parameters();

    std::vector<int> argnums(params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    bot::report("[train] {} parameters, {} training positions, {} held out",
                net.parameterCount(), training.size(), validationSize);
    bot::report("{:>7}  {:>9}  {:>9}  {:>9}  {:>9}", "step", "policy", "value", "held.pol", "held.val");

    bot::Adam adam{params};
    std::vector<size_t> picks(batchSize);

    for (int step = 1; step <= steps; ++step)
    {
        for (size_t& pick : picks) {
            pick = training[std::uniform_int_distribution<size_t>{0, training.size() - 1}(rng)];
        }
        const bot::Batch batch = bot::batchFrom(samples, picks);

        const auto lossFn = [&](const std::vector<mlx::core::array>& p) { return bot::loss(p, shape, batch, decay); };
        auto [values, gradients] = mlx::core::value_and_grad(lossFn, argnums)(params);

        params = adam.step(params, gradients, rate);
        mlx::core::eval(params);

        if (step % 100 == 0 || step == 1)
        {
            const std::vector<mlx::core::array> held = bot::loss(params, shape, validationBatch, 0.0f);
            mlx::core::eval(values);
            mlx::core::eval(held);
            bot::report("{:>7}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}", step,
                        values[1].item<float>(), values[2].item<float>(),
                        held[1].item<float>(), held[2].item<float>());
        }
    }

    net.replaceParameters(params);
    net.save(out);
    bot::report("[train] wrote {}", out.string());
}
