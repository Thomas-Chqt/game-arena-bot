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
//
// MODE=match instead loads CHECKPOINT and plays PLAYER against OPPONENT over GAMES
// games, each being one of random, rollout or network, with LEAVES collected per
// batched evaluation. Nothing is generated or trained. This is the honest test: rollout MCTS beat random play 97.5% of the time
// at 200 simulations, so a network that has learned anything should be in that
// region, and a network that has learned a better leaf evaluation should beat
// rollout MCTS head to head.

#include "training.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <memory>
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
    int game = 0;
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

            std::vector<Sample> played = playGame(search, samplingPlies, rng);
            for (Sample& sample : played) {
                sample.game = game;
            }
            perGame[static_cast<size_t>(game)] = std::move(played);

            const int done = completed.fetch_add(1) + 1;
            if (done == 1 || done % 25 == 0 || done == games)
            {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                report("[generate] {}/{} games, {:.0f}s elapsed, {:.0f}s left", done, games, elapsed, elapsed / done * (games - done));
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
        if (last == 0.0f)
            ++draws;
        else if (game.back().board.whiteToMove == (last > 0.0f))
            ++whiteWins;
        else
            ++blackWins;
    }

    report("[generate] {} positions from {} games: {} White, {} Black, {} drawn", samples.size(), games, whiteWins, blackWins, draws);
    return samples;
}

} // namespace

// ---------------------------------------------------------------------------
// Playing a match
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
    SearchPlayer(Evaluator& evaluator, int simulations, int leaves)
        : m_search(evaluator, {.simulations = simulations, .batchSize = leaves})
    {
    }

    uint16_t pick(const amoeba::Board& board, std::span<const uint64_t> history) override
    {
        return bestMove(m_search.run(board, history));
    }

private:
    Search m_search;
};

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

// A player named by a string, so a match can be spelled out with two env vars.
// Returns nothing for "random"; the caller owns whatever the network needs.
std::unique_ptr<Player> makePlayer(const std::string& kind, int simulations, int leaves, uint64_t seed,
                                   std::vector<std::unique_ptr<Evaluator>>& owned, const Network& net)
{
    if (kind == "random")
        return std::make_unique<RandomPlayer>(seed);

    if (kind == "rollout")
        owned.push_back(std::make_unique<RolloutEvaluator>(seed));
    else if (kind == "network")
        owned.push_back(std::make_unique<NetworkEvaluator>(net));
    else
        throw std::runtime_error(std::format("PLAYER/OPPONENT must be random, rollout or network, not {}", kind));

    // A rollout gains nothing from batching, so only the network asks for leaves.
    return std::make_unique<SearchPlayer>(*owned.back(), simulations, kind == "network" ? leaves : 1);
}

// Alternates colours so the first-move advantage cancels out rather than being
// handed to whichever side happens to be listed first.
void runMatch(const std::string& playerKind, const std::string& opponentKind, int games, int simulations,
              int leaves, uint64_t seed, const Network& net)
{
    report("[match] {} vs {} over {} games at {} simulations, {} leaves per batch, alternating colours",
           playerKind, opponentKind, games, simulations, leaves);

    int wins = 0, losses = 0, draws = 0, plies = 0;
    const auto start = std::chrono::steady_clock::now();

    for (int game = 0; game < games; ++game)
    {
        std::vector<std::unique_ptr<Evaluator>> owned;
        const std::unique_ptr<Player> player =
            makePlayer(playerKind, simulations, leaves, seed + static_cast<uint64_t>(game), owned, net);
        const std::unique_ptr<Player> opponent =
            makePlayer(opponentKind, simulations, leaves, seed + 0xabcdef + static_cast<uint64_t>(game), owned, net);

        const bool playerIsWhite = game % 2 == 0;
        const int result = playerIsWhite ? playMatch(*player, *opponent, plies)
                                         : playMatch(*opponent, *player, plies);
        const int forPlayer = playerIsWhite ? result : -result;

        if (forPlayer > 0) ++wins;
        else if (forPlayer < 0) ++losses;
        else ++draws;

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        report("[match] {}/{}: {}-{}-{}  ({:.0f}s elapsed, {:.0f}s left)", game + 1, games, wins, losses, draws,
               elapsed, elapsed / (game + 1) * (games - game - 1));
    }

    // Counting a draw as half a win, the usual convention for a match score.
    const double score = (wins + 0.5 * draws) / games;
    // Standard error of a proportion, so a result is not read as more precise than it is.
    const double error = std::sqrt(score * (1.0 - score) / games);
    report("[match] {} scored {:.1f}% +/- {:.1f}% ({}-{}-{}) over {} games, {} plies average",
           playerKind, 100.0 * score, 100.0 * error, wins, losses, draws, games, plies / games);
}

} // namespace bot

int main()
{
    const std::string mode = bot::envText("MODE", "train");
    const int games = bot::envInt("GAMES", 300);
    const int simulations = bot::envInt("SIMULATIONS", 200);
    const int samplingPlies = bot::envInt("SAMPLING_PLIES", 20);
    const uint64_t seed = static_cast<uint64_t>(bot::envInt("SEED", 20260819));

    const bot::NetworkShape shape{
        .blocks = bot::envInt("BLOCKS", 2),
        .width = bot::envInt("WIDTH", 64),
        .heads = bot::envInt("HEADS", 4)
    };

    const int steps = bot::envInt("STEPS", 600);
    const int batchSize = bot::envInt("BATCH", 256);
    const float rate = bot::envFloat("RATE", 1e-3f);
    const float decay = bot::envFloat("DECAY", 1e-4f);
    const std::filesystem::path out = bot::envText("OUT", "gen1.safetensors");

    // MODE=match loads a checkpoint and plays it; nothing is generated or trained.
    if (mode == "match")
    {
        const std::filesystem::path checkpoint = bot::envText("CHECKPOINT", "gen1.safetensors");
        const bot::Network net{checkpoint};
        bot::report("[config] loaded {}: {} blocks, width {}, {} heads, {} parameters", checkpoint.string(),
                    net.shape().blocks, net.shape().width, net.shape().heads, net.parameterCount());

        bot::runMatch(bot::envText("PLAYER", "network"), bot::envText("OPPONENT", "random"),
                      bot::envInt("GAMES", 20), simulations, bot::envInt("LEAVES", 16), seed, net);
        return 0;
    }

    bot::report("[config] sampling the first {} plies, seed {}", samplingPlies, seed);
    bot::report("[config] {} blocks, width {}, {} heads | {} steps, batch {}, rate {}, decay {}", shape.blocks, shape.width, shape.heads, steps, batchSize, rate, decay);
    bot::report("[config] checkpoint goes to {}", out.string());

    std::vector<bot::Sample> samples = bot::generate(games, simulations, samplingPlies, seed);

    // Hold out whole games. Positions within one game are near-copies of each other
    // and every one of them carries the same outcome label, so splitting by position
    // leaves a held-out position's own game in the training set and the value head
    // scores well by recognising the game rather than by reading the board. That
    // makes the held-out loss look better the fewer games you generate, which is
    // exactly backwards.
    std::mt19937_64 rng{seed};
    std::vector<int> gameOrder(static_cast<size_t>(games));
    std::iota(gameOrder.begin(), gameOrder.end(), 0);
    std::ranges::shuffle(gameOrder, rng);

    std::vector<bool> heldOut(static_cast<size_t>(games), false);
    for (int i = 0; i < std::max(1, games / 10); ++i) {
        heldOut[static_cast<size_t>(gameOrder[static_cast<size_t>(i)])] = true;
    }

    std::vector<size_t> training, validation;
    for (size_t i = 0; i < samples.size(); ++i) {
        (heldOut[static_cast<size_t>(samples[i].game)] ? validation : training).push_back(i);
    }

    const bot::Batch validationBatch = bot::batchFrom(samples, std::span{validation}.subspan(0, std::min<size_t>(1024, validation.size())));

    bot::Network net{shape, seed};
    std::vector<mlx::core::array> params = net.parameters();

    std::vector<int> argnums(params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    bot::report("[train] {} parameters, {} positions from {} games, {} held out from {} games",
                net.parameterCount(), training.size(), games - std::max(1, games / 10),
                validation.size(), std::max(1, games / 10));
    bot::report("{:>7}  {:>9}  {:>9}  {:>9}  {:>9}", "step", "policy", "value", "held.pol", "held.val");

    bot::Adam adam{params};
    std::vector<size_t> picks(static_cast<size_t>(batchSize));
    // Keep the weights from the best step, not the last. Held-out loss turns back
    // up well before training ends, so saving the final parameters ships a network
    // measurably worse than one we already had. Copying the vector is cheap - an
    // mlx::core::array is a handle, and these are already evaluated.
    std::vector<mlx::core::array> best = params;
    float bestHeld = std::numeric_limits<float>::max();
    int bestStep = 0;

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
            // Judge on policy and value together: both feed the search, one orders
            // the moves and the other evaluates the leaves.
            const float combined = held[1].item<float>() + held[2].item<float>();
            if (combined < bestHeld)
            {
                bestHeld = combined;
                bestStep = step;
                best = params;
            }
            bot::report("{:>7}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}", step, values[1].item<float>(), values[2].item<float>(), held[1].item<float>(), held[2].item<float>());
        }
    }

    bot::report("[train] best held-out loss {:.4f} at step {} of {}, keeping those weights", bestHeld, bestStep, steps);

    net.replaceParameters(best);
    net.save(out);
    bot::report("[train] wrote {}", out.string());
}
