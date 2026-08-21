// amoeba_train: improves a network by playing it against itself, forever.
//
//   amoeba_train <weights.safetensors>
//
// The path is required and is created from random weights if it does not exist -
// naming the file is how you say whether this is a new run or a continued one, and
// guessing at it was one silent way to resume the wrong network. Whenever a
// generation beats the one before it, the new weights are written back over that
// same file, which is the file amoeba_bot reads, so a bot running alongside picks
// the improvement up at its next game.
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
// gateResultIsSettled. Both run to the cap only when the extra work is actually buying
// something.

#include "network.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
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
template <typename... Args> void report(std::format_string<Args...> format, Args&&... args)
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
    constexpr double bytesPerGibibyte = 1024.0 * 1024.0 * 1024.0;

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);

#if defined(__APPLE__)
    const double processHighWaterBytes = static_cast<double>(usage.ru_maxrss);
#else
    const double processHighWaterBytes = static_cast<double>(usage.ru_maxrss) * 1024.0;
#endif

    report("[memory] after {}: {:.2f} GiB live, {:.2f} GiB cached, {:.2f} GiB MLX peak, "
           "{:.2f} GiB process high-water",
           phase, static_cast<double>(mlx::core::get_active_memory()) / bytesPerGibibyte,
           static_cast<double>(mlx::core::get_cache_memory()) / bytesPerGibibyte,
           static_cast<double>(mlx::core::get_peak_memory()) / bytesPerGibibyte,
           processHighWaterBytes / bytesPerGibibyte);
}

int readIntegerSetting(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr ? fallback : std::stoi(text);
}

float readFloatSetting(const char* name, float fallback)
{
    const char* text = std::getenv(name);
    return text == nullptr ? fallback : std::stof(text);
}

// The defaults are sized for an overnight run on one Mac, not for a smoke test.
// A generation at these settings is expected to take on the order of an hour;
// if it takes much longer, GAMES is the knob to turn down first, because it is
// the only one that trades directly against how many generations a night holds.
struct TrainingSettings
{
    uint64_t seed = static_cast<uint64_t>(readIntegerSetting("SEED", 20260819));

    // Read only when there is no checkpoint to load - otherwise the shape comes
    // out of the file. ~1.2M parameters, which is the size the encoder and the
    // relative-position bias were designed around.
    NetworkShape networkShape = {.blockCount = readIntegerSetting("BLOCKS", 6),
                                 .embeddingWidth = readIntegerSetting("WIDTH", 128),
                                 .attentionHeadCount = readIntegerSetting("HEADS", 8)};

    // 400 simulations is half of AlphaZero's 800, which is the usual trade on one
    // machine: the policy target is a distribution over visits, and doubling the
    // visits sharpens it far less than doubling the games broadens it. It counts
    // simulations through the root, so a re-rooted tree arrives with a good share
    // of them already spent.
    // One leaf per search, because the batch comes from the other games in flight
    // and there is nothing to gain from guessing at a second leaf before hearing
    // about the first.
    Config searchConfig = {.simulations = readIntegerSetting("SIMULATIONS", 400),
                           .batchSize = readIntegerSetting("LEAVES", 1)};

    // 512 games is roughly 60k positions. Fewer than that and each generation's
    // training set is mostly the previous generation's, so the gate compares two
    // networks that saw nearly the same data and rejects almost everything.
    //
    // Above CONCURRENT, so a slot takes on another game as its own ends rather than
    // going idle. What a game costs does not depend on when it is played - every one
    // of the 512 builds its own tree from nothing and gets its own cheap endgame - so
    // this buys games rather than adding overhead, and it keeps the batch full for
    // the bulk of the run instead of only until the first game ends.
    int selfPlayGameCount = readIntegerSetting("GAMES", 512);
    int samplingPlyCount = readIntegerSetting("SAMPLING_PLIES", 20);
    float rootNoise = readFloatSetting("NOISE", 0.25f);

    // How many games are in flight at once, which is also the batch the network
    // sees: one position per game per round. The evaluator costs 0.155 ms/position
    // at 256 against 3.84 ms on its own, so the field is what buys the throughput.
    //
    // Measure before trusting 256: a field of 64 came out at 1.82x the old code and
    // a field of 256 at parity with it, which the evaluator's own numbers say should
    // be impossible. See "Batching across games" in CLAUDE.md.
    int concurrentSelfPlayGames = readIntegerSetting("CONCURRENT", 256);

    int maximumTrainingSteps = readIntegerSetting("STEPS", 1000);
    int trainingBatchSize = readIntegerSetting("BATCH", 256);
    float learningRate = readFloatSetting("RATE", 1e-3f);
    float weightDecay = readFloatSetting("DECAY", 1e-4f);

    // About the last five generations. Older positions came from networks several
    // generations weaker and hold the current one back; keeping none of them at
    // all makes each generation overfit the games it just played. ~650 MB.
    size_t replayBufferCapacity = static_cast<size_t>(readIntegerSetting("BUFFER", 300000));

    // 200 games puts the gate's error bar at +/-3.5%, which is what it takes for
    // a 55% result to mean anything - at 40 games the bar is +/-8% and promotion
    // is close to a coin flip. It costs as much as the self-play it judges, and
    // that is the price of the only honest signal in the system.
    // 200 is the cap, not the usual cost: the gate stops as soon as the verdict is
    // settled, which for a clearly better candidate is about twenty games. The cap
    // only gets spent when the two networks are genuinely close.
    int gateGameCount = readIntegerSetting("GATE_GAMES", 200);
    float promotionThreshold = readFloatSetting("GATE", 0.55f);

    // A quarter of self-play's field, because a gate that stops early throws away
    // whatever is still in flight. gateResultIsSettled() needs twenty games in, and twenty of a
    // field of 64 land well before the other 44 - which are then abandoned.
    int concurrentGateGames = readIntegerSetting("GATE_CONCURRENT", 256);

    // Ranking two networks needs far less search than generating a training
    // target does: the visit counts are thrown away here, only the result counts.
    int gateSimulationCount = readIntegerSetting("GATE_SIMULATIONS", 200);
};

// ---------------------------------------------------------------------------
// Playing hundreds of games concurrently
// ---------------------------------------------------------------------------

struct TrainingSample
{
    amoeba::Board board;
    VisitCounts visits;
    float outcome = 0.0f;
    int gameId = 0; // unique across generations, so whole games can be held out
};

// Proportional to visits early, argmax afterwards. Without the early sampling
// every game walks the same opening and the training set is far narrower than
// its position count suggests.
uint16_t selectMoveFromVisits(const VisitCounts& visitCounts, int plyCount, int samplingPlyCount,
                              std::mt19937_64& randomEngine)
{
    if (plyCount >= samplingPlyCount)
        return bestMove(visitCounts);

    const uint64_t totalVisits = std::accumulate(visitCounts.begin(), visitCounts.end(), uint64_t{0});
    assert(totalVisits > 0);

    uint64_t remaining = std::uniform_int_distribution<uint64_t>{0, totalVisits - 1}(randomEngine);
    for (uint16_t moveId = 0; moveId < amoeba::moveIdCount; ++moveId)
    {
        if (visitCounts[moveId] > remaining)
            return moveId;
        remaining -= visitCounts[moveId];
    }
    return bestMove(visitCounts);
}

// Which network takes which colour, as indices into the field's network list.
struct NetworkPairing
{
    int whiteEvaluator;
    int blackEvaluator;
};

// One game in flight.
struct ActiveGame
{
    amoeba::Board board;
    std::vector<uint64_t> positionHistory;
    std::vector<TrainingSample> trainingSamples;
    std::mt19937_64 randomEngine;
    NetworkPairing pairing{0, 0};
    int gameId = 0;
    bool isActive = false;

    // Reported when the game ends. `trainingSamples` is one per move, so the move count is
    // already there; the evaluations have to be counted as they are absorbed.
    std::chrono::steady_clock::time_point startTime;
    long long evaluatedPositionCount = 0;

    // One tree per network, not per colour. Self-play has a single network and so
    // a single tree, which it re-roots after every ply instead of every other one -
    // that is where most of the reuse comes from, since the tree keeps what both
    // sides found. The gate has two, because a tree's statistics are worth exactly
    // what the network that produced them is.
    std::vector<Search> searches;

    // Set fresh each round: the boards this game cannot go on without, which
    // network owes it the answers, and where they sit in that network's batch.
    std::span<const amoeba::Board* const> pendingBoards;
    int evaluatorIndex = 0;
    size_t evaluationBatchOffset = 0;
};

// Plays a whole field of games at once, one simulation each per round, so that a
// single network call answers every game in flight. The batch is the size of the
// field rather than the size of whatever leaves one search could guess at, and
// nothing is stale: each descent sees the statistics its own tree ended the last
// round with, so no virtual loss is needed and none is applied.
class GameBatchRunner
{
public:
    GameBatchRunner(std::span<Evaluator* const> evaluators, Config searchConfig, int samplingPlies)
        : m_evaluators(evaluators)
        , m_searchConfig(searchConfig)
        , m_samplingPlies(samplingPlies)
    {
    }

    // Plays `games` games with `slots` of them in flight, refilling a slot as its
    // game ends so the batch stays full until the work runs out. `pairingFor(game)`
    // says who plays which colour; `finished` is handed each game as it ends, under
    // a lock, so its tally needs no synchronisation of its own. `finished` returning
    // false stops the field there and then - the games still in flight are
    // abandoned, so the tally is exactly what it was when it said stop.
    void playGames(int gameCount, int concurrentGameCount, uint64_t seed, const char* label,
                   const std::function<NetworkPairing(int)>& pairingFor,
                   const std::function<bool(int, std::vector<TrainingSample>&&)>& onGameFinished);

private:
    void initializeGame(ActiveGame&, int gameId, NetworkPairing, uint64_t seed) const;
    void advanceUntilEvaluation(ActiveGame&) const;

    std::span<Evaluator* const> m_evaluators;
    Config m_searchConfig;
    int m_samplingPlies;
};

void GameBatchRunner::initializeGame(ActiveGame& game, int gameId, NetworkPairing pairing, uint64_t seed) const
{
    game.gameId = gameId;
    game.board = amoeba::createStartingBoard();
    game.pairing = pairing;
    game.isActive = true;
    game.startTime = std::chrono::steady_clock::now();
    game.evaluatedPositionCount = 0;
    game.positionHistory.assign(1, game.board.positionHash);
    game.trainingSamples.clear();
    game.randomEngine.seed(seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(gameId) + 1)));

    // Everything random about a game comes from its id, never from which slot or
    // thread happened to pick it up, so a whole run stays reproducible from SEED
    // however the field interleaves.
    game.searches.clear();
    for (size_t evaluatorIndex = 0; evaluatorIndex < m_evaluators.size(); ++evaluatorIndex)
    {
        Config config = m_searchConfig;
        config.noiseSeed = seed + static_cast<uint64_t>(gameId);
        game.searches.emplace_back(config);
    }
    for (Search& search : game.searches)
        search.restart(game.board, game.positionHistory);
}

// Plays as far as it can without an evaluation, leaving the boards it is waiting
// on in `pendingBoards` - or leaving it empty, which means the game is over.
void GameBatchRunner::advanceUntilEvaluation(ActiveGame& game) const
{
    while (game.board.state == amoeba::State::ongoing)
    {
        const int evaluatorIndex = game.board.whiteToMove ? game.pairing.whiteEvaluator : game.pairing.blackEvaluator;
        Search& search = game.searches[static_cast<size_t>(evaluatorIndex)];

        const std::span<const amoeba::Board* const> pendingBoards = search.pendingLeaves();
        if (!pendingBoards.empty())
        {
            game.evaluatorIndex = evaluatorIndex;
            game.pendingBoards = pendingBoards;
            return;
        }

        const VisitCounts visitCounts = search.visits();
        game.trainingSamples.push_back({game.board, visitCounts, 0.0f, game.gameId});

        const uint16_t chosenMoveId =
            selectMoveFromVisits(visitCounts, game.board.plyCount, m_samplingPlies, game.randomEngine);
        game.board = amoeba::applyMove(game.board, amoeba::Move::fromId(chosenMoveId), game.positionHistory);
        game.positionHistory.push_back(game.board.positionHash);

        // Every tree follows the game, not just the one that was searching: a tree
        // can only keep a subtree while its root is where the game is.
        for (Search& searchToAdvance : game.searches)
            searchToAdvance.advance(chosenMoveId, game.board, game.positionHistory);
    }
    game.pendingBoards = {};
}

void GameBatchRunner::playGames(int gameCount, int concurrentGameCount, uint64_t seed, const char* label,
                                const std::function<NetworkPairing(int)>& pairingFor,
                                const std::function<bool(int, std::vector<TrainingSample>&&)>& onGameFinished)
{
    if (gameCount <= 0)
        return;

    std::vector<ActiveGame> activeGames(static_cast<size_t>(std::max(1, std::min(gameCount, concurrentGameCount))));

    std::mutex completionMutex;
    std::atomic<bool> shouldContinue{true};
    int completedGameCount = 0;
    int startedGameCount = 0;

    // The field is walked in two halves that take turns: while one half's batch is on
    // the device, the other half's trees are walked on the pool. A round used to be
    // every descent with the device idle and then one network call with every core
    // idle - 15 ms and 40 ms of a 55 ms round, neither overlapping the other.
    struct Half
    {
        size_t firstGameIndex = 0;
        size_t gameCount = 0;
        std::vector<std::vector<const amoeba::Board*>> pendingBoardsByEvaluator;
        std::vector<std::vector<Evaluation>> evaluationsByEvaluator;
        size_t gatheredBoardCount = 0;
    };

    std::array<Half, 2> halves;
    halves[0].gameCount = activeGames.size() - activeGames.size() / 2;
    halves[1].firstGameIndex = halves[0].gameCount;
    halves[1].gameCount = activeGames.size() / 2;
    for (Half& half : halves)
    {
        half.pendingBoardsByEvaluator.resize(m_evaluators.size());
        half.evaluationsByEvaluator.resize(m_evaluators.size());
    }

    report("[{}] {} games, {} in flight in halves of {} and {}, at {} simulations, {} leaves per search", label,
           gameCount, activeGames.size(), halves[0].gameCount, halves[1].gameCount, m_searchConfig.simulations,
           m_searchConfig.batchSize);

    for (ActiveGame& game : activeGames)
    {
        initializeGame(game, startedGameCount, pairingFor(startedGameCount), seed);
        ++startedGameCount;
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
    const auto handOver = [&](ActiveGame& game)
    {
        for (TrainingSample& sample : game.trainingSamples)
        {
            sample.outcome = outcomeFor(game.board.state, sample.board.whiteToMove);
        }

        const std::lock_guard guard{completionMutex};

        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - game.startTime).count();
        const size_t moves = game.trainingSamples.size();

        if (!onGameFinished(game.gameId, std::move(game.trainingSamples)))
            shouldContinue = false;
        ++completedGameCount;

        report("[{}] game {}: {:.0f}s, {} moves, {} network calls -- {} games left", label, game.gameId, seconds, moves,
               game.evaluatedPositionCount, gameCount - completedGameCount);

        if (!shouldContinue || startedGameCount >= gameCount)
        {
            game.isActive = false;
            return;
        }
        initializeGame(game, startedGameCount, pairingFor(startedGameCount), seed);
        ++startedGameCount;
    };

    // Absorbs what the device returned for this slot last time round, then takes the
    // game as far as it can before it needs an answer again.
    const auto stepOne = [&](Half& half, size_t slot)
    {
        ActiveGame& game = activeGames[slot];
        if (!game.isActive)
            return;

        if (!game.pendingBoards.empty())
        {
            game.evaluatedPositionCount += static_cast<long long>(game.pendingBoards.size());
            game.searches[static_cast<size_t>(game.evaluatorIndex)].absorb(
                std::span{half.evaluationsByEvaluator[static_cast<size_t>(game.evaluatorIndex)]}.subspan(
                    game.evaluationBatchOffset, game.pendingBoards.size()));
        }

        advanceUntilEvaluation(game);
        while (game.isActive && game.pendingBoards.empty())
        {
            handOver(game);
            if (game.isActive)
                advanceUntilEvaluation(game);
        }
    };

    // Collects one half's waiting boards into one batch per network, remembering
    // where each game's own boards landed so absorb() can find them again.
    const auto gather = [&](Half& half)
    {
        half.gatheredBoardCount = 0;
        for (std::vector<const amoeba::Board*>& evaluatorBatch : half.pendingBoardsByEvaluator)
        {
            evaluatorBatch.clear();
        }

        for (size_t slot = half.firstGameIndex; slot < half.firstGameIndex + half.gameCount; ++slot)
        {
            ActiveGame& game = activeGames[slot];
            if (!game.isActive || game.pendingBoards.empty())
                continue;

            std::vector<const amoeba::Board*>& evaluatorBatch =
                half.pendingBoardsByEvaluator[static_cast<size_t>(game.evaluatorIndex)];
            game.evaluationBatchOffset = evaluatorBatch.size();
            evaluatorBatch.insert(evaluatorBatch.end(), game.pendingBoards.begin(), game.pendingBoards.end());
            half.gatheredBoardCount += game.pendingBoards.size();
        }
    };

    Half* stepping = nullptr;
    Half* evaluating = nullptr;

    // Task 0 is the device call, tasks 1.. are the other half's trees, all in one
    // round of the pool. No extra thread, and only ever one thread inside MLX: its
    // own encode is a nested forEach and so runs serially there, which costs 0.3 ms
    // against a call of twenty.
    const std::function<void(size_t)> round = [&](size_t index)
    {
        if (index == 0)
        {
            for (size_t evaluatorIndex = 0; evaluatorIndex < m_evaluators.size(); ++evaluatorIndex)
            {
                if (evaluating->pendingBoardsByEvaluator[evaluatorIndex].empty())
                    continue;
                evaluating->evaluationsByEvaluator[evaluatorIndex].resize(
                    evaluating->pendingBoardsByEvaluator[evaluatorIndex].size());
                m_evaluators[evaluatorIndex]->evaluate(evaluating->pendingBoardsByEvaluator[evaluatorIndex],
                                                       evaluating->evaluationsByEvaluator[evaluatorIndex]);
            }
            return;
        }
        stepOne(*stepping, stepping->firstGameIndex + index - 1);
    };

    // One half has to have something to evaluate before the two can start taking
    // turns; there is nothing to overlap this once.
    ThreadPool::global().forEach(halves[0].gameCount,
                                 [&](size_t gameOffset) { stepOne(halves[0], halves[0].firstGameIndex + gameOffset); });
    gather(halves[0]);

    for (size_t turn = 0;; turn ^= 1)
    {
        evaluating = &halves[turn];
        stepping = &halves[turn ^ 1];

        ThreadPool::global().forEach(1 + stepping->gameCount, round);
        if (!shouldContinue)
            return;

        gather(*stepping);

        // Each half is evaluated on one turn and stepped on the next, so nothing is
        // left holding an answer nobody absorbed when both come up empty.
        if (halves[0].gatheredBoardCount == 0 && halves[1].gatheredBoardCount == 0)
            return;
    }
}

std::vector<TrainingSample> generateSelfPlaySamples(const Network& champion, const TrainingSettings& settings, uint64_t seed)
{
    Config searchConfig = settings.searchConfig;
    searchConfig.rootNoise = settings.rootNoise;

    NetworkEvaluator evaluator{champion};
    Evaluator* const evaluators[]{&evaluator};

    std::vector<TrainingSample> trainingSamples;
    int whiteWins = 0, blackWins = 0, draws = 0;

    GameBatchRunner gameRunner{evaluators, searchConfig, settings.samplingPlyCount};
    gameRunner.playGames(
        settings.selfPlayGameCount, settings.concurrentSelfPlayGames, seed, "selfplay",
        [](int) { return NetworkPairing{0, 0}; },
        [&](int, std::vector<TrainingSample>&& played)
        {
            // The last sample's mover lost, drew, or was adjudicated against;
            // read the result off it rather than threading the final Board out.
            const float last = played.back().outcome;
            if (last == 0.0f)
                ++draws;
            else if (played.back().board.whiteToMove == (last > 0.0f))
                ++whiteWins;
            else
                ++blackWins;

            trainingSamples.insert(trainingSamples.end(), std::make_move_iterator(played.begin()), std::make_move_iterator(played.end()));
            return true; // self-play always plays every game it was asked for
        });

    report("[selfplay] {} positions from {} games: {} White, {} Black, {} drawn", trainingSamples.size(),
           settings.selfPlayGameCount, whiteWins, blackWins, draws);
    return trainingSamples;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

TrainingBatch createBatchFromSamples(const std::vector<TrainingSample>& samples, std::span<const size_t> picks)
{
    std::vector<const amoeba::Board*> boards;
    std::vector<VisitCounts> visits;
    std::vector<float> outcomes;
    boards.reserve(picks.size());
    visits.reserve(picks.size());
    outcomes.reserve(picks.size());

    for (const size_t sampleIndex : picks)
    {
        boards.push_back(&samples[sampleIndex].board);
        visits.push_back(samples[sampleIndex].visits);
        outcomes.push_back(samples[sampleIndex].outcome);
    }
    return makeTrainingBatch(boards, visits, outcomes);
}

// Holds out whole games. Positions within a game are near-copies of each other and
// every one carries the same outcome label, so splitting by position leaves a
// held-out position's own game in the training set and the value head can score
// well by recognising the game rather than by reading the board. The symptom is a
// held-out loss that improves as you generate *fewer* games.
struct DatasetSplit
{
    std::vector<size_t> training;
    std::vector<size_t> validation;
};

DatasetSplit splitDatasetByGame(const std::vector<TrainingSample>& samples, std::mt19937_64& randomEngine)
{
    std::set<int> ids;
    for (const TrainingSample& sample : samples)
    {
        ids.insert(sample.gameId);
    }

    std::vector<int> order(ids.begin(), ids.end());
    std::ranges::shuffle(order, randomEngine);
    const size_t heldOutCount = std::max<size_t>(1, order.size() / 10);
    const std::set<int> heldOut(order.begin(), order.begin() + static_cast<long>(heldOutCount));

    DatasetSplit split;
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex)
    {
        (heldOut.contains(samples[sampleIndex].gameId) ? split.validation : split.training).push_back(sampleIndex);
    }
    return split;
}

// Leaves `candidate` holding the weights from the best step, not the last. Held-out loss
// turns back up well before training ends, so keeping the final parameters ships a
// network measurably worse than one already in hand.
void trainCandidate(Network& candidate, const std::vector<TrainingSample>& samples, const TrainingSettings& settings,
                    uint64_t seed)
{
    std::mt19937_64 randomEngine{seed};
    const DatasetSplit split = splitDatasetByGame(samples, randomEngine);
    if (split.training.empty() || split.validation.empty())
        throw std::runtime_error("not enough games to hold any out");

    const TrainingBatch validation = createBatchFromSamples(
        samples, std::span{split.validation}.subspan(0, std::min<size_t>(1024, split.validation.size())));

    std::vector<mlx::core::array> parameters = candidate.parameters();
    std::vector<int> argnums(parameters.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    report("[train] {} parameters, {} positions, {} held out, {} steps at batch {}", candidate.parameterCount(),
           split.training.size(), split.validation.size(), settings.maximumTrainingSteps, settings.trainingBatchSize);
    report("{:>7}  {:>9}  {:>9}  {:>9}  {:>9}", "step", "policy", "value", "held.pol", "held.val");

    // Checked four times more often than it is printed, because the optimum moves
    // as the replay buffer fills: at 22k positions it lands around step 100, and
    // at the buffer's full 200k the same step count is barely one pass over the
    // data and the optimum is far later. Stopping on patience rather than on a
    // fixed count is what lets one STEPS serve both.
    constexpr int validationCheckInterval = 25;
    constexpr int validationPatience = 8;

    Adam adam{parameters};
    std::vector<size_t> picks(static_cast<size_t>(settings.trainingBatchSize));
    std::vector<mlx::core::array> bestParameters = parameters;
    float lowestValidationLoss = std::numeric_limits<float>::max();
    int bestStep = 0;
    int checksWithoutImprovement = 0;

    for (int step = 1; step <= settings.maximumTrainingSteps; ++step)
    {
        for (size_t& pick : picks)
        {
            pick = split.training[std::uniform_int_distribution<size_t>{0, split.training.size() - 1}(randomEngine)];
        }
        const TrainingBatch batch = createBatchFromSamples(samples, picks);

        const auto lossFunction = [&](const std::vector<mlx::core::array>& currentParameters)
        { return computeLoss(currentParameters, candidate.shape(), batch, settings.weightDecay); };
        auto [values, gradients] = mlx::core::value_and_grad(lossFunction, argnums)(parameters);

        parameters = adam.updateParameters(parameters, gradients, settings.learningRate);
        mlx::core::eval(parameters);

        if (step % validationCheckInterval != 0 && step != 1)
            continue;

        const std::vector<mlx::core::array> held = computeLoss(parameters, candidate.shape(), validation, 0.0f);
        mlx::core::eval(values);
        mlx::core::eval(held);

        // Judged on policy and value together: one orders the moves, the other
        // evaluates the leaves, and the search needs both.
        const float combined = held[1].item<float>() + held[2].item<float>();
        if (combined < lowestValidationLoss)
        {
            lowestValidationLoss = combined;
            bestStep = step;
            bestParameters = parameters;
            checksWithoutImprovement = 0;
        }
        else
        {
            ++checksWithoutImprovement;
        }

        if (step % 100 == 0 || step == 1)
            report("{:>7}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}", step, values[1].item<float>(),
                   values[2].item<float>(), held[1].item<float>(), held[2].item<float>());

        if (checksWithoutImprovement >= validationPatience)
        {
            report("[train] held-out loss has not improved in {} checks, stopping at step {}", validationPatience,
                   step);
            break;
        }
    }

    report("[train] best held-out loss {:.4f} at step {} of {}, keeping those weights", lowestValidationLoss, bestStep,
           settings.maximumTrainingSteps);
    candidate.replaceParameters(bestParameters);
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
bool gateResultIsSettled(int played, double score, float threshold)
{
    constexpr int minimumGameCount = 20;
    return played >= minimumGameCount && std::abs(score - static_cast<double>(threshold)) > 3.0 * standardError(score, played);
}

// The candidate's score, draws counted as half. Colours alternate so the
// first-move advantage cancels rather than being handed to whoever is listed
// first, and the early sampling in chooseMove is what makes the games differ -
// two argmax players would replay one game GATE_GAMES times.
//
// Ranks with fewer simulations than self-play used, because only the result is
// read here - the visit counts that needed the deeper search are discarded.
double evaluateCandidate(const Network& candidate, const Network& champion, const TrainingSettings& settings, uint64_t seed)
{
    Config searchConfig = settings.searchConfig;
    searchConfig.simulations = settings.gateSimulationCount;
    searchConfig.rootNoise = 0.0f; // competition keeps the network's own opinion

    NetworkEvaluator candidateEval{candidate};
    NetworkEvaluator championEval{champion};
    Evaluator* const evaluators[]{&candidateEval, &championEval};

    int wins = 0, draws = 0, losses = 0;

    // A smaller field than self-play, because everything still in flight when the
    // verdict settles is work thrown away: gateResultIsSettled() is consulted on each game that
    // comes in, and the twentieth of 64 lands long before the rest.
    GameBatchRunner gameRunner{evaluators, searchConfig, settings.samplingPlyCount};
    gameRunner.playGames(
        settings.gateGameCount, settings.concurrentGateGames, seed, "gate",
        [](int game) { return game % 2 == 0 ? NetworkPairing{0, 1} : NetworkPairing{1, 0}; },
        [&](int game, std::vector<TrainingSample>&& played)
        {
            // The final sample is from the point of view of the side that lost or drew.
            const TrainingSample& last = played.back();
            const bool whiteWon = last.outcome < 0.0f ? !last.board.whiteToMove : last.board.whiteToMove;

            if (last.outcome == 0.0f)
                ++draws;
            else if (whiteWon == (game % 2 == 0))
                ++wins;
            else
                ++losses;

            const int total = wins + draws + losses;
            const double score = (wins + 0.5 * draws) / total;
            return !gateResultIsSettled(total, score, settings.promotionThreshold);
        });

    const int total = wins + draws + losses;
    const double score = (wins + 0.5 * draws) / total;

    report("[gate] candidate {:.1f}% +/- {:.1f}% ({}-{}-{}) over {} games at {} simulations", 100.0 * score,
           100.0 * standardError(score, total), wins, draws, losses, total, settings.gateSimulationCount);
    return score;
}

// ---------------------------------------------------------------------------

} // namespace

} // namespace bot

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::println(stderr, "usage: amoeba_train <weights.safetensors>");
        std::println(stderr, "  created from random weights if it does not exist, and written back to");
        std::println(stderr, "  whenever a generation beats the one before it");
        return EXIT_FAILURE;
    }

    const bot::TrainingSettings settings;
    const std::filesystem::path weights{argv[1]};

    // Generation 0 is random weights, written straight away so that amoeba_bot has
    // something to load while the first generation is still being played.
    std::unique_ptr<bot::Network> champion;
    if (std::filesystem::exists(weights))
    {
        champion = std::make_unique<bot::Network>(weights);
        bot::report("[train] resuming from {}: {} blocks, width {}, {} heads, {} parameters", weights.string(),
                    champion->shape().blockCount, champion->shape().embeddingWidth,
                    champion->shape().attentionHeadCount, champion->parameterCount());
    }
    else
    {
        champion = std::make_unique<bot::Network>(settings.networkShape, settings.seed);
        champion->save(weights);
        bot::report("[train] {} did not exist: started from random weights, {} parameters, saved", weights.string(), champion->parameterCount());
    }

    std::vector<bot::TrainingSample> replay;
    int gameIdBase = 0;

    for (int generation = 1;; ++generation)
    {
        bot::report("");
        bot::report("======== generation {} ========", generation);

        std::vector<bot::TrainingSample> fresh = bot::generateSelfPlaySamples(*champion, settings, settings.seed + static_cast<uint64_t>(generation) * 1000);

        // Game ids have to stay unique across generations, or splitDatasetByGame will hold
        // out a game from this generation and train on a different one with the same
        // id from the last.
        for (bot::TrainingSample& sample : fresh)
            sample.gameId += gameIdBase;
        gameIdBase += settings.selfPlayGameCount;

        bot::reportMemory("self-play");

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()), std::make_move_iterator(fresh.end()));
        if (replay.size() > settings.replayBufferCapacity)
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(replay.size() - settings.replayBufferCapacity));
        bot::report("[train] replay buffer holds {} positions", replay.size());

        // The candidate starts from the best weights rather than from scratch: each
        // generation is meant to refine, not to relearn the game.
        bot::Network candidate = *champion;
        bot::trainCandidate(candidate, replay, settings, settings.seed + static_cast<uint64_t>(generation));
        bot::reportMemory("training");

        const double score = bot::evaluateCandidate(candidate, *champion, settings, settings.seed + 7777 + static_cast<uint64_t>(generation));
        bot::reportMemory("the gate");

        if (score < settings.promotionThreshold)
        {
            bot::report("[train] generation {} rejected at {:.1f}%, keeping the previous weights", generation, 100.0 * score);
        }
        else
        {
            *champion = candidate;
            champion->save(weights);
            bot::report("[train] generation {} PROMOTED at {:.1f}%, wrote {}", generation, 100.0 * score, weights.string());
        }
    }
}
