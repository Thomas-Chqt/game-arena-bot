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
//   generating  SEED GAMES CONCURRENT SIMULATIONS SAMPLING_PLIES NOISE
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
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace amoeba
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

    // 400 simulations is half of AlphaZero's 800, which is the usual trade on one
    // machine: the policy target is a distribution over visits, and doubling the
    // visits sharpens it far less than doubling the games broadens it. Every move
    // starts a fresh search and receives this full budget.
    MCTSConfig searchConfig = {.simulations = readIntegerSetting("SIMULATIONS", 400)};

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
    std::optional<amoeba::Outcome> outcome;
    std::vector<uint64_t> positionHistory;
    std::vector<TrainingSample> trainingSamples;
    std::mt19937_64 randomEngine;
    std::mt19937_64 searchRandomEngine;
    NetworkPairing pairing{0, 0};
    int gameId = 0;
    bool isActive = false;

    // Reported when the game ends. `trainingSamples` is one per move, so the move count is
    // already there; the evaluations have to be counted as they are absorbed.
    std::chrono::steady_clock::time_point startTime;
    long long evaluatedPositionCount = 0;

    // One fresh tree for the current move. The caller chooses which network
    // evaluates its leaves, so gate games do not need one tree per network.
    std::optional<MCTS> search;

    // Set fresh each round: the leaf this game cannot go on without, which
    // network owes it the answer, and where it sits in that network's batch.
    const amoeba::Board* pendingLeaf = nullptr;
    int evaluatorIndex = 0;
    size_t evaluationBatchOffset = 0;
};

// Plays a field of independent games in lockstep. Each search offers one leaf per
// round, and the runner batches those leaves by evaluator before continuing.
class GameBatchRunner
{
public:
    GameBatchRunner(std::span<NetworkEvaluator* const> evaluators, MCTSConfig searchConfig, int samplingPlies)
        : m_evaluators(evaluators)
        , m_searchConfig(searchConfig)
        , m_samplingPlies(samplingPlies)
    {
    }

    // Plays `games` games with `slots` of them in flight, refilling a slot as its
    // game ends so the batch stays full until the work runs out. `pairingFor(game)`
    // says who plays which colour. `finished` returning false stops the field there
    // and then; the other games still in flight are abandoned.
    void playGames(int gameCount, int concurrentGameCount, uint64_t seed, const char* label,
                   const std::function<NetworkPairing(int)>& pairingFor,
                   const std::function<bool(int, std::vector<TrainingSample>&&)>& onGameFinished);

private:
    void initializeGame(ActiveGame&, int gameId, NetworkPairing, uint64_t seed) const;
    void startSearch(ActiveGame&) const;
    void advanceUntilEvaluation(ActiveGame&) const;

    std::span<NetworkEvaluator* const> m_evaluators;
    MCTSConfig m_searchConfig;
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
    game.outcome.reset();
    game.trainingSamples.clear();
    game.randomEngine.seed(seed ^ (0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(gameId) + 1)));
    game.searchRandomEngine.seed(seed + static_cast<uint64_t>(gameId));

    // The caller owns noise-seed progression because each MCTS exists for one
    // position only. It stays separate from move sampling so changing one random
    // process does not perturb the other.
    startSearch(game);
    game.pendingLeaf = nullptr;
}

void GameBatchRunner::startSearch(ActiveGame& game) const
{
    MCTSConfig config = m_searchConfig;
    config.noiseSeed = game.searchRandomEngine();
    game.search.emplace(game.board, game.positionHistory, config);
}

// Plays as far as it can without an evaluation, leaving the leaf it is waiting on
// in `pendingLeaf` - or leaving it null, which means the game is over.
void GameBatchRunner::advanceUntilEvaluation(ActiveGame& game) const
{
    while (!game.outcome.has_value())
    {
        const int evaluatorIndex = game.board.whiteToMove ? game.pairing.whiteEvaluator : game.pairing.blackEvaluator;
        if (const amoeba::Board* leaf = game.search->pendingLeaf())
        {
            game.evaluatorIndex = evaluatorIndex;
            game.pendingLeaf = leaf;
            return;
        }

        const VisitCounts visitCounts = game.search->visits();
        game.trainingSamples.push_back({game.board, visitCounts, 0.0f, game.gameId});

        const uint16_t chosenMoveId =
            selectMoveFromVisits(visitCounts, game.board.plyCount, m_samplingPlies, game.randomEngine);
        amoeba::MoveResult result =
            amoeba::applyMove(game.board, amoeba::Move::fromId(chosenMoveId), game.positionHistory);
        if (const auto* outcome = std::get_if<amoeba::Outcome>(&result))
        {
            game.outcome = *outcome;
            break;
        }

        game.board = std::get<amoeba::Board>(std::move(result));
        game.positionHistory.push_back(game.board.positionHash);
        startSearch(game);
    }
    game.pendingLeaf = nullptr;
}

void GameBatchRunner::playGames(int gameCount, int concurrentGameCount, uint64_t seed, const char* label,
                                const std::function<NetworkPairing(int)>& pairingFor,
                                const std::function<bool(int, std::vector<TrainingSample>&&)>& onGameFinished)
{
    if (gameCount <= 0)
        return;

    std::vector<ActiveGame> activeGames(static_cast<size_t>(std::max(1, std::min(gameCount, concurrentGameCount))));

    bool shouldContinue = true;
    int completedGameCount = 0;
    int startedGameCount = 0;

    std::vector<std::vector<const amoeba::Board*>> pendingBoardsByEvaluator(m_evaluators.size());
    std::vector<std::vector<Evaluation>> evaluationsByEvaluator(m_evaluators.size());

    report("[{}] {} games, {} in flight, at {} simulations", label, gameCount, activeGames.size(),
           m_searchConfig.simulations);

    for (ActiveGame& game : activeGames)
    {
        initializeGame(game, startedGameCount, pairingFor(startedGameCount), seed);
        ++startedGameCount;
        advanceUntilEvaluation(game);
    }

    // Hands a finished game over and refills its slot if more work remains.
    const auto handOver = [&](ActiveGame& game)
    {
        for (TrainingSample& sample : game.trainingSamples)
        {
            assert(game.outcome.has_value());
            sample.outcome = outcomeFor(*game.outcome, sample.board.whiteToMove);
        }

        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - game.startTime).count();
        const size_t moves = game.trainingSamples.size();

        if (!onGameFinished(game.gameId, std::move(game.trainingSamples)))
            shouldContinue = false;
        ++completedGameCount;

        report("[{}] game {}: {:.0f}s, {} moves, {} evaluated positions -- {} games left", label, game.gameId,
               seconds, moves,
               game.evaluatedPositionCount, gameCount - completedGameCount);

        if (!shouldContinue || startedGameCount >= gameCount)
        {
            game.isActive = false;
            return;
        }
        initializeGame(game, startedGameCount, pairingFor(startedGameCount), seed);
        ++startedGameCount;
        advanceUntilEvaluation(game);
    };

    for (;;)
    {
        size_t gatheredBoardCount = 0;
        for (std::vector<const amoeba::Board*>& batch : pendingBoardsByEvaluator)
            batch.clear();

        for (ActiveGame& game : activeGames)
        {
            if (!game.isActive || game.pendingLeaf == nullptr)
                continue;

            std::vector<const amoeba::Board*>& batch =
                pendingBoardsByEvaluator[static_cast<size_t>(game.evaluatorIndex)];
            game.evaluationBatchOffset = batch.size();
            batch.push_back(game.pendingLeaf);
            ++gatheredBoardCount;
        }

        if (gatheredBoardCount == 0)
            return;

        for (size_t evaluatorIndex = 0; evaluatorIndex < m_evaluators.size(); ++evaluatorIndex)
        {
            if (pendingBoardsByEvaluator[evaluatorIndex].empty())
                continue;
            evaluationsByEvaluator[evaluatorIndex].resize(pendingBoardsByEvaluator[evaluatorIndex].size());
            m_evaluators[evaluatorIndex]->evaluate(pendingBoardsByEvaluator[evaluatorIndex],
                                                   evaluationsByEvaluator[evaluatorIndex]);
        }

        for (ActiveGame& game : activeGames)
        {
            if (!game.isActive || game.pendingLeaf == nullptr)
                continue;

            game.search->absorb(
                evaluationsByEvaluator[static_cast<size_t>(game.evaluatorIndex)][game.evaluationBatchOffset]);
            game.pendingLeaf = nullptr;
            ++game.evaluatedPositionCount;
            advanceUntilEvaluation(game);

            if (game.pendingLeaf == nullptr)
            {
                handOver(game);
                if (!shouldContinue)
                    return;
            }
        }
    }
}

std::vector<TrainingSample> generateSelfPlaySamples(
    const AmoebaNetwork& champion, const TrainingSettings& settings, uint64_t seed)
{
    MCTSConfig searchConfig = settings.searchConfig;
    searchConfig.rootNoise = settings.rootNoise;

    NetworkEvaluator evaluator{champion};
    NetworkEvaluator* const evaluators[]{&evaluator};

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
void trainCandidate(AmoebaNetwork& candidate, const std::vector<TrainingSample>& samples,
                    const TrainingSettings& settings, uint64_t seed)
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

        const auto lossFunction = [&](const std::vector<mlx::core::array>& currentParameters) {
            return computeLoss(candidate, currentParameters, batch, settings.weightDecay);
        };
        auto [values, gradients] = mlx::core::value_and_grad(lossFunction, argnums)(parameters);

        parameters = adam.updateParameters(parameters, gradients, settings.learningRate);
        mlx::core::eval(parameters);

        if (step % validationCheckInterval != 0 && step != 1)
            continue;

        const std::vector<mlx::core::array> held = computeLoss(candidate, parameters, validation, 0.0f);
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
double evaluateCandidate(const AmoebaNetwork& candidate, const AmoebaNetwork& champion,
                         const TrainingSettings& settings, uint64_t seed)
{
    MCTSConfig searchConfig = settings.searchConfig;
    searchConfig.simulations = settings.gateSimulationCount;
    searchConfig.rootNoise = 0.0f; // competition keeps the network's own opinion

    NetworkEvaluator candidateEval{candidate};
    NetworkEvaluator championEval{champion};
    NetworkEvaluator* const evaluators[]{&candidateEval, &championEval};

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

} // namespace amoeba

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::println(stderr, "usage: amoeba_train <weights.safetensors>");
        std::println(stderr, "  created from random weights if it does not exist, and written back to");
        std::println(stderr, "  whenever a generation beats the one before it");
        return EXIT_FAILURE;
    }

    const amoeba::TrainingSettings settings;
    const std::filesystem::path weights{argv[1]};

    // Generation 0 is random weights, written straight away so that amoeba_bot has
    // something to load while the first generation is still being played.
    std::unique_ptr<amoeba::AmoebaNetwork> champion;
    if (std::filesystem::exists(weights))
    {
        champion = std::make_unique<amoeba::AmoebaNetwork>(weights);
        amoeba::report("[train] resuming from {}: {}, {} parameters", weights.string(),
                       amoeba::AmoebaNetwork::name, champion->parameterCount());
    }
    else
    {
        champion = std::make_unique<amoeba::AmoebaNetwork>(settings.seed);
        champion->save(weights);
        amoeba::report("[train] {} did not exist: started from random weights, {} parameters, saved",
                       weights.string(), champion->parameterCount());
    }

    std::vector<amoeba::TrainingSample> replay;
    int gameIdBase = 0;

    for (int generation = 1;; ++generation)
    {
        amoeba::report("");
        amoeba::report("======== generation {} ========", generation);

        std::vector<amoeba::TrainingSample> fresh = amoeba::generateSelfPlaySamples(
            *champion, settings, settings.seed + static_cast<uint64_t>(generation) * 1000);

        // Game ids have to stay unique across generations, or splitDatasetByGame will hold
        // out a game from this generation and train on a different one with the same
        // id from the last.
        for (amoeba::TrainingSample& sample : fresh)
            sample.gameId += gameIdBase;
        gameIdBase += settings.selfPlayGameCount;

        amoeba::reportMemory("self-play");

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()), std::make_move_iterator(fresh.end()));
        if (replay.size() > settings.replayBufferCapacity)
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(replay.size() - settings.replayBufferCapacity));
        amoeba::report("[train] replay buffer holds {} positions", replay.size());

        // The candidate starts from the best weights rather than from scratch: each
        // generation is meant to refine, not to relearn the game.
        amoeba::AmoebaNetwork candidate = *champion;
        amoeba::trainCandidate(candidate, replay, settings, settings.seed + static_cast<uint64_t>(generation));
        amoeba::reportMemory("training");

        const double score = amoeba::evaluateCandidate(
            candidate, *champion, settings, settings.seed + 7777 + static_cast<uint64_t>(generation));
        amoeba::reportMemory("the gate");

        if (score < settings.promotionThreshold)
        {
            amoeba::report("[train] generation {} rejected at {:.1f}%, keeping the previous weights",
                           generation, 100.0 * score);
        }
        else
        {
            *champion = candidate;
            champion->save(weights);
            amoeba::report("[train] generation {} PROMOTED at {:.1f}%, wrote {}",
                           generation, 100.0 * score, weights.string());
        }
    }
}
