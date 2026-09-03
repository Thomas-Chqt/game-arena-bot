#include "training.hpp"

#include "mcts.hpp"
#include "network.hpp"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <format>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace amoeba_bot
{

namespace
{

constexpr uint64_t healthReferenceSeed = 20260819;
constexpr uint64_t evaluationOpeningSeed = healthReferenceSeed + 7777;
// Self-play and the gate use the same search budget so the comparison does not
// reward a candidate under different search conditions.
constexpr int selfPlaySimulationCount = 512;
constexpr int evaluationSimulationCount = 512;
// CPU lanes are sized at runtime from the available hardware threads. Their
// combined pending leaves stay capped here, so a machine with more CPU cores
// creates more, smaller lanes without silently increasing GPU memory use.
constexpr size_t selfPlayEvaluationBatchSize = 4096;
constexpr int selfPlayGameCount = 8192;
// Two hundred fifty-six paired openings give 512 games. Promotion uses the
// lower bound across the 256 pair scores, rather than a raw win-rate threshold.
constexpr int championGameCount = 512;
constexpr int evaluationOpeningPlyCount = 4;
constexpr int randomGameCount = 256;
constexpr int rolloutGameCount = 256;
constexpr int concurrentEvaluationGames = 512;
constexpr int samplingPlyCount = 20;
constexpr float rootNoise = 0.25f;
constexpr float noiseAlpha = 0.35f;
constexpr size_t trainingBatchSize = 1024;
// The split is by game, never by position. One eighth of the games generated
// by a champion are validation-only, including games retained after a rejected
// candidate.
constexpr uint64_t heldOutGameFraction = 8;
constexpr int maximumTrainingEpochs = 8;
constexpr int validationPointsPerEpoch = 8;
constexpr int earlyStoppingPatience = 2;
constexpr float validationImprovement = 1e-4f;
// Each candidate fine-tunes an already trained champion. Use deliberately
// small AdamW updates so held-out validation can select the point before the
// candidate begins fitting this generation's self-play too closely.
constexpr float learningRate = 5e-5f;
constexpr float weightDecay = 1e-2f;
constexpr double gateConfidenceZ = 1.645; // one-sided 95% lower bound
// The 512-visit gate supersedes the v1 256-visit qualification. Bump this
// whenever a fixed baseline, its game count, or its evaluation method changes.
constexpr std::string_view baselineGateVersion = "2";
constexpr std::string_view baselineMetadataPrefix = "training.baseline_gate.";
constexpr std::string_view retainedStepMetadataName = "training.retained_step";

volatile sig_atomic_t stopRequested = 0;

size_t mctsWorkerCount()
{
    return std::max<size_t>(1, std::thread::hardware_concurrency());
}

class ThreadPool
{
public:
    explicit ThreadPool(size_t workerCount)
    {
        assert(workerCount > 0);
        m_workers.reserve(workerCount);
        for (size_t worker = 0; worker < workerCount; ++worker)
            m_workers.emplace_back([this] { run(); });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard lock{m_mutex};
            m_stopping = true;
        }
        m_ready.notify_all();
        for (std::thread& worker : m_workers)
            worker.join();
    }

    template<typename Function>
    auto submit(Function&& function)
        -> std::future<std::invoke_result_t<std::decay_t<Function>>>
    {
        using Result = std::invoke_result_t<std::decay_t<Function>>;
        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::forward<Function>(function));
        std::future<Result> result = task->get_future();
        {
            std::lock_guard lock{m_mutex};
            assert(!m_stopping);
            m_tasks.emplace_back([task] { (*task)(); });
        }
        m_ready.notify_one();
        return result;
    }

private:
    void run()
    {
        for (;;)
        {
            std::function<void()> task;
            {
                std::unique_lock lock{m_mutex};
                m_ready.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
                if (m_stopping && m_tasks.empty())
                    return;
                task = std::move(m_tasks.front());
                m_tasks.pop_front();
            }
            task();
        }
    }

    std::mutex m_mutex;
    std::condition_variable m_ready;
    std::deque<std::function<void()>> m_tasks;
    std::vector<std::thread> m_workers;
    bool m_stopping = false;
};

struct MatchScore
{
    int wins = 0;
    int draws = 0;
    int losses = 0;
    uint64_t totalPlies = 0;
    std::vector<double> observations;

    void add(Outcome outcome, bool candidateWhite, uint64_t plyCount)
    {
        if (outcome == Outcome::draw)
        {
            ++draws;
            observations.push_back(0.5);
        }
        else if ((outcome == Outcome::whiteWins) == candidateWhite)
        {
            ++wins;
            observations.push_back(1.0);
        }
        else
        {
            ++losses;
            observations.push_back(0.0);
        }
        totalPlies += plyCount;
    }
    int games() const { return wins + draws + losses; }
    double score() const
    {
        assert(games() > 0);
        return (wins + 0.5 * draws) / games();
    }
    double averagePlies() const
    {
        assert(games() > 0);
        return static_cast<double>(totalPlies) / games();
    }
    double lowerConfidenceBound() const;
};

struct SelfPlayStatus
{
    uint64_t positions = 0;
    uint64_t totalPositions = 0;
    uint64_t games = 0;
    uint64_t totalPlies = 0;
    double positionsPerSecond = 0.0;
    std::chrono::steady_clock::time_point startedAt =
        std::chrono::steady_clock::now();
    bool visible = false;
};

SelfPlayStatus selfPlayStatus;

struct GateStatus
{
    uint64_t generation = 0;
    std::string_view opponent;
    int completedGames = 0;
    int totalGames = 0;
    double score = 0.0;
    int wins = 0;
    int draws = 0;
    int losses = 0;
    uint64_t totalPlies = 0;
    bool visible = false;
};

GateStatus gateStatus;

uint64_t completedGameRounds(const Board& board)
{
    // applyMove returns an Outcome instead of the terminal Board, so the board
    // retained by a finished runner is still one move behind the game result.
    return static_cast<uint64_t>(board.plyCount) + 1;
}

uint64_t makeRunSeed()
{
    std::random_device randomDevice;
    const uint64_t entropy =
        (static_cast<uint64_t>(randomDevice()) << 32) ^ randomDevice();
    const uint64_t time = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t process = static_cast<uint64_t>(getpid());
    return entropy ^ time ^ (0x9e3779b97f4a7c15ULL * process);
}

void clearTerminalLine()
{
    if (isatty(STDOUT_FILENO))
    {
        std::print("\r\x1b[2K");
        std::fflush(stdout);
    }
}

void drawGateStatus()
{
    if (!gateStatus.visible || !isatty(STDOUT_FILENO))
        return;
    if (gateStatus.completedGames == 0)
        std::print("\r\x1b[2K[gate] generation {} vs {}: 0/{} games",
                   gateStatus.generation, gateStatus.opponent,
                   gateStatus.totalGames);
    else
        std::print("\r\x1b[2K[gate] generation {} vs {}: {}/{} games "
                   "({:.1f}% win rate, {}W/{}D/{}L, {:.1f} rounds/game)",
                   gateStatus.generation, gateStatus.opponent,
                   gateStatus.completedGames, gateStatus.totalGames,
                   100.0 * gateStatus.score, gateStatus.wins,
                   gateStatus.draws, gateStatus.losses,
                   static_cast<double>(gateStatus.totalPlies)
                       / gateStatus.completedGames);
    std::fflush(stdout);
}

void startGateStatus(uint64_t generation, std::string_view opponent,
                     int totalGames)
{
    assert(totalGames > 0);
    assert(!gateStatus.visible);
    gateStatus = GateStatus{
        .generation = generation,
        .opponent = opponent,
        .completedGames = 0,
        .totalGames = totalGames,
        .score = 0.0,
        .wins = 0,
        .draws = 0,
        .losses = 0,
        .totalPlies = 0,
        .visible = true,
    };
    drawGateStatus();
}

void updateGateStatus(const MatchScore& score)
{
    assert(gateStatus.visible);
    assert(score.games() > gateStatus.completedGames);
    assert(score.games() <= gateStatus.totalGames);
    gateStatus.completedGames = score.games();
    gateStatus.score = score.score();
    gateStatus.wins = score.wins;
    gateStatus.draws = score.draws;
    gateStatus.losses = score.losses;
    gateStatus.totalPlies = score.totalPlies;
    drawGateStatus();
}

void finishGateStatus()
{
    if (!gateStatus.visible)
        return;
    clearTerminalLine();
    gateStatus.visible = false;
}

void drawSelfPlayStatus()
{
    if (!selfPlayStatus.visible || !isatty(STDOUT_FILENO))
        return;
    std::print("\r\x1b[2K[selfplay] generated {} positions from {} games, "
               "{:.1f} rounds/game, {:.1f} positions/s average",
               selfPlayStatus.totalPositions, selfPlayStatus.games,
               static_cast<double>(selfPlayStatus.totalPlies) / selfPlayStatus.games,
               selfPlayStatus.positionsPerSecond);
    std::fflush(stdout);
}

template<typename... Args>
void report(std::format_string<Args...> format, Args&&... args)
{
    if (selfPlayStatus.visible && isatty(STDOUT_FILENO))
        clearTerminalLine();
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
    if (selfPlayStatus.visible)
        selfPlayStatus.positions = 0;
}

void reportSelfPlayStatus(uint64_t generatedPositions, uint64_t completedGames,
                          uint64_t completedPlies)
{
    assert(completedGames > 0);
    selfPlayStatus.positions += generatedPositions;
    selfPlayStatus.totalPositions += generatedPositions;
    selfPlayStatus.games += completedGames;
    selfPlayStatus.totalPlies += completedPlies;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selfPlayStatus.startedAt).count();
    selfPlayStatus.positionsPerSecond = selfPlayStatus.totalPositions / elapsed;
    selfPlayStatus.visible = true;
    if (isatty(STDOUT_FILENO))
        drawSelfPlayStatus();
    else
    {
        std::println("[selfplay] generated {} positions from {} games, "
                     "{:.1f} rounds/game, {:.1f} positions/s average",
                     selfPlayStatus.totalPositions, selfPlayStatus.games,
                     static_cast<double>(selfPlayStatus.totalPlies)
                         / selfPlayStatus.games,
                     selfPlayStatus.positionsPerSecond);
        std::fflush(stdout);
    }
}

void finishSelfPlayStatus()
{
    if (!selfPlayStatus.visible)
        return;
    clearTerminalLine();
    selfPlayStatus.visible = false;
}

void requestStop(int)
{
    stopRequested = 1;
}

void installSignalHandlers()
{
    struct sigaction action{};
    action.sa_handler = requestStop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) != 0
        || sigaction(SIGTERM, &action, nullptr) != 0)
        throw std::runtime_error(std::format("sigaction: {}", std::strerror(errno)));
}

struct TrainingSample
{
    Board board;
    VisitCounts visits;
    float outcome;
    uint64_t gameId;
};

struct CompletedSelfPlay
{
    std::vector<TrainingSample> samples;
    uint64_t games = 0;
    uint64_t totalPlies = 0;
    double positionsPerSecond = 0.0;

    double averagePlies() const
    {
        assert(games > 0);
        return static_cast<double>(totalPlies) / games;
    }
};

// splitmix64's finalizer, so the split depends on the game id alone and is
// identical across restarts. Game ids are unique across generations, so a game
// held out once is held out forever.
bool isHeldOutGame(uint64_t gameId)
{
    uint64_t mixed = gameId + 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    return (mixed ^ (mixed >> 31)) % heldOutGameFraction == 0;
}

uint16_t randomLegalMove(const Board& board, std::mt19937_64& randomEngine)
{
    assert(board.legalMoveCount > 0);
    uint16_t remaining = std::uniform_int_distribution<uint16_t>{
        0, static_cast<uint16_t>(board.legalMoveCount - 1)}(randomEngine);
    uint16_t selected = 0;
    bool found = false;
    board.forEachLegal([&](uint16_t moveId) {
        if (!found && remaining == 0)
        {
            selected = moveId;
            found = true;
        }
        else if (!found)
            --remaining;
    });
    assert(found);
    return selected;
}

uint16_t selectMoveFromVisits(const VisitCounts& visitCounts, int plyCount,
                              std::mt19937_64& randomEngine)
{
    if (plyCount >= samplingPlyCount)
        return bestMove(visitCounts);
    const uint64_t totalVisits =
        std::accumulate(visitCounts.begin(), visitCounts.end(), uint64_t{0});
    assert(totalVisits > 0);
    uint64_t remaining =
        std::uniform_int_distribution<uint64_t>{0, totalVisits - 1}(randomEngine);
    for (uint16_t moveId = 0; moveId < moveIdCount; ++moveId)
    {
        if (visitCounts[moveId] > remaining)
            return moveId;
        remaining -= visitCounts[moveId];
    }
    return bestMove(visitCounts);
}

struct NetworkPairing
{
    int white;
    int black;
};

template<int SimulationCount>
struct ActiveGame
{
    Board board;
    std::optional<Outcome> outcome;
    std::vector<uint64_t> positionHistory;
    std::vector<TrainingSample> samples;
    std::mt19937_64 moveRandom;
    std::mt19937_64 noiseRandom;
    NetworkPairing pairing{0, 0};
    uint64_t gameId = 0;
    bool active = false;
    std::optional<MCTS<SimulationCount>> search;
    const Board* pendingLeaf = nullptr;
    int networkIndex = 0;
    size_t evaluationOffset = 0;
    bool firstLeafOfSearch = true;
};

template<int SimulationCount>
void startSearch(ActiveGame<SimulationCount>& game)
{
    game.search.emplace(game.board, game.positionHistory);
    game.pendingLeaf = nullptr;
    game.firstLeafOfSearch = true;
}

template<int SimulationCount>
void initialize(ActiveGame<SimulationCount>& game, uint64_t gameId,
                NetworkPairing pairing, uint64_t randomSeed)
{
    initialize(game, gameId, pairing, randomSeed, Board::startingBoard(), {});
}

template<int SimulationCount>
void initialize(ActiveGame<SimulationCount>& game, uint64_t gameId,
                NetworkPairing pairing, uint64_t randomSeed,
                const Board& initialBoard, std::span<const uint64_t> initialHistory)
{
    game.board = initialBoard;
    game.outcome.reset();
    if (initialHistory.empty())
        game.positionHistory.assign(1, game.board.positionHash);
    else
    {
        assert(initialHistory.back() == game.board.positionHash);
        game.positionHistory.assign(initialHistory.begin(), initialHistory.end());
    }
    game.samples.clear();
    game.moveRandom.seed(randomSeed ^ (0x9e3779b97f4a7c15ULL * (gameId + 1)));
    game.noiseRandom.seed(randomSeed + gameId);
    game.pairing = pairing;
    game.gameId = gameId;
    game.active = true;
    startSearch(game);
}

struct EvaluationOpening
{
    Board board;
    std::vector<uint64_t> positionHistory;
};

std::optional<EvaluationOpening> makeEvaluationOpening(std::mt19937_64& randomEngine)
{
    EvaluationOpening opening{.board = Board::startingBoard()};
    opening.positionHistory.push_back(opening.board.positionHash);
    for (int ply = 0; ply < evaluationOpeningPlyCount; ++ply)
    {
        const MoveResult result = applyMove(
            opening.board, Move::fromId(randomLegalMove(opening.board, randomEngine)),
            opening.positionHistory);
        if (std::holds_alternative<Outcome>(result))
            return std::nullopt;
        opening.board = std::get<Board>(result);
        opening.positionHistory.push_back(opening.board.positionHash);
    }
    return opening;
}

std::vector<EvaluationOpening> makeEvaluationOpenings(size_t count)
{
    // Every candidate sees this same suite. Pairing both colours from each
    // position adds position diversity without injecting noise into search.
    std::mt19937_64 randomEngine{evaluationOpeningSeed};
    std::unordered_set<uint64_t> positionHashes;
    std::vector<EvaluationOpening> openings;
    openings.reserve(count);
    while (openings.size() < count)
    {
        std::optional<EvaluationOpening> opening = makeEvaluationOpening(randomEngine);
        if (opening.has_value() && positionHashes.insert(opening->board.positionHash).second)
            openings.push_back(std::move(*opening));
    }
    return openings;
}

template<int SimulationCount, bool SampleOpeningMoves>
void advance(ActiveGame<SimulationCount>& game)
{
    while (!game.outcome.has_value())
    {
        if (const Board* leaf = game.search->pendingLeaf())
        {
            game.networkIndex = game.board.whiteToMove
                ? game.pairing.white : game.pairing.black;
            game.pendingLeaf = leaf;
            return;
        }
        const VisitCounts visits = game.search->visits();
        if constexpr (SampleOpeningMoves)
            game.samples.push_back({game.board, visits, 0.0f, game.gameId});
        const uint16_t moveId = SampleOpeningMoves
            ? selectMoveFromVisits(visits, game.board.plyCount, game.moveRandom)
            : bestMove(visits);
        MoveResult result = applyMove(
            game.board, Move::fromId(moveId), game.positionHistory);
        if (const auto* outcome = std::get_if<Outcome>(&result))
        {
            game.outcome = *outcome;
            break;
        }
        game.board = std::get<Board>(std::move(result));
        game.positionHistory.push_back(game.board.positionHash);
        startSearch(game);
    }
    game.pendingLeaf = nullptr;
}

template<int SimulationCount>
void finish(ActiveGame<SimulationCount>& game)
{
    assert(game.outcome.has_value());
    const uint64_t completedRounds = completedGameRounds(game.board);
    for (TrainingSample& sample : game.samples)
    {
        assert(completedRounds > sample.board.plyCount);
        const uint64_t remainingMoves = completedRounds - sample.board.plyCount;
        const float timeDiscount = std::pow(
            terminalValueDiscountPerMove, static_cast<float>(remainingMoves));
        sample.outcome = timeDiscount
            * outcomeFor(*game.outcome, sample.board.whiteToMove);
    }
}

void addExplorationNoise(Evaluation& evaluation, const Board& board,
                         std::mt19937_64& randomEngine)
{
    std::gamma_distribution<float> gamma{noiseAlpha, 1.0f};
    std::array<float, moveIdCount> noise{};
    float total = 0.0f;
    board.forEachLegal([&](uint16_t moveId) {
        noise[moveId] = gamma(randomEngine);
        total += noise[moveId];
    });
    if (total <= 0.0f)
        return;
    board.forEachLegal([&](uint16_t moveId) {
        evaluation.policy[moveId] =
            (1.0f - rootNoise) * evaluation.policy[moveId]
            + rootNoise * noise[moveId] / total;
    });
}

template<int SimulationCount>
class NetworkMatchRunner
{
public:
    explicit NetworkMatchRunner(std::span<const Network* const> networks)
        : m_networks(networks)
    {
        assert(networks.size() == 2);
    }

    template<int GameCount, typename GameFinished>
    bool play(uint64_t firstGameId, uint64_t randomSeed,
              GameFinished&& gameFinished)
    {
        static_assert(GameCount > 0);
        static_assert(GameCount % 2 == 0);
        const std::vector<EvaluationOpening> openings =
            makeEvaluationOpenings(static_cast<size_t>(GameCount / 2));
        const int slotCount = std::min(GameCount, concurrentEvaluationGames);
        std::vector<ActiveGame<SimulationCount>> games(static_cast<size_t>(slotCount));
        int started = 0;
        int completed = 0;
        const auto initializeGame = [&](ActiveGame<SimulationCount>& game) {
            const size_t openingIndex = static_cast<size_t>(started / 2);
            initialize(game, firstGameId + started,
                       started % 2 == 0 ? NetworkPairing{0, 1}
                                        : NetworkPairing{1, 0},
                       randomSeed, openings[openingIndex].board,
                       openings[openingIndex].positionHistory);
            ++started;
        };
        for (ActiveGame<SimulationCount>& game : games)
            initializeGame(game);
        ThreadPool workers{mctsWorkerCount()};
        std::array<std::vector<const Board*>, 2> pendingBoards;
        std::array<std::vector<Evaluation>, 2> evaluations;
        while (completed < GameCount)
        {
            std::vector<std::future<void>> advanced;
            advanced.reserve(games.size());
            for (ActiveGame<SimulationCount>& game : games)
            {
                if (game.active)
                    advanced.push_back(workers.submit([&game] {
                        advance<SimulationCount, false>(game);
                    }));
            }
            for (std::future<void>& future : advanced)
                future.get();

            for (std::vector<const Board*>& boards : pendingBoards)
                boards.clear();
            for (ActiveGame<SimulationCount>& game : games)
            {
                if (!game.active)
                    continue;
                if (game.pendingLeaf == nullptr)
                {
                    finish(game);
                    const MatchScore& score = gameFinished(
                        game.gameId, *game.outcome,
                        completedGameRounds(game.board));
                    ++completed;
                    assert(score.games() == completed);
                    updateGateStatus(score);
                    if (started < GameCount)
                        initializeGame(game);
                    else
                    {
                        game.active = false;
                        continue;
                    }
                }
                if (!game.active || game.pendingLeaf == nullptr)
                    continue;
                std::vector<const Board*>& boards =
                    pendingBoards[static_cast<size_t>(game.networkIndex)];
                game.evaluationOffset = boards.size();
                boards.push_back(game.pendingLeaf);
            }
            if (completed == GameCount)
                break;
            for (size_t networkIndex = 0; networkIndex < m_networks.size(); ++networkIndex)
            {
                if (pendingBoards[networkIndex].empty())
                    continue;
                evaluations[networkIndex].resize(pendingBoards[networkIndex].size());
                (*m_networks[networkIndex])(
                    pendingBoards[networkIndex], evaluations[networkIndex]);
            }
            for (ActiveGame<SimulationCount>& game : games)
            {
                if (!game.active || game.pendingLeaf == nullptr)
                    continue;
                const Evaluation& evaluation =
                    evaluations[static_cast<size_t>(game.networkIndex)][game.evaluationOffset];
                game.search->absorb(evaluation.policy, evaluation.value);
                game.firstLeafOfSearch = false;
                game.pendingLeaf = nullptr;
            }
            if (stopRequested)
                return false;
        }
        return true;
    }

private:
    std::span<const Network* const> m_networks;
};

class SelfPlayLane
{
public:
    SelfPlayLane(uint64_t firstGameId, size_t gameCount, size_t activeGameCount,
                 uint64_t randomSeed)
        : m_games(std::min(activeGameCount, gameCount)),
          m_firstGameId(firstGameId), m_randomSeed(randomSeed),
          m_gameCount(gameCount), m_activeGames(m_games.size())
    {
        assert(gameCount > 0);
        assert(activeGameCount > 0);
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
            initializeNext(game);
    }

    CompletedSelfPlay prepare()
    {
        assert(!m_prepared);
        m_pendingBoards.clear();
        CompletedSelfPlay completed;
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            if (!game.active)
                continue;
            advance<selfPlaySimulationCount, true>(game);
            if (game.pendingLeaf == nullptr)
            {
                finish(game);
                completed.samples.insert(
                    completed.samples.end(),
                    std::make_move_iterator(game.samples.begin()),
                    std::make_move_iterator(game.samples.end()));
                ++completed.games;
                completed.totalPlies += completedGameRounds(game.board);
                ++m_completedGames;
                if (m_startedGames < m_gameCount)
                {
                    initializeNext(game);
                    advance<selfPlaySimulationCount, true>(game);
                }
                else
                {
                    game.active = false;
                    --m_activeGames;
                    continue;
                }
            }
            game.evaluationOffset = m_pendingBoards.size();
            m_pendingBoards.push_back(game.pendingLeaf);
        }
        m_prepared = !m_pendingBoards.empty();
        return completed;
    }

    std::span<const Board* const> preparedBoards() const
    {
        assert(m_prepared);
        return m_pendingBoards;
    }

    void absorb(std::span<const Evaluation> evaluations)
    {
        assert(m_prepared);
        assert(evaluations.size() == m_pendingBoards.size());
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            if (!game.active)
                continue;
            assert(game.pendingLeaf != nullptr);
            Evaluation evaluation = evaluations[game.evaluationOffset];
            if (game.firstLeafOfSearch)
                addExplorationNoise(evaluation, *game.pendingLeaf, game.noiseRandom);
            game.search->absorb(evaluation.policy, evaluation.value);
            game.firstLeafOfSearch = false;
            game.pendingLeaf = nullptr;
        }
        m_prepared = false;
    }

    bool hasPreparedEvaluations() const { return m_prepared; }
    bool finished() const
    {
        return m_activeGames == 0 && !m_prepared;
    }
    size_t completedGames() const { return m_completedGames; }

private:
    void initializeNext(ActiveGame<selfPlaySimulationCount>& game)
    {
        assert(m_startedGames < m_gameCount);
        initialize(game, m_firstGameId + m_startedGames,
                   NetworkPairing{0, 0}, m_randomSeed);
        ++m_startedGames;
    }

    std::vector<ActiveGame<selfPlaySimulationCount>> m_games;
    std::vector<const Board*> m_pendingBoards;
    uint64_t m_firstGameId;
    uint64_t m_randomSeed;
    size_t m_gameCount;
    size_t m_startedGames = 0;
    size_t m_completedGames = 0;
    size_t m_activeGames;
    bool m_prepared = false;
};

CompletedSelfPlay generateSelfPlay(const Network& network,
                                   uint64_t& nextGameId,
                                   uint64_t runSeed)
{
    selfPlayStatus = SelfPlayStatus{};
    const size_t laneCount = std::min(
        {mctsWorkerCount(), selfPlayEvaluationBatchSize,
         static_cast<size_t>(selfPlayGameCount)});
    const size_t gamesPerLane = selfPlayGameCount / laneCount;
    const size_t extraGames = selfPlayGameCount % laneCount;
    const size_t leavesPerLane = selfPlayEvaluationBatchSize / laneCount;
    const size_t extraLeaves = selfPlayEvaluationBatchSize % laneCount;
    std::vector<SelfPlayLane> lanes;
    lanes.reserve(laneCount);
    uint64_t firstGameId = nextGameId;
    for (size_t laneIndex = 0; laneIndex < laneCount; ++laneIndex)
    {
        const size_t laneGames = gamesPerLane + (laneIndex < extraGames ? 1 : 0);
        const size_t activeGames = leavesPerLane + (laneIndex < extraLeaves ? 1 : 0);
        lanes.emplace_back(firstGameId, laneGames, activeGames, runSeed ^ firstGameId);
        firstGameId += laneGames;
    }
    nextGameId += selfPlayGameCount;

    ThreadPool workers{laneCount};
    CompletedSelfPlay result;
    const auto collect = [&](CompletedSelfPlay completed) {
        if (!completed.samples.empty())
        {
            reportSelfPlayStatus(
                completed.samples.size(), completed.games, completed.totalPlies);
            result.games += completed.games;
            result.totalPlies += completed.totalPlies;
            result.samples.insert(
                result.samples.end(),
                std::make_move_iterator(completed.samples.begin()),
                std::make_move_iterator(completed.samples.end()));
        }
    };

    for (;;)
    {
        std::vector<std::future<CompletedSelfPlay>> prepared;
        prepared.reserve(lanes.size());
        for (SelfPlayLane& lane : lanes)
        {
            if (!lane.finished() && !lane.hasPreparedEvaluations())
                prepared.push_back(workers.submit([&lane] { return lane.prepare(); }));
        }
        for (std::future<CompletedSelfPlay>& future : prepared)
            collect(future.get());
        if (stopRequested)
            break;

        std::vector<const Board*> boards;
        std::vector<size_t> offsets;
        offsets.reserve(lanes.size() + 1);
        for (const SelfPlayLane& lane : lanes)
        {
            offsets.push_back(boards.size());
            if (!lane.hasPreparedEvaluations())
                continue;
            const std::span<const Board* const> pending = lane.preparedBoards();
            boards.insert(boards.end(), pending.begin(), pending.end());
        }
        offsets.push_back(boards.size());

        if (boards.empty())
            break;

        // MLX remains confined to this coordinator thread. The worker pool owns
        // the irregular Board/MCTS work; this one call gives CUDA a large dense
        // batch without requiring MLX arrays to cross worker-thread boundaries.
        std::vector<Evaluation> evaluations(boards.size());
        network(boards, evaluations);
        for (size_t laneIndex = 0; laneIndex < lanes.size(); ++laneIndex)
        {
            if (!lanes[laneIndex].hasPreparedEvaluations())
                continue;
            const size_t offset = offsets[laneIndex];
            const size_t count = offsets[laneIndex + 1] - offset;
            lanes[laneIndex].absorb(
                std::span<const Evaluation>{evaluations.data() + offset, count});
        }
    }

    finishSelfPlayStatus();
    result.positionsPerSecond = selfPlayStatus.positionsPerSecond;
    if (!stopRequested)
    {
        const size_t completedGames = std::accumulate(
            lanes.begin(), lanes.end(), size_t{0},
            [](size_t total, const SelfPlayLane& lane) {
                return total + lane.completedGames();
            });
        assert(completedGames == selfPlayGameCount);
        assert(result.games == selfPlayGameCount);
    }
    return result;
}

constexpr Coordinate transformCoordinate(Coordinate coordinate, uint8_t symmetry)
{
    assert(symmetry < 12);
    if (symmetry >= 6)
        std::swap(coordinate.q, coordinate.r);
    for (uint8_t rotation = 0; rotation < symmetry % 6; ++rotation)
        coordinate = Coordinate{
            static_cast<int8_t>(-coordinate.r),
            static_cast<int8_t>(coordinate.q + coordinate.r)};
    return coordinate;
}

struct SymmetryTables
{
    std::array<std::array<uint8_t, hexCount>, 12> hexes{};
    std::array<std::array<uint8_t, directionCount>, 12> directions{};
};

constexpr SymmetryTables symmetryTables = [] {
    SymmetryTables tables;
    for (uint8_t symmetry = 0; symmetry < 12; ++symmetry)
    {
        for (uint8_t hex = 0; hex < hexCount; ++hex)
            tables.hexes[symmetry][hex] =
                hexIndex(transformCoordinate(hexCoord(hex), symmetry)).value();
        for (uint8_t direction = 0; direction < directionCount; ++direction)
        {
            const Coordinate transformed = transformCoordinate(
                Coordinate{directions[direction].q, directions[direction].r}, symmetry);
            tables.directions[symmetry][direction] = directionIdx(
                Direction{transformed.q, transformed.r});
        }
    }
    return tables;
}();

static_assert([] {
    for (uint8_t symmetry = 0; symmetry < 12; ++symmetry)
    {
        std::array<bool, hexCount> seenHexes{};
        std::array<bool, directionCount> seenDirections{};
        for (uint8_t hex : symmetryTables.hexes[symmetry])
            seenHexes[hex] = true;
        for (uint8_t direction : symmetryTables.directions[symmetry])
            seenDirections[direction] = true;
        if (!std::ranges::all_of(seenHexes, std::identity{})
            || !std::ranges::all_of(seenDirections, std::identity{}))
            return false;
    }
    return true;
}());

constexpr uint16_t transformedMoveId(uint16_t moveId, uint8_t symmetry)
{
    Move move = Move::fromId(moveId);
    move.sourceCoord = symmetryTables.hexes[symmetry][move.sourceCoord];
    move.direction = symmetryTables.directions[symmetry][move.direction];
    return move.id();
}

static_assert([] {
    for (uint8_t symmetry = 0; symmetry < 12; ++symmetry)
    {
        std::array<bool, moveIdCount> seen{};
        for (uint16_t moveId = 0; moveId < moveIdCount; ++moveId)
            seen[transformedMoveId(moveId, symmetry)] = true;
        if (!std::ranges::all_of(seen, std::identity{}))
            return false;
    }
    return true;
}());

TrainingSample transformSample(const TrainingSample& sample, uint8_t symmetry)
{
    TrainingSample transformed = sample;
    for (uint8_t hex = 0; hex < hexCount; ++hex)
        transformed.board.hexes[symmetryTables.hexes[symmetry][hex]] = sample.board.hexes[hex];
    transformed.board.whiteKernelIndex =
        symmetryTables.hexes[symmetry][sample.board.whiteKernelIndex];
    transformed.board.blackKernelIndex =
        symmetryTables.hexes[symmetry][sample.board.blackKernelIndex];
    transformed.board.positionHash = 0;
    transformed.board.clearLegal();
    transformed.visits.fill(0);
    sample.board.forEachLegal([&](uint16_t moveId) {
        const uint16_t transformedMove = transformedMoveId(moveId, symmetry);
        transformed.board.setLegal(transformedMove);
        ++transformed.board.legalMoveCount;
    });
    for (uint16_t moveId = 0; moveId < moveIdCount; ++moveId)
        transformed.visits[transformedMoveId(moveId, symmetry)] = sample.visits[moveId];
    assert(transformed.board.legalMoveCount == sample.board.legalMoveCount);
    assert(std::accumulate(transformed.visits.begin(), transformed.visits.end(), uint64_t{0})
           == std::accumulate(sample.visits.begin(), sample.visits.end(), uint64_t{0}));
    return transformed;
}

TrainingBatch createBatch(const std::vector<TrainingSample>& replay,
                          std::span<const size_t> picks,
                          std::mt19937_64& randomEngine,
                          bool augment)
{
    const auto policyIndexToMoveId = [](uint16_t policyIndex, bool whiteToMove) {
        Move move = Move::fromId(policyIndex);
        if (!whiteToMove)
        {
            move.sourceCoord = rotatedHex(move.sourceCoord);
            move.direction = oppositeDirection(move.direction);
        }
        return move.id();
    };
    assert(!picks.empty());
    assert(picks.size() <= trainingBatchSize);
    const size_t batchSize = picks.size();
    std::vector<mlx::core::array> inputs;
    inputs.reserve(batchSize);
    std::vector<float> legal(batchSize * moveIdCount);
    std::vector<float> policy(batchSize * moveIdCount);
    std::vector<float> outcomes(batchSize);
    for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex)
    {
        // Training uses the six rotations requested by the owner. Validation
        // stays at identity so its epoch-to-epoch comparison has no augmentation
        // noise in it.
        const uint8_t symmetry = augment
            ? std::uniform_int_distribution<uint8_t>{0, 5}(randomEngine)
            : 0;
        const TrainingSample sample = transformSample(replay[picks[batchIndex]], symmetry);
        inputs.push_back(sample.board.tensorEncoding());
        const uint64_t totalVisits =
            std::accumulate(sample.visits.begin(), sample.visits.end(), uint64_t{0});
        assert(totalVisits > 0);
        const size_t offset = batchIndex * moveIdCount;
        for (uint16_t policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
        {
            const uint16_t moveId = policyIndexToMoveId(
                policyIndex, sample.board.whiteToMove);
            legal[offset + policyIndex] = sample.board.isLegal(moveId) ? 1.0f : 0.0f;
            policy[offset + policyIndex] =
                static_cast<float>(sample.visits[moveId]) / totalVisits;
        }
        outcomes[batchIndex] = sample.outcome;
    }
    return {
        mlx::core::stack(inputs),
        mlx::core::array(legal.data(), {static_cast<int>(batchSize), moveIdCount}, mlx::core::float32),
        mlx::core::array(policy.data(), {static_cast<int>(batchSize), moveIdCount}, mlx::core::float32),
        mlx::core::array(outcomes.data(), {static_cast<int>(batchSize)}, mlx::core::float32),
    };
}

std::filesystem::path temporaryPath(const std::filesystem::path& path)
{
    return path.parent_path()
        / (path.filename().string() + ".tmp.safetensors");
}

void saveAtomically(const Network& network, const std::filesystem::path& path,
                    const CheckpointMetadata& metadata = {})
{
    const std::filesystem::path temporary = temporaryPath(path);
    std::filesystem::remove(temporary);
    network.save(temporary, metadata);
    std::filesystem::rename(temporary, path);
}

double lowerConfidenceBound(std::span<const double> observations)
{
    assert(observations.size() >= 2);
    const double mean = std::accumulate(
        observations.begin(), observations.end(), 0.0) / observations.size();
    double squaredError = 0.0;
    for (const double observation : observations)
        squaredError += (observation - mean) * (observation - mean);
    const double variance = squaredError / (observations.size() - 1);
    return std::max(0.0, mean - gateConfidenceZ
        * std::sqrt(variance / observations.size()));
}

double MatchScore::lowerConfidenceBound() const
{
    return amoeba_bot::lowerConfidenceBound(observations);
}

struct PairedMatchScore
{
    explicit PairedMatchScore(size_t pairCount)
        : pairScores(pairCount), gamesPerPair(pairCount)
    {
    }

    void add(size_t gameIndex, Outcome outcome, bool candidateWhite,
             uint64_t plyCount)
    {
        assert(gameIndex / 2 < pairScores.size());
        games.add(outcome, candidateWhite, plyCount);
        const double gameScore = games.observations.back();
        pairScores[gameIndex / 2] += 0.5 * gameScore;
        ++gamesPerPair[gameIndex / 2];
    }

    double score() const { return games.score(); }
    int gameCount() const { return games.games(); }
    double lowerConfidenceBound() const
    {
        assert(std::ranges::all_of(gamesPerPair,
                                  [](uint8_t count) { return count == 2; }));
        return amoeba_bot::lowerConfidenceBound(pairScores);
    }

    MatchScore games;
    std::vector<double> pairScores;
    std::vector<uint8_t> gamesPerPair;
};

std::string baselineMetadataName(std::string_view suffix)
{
    return std::format("{}{}", baselineMetadataPrefix, suffix);
}

bool hasBaselineQualification(const CheckpointMetadata& metadata)
{
    const auto version = metadata.find(baselineMetadataName("version"));
    const auto passed = metadata.find(baselineMetadataName("passed"));
    return version != metadata.end() && version->second == baselineGateVersion
        && passed != metadata.end() && passed->second == "true";
}

void storeBaselineScore(CheckpointMetadata& metadata, std::string_view opponent,
                        const MatchScore& score)
{
    const auto name = [&](std::string_view field) {
        return baselineMetadataName(std::format("{}.{}", opponent, field));
    };
    metadata.insert_or_assign(name("score"), std::format("{}", score.score()));
    metadata.insert_or_assign(
        name("lower_bound"), std::format("{}", score.lowerConfidenceBound()));
    metadata.insert_or_assign(name("wins"), std::format("{}", score.wins));
    metadata.insert_or_assign(name("draws"), std::format("{}", score.draws));
    metadata.insert_or_assign(name("losses"), std::format("{}", score.losses));
    metadata.insert_or_assign(name("games"), std::format("{}", score.games()));
    metadata.insert_or_assign(
        name("total_plies"), std::format("{}", score.totalPlies));
}

void storeBaselineQualification(CheckpointMetadata& metadata,
                                const MatchScore& randomScore,
                                const MatchScore& rolloutScore)
{
    metadata.insert_or_assign(
        baselineMetadataName("version"), std::string{baselineGateVersion});
    metadata.insert_or_assign(baselineMetadataName("passed"), "true");
    storeBaselineScore(metadata, "random", randomScore);
    storeBaselineScore(metadata, "rollout_mcts", rolloutScore);
}

uint64_t loadRetainedStep(const CheckpointMetadata& metadata)
{
    const auto found = metadata.find(std::string{retainedStepMetadataName});
    if (found == metadata.end())
        return 0;

    uint64_t step = 0;
    const char* const begin = found->second.data();
    const char* const end = begin + found->second.size();
    const auto [parsedEnd, error] = std::from_chars(begin, end, step);
    if (error != std::errc{} || parsedEnd != end)
        throw std::runtime_error(std::format(
            "checkpoint metadata {} is not a valid step: {}",
            retainedStepMetadataName, found->second));
    return step;
}

void storeRetainedStep(CheckpointMetadata& metadata, uint64_t step)
{
    metadata.insert_or_assign(
        std::string{retainedStepMetadataName}, std::format("{}", step));
}

Evaluation rolloutEvaluation(const Board& leaf, std::mt19937_64& randomEngine)
{
    Evaluation evaluation;
    const float prior = 1.0f / leaf.legalMoveCount;
    leaf.forEachLegal([&](uint16_t moveId) { evaluation.policy[moveId] = prior; });
    Board board = leaf;
    std::vector<uint64_t> history{board.positionHash};
    for (;;)
    {
        MoveResult result = applyMove(board,
            Move::fromId(randomLegalMove(board, randomEngine)), history);
        if (const auto* outcome = std::get_if<Outcome>(&result))
        {
            evaluation.value = outcomeFor(*outcome, leaf.whiteToMove);
            return evaluation;
        }
        board = std::get<Board>(std::move(result));
        history.push_back(board.positionHash);
    }
}

enum class Baseline { random, rollout };

struct BaselineGame
{
    Board board;
    std::vector<uint64_t> history;
    std::optional<MCTS<evaluationSimulationCount>> search;
    const Board* pendingLeaf = nullptr;
    std::optional<Outcome> outcome;
    std::mt19937_64 random;
    uint64_t gameId = 0;
    size_t evaluationOffset = 0;
    bool candidateWhite = false;
};

void initialize(BaselineGame& game, uint64_t gameId, uint64_t randomSeed)
{
    game.board = Board::startingBoard();
    game.history.assign(1, game.board.positionHash);
    game.search.reset();
    game.pendingLeaf = nullptr;
    game.outcome.reset();
    game.random.seed(randomSeed ^ (0x9e3779b97f4a7c15ULL * (gameId + 1)));
    game.gameId = gameId;
    game.candidateWhite = gameId % 2 == 0;
}

void applyEvaluationMove(BaselineGame& game, uint16_t moveId)
{
    MoveResult result = applyMove(game.board, Move::fromId(moveId), game.history);
    game.search.reset();
    game.pendingLeaf = nullptr;
    if (const auto* outcome = std::get_if<Outcome>(&result))
        game.outcome = *outcome;
    else
    {
        game.board = std::get<Board>(std::move(result));
        game.history.push_back(game.board.positionHash);
    }
}

void advance(BaselineGame& game, Baseline baseline)
{
    while (!game.outcome.has_value())
    {
        const bool candidateTurn = game.board.whiteToMove == game.candidateWhite;
        if (!candidateTurn && baseline == Baseline::random)
        {
            applyEvaluationMove(game, randomLegalMove(game.board, game.random));
            continue;
        }
        if (!game.search.has_value())
            game.search.emplace(game.board, game.history);
        if (const Board* leaf = game.search->pendingLeaf())
        {
            if (candidateTurn)
            {
                game.pendingLeaf = leaf;
                return;
            }
            const Evaluation evaluation = rolloutEvaluation(*leaf, game.random);
            game.search->absorb(evaluation.policy, evaluation.value);
            continue;
        }
        applyEvaluationMove(game, bestMove(game.search->visits()));
    }
}

template<int GameCount>
bool playBaselineGames(const Network& candidate, Baseline baseline,
                       uint64_t firstGameId, uint64_t randomSeed,
                       MatchScore& score)
{
    const int slotCount = std::min(GameCount, concurrentEvaluationGames);
    std::vector<BaselineGame> games(static_cast<size_t>(slotCount));
    int started = 0;
    int completed = 0;
    for (BaselineGame& game : games)
        initialize(game, firstGameId + started++, randomSeed);
    ThreadPool workers{mctsWorkerCount()};
    std::vector<const Board*> pendingBoards;
    std::vector<Evaluation> evaluations;
    while (completed < GameCount)
    {
        std::vector<std::future<void>> advanced;
        advanced.reserve(games.size());
        for (BaselineGame& game : games)
        {
            if (!game.outcome.has_value() || started < GameCount)
                advanced.push_back(workers.submit([&game, baseline] {
                    advance(game, baseline);
                }));
        }
        for (std::future<void>& future : advanced)
            future.get();

        pendingBoards.clear();
        for (BaselineGame& game : games)
        {
            if (game.outcome.has_value() && started >= GameCount)
                continue;
            if (game.outcome.has_value())
            {
                score.add(*game.outcome, game.candidateWhite,
                          completedGameRounds(game.board));
                ++completed;
                assert(score.games() == completed);
                updateGateStatus(score);
                if (started >= GameCount)
                    continue;
                initialize(game, firstGameId + started++, randomSeed);
            }
            if (!game.outcome.has_value())
            {
                game.evaluationOffset = pendingBoards.size();
                pendingBoards.push_back(game.pendingLeaf);
            }
        }
        if (completed == GameCount)
            break;
        evaluations.resize(pendingBoards.size());
        candidate(pendingBoards, evaluations);
        for (BaselineGame& game : games)
        {
            if (game.outcome.has_value() || game.pendingLeaf == nullptr)
                continue;
            const Evaluation& evaluation = evaluations[game.evaluationOffset];
            game.search->absorb(evaluation.policy, evaluation.value);
            game.pendingLeaf = nullptr;
        }
        if (stopRequested)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Parameter health tripwire.
//
// The promotion gate cannot see a network whose tensors are dying: the 2026
// collapse promoted a champion that had already lost three quarters of its
// parameters, because a mutilated network still beats fixed baselines. A
// LayerNorm scale at zero also gates its whole branch off from gradient
// forever, so by the time play degrades nothing is trainable. The only
// reliable alarm is the parameter magnitudes themselves, compared against a
// fresh initialization.
//
// Negative test (an alarm that has never fired is not an alarm): on a scratch
// weights file, temporarily re-add the old L2 penalty term to the loss in
// Network::computeLoss and confirm a WARNING within ~150 steps and an abort
// well before step 2,500 - or zero one norm scale in the live file and
// confirm the startup sweep refuses it. Then revert.
// ---------------------------------------------------------------------------

// LayerNorm scales initialize at exactly 1.0, so their floor is absolute and
// survives restarts; matrices are judged relative to their own fresh-init rms.
constexpr float scaleAbortRms = 0.1f;
constexpr float scaleWarnRms = 0.5f;
constexpr float scaleInflateWarnRms = 4.0f;
constexpr float matrixAbortRatio = 0.10f;
constexpr float matrixWarnRatio = 0.25f;
constexpr float matrixInflateWarnRatio = 8.0f;
constexpr int matrixAbortSweeps = 2;
// Zero-initialized tensors (biases, norm shifts, relation biases) have no
// meaningful reference magnitude and are skipped; their branches are covered
// through the sibling tensors that always die with them.
constexpr float monitoredReferenceRms = 1e-3f;

// A consecutive streak catches a broken network; the window catches a steady
// trickle (e.g. one poisoned replay sample resampled at ~0.085% per batch,
// which would never produce 10 in a row).
constexpr int nonFiniteAbortConsecutive = 10;
constexpr int nonFiniteAbortPerWindow = 50;

std::vector<float> tensorRms(const std::vector<mlx::core::array>& tensors)
{
    std::vector<mlx::core::array> values;
    values.reserve(tensors.size());
    for (const mlx::core::array& tensor : tensors)
        values.push_back(mlx::core::sqrt(mlx::core::mean(mlx::core::square(tensor))));
    mlx::core::eval(values);

    std::vector<float> result;
    result.reserve(values.size());
    for (mlx::core::array& value : values)
        result.push_back(value.item<float>());
    return result;
}

class ParameterHealthMonitor
{
public:
    explicit ParameterHealthMonitor(const Network& network)
    {
        // The reference is always a fresh network built from the layout's own
        // initialization, never from loaded weights: re-anchoring to a resumed
        // checkpoint would silently accept whatever erosion it already has.
        const std::unique_ptr<Network> reference =
            createNetwork(network.name(), healthReferenceSeed);
        m_referenceRms = tensorRms(reference->parameters());
        m_names.reserve(m_referenceRms.size());
        for (size_t index = 0; index < m_referenceRms.size(); ++index)
            m_names.emplace_back(network.parameterName(index));
        m_strikes.assign(m_names.size(), 0);
        m_warnedLow.assign(m_names.size(), 0);
        m_warnedHigh.assign(m_names.size(), 0);
    }

    // Throws to stop the run. Generation training never touches the champion
    // checkpoint until a candidate has passed the gate, so a failed candidate
    // needs no checkpoint quarantine.
    void check(const std::vector<mlx::core::array>& parameters, uint64_t step)
    {
        assert(parameters.size() == m_names.size());
        const std::vector<float> rms = tensorRms(parameters);

        float worstRatio = std::numeric_limits<float>::infinity();
        float largestRatio = 0.0f;
        size_t worstIndex = 0;
        size_t largestIndex = 0;
        for (size_t index = 0; index < rms.size(); ++index)
        {
            // rms < threshold is false for NaN, so non-finite needs its own test.
            if (!std::isfinite(rms[index]))
                throw std::runtime_error(std::format(
                    "parameter health: {} is non-finite at step {}", m_names[index], step));

            if (m_names[index].ends_with(".scale"))
            {
                if (rms[index] < scaleAbortRms)
                    throw std::runtime_error(std::format(
                        "parameter health: norm scale {} has collapsed to rms {:.4f} at step {}"
                        " - its branch is (or is about to be) cut off from gradient",
                        m_names[index], rms[index], step));
                warnOnTransition(index, rms[index], scaleWarnRms, scaleInflateWarnRms, step);
            }
            else if (m_referenceRms[index] > monitoredReferenceRms)
            {
                const float ratio = rms[index] / m_referenceRms[index];
                if (ratio < matrixAbortRatio)
                {
                    // Two consecutive sweeps, because a single dip below 10%
                    // is no longer an absorbing state under decoupled decay
                    // and a false 3 a.m. abort is the one failure this
                    // monitor could newly introduce.
                    if (++m_strikes[index] >= matrixAbortSweeps)
                        throw std::runtime_error(std::format(
                            "parameter health: {} fell to {:.1f}% of its initialization"
                            " for {} consecutive sweeps at step {}",
                            m_names[index], 100.0f * ratio, matrixAbortSweeps, step));
                }
                else
                    m_strikes[index] = 0;
                warnOnTransition(index, ratio, matrixWarnRatio, matrixInflateWarnRatio, step);

                if (ratio < worstRatio)
                {
                    worstRatio = ratio;
                    worstIndex = index;
                }
                if (ratio > largestRatio)
                {
                    largestRatio = ratio;
                    largestIndex = index;
                }
            }
        }

        if (step % 1000 == 0)
            report("[health] step {}: weakest {} at {:.0f}% of init, largest {} at {:.1f}x init",
                   step, m_names[worstIndex], 100.0f * worstRatio,
                   m_names[largestIndex], largestRatio);
    }

private:
    // Warnings fire on entering the bad band, not every sweep: a repeated
    // warning becomes wallpaper, and wallpaper is how the last collapse ran
    // for a whole night unnoticed.
    void warnOnTransition(size_t index, float value, float low, float high, uint64_t step)
    {
        const bool isLow = value < low;
        const bool isHigh = value > high;
        if (isLow && !m_warnedLow[index])
            report("[health] WARNING step {}: {} is down to {:.3f} (warn threshold {})",
                   step, m_names[index], value, low);
        if (isHigh && !m_warnedHigh[index])
            report("[health] WARNING step {}: {} has grown to {:.2f} (warn threshold {})",
                   step, m_names[index], value, high);
        m_warnedLow[index] = isLow;
        m_warnedHigh[index] = isHigh;
    }

    std::vector<std::string> m_names;
    std::vector<float> m_referenceRms;
    std::vector<uint8_t> m_strikes;
    std::vector<uint8_t> m_warnedLow;
    std::vector<uint8_t> m_warnedHigh;
};

} // namespace

int runTraining(const std::filesystem::path& weights,
                std::string_view networkIdentifier)
{
    stopRequested = 0;
    try
    {
        installSignalHandlers();
        const uint64_t runSeed = makeRunSeed();
        std::filesystem::path legacyLiveWeights = weights;
        legacyLiveWeights += ".live.safetensors";
        const bool importLegacyLive = std::filesystem::exists(legacyLiveWeights)
            && (!std::filesystem::exists(weights)
                || std::filesystem::last_write_time(legacyLiveWeights)
                    > std::filesystem::last_write_time(weights));
        std::unique_ptr<Network> network;
        if (importLegacyLive)
            network = loadNetwork(legacyLiveWeights, networkIdentifier);
        else if (std::filesystem::exists(weights))
            network = loadNetwork(weights, networkIdentifier);
        else if (!networkIdentifier.empty())
            network = createNetwork(networkIdentifier, runSeed);
        else
            throw std::runtime_error(std::format(
                "{} does not exist; select a network with --network",
                weights.string()));
        CheckpointMetadata checkpointMetadata = network->checkpointMetadata();
        bool baselineQualified = hasBaselineQualification(checkpointMetadata);
        uint64_t optimizerStep = loadRetainedStep(checkpointMetadata);
        storeRetainedStep(checkpointMetadata, optimizerStep);
        ParameterHealthMonitor healthMonitor{*network};
        healthMonitor.check(network->parameters(), optimizerStep);
        if (importLegacyLive)
        {
            // The previous method's .live file contains its newest network plus
            // Adam state. Import only the network once: saving the champion now
            // makes weights newer than the untouched legacy sidecar, so a later
            // restart cannot roll a promoted champion back to stale live data.
            saveAtomically(*network, weights, checkpointMetadata);
            report("[train] imported current network from {} and reset optimizer state",
                   legacyLiveWeights.string());
        }
        else if (!std::filesystem::exists(weights))
            saveAtomically(*network, weights, checkpointMetadata);
        report("[train] {}: {}, {} parameters", weights.string(),
               network->name(), network->parameterCount());
        report("[train] run seed: {}", runSeed);
        report("[train] retained step: {}", optimizerStep);
        if (baselineQualified)
            report("[train] checkpoint already passed baseline gate version {}; "
                   "skipping random and rollout-mcts",
                   baselineGateVersion);
        report("[train] generation method: {} self-play games, {} MCTS workers, "
               "up to {} leaves/batch, "
               "{} visits, AdamW lr {}, terminal discount/move {}, up to {} epochs, "
               "{} validations/epoch",
               selfPlayGameCount, mctsWorkerCount(), selfPlayEvaluationBatchSize,
               selfPlaySimulationCount, learningRate, terminalValueDiscountPerMove,
               maximumTrainingEpochs,
               validationPointsPerEpoch);

        std::mt19937_64 randomEngine{runSeed};
        uint64_t nextGameId = 0;
        uint64_t generation = 1;
        size_t pooledGames = 0;
        std::vector<TrainingSample> training;
        std::vector<TrainingSample> heldOut;

        while (!stopRequested)
        {
            const uint64_t championStep = optimizerStep;
            report("[generation {}] self-play started from the champion", generation);
            CompletedSelfPlay generated =
                generateSelfPlay(*network, nextGameId, runSeed);
            if (stopRequested)
                break;

            const size_t generatedPositionCount = generated.samples.size();
            training.reserve(training.size() + generatedPositionCount);
            heldOut.reserve(
                heldOut.size() + generatedPositionCount / heldOutGameFraction + 1);
            for (TrainingSample& sample : generated.samples)
            {
                std::vector<TrainingSample>& destination =
                    isHeldOutGame(sample.gameId) ? heldOut : training;
                destination.push_back(std::move(sample));
            }
            pooledGames += selfPlayGameCount;
            if (training.empty() || heldOut.empty())
                throw std::runtime_error(
                    "self-play generation did not produce both training and held-out positions");
            report("[generation {}] {} new positions, {:.1f} rounds/game, "
                   "{:.1f} positions/s; champion pool: {} games, "
                   "{} positions ({} train, {} held out)",
                   generation, generatedPositionCount, generated.averagePlies(),
                   generated.positionsPerSecond, pooledGames,
                   training.size() + heldOut.size(), training.size(), heldOut.size());

            std::vector<mlx::core::array> parameters = network->parameters();
            Adam adam{parameters};
            const std::vector<uint8_t> decayMask = network->weightDecayMask();
            std::vector<size_t> trainingOrder(training.size());
            std::iota(trainingOrder.begin(), trainingOrder.end(), size_t{0});
            std::vector<size_t> heldOutOrder(heldOut.size());
            std::iota(heldOutOrder.begin(), heldOutOrder.end(), size_t{0});
            std::vector<mlx::core::array> bestParameters = parameters;
            int bestEpoch = 0;
            int bestEpochPercent = 0;
            uint64_t bestOptimizerStep = optimizerStep;
            int staleEpochs = 0;
            int consecutiveNonFinite = 0;
            int nonFiniteInWindow = 0;
            size_t batchAttemptCount = 0;
            std::array<bool, 1000> nonFiniteWindow{};

            const auto validationLoss = [&](const std::vector<mlx::core::array>& values) {
                double policy = 0.0;
                double value = 0.0;
                size_t positions = 0;
                for (size_t offset = 0; offset < heldOutOrder.size();
                     offset += trainingBatchSize)
                {
                    const size_t count = std::min(
                        trainingBatchSize, heldOutOrder.size() - offset);
                    const TrainingBatch batch = createBatch(
                        heldOut,
                        std::span<const size_t>{heldOutOrder.data() + offset, count},
                        randomEngine, false);
                    const Loss loss = network->loss(values, batch);
                    mlx::core::eval({loss.policy, loss.value});
                    policy += count * loss.policy.item<float>();
                    value += count * loss.value.item<float>();
                    positions += count;
                }
                return std::array<double, 2>{policy / positions, value / positions};
            };

            const std::array<double, 2> baselineHeldOutLoss =
                validationLoss(parameters);
            double bestValidation =
                baselineHeldOutLoss[0] + baselineHeldOutLoss[1];
            report("[train] generation {}, baseline step {}: held-out {:.4f}+{:.4f}",
                   generation, optimizerStep,
                   baselineHeldOutLoss[0], baselineHeldOutLoss[1]);

            for (int epoch = 1; epoch <= maximumTrainingEpochs && !stopRequested;
                 ++epoch)
            {
                std::ranges::shuffle(trainingOrder, randomEngine);
                const size_t epochBatchCount =
                    (trainingOrder.size() + trainingBatchSize - 1) / trainingBatchSize;
                int nextValidationPoint = 1;
                bool epochImproved = false;
                double trainPolicy = 0.0;
                double trainValue = 0.0;
                double gradientNormTotal = 0.0;
                size_t trainedPositions = 0;
                size_t completedBatches = 0;
                for (size_t offset = 0; offset < trainingOrder.size();
                     offset += trainingBatchSize)
                {
                    const size_t count = std::min(
                        trainingBatchSize, trainingOrder.size() - offset);
                    const TrainingBatch batch = createBatch(
                        training,
                        std::span<const size_t>{trainingOrder.data() + offset, count},
                        randomEngine, true);
                    const LossAndGrad result = network->valueAndGrad(parameters, batch);
                    mlx::core::array gradientNormSquared = mlx::core::array(0.0f);
                    for (const mlx::core::array& gradient : result.gradients)
                        gradientNormSquared = gradientNormSquared
                            + mlx::core::sum(mlx::core::square(gradient));
                    mlx::core::eval(
                        {result.loss.policy, result.loss.value, gradientNormSquared});
                    const float policyLoss = result.loss.policy.item<float>();
                    const float valueLoss = result.loss.value.item<float>();
                    const float gradientNorm =
                        std::sqrt(gradientNormSquared.item<float>());
                    const bool nonFinite = !std::isfinite(policyLoss)
                        || !std::isfinite(valueLoss) || !std::isfinite(gradientNorm);
                    nonFiniteInWindow -=
                        nonFiniteWindow[batchAttemptCount % nonFiniteWindow.size()];
                    nonFiniteWindow[batchAttemptCount % nonFiniteWindow.size()] = nonFinite;
                    nonFiniteInWindow += nonFinite;
                    ++batchAttemptCount;
                    if (nonFinite)
                    {
                        ++consecutiveNonFinite;
                        report("[train] skipped a non-finite batch ({} consecutive, "
                               "{} in the last {} attempts)",
                               consecutiveNonFinite, nonFiniteInWindow,
                               nonFiniteWindow.size());
                        if (consecutiveNonFinite >= nonFiniteAbortConsecutive
                            || nonFiniteInWindow > nonFiniteAbortPerWindow)
                            throw std::runtime_error(
                                "training keeps producing non-finite batches");
                    }
                    else
                    {
                        consecutiveNonFinite = 0;
                        parameters = adam.updateParameters(
                            parameters, result.gradients, learningRate,
                            weightDecay, decayMask);
                        mlx::core::eval(parameters);
                        ++optimizerStep;
                        trainPolicy += count * policyLoss;
                        trainValue += count * valueLoss;
                        gradientNormTotal += gradientNorm;
                        trainedPositions += count;
                        ++completedBatches;
                    }
                    if (stopRequested)
                        break;

                    const size_t processedBatches =
                        offset / trainingBatchSize + 1;
                    while (nextValidationPoint <= validationPointsPerEpoch
                           && processedBatches * validationPointsPerEpoch
                               >= epochBatchCount * nextValidationPoint)
                    {
                        const int epochPercent =
                            100 * nextValidationPoint / validationPointsPerEpoch;
                        const std::array<double, 2> heldOutLoss =
                            validationLoss(parameters);
                        const double validation = heldOutLoss[0] + heldOutLoss[1];
                        report("[train] generation {}, epoch {}/{} {}%, step {}: "
                               "held-out {:.4f}+{:.4f}",
                               generation, epoch, maximumTrainingEpochs, epochPercent,
                               optimizerStep, heldOutLoss[0], heldOutLoss[1]);
                        if (validation + validationImprovement < bestValidation)
                        {
                            bestValidation = validation;
                            bestParameters = parameters;
                            bestEpoch = epoch;
                            bestEpochPercent = epochPercent;
                            bestOptimizerStep = optimizerStep;
                            epochImproved = true;
                        }
                        ++nextValidationPoint;
                    }
                }
                if (stopRequested)
                    break;
                if (trainedPositions == 0)
                    throw std::runtime_error("an epoch contained no finite training batch");

                healthMonitor.check(parameters, optimizerStep);
                report("[train] generation {}, epoch {}/{} complete, step {}: "
                       "policy {:.4f}, value {:.4f}, |g| {:.2f}",
                       generation, epoch, maximumTrainingEpochs, optimizerStep,
                       trainPolicy / trainedPositions, trainValue / trainedPositions,
                       gradientNormTotal / completedBatches);
                if (epochImproved)
                    staleEpochs = 0;
                else if (++staleEpochs >= earlyStoppingPatience)
                {
                    if (bestEpoch == 0)
                        report("[train] generation {} stopped after epoch {}; "
                               "held-out loss never improved on the baseline",
                               generation, epoch);
                    else
                        report("[train] generation {} stopped after epoch {}; "
                               "held-out loss last improved at epoch {} {}%",
                               generation, epoch, bestEpoch, bestEpochPercent);
                    break;
                }
            }
            if (stopRequested)
                break;
            parameters = std::move(bestParameters);
            optimizerStep = bestOptimizerStep;
            healthMonitor.check(parameters, optimizerStep);
            network->replaceParameters(parameters);
            if (bestEpoch == 0)
            {
                report("[train] generation {}: no checkpoint improved on the baseline; "
                       "gate skipped", generation);
                report("[generation {}] retaining {} champion self-play games and "
                       "generating more", generation, pooledGames);
                ++generation;
                continue;
            }
            report("[train] generation {} candidate uses epoch {} {}%, step {}",
                   generation, bestEpoch, bestEpochPercent, optimizerStep);

            const uint64_t gateBase = generation * 1'000'000;
            bool baselinesPassed = true;
            std::optional<MatchScore> randomScore;
            std::optional<MatchScore> rolloutScore;
            if (!baselineQualified)
            {
                randomScore.emplace();
                startGateStatus(generation, "random", randomGameCount);
                const bool randomCompleted = playBaselineGames<randomGameCount>(
                    *network, Baseline::random, gateBase + championGameCount,
                    runSeed + 8888 + generation, *randomScore);
                finishGateStatus();
                if (!randomCompleted)
                    break;
                report("[gate] generation {} vs random: {:.1f}% "
                       "({}W/{}D/{}L, 95% lower {:.1f}%, {:.1f} rounds/game) "
                       "over {} games",
                       generation, 100.0 * randomScore->score(),
                       randomScore->wins, randomScore->draws, randomScore->losses,
                       100.0 * randomScore->lowerConfidenceBound(),
                       randomScore->averagePlies(), randomScore->games());

                rolloutScore.emplace();
                startGateStatus(generation, "rollout-mcts", rolloutGameCount);
                const bool rolloutCompleted = playBaselineGames<rolloutGameCount>(
                    *network, Baseline::rollout,
                    gateBase + championGameCount + randomGameCount,
                    runSeed + 9999 + generation, *rolloutScore);
                finishGateStatus();
                if (!rolloutCompleted)
                    break;
                report("[gate] generation {} vs rollout-mcts: {:.1f}% "
                       "({}W/{}D/{}L, 95% lower {:.1f}%, {:.1f} rounds/game) "
                       "over {} games",
                       generation, 100.0 * rolloutScore->score(),
                       rolloutScore->wins, rolloutScore->draws, rolloutScore->losses,
                       100.0 * rolloutScore->lowerConfidenceBound(),
                       rolloutScore->averagePlies(), rolloutScore->games());
                baselinesPassed = randomScore->lowerConfidenceBound() > 0.5
                    && rolloutScore->lowerConfidenceBound() > 0.5;
            }
            if (stopRequested)
                break;

            std::unique_ptr<Network> champion = loadNetwork(weights, network->name());
            PairedMatchScore championScore(championGameCount / 2);
            const Network* gateNetworks[]{network.get(), champion.get()};
            NetworkMatchRunner<evaluationSimulationCount> championRunner{gateNetworks};
            startGateStatus(generation, "champion", championGameCount);
            const bool championCompleted =
                championRunner.template play<championGameCount>(
                    gateBase, runSeed + 7777 + generation,
                    [&](uint64_t gameId, Outcome outcome,
                        uint64_t plyCount) -> const MatchScore& {
                        const size_t gameIndex = static_cast<size_t>(gameId - gateBase);
                        championScore.add(
                            gameIndex, outcome, gameIndex % 2 == 0, plyCount);
                        return championScore.games;
                    });
            finishGateStatus();
            if (!championCompleted)
                break;
            const double championLowerBound =
                championScore.lowerConfidenceBound();
            report("[gate] generation {} vs champion: {:.1f}% "
                   "({}W/{}D/{}L, paired 95% lower {:.1f}%, "
                   "{:.1f} rounds/game) over {} games",
                   generation, 100.0 * championScore.score(),
                   championScore.games.wins, championScore.games.draws,
                   championScore.games.losses, 100.0 * championLowerBound,
                   championScore.games.averagePlies(), championScore.gameCount());

            const bool promoted = baselinesPassed && championLowerBound > 0.5;
            report("[gate] generation {}: {}", generation,
                   promoted ? "PROMOTED" : "rejected");
            if (promoted)
            {
                bool storedBaselineQualification = false;
                if (!baselineQualified)
                {
                    assert(randomScore.has_value());
                    assert(rolloutScore.has_value());
                    assert(baselinesPassed);
                    storeBaselineQualification(
                        checkpointMetadata, *randomScore, *rolloutScore);
                    baselineQualified = true;
                    storedBaselineQualification = true;
                }
                storeRetainedStep(checkpointMetadata, optimizerStep);
                saveAtomically(*network, weights, checkpointMetadata);
                if (storedBaselineQualification)
                    report("[gate] generation {} baseline qualification saved "
                           "in the champion checkpoint", generation);
                report("[generation {}] promoted; cleared {} champion self-play games",
                       generation, pooledGames);
                std::vector<TrainingSample>{}.swap(training);
                std::vector<TrainingSample>{}.swap(heldOut);
                pooledGames = 0;
            }
            else
            {
                optimizerStep = championStep;
                network = std::move(champion);
                report("[generation {}] rejected; restored champion step {}, "
                       "retaining {} champion self-play games",
                       generation, optimizerStep, pooledGames);
            }
            ++generation;
        }

        finishSelfPlayStatus();
        report("[train] shutdown complete; champion remains {}", weights.string());
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        finishSelfPlayStatus();
        finishGateStatus();
        std::println(stderr, "[train] {}", error.what());
        return EXIT_FAILURE;
    }
}

} // namespace amoeba_bot
