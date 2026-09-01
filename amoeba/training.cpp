#include "training.hpp"

#include "mcts.hpp"
#include "network.hpp"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace amoeba_bot
{

namespace
{

constexpr uint64_t seed = 20260819;
// A 256-visit search is large enough to make the policy target materially less
// sparse than the old 200-visit target without doubling the dominant cost of
// every generation. Self-play and the gate use the same search budget so the
// comparison does not reward a candidate under different search conditions.
constexpr int selfPlaySimulationCount = 256;
constexpr int evaluationSimulationCount = 256;
// One lane is one MLX inference batch. While its 128 leaves are on the GPU,
// the host advances MCTS for the other lane.
constexpr int selfPlayLaneSize = 128;
constexpr int selfPlayGameCount = 2 * selfPlayLaneSize;
// Two hundred paired openings give 400 games. Promotion uses the lower bound
// across the 200 pair scores, rather than a raw win-rate threshold.
constexpr int championGameCount = 400;
constexpr int evaluationOpeningPlyCount = 4;
constexpr int randomGameCount = 128;
constexpr int rolloutGameCount = 128;
constexpr int concurrentEvaluationGames = 128;
constexpr int samplingPlyCount = 20;
constexpr float rootNoise = 0.25f;
constexpr float noiseAlpha = 0.35f;
constexpr size_t trainingBatchSize = 256;
// The split is by game, never by position. Thirty-two of each generation's
// 256 games are expected to be validation-only, which is enough to stop an
// epoch sequence that has begun fitting its game outcomes instead of learning
// a better position evaluator.
constexpr uint64_t heldOutGameFraction = 8;
constexpr int maximumTrainingEpochs = 8;
constexpr int earlyStoppingPatience = 2;
constexpr float validationImprovement = 1e-4f;
// Each generation fine-tunes an already trained champion on only 256 new
// games. AdamW at 1e-4 makes that update deliberate; the gate, not a large
// optimizer jump, decides whether it was useful.
constexpr float learningRate = 1e-4f;
constexpr float weightDecay = 1e-2f;
constexpr double gateConfidenceZ = 1.645; // one-sided 95% lower bound

volatile sig_atomic_t stopRequested = 0;

struct SelfPlayStatus
{
    uint64_t positions = 0;
    uint64_t totalPositions = 0;
    double positionsPerSecond = 0.0;
    std::chrono::steady_clock::time_point startedAt =
        std::chrono::steady_clock::now();
    bool visible = false;
};

SelfPlayStatus selfPlayStatus;

void clearTerminalLine()
{
    if (isatty(STDOUT_FILENO))
    {
        std::print("\r\x1b[2K");
        std::fflush(stdout);
    }
}

void drawSelfPlayStatus()
{
    if (!selfPlayStatus.visible || !isatty(STDOUT_FILENO))
        return;
    std::print("\r\x1b[2K[selfplay] generated {} positions, {:.1f} positions/s average",
               selfPlayStatus.positions, selfPlayStatus.positionsPerSecond);
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

void reportSelfPlayStatus(uint64_t generatedPositions)
{
    selfPlayStatus.positions += generatedPositions;
    selfPlayStatus.totalPositions += generatedPositions;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selfPlayStatus.startedAt).count();
    selfPlayStatus.positionsPerSecond = selfPlayStatus.totalPositions / elapsed;
    selfPlayStatus.visible = true;
    if (isatty(STDOUT_FILENO))
        drawSelfPlayStatus();
    else
    {
        std::println("[selfplay] generated {} positions, {:.1f} positions/s average",
                     selfPlayStatus.positions, selfPlayStatus.positionsPerSecond);
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
    std::mt19937_64 randomEngine{seed + 7777};
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
    for (TrainingSample& sample : game.samples)
        sample.outcome = outcomeFor(*game.outcome, sample.board.whiteToMove);
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
        std::array<std::vector<const Board*>, 2> pendingBoards;
        std::array<std::vector<Evaluation>, 2> evaluations;
        while (completed < GameCount)
        {
            for (std::vector<const Board*>& boards : pendingBoards)
                boards.clear();
            for (ActiveGame<SimulationCount>& game : games)
            {
                if (!game.active)
                    continue;
                advance<SimulationCount, false>(game);
                if (game.pendingLeaf == nullptr)
                {
                    finish(game);
                    gameFinished(game.gameId, *game.outcome);
                    ++completed;
                    if (started < GameCount)
                    {
                        initializeGame(game);
                        advance<SimulationCount, false>(game);
                    }
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
    SelfPlayLane(uint64_t firstGameId, uint64_t randomSeed)
        : m_games(selfPlayLaneSize), m_activeGames(selfPlayLaneSize)
    {
        for (size_t gameIndex = 0; gameIndex < m_games.size(); ++gameIndex)
            initialize(m_games[gameIndex], firstGameId + gameIndex,
                       NetworkPairing{0, 0}, randomSeed);
    }

    std::vector<TrainingSample> prepare()
    {
        assert(!m_pending.has_value());
        assert(!m_prepared);
        m_pendingBoards.clear();
        std::vector<TrainingSample> completed;
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            if (!game.active)
                continue;
            advance<selfPlaySimulationCount, true>(game);
            if (game.pendingLeaf == nullptr)
            {
                finish(game);
                completed.insert(completed.end(),
                                 std::make_move_iterator(game.samples.begin()),
                                 std::make_move_iterator(game.samples.end()));
                game.active = false;
                --m_activeGames;
                continue;
            }
            game.evaluationOffset = m_pendingBoards.size();
            m_pendingBoards.push_back(game.pendingLeaf);
        }
        m_prepared = !m_pendingBoards.empty();
        return completed;
    }

    void submit(const Network& network)
    {
        assert(m_prepared);
        assert(!m_pending.has_value());
        m_evaluations.resize(m_pendingBoards.size());
        m_pending.emplace(network.submit(m_pendingBoards));
        m_prepared = false;
    }

    void absorb(const Network& network)
    {
        assert(m_pending.has_value());
        network.finish(std::move(*m_pending), m_evaluations);
        m_pending.reset();
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            if (!game.active)
                continue;
            assert(game.pendingLeaf != nullptr);
            Evaluation& evaluation = m_evaluations[game.evaluationOffset];
            if (game.firstLeafOfSearch)
                addExplorationNoise(evaluation, *game.pendingLeaf, game.noiseRandom);
            game.search->absorb(evaluation.policy, evaluation.value);
            game.firstLeafOfSearch = false;
            game.pendingLeaf = nullptr;
        }
    }

    bool hasPreparedEvaluations() const { return m_prepared; }
    bool finished() const
    {
        return m_activeGames == 0 && !m_prepared && !m_pending.has_value();
    }
    size_t completedGames() const { return selfPlayLaneSize - m_activeGames; }

private:
    std::vector<ActiveGame<selfPlaySimulationCount>> m_games;
    std::vector<const Board*> m_pendingBoards;
    std::vector<Evaluation> m_evaluations;
    std::optional<PendingEvaluations> m_pending;
    size_t m_activeGames;
    bool m_prepared = false;
};

std::vector<TrainingSample> generateSelfPlay(const Network& network,
                                             uint64_t& nextGameId)
{
    selfPlayStatus = SelfPlayStatus{};
    std::array<SelfPlayLane, 2> lanes{
        SelfPlayLane{nextGameId, seed ^ nextGameId},
        SelfPlayLane{nextGameId + selfPlayLaneSize,
                     seed ^ (nextGameId + selfPlayLaneSize)},
    };
    nextGameId += selfPlayGameCount;

    std::vector<TrainingSample> samples;
    const auto prepare = [&](SelfPlayLane& lane) {
        std::vector<TrainingSample> completed = lane.prepare();
        if (!completed.empty())
        {
            reportSelfPlayStatus(completed.size());
            samples.insert(samples.end(),
                           std::make_move_iterator(completed.begin()),
                           std::make_move_iterator(completed.end()));
        }
    };

    int submitted = 0;
    prepare(lanes[submitted]);
    assert(lanes[submitted].hasPreparedEvaluations());
    lanes[submitted].submit(network);
    for (;;)
    {
        const int preparing = 1 - submitted;
        if (!lanes[preparing].finished()
            && !lanes[preparing].hasPreparedEvaluations())
            prepare(lanes[preparing]);

        // finish() is the synchronization point. Everything in prepare() above
        // ran while this lane's MLX batch was executing on the GPU.
        lanes[submitted].absorb(network);
        if (stopRequested)
            break;

        if (lanes[preparing].hasPreparedEvaluations())
        {
            lanes[preparing].submit(network);
            submitted = preparing;
            continue;
        }

        if (!lanes[submitted].finished())
        {
            prepare(lanes[submitted]);
            if (lanes[submitted].hasPreparedEvaluations())
            {
                lanes[submitted].submit(network);
                continue;
            }
        }
        break;
    }

    finishSelfPlayStatus();
    if (!stopRequested)
        assert(lanes[0].completedGames() + lanes[1].completedGames()
               == selfPlayGameCount);
    return samples;
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

void saveAtomically(const Network& network, const std::filesystem::path& path)
{
    const std::filesystem::path temporary = temporaryPath(path);
    std::filesystem::remove(temporary);
    network.save(temporary);
    std::filesystem::rename(temporary, path);
}

struct MatchScore
{
    int wins = 0;
    int draws = 0;
    int losses = 0;
    std::vector<double> observations;

    void add(Outcome outcome, bool candidateWhite)
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
    }
    int games() const { return wins + draws + losses; }
    double score() const { return (wins + 0.5 * draws) / games(); }
    double lowerConfidenceBound() const;
};

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

    void add(size_t gameIndex, Outcome outcome, bool candidateWhite)
    {
        assert(gameIndex / 2 < pairScores.size());
        games.add(outcome, candidateWhite);
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
    std::vector<const Board*> pendingBoards;
    std::vector<Evaluation> evaluations;
    while (completed < GameCount)
    {
        pendingBoards.clear();
        for (BaselineGame& game : games)
        {
            if (game.outcome.has_value() && started >= GameCount)
                continue;
            advance(game, baseline);
            if (game.outcome.has_value())
            {
                score.add(*game.outcome, game.candidateWhite);
                ++completed;
                if (started >= GameCount)
                    continue;
                initialize(game, firstGameId + started++, randomSeed);
                advance(game, baseline);
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
        const std::unique_ptr<Network> reference = createNetwork(network.name(), seed);
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
            network = createNetwork(networkIdentifier, seed);
        else
            throw std::runtime_error(std::format(
                "{} does not exist; select a network with --network",
                weights.string()));
        ParameterHealthMonitor healthMonitor{*network};
        healthMonitor.check(network->parameters(), 0);
        if (importLegacyLive)
        {
            // The previous method's .live file contains its newest network plus
            // Adam state. Import only the network once: saving the champion now
            // makes weights newer than the untouched legacy sidecar, so a later
            // restart cannot roll a promoted champion back to stale live data.
            saveAtomically(*network, weights);
            report("[train] imported current network from {} and reset optimizer state",
                   legacyLiveWeights.string());
        }
        else if (!std::filesystem::exists(weights))
            saveAtomically(*network, weights);
        report("[train] {}: {}, {} parameters", weights.string(),
               network->name(), network->parameterCount());
        report("[train] generation method: {} self-play games, {}+{} lanes, "
               "{} visits, AdamW lr {}, up to {} epochs",
               selfPlayGameCount, selfPlayLaneSize, selfPlayLaneSize,
               selfPlaySimulationCount, learningRate, maximumTrainingEpochs);

        std::mt19937_64 randomEngine{seed};
        uint64_t nextGameId = 0;
        uint64_t optimizerStep = 0;
        uint64_t generation = 1;
        bool firstGate = true;

        while (!stopRequested)
        {
            report("[generation {}] self-play started from the champion", generation);
            std::vector<TrainingSample> generated =
                generateSelfPlay(*network, nextGameId);
            if (stopRequested)
                break;

            std::vector<TrainingSample> training;
            std::vector<TrainingSample> heldOut;
            training.reserve(generated.size());
            heldOut.reserve(generated.size() / heldOutGameFraction + 1);
            for (TrainingSample& sample : generated)
            {
                std::vector<TrainingSample>& destination =
                    isHeldOutGame(sample.gameId) ? heldOut : training;
                destination.push_back(std::move(sample));
            }
            if (training.empty() || heldOut.empty())
                throw std::runtime_error(
                    "self-play generation did not produce both training and held-out positions");
            report("[generation {}] {} positions: {} train, {} held out",
                   generation, generated.size(), training.size(), heldOut.size());

            std::vector<mlx::core::array> parameters = network->parameters();
            Adam adam{parameters};
            const std::vector<uint8_t> decayMask = network->weightDecayMask();
            std::vector<size_t> trainingOrder(training.size());
            std::iota(trainingOrder.begin(), trainingOrder.end(), size_t{0});
            std::vector<size_t> heldOutOrder(heldOut.size());
            std::iota(heldOutOrder.begin(), heldOutOrder.end(), size_t{0});
            std::vector<mlx::core::array> bestParameters = parameters;
            double bestValidation = std::numeric_limits<double>::infinity();
            int bestEpoch = 0;
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

            for (int epoch = 1; epoch <= maximumTrainingEpochs && !stopRequested;
                 ++epoch)
            {
                std::ranges::shuffle(trainingOrder, randomEngine);
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
                        continue;
                    }
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
                    if (stopRequested)
                        break;
                }
                if (stopRequested)
                    break;
                if (trainedPositions == 0)
                    throw std::runtime_error("an epoch contained no finite training batch");

                healthMonitor.check(parameters, optimizerStep);
                const std::array<double, 2> heldOutLoss = validationLoss(parameters);
                const double validation = heldOutLoss[0] + heldOutLoss[1];
                report("[train] generation {}, epoch {}/{}, step {}: policy {:.4f}, "
                       "value {:.4f}, held-out {:.4f}+{:.4f}, |g| {:.2f}",
                       generation, epoch, maximumTrainingEpochs, optimizerStep,
                       trainPolicy / trainedPositions, trainValue / trainedPositions,
                       heldOutLoss[0], heldOutLoss[1],
                       gradientNormTotal / completedBatches);
                if (validation + validationImprovement < bestValidation)
                {
                    bestValidation = validation;
                    bestParameters = parameters;
                    bestEpoch = epoch;
                    staleEpochs = 0;
                }
                else if (++staleEpochs >= earlyStoppingPatience)
                {
                    report("[train] generation {} stopped after epoch {}; "
                           "held-out loss last improved at epoch {}",
                           generation, epoch, bestEpoch);
                    break;
                }
            }
            if (stopRequested)
                break;
            if (bestEpoch == 0)
                throw std::runtime_error("training did not produce a candidate epoch");
            parameters = std::move(bestParameters);
            healthMonitor.check(parameters, optimizerStep);
            network->replaceParameters(parameters);
            report("[train] generation {} candidate uses epoch {}", generation, bestEpoch);

            const uint64_t gateBase = generation * 1'000'000;
            bool baselinesPassed = true;
            if (firstGate)
            {
                MatchScore randomScore;
                if (!playBaselineGames<randomGameCount>(
                        *network, Baseline::random, gateBase + championGameCount,
                        seed + 8888 + generation, randomScore))
                    break;
                report("[gate] generation {} vs random: {:.1f}% "
                       "(95% lower {:.1f}%) over {} games",
                       generation, 100.0 * randomScore.score(),
                       100.0 * randomScore.lowerConfidenceBound(), randomScore.games());

                MatchScore rolloutScore;
                if (!playBaselineGames<rolloutGameCount>(
                        *network, Baseline::rollout,
                        gateBase + championGameCount + randomGameCount,
                        seed + 9999 + generation, rolloutScore))
                    break;
                report("[gate] generation {} vs rollout-mcts: {:.1f}% "
                       "(95% lower {:.1f}%) over {} games",
                       generation, 100.0 * rolloutScore.score(),
                       100.0 * rolloutScore.lowerConfidenceBound(), rolloutScore.games());
                baselinesPassed = randomScore.lowerConfidenceBound() > 0.5
                    && rolloutScore.lowerConfidenceBound() > 0.5;
            }
            if (stopRequested)
                break;

            std::unique_ptr<Network> champion = loadNetwork(weights, network->name());
            PairedMatchScore championScore(championGameCount / 2);
            const Network* gateNetworks[]{network.get(), champion.get()};
            NetworkMatchRunner<evaluationSimulationCount> championRunner{gateNetworks};
            if (!championRunner.template play<championGameCount>(
                    gateBase, seed + 7777 + generation,
                    [&](uint64_t gameId, Outcome outcome) {
                        const size_t gameIndex = static_cast<size_t>(gameId - gateBase);
                        championScore.add(gameIndex, outcome, gameIndex % 2 == 0);
                    }))
                break;
            const double championLowerBound =
                championScore.lowerConfidenceBound();
            report("[gate] generation {} vs champion: {:.1f}% "
                   "(paired 95% lower {:.1f}%) over {} games",
                   generation, 100.0 * championScore.score(),
                   100.0 * championLowerBound, championScore.gameCount());

            const bool promoted = baselinesPassed && championLowerBound > 0.5;
            report("[gate] generation {}: {}", generation,
                   promoted ? "PROMOTED" : "rejected");
            firstGate = false;
            if (promoted)
                saveAtomically(*network, weights);
            else
                network = std::move(champion);
            ++generation;
        }

        finishSelfPlayStatus();
        report("[train] shutdown complete; champion remains {}", weights.string());
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        finishSelfPlayStatus();
        std::println(stderr, "[train] {}", error.what());
        return EXIT_FAILURE;
    }
}

} // namespace amoeba_bot
