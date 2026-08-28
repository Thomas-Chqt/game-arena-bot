#include "training.hpp"

#include "mcts.hpp"
#include "network.hpp"

#include <limits.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <charconv>
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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amoeba_bot
{

namespace
{

constexpr uint64_t seed = 20260819;
constexpr int selfPlaySimulationCount = 200;
constexpr int evaluationSimulationCount = 200;
constexpr int selfPlayLaneSize = 256;
constexpr int championGameCount = 200;
constexpr int randomGameCount = 100;
constexpr int rolloutGameCount = 100;
constexpr int concurrentEvaluationGames = 64;
constexpr int samplingPlyCount = 20;
constexpr float rootNoise = 0.25f;
constexpr float noiseAlpha = 0.35f;
constexpr size_t replayBufferCapacity = 300'000;
constexpr size_t minimumReplaySize = 2'048;
constexpr size_t trainingBatchSize = 256;
constexpr double targetReuse = 4.0;
constexpr float learningRate = 1e-3f;
constexpr float weightDecay = 1e-4f;
constexpr uint64_t selfPlayRefreshSteps = 50;
constexpr uint64_t evaluationSteps = 1'000;
constexpr float promotionThreshold = 0.55f;
constexpr int workerSocket = 3;
constexpr char workerRoleEnvironment[] = "AMOEBA_TRAINING_WORKER";
constexpr char workerSocketEnvironment[] = "AMOEBA_TRAINING_SOCKET";

volatile sig_atomic_t stopRequested = 0;

struct SelfPlayStatus
{
    uint64_t rounds = 0;
    uint64_t totalRounds = 0;
    double roundsPerSecond = 0.0;
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
    std::print("\r\x1b[2K[selfplay] received {} rounds, {:.1f} rounds/s average",
               selfPlayStatus.rounds, selfPlayStatus.roundsPerSecond);
    std::fflush(stdout);
}

template<typename... Args>
void report(std::format_string<Args...> format, Args&&... args)
{
    if (selfPlayStatus.visible && isatty(STDOUT_FILENO))
        std::println();
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
    if (selfPlayStatus.visible)
    {
        selfPlayStatus.rounds = 0;
        drawSelfPlayStatus();
    }
}

void reportSelfPlayStatus(uint64_t receivedRounds)
{
    selfPlayStatus.rounds += receivedRounds;
    selfPlayStatus.totalRounds += receivedRounds;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selfPlayStatus.startedAt).count();
    selfPlayStatus.roundsPerSecond = selfPlayStatus.totalRounds / elapsed;
    selfPlayStatus.visible = true;
    if (!isatty(STDOUT_FILENO))
    {
        std::println("[selfplay] received {} rounds, {:.1f} rounds/s average",
                     selfPlayStatus.rounds, selfPlayStatus.roundsPerSecond);
        std::fflush(stdout);
        return;
    }
    drawSelfPlayStatus();
}

void finishSelfPlayStatus()
{
    if (!selfPlayStatus.visible)
        return;
    if (!isatty(STDOUT_FILENO))
    {
        selfPlayStatus.visible = false;
        return;
    }
    clearTerminalLine();
    selfPlayStatus.visible = false;
    std::println("[selfplay] received {} rounds, {:.1f} rounds/s average",
                 selfPlayStatus.rounds, selfPlayStatus.roundsPerSecond);
    std::fflush(stdout);
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

static_assert(std::is_trivially_copyable_v<TrainingSample>);

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
    game.board = Board::startingBoard();
    game.outcome.reset();
    game.positionHistory.assign(1, game.board.positionHash);
    game.samples.clear();
    game.moveRandom.seed(randomSeed ^ (0x9e3779b97f4a7c15ULL * (gameId + 1)));
    game.noiseRandom.seed(randomSeed + gameId);
    game.pairing = pairing;
    game.gameId = gameId;
    game.active = true;
    startSearch(game);
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

bool stopCommandReceived(int socket);

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
    bool play(uint64_t firstGameId, uint64_t randomSeed, int socket,
              GameFinished&& gameFinished)
    {
        static_assert(GameCount > 0);
        const int slotCount = std::min(GameCount, concurrentEvaluationGames);
        std::vector<ActiveGame<SimulationCount>> games(static_cast<size_t>(slotCount));
        int started = 0;
        int completed = 0;
        for (ActiveGame<SimulationCount>& game : games)
        {
            initialize(game, firstGameId + started,
                       started % 2 == 0 ? NetworkPairing{0, 1}
                                        : NetworkPairing{1, 0}, randomSeed);
            ++started;
        }
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
                        initialize(game, firstGameId + started,
                                   started % 2 == 0 ? NetworkPairing{0, 1}
                                                    : NetworkPairing{1, 0}, randomSeed);
                        ++started;
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
            if (stopCommandReceived(socket))
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
    SelfPlayLane(uint64_t& nextGameId, uint64_t randomSeed)
        : m_nextGameId(nextGameId), m_randomSeed(randomSeed), m_games(selfPlayLaneSize)
    {
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
            initializeNext(game);
    }

    std::vector<TrainingSample> prepare()
    {
        assert(!m_pending.has_value());
        m_pendingBoards.clear();
        std::vector<TrainingSample> completed;
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            advance<selfPlaySimulationCount, true>(game);
            if (game.pendingLeaf == nullptr)
            {
                finish(game);
                completed.insert(completed.end(),
                                 std::make_move_iterator(game.samples.begin()),
                                 std::make_move_iterator(game.samples.end()));
                initializeNext(game);
                advance<selfPlaySimulationCount, true>(game);
            }
            game.evaluationOffset = m_pendingBoards.size();
            m_pendingBoards.push_back(game.pendingLeaf);
        }
        assert(m_pendingBoards.size() == selfPlayLaneSize);
        return completed;
    }

    void submit(const Network& network)
    {
        assert(!m_pendingBoards.empty());
        assert(!m_pending.has_value());
        m_evaluations.resize(m_pendingBoards.size());
        m_pending.emplace(network.submit(m_pendingBoards));
    }

    void absorb(const Network& network)
    {
        assert(m_pending.has_value());
        network.finish(std::move(*m_pending), m_evaluations);
        m_pending.reset();
        for (ActiveGame<selfPlaySimulationCount>& game : m_games)
        {
            Evaluation& evaluation = m_evaluations[game.evaluationOffset];
            if (game.firstLeafOfSearch)
                addExplorationNoise(evaluation, *game.pendingLeaf, game.noiseRandom);
            game.search->absorb(evaluation.policy, evaluation.value);
            game.firstLeafOfSearch = false;
            game.pendingLeaf = nullptr;
        }
    }

private:
    void initializeNext(ActiveGame<selfPlaySimulationCount>& game)
    {
        initialize(game, m_nextGameId++, NetworkPairing{0, 0}, m_randomSeed);
    }
    uint64_t& m_nextGameId;
    uint64_t m_randomSeed;
    std::vector<ActiveGame<selfPlaySimulationCount>> m_games;
    std::vector<const Board*> m_pendingBoards;
    std::vector<Evaluation> m_evaluations;
    std::optional<PendingEvaluations> m_pending;
};

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
                          std::span<const size_t, trainingBatchSize> picks,
                          std::mt19937_64& randomEngine)
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
    std::vector<mlx::core::array> inputs;
    inputs.reserve(trainingBatchSize);
    std::vector<float> legal(trainingBatchSize * moveIdCount);
    std::vector<float> policy(trainingBatchSize * moveIdCount);
    std::array<float, trainingBatchSize> outcomes{};
    for (size_t batchIndex = 0; batchIndex < trainingBatchSize; ++batchIndex)
    {
        const uint8_t symmetry =
            std::uniform_int_distribution<uint8_t>{0, 11}(randomEngine);
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
        mlx::core::array(legal.data(), {static_cast<int>(trainingBatchSize), moveIdCount}, mlx::core::float32),
        mlx::core::array(policy.data(), {static_cast<int>(trainingBatchSize), moveIdCount}, mlx::core::float32),
        mlx::core::array(outcomes.data(), {static_cast<int>(trainingBatchSize)}, mlx::core::float32),
    };
}

enum class MessageType : uint32_t
{
    start,
    samples,
    liveWeights,
    liveWeightsApplied,
    evaluate,
    evaluationResult,
    stop,
};

struct Message
{
    MessageType type{};
    uint32_t count{};
    uint64_t version{};
    std::array<uint32_t, 3> gameCounts{};
    std::array<double, 3> scores{};
    uint32_t promoted{};
};

bool readAll(int socket, void* output, size_t size)
{
    char* destination = static_cast<char*>(output);
    while (size > 0)
    {
        const ssize_t received = recv(socket, destination, size, 0);
        if (received == 0)
            return false;
        if (received < 0)
        {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::format("recv: {}", std::strerror(errno)));
        }
        destination += received;
        size -= static_cast<size_t>(received);
    }
    return true;
}

void writeAll(int socket, const void* input, size_t size)
{
    const char* source = static_cast<const char*>(input);
    while (size > 0)
    {
        const ssize_t written = send(socket, source, size, MSG_NOSIGNAL);
        if (written < 0)
        {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::format("send: {}", std::strerror(errno)));
        }
        source += written;
        size -= static_cast<size_t>(written);
    }
}

void sendMessage(int socket, const Message& message) { writeAll(socket, &message, sizeof(message)); }
bool receiveMessage(int socket, Message& message) { return readAll(socket, &message, sizeof(message)); }

void sendSamples(int socket, const std::vector<TrainingSample>& samples)
{
    if (samples.empty())
        return;
    assert(samples.size() <= std::numeric_limits<uint32_t>::max());
    sendMessage(socket, Message{.type = MessageType::samples,
                                .count = static_cast<uint32_t>(samples.size())});
    writeAll(socket, samples.data(), samples.size() * sizeof(TrainingSample));
}

std::filesystem::path internalPath(const std::filesystem::path& weights,
                                   std::string_view suffix)
{
    std::filesystem::path result = weights;
    result += suffix;
    return result;
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

void saveTrainingAtomically(const Network& network, const Adam& adam,
                            const std::filesystem::path& path)
{
    const std::filesystem::path temporary = temporaryPath(path);
    std::filesystem::remove(temporary);
    network.save(temporary);
    auto [tensors, metadata] = mlx::core::load_safetensors(temporary.string());
    for (size_t index = 0; index < adam.mean().size(); ++index)
    {
        tensors.emplace(std::format("adam.mean.{}", index), adam.mean()[index]);
        tensors.emplace(std::format("adam.variance.{}", index), adam.variance()[index]);
    }
    metadata["adam_steps"] = std::to_string(adam.steps());
    std::filesystem::remove(temporary);
    mlx::core::save_safetensors(temporary.string(), tensors, metadata);
    std::filesystem::rename(temporary, path);
}

bool restoreAdam(const std::filesystem::path& path, Adam& adam)
{
    auto [tensors, metadata] = mlx::core::load_safetensors(path.string());
    const auto storedSteps = metadata.find("adam_steps");
    if (storedSteps == metadata.end())
        return false;
    int steps = 0;
    const char* begin = storedSteps->second.data();
    const char* end = begin + storedSteps->second.size();
    const auto [position, error] = std::from_chars(begin, end, steps);
    if (error != std::errc{} || position != end)
        throw std::runtime_error("Adam checkpoint has an invalid step count");
    std::vector<mlx::core::array> mean;
    std::vector<mlx::core::array> variance;
    mean.reserve(adam.mean().size());
    variance.reserve(adam.variance().size());
    for (size_t index = 0; index < adam.mean().size(); ++index)
    {
        const auto storedMean = tensors.find(std::format("adam.mean.{}", index));
        const auto storedVariance = tensors.find(std::format("adam.variance.{}", index));
        if (storedMean == tensors.end() || storedVariance == tensors.end())
            throw std::runtime_error(std::format("Adam checkpoint is missing tensor {}", index));
        mean.push_back(storedMean->second);
        variance.push_back(storedVariance->second);
    }
    adam.restore(std::move(mean), std::move(variance), steps);
    return true;
}

bool stopCommandReceived(int socket)
{
    if (stopRequested)
        return true;
    pollfd descriptor{.fd = socket, .events = POLLIN, .revents = 0};
    const int ready = poll(&descriptor, 1, 0);
    if (ready < 0)
    {
        if (errno == EINTR)
            return stopRequested;
        throw std::runtime_error(std::format("poll: {}", std::strerror(errno)));
    }
    if (ready == 0)
        return false;
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        return true;
    Message message;
    if (!receiveMessage(socket, message))
        return true;
    if (message.type != MessageType::stop)
        throw std::runtime_error("worker received an invalid command while busy");
    return true;
}

bool refreshSelfPlayNetwork(int socket, const std::filesystem::path& liveWeights,
                            std::unique_ptr<Network>& network)
{
    if (stopRequested)
        return false;
    std::optional<uint64_t> newestVersion;
    for (;;)
    {
        pollfd descriptor{.fd = socket, .events = POLLIN, .revents = 0};
        const int ready = poll(&descriptor, 1, 0);
        if (ready < 0)
        {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::format("poll: {}", std::strerror(errno)));
        }
        if (ready == 0)
            break;
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
            return false;
        Message message;
        if (!receiveMessage(socket, message) || message.type == MessageType::stop)
            return false;
        if (message.type != MessageType::liveWeights)
            throw std::runtime_error("self-play worker received an invalid command");
        newestVersion = message.version;
    }
    if (newestVersion.has_value())
    {
        network = loadNetwork(liveWeights);
        sendMessage(socket, Message{
            .type = MessageType::liveWeightsApplied,
            .version = *newestVersion,
        });
    }
    return true;
}

struct MatchScore
{
    int wins = 0;
    int draws = 0;
    int losses = 0;
    void add(Outcome outcome, bool candidateWhite)
    {
        if (outcome == Outcome::draw)
            ++draws;
        else if ((outcome == Outcome::whiteWins) == candidateWhite)
            ++wins;
        else
            ++losses;
    }
    int games() const { return wins + draws + losses; }
    double score() const { return (wins + 0.5 * draws) / games(); }
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
                       MatchScore& score, int socket)
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
        if (stopCommandReceived(socket))
            return false;
    }
    return true;
}

int runSelfPlayWorker(const std::filesystem::path& weights, int socket)
try
{
    Message start;
    if (!receiveMessage(socket, start) || start.type == MessageType::stop)
        return EXIT_SUCCESS;
    if (start.type != MessageType::start)
        throw std::runtime_error("self-play worker did not receive its start command");
    const std::filesystem::path liveWeights = internalPath(weights, ".live.safetensors");
    std::unique_ptr<Network> network = loadNetwork(liveWeights);
    uint64_t nextGameId = 0;
    std::array<SelfPlayLane, 2> lanes{
        SelfPlayLane{nextGameId, seed},
        SelfPlayLane{nextGameId, seed + selfPlayLaneSize},
    };
    int submitted = 0;
    int preparing = 1;
    sendSamples(socket, lanes[submitted].prepare());
    lanes[submitted].submit(*network);
    for (;;)
    {
        sendSamples(socket, lanes[preparing].prepare());
        lanes[submitted].absorb(*network);
        if (!refreshSelfPlayNetwork(socket, liveWeights, network))
            return EXIT_SUCCESS;
        std::swap(submitted, preparing);
        lanes[submitted].submit(*network);
    }
}
catch (const std::exception& error)
{
    clearTerminalLine();
    std::println(stderr, "[selfplay] {}", error.what());
    return EXIT_FAILURE;
}

int runEvaluationWorker(const std::filesystem::path& weights, int socket)
try
{
    Message start;
    if (!receiveMessage(socket, start) || start.type == MessageType::stop)
        return EXIT_SUCCESS;
    if (start.type != MessageType::start)
        throw std::runtime_error("evaluation worker did not receive its start command");
    const std::filesystem::path candidateWeights =
        internalPath(weights, ".candidate.safetensors");
    for (;;)
    {
        Message command;
        if (!receiveMessage(socket, command) || command.type == MessageType::stop)
            return EXIT_SUCCESS;
        if (command.type != MessageType::evaluate)
            throw std::runtime_error("evaluation worker received an invalid command");
        std::unique_ptr<Network> candidate = loadNetwork(candidateWeights);
        std::unique_ptr<Network> champion = loadNetwork(weights);
        std::array<MatchScore, 3> scores;
        const Network* networks[]{candidate.get(), champion.get()};
        NetworkMatchRunner<evaluationSimulationCount> championRunner{networks};
        const bool championComplete = championRunner.template play<championGameCount>(
            command.version * 1'000, seed + 7777 + command.version, socket,
            [&](uint64_t gameId, Outcome outcome) {
                scores[0].add(outcome, gameId % 2 == 0);
            });
        if (!championComplete)
            return EXIT_SUCCESS;
        if (!playBaselineGames<randomGameCount>(
                *candidate, Baseline::random, command.version * 1'000 + 200,
                seed + 8888 + command.version, scores[1], socket))
            return EXIT_SUCCESS;
        if (!playBaselineGames<rolloutGameCount>(
                *candidate, Baseline::rollout, command.version * 1'000 + 300,
                seed + 9999 + command.version, scores[2], socket))
            return EXIT_SUCCESS;
        const bool promoted = std::ranges::all_of(scores, [](const MatchScore& score) {
            return score.score() >= promotionThreshold;
        });
        if (promoted)
            saveAtomically(*candidate, weights);
        sendMessage(socket, Message{
            .type = MessageType::evaluationResult,
            .version = command.version,
            .gameCounts = {static_cast<uint32_t>(scores[0].games()),
                           static_cast<uint32_t>(scores[1].games()),
                           static_cast<uint32_t>(scores[2].games())},
            .scores = {scores[0].score(), scores[1].score(), scores[2].score()},
            .promoted = promoted,
        });
    }
}
catch (const std::exception& error)
{
    clearTerminalLine();
    std::println(stderr, "[evaluation] {}", error.what());
    return EXIT_FAILURE;
}

struct Worker { pid_t process = -1; int socket = -1; };
struct Workers { Worker selfPlay; Worker evaluation; };

void stopWorker(Worker& worker)
{
    bool sent = false;
    if (worker.socket >= 0)
    {
        const Message stop{.type = MessageType::stop};
        sent = send(worker.socket, &stop, sizeof(stop), MSG_NOSIGNAL)
            == static_cast<ssize_t>(sizeof(stop));
        close(worker.socket);
        worker.socket = -1;
    }
    if (worker.process > 0)
    {
        if (!sent)
            kill(worker.process, SIGTERM);
        while (waitpid(worker.process, nullptr, 0) < 0 && errno == EINTR) {}
        worker.process = -1;
    }
}

void stopWorkers(Workers& workers)
{
    stopWorker(workers.selfPlay);
    stopWorker(workers.evaluation);
}

Workers forkWorkers(const std::filesystem::path& weights)
{
    int selfPlayChannels[2];
    int evaluationChannels[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, selfPlayChannels) != 0)
        throw std::runtime_error(std::format("socketpair: {}", std::strerror(errno)));
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, evaluationChannels) != 0)
    {
        close(selfPlayChannels[0]); close(selfPlayChannels[1]);
        throw std::runtime_error(std::format("socketpair: {}", std::strerror(errno)));
    }
    std::array<char, PATH_MAX> executableBuffer{};
    uint32_t executableSize = executableBuffer.size();
    if (_NSGetExecutablePath(executableBuffer.data(), &executableSize) != 0)
    {
        close(selfPlayChannels[0]); close(selfPlayChannels[1]);
        close(evaluationChannels[0]); close(evaluationChannels[1]);
        throw std::runtime_error("executable path is too long");
    }

    std::string executable{executableBuffer.data()};
    std::string weightsArgument = weights.string();
    std::string socketArgument = std::to_string(workerSocket);
    char trainArgument[] = "--train";
    std::array<char*, 4> arguments{
        executable.data(), trainArgument, weightsArgument.data(), nullptr};
    const std::array<int, 4> channels{
        selfPlayChannels[0], selfPlayChannels[1],
        evaluationChannels[0], evaluationChannels[1]};

    const auto startWorker = [&](std::string_view role, int socket) {
        const std::string roleArgument{role};
        if (setenv(workerRoleEnvironment, roleArgument.c_str(), 1) != 0)
            throw std::runtime_error(std::format("setenv: {}", std::strerror(errno)));
        if (setenv(workerSocketEnvironment, socketArgument.c_str(), 1) != 0)
        {
            const int environmentError = errno;
            unsetenv(workerRoleEnvironment);
            errno = environmentError;
            throw std::runtime_error(std::format("setenv: {}", std::strerror(errno)));
        }

        const pid_t process = fork();
        const int forkError = errno;
        if (process == 0)
        {
            if (socket != workerSocket && dup2(socket, workerSocket) < 0)
                _exit(EXIT_FAILURE);
            for (int channel : channels)
                if (channel != workerSocket)
                    close(channel);
            execv(executable.c_str(), arguments.data());
            constexpr char message[] = "[worker] execv failed\n";
            write(STDERR_FILENO, message, sizeof(message) - 1);
            _exit(EXIT_FAILURE);
        }

        unsetenv(workerRoleEnvironment);
        unsetenv(workerSocketEnvironment);
        if (process < 0)
        {
            errno = forkError;
            throw std::runtime_error(std::format("fork: {}", std::strerror(errno)));
        }
        return process;
    };

    pid_t selfPlayProcess = -1;
    pid_t evaluationProcess = -1;
    try
    {
        selfPlayProcess = startWorker("selfplay", selfPlayChannels[1]);
        evaluationProcess = startWorker("evaluation", evaluationChannels[1]);
    }
    catch (...)
    {
        close(selfPlayChannels[0]); close(selfPlayChannels[1]);
        close(evaluationChannels[0]); close(evaluationChannels[1]);
        if (selfPlayProcess > 0)
        {
            kill(selfPlayProcess, SIGTERM);
            waitpid(selfPlayProcess, nullptr, 0);
        }
        throw;
    }
    close(selfPlayChannels[1]);
    close(evaluationChannels[1]);
    return {.selfPlay = {selfPlayProcess, selfPlayChannels[0]},
            .evaluation = {evaluationProcess, evaluationChannels[0]}};
}

} // namespace

int runTraining(const std::filesystem::path& weights,
                std::string_view networkIdentifier)
{
    if (const char* role = std::getenv(workerRoleEnvironment))
    {
        const char* socketText = std::getenv(workerSocketEnvironment);
        int socket = -1;
        if (socketText == nullptr)
        {
            std::println(stderr, "[worker] missing socket environment variable");
            return EXIT_FAILURE;
        }
        const std::string_view text{socketText};
        const auto [end, error] = std::from_chars(
            text.data(), text.data() + text.size(), socket);
        if (error != std::errc{} || end != text.data() + text.size() || socket < 0)
        {
            std::println(stderr, "[worker] invalid socket environment variable");
            return EXIT_FAILURE;
        }
        if (std::string_view{role} == "selfplay")
            return runSelfPlayWorker(weights, socket);
        if (std::string_view{role} == "evaluation")
            return runEvaluationWorker(weights, socket);
        std::println(stderr, "[worker] invalid role {}", role);
        return EXIT_FAILURE;
    }

    stopRequested = 0;
    Workers workers;
    const std::filesystem::path liveWeights = internalPath(weights, ".live.safetensors");
    const std::filesystem::path candidateWeights = internalPath(weights, ".candidate.safetensors");
    try
    {
        installSignalHandlers();
        workers = forkWorkers(weights);
        const bool resumed = std::filesystem::exists(liveWeights);
        std::unique_ptr<Network> network;
        if (std::filesystem::exists(weights))
        {
            network = loadNetwork(weights, networkIdentifier);
            if (resumed)
                network = loadNetwork(liveWeights, network->name());
        }
        else if (resumed)
            network = loadNetwork(liveWeights, networkIdentifier);
        else if (!networkIdentifier.empty())
            network = createNetwork(networkIdentifier, seed);
        else
            throw std::runtime_error(std::format(
                "{} does not exist; select a network with --network",
                weights.string()));
        if (!std::filesystem::exists(weights))
            saveAtomically(*network, weights);
        std::vector<mlx::core::array> parameters = network->parameters();
        Adam adam{parameters};
        const bool restoredAdam = resumed && restoreAdam(liveWeights, adam);
        saveTrainingAtomically(*network, adam, liveWeights);
        report("[train] {}: {}, {} parameters", weights.string(),
               network->name(), network->parameterCount());
        if (restoredAdam)
            report("[train] restored Adam at optimizer step {}", adam.steps());
        sendMessage(workers.selfPlay.socket, Message{.type = MessageType::start});
        sendMessage(workers.evaluation.socket, Message{.type = MessageType::start});

        std::vector<TrainingSample> replay;
        replay.reserve(replayBufferCapacity);
        size_t replayWrite = 0;
        double credits = 0.0;
        std::mt19937_64 randomEngine{seed};
        uint64_t step = 0;
        bool evaluationBusy = false;
        bool evaluationPending = false;
        const auto publishCandidate = [&] {
            network->replaceParameters(parameters);
            saveAtomically(*network, candidateWeights);
            report("[evaluation] step {}: started", step);
            sendMessage(workers.evaluation.socket,
                        Message{.type = MessageType::evaluate, .version = step});
            evaluationBusy = true;
            evaluationPending = false;
        };

        while (!stopRequested)
        {
            const bool canTrain = replay.size() >= minimumReplaySize
                                  && credits >= trainingBatchSize;
            pollfd descriptors[]{
                {.fd = workers.selfPlay.socket, .events = POLLIN, .revents = 0},
                {.fd = workers.evaluation.socket, .events = POLLIN, .revents = 0},
            };
            const int ready = poll(descriptors, 2, canTrain ? 0 : -1);
            if (ready < 0 && errno != EINTR)
                throw std::runtime_error(std::format("poll: {}", std::strerror(errno)));
            if (stopRequested)
                break;
            if ((descriptors[0].revents & POLLIN) != 0)
            {
                Message message;
                if (!receiveMessage(workers.selfPlay.socket, message))
                    throw std::runtime_error("self-play worker stopped");
                if (message.type == MessageType::liveWeightsApplied)
                {
                    report("[selfplay] network replaced at step {}", message.version);
                    continue;
                }
                if (message.type != MessageType::samples)
                    throw std::runtime_error("trainer received an invalid self-play message");
                std::vector<TrainingSample> fresh(message.count);
                if (!readAll(workers.selfPlay.socket, fresh.data(),
                             fresh.size() * sizeof(TrainingSample)))
                    throw std::runtime_error("self-play worker stopped inside a sample message");
                for (TrainingSample& sample : fresh)
                {
                    if (replay.size() < replayBufferCapacity)
                        replay.push_back(std::move(sample));
                    else
                    {
                        replay[replayWrite] = std::move(sample);
                        replayWrite = (replayWrite + 1) % replayBufferCapacity;
                    }
                }
                credits += fresh.size() * targetReuse;
                reportSelfPlayStatus(fresh.size());
            }
            if ((descriptors[1].revents & POLLIN) != 0)
            {
                Message result;
                if (!receiveMessage(workers.evaluation.socket, result))
                    throw std::runtime_error("evaluation worker stopped");
                if (result.type != MessageType::evaluationResult)
                    throw std::runtime_error("trainer received an invalid evaluation message");
                constexpr std::array<std::string_view, 3> opponentNames{
                    "champion", "random", "rollout-mcts"};
                for (size_t index = 0; index < opponentNames.size(); ++index)
                    report("[evaluation] step {} vs {}: {:.1f}% over {} games",
                           result.version, opponentNames[index],
                           100.0 * result.scores[index], result.gameCounts[index]);
                report("[evaluation] step {}: {}", result.version,
                       result.promoted ? "PROMOTED" : "rejected");
                evaluationBusy = false;
                if (evaluationPending)
                    publishCandidate();
            }
            if ((descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
                throw std::runtime_error("self-play worker failed");
            if ((descriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
                throw std::runtime_error("evaluation worker failed");
            if (!canTrain)
                continue;

            std::array<size_t, trainingBatchSize> picks;
            for (size_t& pick : picks)
                pick = std::uniform_int_distribution<size_t>{0, replay.size() - 1}(randomEngine);
            const TrainingBatch batch = createBatch(replay, picks, randomEngine);
            const LossAndGrad result = network->valueAndGrad(parameters, batch, weightDecay);
            parameters = adam.updateParameters(parameters, result.gradients, learningRate);
            mlx::core::eval({result.loss.policy, result.loss.value});
            mlx::core::eval(parameters);
            credits -= trainingBatchSize;
            ++step;
            const float policyLoss = result.loss.policy.item<float>();
            const float valueLoss = result.loss.value.item<float>();
            if (!std::isfinite(policyLoss) || !std::isfinite(valueLoss))
                throw std::runtime_error("training produced a non-finite loss");
            if (step % 10 == 0)
            {
                report("[train] step {}, policy {:.4f}, value {:.4f}, credits {:.0f}",
                       step, policyLoss, valueLoss, credits);
            }
            if (step % selfPlayRefreshSteps == 0)
            {
                network->replaceParameters(parameters);
                saveTrainingAtomically(*network, adam, liveWeights);
                sendMessage(workers.selfPlay.socket,
                            Message{.type = MessageType::liveWeights, .version = step});
            }
            if (step % evaluationSteps == 0)
            {
                if (evaluationBusy)
                    evaluationPending = true;
                else
                    publishCandidate();
            }
        }

        finishSelfPlayStatus();
        network->replaceParameters(parameters);
        report("[train] stopping, saving {}", liveWeights.string());
        saveTrainingAtomically(*network, adam, liveWeights);
        stopWorkers(workers);
        std::filesystem::remove(candidateWeights);
        report("[train] shutdown complete");
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        stopWorkers(workers);
        finishSelfPlayStatus();
        std::println(stderr, "[train] {}", error.what());
        return EXIT_FAILURE;
    }
}

} // namespace amoeba_bot
