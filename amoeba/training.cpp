#include "training.hpp"

#include "mcts.hpp"
#include "network.hpp"

#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
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
#include <utility>
#include <vector>

extern char** environ;

namespace amoeba_bot
{

namespace
{

constexpr uint64_t seed = 20260819;
constexpr int selfPlaySimulationCount = 200;
constexpr int evaluationSimulationCount = 200;
constexpr int gamesPerSelfPlayField = 128;
constexpr int concurrentSelfPlayGames = 128;
constexpr int evaluationGameCount = 200;
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
constexpr uint64_t evaluationSteps = 500;
constexpr float promotionThreshold = 0.55f;
constexpr int workerChannel = 3;

template<typename... Args>
void report(std::format_string<Args...> format, Args&&... args)
{
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
}

struct TrainingSample
{
    Board board;
    VisitCounts visits;
    float outcome;
    uint64_t gameId;
};

static_assert(std::is_trivially_copyable_v<TrainingSample>);

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
    uint64_t evaluatedPositions = 0;
    bool firstLeafOfSearch = true;
};

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

template<int SimulationCount, bool AddRootNoise>
class GameBatchRunner
{
public:
    explicit GameBatchRunner(std::span<const Network* const> networks)
        : m_networks(networks)
    {
        assert(!networks.empty() && networks.size() <= 2);
    }

    template<int GameCount, int ConcurrentGameCount, typename PairingFor,
             typename GameFinished>
    void play(uint64_t firstGameId, uint64_t randomSeed,
              PairingFor&& pairingFor, GameFinished&& gameFinished)
    {
        static_assert(GameCount > 0);
        static_assert(ConcurrentGameCount > 0);

        const int slotCount = std::min(GameCount, ConcurrentGameCount);
        std::vector<ActiveGame<SimulationCount>> games(static_cast<size_t>(slotCount));
        int started = 0;
        int completed = 0;
        bool keepPlaying = true;

        for (ActiveGame<SimulationCount>& game : games)
        {
            initialize(game, firstGameId + started, pairingFor(started), randomSeed);
            ++started;
        }

        std::array<std::vector<const Board*>, 2> pendingBoards;
        std::array<std::vector<Evaluation>, 2> evaluations;

        while (keepPlaying)
        {
            size_t pendingCount = 0;
            for (std::vector<const Board*>& boards : pendingBoards)
                boards.clear();

            for (ActiveGame<SimulationCount>& game : games)
            {
                if (!game.active)
                    continue;
                advance(game);
                if (game.pendingLeaf == nullptr)
                {
                    finish(game);
                    keepPlaying = gameFinished(game.gameId, std::move(game.samples));
                    ++completed;
                    if (!keepPlaying || started == GameCount)
                    {
                        game.active = false;
                        continue;
                    }
                    initialize(game, firstGameId + started, pairingFor(started), randomSeed);
                    ++started;
                    advance(game);
                }

                if (!game.active)
                    continue;
                std::vector<const Board*>& boards =
                    pendingBoards[static_cast<size_t>(game.networkIndex)];
                game.evaluationOffset = boards.size();
                boards.push_back(game.pendingLeaf);
                ++pendingCount;
            }

            if (!keepPlaying || pendingCount == 0)
                break;

            for (size_t networkIndex = 0; networkIndex < m_networks.size(); ++networkIndex)
            {
                if (pendingBoards[networkIndex].empty())
                    continue;
                evaluations[networkIndex].resize(pendingBoards[networkIndex].size());
                (*m_networks[networkIndex])(pendingBoards[networkIndex], evaluations[networkIndex]);
            }

            for (ActiveGame<SimulationCount>& game : games)
            {
                if (!game.active || game.pendingLeaf == nullptr)
                    continue;
                Evaluation& evaluation =
                    evaluations[static_cast<size_t>(game.networkIndex)][game.evaluationOffset];
                if constexpr (AddRootNoise)
                {
                    if (game.firstLeafOfSearch)
                        addExplorationNoise(evaluation, *game.pendingLeaf, game.noiseRandom);
                }
                game.search->absorb(evaluation.policy, evaluation.value);
                game.firstLeafOfSearch = false;
                game.pendingLeaf = nullptr;
                ++game.evaluatedPositions;
            }
        }

        report("[games] completed {} of {} games at {} simulations",
               completed, GameCount, SimulationCount);
    }

private:
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
        game.evaluatedPositions = 0;
        startSearch(game);
    }

    void startSearch(ActiveGame<SimulationCount>& game)
    {
        game.search.emplace(game.board, game.positionHistory);
        game.pendingLeaf = nullptr;
        game.firstLeafOfSearch = true;
    }

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
            const uint16_t moveId = selectMoveFromVisits(
                visits, game.board.plyCount, game.moveRandom);
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

    void finish(ActiveGame<SimulationCount>& game)
    {
        assert(game.outcome.has_value());
        for (TrainingSample& sample : game.samples)
            sample.outcome = outcomeFor(*game.outcome, sample.board.whiteToMove);
    }

    std::span<const Network* const> m_networks;
};

TrainingBatch createBatch(
    const std::vector<TrainingSample>& replay,
    std::span<const size_t, trainingBatchSize> picks)
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
        const TrainingSample& sample = replay[picks[batchIndex]];
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
        mlx::core::array(legal.data(),
                         {static_cast<int>(trainingBatchSize), moveIdCount},
                         mlx::core::float32),
        mlx::core::array(policy.data(),
                         {static_cast<int>(trainingBatchSize), moveIdCount},
                         mlx::core::float32),
        mlx::core::array(outcomes.data(),
                         {static_cast<int>(trainingBatchSize)},
                         mlx::core::float32),
    };
}

enum class MessageType : uint32_t
{
    samples,
    liveWeights,
    evaluate,
    evaluationResult,
    stop,
};

struct Message
{
    MessageType type{};
    uint32_t count{};
    uint64_t version{};
    double score{};
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

void sendMessage(int socket, const Message& message)
{
    writeAll(socket, &message, sizeof(message));
}

bool receiveMessage(int socket, Message& message)
{
    return readAll(socket, &message, sizeof(message));
}

void sendSamples(int socket, const std::vector<TrainingSample>& samples)
{
    assert(samples.size() <= std::numeric_limits<uint32_t>::max());
    sendMessage(socket, Message{
        .type = MessageType::samples,
        .count = static_cast<uint32_t>(samples.size()),
    });
    writeAll(socket, samples.data(), samples.size() * sizeof(TrainingSample));
}

std::filesystem::path internalPath(
    const std::filesystem::path& weights, std::string_view suffix)
{
    std::filesystem::path result = weights;
    result += suffix;
    return result;
}

void saveAtomically(const Network& network, const std::filesystem::path& path)
{
    const std::filesystem::path temporary =
        path.parent_path() / (path.stem().string() + ".tmp" + path.extension().string());
    std::filesystem::remove(temporary);
    network.save(temporary);
    std::filesystem::rename(temporary, path);
}

struct Worker
{
    pid_t process = -1;
    int socket = -1;
};

Worker spawnWorker(const std::filesystem::path& executable,
                   std::string_view role,
                   const std::filesystem::path& weights)
{
    int channels[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, channels) != 0)
        throw std::runtime_error(std::format("socketpair: {}", std::strerror(errno)));

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, channels[0]);
    if (channels[1] != workerChannel)
    {
        posix_spawn_file_actions_adddup2(&actions, channels[1], workerChannel);
        posix_spawn_file_actions_addclose(&actions, channels[1]);
    }

    std::string executableText = executable.string();
    std::string roleText{role};
    std::string weightsText = weights.string();
    char* arguments[]{executableText.data(), roleText.data(), weightsText.data(), nullptr};

    pid_t process = -1;
    const int status = posix_spawnp(
        &process, executableText.c_str(), &actions, nullptr, arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(channels[1]);
    if (status != 0)
    {
        close(channels[0]);
        throw std::runtime_error(std::format("posix_spawn: {}", std::strerror(status)));
    }
    return {process, channels[0]};
}

void stopWorker(Worker& worker)
{
    if (worker.socket >= 0)
    {
        close(worker.socket);
        worker.socket = -1;
    }
    if (worker.process > 0)
    {
        kill(worker.process, SIGTERM);
        waitpid(worker.process, nullptr, 0);
        worker.process = -1;
    }
}

volatile sig_atomic_t stopRequested = 0;

void requestStop(int)
{
    stopRequested = 1;
}

double standardError(double score, int games)
{
    return std::sqrt(std::max(score * (1.0 - score), 0.01) / games);
}

bool gateSettled(int played, double score)
{
    constexpr int minimumGameCount = 20;
    return played >= minimumGameCount
        && std::abs(score - promotionThreshold) > 3.0 * standardError(score, played);
}

} // namespace

int runSelfPlayWorker(const std::filesystem::path& weights)
try
{
    const std::filesystem::path liveWeights = internalPath(weights, ".live.safetensors");
    uint64_t firstGameId = 0;
    uint64_t field = 0;

    for (;;)
    {
        std::unique_ptr<Network> network = loadNetwork(liveWeights);
        const Network* networks[]{network.get()};
        GameBatchRunner<selfPlaySimulationCount, true> runner{networks};
        runner.template play<gamesPerSelfPlayField, concurrentSelfPlayGames>(
            firstGameId, seed + field,
            [](int) { return NetworkPairing{0, 0}; },
            [&](uint64_t, std::vector<TrainingSample>&& samples) {
                sendSamples(workerChannel, samples);
                return true;
            });
        firstGameId += gamesPerSelfPlayField;
        ++field;

        pollfd descriptor{.fd = workerChannel, .events = POLLIN, .revents = 0};
        while (poll(&descriptor, 1, 0) > 0 && (descriptor.revents & POLLIN) != 0)
        {
            Message message;
            if (!receiveMessage(workerChannel, message) || message.type == MessageType::stop)
                return EXIT_SUCCESS;
            descriptor.revents = 0;
        }
    }
}
catch (const std::exception& error)
{
    std::println(stderr, "[selfplay] {}", error.what());
    return EXIT_FAILURE;
}

int runEvaluationWorker(const std::filesystem::path& weights)
try
{
    const std::filesystem::path candidateWeights =
        internalPath(weights, ".candidate.safetensors");

    for (;;)
    {
        Message command;
        if (!receiveMessage(workerChannel, command) || command.type == MessageType::stop)
            return EXIT_SUCCESS;
        if (command.type != MessageType::evaluate)
            throw std::runtime_error("evaluation worker received an invalid command");

        std::unique_ptr<Network> candidate = loadNetwork(candidateWeights);
        std::unique_ptr<Network> champion = loadNetwork(weights);
        const Network* networks[]{candidate.get(), champion.get()};
        int wins = 0;
        int draws = 0;
        int losses = 0;

        GameBatchRunner<evaluationSimulationCount, false> runner{networks};
        runner.template play<evaluationGameCount, concurrentEvaluationGames>(
            command.version * evaluationGameCount,
            seed + 7777 + command.version,
            [](int game) {
                return game % 2 == 0 ? NetworkPairing{0, 1}
                                     : NetworkPairing{1, 0};
            },
            [&](uint64_t gameId, std::vector<TrainingSample>&& samples) {
                const TrainingSample& last = samples.back();
                if (last.outcome == 0.0f)
                    ++draws;
                else
                {
                    const bool whiteWon =
                        (last.outcome > 0.0f) == last.board.whiteToMove;
                    const bool candidateWasWhite = gameId % 2 == 0;
                    if (whiteWon == candidateWasWhite)
                        ++wins;
                    else
                        ++losses;
                }
                const int total = wins + draws + losses;
                const double score = (wins + 0.5 * draws) / total;
                return !gateSettled(total, score);
            });

        const int total = wins + draws + losses;
        const double score = (wins + 0.5 * draws) / total;
        const bool promoted = score >= promotionThreshold;
        if (promoted)
            saveAtomically(*candidate, weights);
        sendMessage(workerChannel, Message{
            .type = MessageType::evaluationResult,
            .count = static_cast<uint32_t>(total),
            .version = command.version,
            .score = score,
            .promoted = promoted,
        });
    }
}
catch (const std::exception& error)
{
    std::println(stderr, "[evaluation] {}", error.what());
    return EXIT_FAILURE;
}

int runTraining(const std::filesystem::path& weights,
                const std::filesystem::path& executable)
{
    Worker selfPlay;
    Worker evaluation;
    const std::filesystem::path liveWeights = internalPath(weights, ".live.safetensors");
    const std::filesystem::path candidateWeights =
        internalPath(weights, ".candidate.safetensors");

    try
    {
        std::unique_ptr<Network> network;
        if (std::filesystem::exists(weights))
            network = loadNetwork(weights);
        else
        {
            network = createDefaultNetwork(seed);
            saveAtomically(*network, weights);
        }

        saveAtomically(*network, liveWeights);

        report("[train] {}: {}, {} parameters", weights.string(),
               network->name(), network->parameterCount());

        selfPlay = spawnWorker(executable, "--self-play-worker", weights);
        evaluation = spawnWorker(executable, "--evaluation-worker", weights);
        signal(SIGINT, requestStop);
        signal(SIGTERM, requestStop);

        std::vector<TrainingSample> replay;
        replay.reserve(replayBufferCapacity);
        size_t replayWrite = 0;
        double credits = 0.0;
        std::mt19937_64 randomEngine{ seed };
        std::vector<mlx::core::array> parameters = network->parameters();
        Adam adam{ parameters };
        uint64_t step = 0;
        bool evaluationBusy = false;
        bool evaluationPending = false;

        const auto publishCandidate = [&] {
            network->replaceParameters(parameters);
            saveAtomically(*network, candidateWeights);
            sendMessage(evaluation.socket, Message{
                                               .type = MessageType::evaluate,
                                               .version = step,
                                           });
            evaluationBusy = true;
            evaluationPending = false;
        };

        while (!stopRequested)
        {
            const bool canTrain = replay.size() >= minimumReplaySize
                                  && credits >= trainingBatchSize;
            pollfd descriptors[]{
                { .fd = selfPlay.socket, .events = POLLIN, .revents = 0 },
                { .fd = evaluation.socket, .events = POLLIN, .revents = 0 },
            };
            const int ready = poll(descriptors, 2, canTrain ? 0 : -1);
            if (ready < 0 && errno != EINTR)
                throw std::runtime_error(std::format("poll: {}", std::strerror(errno)));
            if (stopRequested)
                break;

            if ((descriptors[0].revents & POLLIN) != 0)
            {
                Message message;
                if (!receiveMessage(selfPlay.socket, message))
                    throw std::runtime_error("self-play worker stopped");
                if (message.type != MessageType::samples)
                    throw std::runtime_error("trainer received an invalid self-play message");

                std::vector<TrainingSample> fresh(message.count);
                if (!readAll(selfPlay.socket, fresh.data(), fresh.size() * sizeof(TrainingSample)))
                    throw std::runtime_error("self-play worker stopped inside a sample batch");
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
            }

            if ((descriptors[1].revents & POLLIN) != 0)
            {
                Message result;
                if (!receiveMessage(evaluation.socket, result))
                    throw std::runtime_error("evaluation worker stopped");
                if (result.type != MessageType::evaluationResult)
                    throw std::runtime_error("trainer received an invalid evaluation message");
                report("[evaluation] step {}: {:.1f}% over {} games -- {}",
                       result.version, 100.0 * result.score, result.count,
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
                pick = std::uniform_int_distribution<size_t>{ 0, replay.size() - 1 }(randomEngine);
            const TrainingBatch batch = createBatch(replay, picks);
            const LossAndGrad result = network->valueAndGrad(parameters, batch, weightDecay);
            parameters = adam.updateParameters(parameters, result.gradients, learningRate);
            mlx::core::eval({ result.loss.policy, result.loss.value });
            mlx::core::eval(parameters);
            credits -= trainingBatchSize;
            ++step;

            const float policyLoss = result.loss.policy.item<float>();
            const float valueLoss = result.loss.value.item<float>();
            if (!std::isfinite(policyLoss) || !std::isfinite(valueLoss))
                throw std::runtime_error("training produced a non-finite loss");

            if (step % 10 == 0)
            {
                report("[train] step {}, replay {}, policy {:.4f}, value {:.4f}, credits {:.0f}",
                       step, replay.size(), policyLoss, valueLoss, credits);
            }

            if (step % selfPlayRefreshSteps == 0)
            {
                network->replaceParameters(parameters);
                saveAtomically(*network, liveWeights);
                sendMessage(selfPlay.socket, Message{
                                                 .type = MessageType::liveWeights,
                                                 .version = step,
                                             });
            }

            if (step % evaluationSteps == 0)
            {
                if (evaluationBusy)
                    evaluationPending = true;
                else
                    publishCandidate();
            }
        }

        stopWorker(selfPlay);
        stopWorker(evaluation);
        std::filesystem::remove(liveWeights);
        std::filesystem::remove(candidateWeights);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        stopWorker(selfPlay);
        stopWorker(evaluation);
        std::filesystem::remove(liveWeights);
        std::filesystem::remove(candidateWeights);
        std::println(stderr, "[train] {}", error.what());
        return EXIT_FAILURE;
    }
}

} // namespace amoeba_bot
