// amoeba_bot either plays Amoeba on Game Arena or trains the same network:
//
//   amoeba_bot <weights.safetensors>
//   amoeba_bot --train <weights.safetensors>
//
// Bot mode requires an existing checkpoint and keeps it for the process lifetime.
// Training mode creates a missing checkpoint and replaces it whenever a candidate
// defeats the current champion.
//
// Credentials come from BOT1_ID and BOT1_KEY; ROOM_ID picks a practice room, and
// without it the bot queues for ranked games back to back.
//
// The SDK side is the same shape as the random bot in the reference: fill
// arena_move_t with the strings the server sent, and copy the piece's splitting
// flag straight back.
//
// The server sends a position and nothing else - no ply count, no history - so
// the bot keeps its own board and replays the opponent's move onto it. That is
// what keeps ply, staleness and the repetition history right, and the search
// needs all three to see draws and adjudications coming.

#include "mcts.hpp"
#include "amoeba_network.hpp"
#include "training.hpp"

#include <arena/arena.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace amoeba
{

namespace
{

// Every move starts a fresh search and runs to the full simulation count. There
// is no clock on it, so elapsed time is a measurement rather than a limit. The
// server allows 5 s a move, and nothing here enforces that.
constexpr int searchSimulationCount = 1000;

// A C++ exception unwinding through the SDK's C frames is undefined behaviour,
// so nothing may leave a callback. A turn that goes unanswered times out and
// loses one game; a crash here loses every game that would have followed.
template <typename Callback> void runGuardedCallback(const char* callbackName, Callback&& callback)
{
    try
    {
        callback();
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "[bot] {} threw: {}", callbackName, error.what());
    }
    catch (...)
    {
        std::println(stderr, "[bot] {} threw something that is not an exception", callbackName);
    }
}

// The model is loaded once at startup and held for the process lifetime. Each
// game gets a fresh search root, but changes to the checkpoint on disk are not
// picked up until the bot is restarted.
struct BotContext
{
    std::filesystem::path checkpointPath;
    std::unique_ptr<AmoebaNetwork> network;
    std::optional<MCTS> search;
    amoeba::Board board;
    std::optional<amoeba::Outcome> outcome;
    uint16_t plyCount = 0;
    std::vector<uint64_t> positionHistory;
};

// ---------------------------------------------------------------------------
// Keeping the local board in step with the server's
// ---------------------------------------------------------------------------

bool hasSamePosition(const amoeba::Board& left, const amoeba::Board& right)
{
    return left.whiteToMove == right.whiteToMove && std::ranges::equal(left.hexes, right.hexes);
}

void startSearch(BotContext& context)
{
    context.search.emplace(context.board, context.positionHistory,
                           MCTSConfig{.simulations = searchSimulationCount});
}

VisitCounts finishSearch(MCTS& search, const Network& network)
{
    while (const amoeba::Board* leaf = search.pendingLeaf())
    {
        const amoeba::Board* boards[]{leaf};
        Evaluation evaluation;
        network(boards, std::span{&evaluation, 1});
        search.absorb(evaluation);
    }
    return search.visits();
}

// Plays a move locally and creates a fresh search for the resulting position.
void advanceLocalGame(BotContext& context, amoeba::Move move)
{
    amoeba::MoveResult result = amoeba::applyMove(context.board, move, context.positionHistory);
    if (const auto* outcome = std::get_if<amoeba::Outcome>(&result))
    {
        context.outcome = *outcome;
        context.plyCount = static_cast<uint16_t>(context.board.plyCount + 1);
        return;
    }

    context.board = std::get<amoeba::Board>(std::move(result));
    context.plyCount = context.board.plyCount;
    context.positionHistory.push_back(context.board.positionHash);
    startSearch(context);
}

void synchronizeWithServer(BotContext& context, const arena_game_state_t& state)
{
    const amoeba::Board serverBoard = boardFromArena(state);
    if (hasSamePosition(context.board, serverBoard))
        return;

    // Between two of our turns the opponent plays exactly one move; find which.
    std::optional<amoeba::Move> opponentMove;
    context.board.forEachLegal(
        [&](uint16_t moveId)
        {
            if (opponentMove.has_value())
                return;
            const amoeba::Move move = amoeba::Move::fromId(moveId);
            const amoeba::MoveResult result = amoeba::applyMove(context.board, move, context.positionHistory);
            const amoeba::Board* nextBoard = std::get_if<amoeba::Board>(&result);
            if (nextBoard != nullptr && hasSamePosition(*nextBoard, serverBoard))
                opponentMove = move;
        });

    if (opponentMove.has_value())
    {
        advanceLocalGame(context, *opponentMove);
        return;
    }

    // The engine and the server disagree about what some move does. Adopting the
    // server's position costs the staleness and repetition counters, but that
    // beats playing out the rest of the game from a board that never existed.
    std::println(stderr, "[bot] resync at ply {}: no legal move reaches {}", context.board.plyCount,
                 state.board);
    const uint16_t plyCount = context.board.plyCount;
    context.board = serverBoard;
    context.board.plyCount = static_cast<uint16_t>(plyCount + 1);
    context.outcome.reset();
    context.plyCount = context.board.plyCount;
    context.positionHistory.assign(1, context.board.positionHash);
    startSearch(context);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

// The clock starts when the callback does, not when the search does: the server
// times the whole reply, so sync and translation count against the budget too.
const ArenaMove& selectMove(BotContext& context, const std::vector<ArenaMove>& serverMoves,
                            std::chrono::steady_clock::time_point turnStart)
{
    if (context.outcome.has_value())
    {
        std::println(stderr, "[bot] engine calls the game over at ply {}, deferring to the server", context.plyCount);
        return serverMoves.front();
    }

    const VisitCounts visitCounts = finishSearch(*context.search, *context.network);
    const uint16_t chosenMoveId = bestMove(visitCounts);

    const auto chosenMove = std::ranges::find_if(
        serverMoves, [chosenMoveId](const ArenaMove& move) { return move.move.id() == chosenMoveId; });
    if (chosenMove == serverMoves.end())
    {
        std::println(stderr, "[bot] search picked a move the server did not offer, falling back");
        return serverMoves.front();
    }

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - turnStart;
    const uint32_t totalVisits = std::accumulate(visitCounts.begin(), visitCounts.end(), uint32_t{0});
    std::println("[bot] ply {}: {} -> {}{}  {}/{} visits over {} moves in {:.0f} ms ({:.2f} s)",
        context.board.plyCount,
        chosenMove->sourcePosition,
        chosenMove->destinationPosition,
        chosenMove->move.splitsStack ? " sow" : "",
        visitCounts[chosenMoveId], totalVisits,
        context.board.legalMoveCount,
        elapsed.count(),
        elapsed.count() / 1000.0
    );
    return *chosenMove;
}

void loadCheckpoint(BotContext& context)
{
    context.network = std::make_unique<AmoebaNetwork>(context.checkpointPath);

    std::println("[bot] {}: {}, {} parameters", context.checkpointPath.filename().string(),
                 AmoebaNetwork::name, context.network->parameterCount());
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    runGuardedCallback("on_game_start", [&] {
        BotContext& context = *static_cast<BotContext*>(userData);

        context.board = boardFromArena(*state);
        context.outcome.reset();
        context.plyCount = context.board.plyCount;
        context.positionHistory.assign(1, context.board.positionHash);
        startSearch(context);

        std::println("[bot] game start, I am {}, {} to move", arena_side_str(state->my_side), arena_side_str(state->current_turn));
    });
}

void onMove(const arena_game_state_t* state, arena_move_t* output, void* userData)
{
    runGuardedCallback("on_move", [&] {
        const std::chrono::steady_clock::time_point turnStart = std::chrono::steady_clock::now();

        BotContext& context = *static_cast<BotContext*>(userData);
        if (!context.search.has_value())
        {
            std::println(stderr, "[bot] no model loaded, cannot move");
            return;
        }

        synchronizeWithServer(context, *state);

        const std::vector<ArenaMove> serverMoves = movesFromArena(*state, context.board);
        if (serverMoves.empty())
        {
            std::println(stderr, "[bot] no usable move in the server's list for {}", state->board);
            return;
        }

        const ArenaMove& chosenMove = selectMove(context, serverMoves, turnStart);

        // The SDK owns these strings for the duration of the callback, and they
        // carry the server's own spelling of the move.
        *output = moveToArena(chosenMove);

        advanceLocalGame(context, chosenMove.move);
    });
}

void onGameEnd(const arena_game_end_t* state, void* userData)
{
    runGuardedCallback("on_game_end", [&] {
        const BotContext& context = *static_cast<BotContext*>(userData);
        const char* result = !state->has_winner ? "draw" : state->winner == state->my_side ? "won" : "lost";
        std::println("[bot] {} after {} plies", result, context.plyCount);
    });
}

void onDisconnect(const char* reason, void*)
{
    std::println(stderr, "[bot] disconnected: {}", reason == nullptr ? "unknown reason" : reason);
}

// Flush high-level training progress even when stdout is redirected to a file.
template <typename... Args> void report(std::format_string<Args...> format, Args&&... args)
{
    std::println(format, std::forward<Args>(args)...);
    std::fflush(stdout);
}

void runTraining(const std::filesystem::path& weights)
{
    const TrainingSettings settings;

    // Generation zero is saved immediately so bot mode has a checkpoint while
    // the first generation is still being produced.
    std::unique_ptr<AmoebaNetwork> champion;
    if (std::filesystem::exists(weights))
    {
        champion = std::make_unique<AmoebaNetwork>(weights);
        report("[train] resuming from {}: {}, {} parameters", weights.string(),
               AmoebaNetwork::name, champion->parameterCount());
    }
    else
    {
        champion = std::make_unique<AmoebaNetwork>(settings.seed);
        champion->save(weights);
        report("[train] {} did not exist: started from random weights, {} parameters, saved",
               weights.string(), champion->parameterCount());
    }

    std::vector<TrainingSample> replay;
    int gameIdBase = 0;

    for (int generation = 1;; ++generation)
    {
        report("");
        report("======== generation {} ========", generation);

        std::vector<TrainingSample> fresh = generateSelfPlaySamples(
            *champion, settings,
            settings.seed + static_cast<uint64_t>(generation) * 1000);

        // IDs stay unique across generations so holding out one game cannot
        // accidentally retain positions from an older game with the same ID.
        for (TrainingSample& sample : fresh)
            sample.gameId += gameIdBase;
        gameIdBase += settings.selfPlayGameCount;

        reportMemory("self-play");

        replay.insert(replay.end(), std::make_move_iterator(fresh.begin()),
                      std::make_move_iterator(fresh.end()));
        if (replay.size() > settings.replayBufferCapacity)
        {
            replay.erase(replay.begin(), replay.begin() + static_cast<long>(
                replay.size() - settings.replayBufferCapacity));
        }
        report("[train] replay buffer holds {} positions", replay.size());

        // Training refines a copy; the champion remains unchanged until the
        // candidate proves itself in games.
        AmoebaNetwork candidate = *champion;
        trainCandidate(candidate, replay, settings,
                       settings.seed + static_cast<uint64_t>(generation));
        reportMemory("training");

        const double score = evaluateCandidate(
            candidate, *champion, settings,
            settings.seed + 7777 + static_cast<uint64_t>(generation));
        reportMemory("the gate");

        if (score < settings.promotionThreshold)
        {
            report("[train] generation {} rejected at {:.1f}%, keeping the previous weights",
                   generation, 100.0 * score);
        }
        else
        {
            *champion = candidate;
            champion->save(weights);
            report("[train] generation {} PROMOTED at {:.1f}%, wrote {}",
                   generation, 100.0 * score, weights.string());
        }
    }
}

int runBot(const std::filesystem::path& weights)
{
    const char* botId = std::getenv("BOT1_ID");
    const char* apiKey = std::getenv("BOT1_KEY");
    if (botId == nullptr || apiKey == nullptr)
    {
        std::println(stderr, "set BOT1_ID and BOT1_KEY");
        return EXIT_FAILURE;
    }

    BotContext context;
    try
    {
        context.checkpointPath = weights;
        if (!std::filesystem::exists(context.checkpointPath))
            throw std::runtime_error(std::format("{} does not exist", context.checkpointPath.string()));
        loadCheckpoint(context);
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "[bot] {}", error.what());
        return EXIT_FAILURE;
    }

    const arena_bot_config_t config{
        .bot_id = botId,
        .api_key = apiKey,
        .callbacks =
            {
                .on_move = onMove,
                .on_game_start = onGameStart,
                .on_game_end = onGameEnd,
                .on_disconnect = onDisconnect,
            },
        .user_data = &context,
    };

    // A practice room when one is named, otherwise ranked games back to back for
    // as long as the process lives. arena_start_continuous only returns on error.
    if (const char* roomId = std::getenv("ROOM_ID"))
        return arena_start_practice(&config, roomId);

    const int status = arena_start_continuous(&config, ARENA_GAME_AMOEBA);
    std::println(stderr, "[bot] continuous play stopped with status {}", status);
    return status;
}

} // namespace

} // namespace amoeba

int main(int argc, char** argv)
{
    if (argc == 3 && std::string_view{argv[1]} == "--train")
    {
        amoeba::runTraining(std::filesystem::path{argv[2]});
        return EXIT_SUCCESS;
    }

    if (argc == 2 && std::string_view{argv[1]} != "--train")
        return amoeba::runBot(std::filesystem::path{argv[1]});

    std::println(stderr, "usage: amoeba_bot <weights.safetensors>");
    std::println(stderr, "       amoeba_bot --train <weights.safetensors>");
    return EXIT_FAILURE;
}
