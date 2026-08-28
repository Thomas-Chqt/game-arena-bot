// amoeba_bot either plays Amoeba on Game Arena or trains the same network:
//
//   amoeba_bot [--network <identifier>] <weights.safetensors>
//   amoeba_bot --train [--network <identifier>] <weights.safetensors>
//
// An existing checkpoint identifies its own architecture. --network checks that
// identifier, or selects the architecture when creating a missing checkpoint.
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
#include "network.hpp"
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

namespace amoeba_bot
{

namespace
{

// Every move starts a fresh search and runs to the full simulation count. There
// is no clock on it, so elapsed time is a measurement rather than a limit. The
// server allows 5 s a move, and nothing here enforces that.
constexpr int searchSimulationCount = 1000;
constexpr uint64_t initializationSeed = 20260819;

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
    std::unique_ptr<Network> network;
    std::optional<MCTS<searchSimulationCount>> search;
    Board board;
    std::optional<Outcome> outcome;
    uint16_t plyCount = 0;
    std::vector<uint64_t> positionHistory;
};

// ---------------------------------------------------------------------------
// Keeping the local board in step with the server's
// ---------------------------------------------------------------------------

bool hasSamePosition(const Board& left, const Board& right)
{
    return left.whiteToMove == right.whiteToMove && std::ranges::equal(left.hexes, right.hexes);
}

void startSearch(BotContext& context)
{
    context.search.emplace(context.board, context.positionHistory);
}

VisitCounts finishSearch(MCTS<searchSimulationCount>& search, const Network& network)
{
    while (const Board* leaf = search.pendingLeaf())
    {
        const Board* boards[]{leaf};
        Evaluation evaluation;
        network(boards, std::span{&evaluation, 1});
        search.absorb(evaluation.policy, evaluation.value);
    }
    return search.visits();
}

// Plays a move locally and creates a fresh search for the resulting position.
void advanceLocalGame(BotContext& context, Move move, const Board* serverBoard = nullptr)
{
    MoveResult result = applyMove(context.board, move, context.positionHistory);
    if (const auto* outcome = std::get_if<Outcome>(&result))
    {
        context.outcome = *outcome;
        context.plyCount = static_cast<uint16_t>(context.board.plyCount + 1);
        return;
    }

    context.board = std::get<Board>(std::move(result));
    if (serverBoard != nullptr)
    {
        assert(hasSamePosition(context.board, *serverBoard));
        for (int word = 0; word < legalMoveWordCount; ++word)
            context.board.legalMoveBits[word] = serverBoard->legalMoveBits[word];
        context.board.legalMoveCount = serverBoard->legalMoveCount;
    }
    context.plyCount = context.board.plyCount;
    context.positionHistory.push_back(context.board.positionHash);
    startSearch(context);
}

void synchronizeWithServer(BotContext& context, const arena_game_state_t& state)
{
    const Board serverBoard{state};
    if (hasSamePosition(context.board, serverBoard))
        return;

    // Between two of our turns the opponent plays exactly one move; find which.
    std::optional<Move> opponentMove;
    context.board.forEachLegal(
        [&](uint16_t moveId)
        {
            if (opponentMove.has_value())
                return;
            const Move move = Move::fromId(moveId);
            const MoveResult result = applyMove(context.board, move, context.positionHistory);
            const Board* nextBoard = std::get_if<Board>(&result);
            if (nextBoard != nullptr && hasSamePosition(*nextBoard, serverBoard))
                opponentMove = move;
        });

    if (opponentMove.has_value())
    {
        advanceLocalGame(context, *opponentMove, &serverBoard);
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
Move selectMove(BotContext& context, std::chrono::steady_clock::time_point turnStart)
{
    if (context.outcome.has_value())
    {
        std::println(stderr, "[bot] engine calls the game over at ply {}, deferring to the server", context.plyCount);
        std::optional<Move> firstMove;
        context.board.forEachLegal([&](uint16_t id) {
            if (!firstMove.has_value())
                firstMove = Move::fromId(id);
        });
        assert(firstMove.has_value());
        return firstMove.value();
    }

    const VisitCounts visitCounts = finishSearch(*context.search, *context.network);
    const uint16_t chosenMoveId = bestMove(visitCounts);

    if (!context.board.isLegal(chosenMoveId))
    {
        std::println(stderr, "[bot] search picked a move the server did not offer, falling back");
        std::optional<Move> firstMove;
        context.board.forEachLegal([&](uint16_t id) {
            if (!firstMove.has_value())
                firstMove = Move::fromId(id);
        });
        assert(firstMove.has_value());
        return firstMove.value();
    }

    const Move chosenMove = Move::fromId(chosenMoveId);
    const arena_move_t arenaMove = chosenMove.toArena(context.board.hexes[chosenMove.sourceCoord].height());

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - turnStart;
    const uint32_t totalVisits = std::accumulate(visitCounts.begin(), visitCounts.end(), uint32_t{0});
    std::println("[bot] ply {}: {} -> {}{}  {}/{} visits over {} moves in {:.0f} ms ({:.2f} s)",
        context.board.plyCount,
        arenaMove.from_pos,
        arenaMove.to_pos,
        chosenMove.splitsStack ? " sow" : "",
        visitCounts[chosenMoveId], totalVisits,
        context.board.legalMoveCount,
        elapsed.count(),
        elapsed.count() / 1000.0
    );
    return chosenMove;
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    runGuardedCallback("on_game_start", [&] {
        BotContext& context = *static_cast<BotContext*>(userData);

        context.board = Board(*state);
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

        if (context.board.legalMoveCount == 0)
        {
            std::println(stderr, "[bot] no usable move in the server's list for {}", state->board);
            return;
        }

        const Move chosenMove = selectMove(context, turnStart);

        *output = chosenMove.toArena(context.board.hexes[chosenMove.sourceCoord].height());

        advanceLocalGame(context, chosenMove);
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

int runBot(const std::filesystem::path& weights,
           std::string_view networkIdentifier)
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
        {
            if (networkIdentifier.empty())
                throw std::runtime_error(std::format(
                    "{} does not exist; select a network with --network",
                    context.checkpointPath.string()));
            context.network = createNetwork(networkIdentifier, initializationSeed);
            context.network->save(context.checkpointPath);
        }
        else
            context.network = loadNetwork(context.checkpointPath, networkIdentifier);
        std::println("[bot] {}: {}, {} parameters",
                     context.checkpointPath.filename().string(),
                     context.network->name(), context.network->parameterCount());
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

} // namespace amoeba_bot

int main(int argc, char** argv)
{
    bool train = false;
    bool valid = true;
    bool networkSpecified = false;
    std::string_view networkIdentifier;
    std::filesystem::path weights;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument{argv[index]};
        if (argument == "--train")
            train = true;
        else if (argument == "--network" && !networkSpecified
                 && index + 1 < argc)
        {
            networkSpecified = true;
            networkIdentifier = argv[++index];
            if (networkIdentifier.empty() || networkIdentifier.starts_with('-'))
                valid = false;
        }
        else if (argument.starts_with('-') || !weights.empty())
            valid = false;
        else
            weights = argument;
    }

    if (valid && !weights.empty())
    {
        if (train)
            return amoeba_bot::runTraining(weights, networkIdentifier);
        return amoeba_bot::runBot(weights, networkIdentifier);
    }

    std::println(stderr,
                 "usage: amoeba_bot [--network <identifier>] <weights.safetensors>");
    std::println(stderr,
                 "       amoeba_bot --train [--network <identifier>] <weights.safetensors>");
    return EXIT_FAILURE;
}
