// amoeba_bot: plays Amoeba on Game Arena with a trained network, and does
// nothing else. Training lives in amoeba_train; the two programs only ever meet
// in a .safetensors file, so either can be killed and restarted without the
// other noticing.
//
//   amoeba_bot <weights.safetensors>
//
// The path is required and must exist: this program never trains and never
// writes weights, so there is nothing sensible to do without them, and guessing
// at which file to load is how you end up playing rated games with a network you
// did not mean to. The file is loaded once at startup and that model is used
// until the program exits.
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

#include <arena/arena.h>

#include <algorithm>
#include <charconv>
#include <chrono>
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

namespace bot
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

struct TranslatedMove
{
    amoeba::Move move;
    const char* sourcePosition;
    const char* destinationPosition;
    bool serverSplittingFlag;
};

// The model is loaded once at startup and held for the process lifetime. Each
// game gets a fresh search root, but changes to the checkpoint on disk are not
// picked up until the bot is restarted.
struct BotContext
{
    std::filesystem::path checkpointPath;
    std::unique_ptr<Network> network;
    std::unique_ptr<NetworkEvaluator> evaluator;
    std::optional<MCTS> search;
    amoeba::Board board;
    std::optional<amoeba::Outcome> outcome;
    uint16_t plyCount = 0;
    std::vector<uint64_t> positionHistory;
};

// ---------------------------------------------------------------------------
// Translation between the server's strings and the engine's move ids
// ---------------------------------------------------------------------------

int8_t parseHexIndex(std::string_view coordinate)
{
    const size_t comma = coordinate.find(',');
    if (comma == std::string_view::npos)
        return -1;

    int q{};
    int r{};
    if (std::from_chars(coordinate.data(), coordinate.data() + comma, q).ec != std::errc{})
        return -1;
    if (std::from_chars(coordinate.data() + comma + 1, coordinate.data() + coordinate.size(), r).ec != std::errc{})
        return -1;
    return amoeba::hexIndex(q, r);
}

// The engine indexes a move by direction, the server names its destination.
std::optional<uint8_t> findMoveDirection(uint8_t sourceHex, uint8_t destinationHex, uint8_t distance)
{
    for (uint8_t direction = 0; direction < amoeba::directionCount; ++direction)
    {
        if (amoeba::destinationHex(sourceHex, direction, distance) == static_cast<int8_t>(destinationHex))
            return direction;
    }
    return std::nullopt;
}

std::vector<TranslatedMove> translateServerMoves(const arena_game_state_t& state, const amoeba::Board& board)
{
    std::vector<TranslatedMove> translatedMoves;
    for (const arena_piece_moves_t& piece : std::span(state.legal_moves, state.legal_moves_count))
    {
        const int8_t sourceHex = parseHexIndex(piece.pos);
        if (sourceHex < 0)
            continue;

        const uint8_t stackHeight = board.hexes[sourceHex].height();
        if (stackHeight == 0)
            continue;

        // A one-piece stack sows and moves identically, so the engine only ever
        // emits the whole-stack form of it.
        const bool splitsStack = piece.splitting && stackHeight >= 2;

        for (const char* destination : std::span(piece.valid_moves, piece.valid_moves_count))
        {
            const int8_t destinationHex = parseHexIndex(destination);
            if (destinationHex < 0)
                continue;

            const std::optional<uint8_t> direction =
                findMoveDirection(static_cast<uint8_t>(sourceHex), static_cast<uint8_t>(destinationHex), stackHeight);
            if (!direction.has_value())
                continue;

            translatedMoves.push_back({amoeba::Move{static_cast<uint8_t>(sourceHex), *direction, splitsStack},
                                       piece.pos, destination, piece.splitting});
        }
    }
    return translatedMoves;
}

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

VisitCounts finishSearch(MCTS& search, NetworkEvaluator& evaluator)
{
    while (const amoeba::Board* leaf = search.pendingLeaf())
    {
        const amoeba::Board* boards[]{leaf};
        Evaluation evaluation;
        evaluator.evaluate(boards, std::span{&evaluation, 1});
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

void synchronizeWithServer(BotContext& context, const char* serializedServerBoard, arena_side_t currentTurn)
{
    const amoeba::Board serverBoard = amoeba::parseBoard(serializedServerBoard, currentTurn == ARENA_SIDE_WHITE);
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
                 serializedServerBoard);
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
const TranslatedMove& selectMove(BotContext& context, const std::vector<TranslatedMove>& serverMoves, std::chrono::steady_clock::time_point turnStart)
{
    if (context.outcome.has_value())
    {
        std::println(stderr, "[bot] engine calls the game over at ply {}, deferring to the server", context.plyCount);
        return serverMoves.front();
    }

    const VisitCounts visitCounts = finishSearch(*context.search, *context.evaluator);
    const uint16_t chosenMoveId = bestMove(visitCounts);

    const auto chosenMove = std::ranges::find_if(serverMoves, [chosenMoveId](const TranslatedMove& move) { return move.move.id() == chosenMoveId; });
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
    context.network = std::make_unique<Network>(context.checkpointPath);
    context.evaluator = std::make_unique<NetworkEvaluator>(*context.network);

    std::println("[bot] {}: {} blocks, width {}, {} heads, {} parameters", context.checkpointPath.filename().string(),
                 context.network->shape().blockCount, context.network->shape().embeddingWidth,
                 context.network->shape().attentionHeadCount, context.network->parameterCount());
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    runGuardedCallback("on_game_start", [&] {
        BotContext& context = *static_cast<BotContext*>(userData);

        context.board = amoeba::parseBoard(state->board, state->current_turn == ARENA_SIDE_WHITE);
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

        synchronizeWithServer(context, state->board, state->current_turn);

        const std::vector<TranslatedMove> serverMoves = translateServerMoves(*state, context.board);
        if (serverMoves.empty())
        {
            std::println(stderr, "[bot] no usable move in the server's list for {}", state->board);
            return;
        }

        const TranslatedMove& chosenMove = selectMove(context, serverMoves, turnStart);

        // The SDK owns these strings for the duration of the callback, and they
        // carry the server's own spelling of the move.
        output->from_pos = chosenMove.sourcePosition;
        output->to_pos = chosenMove.destinationPosition;
        output->side = nullptr;
        output->splitting = chosenMove.serverSplittingFlag;

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

} // namespace

} // namespace bot

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::println(stderr, "usage: amoeba_bot <weights.safetensors>");
        std::println(stderr, "  the file must exist - this program only ever reads it");
        return EXIT_FAILURE;
    }

    const char* botId = std::getenv("BOT1_ID");
    const char* apiKey = std::getenv("BOT1_KEY");
    if (botId == nullptr || apiKey == nullptr)
    {
        std::println(stderr, "set BOT1_ID and BOT1_KEY");
        return EXIT_FAILURE;
    }

    bot::BotContext context;
    try
    {
        context.checkpointPath = std::filesystem::path{argv[1]};
        if (!std::filesystem::exists(context.checkpointPath))
            throw std::runtime_error(std::format("{} does not exist", context.checkpointPath.string()));
        bot::loadCheckpoint(context);
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
                .on_move = bot::onMove,
                .on_game_start = bot::onGameStart,
                .on_game_end = bot::onGameEnd,
                .on_disconnect = bot::onDisconnect,
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
