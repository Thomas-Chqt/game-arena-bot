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
// did not mean to. Point it at the file amoeba_train is writing to.
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

// Every search runs to the full simulation count - there is no clock on it, so
// the move logged is always the one `searchSimulationCount` simulations chose,
// and the elapsed time is a measurement rather than a limit. The server allows 5 s a move, and
// nothing here enforces that. The count is simulations *through the root*, so a
// re-rooted tree arrives having spent part of it on an earlier turn: the same
// total search behind the move, but fewer new simulations.
//
// Leaves are batched 16 at a time because the network costs 1.70 ms for a single
// position and 0.16 ms each for 64 - one position per forward pass would spend
// the whole turn on overhead. Self-play fills its batch from the other games in
// flight and needs none of this; a match has one game and no such option.
constexpr int searchSimulationCount = 1000;
constexpr int searchLeavesPerBatch = 16;

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

// The model is loaded once per game and held for the whole of it. Reloading it
// mid-game would mean one tree scoring its positions with two different
// networks, which is incoherent and would not show up in any log; between games
// is the only safe moment, and it is how a promotion by a trainer running
// alongside gets picked up.
struct BotContext
{
    std::filesystem::path checkpointPath;
    std::unique_ptr<Network> network;
    std::unique_ptr<NetworkEvaluator> evaluator;
    std::optional<Search> search;
    amoeba::Board board;
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

// Re-roots the tree on the move as well as playing it, so the simulations that
// already went into that move are still there next turn instead of being thrown
// away - our own moves and the opponent's alike, since a reply the search looked
// at is a subtree it can keep.
void advanceLocalGame(BotContext& context, amoeba::Move move)
{
    context.board = amoeba::applyMove(context.board, move, context.positionHistory);
    context.positionHistory.push_back(context.board.positionHash);
    context.search->advance(move.id(), context.board, context.positionHistory);
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
            if (hasSamePosition(amoeba::applyMoveWithoutUpdatingState(context.board, move), serverBoard))
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
    context.positionHistory.assign(1, context.board.positionHash);
    context.search->restart(context.board, context.positionHistory);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

// The clock starts when the callback does, not when the search does: the server
// times the whole reply, so sync and translation count against the budget too.
const TranslatedMove& selectMove(BotContext& context, const std::vector<TranslatedMove>& serverMoves,
                                 std::chrono::steady_clock::time_point turnStart)
{
    if (context.board.state != amoeba::State::ongoing)
    {
        std::println(stderr, "[bot] engine calls the game over at ply {}, deferring to the server",
                     context.board.plyCount);
        return serverMoves.front();
    }

    const VisitCounts visitCounts = runSearch(*context.search, *context.evaluator);
    const uint16_t chosenMoveId = bestMove(visitCounts);

    const auto chosenMove = std::ranges::find_if(serverMoves, [chosenMoveId](const TranslatedMove& move)
                                                 { return move.move.id() == chosenMoveId; });
    if (chosenMove == serverMoves.end())
    {
        std::println(stderr, "[bot] search picked a move the server did not offer, falling back");
        return serverMoves.front();
    }

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - turnStart;
    const uint32_t totalVisits = std::accumulate(visitCounts.begin(), visitCounts.end(), uint32_t{0});
    std::println("[bot] ply {}: {} -> {}{}  {}/{} visits over {} moves in {:.0f} ms ({:.2f} s)", context.board.plyCount,
                 chosenMove->sourcePosition, chosenMove->destinationPosition,
                 chosenMove->move.splitsStack ? " sow" : "", visitCounts[chosenMoveId], totalVisits,
                 context.board.legalMoveCount, elapsed.count(), elapsed.count() / 1000.0);
    return *chosenMove;
}

void loadCheckpoint(BotContext& context)
{
    context.network = std::make_unique<Network>(context.checkpointPath);
    context.evaluator = std::make_unique<NetworkEvaluator>(*context.network);
    context.search.emplace(Config{.simulations = searchSimulationCount, .batchSize = searchLeavesPerBatch});

    std::println("[bot] {}: {} blocks, width {}, {} heads, {} parameters", context.checkpointPath.filename().string(),
                 context.network->shape().blockCount, context.network->shape().embeddingWidth,
                 context.network->shape().attentionHeadCount, context.network->parameterCount());
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    runGuardedCallback("on_game_start",
                       [&]
                       {
                           BotContext& context = *static_cast<BotContext*>(userData);
                           loadCheckpoint(context);

                           context.board = amoeba::parseBoard(state->board, state->current_turn == ARENA_SIDE_WHITE);
                           context.positionHistory.assign(1, context.board.positionHash);
                           context.search->restart(context.board, context.positionHistory);

                           std::println("[bot] game start, I am {}, {} to move", arena_side_str(state->my_side),
                                        arena_side_str(state->current_turn));
                       });
}

void onMove(const arena_game_state_t* state, arena_move_t* output, void* userData)
{
    runGuardedCallback("on_move",
                       [&]
                       {
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
    runGuardedCallback(
        "on_game_end",
        [&]
        {
            const BotContext& context = *static_cast<BotContext*>(userData);
            const char* result = !state->has_winner ? "draw" : state->winner == state->my_side ? "won" : "lost";
            std::println("[bot] {} after {} plies", result, context.board.plyCount);
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
