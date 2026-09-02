// amoeba_2_bot is a non-neural Amoeba bot.  It keeps the same Game Arena SDK
// callbacks and environment variables as amoeba_bot, but needs no checkpoint:
//
//   amoeba_2_bot [--continuous]

#include "alpha_beta.hpp"

#include <arena/arena.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace amoeba_bot
{
namespace
{

constexpr auto searchBudget = std::chrono::milliseconds{4'400};

template <typename Callback> void runGuardedCallback(const char* callbackName, Callback&& callback)
{
    try { callback(); }
    catch (const std::exception& error) { std::println(stderr, "[amoeba_2] {} threw: {}", callbackName, error.what()); }
    catch (...) { std::println(stderr, "[amoeba_2] {} threw an unknown exception", callbackName); }
}

struct BotContext
{
    Board board;
    std::optional<Outcome> outcome;
    uint16_t plyCount{};
    std::vector<uint64_t> positionHistory;
};

bool hasSamePosition(const Board& left, const Board& right)
{
    return left.whiteToMove == right.whiteToMove && std::ranges::equal(left.hexes, right.hexes);
}

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
}

void synchronizeWithServer(BotContext& context, const arena_game_state_t& state)
{
    const Board serverBoard{state};
    if (hasSamePosition(context.board, serverBoard))
        return;

    std::optional<Move> opponentMove;
    context.board.forEachLegal([&](uint16_t moveId) {
        if (opponentMove.has_value())
            return;
        const MoveResult result = applyMove(context.board, Move::fromId(moveId), context.positionHistory);
        if (const Board* next = std::get_if<Board>(&result); next != nullptr && hasSamePosition(*next, serverBoard))
            opponentMove = Move::fromId(moveId);
    });
    if (opponentMove.has_value())
    {
        advanceLocalGame(context, *opponentMove, &serverBoard);
        return;
    }

    // Preserve the original bot's safe resynchronization behaviour.  The server
    // does not provide past positions, so this intentionally restarts history.
    std::println(stderr, "[amoeba_2] resync at ply {}: no legal move reaches {}", context.board.plyCount, state.board);
    context.board = serverBoard;
    context.outcome.reset();
    context.plyCount = context.board.plyCount;
    context.positionHistory.assign(1, context.board.positionHash);
}

Move selectMove(BotContext& context, std::chrono::steady_clock::time_point turnStart)
{
    const SearchResult result = chooseAlphaBetaMove(context.board, context.positionHistory, turnStart + searchBudget);
    const Move move = Move::fromId(result.moveId);
    const arena_move_t arenaMove = move.toArena(context.board.hexes[move.sourceCoord].height());
    const auto elapsed = std::chrono::steady_clock::now() - turnStart;
    std::println("[amoeba_2] ply {}: {} -> {}{}  depth {}, {} nodes, score {} in {:.0f} ms",
                 context.board.plyCount, arenaMove.from_pos, arenaMove.to_pos,
                 move.splitsStack ? " sow" : "", result.completedDepth, result.nodes, result.score,
                 std::chrono::duration<double, std::milli>{elapsed}.count());
    return move;
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    runGuardedCallback("on_game_start", [&] {
        BotContext& context = *static_cast<BotContext*>(userData);
        context.board = Board(*state);
        context.outcome.reset();
        context.plyCount = context.board.plyCount;
        context.positionHistory.assign(1, context.board.positionHash);
        std::println("[amoeba_2] game start, I am {}, {} to move", arena_side_str(state->my_side), arena_side_str(state->current_turn));
    });
}

void onMove(const arena_game_state_t* state, arena_move_t* output, void* userData)
{
    runGuardedCallback("on_move", [&] {
        const auto turnStart = std::chrono::steady_clock::now();
        BotContext& context = *static_cast<BotContext*>(userData);
        synchronizeWithServer(context, *state);
        if (context.board.legalMoveCount == 0)
        {
            std::println(stderr, "[amoeba_2] no usable move in the server's list for {}", state->board);
            return;
        }
        const Move move = selectMove(context, turnStart);
        *output = move.toArena(context.board.hexes[move.sourceCoord].height());
        advanceLocalGame(context, move);
    });
}

void onGameEnd(const arena_game_end_t* state, void* userData)
{
    runGuardedCallback("on_game_end", [&] {
        const BotContext& context = *static_cast<BotContext*>(userData);
        const char* result = !state->has_winner ? "draw" : state->winner == state->my_side ? "won" : "lost";
        std::println("[amoeba_2] {} after {} plies", result, context.plyCount);
    });
}

void onDisconnect(const char* reason, void*)
{
    std::println(stderr, "[amoeba_2] disconnected: {}", reason == nullptr ? "unknown reason" : reason);
}

int runBot(bool continuous)
{
    const char* botId = std::getenv("BOT1_ID");
    const char* apiKey = std::getenv("BOT1_KEY");
    if (botId == nullptr || apiKey == nullptr)
    {
        std::println(stderr, "set BOT1_ID and BOT1_KEY");
        return EXIT_FAILURE;
    }
    BotContext context;
    const arena_bot_config_t config{
        .bot_id = botId, .api_key = apiKey,
        .callbacks = {.on_move = onMove, .on_game_start = onGameStart,
                      .on_game_end = onGameEnd, .on_disconnect = onDisconnect},
        .user_data = &context,
    };
    if (const char* roomId = std::getenv("ROOM_ID"))
        return arena_start_practice(&config, roomId);
    return continuous ? arena_start_continuous(&config, ARENA_GAME_AMOEBA)
                      : arena_start(&config, ARENA_GAME_AMOEBA);
}

} // namespace
} // namespace amoeba_bot

int main(int argc, char** argv)
{
    const bool continuous = argc == 2 && std::string_view{argv[1]} == "--continuous";
    if (argc == 1 || continuous)
        return amoeba_bot::runBot(continuous);
    std::println(stderr, "usage: amoeba_2_bot [--continuous]");
    return EXIT_FAILURE;
}
