// Plays random legal moves against a live Game Arena practice room and checks,
// every ply, that the local rules engine produces exactly the move set the
// server does. Any disagreement is a rules bug and stops the run.

#include <amoeba/amoeba.hpp>

#include <arena/arena.h>

#include <algorithm>
#include <bitset>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace amoeba;

namespace {

struct ServerMove
{
    Move                       move;
    const arena_piece_moves_t* piece;
    const char*                destination;
};

struct Context
{
    std::mt19937_64       random{std::random_device{}()};
    Board                 board;
    std::vector<uint64_t> history;
    bool                  completed{};
};

template <typename... Args>
[[noreturn]] void fail(std::format_string<Args...> format, Args&&... args)
{
    std::println(stderr, "[test] FAILED: {}", std::format(format, std::forward<Args>(args)...));
    std::exit(EXIT_FAILURE);
}

// ---------------------------------------------------------------------------
// Coordinates
// ---------------------------------------------------------------------------

[[nodiscard]] int8_t parseHex(std::string_view text)
{
    const size_t comma = text.find(',');
    if (comma == std::string_view::npos)
        return -1;

    int q{};
    int r{};
    if (std::from_chars(text.data(), text.data() + comma, q).ec != std::errc{})
        return -1;
    if (std::from_chars(text.data() + comma + 1, text.data() + text.size(), r).ec != std::errc{})
        return -1;
    return hexIndex(q, r);
}

[[nodiscard]] std::string hexName(uint8_t hex)
{
    const Coordinate coordinate = kCoordinates[hex];
    return std::format("{},{}", coordinate.q, coordinate.r);
}

// The engine indexes a move by direction, the server names its destination.
[[nodiscard]] std::optional<uint8_t> findDirection(uint8_t from, uint8_t to, uint8_t steps)
{
    for (uint8_t dir = 0; dir < kNumDirs; ++dir)
        if (ray(from, dir, steps) == static_cast<int8_t>(to))
            return dir;
    return std::nullopt;
}

[[nodiscard]] std::string describe(const Board& board, Move move)
{
    const int8_t to = ray(move.from, move.dir, board.hexes[move.from].height());
    return std::format("{} -> {} ({})",
                       hexName(move.from),
                       to < 0 ? std::string{"off-board"} : hexName(static_cast<uint8_t>(to)),
                       move.splitting ? "sow" : "whole");
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

[[nodiscard]] bool samePosition(const Board& left, const Board& right)
{
    return left.whiteToMove == right.whiteToMove && std::ranges::equal(left.hexes, right.hexes);
}

// fromString ignores anything it cannot parse, so a format we misread shows up
// as pieces going missing rather than as an error.
[[nodiscard]] Board parsePosition(const char* board, bool whiteToMove)
{
    const Board parsed = fromString(board, whiteToMove);
    const int   pieces = std::accumulate(std::begin(parsed.hexes),
                                       std::end(parsed.hexes),
                                       0,
                                       [](int sum, const Hex& hex) { return sum + hex.height(); });
    if (pieces != kMaxHeight)
        fail("parsed {} pieces, expected {}\n  server: {}\n  local:  {}", pieces, kMaxHeight, board, toString(parsed));
    return parsed;
}

// Between two of our turns the opponent plays exactly one move. Replaying it
// locally is what checks that apply() lands split pieces where the server does.
void replayOpponent(Context& context, const char* board, arena_side_t currentTurn)
{
    const Board remote = parsePosition(board, currentTurn == ARENA_SIDE_WHITE);
    if (samePosition(context.board, remote))
        return;

    context.history.push_back(context.board.hash);

    Move                 played{};
    std::optional<Board> successor;
    context.board.forEachLegal([&](uint16_t id) {
        if (successor.has_value())
            return;
        const Move  move      = Move::fromId(id);
        const Board candidate = apply(context.board, move, context.history);
        if (samePosition(candidate, remote))
        {
            played    = move;
            successor = candidate;
        }
    });

    if (!successor.has_value())
        fail("server position is not reachable by any legal local move\n  server: {}\n  local:  {}",
             board,
             toString(context.board));

    std::println("[test] opponent played {}", describe(context.board, played));
    context.board = *successor;
}

// ---------------------------------------------------------------------------
// Move sets
// ---------------------------------------------------------------------------

[[nodiscard]] std::vector<ServerMove> collectServerMoves(const arena_game_state_t& state, const Board& board)
{
    std::vector<ServerMove> moves;
    for (const arena_piece_moves_t& piece : std::span(state.legal_moves, state.legal_moves_count))
    {
        if (!piece.has_splitting)
            fail("server move is missing its splitting field");

        const int8_t from = parseHex(piece.pos);
        if (from < 0)
            fail("server sent an unparseable position: {}", piece.pos);

        const uint8_t height = board.hexes[from].height();
        if (height == 0)
            fail("server can move an empty hex: {}", piece.pos);

        // A one-piece stack sows and moves identically, so the engine only ever
        // emits the whole-stack form of it.
        const bool splitting = piece.splitting && height >= 2;

        for (const char* destination : std::span(piece.valid_moves, piece.valid_moves_count))
        {
            const int8_t                 to  = parseHex(destination);
            const std::optional<uint8_t> dir = to < 0
                ? std::nullopt
                : findDirection(static_cast<uint8_t>(from), static_cast<uint8_t>(to), height);
            if (!dir.has_value())
                fail("server move is not a {}-step ray: {} -> {}", height, piece.pos, destination);

            moves.push_back({Move{static_cast<uint8_t>(from), *dir, splitting}, &piece, destination});
        }
    }
    return moves;
}

void compareMoveSets(const Board& board, const std::vector<ServerMove>& server)
{
    std::bitset<kNumMoveIds> remote;
    for (const ServerMove& move : server)
        remote.set(move.move.id());

    std::string differences;
    for (uint16_t id = 0; id < kNumMoveIds; ++id)
        if (board.isLegal(id) != remote.test(id))
            differences += std::format("\n  {:<13}{}",
                                       board.isLegal(id) ? "local only:" : "server only:",
                                       describe(board, Move::fromId(id)));

    if (!differences.empty())
        fail("legal-move mismatch\n  position: {}\n  side: {}{}",
             toString(board),
             board.whiteToMove ? "white" : "black",
             differences);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void onGameStart(const arena_game_state_t* state, void* userData)
{
    Context& context = *static_cast<Context*>(userData);
    context.board    = parsePosition(state->board, state->current_turn == ARENA_SIDE_WHITE);
    context.history.clear();

    std::println("[test] start: {}, I am {}", toString(context.board), arena_side_str(state->my_side));
}

void onMove(const arena_game_state_t* state, arena_move_t* output, void* userData)
{
    Context& context = *static_cast<Context*>(userData);
    replayOpponent(context, state->board, state->current_turn);
    if (context.board.state != State::Ongoing)
        fail("server continued a game the engine considers finished");

    const std::vector<ServerMove> server = collectServerMoves(*state, context.board);
    compareMoveSets(context.board, server);
    if (server.empty())
        fail("server asked for a move with no legal actions");

    const ServerMove& chosen = server[std::uniform_int_distribution<size_t>(0, server.size() - 1)(context.random)];
    std::println("[test] ply {}: {} moves matched, playing {}",
                 context.board.ply,
                 server.size(),
                 describe(context.board, chosen.move));

    // The SDK owns these strings for the duration of the callback, and they
    // carry the server's own spelling of the move.
    output->from_pos  = chosen.piece->pos;
    output->to_pos    = chosen.destination;
    output->side      = nullptr;
    output->splitting = chosen.piece->splitting;

    context.history.push_back(context.board.hash);
    context.board = apply(context.board, chosen.move, context.history);
}

void onGameEnd(const arena_game_end_t* state, void* userData)
{
    Context& context = *static_cast<Context*>(userData);
    replayOpponent(context, state->board, state->current_turn);

    // The engine's move cap and repetition limit are guesses; the ply the server
    // stops at is the evidence for what they really are.
    if (context.board.state == State::Ongoing)
        fail("server ended the game at ply {} while the engine considers it ongoing "
             "(position seen {} times; kMoveCap={}, kRepetitionLimit={})",
             context.board.ply,
             std::ranges::count(context.history, context.board.hash) + 1,
             kMoveCap,
             kRepetitionLimit);

    const bool draw = context.board.state == State::Draw;
    if (state->has_winner == draw)
        fail("result mismatch: one implementation reports a draw");

    const bool whiteWins = context.board.state == State::WhiteWins;
    if (state->has_winner && (state->winner == ARENA_SIDE_WHITE) != whiteWins)
        fail("winner mismatch: server={}, local={}", arena_side_str(state->winner), whiteWins ? "white" : "black");

    context.completed = true;
    std::println("[test] PASSED after {} plies", context.board.ply);
}

void onDisconnect(const char* reason, void*)
{
    std::println(stderr, "[test] disconnected: {}", reason == nullptr ? "unknown reason" : reason);
}

} // namespace

int main()
{
    const char* botId  = std::getenv("BOT1_ID");
    const char* apiKey = std::getenv("BOT1_KEY");
    if (botId == nullptr || apiKey == nullptr || std::getenv("ROOM_ID") == nullptr)
    {
        std::println(stderr, "set BOT1_ID, BOT1_KEY and ROOM_ID");
        return EXIT_FAILURE;
    }

    Context context;
    const arena_bot_config_t config{
        .bot_id   = botId,
        .api_key  = apiKey,
        .callbacks = {
            .on_move       = onMove,
            .on_game_start = onGameStart,
            .on_game_end   = onGameEnd,
            .on_disconnect = onDisconnect,
        },
        .user_data = &context,
    };

    if (const int result = arena_start_practice(&config, nullptr); result != 0)
        return result;
    if (!context.completed)
    {
        std::println(stderr, "[test] no verified complete game");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
