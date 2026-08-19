// Plays random legal moves against a live Game Arena practice room and checks,
// every ply, that the local rules engine produces exactly the move set the
// server does. Any disagreement is a rules bug and stops the run.

#include <amoeba/amoeba.hpp>

#include <arena/arena.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace game = amoeba;

using Mask = std::array<std::uint64_t, game::kMaskWords>;

struct ServerAction {
    std::uint16_t id;
    const arena_piece_moves_t* piece;
    const char* destination;
};

struct ServerMoves {
    Mask mask{};
    std::vector<ServerAction> actions;
};

struct Context {
    Context(const char* id, std::uint64_t random_seed)
        : bot_id(id)
        , random(random_seed)
        , seed(random_seed)
    {
    }

    const char* bot_id;
    std::mt19937_64 random;
    std::uint64_t seed;
    game::Board board;
    std::vector<std::uint64_t> history;
    bool started{};
    bool white{true};
    std::size_t compared_turns{};
    std::size_t inferred_opponent_moves{};
    std::size_t normalised_splits{};
    bool completed{};
    bool disconnected{};
};

[[noreturn]] void fail(const Context& context, const std::string& message)
{
    std::fprintf(stderr, "[test %.8s] FAILED: %s\n", context.bot_id, message.c_str());
    std::fflush(stderr);
    std::exit(EXIT_FAILURE);
}

[[nodiscard]] bool is_white(arena_side_t side) noexcept
{
    return side == ARENA_SIDE_WHITE;
}

// ---------------------------------------------------------------------------
// Coordinates and rays
// ---------------------------------------------------------------------------

[[nodiscard]] std::int8_t parse_coord(const char* text)
{
    if (text == nullptr) {
        return -1;
    }
    const std::string_view view(text);
    const std::size_t comma = view.find(',');
    if (comma == std::string_view::npos) {
        return -1;
    }

    int q = 0;
    int r = 0;
    const auto q_result = std::from_chars(view.data(), view.data() + comma, q);
    const auto r_result =
        std::from_chars(view.data() + comma + 1, view.data() + view.size(), r);
    if (q_result.ec != std::errc{} || r_result.ec != std::errc{}) {
        return -1;
    }
    return game::hexIndex(q, r);
}

[[nodiscard]] std::string encode_coord(std::uint8_t hex)
{
    const game::Coordinate coordinate = game::kCoordinates[hex];
    return std::to_string(coordinate.q) + ',' + std::to_string(coordinate.r);
}

struct Ray {
    std::uint8_t dir;
    std::uint8_t steps;
};

[[nodiscard]] std::optional<Ray> find_ray(std::uint8_t from, std::uint8_t to)
{
    for (std::uint8_t dir = 0; dir < game::kNumDirs; ++dir) {
        for (std::uint8_t steps = 1; steps <= game::kMovableMax; ++steps) {
            if (game::ray(from, dir, steps) == static_cast<std::int8_t>(to)) {
                return Ray{dir, steps};
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::string describe(const game::Board& board, game::Move move)
{
    const std::int8_t to = game::ray(move.from, move.dir, board.hexes[move.from].height());
    std::string text = encode_coord(move.from) + " -> ";
    text += to >= 0 ? encode_coord(static_cast<std::uint8_t>(to)) : std::string("off-board");
    text += move.splitting ? " (sow)" : " (whole)";
    return text;
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

[[nodiscard]] std::string board_text(const char* board)
{
    return board == nullptr ? std::string{} : std::string(board);
}

[[nodiscard]] bool same_position(const game::Board& left, const game::Board& right)
{
    if (left.whiteToMove != right.whiteToMove) {
        return false;
    }
    for (std::uint8_t hex = 0; hex < game::kNumHexes; ++hex) {
        if (left.hexes[hex] != right.hexes[hex]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] int piece_count(const game::Board& board)
{
    int total = 0;
    for (std::uint8_t hex = 0; hex < game::kNumHexes; ++hex) {
        total += board.hexes[hex].height();
    }
    return total;
}

// fromString ignores anything it cannot parse, so a format we misread shows up
// as pieces going missing rather than as an error.
[[nodiscard]] game::Board parse_position(const char* board, bool white, const Context& context)
{
    const game::Board parsed = game::fromString(board_text(board), white);
    if (piece_count(parsed) != game::kMaxHeight) {
        fail(
            context,
            "parsed " + std::to_string(piece_count(parsed)) + " pieces, expected "
                + std::to_string(game::kMaxHeight) + "\n  server: " + board_text(board)
                + "\n  local:  " + game::toString(parsed));
    }
    return parsed;
}

// Between two of our turns the opponent plays exactly one move. Replaying it
// locally is what checks that apply() lands split pieces where the server does.
void advance_to(Context& context, const char* board, arena_side_t current_turn)
{
    const game::Board remote = parse_position(board, is_white(current_turn), context);
    if (same_position(context.board, remote)) {
        return;
    }

    std::optional<game::Move> matched;
    game::Board successor;
    context.board.forEachLegal([&](std::uint16_t id) {
        if (matched.has_value()) {
            return;
        }
        const game::Move move = game::Move::fromId(id);
        const game::Board candidate = game::apply(context.board, move, context.history);
        if (same_position(candidate, remote)) {
            matched = move;
            successor = candidate;
        }
    });

    if (!matched.has_value()) {
        fail(
            context,
            "server position is not reachable by any legal local move\n  server: "
                + board_text(board) + "\n  local:  " + game::toString(context.board));
    }

    std::printf(
        "[test %.8s] Opponent move replayed: %s\n",
        context.bot_id,
        describe(context.board, *matched).c_str());
    context.history.push_back(context.board.hash);
    context.board = successor;
    ++context.inferred_opponent_moves;
}

// ---------------------------------------------------------------------------
// Move sets
// ---------------------------------------------------------------------------

void set_bit(Mask& mask, std::uint16_t id)
{
    mask[id >> 6] |= 1ULL << (id & 63);
}

[[nodiscard]] bool test_bit(const Mask& mask, std::uint16_t id)
{
    return ((mask[id >> 6] >> (id & 63)) & 1ULL) != 0;
}

[[nodiscard]] ServerMoves collect_server_moves(
    const arena_game_state_t& state,
    const game::Board& board,
    Context& context)
{
    ServerMoves moves;
    for (std::size_t piece_index = 0; piece_index < state.legal_moves_count; ++piece_index) {
        const arena_piece_moves_t& piece = state.legal_moves[piece_index];
        if (!piece.has_splitting) {
            fail(context, "server Amoeba move is missing its splitting field");
        }

        const std::int8_t from = parse_coord(piece.pos);
        if (from < 0) {
            fail(context, "server sent an unparseable position: " + board_text(piece.pos));
        }
        const std::uint8_t height = board.hexes[from].height();
        if (height == 0) {
            fail(context, "server can move an empty hex: " + board_text(piece.pos));
        }

        // A one-piece stack sows and moves identically, so the engine only ever
        // emits the whole-stack form of it.
        bool splitting = piece.splitting;
        if (splitting && height < 2) {
            splitting = false;
            ++context.normalised_splits;
        }

        for (std::size_t index = 0; index < piece.valid_moves_count; ++index) {
            const char* destination = piece.valid_moves[index];
            const std::int8_t to = parse_coord(destination);
            if (to < 0) {
                fail(context, "server sent an unparseable destination: " + board_text(destination));
            }

            const std::optional<Ray> ray =
                find_ray(static_cast<std::uint8_t>(from), static_cast<std::uint8_t>(to));
            if (!ray.has_value()) {
                fail(
                    context,
                    "server move is not a straight line: " + board_text(piece.pos) + " -> "
                        + board_text(destination));
            }
            if (ray->steps != height) {
                fail(
                    context,
                    "server move distance " + std::to_string(ray->steps) + " does not match stack height "
                        + std::to_string(height) + ": " + board_text(piece.pos) + " -> "
                        + board_text(destination));
            }

            const game::Move move{static_cast<std::uint8_t>(from), ray->dir, splitting};
            moves.actions.push_back({move.id(), &piece, destination});
            set_bit(moves.mask, move.id());
        }
    }
    return moves;
}

void compare_move_sets(const game::Board& board, const ServerMoves& server, const Context& context)
{
    std::string differences;
    for (std::uint16_t id = 0; id < game::kNumMoveIds; ++id) {
        const bool local = board.isLegal(id);
        const bool remote = test_bit(server.mask, id);
        if (local == remote) {
            continue;
        }
        differences += std::string("\n  ") + (local ? "local only:  " : "server only: ")
            + describe(board, game::Move::fromId(id));
    }

    if (!differences.empty()) {
        fail(
            context,
            "legal-move mismatch\n  position: " + game::toString(board) + "\n  side: "
                + (board.whiteToMove ? "white" : "black") + differences);
    }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void on_ready(void* user_data)
{
    const auto& context = *static_cast<Context*>(user_data);
    std::printf("[test %.8s] Connected and ready.\n", context.bot_id);
}

void on_room_joined(const char* room_id, void* user_data)
{
    const auto& context = *static_cast<Context*>(user_data);
    std::printf("[test %.8s] Joined practice room: %.8s...\n", context.bot_id, room_id);
}

void on_game_start(const arena_game_state_t* state, void* user_data)
{
    auto& context = *static_cast<Context*>(user_data);
    if (state == nullptr) {
        fail(context, "SDK passed a null game-start state");
    }

    context.white = is_white(state->my_side);
    context.board = parse_position(state->board, is_white(state->current_turn), context);
    context.history.clear();
    context.started = true;

    // Game start reports this bot's moves even when the opponent is to move.
    const game::Board view = is_white(state->current_turn) == context.white
        ? context.board
        : parse_position(state->board, context.white, context);
    compare_move_sets(view, collect_server_moves(*state, view, context), context);

    std::printf(
        "[test %.8s] Start position matched: %s\n[test %.8s] I am %s; %s moves first.\n",
        context.bot_id,
        game::toString(context.board).c_str(),
        context.bot_id,
        arena_side_str(state->my_side),
        arena_side_str(state->current_turn));
}

void on_move(const arena_game_state_t* state, arena_move_t* output, void* user_data)
{
    auto& context = *static_cast<Context*>(user_data);
    if (state == nullptr || output == nullptr) {
        fail(context, "SDK passed a null move callback argument");
    }
    if (!context.started) {
        fail(context, "SDK requested a move before game start");
    }
    if (is_white(state->current_turn) != context.white) {
        fail(context, "SDK requested a move for the wrong side");
    }

    advance_to(context, state->board, state->current_turn);
    if (context.board.state != game::State::Ongoing) {
        fail(context, "server continued a game the engine considers finished");
    }

    const ServerMoves server = collect_server_moves(*state, context.board, context);
    compare_move_sets(context.board, server, context);
    if (server.actions.empty()) {
        fail(context, "SDK requested a move with no legal actions");
    }
    ++context.compared_turns;

    std::uniform_int_distribution<std::size_t> distribution(0, server.actions.size() - 1);
    const ServerAction& action = server.actions[distribution(context.random)];
    const game::Move move = game::Move::fromId(action.id);

    // The SDK owns these strings for the duration of the callback, and they
    // carry the server's own spelling of the move.
    output->from_pos = action.piece->pos;
    output->to_pos = action.destination;
    output->side = nullptr;
    output->splitting = action.piece->splitting;

    std::printf(
        "[test %.8s] Ply %u: %zu moves matched; sending %s\n",
        context.bot_id,
        context.board.ply,
        server.actions.size(),
        describe(context.board, move).c_str());

    context.history.push_back(context.board.hash);
    context.board = game::apply(context.board, move, context.history);
}

void on_game_end(const arena_game_end_t* state, void* user_data)
{
    auto& context = *static_cast<Context*>(user_data);
    if (state == nullptr) {
        fail(context, "SDK passed a null game-end state");
    }
    if (!context.started) {
        fail(context, "SDK ended the game before game start");
    }

    advance_to(context, state->board, state->current_turn);

    // The engine's move cap and repetition limit are guesses; the ply the server
    // stops at is the evidence for what they really are.
    if (context.board.state == game::State::Ongoing) {
        int repeats = 0;
        for (std::uint64_t past : context.history) {
            if (past == context.board.hash) {
                ++repeats;
            }
        }
        fail(
            context,
            "server ended the game at ply " + std::to_string(context.board.ply)
                + " while the engine considers it ongoing (position seen "
                + std::to_string(repeats + 1) + " times; kMoveCap=" + std::to_string(game::kMoveCap)
                + ", kRepetitionLimit=" + std::to_string(game::kRepetitionLimit) + ")");
    }

    const bool local_draw = context.board.state == game::State::Draw;
    if (state->has_winner == local_draw) {
        fail(context, "winner mismatch: one implementation reports a draw");
    }
    if (state->has_winner) {
        const bool local_white_wins = context.board.state == game::State::WhiteWins;
        if (is_white(state->winner) != local_white_wins) {
            fail(
                context,
                "winner mismatch: server=" + std::string(arena_side_str(state->winner))
                    + ", local=" + (local_white_wins ? "white" : "black"));
        }
    }

    context.completed = true;
    std::printf(
        "[test %.8s] PASSED after %u plies: %zu turns compared, %zu opponent moves replayed",
        context.bot_id,
        context.board.ply,
        context.compared_turns,
        context.inferred_opponent_moves);
    if (context.normalised_splits > 0) {
        std::printf(", %zu height-1 sows normalised", context.normalised_splits);
    }
    std::printf(".\n");
}

void on_disconnect(const char* reason, void* user_data)
{
    auto& context = *static_cast<Context*>(user_data);
    context.disconnected = true;
    std::fprintf(
        stderr,
        "[test %.8s] Disconnected before a verified result: %s\n",
        context.bot_id,
        reason == nullptr ? "unknown reason" : reason);
}

[[nodiscard]] std::uint64_t parse_seed(std::string_view value)
{
    std::uint64_t parsed{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid --seed value");
    }
    return parsed;
}

[[nodiscard]] std::uint64_t parse_options(int argc, char** argv)
{
    std::uint64_t seed = 1;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--seed" && index + 1 < argc) {
            seed = parse_seed(argv[++index]);
        } else if (argument == "--help") {
            std::printf(
                "Usage: amoeba_random_test [--seed N]\n"
                "Requires BOT1_ID, BOT1_KEY, and ROOM_ID. Runs in practice mode only.\n");
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown or incomplete option: " + std::string(argument));
        }
    }
    return seed;
}

} // namespace

int main(int argc, char** argv)
try {
    const std::uint64_t seed = parse_options(argc, argv);
    const char* bot_id = std::getenv("BOT1_ID");
    const char* api_key = std::getenv("BOT1_KEY");
    if (bot_id == nullptr || api_key == nullptr || std::getenv("ROOM_ID") == nullptr) {
        std::fprintf(stderr, "Error: set BOT1_ID, BOT1_KEY, and ROOM_ID.\n");
        return EXIT_FAILURE;
    }

    Context context(bot_id, seed);
    const arena_bot_config_t config{
        .bot_id = bot_id,
        .api_key = api_key,
        .callbacks = {
            .on_move = on_move,
            .on_ready = on_ready,
            .on_queue_entry = nullptr,
            .on_queue_exit = nullptr,
            .on_match_found = nullptr,
            .on_room_joined = on_room_joined,
            .on_game_start = on_game_start,
            .on_game_end = on_game_end,
            .on_disconnect = on_disconnect,
        },
        .user_data = &context,
    };

    std::printf(
        "[test %.8s] Starting Amoeba practice comparison with seed %llu.\n",
        bot_id,
        static_cast<unsigned long long>(context.seed));

    const int result = arena_start_practice(&config, nullptr);
    if (result != 0) {
        std::fprintf(stderr, "[test %.8s] SDK exited with error code %d.\n", bot_id, result);
        return result;
    }
    if (context.disconnected || !context.completed) {
        std::fprintf(stderr, "[test %.8s] No verified complete game.\n", bot_id);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::fprintf(stderr, "Error: %s\n", error.what());
    return EXIT_FAILURE;
}
