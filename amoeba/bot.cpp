// amoeba_bot: plays Amoeba on Game Arena with a trained network, and does
// nothing else. Training lives in amoeba_train; the two programs only ever meet
// in a .safetensors file, so either can be killed and restarted without the
// other noticing.
//
//   amoeba_bot [weights.safetensors]
//
// With no argument it takes the first .safetensors file in the working
// directory - the same one amoeba_train writes to. Credentials come from
// BOT1_ID and BOT1_KEY; ROOM_ID picks a practice room, and without it the bot
// queues for ranked games back to back.
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
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <vector>

namespace bot
{

namespace
{

// Every search runs to the full simulation count - there is no clock on it, so
// the move logged is always the one 800 simulations chose and the elapsed time
// is a measurement rather than a limit. The server allows 5 s a move, and
// nothing here enforces that.
//
// Leaves are batched 16 at a time because the network costs 1.70 ms for a single
// position and 0.16 ms each for 64 - one position per forward pass would spend
// the whole turn on overhead.
constexpr int kSimulations = 800;
constexpr int kLeaves      = 16;

// A C++ exception unwinding through the SDK's C frames is undefined behaviour,
// so nothing may leave a callback. A turn that goes unanswered times out and
// loses one game; a crash here loses every game that would have followed.
template <typename Fn>
void guarded(const char* callback, Fn&& body)
{
    try
    {
        body();
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "[bot] {} threw: {}", callback, error.what());
    }
    catch (...)
    {
        std::println(stderr, "[bot] {} threw something that is not an exception", callback);
    }
}

struct ServerMove
{
    amoeba::Move move;
    const char*  from;
    const char*  to;
    bool         splitting;   // the server's own per-piece flag, echoed verbatim
};

// The model is loaded once per game and held for the whole of it. Reloading it
// mid-game would mean one tree scoring its positions with two different
// networks, which is incoherent and would not show up in any log; between games
// is the only safe moment, and it is how a promotion by a trainer running
// alongside gets picked up.
struct Context
{
    std::filesystem::path             weights;
    std::unique_ptr<Network>          network;
    std::unique_ptr<NetworkEvaluator> evaluator;
    std::optional<Search>             search;
    amoeba::Board                     board;
    std::vector<uint64_t>             history;
};

// ---------------------------------------------------------------------------
// Translation between the server's strings and the engine's move ids
// ---------------------------------------------------------------------------

int8_t parseHex(std::string_view text)
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
    return amoeba::hexIndex(q, r);
}

// The engine indexes a move by direction, the server names its destination.
std::optional<uint8_t> findDirection(uint8_t from, uint8_t to, uint8_t steps)
{
    for (uint8_t dir = 0; dir < amoeba::kNumDirs; ++dir)
        if (amoeba::ray(from, dir, steps) == static_cast<int8_t>(to))
            return dir;
    return std::nullopt;
}

std::vector<ServerMove> collectServerMoves(const arena_game_state_t& state, const amoeba::Board& board)
{
    std::vector<ServerMove> moves;
    for (const arena_piece_moves_t& piece : std::span(state.legal_moves, state.legal_moves_count))
    {
        const int8_t from = parseHex(piece.pos);
        if (from < 0)
            continue;

        const uint8_t height = board.hexes[from].height();
        if (height == 0)
            continue;

        // A one-piece stack sows and moves identically, so the engine only ever
        // emits the whole-stack form of it.
        const bool splitting = piece.splitting && height >= 2;

        for (const char* destination : std::span(piece.valid_moves, piece.valid_moves_count))
        {
            const int8_t to = parseHex(destination);
            if (to < 0)
                continue;

            const std::optional<uint8_t> dir =
                findDirection(static_cast<uint8_t>(from), static_cast<uint8_t>(to), height);
            if (!dir.has_value())
                continue;

            moves.push_back({amoeba::Move{static_cast<uint8_t>(from), *dir, splitting},
                             piece.pos,
                             destination,
                             piece.splitting});
        }
    }
    return moves;
}

// ---------------------------------------------------------------------------
// Keeping the local board in step with the server's
// ---------------------------------------------------------------------------

bool samePosition(const amoeba::Board& left, const amoeba::Board& right)
{
    return left.whiteToMove == right.whiteToMove && std::ranges::equal(left.hexes, right.hexes);
}

void advance(Context& context, amoeba::Move move)
{
    context.board = amoeba::apply(context.board, move, context.history);
    context.history.push_back(context.board.hash);
}

void syncToServer(Context& context, const char* serverBoard, arena_side_t currentTurn)
{
    const amoeba::Board remote = amoeba::fromString(serverBoard, currentTurn == ARENA_SIDE_WHITE);
    if (samePosition(context.board, remote))
        return;

    // Between two of our turns the opponent plays exactly one move; find which.
    std::optional<amoeba::Move> played;
    context.board.forEachLegal([&](uint16_t id) {
        if (played.has_value())
            return;
        const amoeba::Move move = amoeba::Move::fromId(id);
        if (samePosition(amoeba::applyRaw(context.board, move), remote))
            played = move;
    });

    if (played.has_value())
    {
        advance(context, *played);
        return;
    }

    // The engine and the server disagree about what some move does. Adopting the
    // server's position costs the staleness and repetition counters, but that
    // beats playing out the rest of the game from a board that never existed.
    std::println(stderr, "[bot] resync at ply {}: no legal move reaches {}", context.board.ply, serverBoard);
    const uint16_t ply = context.board.ply;
    context.board     = remote;
    context.board.ply = static_cast<uint16_t>(ply + 1);
    context.history.assign(1, context.board.hash);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

// The clock starts when the callback does, not when the search does: the server
// times the whole reply, so sync and translation count against the budget too.
const ServerMove& chooseMove(Context& context, const std::vector<ServerMove>& moves,
                             std::chrono::steady_clock::time_point turnStart)
{
    if (context.board.state != amoeba::State::Ongoing)
    {
        std::println(stderr, "[bot] engine calls the game over at ply {}, deferring to the server",
                     context.board.ply);
        return moves.front();
    }

    const VisitCounts counts = context.search->run(context.board, context.history);
    const uint16_t    chosen = bestMove(counts);

    const auto found = std::ranges::find_if(moves, [chosen](const ServerMove& m) { return m.move.id() == chosen; });
    if (found == moves.end())
    {
        std::println(stderr, "[bot] search picked a move the server did not offer, falling back");
        return moves.front();
    }

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - turnStart;
    const uint32_t simulations = std::accumulate(counts.begin(), counts.end(), uint32_t{0});
    std::println("[bot] ply {}: {} -> {}{}  {}/{} visits over {} moves in {:.0f} ms ({:.2f} s)",
                 context.board.ply, found->from, found->to, found->move.splitting ? " sow" : "",
                 counts[chosen], simulations, context.board.moveCount, elapsed.count(),
                 elapsed.count() / 1000.0);
    return *found;
}

void loadNetwork(Context& context)
{
    context.network = std::make_unique<Network>(context.weights);
    context.evaluator = std::make_unique<NetworkEvaluator>(*context.network);
    context.search.emplace(*context.evaluator,
                           Config{.simulations = kSimulations, .batchSize = kLeaves});

    std::println("[bot] {}: {} blocks, width {}, {} heads, {} parameters",
                 context.weights.filename().string(), context.network->shape().blocks,
                 context.network->shape().width, context.network->shape().heads,
                 context.network->parameterCount());
}

void onGameStart(const arena_game_state_t* state, void* userData)
{
    guarded("on_game_start", [&] {
        Context& context = *static_cast<Context*>(userData);
        loadNetwork(context);

        context.board = amoeba::fromString(state->board, state->current_turn == ARENA_SIDE_WHITE);
        context.history.assign(1, context.board.hash);

        std::println("[bot] game start, I am {}, {} to move", arena_side_str(state->my_side), arena_side_str(state->current_turn));
    });
}

void onMove(const arena_game_state_t* state, arena_move_t* output, void* userData)
{
    guarded("on_move", [&] {
        const std::chrono::steady_clock::time_point turnStart = std::chrono::steady_clock::now();

        Context& context = *static_cast<Context*>(userData);
        if (!context.search.has_value())
        {
            std::println(stderr, "[bot] no model loaded, cannot move");
            return;
        }

        syncToServer(context, state->board, state->current_turn);

        const std::vector<ServerMove> moves = collectServerMoves(*state, context.board);
        if (moves.empty())
        {
            std::println(stderr, "[bot] no usable move in the server's list for {}", state->board);
            return;
        }

        const ServerMove& chosen = chooseMove(context, moves, turnStart);

        // The SDK owns these strings for the duration of the callback, and they
        // carry the server's own spelling of the move.
        output->from_pos  = chosen.from;
        output->to_pos    = chosen.to;
        output->side      = nullptr;
        output->splitting = chosen.splitting;

        advance(context, chosen.move);
    });
}

void onGameEnd(const arena_game_end_t* state, void* userData)
{
    guarded("on_game_end", [&] {
        const Context& context = *static_cast<Context*>(userData);
        const char*    result  = !state->has_winner ? "draw" : state->winner == state->my_side ? "won" : "lost";
        std::println("[bot] {} after {} plies", result, context.board.ply);
    });
}

void onDisconnect(const char* reason, void*)
{
    std::println(stderr, "[bot] disconnected: {}", reason == nullptr ? "unknown reason" : reason);
}

// The trainer writes one file and keeps writing to it, so with no argument
// there is normally exactly one candidate in the directory.
std::filesystem::path firstCheckpointHere()
{
    std::vector<std::filesystem::path> found;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator("."))
        if (entry.path().extension() == ".safetensors")
            found.push_back(entry.path());

    if (found.empty())
        throw std::runtime_error("no .safetensors file here - name one, or run amoeba_train first");

    std::ranges::sort(found);
    return found.front();
}

} // namespace

} // namespace bot

int main(int argc, char** argv)
{
    const char* botId  = std::getenv("BOT1_ID");
    const char* apiKey = std::getenv("BOT1_KEY");
    if (botId == nullptr || apiKey == nullptr)
    {
        std::println(stderr, "set BOT1_ID and BOT1_KEY");
        return EXIT_FAILURE;
    }

    bot::Context context;
    try
    {
        context.weights = argc > 1 ? std::filesystem::path{argv[1]} : bot::firstCheckpointHere();
        bot::loadNetwork(context);
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "[bot] {}", error.what());
        return EXIT_FAILURE;
    }

    const arena_bot_config_t config{
        .bot_id    = botId,
        .api_key   = apiKey,
        .callbacks = {
            .on_move       = bot::onMove,
            .on_game_start = bot::onGameStart,
            .on_game_end   = bot::onGameEnd,
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
