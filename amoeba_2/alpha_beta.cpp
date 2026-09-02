#include "alpha_beta.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace amoeba_bot
{

namespace
{

constexpr int winScore = 1'000'000;
constexpr int infinity = 2 * winScore;
constexpr int maximumDepth = 32;
constexpr int maximumTacticalExtensions = 2;
constexpr uint16_t lowMobilityLimit = 2;

enum class Bound : uint8_t { exact, lower, upper };

struct TranspositionEntry
{
    int score;
    int depth;
    uint16_t bestMove;
    Bound bound;
};

constexpr uint64_t mix(uint64_t value)
{
    value ^= value >> 30;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Repetition depends on how often previous positions occurred.  XOR alone
// would erase an even number of identical hashes, so pair it with a sum.
struct HistorySignature
{
    uint64_t xorValue{};
    uint64_t sumValue{};
};

constexpr HistorySignature appendHistory(HistorySignature signature, uint64_t positionHash)
{
    const uint64_t contribution = mix(positionHash + 0x9E3779B97F4A7C15ULL);
    signature.xorValue ^= contribution;
    signature.sumValue += contribution;
    return signature;
}

bool isWhite(Piece piece)
{
    return piece == Piece::white || piece == Piece::whiteKernel;
}

// Positive means White is better.  Stack control and prisoners mirror the
// official adjudication rule; mobility rewards keeping stacks usable.
int scoreForWhite(const Board& board)
{
    int score = 0;
    for (uint8_t hex = 0; hex < hexCount; ++hex)
    {
        const Hex& stack = board.hexes[hex];
        const Piece top = stack.topPiece();
        if (top == Piece::empty)
            continue;

        const int sign = isWhite(top) ? 1 : -1;
        score += sign * 120;
        for (uint8_t depth = 0; depth < stack.height(); ++depth)
        {
            const Piece piece = stack.pieceAt(depth);
            if (isWhite(piece) != isWhite(top))
                score += sign * 24;
        }

        // A stack above six is frozen by the board geometry.  Smaller stacks
        // remain potential sources of legal moves and tactical threats.
        if (stack.height() <= maximumMovableStackHeight)
            score += sign * 5;
    }

    const uint8_t ownKernel = board.whiteToMove ? board.whiteKernelIndex : board.blackKernelIndex;
    const uint8_t opponentKernel = board.whiteToMove ? board.blackKernelIndex : board.whiteKernelIndex;
    score += static_cast<int>(board.hexes[ownKernel].height()) * (board.whiteToMove ? 4 : -4);
    score += static_cast<int>(board.hexes[opponentKernel].height()) * (board.whiteToMove ? -4 : 4);
    return score;
}

int evaluate(const Board& board)
{
    const int positional = board.whiteToMove ? scoreForWhite(board) : -scoreForWhite(board);
    return positional + static_cast<int>(board.legalMoveCount) * 8;
}

int outcomeScore(Outcome outcome, bool sideToMoveIsWhite, int ply)
{
    if (outcome == Outcome::draw)
        return 0;
    const bool sideToMoveWins = (outcome == Outcome::whiteWins) == sideToMoveIsWhite;
    return sideToMoveWins ? winScore - ply : -winScore + ply;
}

class AlphaBetaSearch
{
public:
    AlphaBetaSearch(const Board& root, std::span<const uint64_t> history,
                    std::chrono::steady_clock::time_point deadline)
        : m_deadline(deadline)
    {
        assert(!history.empty());
        assert(history.size() <= m_history.size());
        std::copy(history.begin(), history.end(), m_history.begin());
        m_historySize = history.size();
        for (uint64_t hash : history)
            m_historySignature = appendHistory(m_historySignature, hash);
        m_table.reserve(1 << 20);
        root.forEachLegal([&](uint16_t moveId) { m_fallbackMove = moveId; });
    }

    SearchResult run(const Board& root)
    {
        SearchResult result{.moveId = m_fallbackMove, .completedDepth = 0, .nodes = 0,
                            .score = 0, .transpositionHits = 0, .betaCutoffs = 0,
                            .tacticalExtensions = 0};
        for (int depth = 1; depth <= maximumDepth; ++depth)
        {
            m_stopped = false;
            if (deadlineReached())
                break;
            const RootResult iteration = searchRoot(root, depth);
            if (m_stopped)
                break;
            result = {.moveId = iteration.moveId, .completedDepth = depth,
                      .nodes = m_nodes, .score = iteration.score,
                      .transpositionHits = m_transpositionHits,
                      .betaCutoffs = m_betaCutoffs,
                      .tacticalExtensions = m_tacticalExtensions};
            m_previousBestMove = iteration.moveId;
            if (std::abs(iteration.score) >= winScore - maximumDepth)
                break;
        }
        // The move/score must come from the last complete iteration, but the
        // node and pruning counters should include useful work in the final
        // interrupted iteration as well.
        result.nodes = m_nodes;
        result.transpositionHits = m_transpositionHits;
        result.betaCutoffs = m_betaCutoffs;
        result.tacticalExtensions = m_tacticalExtensions;
        return result;
    }

private:
    struct RootResult { uint16_t moveId; int score; };

    bool outOfTime()
    {
        if ((++m_nodes & 0x3FF) != 0)
            return m_stopped;
        m_stopped = std::chrono::steady_clock::now() >= m_deadline;
        return m_stopped;
    }

    bool deadlineReached()
    {
        if (std::chrono::steady_clock::now() < m_deadline)
            return false;
        m_stopped = true;
        return true;
    }

    uint64_t key(const Board& board, HistorySignature historySignature, int extensionsRemaining) const
    {
        return board.positionHash ^ mix(historySignature.xorValue) ^ mix(historySignature.sumValue)
            ^ mix((static_cast<uint64_t>(board.plyCount) << 16) | board.stalenessCount)
            ^ mix(static_cast<uint64_t>(extensionsRemaining));
    }

    struct OrderedMove
    {
        uint16_t moveId;
        int order;
        std::optional<MoveResult> result;
    };

    int staticMoveOrder(const Board& board, Move move) const
    {
        const uint8_t height = board.hexes[move.sourceCoord].height();
        const uint8_t destination = destinationHex(
            move.sourceCoord, directions[move.direction], height).value();
        const Piece destinationTop = board.hexes[destination].topPiece();
        int order = 0;
        const uint8_t opponentKernel = board.whiteToMove ? board.blackKernelIndex : board.whiteKernelIndex;
        if (destination == opponentKernel)
            order += 900'000;
        if (destinationTop != Piece::empty && isWhite(destinationTop) != board.whiteToMove)
            order += 600;

        if (move.splitsStack)
        {
            bool pure = true;
            for (uint8_t depth = 0; depth < height; ++depth)
                pure &= isWhite(board.hexes[move.sourceCoord].pieceAt(depth)) == board.whiteToMove;
            if (pure)
                order += 80;
        }
        return order;
    }

    // Exact reply-count ordering is valuable at the top of the tree.  Further
    // down it costs a full move generation for branches alpha-beta may prune,
    // so use only the cheap structural ordering there.
    const std::vector<OrderedMove>& orderedMoves(const Board& board, std::optional<uint16_t> preferred,
                                                 int ply, bool exactTacticalOrdering)
    {
        assert(ply >= 0 && static_cast<size_t>(ply) < m_moveBuffers.size());
        std::vector<OrderedMove>& result = m_moveBuffers[ply];
        result.clear();
        if (result.capacity() < board.legalMoveCount)
            result.reserve(board.legalMoveCount);
        board.forEachLegal([&](uint16_t moveId) {
            // Ordering evaluates every candidate, so it needs its own deadline
            // check rather than waiting for negamax's periodic node check.
            if (m_stopped || deadlineReached())
                return;
            const Move move = Move::fromId(moveId);
            int order = staticMoveOrder(board, move);
            std::optional<MoveResult> moveResult;
            if (exactTacticalOrdering)
            {
                moveResult.emplace(applyMove(
                    board, move, std::span{m_history.data(), m_historySize}));
                if (const auto* outcome = std::get_if<Outcome>(&*moveResult))
                    order += outcomeScore(*outcome, board.whiteToMove, 0) > 0 ? 1'000'000 : -1'000'000;
                else
                {
                    const Board& child = std::get<Board>(*moveResult);
                    // The child belongs to the opponent.  Fewer replies and a
                    // large swing in the current side's static score are tactics.
                    order += static_cast<int>(board.legalMoveCount - child.legalMoveCount) * 12;
                    order += (16 - std::min(16, static_cast<int>(child.legalMoveCount))) * 48;
                    order += -evaluate(child) / 16;
                }
            }
            if (preferred == moveId)
                order += 2'000'000;
            result.push_back({moveId, order, std::move(moveResult)});
        });
        std::ranges::stable_sort(result, std::greater{}, &OrderedMove::order);
        return result;
    }

    RootResult searchRoot(const Board& root, int depth)
    {
        int alpha = -infinity;
        int beta = infinity;
        uint16_t bestMove = m_previousBestMove.value_or(m_fallbackMove);
        int bestScore = -infinity;
        const std::vector<OrderedMove>& candidates = orderedMoves(root, m_previousBestMove, 0, true);
        if (m_stopped)
            return {bestMove, bestScore};
        for (const OrderedMove& candidate : candidates)
        {
            int score;
            assert(candidate.result.has_value());
            if (const auto* outcome = std::get_if<Outcome>(&*candidate.result))
                score = outcomeScore(*outcome, root.whiteToMove, 0);
            else
            {
                const Board& child = std::get<Board>(*candidate.result);
                assert(m_historySize < m_history.size());
                m_history[m_historySize++] = child.positionHash;
                const HistorySignature childSignature = appendHistory(m_historySignature, child.positionHash);
                score = -negamax(child, depth - 1, -beta, -alpha, 1, childSignature,
                                 maximumTacticalExtensions);
                --m_historySize;
            }
            if (m_stopped)
                return {bestMove, bestScore};
            if (score > bestScore)
            {
                bestScore = score;
                bestMove = candidate.moveId;
            }
            alpha = std::max(alpha, score);
        }
        return {bestMove, bestScore};
    }

    int negamax(const Board& board, int depth, int alpha, int beta, int ply,
                HistorySignature historySignature, int extensionsRemaining)
    {
        if (outOfTime())
            return 0;
        if (depth == 0)
        {
            if (extensionsRemaining == 0 || board.legalMoveCount > lowMobilityLimit)
                return evaluate(board);
            depth = 1;
            --extensionsRemaining;
            ++m_tacticalExtensions;
        }

        const uint64_t positionKey = key(board, historySignature, extensionsRemaining);
        std::optional<uint16_t> preferred;
        const auto found = m_table.find(positionKey);
        if (found != m_table.end())
        {
            ++m_transpositionHits;
            preferred = found->second.bestMove;
            if (found->second.depth >= depth)
            {
                if (found->second.bound == Bound::exact)
                    return found->second.score;
                if (found->second.bound == Bound::lower)
                    alpha = std::max(alpha, found->second.score);
                else
                    beta = std::min(beta, found->second.score);
                if (alpha >= beta)
                {
                    ++m_betaCutoffs;
                    return found->second.score;
                }
            }
        }

        const int originalAlpha = alpha;
        const int originalBeta = beta;
        int bestScore = -infinity;
        uint16_t bestMove = 0;
        const std::vector<OrderedMove>& candidates = orderedMoves(
            board, preferred, ply, ply <= 1);
        if (m_stopped)
            return 0;
        for (const OrderedMove& candidate : candidates)
        {
            MoveResult searchedResult;
            const MoveResult* moveResult = candidate.result ? &*candidate.result : nullptr;
            if (moveResult == nullptr)
            {
                searchedResult = applyMove(board, Move::fromId(candidate.moveId),
                    std::span{m_history.data(), m_historySize});
                moveResult = &searchedResult;
            }
            int score;
            if (const auto* outcome = std::get_if<Outcome>(moveResult))
                score = outcomeScore(*outcome, board.whiteToMove, ply);
            else
            {
                const Board& child = std::get<Board>(*moveResult);
                assert(m_historySize < m_history.size());
                m_history[m_historySize++] = child.positionHash;
                const HistorySignature childSignature = appendHistory(historySignature, child.positionHash);
                score = -negamax(child, depth - 1, -beta, -alpha, ply + 1, childSignature,
                                 extensionsRemaining);
                --m_historySize;
            }
            if (m_stopped)
                return 0;
            if (score > bestScore)
            {
                bestScore = score;
                bestMove = candidate.moveId;
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta)
            {
                ++m_betaCutoffs;
                break;
            }
        }

        const Bound bound = bestScore <= originalAlpha ? Bound::upper
            : bestScore >= originalBeta ? Bound::lower : Bound::exact;
        m_table[positionKey] = {bestScore, depth, bestMove, bound};
        return bestScore;
    }

    std::array<uint64_t, moveLimit + 1> m_history{};
    size_t m_historySize{};
    HistorySignature m_historySignature{};
    std::chrono::steady_clock::time_point m_deadline;
    std::unordered_map<uint64_t, TranspositionEntry> m_table;
    // A recursive child only needs a distinct list from its ancestors.  Keeping
    // these buffers by ply lets iterative deepening reuse their allocations.
    std::array<std::vector<OrderedMove>, maximumDepth + maximumTacticalExtensions + 1> m_moveBuffers;
    std::optional<uint16_t> m_previousBestMove;
    uint16_t m_fallbackMove{};
    uint64_t m_nodes{};
    uint64_t m_transpositionHits{};
    uint64_t m_betaCutoffs{};
    uint64_t m_tacticalExtensions{};
    bool m_stopped{};
};

} // namespace

SearchResult chooseAlphaBetaMove(const Board& root, std::span<const uint64_t> history,
                                 std::chrono::steady_clock::time_point deadline)
{
    assert(root.legalMoveCount > 0);
    AlphaBetaSearch search{root, history, deadline};
    return search.run(root);
}

} // namespace amoeba_bot
