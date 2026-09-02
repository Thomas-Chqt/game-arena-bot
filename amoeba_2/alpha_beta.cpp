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
            m_historySignature ^= mix(hash + 0x9E3779B97F4A7C15ULL);
        m_table.reserve(1 << 20);
        root.forEachLegal([&](uint16_t moveId) { m_fallbackMove = moveId; });
    }

    SearchResult run(const Board& root)
    {
        SearchResult result{.moveId = m_fallbackMove, .completedDepth = 0, .nodes = 0, .score = 0};
        for (int depth = 1; depth <= maximumDepth; ++depth)
        {
            m_stopped = false;
            const RootResult iteration = searchRoot(root, depth);
            if (m_stopped)
                break;
            result = {.moveId = iteration.moveId, .completedDepth = depth,
                      .nodes = m_nodes, .score = iteration.score};
            m_previousBestMove = iteration.moveId;
            if (std::abs(iteration.score) >= winScore - maximumDepth)
                break;
        }
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

    uint64_t key(const Board& board, uint64_t historySignature) const
    {
        return board.positionHash ^ mix(historySignature)
            ^ mix((static_cast<uint64_t>(board.plyCount) << 16) | board.stalenessCount);
    }

    std::vector<uint16_t> moves(const Board& board, std::optional<uint16_t> preferred) const
    {
        std::vector<uint16_t> result;
        result.reserve(board.legalMoveCount);
        board.forEachLegal([&](uint16_t moveId) { result.push_back(moveId); });
        if (preferred.has_value())
        {
            const auto found = std::find(result.begin(), result.end(), *preferred);
            if (found != result.end())
                std::iter_swap(result.begin(), found);
        }
        return result;
    }

    RootResult searchRoot(const Board& root, int depth)
    {
        int alpha = -infinity;
        int beta = infinity;
        uint16_t bestMove = m_previousBestMove.value_or(m_fallbackMove);
        int bestScore = -infinity;
        for (uint16_t moveId : moves(root, m_previousBestMove))
        {
            const MoveResult moveResult = applyMove(root, Move::fromId(moveId),
                                                     std::span{m_history.data(), m_historySize});
            int score;
            if (const auto* outcome = std::get_if<Outcome>(&moveResult))
                score = outcomeScore(*outcome, root.whiteToMove, 0);
            else
            {
                const Board& child = std::get<Board>(moveResult);
                assert(m_historySize < m_history.size());
                m_history[m_historySize++] = child.positionHash;
                const uint64_t childSignature = m_historySignature ^ mix(child.positionHash + 0x9E3779B97F4A7C15ULL);
                score = -negamax(child, depth - 1, -beta, -alpha, 1, childSignature);
                --m_historySize;
            }
            if (m_stopped)
                return {bestMove, bestScore};
            if (score > bestScore)
            {
                bestScore = score;
                bestMove = moveId;
            }
            alpha = std::max(alpha, score);
        }
        return {bestMove, bestScore};
    }

    int negamax(const Board& board, int depth, int alpha, int beta, int ply,
                uint64_t historySignature)
    {
        if (outOfTime())
            return 0;
        if (depth == 0)
            return evaluate(board);

        const uint64_t positionKey = key(board, historySignature);
        std::optional<uint16_t> preferred;
        const auto found = m_table.find(positionKey);
        if (found != m_table.end())
        {
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
                    return found->second.score;
            }
        }

        const int originalAlpha = alpha;
        const int originalBeta = beta;
        int bestScore = -infinity;
        uint16_t bestMove = 0;
        for (uint16_t moveId : moves(board, preferred))
        {
            const MoveResult moveResult = applyMove(board, Move::fromId(moveId),
                                                     std::span{m_history.data(), m_historySize});
            int score;
            if (const auto* outcome = std::get_if<Outcome>(&moveResult))
                score = outcomeScore(*outcome, board.whiteToMove, ply);
            else
            {
                const Board& child = std::get<Board>(moveResult);
                assert(m_historySize < m_history.size());
                m_history[m_historySize++] = child.positionHash;
                const uint64_t childSignature = historySignature ^ mix(child.positionHash + 0x9E3779B97F4A7C15ULL);
                score = -negamax(child, depth - 1, -beta, -alpha, ply + 1, childSignature);
                --m_historySize;
            }
            if (m_stopped)
                return 0;
            if (score > bestScore)
            {
                bestScore = score;
                bestMove = moveId;
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta)
                break;
        }

        const Bound bound = bestScore <= originalAlpha ? Bound::upper
            : bestScore >= originalBeta ? Bound::lower : Bound::exact;
        m_table[positionKey] = {bestScore, depth, bestMove, bound};
        return bestScore;
    }

    std::array<uint64_t, moveLimit + 1> m_history{};
    size_t m_historySize{};
    uint64_t m_historySignature{};
    std::chrono::steady_clock::time_point m_deadline;
    std::unordered_map<uint64_t, TranspositionEntry> m_table;
    std::optional<uint16_t> m_previousBestMove;
    uint16_t m_fallbackMove{};
    uint64_t m_nodes{};
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
