#include "mcts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <ranges>

namespace amoeba_bot
{

namespace
{

// Seen by whoever is to move in it - which, at a terminal position, is the side
// that was just mated or ran out of moves.
float terminalValue(const Board& board)
{
    assert(board.state != State::ongoing);
    return outcomeFor(board.state, board.whiteToMove);
}

} // namespace

// ---------------------------------------------------------------------------

float outcomeFor(State state, bool whiteToMove)
{
    assert(state != State::ongoing);
    if (state == State::draw)
        return 0.0f;
    return (state == State::whiteWins) == whiteToMove ? 1.0f : -1.0f;
}

uint16_t bestMove(const VisitCounts& counts)
{
    return static_cast<uint16_t>(std::ranges::max_element(counts) - counts.begin());
}

// ---------------------------------------------------------------------------

// Appends the node and its outgoing edges from an evaluation already in hand. A
// terminal board gets a node with no edges and its Evaluation is ignored.
uint32_t MCTS::addNode(const Board& board, const Evaluation& evaluation)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{board, static_cast<uint32_t>(m_edges.size()), 0, 0});

    if (board.state != State::ongoing)
        return index;

    float totalPrior = 0.0f;
    board.forEachLegal([&](uint16_t moveId) { totalPrior += evaluation.policy[moveId]; });
    assert(totalPrior > 0.0f);
    board.forEachLegal([&](uint16_t moveId)
                       { m_edges.push_back(Edge{evaluation.policy[moveId] / totalPrior, 0.0f, 0, -1, moveId}); });

    m_nodes[index].edgeCount = board.legalMoveCount;
    return index;
}

uint32_t MCTS::selectEdgeToExplore(uint32_t node) const
{
    const Node& currentNode = m_nodes[node];

    // currentNode.visits was already incremented for this simulation, so the bonus is
    // non-zero on a node's first descent and the priors alone decide.
    const float exploration = m_config.explorationConstant * std::sqrt(static_cast<float>(currentNode.visits));

    const auto score = [&](uint32_t edgeIndex)
    {
        const Edge& edge = m_edges[edgeIndex];
        const float meanValue = edge.visits > 0 ? edge.valueSum / static_cast<float>(edge.visits) : 0.0f;
        return meanValue + exploration * edge.prior / static_cast<float>(1 + edge.visits);
    };

    return *std::ranges::max_element(
        std::views::iota(currentNode.firstEdgeIndex, currentNode.firstEdgeIndex + currentNode.edgeCount), {}, score);
}

// A Dirichlet(alpha, ..., alpha) draw is independent Gamma(alpha, 1) draws
// normalised to sum to one.
void MCTS::addExplorationNoise()
{
    const Node& root = m_nodes[0];
    if (root.edgeCount == 0)
        return;

    std::mt19937_64 randomEngine{m_config.noiseSeed};
    std::gamma_distribution<float> gamma{m_config.noiseAlpha, 1.0f};
    std::array<float, moveIdCount> noiseValues;
    float total = 0.0f;
    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        noiseValues[i] = gamma(randomEngine);
        total += noiseValues[i];
    }

    // alpha below 1 puts most draws near zero, so a whole set underflowing is
    // unlikely but not impossible; leaving the priors alone is the honest answer.
    if (total <= 0.0f)
        return;

    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        Edge& edge = m_edges[root.firstEdgeIndex + i];
        edge.prior = (1.0f - m_config.rootNoise) * edge.prior + m_config.rootNoise * (noiseValues[i] / total);
    }
}

// Performs one simulation. Returning true means the descent reached an ongoing
// leaf and needs the caller to evaluate it. Terminal leaves already have an exact
// value, so they are expanded and backed up here.
bool MCTS::descend()
{
    std::array<uint64_t, moveLimit + 1> positionHistory;
    std::ranges::copy_n(m_gameHistory.begin(), m_gameHistorySize, positionHistory.begin());
    size_t positionHistorySize = m_gameHistorySize;
    m_edgeTrailSize = 0;

    uint32_t node = 0;
    for (;;)
    {
        ++m_nodes[node].visits;

        if (m_nodes[node].edgeCount == 0)
        {
            backpropagate(terminalValue(m_nodes[node].board));
            return false;
        }

        const uint32_t edgeIndex = selectEdgeToExplore(node);
        assert(m_edgeTrailSize < m_edgeTrail.size());
        m_edgeTrail[m_edgeTrailSize++] = edgeIndex;

        if (m_edges[edgeIndex].childNode < 0)
        {
            const Move move = Move::fromId(m_edges[edgeIndex].moveId);
            m_pendingLeaf = applyMove(
                m_nodes[node].board, move, std::span{positionHistory.data(), positionHistorySize});

            if (m_pendingLeaf->state == State::ongoing)
                return true;

            m_edges[edgeIndex].childNode = static_cast<int32_t>(addNode(*m_pendingLeaf, Evaluation{}));
            backpropagate(terminalValue(*m_pendingLeaf));
            m_pendingLeaf.reset();
            return false;
        }

        node = static_cast<uint32_t>(m_edges[edgeIndex].childNode);
        assert(positionHistorySize < positionHistory.size());
        positionHistory[positionHistorySize++] = m_nodes[node].board.positionHash;
    }
}

void MCTS::backpropagate(float value)
{
    for (size_t trailIndex = m_edgeTrailSize; trailIndex-- > 0;)
    {
        value = -value;
        m_edges[m_edgeTrail[trailIndex]].valueSum += value;
        ++m_edges[m_edgeTrail[trailIndex]].visits;
    }
}

MCTS::MCTS(const Board& root, std::span<const uint64_t> history, MCTSConfig config)
    : m_config(config)
    , m_gameHistorySize(history.size())
{
    assert(m_config.simulations > 0);
    assert(history.size() <= m_gameHistory.size());

    m_nodes.reserve(static_cast<size_t>(m_config.simulations) + 1);
    std::ranges::copy(history, m_gameHistory.begin());

    // Nothing worth asking about a position the rules have already settled, and
    // nothing to search from it either.
    if (root.state != State::ongoing)
        addNode(root, Evaluation{});
    else
        m_pendingLeaf = root;
}

const Board* MCTS::pendingLeaf()
{
    if (m_pendingLeaf.has_value())
        return &*m_pendingLeaf;

    while (m_nodes[0].edgeCount != 0 && static_cast<int>(m_nodes[0].visits) < m_config.simulations)
    {
        if (descend())
            return &*m_pendingLeaf;
    }
    return nullptr;
}

void MCTS::absorb(const Evaluation& evaluation)
{
    assert(m_pendingLeaf.has_value());

    if (m_nodes.empty())
    {
        addNode(*m_pendingLeaf, evaluation);
        if (m_config.rootNoise > 0.0f)
            addExplorationNoise();
    }
    else
    {
        assert(m_edgeTrailSize > 0);
        const uint32_t expandedEdgeIndex = m_edgeTrail[m_edgeTrailSize - 1];
        assert(m_edges[expandedEdgeIndex].childNode < 0);
        m_edges[expandedEdgeIndex].childNode = static_cast<int32_t>(addNode(*m_pendingLeaf, evaluation));
        backpropagate(evaluation.value);
    }
    m_pendingLeaf.reset();
}

VisitCounts MCTS::visits() const
{
    VisitCounts counts{};
    if (m_nodes.empty())
        return counts;

    const Node& root = m_nodes[0];
    for (uint32_t edgeIndex = root.firstEdgeIndex; edgeIndex < root.firstEdgeIndex + root.edgeCount; ++edgeIndex)
        counts[m_edges[edgeIndex].moveId] = m_edges[edgeIndex].visits;
    return counts;
}

} // namespace amoeba_bot
