#include "mcts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ranges>
#include <utility>

namespace amoeba_bot
{

float outcomeFor(Outcome outcome, bool whiteToMove)
{
    if (outcome == Outcome::draw)
        return 0.0f;
    return (outcome == Outcome::whiteWins) == whiteToMove ? 1.0f : -1.0f;
}

uint16_t bestMove(const VisitCounts& counts)
{
    return static_cast<uint16_t>(std::ranges::max_element(counts) - counts.begin());
}

template<int SimulationCount>
MCTS<SimulationCount>::MCTS(const Board& root, std::span<const uint64_t> history)
    : m_gameHistorySize(history.size())
    , m_pendingLeaf(root)
{
    assert(history.size() <= m_gameHistory.size());
    m_nodes.reserve(SimulationCount + 1);
    std::ranges::copy(history, m_gameHistory.begin());
}

template<int SimulationCount>
const Board* MCTS<SimulationCount>::pendingLeaf()
{
    if (m_pendingLeaf.has_value())
        return &*m_pendingLeaf;

    while (std::get<Node>(m_nodes[0]).edgeCount != 0
           && std::get<Node>(m_nodes[0]).visits < SimulationCount)
    {
        if (descend())
            return &*m_pendingLeaf;
    }
    return nullptr;
}

template<int SimulationCount>
void MCTS<SimulationCount>::absorb(std::span<const float, moveIdCount> policy, float value)
{
    assert(m_pendingLeaf.has_value());

    if (m_nodes.empty())
    {
        addNode(*m_pendingLeaf, policy);
    }
    else
    {
        assert(m_edgeTrailSize > 0);
        const uint32_t expandedEdgeIndex = m_edgeTrail[m_edgeTrailSize - 1];
        assert(!m_edges[expandedEdgeIndex].node.has_value());
        const uint32_t childNode = addNode(*m_pendingLeaf, policy);
        m_edges[expandedEdgeIndex].node = childNode;
        backpropagate(value);
    }
    m_pendingLeaf.reset();
}

template<int SimulationCount>
VisitCounts MCTS<SimulationCount>::visits() const
{
    VisitCounts counts{};
    if (m_nodes.empty())
        return counts;

    const Node& root = std::get<Node>(m_nodes[0]);
    for (uint32_t edgeIndex = root.firstEdgeIndex;
         edgeIndex < root.firstEdgeIndex + root.edgeCount; ++edgeIndex)
    {
        counts[m_edges[edgeIndex].moveId] = m_edges[edgeIndex].visits;
    }
    return counts;
}

template<int SimulationCount>
uint32_t MCTS<SimulationCount>::addNode(
    const Board& board, std::span<const float, moveIdCount> policy)
{
    float totalPrior = 0.0f;
    board.forEachLegal([&](uint16_t moveId) { totalPrior += policy[moveId]; });
    assert(totalPrior > 0.0f);

    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{
        .board = board,
        .visits = 0,
        .firstEdgeIndex = static_cast<uint32_t>(m_edges.size()),
        .edgeCount = board.legalMoveCount,
    });

    board.forEachLegal([&](uint16_t moveId) {
        m_edges.push_back(Edge{
            .moveId = moveId,
            .prior = policy[moveId] / totalPrior,
            .valueSum = 0.0f,
            .visits = 0,
            .node = std::nullopt,
        });
    });
    return index;
}

template<int SimulationCount>
uint32_t MCTS<SimulationCount>::addTerminalNode(float value)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(value);
    return index;
}

template<int SimulationCount>
uint32_t MCTS<SimulationCount>::selectEdgeToExplore(uint32_t nodeIndex) const
{
    constexpr float explorationConstant = 1.5f;
    const Node& node = std::get<Node>(m_nodes[nodeIndex]);
    const float exploration = explorationConstant * std::sqrt(static_cast<float>(node.visits));

    const auto score = [&](uint32_t edgeIndex) {
        const Edge& edge = m_edges[edgeIndex];
        const float meanValue = edge.visits == 0 ? 0.0f : edge.valueSum / edge.visits;
        return meanValue + exploration * edge.prior / static_cast<float>(1 + edge.visits);
    };

    return *std::ranges::max_element(
        std::views::iota(node.firstEdgeIndex, node.firstEdgeIndex + node.edgeCount), {}, score);
}

template<int SimulationCount>
bool MCTS<SimulationCount>::descend()
{
    std::array<uint64_t, moveLimit + 1> positionHistory;
    std::ranges::copy_n(m_gameHistory.begin(), m_gameHistorySize, positionHistory.begin());
    size_t positionHistorySize = m_gameHistorySize;
    m_edgeTrailSize = 0;

    uint32_t nodeIndex = 0;
    for (;;)
    {
        if (float* terminalValue = std::get_if<float>(&m_nodes[nodeIndex]))
        {
            backpropagate(*terminalValue);
            return false;
        }

        Node& node = std::get<Node>(m_nodes[nodeIndex]);
        ++node.visits;

        const uint32_t edgeIndex = selectEdgeToExplore(nodeIndex);
        assert(m_edgeTrailSize < m_edgeTrail.size());
        m_edgeTrail[m_edgeTrailSize++] = edgeIndex;
        Edge& edge = m_edges[edgeIndex];

        if (!edge.node.has_value())
        {
            MoveResult result = applyMove(
                node.board, Move::fromId(edge.moveId),
                std::span{positionHistory.data(), positionHistorySize});
            if (const auto* outcome = std::get_if<Outcome>(&result))
            {
                const float value = outcomeFor(*outcome, !node.board.whiteToMove);
                edge.node = addTerminalNode(value);
                backpropagate(value);
                return false;
            }

            m_pendingLeaf = std::get<Board>(std::move(result));
            return true;
        }

        nodeIndex = *edge.node;
        if (const Node* child = std::get_if<Node>(&m_nodes[nodeIndex]))
        {
            assert(positionHistorySize < positionHistory.size());
            positionHistory[positionHistorySize++] = child->board.positionHash;
        }
    }
}

template<int SimulationCount>
void MCTS<SimulationCount>::backpropagate(float value)
{
    for (size_t trailIndex = m_edgeTrailSize; trailIndex-- > 0;)
    {
        value = -value;
        Edge& edge = m_edges[m_edgeTrail[trailIndex]];
        edge.valueSum += value;
        ++edge.visits;
    }
}

template class MCTS<256>;
template class MCTS<1000>;

} // namespace amoeba_bot
