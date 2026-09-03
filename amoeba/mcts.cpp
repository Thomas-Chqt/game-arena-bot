#include "mcts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>
#include <tuple>
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
MCTS<SimulationCount>::MCTS(const Board& root, std::span<const uint64_t> history, bool proveOutcomes)
    : m_gameHistorySize(history.size())
    , m_proveOutcomes(proveOutcomes)
    , m_pendingLeaf(root)
{
    assert(history.size() <= m_gameHistory.size());
    // The root's tactical pass can add solved children before any simulations.
    m_nodes.reserve(SimulationCount + 1 + (proveOutcomes ? root.legalMoveCount : 0));
    std::ranges::copy(history, m_gameHistory.begin());
}

template<int SimulationCount>
const Board* MCTS<SimulationCount>::pendingLeaf()
{
    if (m_pendingLeaf.has_value())
        return &*m_pendingLeaf;

    while (std::get<Node>(m_nodes[0]).edgeCount != 0
           && (!m_proveOutcomes || !provenValue(0).has_value())
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
        if (m_proveOutcomes)
            checkRootTactics();
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
uint16_t MCTS<SimulationCount>::bestMove() const
{
    assert(!m_nodes.empty());
    if (!m_proveOutcomes)
        return amoeba_bot::bestMove(visits());

    const Node& root = std::get<Node>(m_nodes[0]);
    assert(root.edgeCount > 0);
    const auto rank = [&](uint32_t edgeIndex) {
        const Edge& edge = m_edges[edgeIndex];
        const std::optional<float> childValue = edge.node.has_value()
            ? provenValue(*edge.node) : std::nullopt;
        // Values belong to the child side: its loss is our proven win.
        const int outcomeRank = childValue == -1.0f ? 2 : childValue == 1.0f ? 0 : 1;
        return std::tuple{outcomeRank, edge.visits, edge.prior};
    };
    const uint32_t edgeIndex = *std::ranges::max_element(
        std::views::iota(root.firstEdgeIndex, root.firstEdgeIndex + root.edgeCount), {}, rank);
    return m_edges[edgeIndex].moveId;
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
        .provenValue = std::nullopt,
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
uint32_t MCTS<SimulationCount>::addSolvedNode(float value)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(value);
    return index;
}

template<int SimulationCount>
std::optional<float> MCTS<SimulationCount>::provenValue(uint32_t nodeIndex) const
{
    if (const float* value = std::get_if<float>(&m_nodes[nodeIndex]))
        return *value;
    return std::get<Node>(m_nodes[nodeIndex]).provenValue;
}

template<int SimulationCount>
void MCTS<SimulationCount>::updateProvenValue(uint32_t nodeIndex)
{
    Node& node = std::get<Node>(m_nodes[nodeIndex]);
    bool allProven = true;
    float bestValue = -1.0f;
    for (uint32_t edgeIndex = node.firstEdgeIndex;
         edgeIndex < node.firstEdgeIndex + node.edgeCount; ++edgeIndex)
    {
        const Edge& edge = m_edges[edgeIndex];
        const std::optional<float> childValue = edge.node.has_value()
            ? provenValue(*edge.node) : std::nullopt;
        if (!childValue.has_value())
        {
            allProven = false;
            continue;
        }
        bestValue = std::max(bestValue, -*childValue);
        if (bestValue == 1.0f)
        {
            node.provenValue = bestValue;
            return;
        }
    }
    // One winning move proves a win. A draw or loss needs every move solved.
    if (allProven)
        node.provenValue = bestValue;
}

template<int SimulationCount>
void MCTS<SimulationCount>::checkRootTactics()
{
    const Node& root = std::get<Node>(m_nodes[0]);
    std::array<uint64_t, moveLimit + 1> history;
    std::ranges::copy_n(m_gameHistory.begin(), m_gameHistorySize, history.begin());
    const std::span<const uint64_t> rootHistory{history.data(), m_gameHistorySize};

    for (uint32_t edgeIndex = root.firstEdgeIndex;
         edgeIndex < root.firstEdgeIndex + root.edgeCount; ++edgeIndex)
    {
        Edge& edge = m_edges[edgeIndex];
        const MoveResult result = applyMove(root.board, Move::fromId(edge.moveId), rootHistory);
        if (const auto* outcome = std::get_if<Outcome>(&result))
        {
            const float childValue = outcomeFor(*outcome, !root.board.whiteToMove);
            edge.node = addSolvedNode(childValue);
            if (childValue == -1.0f)
                break;
            continue;
        }

        const Board& child = std::get<Board>(result);
        assert(m_gameHistorySize < history.size());
        history[m_gameHistorySize] = child.positionHash;
        child.forEachLegal([&](uint16_t replyId) {
            if (edge.node.has_value())
                return;
            const MoveResult reply = applyMove(child, Move::fromId(replyId),
                std::span<const uint64_t>{history.data(), m_gameHistorySize + 1});
            if (const auto* outcome = std::get_if<Outcome>(&reply);
                outcome != nullptr && outcomeFor(*outcome, child.whiteToMove) == 1.0f)
            {
                // The opponent can force a win immediately after this move.
                edge.node = addSolvedNode(1.0f);
            }
        });
    }
    updateProvenValue(0);
}

template<int SimulationCount>
uint32_t MCTS<SimulationCount>::selectEdgeToExplore(uint32_t nodeIndex) const
{
    constexpr float explorationConstant = 1.5f;
    const Node& node = std::get<Node>(m_nodes[nodeIndex]);
    const float exploration = explorationConstant * std::sqrt(static_cast<float>(node.visits));

    const auto provenLoss = [&](const Edge& edge) {
        return m_proveOutcomes && edge.node.has_value() && provenValue(*edge.node) == 1.0f;
    };
    float availablePrior = 0.0f;
    uint32_t availableMoves = 0;
    if (m_proveOutcomes)
    {
        for (uint32_t edgeIndex = node.firstEdgeIndex;
             edgeIndex < node.firstEdgeIndex + node.edgeCount; ++edgeIndex)
        {
            const Edge& edge = m_edges[edgeIndex];
            if (!provenLoss(edge))
            {
                availablePrior += edge.prior;
                ++availableMoves;
            }
        }
        assert(availableMoves > 0);
    }

    const auto score = [&](uint32_t edgeIndex) {
        const Edge& edge = m_edges[edgeIndex];
        if (provenLoss(edge))
            return -std::numeric_limits<float>::infinity();
        // Filtering a high-prior losing move must not suppress exploration of
        // the remaining moves, even if their probabilities underflowed to zero.
        const float prior = !m_proveOutcomes ? edge.prior
            : availablePrior > 0.0f ? edge.prior / availablePrior : 1.0f / availableMoves;
        const std::optional<float> childValue = m_proveOutcomes && edge.node.has_value()
            ? provenValue(*edge.node) : std::nullopt;
        const float meanValue = childValue.has_value() ? -*childValue
            : edge.visits == 0 ? 0.0f : edge.valueSum / edge.visits;
        return meanValue + exploration * prior / static_cast<float>(1 + edge.visits);
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

        if (m_proveOutcomes && provenValue(nodeIndex).has_value())
        {
            backpropagate(*provenValue(nodeIndex));
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
                edge.node = addSolvedNode(value);
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
        value = -terminalValueDiscountPerMove * value;
        Edge& edge = m_edges[m_edgeTrail[trailIndex]];
        edge.valueSum += value;
        ++edge.visits;
        if (m_proveOutcomes)
        {
            const uint32_t parentIndex = trailIndex == 0
                ? 0 : *m_edges[m_edgeTrail[trailIndex - 1]].node;
            updateProvenValue(parentIndex);
            if (const std::optional<float> proof = provenValue(parentIndex))
                value = *proof;
        }
    }
}

template class MCTS<256>;
template class MCTS<512>;
template class MCTS<1000>;
template class MCTS<1500>;

} // namespace amoeba_bot
