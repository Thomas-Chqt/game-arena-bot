#pragma once

#include "amoeba.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace amoeba_bot
{

using VisitCounts = std::array<uint32_t, moveIdCount>;

template<int SimulationCount>
class MCTS
{
    static_assert(SimulationCount > 0);

public:
    MCTS(const Board& root, std::span<const uint64_t> history);

    const Board* pendingLeaf();
    void absorb(std::span<const float, moveIdCount> policy, float value);
    VisitCounts visits() const;

private:
    struct Edge
    {
        uint16_t moveId;
        float prior;
        float valueSum;
        uint32_t visits;
        std::optional<uint32_t> node;
    };

    struct Node
    {
        Board board;
        uint32_t visits;
        uint32_t firstEdgeIndex;
        uint32_t edgeCount;
    };

    uint32_t addNode(const Board&, std::span<const float, moveIdCount> policy);
    uint32_t addTerminalNode(float value);
    uint32_t selectEdgeToExplore(uint32_t node) const;
    bool descend();
    void backpropagate(float value);

    std::vector<std::variant<Node, float>> m_nodes;
    std::vector<Edge> m_edges;

    std::array<uint64_t, moveLimit + 1> m_gameHistory;
    size_t m_gameHistorySize;

    std::array<uint32_t, moveLimit> m_edgeTrail;
    size_t m_edgeTrailSize = 0;

    std::optional<Board> m_pendingLeaf;
};

uint16_t bestMove(const VisitCounts&);
float outcomeFor(Outcome, bool whiteToMove);

extern template class MCTS<256>;
extern template class MCTS<1000>;

} // namespace amoeba_bot
