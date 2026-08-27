#ifndef MCTS_HPP
#define MCTS_HPP

#include "amoeba.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace amoeba_bot
{

struct MCTSConfig
{
    int simulations = 800;
};

class MCTS
{
public:
    MCTS(const Board& root, std::span<const uint64_t> history, MCTSConfig config = {});

    const Board* pendingLeaf();

    void absorb(std::span<float> policy, float value);

    std::array<uint32_t, moveIdCount> visits() const;

private:
    struct Edge
    {
        uint16_t moveId;

        float prior;
        float valueSum;
        uint32_t visits;

        std::optional<int32_t> node;
    };

    struct Node
    {
        Board board;

        uint32_t visits;

        uint32_t firstEdgeIndex;
        uint32_t edgeCount;
    };

    uint32_t addNode(const Board&, std::span<float> policy);
    uint32_t addTerminalNode(float value);

    uint32_t selectEdgeToExplore(uint32_t node) const;
    void addExplorationNoise();
    bool descend();
    void backpropagate(float value);

    MCTSConfig m_config;

    std::vector<std::variant<Node, float>> m_nodes; // `Node` or float to indicate the game result
    std::vector<Edge> m_edges;

    std::array<uint64_t, moveLimit + 1> m_gameHistory;
    size_t m_gameHistorySize;

    std::array<uint32_t, moveLimit> m_edgeTrail;
    size_t m_edgeTrailSize = 0;

    std::optional<Board> m_pendingLeaf;
};

} // namespace amoeba_bot

#endif // MCTS_HPP
