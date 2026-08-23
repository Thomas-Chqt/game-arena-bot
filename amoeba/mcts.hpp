#ifndef MCTS_HPP
#define MCTS_HPP

// ---------------------------------------------------------------------------
// PUCT Monte Carlo tree search, the AlphaZero variant: no random playouts in
// the search itself, and the caller supplies the value of each new position.
//
// The tree keeps two numbers per move: how many simulations went through it,
// and the sum of the values that came back. Their ratio is the move's running
// average outcome over everything the search has found below it. Selection
// takes the move maximising that average plus a bonus that starts proportional
// to the evaluator's prior and decays as the move gets visited, so a promising
// move is tried early but has to keep earning its visits.
//
// Every stored value is from the point of view of the side to move at the node
// that owns it, which is why the backup flips sign once per level.
//
// Each descent hands its one leaf to the caller, then receives that leaf's
// evaluation before the next descent. A caller driving several independent
// searches may batch those leaves.
// ---------------------------------------------------------------------------

#include "amoeba.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace amoeba
{

// What the search wants to know about a position it has just reached.
//
// `value` is what the position is worth to the side to move there: +1 they win,
// -1 they lose. `policy` is one number per move id; only the legal entries are
// read, and they are renormalised to sum to one, so the rest may be anything.
struct Evaluation
{
    std::array<float, amoeba::moveIdCount> policy;
    float value;
};

struct MCTSConfig
{
    // Every move starts a fresh tree and receives this many simulations through
    // its root.
    int simulations = 800;
    float explorationConstant = 1.5f;

    // Self-play only, and off by default so competition keeps the network's own
    // opinion. A network is deterministic and so is edge selection, so without
    // noise every self-play game from a given network is the same game: the network
    // only ever sees positions it already understands, and training stalls while
    // every loss curve stays healthy.
    //
    // The mix is prior = (1 - rootNoise) * network + rootNoise * Dirichlet(alpha).
    // Applied at the root only, because the root is the position that becomes a
    // training example - noise deeper in the tree would just spoil the search's
    // judgement. alpha below 1 makes each draw spiky, so a different random handful
    // of moves gets promoted each game; AlphaZero scaled it as 10 / average legal
    // moves, and Amoeba averages 27.
    float rootNoise = 0.0f;
    float noiseAlpha = 0.35f;
    uint64_t noiseSeed = 0;
};

// Simulations spent on each move id. The argmax is the move to play; normalised
// it is the policy target the network trains on.
using VisitCounts = std::array<uint32_t, amoeba::moveIdCount>;

class MCTS
{
public:
    // `history` is the hash of every position the real game has passed through,
    // ending with `root`'s own; amoeba::applyMove() needs it to see repetitions
    // that the search walks into. One MCTS searches this one root only.
    MCTS(const amoeba::Board& root, std::span<const uint64_t> history, MCTSConfig config = {});

    // Descends once and returns the leaf that needs a network evaluation. A null
    // pointer means the budget is spent and visits() is ready. Terminal leaves are
    // backed up immediately because their value comes from the rules.
    //
    // The pointer stays valid until absorb() or this MCTS is destroyed.
    const amoeba::Board* pendingLeaf();

    // Expands and backs up the leaf returned by pendingLeaf().
    void absorb(const Evaluation&);

    VisitCounts visits() const;

private:
    struct Edge
    {
        float prior;
        float valueSum;
        uint32_t visits;
        int32_t childNode; // -1 until a simulation goes through this move
        uint16_t moveId;
    };

    struct OngoingNode
    {
        amoeba::Board board;
        uint32_t firstEdgeIndex;
        uint32_t edgeCount;
    };

    struct Node
    {
        std::variant<OngoingNode, float> contents;
        uint32_t visits;
    };

    uint32_t addNode(const amoeba::Board&, const Evaluation&);
    uint32_t addTerminalNode(float value);
    uint32_t selectEdgeToExplore(uint32_t node) const;
    void addExplorationNoise();
    bool descend();
    void backpropagate(float value);

    MCTSConfig m_config;

    std::vector<Node> m_nodes; // node 0 is always the root
    std::vector<Edge> m_edges;

    std::array<uint64_t, amoeba::moveLimit + 1> m_gameHistory;
    size_t m_gameHistorySize;

    std::array<uint32_t, amoeba::moveLimit> m_edgeTrail;
    size_t m_edgeTrailSize = 0;

    std::optional<amoeba::Board> m_pendingLeaf;
};

uint16_t bestMove(const VisitCounts&);

// What a finished game is worth to a given side. Every value in the search and
// every training target is signed this way - from the point of view of the side
// to move - and inverting it trains a bot that reliably plays badly while every
// loss curve looks healthy.
float outcomeFor(amoeba::Outcome, bool whiteToMove);

} // namespace amoeba

#endif // MCTS_HPP
