#pragma once

// ---------------------------------------------------------------------------
// PUCT Monte Carlo tree search, the AlphaZero variant: no random playouts in
// the search itself, the value of a new position comes from an Evaluator.
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
// ---------------------------------------------------------------------------

#include <amoeba/amoeba.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace bot
{

// What the search wants to know about a position it has just reached.
//
// `value` is what the position is worth to the side to move there: +1 they win,
// -1 they lose. `policy` is one number per move id; only the legal entries are
// read, and they are renormalised to sum to one, so the rest may be anything.
struct Evaluation
{
    std::array<float, amoeba::kNumMoveIds> policy;
    float value;
};

class Evaluator
{
public:
    virtual ~Evaluator() = default;

    // `out` is the same length as `boards`. Plural because evaluating a single
    // position on a GPU wastes most of the device - the network implementation
    // will want whole batches, even though the search asks one at a time today.
    virtual void evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out) = 0;
};

// Uniform priors, and a value read off one uniformly random playout to the end
// of the game. It knows nothing about Amoeba, but it makes the search playable
// before the network exists and it is the baseline the network has to beat.
class RolloutEvaluator final : public Evaluator
{
public:
    explicit RolloutEvaluator(uint64_t seed) : m_rng(seed) {}

    void evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out) override;

private:
    float playout(amoeba::Board);

    std::mt19937_64 m_rng;
};

struct Config
{
    int   simulations = 800;
    float cPuct       = 1.5f;

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
    float    rootNoise  = 0.0f;
    float    noiseAlpha = 0.35f;
    uint64_t noiseSeed  = 0;

    // How many leaves to gather before asking the evaluator. A rollout gains
    // nothing from this, but the network costs 1.70 ms for one position and
    // 0.16 ms each for 64, so leave it at 1 for RolloutEvaluator and raise it for
    // NetworkEvaluator. Descents inside a batch see slightly stale statistics, so
    // each simulation is a little less well directed - worth it about tenfold.
    int batchSize = 1;
};

// Simulations spent on each move id. The argmax is the move to play; normalised
// it is the policy target the network trains on.
using VisitCounts = std::array<uint32_t, amoeba::kNumMoveIds>;

class Search
{
public:
    explicit Search(Evaluator& evaluator, Config config = {})
        : m_evaluator(evaluator), m_config(config), m_rng(config.noiseSeed)
    {
    }

    // `history` is the hash of every position the real game has passed through,
    // ending with `root`'s own. amoeba::apply() needs it to see repetitions that
    // the search walks into.
    VisitCounts run(const amoeba::Board& root, std::span<const uint64_t> history);

private:
    struct Edge
    {
        float    prior;
        float    valueSum;
        uint32_t visits;
        int32_t  child;    // -1 until a simulation goes through this move
        uint16_t moveId;
    };

    struct Node
    {
        amoeba::Board board;
        uint32_t      firstEdge;
        uint32_t      edgeCount;   // 0 at a terminal position
        uint32_t      visits;
    };

    // Appends the node and its outgoing edges, and returns its index alongside
    // the value of the position for the side to move there.
    // Where one descent ended, and everything needed to back it up once its leaf
    // has been evaluated.
    struct Descent
    {
        uint32_t trailStart;
        uint32_t trailLength;
        int32_t  edge;    // the edge whose child this descent is creating, -1 if it hit a terminal node
        int32_t  leaf;    // index into m_leaves, -1 if it hit a terminal node
        float    value;   // the terminal value, when there is no leaf to evaluate
    };

    std::pair<uint32_t, float> addNode(const amoeba::Board&);
    uint32_t expand(const amoeba::Board&, const Evaluation&);
    uint32_t selectEdge(uint32_t node) const;
    void addRootNoise();
    void collect(size_t baseLength, int wanted);
    void backUp();

    Evaluator& m_evaluator;
    Config     m_config;

    std::vector<Node>     m_nodes;
    std::vector<Edge>     m_edges;
    std::vector<uint64_t> m_path;    // hashes down the descent being collected

    std::mt19937_64    m_rng;
    std::vector<float> m_noise;

    std::vector<Descent>              m_descents;
    std::vector<uint32_t>             m_trailStore;   // every trail in the batch, concatenated
    std::vector<amoeba::Board>        m_leaves;
    std::vector<const amoeba::Board*> m_leafPointers;
    std::vector<Evaluation>           m_evaluations;
};

uint16_t randomLegalMove(const amoeba::Board&, std::mt19937_64&);
uint16_t bestMove(const VisitCounts&);

} // namespace bot
