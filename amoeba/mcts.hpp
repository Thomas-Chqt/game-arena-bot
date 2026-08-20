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
//
// The search does not own an evaluator and does not call one. It hands out the
// boards it is waiting on and takes the answers back, so the batch that reaches
// the network can be gathered across every game in flight rather than out of one
// search's guesses. runSearch() is the one-game convenience wrapper.
// ---------------------------------------------------------------------------

#include "amoeba.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace bot
{

// One set of threads for the whole process, because everything here wants all of
// the machine and nothing wants to share it with another pool: self-play descends
// hundreds of trees, the trainer encodes hundreds of boards per step, and both
// are bursts of identical work with no reason to own threads between bursts.
class ThreadPool
{
public:
    static ThreadPool& global();

    explicit ThreadPool(unsigned width);
    ~ThreadPool();

    unsigned width() const { return static_cast<unsigned>(m_workers.size()) + 1; }

    // Runs body(0), ..., body(count - 1) somewhere across the pool and returns
    // once every one of them has run. The calling thread takes indices too, so a
    // one-thread pool is just a loop and no work is left waiting on a thread that
    // does not exist.
    //
    // A call from inside a body runs serially: the outer call already has every
    // thread, so waiting for one here would deadlock. An exception from a body is
    // rethrown to the caller once the round has finished.
    void forEach(size_t count, const std::function<void(size_t)>& body);

private:
    void serve();
    void drain();

    std::mutex                         m_mutex;
    std::condition_variable            m_wake;
    std::condition_variable            m_finished;
    const std::function<void(size_t)>* m_body = nullptr;
    size_t                             m_count = 0;
    std::atomic<size_t>                m_next{0};
    uint64_t                           m_round = 0;
    unsigned                           m_busy  = 0;
    bool                               m_stopping = false;
    std::exception_ptr                 m_failure;
    std::vector<std::thread>           m_workers;
};

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
    // position on a GPU wastes most of the device: 1.70 ms for one position
    // against 0.15 ms each for 256.
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
    // Simulations through the root, not new ones: a re-rooted tree arrives with
    // some of them already spent, and that is the whole point of keeping it.
    int   simulations = 800;
    float cPuct       = 1.5f;

    // The search stops at the simulation count or the deadline, whichever comes
    // first, and always runs one batch so the visit counts can never be empty.
    // Match play sets the count high and lets the clock bind: a turn that arrives
    // late is a forfeit, while a turn that only managed 300 simulations is merely
    // a weaker move. Self-play does the reverse, so its data does not depend on
    // how busy the machine was.
    std::chrono::milliseconds deadline = std::chrono::hours{1};

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

    // How many leaves one search offers per round. Leave it at 1 whenever there
    // are other games to batch with: descents inside a round cannot see each
    // other's results, so they need a virtual loss to diverge at all and they
    // still choose on statistics that are one round stale. Raise it only when a
    // single search has to fill the batch by itself, which is match play - there
    // the alternative is one position per forward pass and ten times the cost.
    int batchSize = 1;
};

// Simulations spent on each move id. The argmax is the move to play; normalised
// it is the policy target the network trains on.
using VisitCounts = std::array<uint32_t, amoeba::kNumMoveIds>;

class Search
{
public:
    explicit Search(Config config = {}) : m_config(config), m_rng(config.noiseSeed) {}

    // Throws the tree away and starts again at `root`. `history` is the hash of
    // every position the real game has passed through, ending with `root`'s own;
    // amoeba::apply() needs it to see repetitions that the search walks into.
    void restart(const amoeba::Board& root, std::span<const uint64_t> history);

    // Re-roots on the child `moveId` leads to and keeps its subtree, so the
    // simulations that already went through that move are still there next turn -
    // 20-40% of the next search for free, and more when the move played was the
    // one the search liked. Falls back to restart() when that child was never
    // expanded, which is what happens when the opponent plays something the search
    // never looked at.
    //
    // The path from the new root down is the same path it always was, only with one
    // more ply of it now living in the game history, so every node's
    // repetition-dependent verdict stays true.
    void advance(uint16_t moveId, const amoeba::Board& next, std::span<const uint64_t> history);

    // The boards this search cannot go any further without an evaluation of.
    // Empty means it has spent its budget and visits() is ready to read.
    //
    // The pointers stay valid until the next call to any of these three.
    std::span<const amoeba::Board* const> pendingLeaves();

    // Answers to the last pendingLeaves(), in the same order.
    void absorb(std::span<const Evaluation>);

    VisitCounts visits() const;

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

    uint32_t expand(const amoeba::Board&, const Evaluation&);
    uint32_t selectEdge(uint32_t node) const;
    int32_t  childFor(uint16_t moveId) const;
    void     keepSubtree(uint32_t node);
    void     addRootNoise();
    void     collect(int wanted);
    void     backUp(std::span<const Evaluation>);
    bool     spent() const;

    Config m_config;

    std::vector<Node>     m_nodes;      // node 0 is always the root
    std::vector<Edge>     m_edges;
    std::vector<Node>     m_spareNodes;   // buffers for re-rooting, swapped in
    std::vector<Edge>     m_spareEdges;
    std::vector<uint64_t> m_path;       // the game history, then hashes down the descent being collected
    size_t                m_baseLength = 0;

    amoeba::Board m_rootBoard;
    bool          m_needsRoot = true;   // the root exists but has not been evaluated yet

    // The clock starts when the search does, not when the root is set: match play
    // re-roots as soon as it hears the opponent's move and only searches when it is
    // asked for one, and the wait in between is not part of its turn.
    std::chrono::steady_clock::time_point m_expiry;
    bool                                  m_timing = false;

    std::mt19937_64    m_rng;
    std::vector<float> m_noise;

    std::vector<Descent>              m_descents;
    std::vector<uint32_t>             m_trailStore;   // every trail in the round, concatenated
    std::vector<amoeba::Board>        m_leaves;
    std::vector<const amoeba::Board*> m_leafPointers;
};

// Drives one search to the end against one evaluator. Match play has a single
// game, so its batch can only come from inside a single search; self-play has
// hundreds and gathers across them instead.
VisitCounts runSearch(Search&, Evaluator&);

uint16_t randomLegalMove(const amoeba::Board&, std::mt19937_64&);
uint16_t bestMove(const VisitCounts&);

// What a finished game is worth to a given side. Every value in the search and
// every training target is signed this way - from the point of view of the side
// to move - and inverting it trains a bot that reliably plays badly while every
// loss curve looks healthy.
float outcomeFor(amoeba::State, bool whiteToMove);

} // namespace bot
