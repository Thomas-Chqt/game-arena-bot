#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace bot
{

namespace
{

// Seen by whoever is to move in it - which, at a terminal position, is the side
// that was just mated or ran out of moves.
float terminalValue(const amoeba::Board& b)
{
    return outcomeFor(b.state, b.whiteToMove);
}

// A body running on the pool must not wait for the pool: the round it would be
// waiting on cannot start until it returns.
thread_local bool tInsidePool = false;

} // namespace

// ---------------------------------------------------------------------------

ThreadPool& ThreadPool::global()
{
    static ThreadPool one{std::max(1u, std::thread::hardware_concurrency())};
    return one;
}

ThreadPool::ThreadPool(unsigned width)
{
    // One of the threads is whoever calls forEach, so only width - 1 are hired.
    for (unsigned i = 1; i < std::max(1u, width); ++i)
        m_workers.emplace_back([this] { serve(); });
}

ThreadPool::~ThreadPool()
{
    {
        const std::lock_guard guard{m_mutex};
        m_stopping = true;
    }
    m_wake.notify_all();
    for (std::thread& worker : m_workers)
        worker.join();
}

void ThreadPool::forEach(size_t count, const std::function<void(size_t)>& body)
{
    if (count == 0)
        return;

    if (tInsidePool || m_workers.empty())
    {
        for (size_t i = 0; i < count; ++i)
            body(i);
        return;
    }

    {
        const std::lock_guard guard{m_mutex};
        m_body  = &body;
        m_count = count;
        m_next.store(0, std::memory_order_relaxed);
        m_busy = static_cast<unsigned>(m_workers.size());
        ++m_round;
    }
    m_wake.notify_all();

    drain();

    std::unique_lock lock{m_mutex};
    m_finished.wait(lock, [this] { return m_busy == 0; });
    m_body = nullptr;

    if (m_failure)
        std::rethrow_exception(std::exchange(m_failure, nullptr));
}

void ThreadPool::drain()
{
    tInsidePool = true;
    try
    {
        for (;;)
        {
            const size_t index = m_next.fetch_add(1, std::memory_order_relaxed);
            if (index >= m_count)
                break;
            (*m_body)(index);
        }
    }
    catch (...)
    {
        // Whoever else is still running finishes the index it is on and then stops;
        // the round has to complete either way, or forEach would go on waiting for a
        // thread that has given up.
        m_next.store(m_count, std::memory_order_relaxed);
        const std::lock_guard guard{m_mutex};
        if (!m_failure)
            m_failure = std::current_exception();
    }
    tInsidePool = false;
}

void ThreadPool::serve()
{
    uint64_t seen = 0;
    for (;;)
    {
        std::unique_lock lock{m_mutex};
        m_wake.wait(lock, [this, &seen] { return m_stopping || m_round != seen; });
        if (m_stopping)
            return;
        seen = m_round;
        lock.unlock();

        drain();

        lock.lock();
        if (--m_busy == 0)
            m_finished.notify_one();
    }
}

// ---------------------------------------------------------------------------

float outcomeFor(amoeba::State state, bool whiteToMove)
{
    if (state == amoeba::State::Draw)
        return 0.0f;
    return (state == amoeba::State::WhiteWins) == whiteToMove ? 1.0f : -1.0f;
}

uint16_t randomLegalMove(const amoeba::Board& b, std::mt19937_64& rng)
{
    int remaining = std::uniform_int_distribution<int>(0, b.moveCount - 1)(rng);
    uint16_t chosen = 0;
    b.forEachLegal([&](uint16_t id) { if (remaining-- == 0) chosen = id; });
    return chosen;
}

uint16_t bestMove(const VisitCounts& counts)
{
    return static_cast<uint16_t>(std::ranges::max_element(counts) - counts.begin());
}

// ---------------------------------------------------------------------------

void RolloutEvaluator::evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out)
{
    for (size_t i = 0; i < boards.size(); ++i)
    {
        out[i].policy.fill(1.0f);
        out[i].value = playout(*boards[i]);
    }
}

float RolloutEvaluator::playout(amoeba::Board b)
{
    const bool mover = b.whiteToMove;

    // No history, so repetitions go unseen; the staleness and move caps still
    // end every playout, and a random game's verdict is too crude for one
    // missed draw to matter.
    while (b.state == amoeba::State::Ongoing)
        b = amoeba::apply(b, amoeba::Move::fromId(randomLegalMove(b, m_rng)));

    return outcomeFor(b.state, mover);
}

// ---------------------------------------------------------------------------

// Appends the node and its outgoing edges from an evaluation already in hand. A
// terminal board gets a node with no edges and its Evaluation is ignored.
uint32_t Search::expand(const amoeba::Board& b, const Evaluation& ev)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{b, static_cast<uint32_t>(m_edges.size()), 0, 0});

    if (b.state != amoeba::State::Ongoing)
        return index;

    float total = 0.0f;
    b.forEachLegal([&](uint16_t id) { total += ev.policy[id]; });
    b.forEachLegal([&](uint16_t id) { m_edges.push_back(Edge{ev.policy[id] / total, 0.0f, 0, -1, id}); });

    m_nodes[index].edgeCount = b.moveCount;
    return index;
}

uint32_t Search::selectEdge(uint32_t node) const
{
    const Node& n = m_nodes[node];

    // n.visits was already incremented for this simulation, so the bonus is
    // non-zero on a node's first descent and the priors alone decide.
    const float exploration = m_config.cPuct * std::sqrt(static_cast<float>(n.visits));

    const auto score = [&](uint32_t e) {
        const Edge& edge = m_edges[e];
        const float q = edge.visits > 0 ? edge.valueSum / static_cast<float>(edge.visits) : 0.0f;
        return q + exploration * edge.prior / static_cast<float>(1 + edge.visits);
    };

    return *std::ranges::max_element(std::views::iota(n.firstEdge, n.firstEdge + n.edgeCount), {}, score);
}

int32_t Search::childFor(uint16_t moveId) const
{
    const Node& root = m_nodes[0];
    for (uint32_t e = root.firstEdge; e < root.firstEdge + root.edgeCount; ++e)
        if (m_edges[e].moveId == moveId)
            return m_edges[e].child;
    return -1;
}

// Copies the subtree under `keep` into fresh vectors, renumbering as it goes, and
// leaves it as the whole tree with `keep` at index 0. One breadth-first pass does
// both jobs: a copied node's firstEdge still points into the old edge array until
// its own turn comes round, so nothing has to be visited twice.
void Search::keepSubtree(uint32_t keep)
{
    m_spareNodes.clear();
    m_spareEdges.clear();
    m_spareNodes.push_back(m_nodes[keep]);

    for (uint32_t n = 0; n < m_spareNodes.size(); ++n)
    {
        const uint32_t first = m_spareNodes[n].firstEdge;
        const uint32_t count = m_spareNodes[n].edgeCount;
        m_spareNodes[n].firstEdge = static_cast<uint32_t>(m_spareEdges.size());

        for (uint32_t e = 0; e < count; ++e)
        {
            Edge edge = m_edges[first + e];
            if (edge.child >= 0)
            {
                m_spareNodes.push_back(m_nodes[static_cast<size_t>(edge.child)]);
                edge.child = static_cast<int32_t>(m_spareNodes.size()) - 1;
            }
            m_spareEdges.push_back(edge);
        }
    }

    m_nodes.swap(m_spareNodes);
    m_edges.swap(m_spareEdges);
}

// A Dirichlet(alpha, ..., alpha) draw is independent Gamma(alpha, 1) draws
// normalised to sum to one.
void Search::addRootNoise()
{
    const Node& root = m_nodes[0];
    if (root.edgeCount == 0)
        return;

    std::gamma_distribution<float> gamma{m_config.noiseAlpha, 1.0f};
    m_noise.clear();
    float total = 0.0f;
    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        m_noise.push_back(gamma(m_rng));
        total += m_noise.back();
    }

    // alpha below 1 puts most draws near zero, so a whole set underflowing is
    // unlikely but not impossible; leaving the priors alone is the honest answer.
    if (total <= 0.0f)
        return;

    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        Edge& edge = m_edges[root.firstEdge + i];
        edge.prior = (1.0f - m_config.rootNoise) * edge.prior + m_config.rootNoise * (m_noise[i] / total);
    }
}

// Descends `wanted` times. At wanted == 1 - the shape self-play uses, since its
// batch comes from the other games in flight - this is one plain descent on
// statistics that are completely up to date.
//
// Above that a virtual loss is needed: each descent is provisionally recorded on
// the way down as having come back a loss, which is the only thing that makes the
// next one prefer somewhere else, and backUp() removes it before applying the real
// result. Two descents can still land on the same unexpanded edge. Both are kept:
// the board is evaluated twice, one of the two nodes ends up unreachable, and both
// back up the same correct value.
void Search::collect(int wanted)
{
    m_descents.clear();
    m_trailStore.clear();
    m_leaves.clear();

    for (int i = 0; i < wanted; ++i)
    {
        m_path.resize(m_baseLength);
        Descent descent{static_cast<uint32_t>(m_trailStore.size()), 0, -1, -1, 0.0f};

        uint32_t node = 0;
        for (;;)
        {
            ++m_nodes[node].visits;

            if (m_nodes[node].edgeCount == 0) {
                descent.value = terminalValue(m_nodes[node].board);
                break;
            }

            const uint32_t e = selectEdge(node);
            m_trailStore.push_back(e);

            if (wanted > 1)
            {
                m_edges[e].valueSum -= 1.0f;
                ++m_edges[e].visits;
            }

            if (m_edges[e].child < 0)
            {
                m_leaves.push_back(amoeba::apply(m_nodes[node].board, amoeba::Move::fromId(m_edges[e].moveId), m_path));
                descent.edge = static_cast<int32_t>(e);
                descent.leaf = static_cast<int32_t>(m_leaves.size()) - 1;
                break;
            }

            node = static_cast<uint32_t>(m_edges[e].child);
            m_path.push_back(m_nodes[node].board.hash);
        }

        descent.trailLength = static_cast<uint32_t>(m_trailStore.size()) - descent.trailStart;
        m_descents.push_back(descent);
    }
}

void Search::backUp(std::span<const Evaluation> evaluations)
{
    const bool virtualLoss = m_descents.size() > 1;

    for (const Descent& descent : m_descents)
    {
        float value = descent.value;

        if (descent.leaf >= 0)
        {
            const amoeba::Board leaf = m_leaves[static_cast<size_t>(descent.leaf)];
            const Evaluation& ev = evaluations[static_cast<size_t>(descent.leaf)];

            m_edges[static_cast<size_t>(descent.edge)].child =
                static_cast<int32_t>(expand(leaf, ev));

            // A terminal leaf is worth what the rules say, not what the network
            // guesses. It is still in the batch, because keeping the two arrays
            // parallel is worth more than the handful of wasted evaluations.
            value = leaf.state == amoeba::State::Ongoing ? ev.value : terminalValue(leaf);
        }

        for (uint32_t k = descent.trailLength; k-- > 0;)
        {
            const uint32_t e = m_trailStore[descent.trailStart + k];

            if (virtualLoss)
            {
                m_edges[e].valueSum += 1.0f;
                --m_edges[e].visits;
            }

            value = -value;
            m_edges[e].valueSum += value;
            ++m_edges[e].visits;
        }
    }
}

bool Search::spent() const
{
    const uint32_t through = m_nodes[0].visits;
    if (static_cast<int>(through) >= m_config.simulations)
        return true;

    // One round always runs, so a deadline shorter than a single evaluation still
    // leaves visit counts to read - unless a re-rooted tree arrived with some, in
    // which case there is nothing left to protect.
    return through > 0 && std::chrono::steady_clock::now() >= m_expiry;
}

void Search::restart(const amoeba::Board& root, std::span<const uint64_t> history)
{
    m_nodes.clear();
    m_edges.clear();
    m_nodes.reserve(static_cast<size_t>(m_config.simulations) + 1);

    m_path.assign(history.begin(), history.end());
    m_baseLength = m_path.size();
    m_rootBoard  = root;
    m_needsRoot  = true;
    m_timing     = false;

    // Nothing worth asking about a position the rules have already settled, and
    // nothing to search from it either.
    if (root.state != amoeba::State::Ongoing)
    {
        expand(root, Evaluation{});
        m_needsRoot = false;
    }
}

void Search::advance(uint16_t moveId, const amoeba::Board& next, std::span<const uint64_t> history)
{
    const int32_t child = m_needsRoot || m_nodes.empty() ? -1 : childFor(moveId);
    if (child < 0)
    {
        restart(next, history);
        return;
    }

    keepSubtree(static_cast<uint32_t>(child));
    m_path.assign(history.begin(), history.end());
    m_baseLength = m_path.size();
    m_rootBoard  = next;
    m_timing     = false;

    // Fresh noise on the new root. What it inherited are the network's own priors -
    // noise only ever went on the root above this one - and the handful of moves the
    // last search was told to promote should not go on being promoted here.
    if (m_config.rootNoise > 0.0f)
        addRootNoise();
}

std::span<const amoeba::Board* const> Search::pendingLeaves()
{
    m_leafPointers.clear();

    if (!m_timing)
    {
        m_expiry = std::chrono::steady_clock::now() + m_config.deadline;
        m_timing = true;
    }

    if (m_needsRoot)
    {
        m_leaves.assign(1, m_rootBoard);
        m_leafPointers.push_back(&m_leaves.front());
        return m_leafPointers;
    }

    while (m_nodes[0].edgeCount != 0 && !spent())
    {
        collect(std::min(std::max(1, m_config.batchSize),
                         m_config.simulations - static_cast<int>(m_nodes[0].visits)));

        if (!m_leaves.empty())
        {
            for (const amoeba::Board& leaf : m_leaves)
                m_leafPointers.push_back(&leaf);
            return m_leafPointers;
        }

        // Every descent in the round ended somewhere the rules had already settled,
        // so its results are already in hand and the next round can start at once.
        backUp({});
    }
    return m_leafPointers;
}

void Search::absorb(std::span<const Evaluation> evaluations)
{
    if (evaluations.size() != m_leafPointers.size())
        throw std::runtime_error("absorb() was given a different number of evaluations than pendingLeaves() asked for");

    if (m_needsRoot)
    {
        expand(m_rootBoard, evaluations.front());
        m_needsRoot = false;
        if (m_config.rootNoise > 0.0f)
            addRootNoise();
        return;
    }

    backUp(evaluations);
}

VisitCounts Search::visits() const
{
    VisitCounts counts{};
    if (m_needsRoot || m_nodes.empty())
        return counts;

    const Node& root = m_nodes[0];
    for (uint32_t e = root.firstEdge; e < root.firstEdge + root.edgeCount; ++e)
        counts[m_edges[e].moveId] = m_edges[e].visits;
    return counts;
}

VisitCounts runSearch(Search& search, Evaluator& evaluator)
{
    std::vector<Evaluation> evaluations;
    for (;;)
    {
        const std::span<const amoeba::Board* const> pending = search.pendingLeaves();
        if (pending.empty())
            return search.visits();

        evaluations.resize(pending.size());
        evaluator.evaluate(pending, evaluations);
        search.absorb(evaluations);
    }
}

} // namespace bot
