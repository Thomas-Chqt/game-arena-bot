#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>

namespace bot
{

namespace
{

// The result of a finished game seen by whoever is to move in it - which, at a
// terminal position, is the side that was just mated or ran out of moves.
float terminalValue(const amoeba::Board& b)
{
    if (b.state == amoeba::State::Draw)
        return 0.0f;
    return (b.state == amoeba::State::WhiteWins) == b.whiteToMove ? 1.0f : -1.0f;
}

} // namespace

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

    if (b.state == amoeba::State::Draw)
        return 0.0f;
    return (b.state == amoeba::State::WhiteWins) == mover ? 1.0f : -1.0f;
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

// The root, which has to be evaluated on its own before anything can descend
// through it.
std::pair<uint32_t, float> Search::addNode(const amoeba::Board& b)
{
    if (b.state != amoeba::State::Ongoing)
        return {expand(b, Evaluation{}), terminalValue(b)};

    const amoeba::Board* const one[]{&b};
    Evaluation ev;
    m_evaluator.evaluate(one, std::span{&ev, 1});
    return {expand(b, ev), ev.value};
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

// Descends `wanted` times, applying a virtual loss on the way down so that each
// descent prefers somewhere the previous ones have not been - without it every
// descent in a batch would follow the same path and return the same leaf.
//
// Two descents can still land on the same unexpanded edge. Both are kept: the
// board is evaluated twice, one of the two nodes ends up unreachable, and both
// back up the same correct value. Sharing one evaluation between them would need
// the backup to know about the pairing, which is more machinery than an
// occasional wasted slot is worth.
void Search::collect(size_t baseLength, int wanted)
{
    m_descents.clear();
    m_trailStore.clear();
    m_leaves.clear();

    for (int i = 0; i < wanted; ++i)
    {
        m_path.resize(baseLength);
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

            // The virtual loss itself: this simulation is provisionally recorded as
            // having come back a loss, and backUp() removes it before applying the
            // real result.
            m_edges[e].valueSum -= 1.0f;
            ++m_edges[e].visits;

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

void Search::backUp()
{
    for (const Descent& descent : m_descents)
    {
        float value = descent.value;

        if (descent.leaf >= 0)
        {
            const amoeba::Board leaf = m_leaves[static_cast<size_t>(descent.leaf)];
            const Evaluation& ev = m_evaluations[static_cast<size_t>(descent.leaf)];

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

            m_edges[e].valueSum += 1.0f;
            --m_edges[e].visits;

            value = -value;
            m_edges[e].valueSum += value;
            ++m_edges[e].visits;
        }
    }
}

VisitCounts Search::run(const amoeba::Board& root, std::span<const uint64_t> history)
{
    m_nodes.clear();
    m_edges.clear();
    m_nodes.reserve(static_cast<size_t>(m_config.simulations) + 1);

    m_path.assign(history.begin(), history.end());
    const size_t baseLength = m_path.size();

    addNode(root);
    if (m_config.rootNoise > 0.0f)
        addRootNoise();

    for (int done = 0; done < m_config.simulations;)
    {
        collect(baseLength, std::min(std::max(1, m_config.batchSize), m_config.simulations - done));

        m_leafPointers.clear();
        for (const amoeba::Board& leaf : m_leaves) {
            m_leafPointers.push_back(&leaf);
        }
        m_evaluations.resize(m_leaves.size());
        if (!m_leaves.empty())
            m_evaluator.evaluate(m_leafPointers, m_evaluations);

        backUp();
        done += static_cast<int>(m_descents.size());
    }

    VisitCounts counts{};
    const Node& r = m_nodes[0];
    for (uint32_t e = r.firstEdge; e < r.firstEdge + r.edgeCount; ++e)
        counts[m_edges[e].moveId] = m_edges[e].visits;
    return counts;
}

} // namespace bot
