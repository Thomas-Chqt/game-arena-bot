#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>

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

} // namespace

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

uint16_t sampleMove(const VisitCounts& counts, float temperature, std::mt19937_64& rng)
{
    std::array<double, amoeba::kNumMoveIds> weights{};
    const double exponent = 1.0 / temperature;
    for (size_t id = 0; id < counts.size(); ++id)
        if (counts[id] > 0)
            weights[id] = std::pow(static_cast<double>(counts[id]), exponent);

    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    double       point = std::uniform_real_distribution<double>(0.0, total)(rng);

    for (size_t id = 0; id < weights.size(); ++id)
        if ((point -= weights[id]) < 0.0)
            return static_cast<uint16_t>(id);

    return bestMove(counts);   // only reachable if rounding eats the whole total
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

std::pair<uint32_t, float> Search::addNode(const amoeba::Board& b)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{b, static_cast<uint32_t>(m_edges.size()), 0, 0});

    if (b.state != amoeba::State::Ongoing)
        return {index, terminalValue(b)};

    const amoeba::Board* const one[]{&b};
    Evaluation ev;
    m_evaluator.evaluate(one, std::span{&ev, 1});

    float total = 0.0f;
    b.forEachLegal([&](uint16_t id) { total += ev.policy[id]; });
    b.forEachLegal([&](uint16_t id) { m_edges.push_back(Edge{ev.policy[id] / total, 0.0f, 0, -1, id}); });

    m_nodes[index].edgeCount = b.moveCount;
    return {index, ev.value};
}

void Search::addRootNoise()
{
    const Node& root = m_nodes[0];
    std::gamma_distribution<float> gamma(m_config.rootNoiseAlpha, 1.0f);

    m_noise.clear();
    double total = 0.0;
    for (uint32_t i = 0; i < root.edgeCount; ++i)
        total += m_noise.emplace_back(gamma(m_rng));

    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        Edge& e = m_edges[root.firstEdge + i];
        e.prior = (1.0f - m_config.rootNoiseWeight) * e.prior
                + m_config.rootNoiseWeight * static_cast<float>(m_noise[i] / total);
    }
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

VisitCounts Search::run(const amoeba::Board& root, std::span<const uint64_t> history)
{
    m_nodes.clear();
    m_edges.clear();
    m_nodes.reserve(static_cast<size_t>(m_config.simulations) + 1);

    m_path.assign(history.begin(), history.end());
    const size_t baseLength = m_path.size();

    addNode(root);
    if (m_config.rootNoiseWeight > 0.0f)
        addRootNoise();

    const auto expiry = std::chrono::steady_clock::now() + m_config.deadline;

    for (int sim = 0; sim < m_config.simulations; ++sim)
    {
        // One clock read per simulation is nothing next to an evaluation, and
        // the alternative - checking every k - overshoots by a whole batch. The
        // first simulation always runs: an empty set of visit counts has no
        // move in it to play and no distribution in it to train on.
        if (sim > 0 && std::chrono::steady_clock::now() >= expiry)
            break;

        m_path.resize(baseLength);
        m_trail.clear();

        uint32_t node  = 0;
        float    value = 0.0f;

        for (;;)
        {
            ++m_nodes[node].visits;

            if (m_nodes[node].edgeCount == 0) {
                value = terminalValue(m_nodes[node].board);
                break;
            }

            const uint32_t e = selectEdge(node);
            m_trail.push_back(e);

            if (m_edges[e].child < 0)
            {
                const amoeba::Board leaf = amoeba::apply(m_nodes[node].board, amoeba::Move::fromId(m_edges[e].moveId), m_path);
                const auto [child, leafValue] = addNode(leaf);
                m_edges[e].child = static_cast<int32_t>(child);
                value = leafValue;
                break;
            }

            node = static_cast<uint32_t>(m_edges[e].child);
            m_path.push_back(m_nodes[node].board.hash);
        }

        for (const uint32_t e : m_trail | std::views::reverse)
        {
            value = -value;
            m_edges[e].valueSum += value;
            ++m_edges[e].visits;
        }
    }

    VisitCounts counts{};
    const Node& r = m_nodes[0];
    for (uint32_t e = r.firstEdge; e < r.firstEdge + r.edgeCount; ++e)
        counts[m_edges[e].moveId] = m_edges[e].visits;
    return counts;
}

} // namespace bot
