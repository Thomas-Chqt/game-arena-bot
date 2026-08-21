#include "mcts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace bot
{

namespace
{

// A body running on the pool must not wait for the pool: the round it would be
// waiting on cannot start until it returns.
thread_local bool threadIsInsidePool = false;

} // namespace

// ---------------------------------------------------------------------------

ThreadPool& ThreadPool::global()
{
    static ThreadPool pool{std::max(1u, std::thread::hardware_concurrency())};
    return pool;
}

ThreadPool::ThreadPool(unsigned width)
{
    // One of the threads is whoever calls forEach, so only width - 1 are hired.
    for (unsigned i = 1; i < std::max(1u, width); ++i)
        m_workers.emplace_back([this] { workerLoop(); });
}

ThreadPool::~ThreadPool()
{
    {
        const std::lock_guard guard{m_mutex};
        m_stopping = true;
    }
    m_workAvailable.notify_all();
    for (std::thread& worker : m_workers)
        worker.join();
}

void ThreadPool::forEach(size_t taskCount, const std::function<void(size_t)>& task)
{
    if (taskCount == 0)
        return;

    if (threadIsInsidePool || m_workers.empty())
    {
        for (size_t taskIndex = 0; taskIndex < taskCount; ++taskIndex)
            task(taskIndex);
        return;
    }

    {
        const std::lock_guard guard{m_mutex};
        m_task = &task;
        m_taskCount = taskCount;
        m_nextTaskIndex.store(0, std::memory_order_relaxed);
        m_activeWorkerCount = static_cast<unsigned>(m_workers.size());
        ++m_roundId;
    }
    m_workAvailable.notify_all();

    runAvailableTasks();

    std::unique_lock lock{m_mutex};
    m_roundFinished.wait(lock, [this] { return m_activeWorkerCount == 0; });
    m_task = nullptr;

    if (m_taskFailure)
        std::rethrow_exception(std::exchange(m_taskFailure, nullptr));
}

void ThreadPool::runAvailableTasks()
{
    threadIsInsidePool = true;
    try
    {
        for (;;)
        {
            const size_t index = m_nextTaskIndex.fetch_add(1, std::memory_order_relaxed);
            if (index >= m_taskCount)
                break;
            (*m_task)(index);
        }
    }
    catch (...)
    {
        // Whoever else is still running finishes the index it is on and then stops;
        // the round has to complete either way, or forEach would go on waiting for a
        // thread that has given up.
        m_nextTaskIndex.store(m_taskCount, std::memory_order_relaxed);
        const std::lock_guard guard{m_mutex};
        if (!m_taskFailure)
            m_taskFailure = std::current_exception();
    }
    threadIsInsidePool = false;
}

void ThreadPool::workerLoop()
{
    uint64_t observedRoundId = 0;
    for (;;)
    {
        std::unique_lock lock{m_mutex};
        m_workAvailable.wait(lock, [this, &observedRoundId] { return m_stopping || m_roundId != observedRoundId; });
        if (m_stopping)
            return;
        observedRoundId = m_roundId;
        lock.unlock();

        runAvailableTasks();

        lock.lock();
        if (--m_activeWorkerCount == 0)
            m_roundFinished.notify_one();
    }
}

// ---------------------------------------------------------------------------

float outcomeFor(amoeba::Outcome outcome, bool whiteToMove)
{
    if (outcome == amoeba::Outcome::draw)
        return 0.0f;
    return (outcome == amoeba::Outcome::whiteWins) == whiteToMove ? 1.0f : -1.0f;
}

uint16_t randomLegalMove(const amoeba::Board& board, std::mt19937_64& randomEngine)
{
    assert(board.legalMoveCount > 0);

    int remaining = std::uniform_int_distribution<int>(0, board.legalMoveCount - 1)(randomEngine);
    uint16_t chosenMove = 0;
    board.forEachLegal(
        [&](uint16_t moveId)
        {
            if (remaining-- == 0)
                chosenMove = moveId;
        });
    return chosenMove;
}

uint16_t bestMove(const VisitCounts& counts)
{
    return static_cast<uint16_t>(std::ranges::max_element(counts) - counts.begin());
}

// ---------------------------------------------------------------------------

void RolloutEvaluator::evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> outputs)
{
    for (size_t i = 0; i < boards.size(); ++i)
    {
        outputs[i].policy.fill(1.0f);
        outputs[i].value = playout(*boards[i]);
    }
}

float RolloutEvaluator::playout(amoeba::Board board)
{
    const bool originalMoverIsWhite = board.whiteToMove;

    // No history, so repetitions go unseen; the staleness and move caps still
    // end every playout, and a random game's verdict is too crude for one
    // missed draw to matter.
    for (;;)
    {
        amoeba::MoveResult result =
            amoeba::applyMove(board, amoeba::Move::fromId(randomLegalMove(board, m_randomEngine)));
        if (const auto* outcome = std::get_if<amoeba::Outcome>(&result))
            return outcomeFor(*outcome, originalMoverIsWhite);
        board = std::get<amoeba::Board>(std::move(result));
    }
}

// ---------------------------------------------------------------------------

// Appends an ongoing node and its outgoing edges from an evaluation already in hand.
uint32_t Search::addNode(const amoeba::Board& board, const Evaluation& evaluation)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{OngoingNode{board, static_cast<uint32_t>(m_edges.size()), 0}, 0});

    float totalPrior = 0.0f;
    board.forEachLegal([&](uint16_t moveId) { totalPrior += evaluation.policy[moveId]; });
    assert(totalPrior > 0.0f);
    board.forEachLegal([&](uint16_t moveId)
                       { m_edges.push_back(Edge{evaluation.policy[moveId] / totalPrior, 0.0f, 0, -1, moveId}); });

    std::get<OngoingNode>(m_nodes[index].contents).edgeCount = board.legalMoveCount;
    return index;
}

uint32_t Search::addTerminalNode(float value)
{
    const uint32_t index = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{value, 0});
    return index;
}

uint32_t Search::selectEdgeToExplore(uint32_t node) const
{
    const Node& currentNode = m_nodes[node];
    const OngoingNode& ongoing = std::get<OngoingNode>(currentNode.contents);

    // currentNode.visits was already incremented for this simulation, so the bonus is
    // non-zero on a node's first descent and the priors alone decide.
    const float exploration = m_config.explorationConstant * std::sqrt(static_cast<float>(currentNode.visits));

    const auto score = [&](uint32_t edgeIndex)
    {
        const Edge& edge = m_edges[edgeIndex];
        const float meanValue = edge.visits > 0 ? edge.valueSum / static_cast<float>(edge.visits) : 0.0f;
        return meanValue + exploration * edge.prior / static_cast<float>(1 + edge.visits);
    };

    return *std::ranges::max_element(
        std::views::iota(ongoing.firstEdgeIndex, ongoing.firstEdgeIndex + ongoing.edgeCount), {}, score);
}

int32_t Search::findRootChild(uint16_t moveId) const
{
    const OngoingNode& root = std::get<OngoingNode>(m_nodes[0].contents);
    for (uint32_t edgeIndex = root.firstEdgeIndex; edgeIndex < root.firstEdgeIndex + root.edgeCount; ++edgeIndex)
    {
        if (m_edges[edgeIndex].moveId == moveId)
            return m_edges[edgeIndex].childNode;
    }
    return -1;
}

// Copies the subtree under `keep` into fresh vectors, renumbering as it goes, and
// leaves it as the whole tree with `keep` at index 0. One breadth-first pass does
// both jobs: a copied node's firstEdgeIndex still points into the old edge array until
// its own turn comes round, so nothing has to be visited twice.
void Search::retainSubtree(uint32_t nodeToKeep)
{
    m_spareNodes.clear();
    m_spareEdges.clear();
    m_spareNodes.push_back(m_nodes[nodeToKeep]);

    for (uint32_t nodeIndex = 0; nodeIndex < m_spareNodes.size(); ++nodeIndex)
    {
        OngoingNode* ongoing = std::get_if<OngoingNode>(&m_spareNodes[nodeIndex].contents);
        if (ongoing == nullptr)
            continue;

        const uint32_t oldFirstEdge = ongoing->firstEdgeIndex;
        const uint32_t edgeCount = ongoing->edgeCount;
        ongoing->firstEdgeIndex = static_cast<uint32_t>(m_spareEdges.size());

        for (uint32_t edgeOffset = 0; edgeOffset < edgeCount; ++edgeOffset)
        {
            Edge edge = m_edges[oldFirstEdge + edgeOffset];
            if (edge.childNode >= 0)
            {
                m_spareNodes.push_back(m_nodes[static_cast<size_t>(edge.childNode)]);
                edge.childNode = static_cast<int32_t>(m_spareNodes.size()) - 1;
            }
            m_spareEdges.push_back(edge);
        }
    }

    m_nodes.swap(m_spareNodes);
    m_edges.swap(m_spareEdges);
}

// A Dirichlet(alpha, ..., alpha) draw is independent Gamma(alpha, 1) draws
// normalised to sum to one.
void Search::addExplorationNoise()
{
    const OngoingNode& root = std::get<OngoingNode>(m_nodes[0].contents);
    if (root.edgeCount == 0)
        return;

    std::gamma_distribution<float> gamma{m_config.noiseAlpha, 1.0f};
    m_rootNoiseValues.clear();
    float total = 0.0f;
    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        m_rootNoiseValues.push_back(gamma(m_randomEngine));
        total += m_rootNoiseValues.back();
    }

    // alpha below 1 puts most draws near zero, so a whole set underflowing is
    // unlikely but not impossible; leaving the priors alone is the honest answer.
    if (total <= 0.0f)
        return;

    for (uint32_t i = 0; i < root.edgeCount; ++i)
    {
        Edge& edge = m_edges[root.firstEdgeIndex + i];
        edge.prior = (1.0f - m_config.rootNoise) * edge.prior + m_config.rootNoise * (m_rootNoiseValues[i] / total);
    }
}

// Descends `requestedLeafCount` times. At requestedLeafCount == 1 - the shape self-play uses, since its
// batch comes from the other games in flight - this is one plain descent on
// statistics that are completely up to date.
//
// Above that a virtual loss is needed: each descent is provisionally recorded on
// the way down as having come back a loss, which is the only thing that makes the
// next one prefer somewhere else, and backpropagate() removes it before applying the real
// result. Two descents can still land on the same unexpanded edge. Both are kept:
// the board is evaluated twice, one of the two nodes ends up unreachable, and both
// back up the same correct value.
void Search::collectPendingLeaves(int requestedLeafCount)
{
    m_descents.clear();
    m_edgeTrails.clear();
    m_pendingBoards.clear();

    for (int descentIndex = 0; descentIndex < requestedLeafCount; ++descentIndex)
    {
        m_positionHistory.resize(m_gameHistoryLength);
        Descent descent{static_cast<uint32_t>(m_edgeTrails.size()), 0, -1, -1, 0.0f};

        uint32_t node = 0;
        for (;;)
        {
            Node& currentNode = m_nodes[node];
            ++currentNode.visits;

            const OngoingNode* ongoing = std::get_if<OngoingNode>(&currentNode.contents);
            if (ongoing == nullptr)
            {
                descent.terminalValue = std::get<float>(currentNode.contents);
                break;
            }

            const uint32_t edgeIndex = selectEdgeToExplore(node);
            m_edgeTrails.push_back(edgeIndex);

            if (requestedLeafCount > 1)
            {
                m_edges[edgeIndex].valueSum -= 1.0f;
                ++m_edges[edgeIndex].visits;
            }

            if (m_edges[edgeIndex].childNode < 0)
            {
                const amoeba::Move move = amoeba::Move::fromId(m_edges[edgeIndex].moveId);
                amoeba::MoveResult result = amoeba::applyMove(ongoing->board, move, m_positionHistory);
                if (const auto* outcome = std::get_if<amoeba::Outcome>(&result))
                {
                    // A terminal child is valued for the player who would have moved
                    // next there, matching every other node value in the tree.
                    descent.terminalValue = outcomeFor(*outcome, !ongoing->board.whiteToMove);
                    m_edges[edgeIndex].childNode = static_cast<int32_t>(addTerminalNode(descent.terminalValue));
                }
                else
                {
                    m_pendingBoards.push_back(std::get<amoeba::Board>(std::move(result)));
                    descent.expandedEdgeIndex = static_cast<int32_t>(edgeIndex);
                    descent.pendingBoardIndex = static_cast<int32_t>(m_pendingBoards.size()) - 1;
                }
                break;
            }

            node = static_cast<uint32_t>(m_edges[edgeIndex].childNode);
            if (const OngoingNode* child = std::get_if<OngoingNode>(&m_nodes[node].contents))
                m_positionHistory.push_back(child->board.positionHash);
        }

        descent.edgeTrailLength = static_cast<uint32_t>(m_edgeTrails.size()) - descent.edgeTrailStart;
        m_descents.push_back(descent);
    }
}

void Search::backpropagate(std::span<const Evaluation> evaluations)
{
    const bool virtualLoss = m_descents.size() > 1;

    for (const Descent& descent : m_descents)
    {
        float value = descent.terminalValue;

        if (descent.pendingBoardIndex >= 0)
        {
            const amoeba::Board& leaf = m_pendingBoards[static_cast<size_t>(descent.pendingBoardIndex)];
            const Evaluation& evaluation = evaluations[static_cast<size_t>(descent.pendingBoardIndex)];

            m_edges[static_cast<size_t>(descent.expandedEdgeIndex)].childNode =
                static_cast<int32_t>(addNode(leaf, evaluation));

            value = evaluation.value;
        }

        for (uint32_t trailOffset = descent.edgeTrailLength; trailOffset-- > 0;)
        {
            const uint32_t edgeIndex = m_edgeTrails[descent.edgeTrailStart + trailOffset];

            if (virtualLoss)
            {
                m_edges[edgeIndex].valueSum += 1.0f;
                --m_edges[edgeIndex].visits;
            }

            value = -value;
            m_edges[edgeIndex].valueSum += value;
            ++m_edges[edgeIndex].visits;
        }
    }
}

bool Search::hasSpentBudget() const
{
    const uint32_t through = m_nodes[0].visits;
    if (static_cast<int>(through) >= m_config.simulations)
        return true;

    // One round always runs, so a deadline shorter than a single evaluation still
    // leaves visit counts to read - unless a re-rooted tree arrived with some, in
    // which case there is nothing left to protect.
    return through > 0 && std::chrono::steady_clock::now() >= m_deadline;
}

void Search::restart(const amoeba::Board& root, std::span<const uint64_t> history)
{
    assert(m_config.simulations > 0);
    assert(m_config.batchSize > 0);

    m_nodes.clear();
    m_edges.clear();
    m_nodes.reserve(static_cast<size_t>(m_config.simulations) + 1);

    m_positionHistory.assign(history.begin(), history.end());
    m_gameHistoryLength = m_positionHistory.size();
    m_rootBoard = root;
    m_rootNeedsEvaluation = true;
    m_searchStarted = false;

}

void Search::advance(uint16_t moveId, const amoeba::Board& next, std::span<const uint64_t> history)
{
    const int32_t childNode = m_rootNeedsEvaluation || m_nodes.empty() ? -1 : findRootChild(moveId);
    if (childNode < 0)
    {
        restart(next, history);
        return;
    }

    retainSubtree(static_cast<uint32_t>(childNode));
    m_positionHistory.assign(history.begin(), history.end());
    m_gameHistoryLength = m_positionHistory.size();
    m_rootBoard = next;
    m_searchStarted = false;

    // Fresh noise on the new root. What it inherited are the network's own priors -
    // noise only ever went on the root above this one - and the handful of moves the
    // last search was told to promote should not go on being promoted here.
    if (m_config.rootNoise > 0.0f)
        addExplorationNoise();
}

std::span<const amoeba::Board* const> Search::pendingLeaves()
{
    m_pendingBoardPointers.clear();

    if (!m_searchStarted)
    {
        m_deadline = std::chrono::steady_clock::now() + m_config.deadline;
        m_searchStarted = true;
    }

    if (m_rootNeedsEvaluation)
    {
        m_pendingBoards.assign(1, m_rootBoard);
        m_pendingBoardPointers.push_back(&m_pendingBoards.front());
        return m_pendingBoardPointers;
    }

    while (std::holds_alternative<OngoingNode>(m_nodes[0].contents) && !hasSpentBudget())
    {
        collectPendingLeaves(
            std::min(std::max(1, m_config.batchSize), m_config.simulations - static_cast<int>(m_nodes[0].visits)));

        if (!m_pendingBoards.empty())
        {
            for (const amoeba::Board& pendingBoard : m_pendingBoards)
                m_pendingBoardPointers.push_back(&pendingBoard);
            return m_pendingBoardPointers;
        }

        // Every descent in the round ended somewhere the rules had already settled,
        // so its results are already in hand and the next round can start at once.
        backpropagate({});
    }
    return m_pendingBoardPointers;
}

void Search::absorb(std::span<const Evaluation> evaluations)
{
    if (evaluations.size() != m_pendingBoardPointers.size())
        throw std::runtime_error("absorb() was given a different number of evaluations than pendingLeaves() asked for");

    if (m_rootNeedsEvaluation)
    {
        addNode(m_rootBoard, evaluations.front());
        m_rootNeedsEvaluation = false;
        if (m_config.rootNoise > 0.0f)
            addExplorationNoise();
        return;
    }

    backpropagate(evaluations);
}

VisitCounts Search::visits() const
{
    VisitCounts counts{};
    if (m_rootNeedsEvaluation || m_nodes.empty())
        return counts;

    const OngoingNode& root = std::get<OngoingNode>(m_nodes[0].contents);
    for (uint32_t edgeIndex = root.firstEdgeIndex; edgeIndex < root.firstEdgeIndex + root.edgeCount; ++edgeIndex)
        counts[m_edges[edgeIndex].moveId] = m_edges[edgeIndex].visits;
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
