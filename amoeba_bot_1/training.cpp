#include "training.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace bot
{

Batch makeBatch(std::span<const amoeba::Board* const> boards, std::span<const VisitCounts> visits, std::span<const float> outcomes)
{
    if (boards.size() != visits.size() || boards.size() != outcomes.size())
        throw std::runtime_error(std::format("makeBatch got {} boards, {} visit sets and {} outcomes", boards.size(), visits.size(), outcomes.size()));

    const int batch = static_cast<int>(boards.size());

    std::vector<float> input(static_cast<size_t>(batch) * amoeba::kEncodedSize);
    std::vector<float> legal(static_cast<size_t>(batch) * amoeba::kNumMoveIds);
    std::vector<float> policy(static_cast<size_t>(batch) * amoeba::kNumMoveIds);

    for (int i = 0; i < batch; ++i)
    {
        const size_t inputBase = static_cast<size_t>(i) * amoeba::kEncodedSize;
        amoeba::encode(*boards[i], std::span<float, amoeba::kEncodedSize>(input.data() + inputBase, amoeba::kEncodedSize));

        uint64_t total = 0;
        for (const uint32_t count : visits[i]) {
            total += count;
        }
        if (total == 0)
            throw std::runtime_error(std::format("sample {} has no visits, so it has no policy to learn", i));

        // The search counts moves in absolute ids and forward() answers in the
        // flipped space encode() used, so the target has to cross over. Same table
        // in both directions - it is its own inverse.
        const std::span<const uint16_t, amoeba::kNumMoveIds> toAbsolute = amoeba::policyToAbsolute(boards[i]->whiteToMove);
        const size_t base = static_cast<size_t>(i) * amoeba::kNumMoveIds;

        for (int slot = 0; slot < amoeba::kNumMoveIds; ++slot)
        {
            const uint16_t id = toAbsolute[slot];
            legal[base + slot] = boards[i]->isLegal(id) ? 1.0f : 0.0f;
            policy[base + slot] = static_cast<float>(visits[i][id]) / static_cast<float>(total);
        }
    }

    return {
        mlx::core::array(input.data(), mlx::core::Shape{batch, amoeba::kEncodedSize}, mlx::core::float32),
        mlx::core::array(legal.data(), mlx::core::Shape{batch, amoeba::kNumMoveIds}, mlx::core::float32),
        mlx::core::array(policy.data(), mlx::core::Shape{batch, amoeba::kNumMoveIds}, mlx::core::float32),
        mlx::core::array(outcomes.data(), mlx::core::Shape{batch}, mlx::core::float32)
    };
}

std::vector<mlx::core::array> loss(const std::vector<mlx::core::array>& params, NetworkShape shape, const Batch& batch, float weightDecay)
{
    const Prediction prediction = forward(params, shape, batch.input);

    // Illegal moves go to a large finite penalty, so they end up with no
    // probability and no gradient is spent teaching the network to avoid them.
    // Finite rather than -inf because a position with no legal moves at all would
    // otherwise softmax to NaN.
    const mlx::core::array masked = mlx::core::where(
        mlx::core::greater(batch.legal, mlx::core::array(0.0f)), prediction.policy, mlx::core::array(-1e9f)
    );

    // log(softmax(x)) written as x - logsumexp(x): exponentiating first overflows.
    const mlx::core::array logProbability = masked - mlx::core::logsumexp(masked, -1, true);

    // Cross-entropy against the whole visit distribution, not just the best move.
    // "the search spent 40% of its thinking here and 15% there" carries far more
    // than "this one won".
    const mlx::core::array policyLoss = mlx::core::mean(-mlx::core::sum(batch.policyTarget * logProbability, -1));
    const mlx::core::array valueLoss = mlx::core::mean(mlx::core::square(prediction.value - batch.valueTarget));

    mlx::core::array total = policyLoss + valueLoss;

    if (weightDecay > 0.0f)
    {
        mlx::core::array penalty = mlx::core::array(0.0f);
        for (const mlx::core::array& tensor : params) {
            penalty = penalty + mlx::core::sum(mlx::core::square(tensor));
        }
        total = total + penalty * weightDecay;
    }

    return {total, policyLoss, valueLoss};
}

Adam::Adam(const std::vector<mlx::core::array>& params, AdamConfig config)
    : m_config(config)
{
    for (const mlx::core::array& tensor : params) {
        m_mean.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
        m_variance.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
    }
}

std::vector<mlx::core::array> Adam::step(const std::vector<mlx::core::array>& params, const std::vector<mlx::core::array>& gradients, float rate)
{
    if (params.size() != m_mean.size() || params.size() != gradients.size())
        throw std::runtime_error(std::format("Adam was built for {} tensors but got {} parameters and {} gradients", m_mean.size(), params.size(), gradients.size()));
    ++m_steps;

    // Both averages start at zero, so for the first few hundred steps they read
    // far too low - and the variance more severely than the mean, which would make
    // the steps enormous rather than merely small. These undo that exactly, and
    // fade to 1 as the averages fill up.
    const float meanScale = 1.0f - std::pow(m_config.meanDecay, m_steps);
    const float varianceScale = 1.0f - std::pow(m_config.varianceDecay, m_steps);

    std::vector<mlx::core::array> updated;
    updated.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i)
    {
        m_mean[i] = m_mean[i] * m_config.meanDecay + gradients[i] * (1.0f - m_config.meanDecay);
        m_variance[i] = m_variance[i] * m_config.varianceDecay + mlx::core::square(gradients[i]) * (1.0f - m_config.varianceDecay);

        const mlx::core::array typical = mlx::core::sqrt(m_variance[i] / varianceScale) + m_config.epsilon;
        updated.push_back(params[i] - (m_mean[i] / meanScale) / typical * rate);
    }
    return updated;
}

} // namespace bot
