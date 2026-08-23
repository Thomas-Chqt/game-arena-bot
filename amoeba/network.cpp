#include "network.hpp"

#include <cassert>
#include <format>
#include <numeric>
#include <stdexcept>

namespace amoeba_bot
{

// Module composition produces names such as blocks.2.attention.query.weight.
// These names make the flat tensor vector readable and form the checkpoint keys.
std::string childName(std::string_view parent, std::string_view child)
{
    return parent.empty() ? std::string{child} : std::format("{}.{}", parent, child);
}

std::string indexedName(std::string_view parent, size_t index)
{
    return childName(parent, std::format("{}", index));
}

Adam::Adam(const std::vector<mlx::core::array>& parameters, AdamConfig config)
    : m_config(config)
{
    // Adam keeps two tensors with the same shape as every trained parameter:
    // a moving mean of gradients and a moving mean of squared gradients.
    for (const mlx::core::array& tensor : parameters)
    {
        m_mean.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
        m_variance.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
    }
}

std::vector<mlx::core::array> Adam::updateParameters(const std::vector<mlx::core::array>& parameters,
                                                      const std::vector<mlx::core::array>& gradients, float rate)
{
    if (parameters.size() != m_mean.size() || parameters.size() != gradients.size())
        throw std::runtime_error(std::format("Adam was built for {} tensors but got {} parameters and {} gradients",
                                             m_mean.size(), parameters.size(), gradients.size()));
    ++m_steps;

    // Both moving averages start at zero. These factors remove that initial bias
    // toward zero, which would otherwise make the first updates too small.
    const float meanScale = 1.0f - std::pow(m_config.meanDecay, m_steps);
    const float varianceScale = 1.0f - std::pow(m_config.varianceDecay, m_steps);

    std::vector<mlx::core::array> updated;
    updated.reserve(parameters.size());
    for (size_t index = 0; index < parameters.size(); ++index)
    {
        // Smooth noisy minibatch gradients into a direction and typical size.
        m_mean[index] = m_mean[index] * m_config.meanDecay + gradients[index] * (1.0f - m_config.meanDecay);
        m_variance[index] = m_variance[index] * m_config.varianceDecay
            + mlx::core::square(gradients[index]) * (1.0f - m_config.varianceDecay);

        // Divide by the typical gradient magnitude before taking the step.
        // epsilon only prevents division by zero for a parameter with no gradient.
        const mlx::core::array typicalGradient =
            mlx::core::sqrt(m_variance[index] / varianceScale) + m_config.epsilon;
        updated.push_back(parameters[index] - (m_mean[index] / meanScale) / typicalGradient * rate);
    }
    return updated;
}

void NetworkEvaluator::evaluate(std::span<const Board* const> boards, std::span<Evaluation> outputs)
{
    assert(boards.size() == outputs.size());
    const int batchSize = static_cast<int>(boards.size());

    // Collect several MCTS leaf positions into one tensor so MLX evaluates them
    // together on the GPU rather than launching one network call per board.
    std::vector<float> encodedBoards(static_cast<size_t>(batchSize) * encodedBoardSize);
    std::vector<float> legalMoveMask(static_cast<size_t>(batchSize) * moveIdCount);

    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        const size_t encodedBoardOffset = boardIndex * encodedBoardSize;
        const size_t maskOffset = boardIndex * moveIdCount;

        encodeBoard(*boards[boardIndex], std::span<float, encodedBoardSize>(
                                                      encodedBoards.data() + encodedBoardOffset,
                                                      encodedBoardSize));

        const std::span<const uint16_t, moveIdCount> moveIdsByPolicyIndex =
            policyIndicesToMoveIds(boards[boardIndex]->whiteToMove);
        for (int policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
            legalMoveMask[maskOffset + policyIndex] =
                boards[boardIndex]->isLegal(moveIdsByPolicyIndex[policyIndex]) ? 1.0f : 0.0f;
    }

    const mlx::core::array input(encodedBoards.data(), mlx::core::Shape{batchSize, encodedBoardSize},
                          mlx::core::float32);
    const std::vector<mlx::core::array> networkOutput = m_network(input);
    assert(networkOutput.size() == 2);
    const Prediction prediction{networkOutput[0], networkOutput[1]};

    const mlx::core::array legal(legalMoveMask.data(), mlx::core::Shape{batchSize, moveIdCount},
                          mlx::core::float32);
    // The network deliberately predicts all 444 move ids. Illegal logits receive
    // a huge negative value before softmax, making them effectively zero whenever
    // the position has at least one legal move.
    const mlx::core::array masked = mlx::core::where(
        mlx::core::greater(legal, mlx::core::array(0.0f)), prediction.policy,
        mlx::core::array(-1e9f));
    const mlx::core::array probabilities = mlx::core::softmax(masked, -1);
    mlx::core::eval({probabilities, prediction.value});

    const float* policyData = probabilities.data<float>();
    const float* valueData = prediction.value.data<float>();

    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        const std::span<const uint16_t, moveIdCount> moveIdsByPolicyIndex =
            policyIndicesToMoveIds(boards[boardIndex]->whiteToMove);
        const size_t policyOffset = boardIndex * moveIdCount;
        // encodeBoard may rotate the position into the side-to-move perspective.
        // Convert policy indices back to absolute Move::id values for MCTS.
        for (int policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
            outputs[boardIndex].policy[moveIdsByPolicyIndex[policyIndex]] = policyData[policyOffset + policyIndex];
        outputs[boardIndex].value = valueData[boardIndex];
    }
}

TrainingBatch makeTrainingBatch(std::span<const Board* const> boards,
                                std::span<const VisitCounts> visits,
                                std::span<const float> outcomes)
{
    if (boards.size() != visits.size() || boards.size() != outcomes.size())
        throw std::runtime_error(std::format("makeTrainingBatch got {} boards, {} visit sets and {} outcomes",
                                             boards.size(), visits.size(), outcomes.size()));

    const int batch = static_cast<int>(boards.size());
    std::vector<float> input(static_cast<size_t>(batch) * encodedBoardSize);
    std::vector<float> legal(static_cast<size_t>(batch) * moveIdCount);
    std::vector<float> policy(static_cast<size_t>(batch) * moveIdCount);

    for (size_t sampleIndex = 0; sampleIndex < boards.size(); ++sampleIndex)
    {
        const size_t inputOffset = sampleIndex * encodedBoardSize;
        encodeBoard(*boards[sampleIndex], std::span<float, encodedBoardSize>(
                                                      input.data() + inputOffset, encodedBoardSize));

        // AlphaZero does not train policy on only the selected move. It normalizes
        // all MCTS visit counts into a probability distribution, preserving how
        // strongly the search preferred each explored move.
        const uint64_t totalVisits = std::accumulate(visits[sampleIndex].begin(), visits[sampleIndex].end(),
                                                     uint64_t{0});
        if (totalVisits == 0)
            throw std::runtime_error(std::format("sample {} has no visits, so it has no policy to learn",
                                                 sampleIndex));

        const std::span<const uint16_t, moveIdCount> moveIdsByPolicyIndex =
            policyIndicesToMoveIds(boards[sampleIndex]->whiteToMove);
        const size_t policyOffset = sampleIndex * moveIdCount;

        for (int policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
        {
            const uint16_t moveId = moveIdsByPolicyIndex[policyIndex];
            legal[policyOffset + policyIndex] = boards[sampleIndex]->isLegal(moveId) ? 1.0f : 0.0f;
            policy[policyOffset + policyIndex] =
                static_cast<float>(visits[sampleIndex][moveId]) / static_cast<float>(totalVisits);
        }
    }

    return {mlx::core::array(input.data(), mlx::core::Shape{batch, encodedBoardSize}, mlx::core::float32),
            mlx::core::array(legal.data(), mlx::core::Shape{batch, moveIdCount}, mlx::core::float32),
            mlx::core::array(policy.data(), mlx::core::Shape{batch, moveIdCount}, mlx::core::float32),
            mlx::core::array(outcomes.data(), mlx::core::Shape{batch}, mlx::core::float32)};
}

std::vector<mlx::core::array> computeLoss(const AmoebaNetwork& network,
                                          const std::vector<mlx::core::array>& parameters,
                                          const TrainingBatch& batch, float weightDecay)
{
    const std::vector<mlx::core::array> networkOutput = network(batch.input, parameters);
    assert(networkOutput.size() == 2);
    const Prediction prediction{networkOutput[0], networkOutput[1]};

    // Cross-entropy asks the policy logits to reproduce the normalized MCTS visit
    // distribution. Masking here also prevents illegal moves entering the loss.
    const mlx::core::array masked = mlx::core::where(mlx::core::greater(batch.legal, mlx::core::array(0.0f)),
                                                      prediction.policy, mlx::core::array(-1e9f));
    const mlx::core::array logProbability = masked - mlx::core::logsumexp(masked, -1, true);
    const mlx::core::array policyLoss = mlx::core::mean(-mlx::core::sum(batch.policyTarget * logProbability, -1));
    // Mean squared error trains the scalar value against the eventual game result.
    const mlx::core::array valueLoss =
        mlx::core::mean(mlx::core::square(prediction.value - batch.valueTarget));

    mlx::core::array total = policyLoss + valueLoss;
    if (weightDecay > 0.0f)
    {
        // L2 weight decay discourages parameters from becoming unnecessarily
        // large, which reduces memorization of the replay buffer.
        mlx::core::array penalty = mlx::core::array(0.0f);
        for (const mlx::core::array& tensor : parameters)
            penalty = penalty + mlx::core::sum(mlx::core::square(tensor));
        total = total + penalty * weightDecay;
    }

    return {total, policyLoss, valueLoss};
}

} // namespace amoeba_bot
