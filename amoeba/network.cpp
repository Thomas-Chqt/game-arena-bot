#include "network.hpp"
#include "hex_ray_lite_network.hpp"
#include "transformer_network.hpp"

#include <cassert>
#include <format>
#include <numeric>
#include <stdexcept>

namespace amoeba_bot
{

Prediction Network::evaluate(mlx::core::array input,
                             std::span<const mlx::core::array> parameters) const
{
    assert(parameters.size() == m_parameters.size());
    for (size_t index = 0; index < parameters.size(); ++index)
    {
        assert(parameters[index].shape() == m_layout[index].shape);
        assert(parameters[index].dtype() == mlx::core::float32);
    }

    // MLX compile accepts one flat vector. The input is entry 0 and every
    // parameter follows in layout order. Parameters must be explicit inputs:
    // arrays captured by compile() would become constants and could not train.
    std::vector<mlx::core::array> inputs;
    inputs.reserve(parameters.size() + 1);
    inputs.push_back(std::move(input));
    inputs.insert(inputs.end(), parameters.begin(), parameters.end());

    // MLX traces this callable on its first use for each input signature and
    // caches the compiled graph. The public API does not expose its vector form.
    if (!m_compiledForward)
        createCompiledForward();
    std::vector<mlx::core::array> output = m_compiledForward(inputs);
    assert(output.size() == 2);
    return Prediction{std::move(output[0]), std::move(output[1])};
}

void Network::createCompiledForward() const
{
    std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> function =
        [this](const std::vector<mlx::core::array>& inputs)
    {
        assert(!inputs.empty());
        const Prediction prediction = forward(inputs.front(), std::span{inputs}.subspan(1));
        return std::vector<mlx::core::array>{prediction.policy, prediction.value};
    };
    m_compiledForward = mlx::core::compile(std::move(function));
}

PendingEvaluations Network::submit(std::span<const Board* const> boards) const
{
    const auto policyIndexToMoveId = [](uint16_t policyIndex, bool whiteToMove) {
        Move move = Move::fromId(policyIndex);
        if (!whiteToMove)
        {
            move.sourceCoord = rotatedHex(move.sourceCoord);
            move.direction = oppositeDirection(move.direction);
        }
        return move.id();
    };

    const int batchSize = static_cast<int>(boards.size());
    assert(batchSize > 0);

    // Collect several MCTS leaves into one tensor so MLX evaluates them in one
    // GPU batch instead of launching one network call per position.
    std::vector<mlx::core::array> encodedBoards;
    encodedBoards.reserve(boards.size());
    std::vector<float> legalMoveMask(static_cast<size_t>(batchSize) * moveIdCount);
    std::vector<uint8_t> whiteToMove;
    whiteToMove.reserve(boards.size());

    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        assert(boards[boardIndex] != nullptr);
        const size_t maskOffset = boardIndex * moveIdCount;

        encodedBoards.push_back(boards[boardIndex]->tensorEncoding());
        whiteToMove.push_back(boards[boardIndex]->whiteToMove);

        for (int policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
        {
            const uint16_t moveId = policyIndexToMoveId(
                static_cast<uint16_t>(policyIndex), boards[boardIndex]->whiteToMove);
            legalMoveMask[maskOffset + policyIndex] =
                boards[boardIndex]->isLegal(moveId) ? 1.0f : 0.0f;
        }
    }

    const mlx::core::array input = mlx::core::stack(encodedBoards);
    const Prediction prediction = evaluate(input, m_parameters);

    const mlx::core::array legal(legalMoveMask.data(),
                                 mlx::core::Shape{batchSize, moveIdCount},
                                 mlx::core::float32);
    // Illegal logits become effectively zero after softmax. The game guarantees
    // that every board evaluated by MCTS has at least one legal move.
    const mlx::core::array masked = mlx::core::where(
        mlx::core::greater(legal, mlx::core::array(0.0f)), prediction.policy,
        mlx::core::array(-1e9f));
    const mlx::core::array probabilities = mlx::core::softmax(masked, -1);
    std::vector<mlx::core::array> outputs{probabilities, prediction.value};
    mlx::core::async_eval(outputs);
    return {std::move(outputs), std::move(whiteToMove)};
}

void Network::finish(PendingEvaluations pending,
                     std::span<Evaluation> outputs) const
{
    const auto policyIndexToMoveId = [](uint16_t policyIndex, bool whiteToMove) {
        Move move = Move::fromId(policyIndex);
        if (!whiteToMove)
        {
            move.sourceCoord = rotatedHex(move.sourceCoord);
            move.direction = oppositeDirection(move.direction);
        }
        return move.id();
    };

    assert(pending.outputs.size() == 2);
    assert(pending.whiteToMove.size() == outputs.size());
    mlx::core::eval(pending.outputs);

    const float* policyData = pending.outputs[0].data<float>();
    const float* valueData = pending.outputs[1].data<float>();
    for (size_t boardIndex = 0; boardIndex < outputs.size(); ++boardIndex)
    {
        const size_t policyOffset = boardIndex * moveIdCount;

        // tensorEncoding rotates into the side-to-move perspective. Convert the
        // policy indices back to absolute Move::id values before MCTS sees them.
        for (int policyIndex = 0; policyIndex < moveIdCount; ++policyIndex)
            outputs[boardIndex].policy[policyIndexToMoveId(
                static_cast<uint16_t>(policyIndex), pending.whiteToMove[boardIndex])] =
                policyData[policyOffset + policyIndex];
        outputs[boardIndex].value = valueData[boardIndex];
    }
}

void Network::operator()(std::span<const Board* const> boards,
                         std::span<Evaluation> outputs) const
{
    assert(boards.size() == outputs.size());
    finish(submit(boards), outputs);
}

std::vector<mlx::core::array> Network::computeLoss(
    const std::vector<mlx::core::array>& parameters,
    const TrainingBatch& batch) const
{
    const Prediction prediction = evaluate(batch.input, parameters);

    // Cross-entropy asks the policy logits to reproduce the normalized MCTS
    // visit distribution. Masking prevents illegal moves entering that loss.
    const mlx::core::array masked = mlx::core::where(
        mlx::core::greater(batch.legal, mlx::core::array(0.0f)),
        prediction.policy, mlx::core::array(-1e9f));
    const mlx::core::array logProbability =
        masked - mlx::core::logsumexp(masked, -1, true);
    const mlx::core::array policyLoss = mlx::core::mean(
        -mlx::core::sum(batch.policyTarget * logProbability, -1));

    // Mean squared error trains value against the eventual game result.
    const mlx::core::array valueLoss = mlx::core::mean(
        mlx::core::square(prediction.value - batch.valueTarget));

    // MLX differentiates the first returned value and carries the remaining
    // values through for logging.
    return {policyLoss + valueLoss, policyLoss, valueLoss};
}

Loss Network::loss(const std::vector<mlx::core::array>& parameters,
                   const TrainingBatch& batch) const
{
    std::vector<mlx::core::array> values = computeLoss(parameters, batch);
    assert(values.size() == 3);
    return Loss{std::move(values[0]), std::move(values[1]), std::move(values[2])};
}

LossAndGrad Network::valueAndGrad(
    const std::vector<mlx::core::array>& parameters,
    const TrainingBatch& batch) const
{
    std::vector<int> parameterIndices(parameters.size());
    std::iota(parameterIndices.begin(), parameterIndices.end(), 0);

    const auto lossFunction = [&](const std::vector<mlx::core::array>& currentParameters)
    {
        return computeLoss(currentParameters, batch);
    };
    auto [values, gradients] =
        mlx::core::value_and_grad(lossFunction, parameterIndices)(parameters);
    assert(values.size() == 3);
    return LossAndGrad{
        Loss{std::move(values[0]), std::move(values[1]), std::move(values[2])},
        std::move(gradients)};
}

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

void Network::save(const std::filesystem::path& checkpoint,
                   const CheckpointMetadata& additionalMetadata) const
{
    std::unordered_map<std::string, mlx::core::array> tensors;
    for (size_t index = 0; index < m_layout.size(); ++index)
        tensors.emplace(m_layout[index].name, m_parameters[index]);

    CheckpointMetadata metadata = m_checkpointMetadata;
    for (const auto& [name, value] : additionalMetadata)
        metadata.insert_or_assign(name, value);
    metadata.insert_or_assign("network", m_name);

    if (checkpoint.extension() == ".safetensors")
    {
        mlx::core::save_safetensors(checkpoint.string(), tensors, metadata);
        return;
    }

    std::filesystem::path temporary = checkpoint;
    temporary += ".tmp.safetensors";
    std::filesystem::remove(temporary);
    mlx::core::save_safetensors(temporary.string(), tensors, metadata);
    std::filesystem::rename(temporary, checkpoint);
}

std::unique_ptr<Network> createNetwork(std::string_view identifier, uint64_t seed)
{
    if (identifier == TransformerNetwork::identifier)
        return std::make_unique<TransformerNetwork>(seed);
    if (identifier == HexRayLiteNetwork::identifier)
        return std::make_unique<HexRayLiteNetwork>(seed);
    throw std::runtime_error(std::format("unknown network identifier {}", identifier));
}

std::unique_ptr<Network> loadNetwork(
    const std::filesystem::path& checkpoint,
    std::string_view expectedIdentifier)
{
    auto [tensors, metadata] = mlx::core::load_safetensors(checkpoint.string());
    const auto storedName = metadata.find("network");
    if (storedName == metadata.end())
        throw std::runtime_error(std::format(
            "{}: checkpoint has no network identifier", checkpoint.string()));
    const std::string_view identifier = storedName->second;

    if (!expectedIdentifier.empty() && identifier != expectedIdentifier)
        throw std::runtime_error(std::format(
            "{}: checkpoint network {} does not match requested network {}",
            checkpoint.string(), identifier, expectedIdentifier));

    if (identifier == TransformerNetwork::identifier)
        return std::make_unique<TransformerNetwork>(checkpoint);
    if (identifier == HexRayLiteNetwork::identifier)
        return std::make_unique<HexRayLiteNetwork>(checkpoint);
    throw std::runtime_error(std::format(
        "{}: unknown network identifier {}", checkpoint.string(), identifier));
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

void Adam::restore(std::vector<mlx::core::array> mean,
                   std::vector<mlx::core::array> variance, int steps)
{
    if (mean.size() != m_mean.size() || variance.size() != m_variance.size())
        throw std::runtime_error("Adam checkpoint has the wrong number of tensors");
    if (steps < 0)
        throw std::runtime_error("Adam checkpoint has a negative step count");

    for (size_t index = 0; index < m_mean.size(); ++index)
    {
        if (mean[index].shape() != m_mean[index].shape()
            || variance[index].shape() != m_variance[index].shape()
            || mean[index].dtype() != m_mean[index].dtype()
            || variance[index].dtype() != m_variance[index].dtype())
            throw std::runtime_error(std::format(
                "Adam checkpoint tensor {} has the wrong shape or type", index));
    }

    m_mean = std::move(mean);
    m_variance = std::move(variance);
    m_steps = steps;
    mlx::core::eval(m_mean);
    mlx::core::eval(m_variance);
}

std::vector<mlx::core::array> Adam::updateParameters(
    const std::vector<mlx::core::array>& parameters,
    const std::vector<mlx::core::array>& gradients, float rate, float weightDecay,
    std::span<const uint8_t> decayMask)
{
    if (parameters.size() != m_mean.size() || parameters.size() != gradients.size())
        throw std::runtime_error(std::format("Adam was built for {} tensors but got {} parameters and {} gradients",
                                             m_mean.size(), parameters.size(), gradients.size()));
    if (decayMask.size() != parameters.size())
        throw std::runtime_error(std::format("Adam got a decay mask of {} entries for {} parameters",
                                             decayMask.size(), parameters.size()));
    assert(rate > 0.0f);
    assert(weightDecay >= 0.0f);
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
        // AdamW applies decay directly to the parameter, outside Adam's
        // gradient normalization. This keeps a weak task gradient from turning
        // regularization into a full-sized update toward zero.
        const float decay = decayMask[index] != 0 ? rate * weightDecay : 0.0f;
        updated.push_back(parameters[index] * (1.0f - decay)
                          - (m_mean[index] / meanScale) / typicalGradient * rate);
    }
    return updated;
}

} // namespace amoeba_bot
