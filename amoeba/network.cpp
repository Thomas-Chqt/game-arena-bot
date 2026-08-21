#include "network.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace bot
{

namespace
{

// x * 0.5 * (1 + erf(x / sqrt(2))). The only nonlinearity in the trunk besides
// the value head's tanh; without one, six stacked blocks would collapse
// algebraically into a single matrix.
mlx::core::array gelu(const mlx::core::array& input)
{
    return input * 0.5f * (mlx::core::erf(input * 0.70710678118654752f) + 1.0f);
}

// [batch, hex, width] -> [batch, head, hex, width/head], so each head attends
// over the 37 hexes independently.
mlx::core::array splitHeads(const mlx::core::array& input, int attentionHeadCount)
{
    return mlx::core::transpose(
        mlx::core::reshape(
            input,
            {input.shape(0), amoeba::hexCount, attentionHeadCount, input.shape(2) / attentionHeadCount}
        ),
        {0, 2, 1, 3}
    );
}

mlx::core::array mergeHeads(const mlx::core::array& input)
{
    return mlx::core::reshape(mlx::core::transpose(input, {0, 2, 1, 3}),
                              {input.shape(0), amoeba::hexCount, input.shape(1) * input.shape(3)});
}

// One learned scalar per (head, bucket), gathered into the [head, hex, hex] shape
// the attention scores are in. This is where the board's geometry enters: the
// network is told which hexes lie on a line and how far apart, instead of having
// to recover it from game outcomes.
mlx::core::array bucketIndices()
{
    std::array<int32_t, amoeba::hexCount * amoeba::hexCount> flattenedIndices{};
    for (int sourceHex = 0; sourceHex < amoeba::hexCount; ++sourceHex)
    {
        for (int destinationHex = 0; destinationHex < amoeba::hexCount; ++destinationHex)
        {
            flattenedIndices[sourceHex * amoeba::hexCount + destinationHex] =
                relativePositionBuckets[sourceHex][destinationHex];
        }
    }

    return mlx::core::array(flattenedIndices.data(), mlx::core::Shape{amoeba::hexCount * amoeba::hexCount},
                            mlx::core::int32);
}

void validateNetworkShape(NetworkShape shape)
{
    if (shape.blockCount <= 0)
        throw std::runtime_error("network must have at least one transformer block");
    if (shape.embeddingWidth <= 0)
        throw std::runtime_error("network embedding width must be positive");
    if (shape.attentionHeadCount <= 0)
        throw std::runtime_error("network must have at least one attention head");
    if (shape.embeddingWidth % shape.attentionHeadCount != 0)
        throw std::runtime_error("network embedding width must be divisible by its attention-head count");
}

mlx::core::array initializeParameter(const std::string& name, const mlx::core::Shape& shape)
{
    if (name.ends_with(".scale"))
        return mlx::core::ones(shape, mlx::core::float32);

    // Both the layer-norm shifts and every Linear bias start here, as does the
    // relative-position bias: no positional preference until training finds one.
    if (name.ends_with(".shift") || name.ends_with(".bias"))
        return mlx::core::zeros(shape, mlx::core::float32);

    // The position embedding is added to a token, not multiplied through it, so
    // fan-in scaling does not apply. Keep it small enough to nudge rather than
    // drown the features it is added to.
    if (name == "position")
        return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, 0.02f);

    // Every remaining tensor is a matrix whose output sums shape[0] inputs.
    // 1/sqrt(fan-in) keeps that sum at roughly unit scale, so activations
    // neither vanish nor explode on the way through the blocks.
    const float fanIn = static_cast<float>(shape.front());
    return mlx::core::random::normal(shape, mlx::core::float32, 0.0f, 1.0f / std::sqrt(fanIn));
}

} // namespace

std::vector<std::pair<std::string, mlx::core::Shape>> Network::parameterLayout(NetworkShape shape)
{
    const int feedForwardWidth = shape.embeddingWidth * 4;

    // Linear weights are stored [in, out] so the forward pass is a plain matmul
    // with no transpose.
    std::vector<std::pair<std::string, mlx::core::Shape>> parameterDefinitions;

    parameterDefinitions.emplace_back("embed.weight", mlx::core::Shape{amoeba::featuresPerHex + amoeba::globalFeatureCount, shape.embeddingWidth});
    parameterDefinitions.emplace_back("embed.bias",   mlx::core::Shape{shape.embeddingWidth});

    parameterDefinitions.emplace_back("position",     mlx::core::Shape{amoeba::hexCount, shape.embeddingWidth});

    for (int block = 0; block < shape.blockCount; ++block)
    {
        const std::string prefix = std::format("block{}.", block);
        parameterDefinitions.emplace_back(prefix + "norm1.scale",  mlx::core::Shape{shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "norm1.shift",  mlx::core::Shape{shape.embeddingWidth});

        parameterDefinitions.emplace_back(prefix + "attn.query",   mlx::core::Shape{shape.embeddingWidth,     shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "attn.key",     mlx::core::Shape{shape.embeddingWidth,     shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "attn.value",   mlx::core::Shape{shape.embeddingWidth,     shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "attn.out",     mlx::core::Shape{shape.embeddingWidth,     shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "attn.bias",    mlx::core::Shape{shape.attentionHeadCount, relativePositionBucketCount});

        parameterDefinitions.emplace_back(prefix + "norm2.scale",  mlx::core::Shape{shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "norm2.shift",  mlx::core::Shape{shape.embeddingWidth});

        parameterDefinitions.emplace_back(prefix + "mlp.in",       mlx::core::Shape{shape.embeddingWidth, feedForwardWidth});
        parameterDefinitions.emplace_back(prefix + "mlp.in.bias",  mlx::core::Shape{feedForwardWidth});
        parameterDefinitions.emplace_back(prefix + "mlp.out",      mlx::core::Shape{feedForwardWidth, shape.embeddingWidth});
        parameterDefinitions.emplace_back(prefix + "mlp.out.bias", mlx::core::Shape{shape.embeddingWidth});
    }

    parameterDefinitions.emplace_back("policy.weight",     mlx::core::Shape{shape.embeddingWidth, policyOutputsPerHex});
    parameterDefinitions.emplace_back("policy.bias",       mlx::core::Shape{policyOutputsPerHex});

    parameterDefinitions.emplace_back("value.hidden",      mlx::core::Shape{shape.embeddingWidth, shape.embeddingWidth});
    parameterDefinitions.emplace_back("value.hidden.bias", mlx::core::Shape{shape.embeddingWidth});

    parameterDefinitions.emplace_back("value.out",         mlx::core::Shape{shape.embeddingWidth, 1});
    parameterDefinitions.emplace_back("value.out.bias",    mlx::core::Shape{1});

    return parameterDefinitions;
}

Network::Network(NetworkShape shape, uint64_t seed)
    : m_shape(shape)
{
    validateNetworkShape(shape);

    // Seeding once and building in parameterLayout() order is what makes a generation
    // reproducible from its seed alone.
    mlx::core::random::seed(seed);

    for (const auto& [name, tensorShape] : parameterLayout(shape))
    {
        m_names.push_back(name);
        m_parameters.push_back(initializeParameter(name, tensorShape));
    }
    mlx::core::eval(m_parameters);
}

Network::Network(const std::filesystem::path& checkpoint)
{
    // MLX spells paths as std::string, so the conversion lives here rather than
    // spreading through every caller.
    const auto [tensors, metadata] = mlx::core::load_safetensors(checkpoint.string());

    m_shape.blockCount = std::stoi(metadata.at("blocks"));
    m_shape.embeddingWidth = std::stoi(metadata.at("width"));
    m_shape.attentionHeadCount = std::stoi(metadata.at("heads"));
    validateNetworkShape(m_shape);

    // Reading through parameterLayout() rather than over the file means a checkpoint from
    // a different architecture is rejected here instead of at the first matmul.
    for (const auto& [name, tensorShape] : parameterLayout(m_shape))
    {
        const auto found = tensors.find(name);
        if (found == tensors.end())
            throw std::runtime_error(std::format("{}: no tensor named {}", checkpoint.string(), name));
        if (found->second.shape() != tensorShape)
            throw std::runtime_error(std::format("{}: {} has the wrong shape", checkpoint.string(), name));

        m_names.push_back(name);
        m_parameters.push_back(found->second);
    }
    mlx::core::eval(m_parameters);
}

void Network::save(const std::filesystem::path& checkpoint) const
{
    std::unordered_map<std::string, mlx::core::array> tensors;
    for (size_t i = 0; i < m_names.size(); ++i)
        tensors.emplace(m_names[i], m_parameters[i]);

    mlx::core::save_safetensors(checkpoint.string(), tensors,
                                {{"blocks", std::format("{}", m_shape.blockCount)},
                                 {"width", std::format("{}", m_shape.embeddingWidth)},
                                 {"heads", std::format("{}", m_shape.attentionHeadCount)}});
}

size_t Network::parameterCount() const
{
    return std::accumulate(m_parameters.begin(), m_parameters.end(), size_t{0}, [](size_t total, const mlx::core::array& tensor) {
        return total + tensor.size();
    });
}

void Network::replaceParameters(std::vector<mlx::core::array> parameters)
{
    if (parameters.size() != m_parameters.size())
        throw std::runtime_error(
            std::format("expected {} parameter tensors, got {}", m_parameters.size(), parameters.size()));
    m_parameters = std::move(parameters);
}

Prediction runInference(const std::vector<mlx::core::array>& parameters, NetworkShape shape, const mlx::core::array& inputBatch)
{
    validateNetworkShape(shape);
    assert(inputBatch.ndim() == 2);
    assert(inputBatch.shape(1) == amoeba::encodedBoardSize);

    // Read strictly in parameterLayout() order. Each tensor gets its own local because the
    // order in which C++ evaluates two reads inside one expression is unspecified.
    size_t cursor = 0;
    const auto next = [&parameters, &cursor]() -> const mlx::core::array& { return parameters[cursor++]; };

    const int batchSize = inputBatch.shape(0);
    constexpr int hexFeatureCount = amoeba::hexCount * amoeba::featuresPerHex;

    const mlx::core::array perHexFeatures = mlx::core::reshape(
        mlx::core::slice(inputBatch, {0, 0}, {batchSize, hexFeatureCount}),
        {batchSize, amoeba::hexCount, amoeba::featuresPerHex});

    // The globals go to every token: staleness or being in check changes what
    // every hex on the board means.
    const mlx::core::array globalFeatures = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::slice(inputBatch, {0, hexFeatureCount}, {batchSize, amoeba::encodedBoardSize}),
            {batchSize, 1, amoeba::globalFeatureCount}
        ),
        {batchSize, amoeba::hexCount, amoeba::globalFeatureCount});

    const mlx::core::array& embedWeight = next();
    const mlx::core::array& embedBias = next();
    const mlx::core::array& position = next();

    mlx::core::array tokenEmbeddings =
        mlx::core::matmul(mlx::core::concatenate({perHexFeatures, globalFeatures}, 2), embedWeight);
    tokenEmbeddings = tokenEmbeddings + embedBias;
    tokenEmbeddings = tokenEmbeddings + position;

    const mlx::core::array positionBucketIndices = bucketIndices();
    const int featuresPerHead = shape.embeddingWidth / shape.attentionHeadCount;
    const float scoreScale = 1.0f / std::sqrt(static_cast<float>(featuresPerHead));

    for (int blockIndex = 0; blockIndex < shape.blockCount; ++blockIndex)
    {
        const mlx::core::array& norm1Scale = next();
        const mlx::core::array& norm1Shift = next();
        const mlx::core::array& queryWeight = next();
        const mlx::core::array& keyWeight = next();
        const mlx::core::array& valueWeight = next();
        const mlx::core::array& outWeight = next();
        const mlx::core::array& positionBias = next();

        const mlx::core::array normalized = mlx::core::fast::layer_norm(tokenEmbeddings, norm1Scale, norm1Shift, 1e-5f);

        const mlx::core::array query = splitHeads(mlx::core::matmul(normalized, queryWeight), shape.attentionHeadCount);
        const mlx::core::array key = splitHeads(mlx::core::matmul(normalized, keyWeight), shape.attentionHeadCount);
        const mlx::core::array value = splitHeads(mlx::core::matmul(normalized, valueWeight), shape.attentionHeadCount);

        const mlx::core::array bias =
            mlx::core::reshape(mlx::core::take(positionBias, positionBucketIndices, 1),
                               {1, shape.attentionHeadCount, amoeba::hexCount, amoeba::hexCount});

        mlx::core::array scores = mlx::core::matmul(query, mlx::core::transpose(key, {0, 1, 3, 2}));
        scores = scores * scoreScale;
        scores = mlx::core::softmax(scores + bias, -1);

        const mlx::core::array attended = mergeHeads(mlx::core::matmul(scores, value));
        tokenEmbeddings = tokenEmbeddings + mlx::core::matmul(attended, outWeight);

        const mlx::core::array& norm2Scale = next();
        const mlx::core::array& norm2Shift = next();
        const mlx::core::array& mlpIn = next();
        const mlx::core::array& mlpInBias = next();
        const mlx::core::array& mlpOut = next();
        const mlx::core::array& mlpOutBias = next();

        const mlx::core::array hidden = mlx::core::matmul(mlx::core::fast::layer_norm(tokenEmbeddings, norm2Scale, norm2Shift, 1e-5f), mlpIn);
        const mlx::core::array activated = gelu(hidden + mlpInBias);
        tokenEmbeddings = tokenEmbeddings + (mlx::core::matmul(activated, mlpOut) + mlpOutBias);
    }

    const mlx::core::array& policyWeight = next();
    const mlx::core::array& policyBias = next();

    // [batch, hex, 12] flattens straight onto Move::id, because the 12 run
    // direction-major and splitting-minor and id is (hex * 6 + dir) * 2 + split.
    const mlx::core::array logits = mlx::core::matmul(tokenEmbeddings, policyWeight) + policyBias;
    const mlx::core::array policy = mlx::core::reshape(logits, {batchSize, amoeba::moveIdCount});

    const mlx::core::array& valueWeight = next();
    const mlx::core::array& valueBias = next();
    const mlx::core::array& valueOutWeight = next();
    const mlx::core::array& valueOutBias = next();

    // Value is a property of the whole position, so the 37 tokens collapse to one.
    const mlx::core::array pooled = mlx::core::mean(tokenEmbeddings, std::vector<int>{1}, false);
    const mlx::core::array hidden = gelu(mlx::core::matmul(pooled, valueWeight) + valueBias);
    const mlx::core::array scalar = mlx::core::matmul(hidden, valueOutWeight) + valueOutBias;

    assert(cursor == parameters.size());
    return {policy, mlx::core::tanh(mlx::core::reshape(scalar, {batchSize}))};
}

void NetworkEvaluator::evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> outputs)
{
    assert(boards.size() == outputs.size());
    const int batchSize = static_cast<int>(boards.size());

    std::vector<float> encodedBoards(static_cast<size_t>(batchSize) * amoeba::encodedBoardSize);
    std::vector<float> legalMoveMask(static_cast<size_t>(batchSize) * amoeba::moveIdCount);

    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        const size_t encodedBoardOffset = boardIndex * amoeba::encodedBoardSize;
        const size_t maskOffset = boardIndex * amoeba::moveIdCount;

        amoeba::encodeBoard(*boards[boardIndex], std::span<float, amoeba::encodedBoardSize>(
                                                      encodedBoards.data() + encodedBoardOffset,
                                                      amoeba::encodedBoardSize));

        const std::span<const uint16_t, amoeba::moveIdCount> moveIdsByPolicyIndex =
            amoeba::policyIndicesToMoveIds(boards[boardIndex]->whiteToMove);
        for (int policyIndex = 0; policyIndex < amoeba::moveIdCount; ++policyIndex)
            legalMoveMask[maskOffset + policyIndex] =
                boards[boardIndex]->isLegal(moveIdsByPolicyIndex[policyIndex]) ? 1.0f : 0.0f;
    }

    const mlx::core::array input(encodedBoards.data(), mlx::core::Shape{batchSize, amoeba::encodedBoardSize}, mlx::core::float32);
    const Prediction prediction = runInference(m_network.parameters(), m_network.shape(), input);

    // A large finite penalty rather than -inf: a terminal position has no legal
    // moves at all, and softmax over an all -inf row is NaN, which would spread
    // into every parameter that touched it. The search creates no edges there, so
    // the resulting uniform row is never read.
    const mlx::core::array legal(legalMoveMask.data(), mlx::core::Shape{batchSize, amoeba::moveIdCount}, mlx::core::float32);
    const mlx::core::array masked = mlx::core::where(mlx::core::greater(legal, mlx::core::array(0.0f)), prediction.policy, mlx::core::array(-1e9f));

    // The search sums the priors of the legal moves and divides, so it needs
    // probabilities; raw logits would give it negative priors.
    const mlx::core::array probabilities = mlx::core::softmax(masked, -1);
    mlx::core::eval({probabilities, prediction.value});

    const float* policyData = probabilities.data<float>();
    const float* valueData = prediction.value.data<float>();

    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        const std::span<const uint16_t, amoeba::moveIdCount> moveIdsByPolicyIndex = amoeba::policyIndicesToMoveIds(boards[boardIndex]->whiteToMove);
        const size_t policyOffset = boardIndex * amoeba::moveIdCount;
        for (int policyIndex = 0; policyIndex < amoeba::moveIdCount; ++policyIndex)
            outputs[boardIndex].policy[moveIdsByPolicyIndex[policyIndex]] = policyData[policyOffset + policyIndex];
        outputs[boardIndex].value = valueData[boardIndex];
    }
}

// ===========================================================================
// Training
// ===========================================================================

TrainingBatch makeTrainingBatch(std::span<const amoeba::Board* const> boards, std::span<const VisitCounts> visits,
                                std::span<const float> outcomes)
{
    if (boards.size() != visits.size() || boards.size() != outcomes.size())
        throw std::runtime_error(std::format("makeTrainingBatch got {} boards, {} visit sets and {} outcomes",
                                             boards.size(), visits.size(), outcomes.size()));

    const int batch = static_cast<int>(boards.size());

    std::vector<float> input(static_cast<size_t>(batch) * amoeba::encodedBoardSize);
    std::vector<float> legal(static_cast<size_t>(batch) * amoeba::moveIdCount);
    std::vector<float> policy(static_cast<size_t>(batch) * amoeba::moveIdCount);

    for (size_t sampleIndex = 0; sampleIndex < boards.size(); ++sampleIndex)
    {
        const size_t inputOffset = sampleIndex * amoeba::encodedBoardSize;
        amoeba::encodeBoard(*boards[sampleIndex], std::span<float, amoeba::encodedBoardSize>(
                                                      input.data() + inputOffset, amoeba::encodedBoardSize));

        uint64_t totalVisits = 0;
        for (const uint32_t count : visits[sampleIndex])
            totalVisits += count;
        if (totalVisits == 0)
            throw std::runtime_error(std::format("sample {} has no visits, so it has no policy to learn", sampleIndex));

        // The search counts moves in absolute ids and runInference() answers in the
        // flipped space encodeBoard() used, so the target has to cross over. Same table
        // in both directions - it is its own inverse.
        const std::span<const uint16_t, amoeba::moveIdCount> moveIdsByPolicyIndex =
            amoeba::policyIndicesToMoveIds(boards[sampleIndex]->whiteToMove);
        const size_t policyOffset = sampleIndex * amoeba::moveIdCount;

        for (int policyIndex = 0; policyIndex < amoeba::moveIdCount; ++policyIndex)
        {
            const uint16_t moveId = moveIdsByPolicyIndex[policyIndex];
            legal[policyOffset + policyIndex] = boards[sampleIndex]->isLegal(moveId) ? 1.0f : 0.0f;
            policy[policyOffset + policyIndex] =
                static_cast<float>(visits[sampleIndex][moveId]) / static_cast<float>(totalVisits);
        }
    }

    return {mlx::core::array(input.data(), mlx::core::Shape{batch, amoeba::encodedBoardSize}, mlx::core::float32),
            mlx::core::array(legal.data(), mlx::core::Shape{batch, amoeba::moveIdCount}, mlx::core::float32),
            mlx::core::array(policy.data(), mlx::core::Shape{batch, amoeba::moveIdCount}, mlx::core::float32),
            mlx::core::array(outcomes.data(), mlx::core::Shape{batch}, mlx::core::float32)};
}

std::vector<mlx::core::array> computeLoss(const std::vector<mlx::core::array>& parameters, NetworkShape shape,
                                          const TrainingBatch& batch, float weightDecay)
{
    const Prediction prediction = runInference(parameters, shape, batch.input);

    // Illegal moves go to a large finite penalty, so they end up with no
    // probability and no gradient is spent teaching the network to avoid them.
    // Finite rather than -inf because a position with no legal moves at all would
    // otherwise softmax to NaN.
    const mlx::core::array masked = mlx::core::where(mlx::core::greater(batch.legal, mlx::core::array(0.0f)),
                                                     prediction.policy, mlx::core::array(-1e9f));

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
        for (const mlx::core::array& tensor : parameters)
        {
            penalty = penalty + mlx::core::sum(mlx::core::square(tensor));
        }
        total = total + penalty * weightDecay;
    }

    return {total, policyLoss, valueLoss};
}

Adam::Adam(const std::vector<mlx::core::array>& parameters, AdamConfig config)
    : m_config(config)
{
    for (const mlx::core::array& tensor : parameters)
    {
        m_mean.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
        m_variance.push_back(mlx::core::zeros(tensor.shape(), tensor.dtype()));
    }
}

std::vector<mlx::core::array> Adam::updateParameters(const std::vector<mlx::core::array>& parameters, const std::vector<mlx::core::array>& gradients, float rate)
{
    if (parameters.size() != m_mean.size() || parameters.size() != gradients.size())
        throw std::runtime_error(std::format("Adam was built for {} tensors but got {} parameters and {} gradients",
                                             m_mean.size(), parameters.size(), gradients.size()));
    ++m_steps;

    // Both averages start at zero, so for the first few hundred steps they read
    // far too low - and the variance more severely than the mean, which would make
    // the steps enormous rather than merely small. These undo that exactly, and
    // fade to 1 as the averages fill up.
    const float meanScale = 1.0f - std::pow(m_config.meanDecay, m_steps);
    const float varianceScale = 1.0f - std::pow(m_config.varianceDecay, m_steps);

    std::vector<mlx::core::array> updated;
    updated.reserve(parameters.size());

    for (size_t parameterIndex = 0; parameterIndex < parameters.size(); ++parameterIndex)
    {
        m_mean[parameterIndex] = m_mean[parameterIndex]
            * m_config.meanDecay
            + gradients[parameterIndex]
            * (1.0f - m_config.meanDecay);

        m_variance[parameterIndex] = m_variance[parameterIndex]
            * m_config.varianceDecay
            + mlx::core::square(gradients[parameterIndex])
            * (1.0f - m_config.varianceDecay);

        const mlx::core::array typicalGradient = mlx::core::sqrt(m_variance[parameterIndex] / varianceScale) + m_config.epsilon;
        updated.push_back(parameters[parameterIndex] - (m_mean[parameterIndex] / meanScale) / typicalGradient * rate);
    }
    return updated;
}

} // namespace bot
