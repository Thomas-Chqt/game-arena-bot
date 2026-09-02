#include "transformer_network.hpp"

namespace amoeba_bot
{

template<size_t Width, size_t HeadCount>
mlx::core::array TransformerBlock<Width, HeadCount>::splitHeads(
    const mlx::core::array& input)
{
    return mlx::core::transpose(
        mlx::core::reshape(
            input, {input.shape(0), hexCount, static_cast<int>(HeadCount),
                    static_cast<int>(Width / HeadCount)}),
        {0, 2, 1, 3});
}

template<size_t Width, size_t HeadCount>
mlx::core::array TransformerBlock<Width, HeadCount>::mergeHeads(
    const mlx::core::array& input)
{
    return mlx::core::reshape(
        mlx::core::transpose(input, {0, 2, 1, 3}),
        {input.shape(0), hexCount, static_cast<int>(Width)});
}

template<size_t Width, size_t HeadCount>
mlx::core::array TransformerBlock<Width, HeadCount>::operator()(
    mlx::core::array input,
    std::span<const mlx::core::array> parameters) const
{
    static_assert(Width % HeadCount == 0);

    const mlx::core::array normalized = m_norm1(input, parameters);
    const mlx::core::array query = splitHeads(
        mlx::core::matmul(normalized, parameters[m_queryIndex]));
    const mlx::core::array key = splitHeads(
        mlx::core::matmul(normalized, parameters[m_keyIndex]));
    const mlx::core::array value = splitHeads(
        mlx::core::matmul(normalized, parameters[m_valueIndex]));

    static constexpr auto flattenedRelationIndices = [] {
        std::array<int32_t, hexCount * hexCount> result{};
        for (uint8_t source = 0; source < hexCount; ++source)
        {
            // A move never lands on its own source, so the diagonal is free
            // for the self bucket and cannot collide with a (direction,
            // distance) bucket.
            result[source * hexCount + source] = 1;
            for (uint8_t direction = 0; direction < directionCount; ++direction)
            {
                for (uint8_t distance = 1; distance <= maximumMovableStackHeight; ++distance)
                {
                    if (const std::optional<uint8_t> destination =
                            destinationHex(source, directions[direction], distance))
                    {
                        result[source * hexCount + *destination] =
                            2 + direction * maximumMovableStackHeight + distance - 1;
                    }
                }
            }
        }
        return result;
    }();
    static const mlx::core::array relationIndices(
        flattenedRelationIndices.data(),
        mlx::core::Shape{static_cast<int>(flattenedRelationIndices.size())},
        mlx::core::int32);
    const mlx::core::array relationBias = mlx::core::reshape(
        mlx::core::take(parameters[m_relationBiasIndex], relationIndices, 1),
        {1, static_cast<int>(HeadCount), hexCount, hexCount});

    const float scoreScale =
        1.0f / std::sqrt(static_cast<float>(static_cast<float>(Width) / static_cast<float>(HeadCount)));
    mlx::core::array scores = mlx::core::matmul(
        query, mlx::core::transpose(key, {0, 1, 3, 2}));
    scores = mlx::core::softmax(scores * scoreScale + relationBias, -1);
    input = input + mlx::core::matmul(
        mergeHeads(mlx::core::matmul(scores, value)), parameters[m_outputIndex]);

    mlx::core::array hidden = mlx::core::matmul(
        m_norm2(input, parameters), parameters[m_mlpInputIndex]);
    hidden = Gelu{}(hidden + parameters[m_mlpInputBiasIndex], parameters);
    return input + mlx::core::matmul(hidden, parameters[m_mlpOutputIndex])
        + parameters[m_mlpOutputBiasIndex];
}

TransformerNetwork::TransformerNetwork(uint64_t seed)
    : Network(identifier, seed)
    , m_embed(*this, "embed")
    , m_positionIndex(addParameter(
          "position", mlx::core::random::normal(
                          {hexCount, embeddingWidth}, mlx::core::float32, 0.0f, 0.02f)))
    , m_block0(*this, "block0", residualInitScale())
    , m_block1(*this, "block1", residualInitScale())
    , m_block2(*this, "block2", residualInitScale())
    , m_block3(*this, "block3", residualInitScale())
    , m_block4(*this, "block4", residualInitScale())
    , m_block5(*this, "block5", residualInitScale())
    , m_finalNorm(*this, "final_norm")
    , m_policy(*this, "policy")
    , m_valueHiddenIndex(addParameter(
          "value.hidden", mlx::core::random::normal(
                              {embeddingWidth, embeddingWidth}, mlx::core::float32,
                              0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueHiddenBiasIndex(addParameter(
          "value.hidden.bias", mlx::core::zeros({embeddingWidth}, mlx::core::float32)))
    , m_valueOutputIndex(addParameter(
          "value.out", mlx::core::random::normal(
                           {embeddingWidth, 1}, mlx::core::float32,
                           0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueOutputBiasIndex(addParameter(
          "value.out.bias", mlx::core::zeros({1}, mlx::core::float32)))
{
    materializeParameters();
}

TransformerNetwork::TransformerNetwork(const std::filesystem::path& checkpoint)
    : Network(identifier, 0)
    , m_embed(*this, "embed")
    , m_positionIndex(addParameter(
          "position", mlx::core::random::normal(
                          {hexCount, embeddingWidth}, mlx::core::float32, 0.0f, 0.02f)))
    , m_block0(*this, "block0", residualInitScale())
    , m_block1(*this, "block1", residualInitScale())
    , m_block2(*this, "block2", residualInitScale())
    , m_block3(*this, "block3", residualInitScale())
    , m_block4(*this, "block4", residualInitScale())
    , m_block5(*this, "block5", residualInitScale())
    , m_finalNorm(*this, "final_norm")
    , m_policy(*this, "policy")
    , m_valueHiddenIndex(addParameter(
          "value.hidden", mlx::core::random::normal(
                              {embeddingWidth, embeddingWidth}, mlx::core::float32,
                              0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueHiddenBiasIndex(addParameter(
          "value.hidden.bias", mlx::core::zeros({embeddingWidth}, mlx::core::float32)))
    , m_valueOutputIndex(addParameter(
          "value.out", mlx::core::random::normal(
                           {embeddingWidth, 1}, mlx::core::float32,
                           0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueOutputBiasIndex(addParameter(
          "value.out.bias", mlx::core::zeros({1}, mlx::core::float32)))
{
    load(checkpoint);
}

Prediction TransformerNetwork::forward(
    mlx::core::array inputBatch,
    std::span<const mlx::core::array> parameters) const
{
    assert(inputBatch.ndim() == 2);
    assert(inputBatch.shape(1) == encodedBoardSize);
    assert(inputBatch.dtype() == mlx::core::float32);

    const int batchSize = inputBatch.shape(0);
    constexpr int hexFeatureCount = hexCount * featuresPerHex;

    const mlx::core::array perHexFeatures = mlx::core::reshape(
        mlx::core::slice(inputBatch, {0, 0}, {batchSize, hexFeatureCount}),
        {batchSize, hexCount, featuresPerHex});
    const mlx::core::array globalFeatures = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::slice(inputBatch, {0, hexFeatureCount},
                             {batchSize, encodedBoardSize}),
            {batchSize, 1, globalFeatureCount}),
        {batchSize, hexCount, globalFeatureCount});

    mlx::core::array tokens = m_embed(
        mlx::core::concatenate({perHexFeatures, globalFeatures}, 2), parameters);
    tokens = tokens + parameters[m_positionIndex];
    tokens = m_block0(std::move(tokens), parameters);
    tokens = m_block1(std::move(tokens), parameters);
    tokens = m_block2(std::move(tokens), parameters);
    tokens = m_block3(std::move(tokens), parameters);
    tokens = m_block4(std::move(tokens), parameters);
    tokens = m_block5(std::move(tokens), parameters);

    // The blocks are pre-norm, so nothing above ever normalized the residual
    // stream itself; without this both heads would read an activation scale
    // that grows with depth and drifts over training.
    tokens = m_finalNorm(std::move(tokens), parameters);

    const mlx::core::array policy = mlx::core::reshape(
        m_policy(tokens, parameters), {batchSize, moveIdCount});
    const mlx::core::array pooled =
        mlx::core::mean(tokens, std::vector<int>{1}, false);
    mlx::core::array valueHidden = mlx::core::matmul(
        pooled, parameters[m_valueHiddenIndex]);
    valueHidden = Gelu{}(
        valueHidden + parameters[m_valueHiddenBiasIndex], parameters);
    const mlx::core::array value = mlx::core::tanh(mlx::core::reshape(
        mlx::core::matmul(valueHidden, parameters[m_valueOutputIndex])
            + parameters[m_valueOutputBiasIndex],
        {batchSize}));
    return {policy, value};
}

TransformerNetworkXL::TransformerNetworkXL(uint64_t seed)
    : Network(identifier, seed)
    , m_embed(*this, "embed")
    , m_positionIndex(addParameter(
          "position", mlx::core::random::normal(
                          {hexCount, embeddingWidth}, mlx::core::float32, 0.0f, 0.02f)))
    , m_block0(*this, "block0", residualInitScale())
    , m_block1(*this, "block1", residualInitScale())
    , m_block2(*this, "block2", residualInitScale())
    , m_block3(*this, "block3", residualInitScale())
    , m_block4(*this, "block4", residualInitScale())
    , m_block5(*this, "block5", residualInitScale())
    , m_block6(*this, "block6", residualInitScale())
    , m_block7(*this, "block7", residualInitScale())
    , m_finalNorm(*this, "final_norm")
    , m_policy(*this, "policy")
    , m_valueHiddenIndex(addParameter(
          "value.hidden", mlx::core::random::normal(
                              {embeddingWidth, embeddingWidth}, mlx::core::float32,
                              0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueHiddenBiasIndex(addParameter(
          "value.hidden.bias", mlx::core::zeros({embeddingWidth}, mlx::core::float32)))
    , m_valueOutputIndex(addParameter(
          "value.out", mlx::core::random::normal(
                           {embeddingWidth, 1}, mlx::core::float32,
                           0.0f, 1.0f / std::sqrt(static_cast<float>(embeddingWidth)))))
    , m_valueOutputBiasIndex(addParameter(
          "value.out.bias", mlx::core::zeros({1}, mlx::core::float32)))
{
    materializeParameters();
}

TransformerNetworkXL::TransformerNetworkXL(const std::filesystem::path& checkpoint)
    : TransformerNetworkXL(0)
{
    load(checkpoint);
}

Prediction TransformerNetworkXL::forward(
    mlx::core::array inputBatch,
    std::span<const mlx::core::array> parameters) const
{
    assert(inputBatch.ndim() == 2);
    assert(inputBatch.shape(1) == encodedBoardSize);
    assert(inputBatch.dtype() == mlx::core::float32);

    const int batchSize = inputBatch.shape(0);
    constexpr int hexFeatureCount = hexCount * featuresPerHex;
    const mlx::core::array perHexFeatures = mlx::core::reshape(
        mlx::core::slice(inputBatch, {0, 0}, {batchSize, hexFeatureCount}),
        {batchSize, hexCount, featuresPerHex});
    const mlx::core::array globalFeatures = mlx::core::broadcast_to(
        mlx::core::reshape(
            mlx::core::slice(inputBatch, {0, hexFeatureCount},
                             {batchSize, encodedBoardSize}),
            {batchSize, 1, globalFeatureCount}),
        {batchSize, hexCount, globalFeatureCount});

    mlx::core::array tokens = m_embed(
        mlx::core::concatenate({perHexFeatures, globalFeatures}, 2), parameters);
    tokens = tokens + parameters[m_positionIndex];
    tokens = m_block0(std::move(tokens), parameters);
    tokens = m_block1(std::move(tokens), parameters);
    tokens = m_block2(std::move(tokens), parameters);
    tokens = m_block3(std::move(tokens), parameters);
    tokens = m_block4(std::move(tokens), parameters);
    tokens = m_block5(std::move(tokens), parameters);
    tokens = m_block6(std::move(tokens), parameters);
    tokens = m_block7(std::move(tokens), parameters);
    tokens = m_finalNorm(std::move(tokens), parameters);

    const mlx::core::array policy = mlx::core::reshape(
        m_policy(tokens, parameters), {batchSize, moveIdCount});
    const mlx::core::array pooled = mlx::core::mean(tokens, std::vector<int>{1}, false);
    mlx::core::array valueHidden = mlx::core::matmul(
        pooled, parameters[m_valueHiddenIndex]);
    valueHidden = Gelu{}(
        valueHidden + parameters[m_valueHiddenBiasIndex], parameters);
    const mlx::core::array value = mlx::core::tanh(mlx::core::reshape(
        mlx::core::matmul(valueHidden, parameters[m_valueOutputIndex])
            + parameters[m_valueOutputBiasIndex],
        {batchSize}));
    return {policy, value};
}

template class TransformerBlock<128, 8>;
template class TransformerBlock<256, 8>;

} // namespace amoeba_bot
