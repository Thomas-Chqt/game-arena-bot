#include "hex_ray_lite_network.hpp"

namespace amoeba_bot
{

namespace
{

mlx::core::array gatherRays(
    const mlx::core::array& tokens, const mlx::core::array& offboard)
{
    static constexpr auto flattenedRayIndices = [] {
        std::array<int32_t, hexCount * directionCount * maximumMovableStackHeight> result{};
        size_t output = 0;
        for (uint8_t source = 0; source < hexCount; ++source)
        {
            for (uint8_t direction = 0; direction < directionCount; ++direction)
            {
                for (uint8_t distance = 1; distance <= maximumMovableStackHeight; ++distance)
                {
                    result[output++] = destinationHex(source, directions[direction], distance)
                                           .value_or(hexCount);
                }
            }
        }
        return result;
    }();
    static const mlx::core::array rayIndices(
        flattenedRayIndices.data(),
        mlx::core::Shape{static_cast<int>(flattenedRayIndices.size())},
        mlx::core::int32);

    const int batchSize = tokens.shape(0);
    const mlx::core::array boundary = mlx::core::broadcast_to(
        mlx::core::reshape(offboard, {1, 1, 24}), {batchSize, 1, 24});
    const mlx::core::array extended =
        mlx::core::concatenate({tokens, boundary}, 1);
    return mlx::core::reshape(
        mlx::core::take(extended, rayIndices, 1),
        {batchSize, hexCount, directionCount,
         maximumMovableStackHeight, 24});
}

} // namespace

HexRayBlock::HexRayBlock(Network& network, std::string_view name)
    : m_rayWeightsIndex(network.addParameter(
          childName(name, "ray_weights"),
          mlx::core::random::normal(
              {directionCount * maximumMovableStackHeight, 24},
              mlx::core::float32, 0.0f, 1.0f / 6.0f)))
    , m_local(network, childName(name, "local"))
    , m_global(network, childName(name, "global"))
    , m_output(network, childName(name, "output"))
{
}

mlx::core::array HexRayBlock::operator()(
    mlx::core::array tokens, const mlx::core::array& globals,
    const mlx::core::array& offboard,
    std::span<const mlx::core::array> parameters) const
{
    const int batchSize = tokens.shape(0);
    const mlx::core::array gathered = mlx::core::reshape(
        gatherRays(tokens, offboard),
        {batchSize, hexCount,
         directionCount * maximumMovableStackHeight, 24});
    const mlx::core::array rayWeights = mlx::core::reshape(
        parameters[m_rayWeightsIndex],
        {1, 1, directionCount * maximumMovableStackHeight, 24});
    const mlx::core::array rays = mlx::core::sum(gathered * rayWeights, 2);

    mlx::core::array hidden = m_local(tokens + rays, parameters);
    hidden = hidden + mlx::core::reshape(
        m_global(globals, parameters), {batchSize, 1, 48});
    return tokens + m_output(Gelu{}(std::move(hidden), parameters), parameters);
}

HexRayLiteNetwork::HexRayLiteNetwork(uint64_t seed)
    : Network(identifier, seed)
    , m_embed(*this, "embed")
    , m_offboardIndex(addParameter(
          "offboard", mlx::core::zeros({24}, mlx::core::float32)))
    , m_block0(*this, "blocks.0")
    , m_block1(*this, "blocks.1")
    , m_block2(*this, "blocks.2")
    , m_block3(*this, "blocks.3")
    , m_policyDistanceWeightsIndex(addParameter(
          "policy.distance_weights",
          mlx::core::random::normal(
              {maximumMovableStackHeight, 24}, mlx::core::float32,
              0.0f, 1.0f / std::sqrt(static_cast<float>(maximumMovableStackHeight)))))
    , m_policy(*this, "policy.head")
    , m_valueWeight(*this, "value.weight")
    , m_value(*this, "value.head")
{
    materializeParameters();
}

HexRayLiteNetwork::HexRayLiteNetwork(const std::filesystem::path& checkpoint)
    : Network(identifier, 0)
    , m_embed(*this, "embed")
    , m_offboardIndex(addParameter(
          "offboard", mlx::core::zeros({24}, mlx::core::float32)))
    , m_block0(*this, "blocks.0")
    , m_block1(*this, "blocks.1")
    , m_block2(*this, "blocks.2")
    , m_block3(*this, "blocks.3")
    , m_policyDistanceWeightsIndex(addParameter(
          "policy.distance_weights",
          mlx::core::random::normal(
              {maximumMovableStackHeight, 24}, mlx::core::float32,
              0.0f, 1.0f / std::sqrt(static_cast<float>(maximumMovableStackHeight)))))
    , m_policy(*this, "policy.head")
    , m_valueWeight(*this, "value.weight")
    , m_value(*this, "value.head")
{
    load(checkpoint);
}

Prediction HexRayLiteNetwork::forward(
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
    const mlx::core::array globals = mlx::core::slice(
        inputBatch, {0, hexFeatureCount}, {batchSize, encodedBoardSize});

    mlx::core::array tokens = m_embed(perHexFeatures, parameters);
    tokens = m_block0(std::move(tokens), globals, parameters[m_offboardIndex], parameters);
    tokens = m_block1(std::move(tokens), globals, parameters[m_offboardIndex], parameters);
    tokens = m_block2(std::move(tokens), globals, parameters[m_offboardIndex], parameters);
    tokens = m_block3(std::move(tokens), globals, parameters[m_offboardIndex], parameters);

    const mlx::core::array gathered = gatherRays(tokens, parameters[m_offboardIndex]);
    const mlx::core::array distanceWeights = mlx::core::reshape(
        parameters[m_policyDistanceWeightsIndex],
        {1, 1, 1, maximumMovableStackHeight, 24});
    const mlx::core::array directional =
        mlx::core::sum(gathered * distanceWeights, 3);
    const mlx::core::array sources = mlx::core::broadcast_to(
        mlx::core::reshape(tokens, {batchSize, hexCount, 1, 24}),
        {batchSize, hexCount, directionCount, 24});
    const mlx::core::array legal = mlx::core::reshape(
        mlx::core::slice(
            perHexFeatures, {0, 0, legalMovesOffset},
            {batchSize, hexCount, featuresPerHex}),
        {batchSize, hexCount, directionCount, 2});
    const mlx::core::array policy = mlx::core::reshape(
        m_policy(mlx::core::concatenate({sources, directional, legal}, 3), parameters),
        {batchSize, moveIdCount});

    const mlx::core::array valueWeights = mlx::core::softmax(
        m_valueWeight(tokens, parameters), 1);
    const mlx::core::array pooled = mlx::core::sum(tokens * valueWeights, 1);
    const mlx::core::array value = mlx::core::reshape(
        m_value(mlx::core::concatenate({pooled, globals}, 1), parameters),
        {batchSize});
    return {policy, value};
}

} // namespace amoeba_bot
