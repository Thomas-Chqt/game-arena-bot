#ifndef AMOEBA_NETWORK_HPP
#define AMOEBA_NETWORK_HPP

#include "mcts.hpp"
#include "network.hpp"

namespace amoeba
{

// Every hex produces six move logits and six sow/split logits. Flattening the
// resulting [37, 12] tensor gives exactly the game's 444 policy indices.
inline constexpr int policyOutputsPerHex = directionCount * 2;

// One relation number for every ordered pair of board hexes. Attention uses
// this fixed table to select a learned score bias that describes board geometry.
struct AmoebaRelationMap
{
    static constexpr size_t tokenCount = hexCount;
    static constexpr size_t bucketCount = 2 + directionCount * maximumMovableStackHeight;

    inline static constexpr auto buckets = []
    {
        std::array<std::array<uint8_t, tokenCount>, tokenCount> table{};
        for (auto& row : table)
            row.fill(1); // Different hexes that do not share a straight line.

        for (size_t source = 0; source < tokenCount; ++source)
        {
            table[source][source] = 0;
            for (uint8_t direction = 0; direction < directionCount; ++direction)
            {
                for (uint8_t distance = 1; distance <= maximumMovableStackHeight; ++distance)
                {
                    const int8_t destination =
                        destinationHex(static_cast<uint8_t>(source), direction, distance);
                    if (destination >= 0)
                    {
                        table[source][destination] = static_cast<uint8_t>(
                            2 + direction * maximumMovableStackHeight + distance - 1);
                    }
                }
            }
        }
        return table;
    }();

    inline static constexpr auto flattened = []
    {
        std::array<int32_t, tokenCount * tokenCount> result{};
        for (size_t source = 0; source < tokenCount; ++source)
        {
            for (size_t destination = 0; destination < tokenCount; ++destination)
                result[source * tokenCount + destination] = buckets[source][destination];
        }
        return result;
    }();

    static const mlx::core::array& indices()
    {
        static const mlx::core::array value(
            flattened.data(), mlx::core::Shape{static_cast<int>(flattened.size())}, mlx::core::int32);
        return value;
    }
};

template<size_t Width, size_t HeadCount>
class TransformerBlock
{
public:
    template<typename NetworkType>
    TransformerBlock(NetworkType& network, std::string_view name)
        : m_norm1(network, childName(name, "norm1"))
        , m_attention(network, childName(name, "attention"))
        , m_norm2(network, childName(name, "norm2"))
        , m_feedForward(network, childName(name, "feed_forward"))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const
    {
        input = input + m_attention(m_norm1(input, parameters), parameters);
        input = input + m_feedForward(m_norm2(input, parameters), parameters);
        return input;
    }

private:
    // Pre-normalization transformer:
    // x = x + attention(norm(x)); x = x + mlp(norm(x)).
    LayerNorm<Width> m_norm1;
    RelationSelfAttention<hexCount, Width, HeadCount, AmoebaRelationMap> m_attention;
    LayerNorm<Width> m_norm2;
    Sequential<Linear<Width, Width * 4>, Gelu, Linear<Width * 4, Width>> m_feedForward;
};

struct Prediction
{
    mlx::core::array policy; // Raw move logits: [batch, 444].
    mlx::core::array value;  // Expected outcome for the side to move: [batch].
};

// This header-only root describes how reusable modules are connected for Amoeba.
// Changing any template argument creates a different C++ network type.
template<size_t BlockCount, size_t Width, size_t HeadCount>
class AmoebaRoot
{
    static_assert(BlockCount > 0, "an Amoeba network needs at least one transformer block");
    static_assert(Width > 0, "an Amoeba network width must be positive");
    static_assert(HeadCount > 0, "an Amoeba network needs at least one attention head");
    static_assert(Width % HeadCount == 0, "network width must be divisible by its attention-head count");

public:
    template<typename NetworkType>
    AmoebaRoot(NetworkType& network, std::string_view)
        : m_embed(network, "embed")
        , m_positionIndex(network.addParameter(
              "position", {hexCount, static_cast<int>(Width)}, Initialization::Normal(0.02f)))
        , m_blocks(network, "blocks")
        , m_finalNorm(network, "final_norm")
        , m_policy(network, "policy")
        , m_value(network, "value")
    {
    }

    std::vector<mlx::core::array> operator()(
        mlx::core::array inputBatch, std::span<const mlx::core::array> parameters) const
    {
        assert(inputBatch.ndim() == 2);
        assert(inputBatch.shape(1) == encodedBoardSize);
        assert(inputBatch.dtype() == mlx::core::float32);
        assert(m_positionIndex < parameters.size());

        const int batchSize = inputBatch.shape(0);
        constexpr int hexFeatureCount = hexCount * featuresPerHex;

        const mlx::core::array perHexFeatures = mlx::core::reshape(
            mlx::core::slice(inputBatch, {0, 0}, {batchSize, hexFeatureCount}),
            {batchSize, hexCount, featuresPerHex});

        const mlx::core::array globalFeatures = mlx::core::broadcast_to(
            mlx::core::reshape(
                mlx::core::slice(inputBatch, {0, hexFeatureCount}, {batchSize, encodedBoardSize}),
                {batchSize, 1, globalFeatureCount}),
            {batchSize, hexCount, globalFeatureCount});

        mlx::core::array tokens = m_embed(
            mlx::core::concatenate({perHexFeatures, globalFeatures}, 2), parameters);
        tokens = tokens + parameters[m_positionIndex];
        tokens = m_blocks(std::move(tokens), parameters);
        tokens = m_finalNorm(std::move(tokens), parameters);

        const mlx::core::array policy = mlx::core::reshape(
            m_policy(tokens, parameters), {batchSize, moveIdCount});
        const mlx::core::array pooled = mlx::core::mean(tokens, std::vector<int>{1}, false);
        const mlx::core::array value = mlx::core::reshape(m_value(pooled, parameters), {batchSize});
        return {policy, value};
    }

private:
    Linear<featuresPerHex + globalFeatureCount, Width> m_embed;
    size_t m_positionIndex;
    Repeat<BlockCount, TransformerBlock<Width, HeadCount>> m_blocks;
    LayerNorm<Width> m_finalNorm;
    Linear<Width, policyOutputsPerHex> m_policy;
    Sequential<Linear<Width, Width>, Gelu, Linear<Width, 1>, Tanh> m_value;
};

using AmoebaNetwork = Network<AmoebaRoot<6, 128, 8>>;

// The name is external to Network and may be chosen freely. A differently sized
// AmoebaRoot is a different type and therefore requires its own specialization.
template<>
struct NetworkName<AmoebaNetwork>
{
    inline static constexpr const char* value = "amoeba-relation-transformer-v2-6x128x8";
};

class NetworkEvaluator
{
public:
    explicit NetworkEvaluator(const AmoebaNetwork& network)
        : m_network(network)
    {
    }

    void evaluate(std::span<const Board* const> boards, std::span<Evaluation> outputs);

private:
    const AmoebaNetwork& m_network;
};

struct TrainingBatch
{
    mlx::core::array input;
    mlx::core::array legal;
    mlx::core::array policyTarget;
    mlx::core::array valueTarget;
};

TrainingBatch makeTrainingBatch(std::span<const Board* const> boards,
                                std::span<const VisitCounts> visits,
                                std::span<const float> outcomes);

std::vector<mlx::core::array> computeLoss(const AmoebaNetwork& network,
                                          const std::vector<mlx::core::array>& parameters,
                                          const TrainingBatch& batch, float weightDecay);

} // namespace amoeba

#endif // AMOEBA_NETWORK_HPP
