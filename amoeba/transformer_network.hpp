#pragma once

#include "network.hpp"

namespace amoeba_bot
{

inline constexpr int policyOutputsPerHex = directionCount * 2;

template<size_t Width, size_t HeadCount>
class TransformerBlock
{
public:
    // residualScale shrinks only the two matrices that write into the residual
    // stream. Every block adds its output onto that shared stream, so with
    // unscaled writes the stream's magnitude grows with depth and the first
    // blocks' contributions drown; 1/sqrt(2 * blocks) keeps the summed stream
    // near the scale of a single write regardless of depth.
    template<NetworkType N>
    TransformerBlock(N& network, std::string_view name, float residualScale)
        : m_norm1(network, childName(name, "norm1"))
        , m_queryIndex(network.addParameter(
              childName(name, "attn.query"), randomMatrix<Width, Width>()))
        , m_keyIndex(network.addParameter(
              childName(name, "attn.key"), randomMatrix<Width, Width>()))
        , m_valueIndex(network.addParameter(
              childName(name, "attn.value"), randomMatrix<Width, Width>()))
        , m_outputIndex(network.addParameter(
              childName(name, "attn.out"), randomMatrix<Width, Width>(residualScale)))
        , m_relationBiasIndex(network.addParameter(
              childName(name, "attn.bias"),
              mlx::core::zeros({static_cast<int>(HeadCount), relationBucketCount},
                               mlx::core::float32)))
        , m_norm2(network, childName(name, "norm2"))
        , m_mlpInputIndex(network.addParameter(
              childName(name, "mlp.in"), randomMatrix<Width, Width * 4>()))
        , m_mlpInputBiasIndex(network.addParameter(
              childName(name, "mlp.in.bias"),
              mlx::core::zeros({static_cast<int>(Width * 4)}, mlx::core::float32)))
        , m_mlpOutputIndex(network.addParameter(
              childName(name, "mlp.out"), randomMatrix<Width * 4, Width>(residualScale)))
        , m_mlpOutputBiasIndex(network.addParameter(
              childName(name, "mlp.out.bias"),
              mlx::core::zeros({static_cast<int>(Width)}, mlx::core::float32)))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const;

private:
    // Bucket 0 is "no shared line", bucket 1 is "this token itself" - without a
    // dedicated self bucket the diagonal would share bucket 0 and a head could
    // not cheaply prefer (or suppress) its own hex. Buckets 2.. are one per
    // (direction, distance) pair a stack move can relate two hexes by.
    inline static constexpr int relationBucketCount =
        2 + directionCount * maximumMovableStackHeight;

    template<size_t Input, size_t Output>
    static mlx::core::array randomMatrix(float scale = 1.0f)
    {
        return mlx::core::random::normal(
            {static_cast<int>(Input), static_cast<int>(Output)}, mlx::core::float32,
            0.0f, scale / std::sqrt(static_cast<float>(Input)));
    }

    static mlx::core::array splitHeads(const mlx::core::array& input);
    static mlx::core::array mergeHeads(const mlx::core::array& input);

    LayerNorm<Width> m_norm1;
    size_t m_queryIndex;
    size_t m_keyIndex;
    size_t m_valueIndex;
    size_t m_outputIndex;
    size_t m_relationBiasIndex;
    LayerNorm<Width> m_norm2;
    size_t m_mlpInputIndex;
    size_t m_mlpInputBiasIndex;
    size_t m_mlpOutputIndex;
    size_t m_mlpOutputBiasIndex;
};

class TransformerNetwork final : public Network
{
public:
    // v2 marks the post-collapse scheme: decoupled weight decay, final
    // LayerNorm, depth-scaled residual init, self relation bucket. Every v1
    // checkpoint predates the decay fix and may contain dead tensors, so it
    // must be rejected at load, never resumed.
    inline static constexpr const char* identifier =
        "amoeba-relation-transformer-v2-6x128x8";

    explicit TransformerNetwork(uint64_t seed);
    explicit TransformerNetwork(const std::filesystem::path& checkpoint);

private:
    static constexpr int embeddingWidth = 128;
    static constexpr int attentionHeadCount = 8;
    static constexpr int blockCount = 6;

    static float residualInitScale()
    {
        return 1.0f / std::sqrt(2.0f * blockCount);
    }

    Prediction forward(mlx::core::array inputBatch,
                       std::span<const mlx::core::array> parameters) const override;

    Linear<featuresPerHex + globalFeatureCount, embeddingWidth> m_embed;
    size_t m_positionIndex;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block0;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block1;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block2;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block3;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block4;
    TransformerBlock<embeddingWidth, attentionHeadCount> m_block5;
    LayerNorm<embeddingWidth> m_finalNorm;
    Linear<embeddingWidth, policyOutputsPerHex> m_policy;
    size_t m_valueHiddenIndex;
    size_t m_valueHiddenBiasIndex;
    size_t m_valueOutputIndex;
    size_t m_valueOutputBiasIndex;
};

} // namespace amoeba_bot
