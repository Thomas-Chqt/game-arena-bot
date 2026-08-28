#pragma once

#include "network.hpp"

namespace amoeba_bot
{

inline constexpr int policyOutputsPerHex = directionCount * 2;

template<size_t Width, size_t HeadCount>
class TransformerBlock
{
public:
    template<NetworkType N>
    TransformerBlock(N& network, std::string_view name)
        : m_norm1(network, childName(name, "norm1"))
        , m_queryIndex(network.addParameter(
              childName(name, "attn.query"), randomMatrix<Width, Width>()))
        , m_keyIndex(network.addParameter(
              childName(name, "attn.key"), randomMatrix<Width, Width>()))
        , m_valueIndex(network.addParameter(
              childName(name, "attn.value"), randomMatrix<Width, Width>()))
        , m_outputIndex(network.addParameter(
              childName(name, "attn.out"), randomMatrix<Width, Width>()))
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
              childName(name, "mlp.out"), randomMatrix<Width * 4, Width>()))
        , m_mlpOutputBiasIndex(network.addParameter(
              childName(name, "mlp.out.bias"),
              mlx::core::zeros({static_cast<int>(Width)}, mlx::core::float32)))
    {
    }

    mlx::core::array operator()(mlx::core::array input,
                                std::span<const mlx::core::array> parameters) const;

private:
    inline static constexpr int relationBucketCount =
        1 + directionCount * maximumMovableStackHeight;

    template<size_t Input, size_t Output>
    static mlx::core::array randomMatrix()
    {
        return mlx::core::random::normal(
            {static_cast<int>(Input), static_cast<int>(Output)}, mlx::core::float32,
            0.0f, 1.0f / std::sqrt(static_cast<float>(Input)));
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
    inline static constexpr const char* identifier =
        "amoeba-relation-transformer-v1-6x128x8";

    explicit TransformerNetwork(uint64_t seed);
    explicit TransformerNetwork(const std::filesystem::path& checkpoint);

private:
    static constexpr int embeddingWidth = 128;
    static constexpr int attentionHeadCount = 8;

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
    Linear<embeddingWidth, policyOutputsPerHex> m_policy;
    size_t m_valueHiddenIndex;
    size_t m_valueHiddenBiasIndex;
    size_t m_valueOutputIndex;
    size_t m_valueOutputBiasIndex;
};

} // namespace amoeba_bot
