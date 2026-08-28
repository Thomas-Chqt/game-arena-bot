#pragma once

#include "network.hpp"

namespace amoeba_bot
{

class HexRayBlock
{
public:
    HexRayBlock(Network& network, std::string_view name);

    mlx::core::array operator()(
        mlx::core::array tokens, const mlx::core::array& globals,
        const mlx::core::array& offboard,
        std::span<const mlx::core::array> parameters) const;

private:
    size_t m_rayWeightsIndex;
    Linear<24, 48> m_local;
    Linear<globalFeatureCount, 48> m_global;
    Linear<48, 24> m_output;
};

class HexRayLiteNetwork final : public Network
{
public:
    inline static constexpr const char* identifier = "amoeba-hex-ray-lite-v1-4x24";

    explicit HexRayLiteNetwork(uint64_t seed);
    explicit HexRayLiteNetwork(const std::filesystem::path& checkpoint);

private:
    Prediction forward(mlx::core::array inputBatch,
                       std::span<const mlx::core::array> parameters) const override;

    Linear<featuresPerHex, 24> m_embed;
    size_t m_offboardIndex;
    HexRayBlock m_block0;
    HexRayBlock m_block1;
    HexRayBlock m_block2;
    HexRayBlock m_block3;
    size_t m_policyDistanceWeightsIndex;
    Sequential<Linear<50, 8>, Gelu, Linear<8, 2>> m_policy;
    Linear<24, 1> m_valueWeight;
    Sequential<Linear<24 + globalFeatureCount, 16>, Gelu,
               Linear<16, 1>, Tanh> m_value;
};

} // namespace amoeba_bot
