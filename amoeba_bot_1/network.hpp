#pragma once

// The network behind bot::Evaluator. Untrained: forward() computes, nothing fits it yet.
//
// encode() already produces 37 hex blocks plus 8 globals, so the natural unit is
// a token per hex: 37 tokens, attention between all of them, a policy head of 12
// logits per token - which is exactly amoeba::Move::id - and one scalar value.
//
// Attention rather than convolution because Amoeba's interactions are long-range
// by construction: a stack jumps over everything and lands at exactly its height,
// so the hex six steps out on your line matters and the adjacent one barely does.
// A relative-position bias hands the network that geometry instead of making it
// rediscover the attack relation from game outcomes.

#include "mcts.hpp"

#include <amoeba/amoeba.hpp>
#include <amoeba/encode.hpp>

#include <mlx/mlx.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace bot
{

// Bucket 0 means "these two hexes share no line"; the rest is one bucket per
// (direction, distance) pair. kMovableMax doubles as the longest straight run on
// the board, which is why it bounds the distance. That this comes to 37, the
// same as kNumHexes, is a coincidence.
inline constexpr int kPositionBuckets = 1 + amoeba::kNumDirs * amoeba::kMovableMax;

// 6 directions x (move, sow). Laid out dir-major so that token t's block of 12
// lands on Move::id == (t * 6 + dir) * 2 + splitting with no permutation.
inline constexpr int kPolicyPerHex = amoeba::kNumDirs * 2;

// Which relative-position bucket joins two hexes. Bucket 0 means no straight
// line joins them at all; otherwise one bucket per (direction, distance) pair,
// which is exactly the attack relation the rules are written in. The stride for
// direction is the number of distance buckets, not the number of directions -
// they are both 6 here, which makes the mistake invisible.
inline constexpr auto kBucket = [] -> std::array<std::array<uint8_t, amoeba::kNumHexes>, amoeba::kNumHexes> {
    std::array<std::array<uint8_t, amoeba::kNumHexes>, amoeba::kNumHexes> table{};
    for (int i = 0; i < amoeba::kNumHexes; ++i) {
        for (uint8_t dir = 0; dir < amoeba::kNumDirs; ++dir) {
            for (uint8_t distance = 1; distance <= amoeba::kMovableMax; ++distance)
            {
                const int8_t reached = amoeba::ray(static_cast<uint8_t>(i), dir, distance);
                if (reached >= 0)
                    table[i][reached] = static_cast<uint8_t>(1 + dir * amoeba::kMovableMax + (distance - 1));
            }
        }
    }
    return table;
}();

// Width and depth are the two knobs worth turning. 6 blocks at width 128 is
// ~1.2M parameters; the first end-to-end run should use 2 blocks instead, since
// its only job is to prove data flows from self-play into training and back.
struct NetworkShape
{
    int blocks = 6;
    int width  = 128;
    int heads  = 8;
};

// Parameters live in one flat vector rather than named fields because that is
// what mlx::core::value_and_grad takes, and the shape it returns gradients in.
// m_names runs parallel to it and matters only for checkpointing.
class Network
{
public:
    Network(NetworkShape shape, uint64_t seed);
    explicit Network(const std::filesystem::path& checkpoint);

    void save(const std::filesystem::path& checkpoint) const;

    NetworkShape shape() const { return m_shape; }
    size_t parameterCount() const;

    const std::vector<mlx::core::array>& parameters() const { return m_params; }

    // The optimiser builds a whole new vector each step rather than mutating in
    // place, because an mlx::core::array is a handle onto a graph, not a buffer.
    void replaceParameters(std::vector<mlx::core::array> params);

private:
    // Which tensors exist, and in what order value_and_grad will see them. The
    // single definition of that: random init and checkpoint loading both read it.
    static std::vector<std::pair<std::string, mlx::core::Shape>> layout(NetworkShape shape);

    NetworkShape m_shape;
    std::vector<std::string> m_names;
    std::vector<mlx::core::array> m_params;
};

// Raw policy logits, [batch, kNumMoveIds] and still in the flipped space encode()
// used, plus one value per position. Masking is the caller's job: inference wants
// probabilities over the legal moves, training wants the logits themselves.
struct Prediction
{
    mlx::core::array policy;
    mlx::core::array value;
};

// `params` is walked positionally, so it must be in Network::layout() order.
Prediction forward(const std::vector<mlx::core::array>& params, NetworkShape shape, const mlx::core::array& input);

// Encodes a batch of boards, runs one forward pass, and hands the search back
// probabilities over legal moves in absolute move ids.
class NetworkEvaluator final : public Evaluator
{
public:
    explicit NetworkEvaluator(const Network& network) : m_network(network) {}

    void evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out) override;

private:
    const Network& m_network;
};

} // namespace bot
