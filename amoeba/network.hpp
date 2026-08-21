#ifndef NETWORK_HPP
#define NETWORK_HPP

// The neural network used to evaluate MCTS leaves. Untrained: runInference()
// computes, nothing fits it yet.
//
// encodeBoard() already produces 37 hex blocks plus 8 globals, so the natural unit is
// a token per hex: 37 tokens, attention between all of them, a policy head of 12
// logits per token - which is exactly amoeba::Move::id - and one scalar value.
//
// Attention rather than convolution because Amoeba's interactions are long-range
// by construction: a stack jumps over everything and lands at exactly its height,
// so the hex six steps out on your line matters and the adjacent one barely does.
// A relative-position bias hands the network that geometry instead of making it
// rediscover the attack relation from game outcomes.

#include "mcts.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace bot
{

// Bucket 0 means "these two hexes share no line"; the rest is one bucket per
// (direction, distance) pair. maximumMovableStackHeight doubles as the longest straight run on
// the board, which is why it bounds the distance. That this comes to 37, the
// same as hexCount, is a coincidence.
inline constexpr int relativePositionBucketCount = 1 + amoeba::directionCount * amoeba::maximumMovableStackHeight;

// 6 directions x (move, sow). Laid out dir-major so that token t's block of 12
// lands on Move::id == (t * 6 + dir) * 2 + splitting with no permutation.
inline constexpr int policyOutputsPerHex = amoeba::directionCount * 2;

// Which relative-position bucket joins two hexes. Bucket 0 means no straight
// line joins them at all; otherwise one bucket per (direction, distance) pair,
// which is exactly the attack relation the rules are written in. The stride for
// direction is the number of distance buckets, not the number of directions -
// they are both 6 here, which makes the mistake invisible.
inline constexpr auto relativePositionBuckets = [] -> std::array<std::array<uint8_t, amoeba::hexCount>, amoeba::hexCount>
{
    std::array<std::array<uint8_t, amoeba::hexCount>, amoeba::hexCount> table{};
    for (int sourceHex = 0; sourceHex < amoeba::hexCount; ++sourceHex)
    {
        for (uint8_t direction = 0; direction < amoeba::directionCount; ++direction)
        {
            for (uint8_t distance = 1; distance <= amoeba::maximumMovableStackHeight; ++distance)
            {
                const int8_t destination = amoeba::destinationHex(static_cast<uint8_t>(sourceHex), direction, distance);
                if (destination >= 0)
                {
                    table[sourceHex][destination] =
                        static_cast<uint8_t>(1 + direction * amoeba::maximumMovableStackHeight + (distance - 1));
                }
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
    int blockCount = 6;
    int embeddingWidth = 128;
    int attentionHeadCount = 8;
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

    const std::vector<mlx::core::array>& parameters() const { return m_parameters; }

    // The optimiser builds a whole new vector each step rather than mutating in
    // place, because an mlx::core::array is a handle onto a graph, not a buffer.
    void replaceParameters(std::vector<mlx::core::array> parameters);

private:
    // Which tensors exist, and in what order value_and_grad will see them. The
    // single definition of that: random init and checkpoint loading both read it.
    static std::vector<std::pair<std::string, mlx::core::Shape>> parameterLayout(NetworkShape shape);

    NetworkShape m_shape;
    std::vector<std::string> m_names;
    std::vector<mlx::core::array> m_parameters;
};

// Raw policy logits, [batch, moveIdCount] and still in the flipped space encodeBoard()
// used, plus one value per position. Masking is the caller's job: inference wants
// probabilities over the legal moves, training wants the logits themselves.
struct Prediction
{
    mlx::core::array policy;
    mlx::core::array value;
};

// `parameters` is walked positionally, so it must be in Network::parameterLayout() order.
Prediction runInference(const std::vector<mlx::core::array>& parameters, NetworkShape shape, const mlx::core::array& input);

// Encodes a batch of boards, runs one forward pass, and hands the search back
// probabilities over legal moves in absolute move ids.
class NetworkEvaluator
{
public:
    explicit NetworkEvaluator(const Network& network)
        : m_network(network)
    {
    }

    void evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> outputs);

private:
    const Network& m_network;
};

// ===========================================================================
// Training: turning search results into a number to minimise
//
// The network predicts two things, so the loss compares two things: how close
// its move preferences are to the visit counts the search settled on, and how
// close its verdict is to who actually won. Adding them is what makes one
// shared trunk serve both heads.
// ===========================================================================

// One training batch, already in the network's own terms: input as encodeBoard() lays
// it out, and the policy target permuted into the same flipped space runInference()
// answers in. makeTrainingBatch() is what puts it there.
struct TrainingBatch
{
    mlx::core::array input;        // [batch, encodedBoardSize]
    mlx::core::array legal;        // [batch, moveIdCount], 1 or 0
    mlx::core::array policyTarget; // [batch, moveIdCount], sums to 1 over legal moves
    mlx::core::array valueTarget;  // [batch], in [-1, 1]
};

// `visits` is indexed by absolute amoeba::Move::id, straight off MCTS::visits().
// `outcomes` is already from the point of view of the side to move at that
// position: +1 if they went on to win, -1 if they lost, 0 for a draw. Getting
// that sign wrong trains a bot that prefers losing while the loss curve stays
// healthy, so it is deliberately the caller's business - only self-play knows the
// result - and this function will not second-guess it.
TrainingBatch makeTrainingBatch(std::span<const amoeba::Board* const> boards, std::span<const VisitCounts> visits, std::span<const float> outcomes);

// Returns { total, policy, value }. The total is the scalar to differentiate; the
// two components come back alongside because they say different things - a value
// loss that sits flat while the policy loss falls means the value head is not
// learning, and one averaged number would hide that.
//
// mlx::core::value_and_grad differentiates the first element, which is why the
// total is first.
//
// weightDecay pulls every parameter gently toward zero, discouraging the network
// from memorising individual positions. It applies to the layer-norm scales and
// biases as well; separating those out is not worth it at this size. Pass 0 when
// overfitting a single batch on purpose, since the penalty otherwise keeps the
// loss off zero and hides whether the gradients actually connect.
std::vector<mlx::core::array> computeLoss(const std::vector<mlx::core::array>& parameters, NetworkShape shape, const TrainingBatch& batch, float weightDecay);

// The decay rates are the conventional beta1 and beta2, named for what they do.
// 0.9 averages roughly the last ten gradients, 0.999 the last thousand. epsilon
// only exists to stop a division by zero where a parameter's gradient has been
// flat at zero.
struct AdamConfig
{
    float meanDecay = 0.9f;
    float varianceDecay = 0.999f;
    float epsilon = 1e-8f;
};

// Gradient descent scaled per parameter.
//
// Plain `w -= rate * gradient` cannot work here: gradient magnitudes across the
// 87 tensors span three orders of magnitude on a single batch, so one rate that
// suits the largest leaves the smallest effectively frozen. Adam divides each
// gradient by the typical size of that parameter's own gradients, which makes the
// step dimensionless and lets one rate serve the whole network.
//
// It keeps two running averages per parameter: the gradient, which smooths batch
// noise into a direction worth trusting, and the gradient squared, which measures
// magnitude regardless of sign.
class Adam
{
public:
    explicit Adam(const std::vector<mlx::core::array>& parameters, AdamConfig config = {});

    // Returns the new parameters; an mlx::core::array is a handle onto a graph
    // rather than a buffer, so nothing is written in place.
    //
    // The caller must mlx::core::eval() the result every step. Without it the
    // graph grows across steps until memory runs out, and the failure looks
    // nothing like the cause. Evaluating the parameters is enough - they depend on
    // both moment estimates, so those are materialised with them.
    //
    // `rate` is an argument rather than config because it is expected to fall on a
    // schedule as training progresses.
    std::vector<mlx::core::array> updateParameters(const std::vector<mlx::core::array>& parameters, const std::vector<mlx::core::array>& gradients, float rate);

    int steps() const { return m_steps; }

private:
    AdamConfig m_config;
    std::vector<mlx::core::array> m_mean;
    std::vector<mlx::core::array> m_variance;
    int m_steps = 0;
};

} // namespace bot

#endif // NETWORK_HPP
