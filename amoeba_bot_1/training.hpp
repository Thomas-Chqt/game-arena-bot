#pragma once

// Turning search results into a number to minimise.
//
// The network predicts two things, so the loss compares two things: how close its
// move preferences are to the visit counts the search settled on, and how close
// its verdict is to who actually won. Adding them is what makes one shared trunk
// serve both heads - whatever tells you a position is winning overlaps heavily
// with what tells you a move is good.

#include "network.hpp"

#include <span>
#include <vector>

namespace bot
{

// One training batch, already in the network's own terms: input as encode() lays
// it out, and the policy target permuted into the same flipped space forward()
// answers in. makeBatch() is what puts it there.
struct Batch
{
    mlx::core::array input;         // [batch, kEncodedSize]
    mlx::core::array legal;         // [batch, kNumMoveIds], 1 or 0
    mlx::core::array policyTarget;  // [batch, kNumMoveIds], sums to 1 over legal moves
    mlx::core::array valueTarget;   // [batch], in [-1, 1]
};

// `visits` is indexed by absolute amoeba::Move::id, straight off Search::run.
// `outcomes` is already from the point of view of the side to move at that
// position: +1 if they went on to win, -1 if they lost, 0 for a draw. Getting
// that sign wrong trains a bot that prefers losing while the loss curve stays
// healthy, so it is deliberately the caller's business - only self-play knows the
// result - and this function will not second-guess it.
Batch makeBatch(std::span<const amoeba::Board* const> boards, std::span<const VisitCounts> visits, std::span<const float> outcomes);

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
std::vector<mlx::core::array> loss(const std::vector<mlx::core::array>& params, NetworkShape shape, const Batch& batch, float weightDecay);

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
    explicit Adam(const std::vector<mlx::core::array>& params, AdamConfig config = {});

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
    std::vector<mlx::core::array> step(const std::vector<mlx::core::array>& params,
                                       const std::vector<mlx::core::array>& gradients, float rate);

    int steps() const { return m_steps; }

private:
    AdamConfig m_config;
    std::vector<mlx::core::array> m_mean;
    std::vector<mlx::core::array> m_variance;
    int m_steps = 0;
};

} // namespace bot
