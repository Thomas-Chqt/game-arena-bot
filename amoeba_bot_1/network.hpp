#pragma once

// ---------------------------------------------------------------------------
// The seam with the network branch. THIS FILE IS DECLARATIONS ONLY - there is
// no network.cpp yet, so both programs compile and then fail to link. That is
// the intended state until the MLX model lands.
//
// Everything below is what play.cpp and train.cpp actually need; the network
// branch owns the private sections and the implementation, and this header is
// where the two branches are expected to meet and disagree once.
//
// Note what is *not* here. The network never sees a hex, a direction or a move:
// it is handed whole Boards and hands back an Evaluation, exactly the interface
// the search already has. amoeba::encode() turns a Board into the input array
// and amoeba::kFlippedMove maps the policy back out of the mover's frame - both
// live in games/amoeba and stay MLX-free, so the encoding can be tested without
// a model.
// ---------------------------------------------------------------------------

#include "mcts.hpp"

#include <amoeba/amoeba.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <span>

namespace bot
{

// One position's worth of training data.
//
// The board rather than its encoding, because it is six times smaller in the
// replay buffer and because the twelve-fold symmetry augmentation needs to
// permute hexes and directions, which it cannot do to a flat array of floats.
//
// `policy` is in absolute move ids, the frame the search works in. Flipping it
// into the mover's frame is the trainer's job, alongside the encode() it has to
// match - keeping both flips in one place is the whole reason they agree.
struct Sample
{
    amoeba::Board                          board;
    std::array<float, amoeba::kNumMoveIds> policy;   // visit counts, normalised over the legal moves
    float                                  value;    // outcomeFor(final state, board.whiteToMove)
};

static_assert(std::is_trivially_copyable_v<Sample>, "the replay buffer writes these to disk raw");

class Network final : public Evaluator
{
public:
    ~Network() override;

    void evaluate(std::span<const amoeba::Board* const> boards, std::span<Evaluation> out) override;

    void save(const std::filesystem::path&) const;

private:
    // Owned by the network branch.
};

std::unique_ptr<Network> loadNetwork(const std::filesystem::path&);

// Generation 0: the untrained starting point, random weights. It plays terribly
// and that is fine - it is the thing generation 1 has to beat.
std::unique_ptr<Network> newNetwork(uint64_t seed);

// Gradient descent on one network. Holds the optimiser state, so it must
// outlive the run of training steps rather than being made per batch.
class Trainer
{
public:
    explicit Trainer(Network&);
    ~Trainer();

    // One gradient step on one batch, returning the loss for logging. Applies
    // the symmetry augmentation and the perspective flip internally.
    float step(std::span<const Sample> batch);

private:
    // Owned by the network branch.
};

} // namespace bot
