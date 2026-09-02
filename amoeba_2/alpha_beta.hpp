#pragma once

#include "amoeba.hpp"

#include <chrono>
#include <cstdint>
#include <span>

namespace amoeba_bot
{

struct SearchResult
{
    uint16_t moveId;
    int completedDepth;
    uint64_t nodes;
    int score;
};

// Iterative-deepening negamax.  The caller supplies the complete position
// history because repetition is part of an Amoeba position's game state.
SearchResult chooseAlphaBetaMove(const Board& root, std::span<const uint64_t> history,
                                 std::chrono::steady_clock::time_point deadline);

} // namespace amoeba_bot
