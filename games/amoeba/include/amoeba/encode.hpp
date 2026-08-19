#pragma once

#include "amoeba/amoeba.hpp"

#include <span>

namespace amoeba
{

// ---------------------------------------------------------------------------
// Board -> network input.
//
// kNumHexes blocks of kHexFeatures floats, one block per hex, followed by
// kGlobalFeatures floats describing the position as a whole. Every value is in
// [0, 1]: the network's first layer weighs all of them against each other, so a
// feature with a much larger range would drown out the rest.
//
// Categories are one-hot rather than a single number, because the network only
// ever multiplies and adds - given `piece = 3` it would treat a black piece as
// three times a white one, and a kernel as the midpoint of two normal pieces.
//
// Everything is written from the point of view of the side to move: for Black
// the colours are swapped and the board is rotated 180 degrees, so the network
// learns the game once instead of once per colour. Block t therefore describes
// what the mover sees at position t, which is absolute hex kFlipped[t] when
// Black is to move, and direction d in a block means absolute opposite(d). A
// policy over this input needs mapping back the same way before it names a move
// the server will accept.
// ---------------------------------------------------------------------------

// Stacks taller than kMovableMax can never move again, so their internal order
// can never matter again either. Six slots covers every stack still in play.
inline constexpr int kSlotDepth     = kMovableMax;
inline constexpr int kPieceCodes    = 5;                // empty, my normal, my kernel, their normal, their kernel
inline constexpr int kHeightBuckets = kMovableMax + 2;  // 0..6 exactly, then 7-and-up

inline constexpr int kHexFeatures =
      kSlotDepth * kPieceCodes   // stack contents, bottom first: slot d is the piece a sow lands d + 1 hexes away
    + kHeightBuckets             // height one-hot - height selects a rule, it is not a magnitude
    + kPieceCodes                // top piece, the only thing above kSlotDepth that still matters
    + 2                          // my kernel / their kernel buried anywhere in this stack
    + 2                          // my pieces / their pieces in this stack
    + kNumDirs * 2;              // legality per direction, move and sow, straight out of Board::legal

inline constexpr int kGlobalFeatures = 8;
inline constexpr int kEncodedSize    = kNumHexes * kHexFeatures + kGlobalFeatures;

static_assert(kHexFeatures == 59);

// Offsets within one hex block.
inline constexpr int kOffSlots   = 0;
inline constexpr int kOffHeight  = kOffSlots + kSlotDepth * kPieceCodes;
inline constexpr int kOffTop     = kOffHeight + kHeightBuckets;
inline constexpr int kOffKernels = kOffTop + kPieceCodes;
inline constexpr int kOffCounts  = kOffKernels + 2;
inline constexpr int kOffLegal   = kOffCounts + 2;

// The globals, in order.
inline constexpr int kGlobalPly            = 0;
inline constexpr int kGlobalStaleness      = 1;
inline constexpr int kGlobalRepeats        = 2;
inline constexpr int kGlobalMyStacks       = 3;
inline constexpr int kGlobalTheirStacks    = 4;
inline constexpr int kGlobalMyPrisoners    = 5;
inline constexpr int kGlobalTheirPrisoners = 6;
inline constexpr int kGlobalInCheck        = 7;

// Requires b.legal and the kernel cache to be populated - apply() and
// fromString() both do it.
void encode(const Board& b, std::span<float, kEncodedSize> out);

} // namespace amoeba
