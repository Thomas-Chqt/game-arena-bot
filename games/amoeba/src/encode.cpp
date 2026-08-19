#include "amoeba/encode.hpp"

#include <algorithm>
#include <cassert>

namespace amoeba
{

namespace
{

// Swapping colours for Black lands exactly on the Piece enum's own numbering, so
// the perspective code is just the swapped enum value: 0 empty, 1 my normal,
// 2 my kernel, 3 their normal, 4 their kernel.
constexpr int perspectiveCode(Piece p, bool whiteToMove)
{
    return static_cast<int>(whiteToMove ? p : kSwapColour[static_cast<uint8_t>(p)]);
}

} // namespace

void encode(const Board& b, std::span<float, kEncodedSize> out)
{
    std::ranges::fill(out, 0.0f);

    const bool  me    = b.whiteToMove;
    const float scale = 1.0f / kPiecesPerSide;

    int myStacks = 0, theirStacks = 0, myPrisoners = 0, theirPrisoners = 0;

    for (int token = 0; token < kNumHexes; ++token)
    {
        const uint8_t source = me ? static_cast<uint8_t>(token) : kFlipped[token];
        const Hex&    hx     = b.hexes[source];
        const uint8_t height = hx.height();
        float* const  block  = out.data() + token * kHexFeatures;

        // Hex::at reports Empty past the top, which is code 0, so short stacks
        // pad themselves.
        for (uint8_t d = 0; d < kSlotDepth; ++d)
            block[kOffSlots + d * kPieceCodes + perspectiveCode(hx.at(d), me)] = 1.0f;

        block[kOffHeight + std::min<int>(height, kMovableMax + 1)] = 1.0f;
        block[kOffTop + perspectiveCode(hx.top(), me)]             = 1.0f;

        int mine = 0, theirs = 0;
        for (uint8_t d = 0; d < height; ++d)
        {
            const int code = perspectiveCode(hx.at(d), me);
            if (code <= 2) ++mine; else ++theirs;
            if (code == 2) block[kOffKernels] = 1.0f;
            if (code == 4) block[kOffKernels + 1] = 1.0f;
        }
        block[kOffCounts]     = static_cast<float>(mine) * scale;
        block[kOffCounts + 1] = static_cast<float>(theirs) * scale;

        if (height > 0)
        {
            if (perspectiveCode(hx.top(), me) <= 2)
            {
                ++myStacks;
                myPrisoners += theirs;
            }
            else
            {
                ++theirStacks;
                theirPrisoners += mine;
            }
        }

        for (uint8_t dir = 0; dir < kNumDirs; ++dir)
        {
            const uint8_t absolute = me ? dir : opposite(dir);
            for (int splitting = 0; splitting < 2; ++splitting)
            {
                const uint16_t id = static_cast<uint16_t>((source * kNumDirs + absolute) * 2 + splitting);
                block[kOffLegal + dir * 2 + splitting] = b.isLegal(id) ? 1.0f : 0.0f;
            }
        }
    }

    float* const globals = out.data() + kNumHexes * kHexFeatures;

    globals[kGlobalPly]       = std::min<float>(static_cast<float>(b.ply), kMoveCap) / kMoveCap;
    globals[kGlobalStaleness] = std::min<float>(static_cast<float>(b.staleness), kStalenessLimit) / kStalenessLimit;
    globals[kGlobalRepeats]   = std::min<float>(static_cast<float>(b.repeats - 1), kRepetitionLimit - 1)
                              / (kRepetitionLimit - 1);

    globals[kGlobalMyStacks]       = static_cast<float>(myStacks) * scale;
    globals[kGlobalTheirStacks]    = static_cast<float>(theirStacks) * scale;
    globals[kGlobalMyPrisoners]    = static_cast<float>(myPrisoners) * scale;
    globals[kGlobalTheirPrisoners] = static_cast<float>(theirPrisoners) * scale;
    globals[kGlobalInCheck]        = kernelAttacked(b, b.ownKernelHex(), !me) ? 1.0f : 0.0f;

    assert(std::ranges::all_of(out, [](float v) { return v >= 0.0f && v <= 1.0f; }));
}

} // namespace amoeba
