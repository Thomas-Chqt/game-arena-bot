// Checks that encode() shows a position to both players identically: the same
// board with the colours swapped and rotated 180 degrees must produce a
// bit-for-bit identical input array. A flip bug is otherwise invisible -
// training still runs, it just quietly learns the game twice and plays worse.
//
// Also asserts the structural properties encode() promises: every value in
// [0, 1], and every one-hot block summing to exactly 1.
//
// And it checks policyToAbsolute() against the legality bits encode() wrote,
// because that permutation is the one whose failure is otherwise silent.

#include <amoeba/amoeba.hpp>
#include <amoeba/encode.hpp>

#include <array>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace amoeba;

namespace {

constexpr int kGames = 200;
constexpr uint64_t kSeed = 20260819;

using Features = std::array<float, kEncodedSize>;

std::string describe(int index)
{
    if (index >= kNumHexes * kHexFeatures)
        return std::format("global {}", index - kNumHexes * kHexFeatures);
    return std::format("hex {} feature {}", index / kHexFeatures, index % kHexFeatures);
}

Board mirrored(const Board& b)
{
    Board m{};
    for (uint8_t i = 0; i < kNumHexes; ++i)
    {
        const Hex& hx = b.hexes[i];
        for (uint8_t d = 0; d < hx.height(); ++d)
            m.hexes[kFlipped[i]].push(kSwapColour[static_cast<uint8_t>(hx.at(d))]);
    }
    m.whiteToMove = !b.whiteToMove;
    m.ply         = b.ply;
    m.staleness   = b.staleness;
    m.repeats     = b.repeats;
    refreshKernels(m);
    m.hash = computeHash(m);
    generateLegal(m);
    return m;
}

bool checkStructure(const Features& f, const Board& b)
{
    for (int i = 0; i < kEncodedSize; ++i)
        if (!(f[i] >= 0.0f && f[i] <= 1.0f))
        {
            std::println(stderr, "[encode] {} is {}, outside [0, 1], on {}", describe(i), f[i], toString(b));
            return false;
        }

    for (int token = 0; token < kNumHexes; ++token)
    {
        const float* block = f.data() + token * kHexFeatures;
        auto sum = [block](int offset, int width) {
            return std::accumulate(block + offset, block + offset + width, 0.0f);
        };

        for (int d = 0; d < kSlotDepth; ++d)
            if (sum(kOffSlots + d * kPieceCodes, kPieceCodes) != 1.0f)
            {
                std::println(stderr, "[encode] hex {} slot {} is not one-hot on {}", token, d, toString(b));
                return false;
            }
        if (sum(kOffHeight, kHeightBuckets) != 1.0f)
        {
            std::println(stderr, "[encode] hex {} height is not one-hot on {}", token, toString(b));
            return false;
        }
        if (sum(kOffTop, kPieceCodes) != 1.0f)
        {
            std::println(stderr, "[encode] hex {} top piece is not one-hot on {}", token, toString(b));
            return false;
        }
    }
    return true;
}

// Ties policyToAbsolute() to the legality bits encode() actually wrote. Slot i of
// the network's policy describes the same move as legality feature i, so the two
// must agree on every position - which is what makes a permutation error loud
// instead of silent.
bool checkPolicyMap(const Features& f, const Board& b)
{
    const std::span<const uint16_t, kNumMoveIds> toAbsolute = policyToAbsolute(b.whiteToMove);

    for (int i = 0; i < kNumMoveIds; ++i)
    {
        const int token     = i / (kNumDirs * 2);
        const int dir       = (i / 2) % kNumDirs;
        const int splitting = i % 2;

        const float feature = f[token * kHexFeatures + kOffLegal + dir * 2 + splitting];
        const float actual  = b.isLegal(toAbsolute[i]) ? 1.0f : 0.0f;

        if (feature != actual)
        {
            std::println(stderr, "[encode] policy slot {} (hex {} dir {}{}) maps to move {}: {} vs {}, on {}",
                         i, token, dir, splitting ? " sow" : "", toAbsolute[i], feature, actual, toString(b));
            return false;
        }
    }
    return true;
}

bool check(const Board& b)
{
    Features direct{}, flipped{};
    encode(b, direct);
    encode(mirrored(b), flipped);

    if (!checkStructure(direct, b)) return false;
    if (!checkPolicyMap(direct, b)) return false;

    for (int i = 0; i < kEncodedSize; ++i)
        if (direct[i] != flipped[i])
        {
            std::println(stderr, "[encode] {} differs under the flip: {} vs {}, on {}",
                         describe(i), direct[i], flipped[i], toString(b));
            return false;
        }
    return true;
}

// The engine records the outcome but not the reason, so infer it. Kernel capture
// is unreachable once every legal move must leave your own kernel safe.
const char* reason(const Board& b)
{
    if (b.staleness >= kStalenessLimit || b.ply >= kMoveCap)
        return "adjudicated";
    if (b.state == State::Draw)
        return "repetition";
    return "no-move";
}

} // namespace

int main()
{
    std::mt19937_64 random{kSeed};

    int positions = 0;
    int noMove = 0, repetition = 0, adjudicated = 0;
    int longest = 0;

    for (int game = 0; game < kGames; ++game)
    {
        Board board = startPosition();
        std::vector<uint64_t> history{board.hash};

        while (board.state == State::Ongoing)
        {
            if (!check(board))
                return 1;
            ++positions;

            std::vector<uint16_t> ids;
            board.forEachLegal([&](uint16_t id) { ids.push_back(id); });

            const size_t pick = std::uniform_int_distribution<size_t>{0, ids.size() - 1}(random);
            board = apply(board, Move::fromId(ids[pick]), history);
            history.push_back(board.hash);
        }

        const std::string why = reason(board);
        if (why == "no-move")
            ++noMove;
        else if (why == "repetition")
            ++repetition;
        else
            ++adjudicated;
        longest = std::max<int>(longest, board.ply);
    }

    std::println("[encode] PASSED: {} positions, {} features each", positions, kEncodedSize);
    std::println("[encode] {} random games: {} no-move, {} repetition, {} adjudicated, longest {} moves", kGames, noMove, repetition, adjudicated, longest);
    return 0;
}
