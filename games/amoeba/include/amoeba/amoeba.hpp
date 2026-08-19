#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace amoeba
{

// ---------------------------------------------------------------------------
// Board geometry: radius-3 hex, axial coordinates, 37 cells.
// Valid iff |q| <= 3, |r| <= 3, |q + r| <= 3.
// ---------------------------------------------------------------------------

inline constexpr int kRadius     = 3;
inline constexpr int kNumHexes   = 37;
inline constexpr int kNumDirs    = 6;
inline constexpr int kMaxHeight  = 22;                        // total pieces on the board
inline constexpr int kPiecesPerSide = 11;                     // 10 standard + 1 kernel, each side
inline constexpr int kMovableMax = 6;                         // stacks taller than this can never move
inline constexpr int kNumMoveIds = kNumHexes * kNumDirs * 2;  // 444, policy size
inline constexpr int kMaskWords  = (kNumMoveIds + 63) / 64;   // 7, one bit per move id

struct Coordinate
{
    int8_t q{};
    int8_t r{};
    constexpr auto operator<=>(const Coordinate&) const = default;
};

struct Direction
{
    int8_t q{};
    int8_t r{};
    constexpr auto operator<=>(const Direction&) const = default;
};

// Ordered so that opposite(d) == (d + 3) % 6, and so the 180-degree perspective flip is just that same mapping.
inline constexpr auto kDirections = [] -> std::array<Direction, kNumDirs> {
    return {{
        { 1,  0},   // 0 east
        { 1, -1},   // 1 northeast
        { 0, -1},   // 2 northwest
        {-1,  0},   // 3 west
        {-1,  1},   // 4 southwest
        { 0,  1},   // 5 southeast
    }};
}();

constexpr uint8_t opposite(uint8_t dir) { return static_cast<uint8_t>((dir + 3) % 6); }

inline constexpr auto kCoordinates = [] -> std::array<Coordinate, kNumHexes> {
    std::array<Coordinate, kNumHexes> coords{};
    int n = 0;
    for (int q = -kRadius; q <= kRadius; ++q)
        for (int r = -kRadius; r <= kRadius; ++r)
            if (q + r >= -kRadius && q + r <= kRadius)
                coords[n++] = Coordinate{static_cast<int8_t>(q), static_cast<int8_t>(r)};
    return coords;
}();

// Reverse lookup: (q + 3) * 7 + (r + 3) -> hex index, or -1 if off board.
inline constexpr auto kIndexOf = [] -> std::array<int8_t, 49> {
    std::array<int8_t, 49> table{};
    for (auto& e : table)
        e = -1;
    for (int i = 0; i < kNumHexes; ++i)
    {
        const auto c = kCoordinates[i];
        table[(c.q + kRadius) * 7 + (c.r + kRadius)] = static_cast<int8_t>(i);
    }
    return table;
}();

constexpr int8_t hexIndex(int q, int r)
{
    if (q < -kRadius || q > kRadius || r < -kRadius || r > kRadius)
        return -1;
    return kIndexOf[(q + kRadius) * 7 + (r + kRadius)];
}

// Step one hex from `from` in direction `dir`. Returns -1 if that leaves the board.
inline constexpr auto kNeighbour = [] -> std::array<std::array<int8_t, kNumDirs>, kNumHexes>
{
    std::array<std::array<int8_t, kNumDirs>, kNumHexes> table{};
    for (int i = 0; i < kNumHexes; ++i)
        for (int d = 0; d < kNumDirs; ++d)
            table[i][d] = hexIndex(kCoordinates[i].q + kDirections[d].q, kCoordinates[i].r + kDirections[d].r);
    return table;
}();

// 180-degree rotation: (q, r) -> (-q, -r). Used by the perspective flip in encode(). Directions flip with opposite().
inline constexpr auto kFlipped = [] -> std::array<uint8_t, kNumHexes>
{
    std::array<uint8_t, kNumHexes> table{};
    for (int i = 0; i < kNumHexes; ++i)
        table[i] = static_cast<uint8_t>(hexIndex(-kCoordinates[i].q, -kCoordinates[i].r));
    return table;
}();

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

// Public-facing codes. Empty exists here but is never stored in a packed stack;
// a slot is empty iff its depth >= height.
enum class Piece : uint8_t
{
    Empty = 0,
    WN    = 1,   // white normal
    WK    = 2,   // white kernel
    BN    = 3,   // black normal
    BK    = 4,   // black kernel
};

constexpr bool isWhite(Piece p)  { return p == Piece::WN || p == Piece::WK; }
constexpr bool isKernel(Piece p) { return p == Piece::WK || p == Piece::BK; }

// Colour/kernel swap, for the perspective flip. Index by uint8_t(Piece).
inline constexpr std::array<Piece, 5> kSwapColour = { Piece::Empty, Piece::BN, Piece::BK, Piece::WN, Piece::WK };

// ---------------------------------------------------------------------------
// Hex: a stack packed into a single 64-bit word.
//
// bits  0..43  : 22 slots x 2 bits, bottom to top. Slot value is Piece minus 1
//                (so 0=WN, 1=WK, 2=BN, 3=BK). Slots at or above `height` are junk.
// bits 44..48  : height, 0..22.
//
// Always go through the accessors; the layout is an implementation detail.
// ---------------------------------------------------------------------------

class Hex
{
public:
    static constexpr int kHeightShift = 44;
    static constexpr uint64_t kHeightMask = 0x1FULL << kHeightShift;

    constexpr Hex() = default;

    constexpr uint8_t height() const
    {
        return static_cast<uint8_t>((bits_ >> kHeightShift) & 0x1F);
    }

    constexpr bool empty() const { return height() == 0; }

    // depth 0 is the bottom of the stack.
    constexpr Piece at(uint8_t depth) const
    {
        if (depth >= height()) return Piece::Empty;
        return static_cast<Piece>(((bits_ >> (depth * 2)) & 0x3) + 1);
    }

    // The piece that controls the stack.
    constexpr Piece top() const
    {
        const uint8_t h = height();
        return h == 0 ? Piece::Empty : at(static_cast<uint8_t>(h - 1));
    }

    constexpr void push(Piece p)
    {
        const uint8_t h = height();
        const uint64_t v = static_cast<uint64_t>(p) - 1;
        bits_ &= ~(0x3ULL << (h * 2));
        bits_ |= v << (h * 2);
        setHeight(static_cast<uint8_t>(h + 1));
    }

    constexpr Piece pop()
    {
        const uint8_t h = height();
        const Piece p = at(static_cast<uint8_t>(h - 1));
        setHeight(static_cast<uint8_t>(h - 1));
        return p;
    }

    constexpr void clear() { bits_ = 0; }

    constexpr auto operator<=>(const Hex&) const = default;

private:
    constexpr void setHeight(uint8_t h)
    {
        bits_ = (bits_ & ~kHeightMask) | (static_cast<uint64_t>(h) << kHeightShift);
    }

    uint64_t bits_{};
};

static_assert(sizeof(Hex) == 8);

// ---------------------------------------------------------------------------
// Move
//
// The distance travelled is always the stack height, so it is not stored.
// A move has a stable id in 0..443, which doubles as the policy head index:
//     id = (from * 6 + dir) * 2 + splitting
// ---------------------------------------------------------------------------

struct Move
{
    uint8_t from{};       // hex index, 0..36
    uint8_t dir{};        // direction index, 0..5
    bool    splitting{};  // false = simple move, true = sow

    constexpr uint16_t id() const
    {
        return static_cast<uint16_t>((from * kNumDirs + dir) * 2 + (splitting ? 1 : 0));
    }

    static constexpr Move fromId(uint16_t id)
    {
        return Move{static_cast<uint8_t>(id / (kNumDirs * 2)),
                    static_cast<uint8_t>((id / 2) % kNumDirs),
                    (id & 1) != 0};
    }

    constexpr auto operator<=>(const Move&) const = default;
};

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

enum class State : uint8_t
{
    Ongoing,
    WhiteWins,
    BlackWins,
    Draw,
};

struct Board
{
    Hex      hexes[kNumHexes]{};       // 296 bytes
    uint64_t hash{};                   // Zobrist, maintained incrementally

    // One bit per move id, filled by apply(). Doubles as the policy mask:
    // feed it straight to the network to set illegal logits to -infinity.
    uint64_t legal[kMaskWords]{};      // 56 bytes
    uint16_t moveCount{};

    uint8_t  whiteKernelHex{};         // cached; the kernel may be buried mid-stack
    uint8_t  blackKernelHex{};
    uint16_t ply{};                    // for the move cap
    uint16_t staleness{};              // moves since the last landing capture
    uint8_t  repeats{1};               // occurrences of this position; only apply() with a history fills it
    bool     whiteToMove{true};
    State    state{State::Ongoing};    // derived data - see note below

    constexpr const Hex& hex(uint8_t i) const { return hexes[i]; }
    constexpr Hex&       hex(uint8_t i)       { return hexes[i]; }

    constexpr bool isLegal(uint16_t id) const
    {
        return ((legal[id >> 6] >> (id & 63)) & 1ULL) != 0;
    }
    constexpr void setLegal(uint16_t id)
    {
        legal[id >> 6] |= 1ULL << (id & 63);
    }
    constexpr void clearLegal()
    {
        for (auto& w : legal) w = 0;
        moveCount = 0;
    }

    // Visit every legal move id. fn is called as fn(uint16_t id).
    //     b.forEachLegal([&](uint16_t id) { Move m = Move::fromId(id); ... });
    template <typename Fn>
    constexpr void forEachLegal(Fn&& fn) const
    {
        for (int w = 0; w < kMaskWords; ++w)
            for (uint64_t bits = legal[w]; bits; bits &= bits - 1)
            {
                const int lsb = __builtin_ctzll(bits);
                fn(static_cast<uint16_t>(w * 64 + lsb));
            }
    }

    constexpr uint8_t ownKernelHex() const
    {
        return whiteToMove ? whiteKernelHex : blackKernelHex;
    }
    constexpr uint8_t enemyKernelHex() const
    {
        return whiteToMove ? blackKernelHex : whiteKernelHex;
    }

    // True if the side to move controls this stack.
    constexpr bool controls(uint8_t i) const
    {
        const Piece t = hexes[i].top();
        return t != Piece::Empty && isWhite(t) == whiteToMove;
    }
};

// `state` is a cache of something computed from the board plus its history.
// It is only trustworthy on a Board produced by apply(). If you build a Board
// by hand - parsing a server string, writing a test - you must populate it
// yourself before reading it.

// ---------------------------------------------------------------------------
// Rule parameters, taken from amoeba-reference.md, which mirrors the server.
// ---------------------------------------------------------------------------

// Occurrences of the same position - same stacks, same side to move - that draw.
inline constexpr int kRepetitionLimit = 3;

// Moves without a landing capture after which the game is adjudicated.
inline constexpr int kStalenessLimit = 80;

// Total moves after which the game is adjudicated.
inline constexpr int kMoveCap = 250;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// Walk `steps` hexes from `start` in direction `dir`. Returns -1 if that leaves
// the board. The board is convex, so if the endpoint is on the board then every
// hex along the way is too - which is why both move types need only this check.
constexpr int8_t ray(uint8_t start, uint8_t dir, uint8_t steps)
{
    int8_t cur = static_cast<int8_t>(start);
    for (uint8_t i = 0; i < steps; ++i)
    {
        cur = kNeighbour[cur][dir];
        if (cur < 0) return -1;
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

// Recompute from scratch. Slow; use it to assert the incremental hash in tests.
uint64_t computeHash(const Board&);

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

// True if `white` controls the stack on `hex` (their piece is on top).
bool controlledBy(const Board&, uint8_t hex, bool white);

// True if the side `attackerIsWhite` has a move that ends with them on top of
// `kernelHex`. Walks the six rays out from the kernel rather than generating
// every move, so it is cheap enough for the legality filter.
bool kernelAttacked(const Board&, uint8_t kernelHex, bool attackerIsWhite);

// Rescan the board for both kernels and update the cache. applyRaw maintains
// the cache itself; this is for boards built by hand.
void refreshKernels(Board&);

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

// Fills b.legal and b.moveCount for the side to move. Excludes moves that lose
// your own kernel immediately or leave it capturable next ply.
void generateLegal(Board&);

// Board mechanics only: move the pieces, update hash / kernel cache / ply, flip
// the side to move. Does not generate moves and does not set `state`.
Board applyRaw(const Board&, Move);

// Full move: applyRaw, then generate the reply moves and settle `state`.
// Pass the hash history - including the hash of `b` itself - to enable
// repetition detection. Without it, repetition is not checked.
Board apply(const Board&, Move, std::span<const uint64_t> history = {});

// ---------------------------------------------------------------------------
// Serialization - the server's format, e.g. "-1,1:WK;0,0:WB;1,-1:BK;1,0:W"
// Stack contents run bottom to top, so "WB" is white with black above it.
// ---------------------------------------------------------------------------

Board fromString(const std::string&, bool whiteToMove = true);

// The opening from section 3 of amoeba-reference.md, White to move. Self-play and
// the encoder test both need it, and two copies of a 22-piece board string would
// eventually disagree.
Board startPosition();
std::string toString(const Board&);

} // namespace amoeba
