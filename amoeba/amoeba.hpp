#ifndef AMOEBA_HPP
#define AMOEBA_HPP

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <span>
#include <string>

namespace amoeba
{

// ---------------------------------------------------------------------------
// Board geometry: radius-3 hex, axial coordinates, 37 cells.
// Valid iff |q| <= 3, |r| <= 3, |q + r| <= 3.
// ---------------------------------------------------------------------------

inline constexpr int boardRadius = 3;
inline constexpr int hexCount = 37;
inline constexpr int directionCount = 6;
inline constexpr int maximumStackHeight = 22;
inline constexpr int piecesPerPlayer = 11;
inline constexpr int maximumMovableStackHeight = 6;
inline constexpr int moveIdCount = hexCount * directionCount * 2;
inline constexpr int legalMoveWordCount = (moveIdCount + 63) / 64;

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

// Opposite directions are three positions apart. The same mapping is used when
// rotating a position into Black's perspective.
inline constexpr std::array<Direction, directionCount> directions = { {
    {  1,  0 }, // 0 east
    {  1, -1 }, // 1 northeast
    {  0, -1 }, // 2 northwest
    { -1,  0 }, // 3 west
    { -1,  1 }, // 4 southwest
    {  0,  1 }, // 5 southeast
}};

constexpr uint8_t oppositeDirection(uint8_t direction)
{
    assert(direction < directionCount);
    return static_cast<uint8_t>((direction + directionCount / 2) % directionCount);
}

inline constexpr auto coordinates = [] -> std::array<Coordinate, hexCount>
{
    std::array<Coordinate, hexCount> result{};
    int index = 0;
    for (int q = -boardRadius; q <= boardRadius; ++q)
    {
        for (int r = -boardRadius; r <= boardRadius; ++r)
        {
            if (q + r >= -boardRadius && q + r <= boardRadius)
                result[index++] = Coordinate{static_cast<int8_t>(q), static_cast<int8_t>(r)};
        }
    }
    return result;
}();

// Reverse lookup: (q + 3) * 7 + (r + 3) -> hex index, or -1 if off board.
inline constexpr auto coordinateIndices = [] -> std::array<int8_t, 49>
{
    std::array<int8_t, 49> table{};
    for (int8_t& index : table)
        index = -1;

    for (int hex = 0; hex < hexCount; ++hex)
    {
        const Coordinate coordinate = coordinates[hex];
        table[(coordinate.q + boardRadius) * 7 + (coordinate.r + boardRadius)] = static_cast<int8_t>(hex);
    }
    return table;
}();

constexpr int8_t hexIndex(int q, int r)
{
    if (q < -boardRadius || q > boardRadius || r < -boardRadius || r > boardRadius)
        return -1;
    return coordinateIndices[(q + boardRadius) * 7 + (r + boardRadius)];
}

// Step one hex from a source in a direction. A negative entry leaves the board.
inline constexpr auto neighboringHexes = [] -> std::array<std::array<int8_t, directionCount>, hexCount>
{
    std::array<std::array<int8_t, directionCount>, hexCount> table{};
    for (int hex = 0; hex < hexCount; ++hex) {
        for (int direction = 0; direction < directionCount; ++direction) {
            table[hex][direction] = hexIndex(coordinates[hex].q + directions[direction].q, coordinates[hex].r + directions[direction].r);
        }
    }
    return table;
}();

// 180-degree rotation: (q, r) -> (-q, -r).
inline constexpr auto rotatedHexes = [] -> std::array<uint8_t, hexCount>
{
    std::array<uint8_t, hexCount> table{};
    for (int hex = 0; hex < hexCount; ++hex)
        table[hex] = static_cast<uint8_t>(hexIndex(-coordinates[hex].q, -coordinates[hex].r));
    return table;
}();

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

// Public-facing codes. Empty exists here but is never stored in a packed stack;
// a slot is empty iff its depth >= height.
enum class Piece : uint8_t
{
    empty       = 0,
    white       = 1,
    whiteKernel = 2,
    black       = 3,
    blackKernel = 4,
};

constexpr bool isWhitePiece(Piece piece)
{
    return piece == Piece::white || piece == Piece::whiteKernel;
}

// Indexed by Piece's underlying value.
inline constexpr std::array<Piece, 5> swappedPieceColors = {
    Piece::empty,
    Piece::black,
    Piece::blackKernel,
    Piece::white,
    Piece::whiteKernel
};

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
    static constexpr int heightShift = 44;
    static constexpr uint64_t heightMask = 0x1FULL << heightShift;

    constexpr Hex() = default;

    constexpr uint8_t height() const { return static_cast<uint8_t>((m_bits >> heightShift) & 0x1F); }

    constexpr bool isEmpty() const { return height() == 0; }

    // Depth zero is the bottom of the stack. Reading above the top returns empty,
    // which lets the encoder pad short stacks without a separate branch.
    constexpr Piece pieceAt(uint8_t depth) const
    {
        if (depth >= height())
            return Piece::empty;
        return static_cast<Piece>(((m_bits >> (depth * 2)) & 0x3) + 1);
    }

    // The piece that controls the stack.
    constexpr Piece topPiece() const
    {
        const uint8_t stackHeight = height();
        return stackHeight == 0 ? Piece::empty : pieceAt(static_cast<uint8_t>(stackHeight - 1));
    }

    constexpr void pushPiece(Piece piece)
    {
        assert(piece != Piece::empty);
        assert(height() < maximumStackHeight);

        const uint8_t stackHeight = height();
        const uint64_t encodedPiece = static_cast<uint64_t>(piece) - 1;
        m_bits &= ~(0x3ULL << (stackHeight * 2));
        m_bits |= encodedPiece << (stackHeight * 2);
        setHeight(static_cast<uint8_t>(stackHeight + 1));
    }

    constexpr void clear() { m_bits = 0; }

    constexpr auto operator<=>(const Hex&) const = default;

private:
    constexpr void setHeight(uint8_t height)
    {
        assert(height <= maximumStackHeight);
        m_bits = (m_bits & ~heightMask) | (static_cast<uint64_t>(height) << heightShift);
    }

    uint64_t m_bits{};
};

static_assert(sizeof(Hex) == 8);

// ---------------------------------------------------------------------------
// Move
//
// The distance travelled is always the stack height, so it is not stored.
// A move has a stable id in 0..443, which doubles as the policy head index:
//     id = (sourceHex * 6 + direction) * 2 + splitsStack
// ---------------------------------------------------------------------------

struct Move
{
    uint8_t sourceHex{};
    uint8_t direction{};
    bool splitsStack{};

    constexpr uint16_t id() const
    {
        assert(sourceHex < hexCount);
        assert(direction < directionCount);
        return static_cast<uint16_t>((sourceHex * directionCount + direction) * 2 + (splitsStack ? 1 : 0));
    }

    static constexpr Move fromId(uint16_t id)
    {
        assert(id < moveIdCount);
        return Move{
            .sourceHex = static_cast<uint8_t>(id / (directionCount * 2)),
            .direction = static_cast<uint8_t>((id / 2) % directionCount),
            .splitsStack = (id & 1) != 0
        };
    }

    constexpr auto operator<=>(const Move&) const = default;
};

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

enum class State : uint8_t
{
    ongoing,
    whiteWins,
    blackWins,
    draw,
};

struct Board
{
    Hex hexes[hexCount]{};
    uint64_t positionHash{};

    // One bit per move id, filled by applyMove(). Doubles as the policy mask:
    // feed it straight to the network to set illegal logits to -infinity.
    uint64_t legalMoveBits[legalMoveWordCount]{};
    uint16_t legalMoveCount{};

    uint8_t whiteKernelIndex{};
    uint8_t blackKernelIndex{};

    uint16_t plyCount{};
    uint16_t stalenessCount{};
    uint8_t repetitionCount{1};

    bool whiteToMove{true};
    State state{State::ongoing};

    constexpr bool isLegal(uint16_t id) const
    {
        assert(id < moveIdCount);
        return ((legalMoveBits[id >> 6] >> (id & 63)) & 1ULL) != 0;
    }

    constexpr void setLegal(uint16_t id)
    {
        assert(id < moveIdCount);
        legalMoveBits[id >> 6] |= 1ULL << (id & 63);
    }

    constexpr void clearLegal()
    {
        for (uint64_t& word : legalMoveBits)
            word = 0;
        legalMoveCount = 0;
    }

    // Visit every legal move id. fn is called as fn(uint16_t id).
    //     b.forEachLegal([&](uint16_t id) { Move m = Move::fromId(id); ... });
    template <typename Callback> constexpr void forEachLegal(Callback&& callback) const
    {
        for (int word = 0; word < legalMoveWordCount; ++word)
        {
            for (uint64_t remaining = legalMoveBits[word]; remaining != 0; remaining &= remaining - 1)
            {
                const int leastSignificantBit = std::countr_zero(remaining);
                callback(static_cast<uint16_t>(word * 64 + leastSignificantBit));
            }
        }
    }

    constexpr uint8_t ownKernelIndex() const { return whiteToMove ? whiteKernelIndex : blackKernelIndex; }
    constexpr uint8_t opponentKernelIndex() const { return whiteToMove ? blackKernelIndex : whiteKernelIndex; }

    // True if the side to move controls this stack.
    constexpr bool controls(uint8_t hexIndex) const
    {
        assert(hexIndex < hexCount);
        const Piece topPiece = hexes[hexIndex].topPiece();
        return topPiece != Piece::empty && isWhitePiece(topPiece) == whiteToMove;
    }
};

// `state` is a cache of something computed from the board plus its history.
// It is only trustworthy on a Board produced by applyMove(). If you build a Board
// by hand - parsing a server string, writing a test - you must populate it
// yourself before reading it.

// ---------------------------------------------------------------------------
// Rule parameters, taken from amoeba-reference.md, which mirrors the server.
// ---------------------------------------------------------------------------

// Occurrences of the same position - same stacks, same side to move - that draw.
inline constexpr int repetitionLimit = 3;

// Moves without a landing capture after which the game is adjudicated.
inline constexpr int stalenessLimit = 80;

// Total moves after which the game is adjudicated.
inline constexpr int moveLimit = 250;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// Walk `steps` hexes from `start` in direction `dir`. Returns -1 if that leaves
// the board. The board is convex, so if the endpoint is on the board then every
// hex along the way is too - which is why both move types need only this check.
constexpr int8_t destinationHex(uint8_t startHex, uint8_t direction, uint8_t distance)
{
    assert(startHex < hexCount);
    assert(direction < directionCount);

    int8_t currentHex = static_cast<int8_t>(startHex);
    for (uint8_t step = 0; step < distance; ++step)
    {
        currentHex = neighboringHexes[currentHex][direction];
        if (currentHex < 0)
            return -1;
    }
    return currentHex;
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

// Recompute from scratch. Slow; use it to assert the incremental hash in tests.
uint64_t computeHash(const Board& board);

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

// True if `white` controls the stack on `hex` (their piece is on top).
bool isControlledBy(const Board& board, uint8_t hex, bool white);

// True if the side `attackerIsWhite` has a move that ends with them on top of
// `kernelHex`. Walks the six rays out from the kernel rather than generating
// every move, so it is cheap enough for the legality filter.
bool isKernelAttacked(const Board& board, uint8_t kernelHex, bool attackerIsWhite);

// Rescan the board for both kernels and update the cache.
// applyMoveWithoutUpdatingState maintains the cache itself; this is for boards
// built by hand.
void refreshKernelPositions(Board& board);

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

// Fills b.legalMoveBits and b.legalMoveCount for the side to move. Excludes
// moves that lose your own kernel immediately or leave it capturable next ply.
void generateLegalMoves(Board& board);

// Board mechanics only: move the pieces, update hash / kernel cache / ply, flip
// the side to move. Does not generate moves and does not set `state`.
// Preconditions: the source stack belongs to the side to move, the endpoint is
// on the board, and a split moves at least two pieces.
Board applyMoveWithoutUpdatingState(const Board& board, Move move);

// Full move: applyMoveWithoutUpdatingState, then generate the reply moves and
// settle `state`.
// Pass the hash history - including the hash of `b` itself - to enable
// repetition detection. Without it, repetition is not checked.
// Precondition: `move` is legal on `board`.
Board applyMove(const Board& board, Move move, std::span<const uint64_t> positionHistory = {});

// ---------------------------------------------------------------------------
// Serialization - the server's format, e.g. "-1,1:WK;0,0:WB;1,-1:BK;1,0:W"
// Stack contents run bottom to top, so "WB" is white with black above it.
// ---------------------------------------------------------------------------

Board parseBoard(const std::string& serializedBoard, bool whiteToMove = true);

// The opening from section 3 of amoeba-reference.md, White to move. Self-play and
// the encoder test both need it, and two copies of a 22-piece board string would
// eventually disagree.
Board createStartingBoard();
std::string serializeBoard(const Board& board);

// ===========================================================================
// Board -> network input
// ===========================================================================

// ---------------------------------------------------------------------------
// Board -> network input.
//
// hexCount blocks of featuresPerHex floats, one block per hex, followed by
// globalFeatureCount floats describing the position as a whole. Every value is in
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
// what the mover sees at position t, which is absolute hex rotatedHexes[t] when
// Black is to move, and direction d in a block means absolute
// oppositeDirection(d). A policy over this input needs mapping back the same way before it names a move
// the server will accept.
// ---------------------------------------------------------------------------

// Stacks taller than maximumMovableStackHeight can never move again, so their
// internal order can never matter again either. Six slots covers every stack
// still in play.
inline constexpr int encodedStackDepth = maximumMovableStackHeight;
inline constexpr int pieceCodeCount = 5; // empty, my normal, my kernel, their normal, their kernel
inline constexpr int heightBucketCount = maximumMovableStackHeight + 2; // 0..6 exactly, then 7-and-up

inline constexpr int featuresPerHex =
    encodedStackDepth * pieceCodeCount // stack contents, bottom first: slot d is the piece a sow lands d + 1 hexes away
    + heightBucketCount                // height one-hot - height selects a rule, it is not a magnitude
    + pieceCodeCount                   // top piece, the only thing above encodedStackDepth that still matters
    + 2                                // my kernel / their kernel buried anywhere in this stack
    + 2                                // my pieces / their pieces in this stack
    + directionCount * 2;              // legality per direction, move and sow, straight out of Board::legal

inline constexpr int globalFeatureCount = 8;
inline constexpr int encodedBoardSize = hexCount * featuresPerHex + globalFeatureCount;

static_assert(featuresPerHex == 59);

// Offsets within one hex block.
inline constexpr int stackSlotsOffset = 0;
inline constexpr int stackHeightOffset = stackSlotsOffset + encodedStackDepth * pieceCodeCount;
inline constexpr int topPieceOffset = stackHeightOffset + heightBucketCount;
inline constexpr int kernelPresenceOffset = topPieceOffset + pieceCodeCount;
inline constexpr int pieceCountsOffset = kernelPresenceOffset + 2;
inline constexpr int legalMovesOffset = pieceCountsOffset + 2;

// The globals, in order.
inline constexpr int globalMoveNumberIndex = 0;
inline constexpr int globalStalenessIndex = 1;
inline constexpr int globalRepetitionIndex = 2;
inline constexpr int globalOwnStackCountIndex = 3;
inline constexpr int globalOpponentStackCountIndex = 4;
inline constexpr int globalOwnPrisonerCountIndex = 5;
inline constexpr int globalOpponentPrisonerCountIndex = 6;
inline constexpr int globalInCheckIndex = 7;

// Requires b.legalMoveBits and the kernel cache to be populated - applyMove() and
// parseBoard() both do it.
void encodeBoard(const Board& board, std::span<float, encodedBoardSize> output);

// ---------------------------------------------------------------------------
// Policy space -> absolute move ids
//
// encodeBoard() writes the slot for (token, dir) from the absolute move
// (rotatedHexes[token], oppositeDirection(dir)) when Black is to move, so a
// policy coming back from the network is indexed in that same flipped space. It has to be
// permuted before any of it names a move, and before a search's visit counts
// become a training target.
//
// This is the one mapping whose failure is silent: a wrong permutation still
// gives a valid distribution over legal moves, the loss still falls, and the bot
// simply plays as though the board were rotated.
//
// rotatedHexes and oppositeDirection() are both involutions, so the permutation
// is its own inverse and the same table maps a target back the other way.
// ---------------------------------------------------------------------------

inline constexpr auto identityPolicyMapping = [] -> std::array<uint16_t, moveIdCount>
{
    std::array<uint16_t, moveIdCount> table{};
    for (int moveId = 0; moveId < moveIdCount; ++moveId)
        table[moveId] = static_cast<uint16_t>(moveId);
    return table;
}();

inline constexpr auto rotatedPolicyMapping = [] -> std::array<uint16_t, moveIdCount>
{
    std::array<uint16_t, moveIdCount> table{};
    for (int token = 0; token < hexCount; ++token)
    {
        for (uint8_t direction = 0; direction < directionCount; ++direction)
        {
            for (int splitsStack = 0; splitsStack < 2; ++splitsStack)
            {
                const uint16_t moveId = (token * directionCount + direction) * 2 + splitsStack;
                const uint16_t rotatedMoveId = static_cast<uint16_t>((rotatedHexes[token] * directionCount + oppositeDirection(direction)) * 2 + splitsStack);
                table[moveId] = rotatedMoveId;
            }
        }
    }
    return table;
}();

static_assert(
    []
    {
        for (int moveId = 0; moveId < moveIdCount; ++moveId)
        {
            if (rotatedPolicyMapping[rotatedPolicyMapping[moveId]] != moveId)
                return false;
        }
        return true;
    }(),
    "the policy flip must be its own inverse");

// Identity for White, because encodeBoard() does not flip then.
constexpr std::span<const uint16_t, moveIdCount> policyIndicesToMoveIds(bool whiteToMove)
{
    return whiteToMove ? std::span<const uint16_t, moveIdCount>{identityPolicyMapping}
                       : std::span<const uint16_t, moveIdCount>{rotatedPolicyMapping};
}

} // namespace amoeba

#endif // AMOEBA_HPP
