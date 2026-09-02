#pragma once

#include <arena/arena.h>

#include <mlx/array.h>
#include <mlx/mlx.h>

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace amoeba_bot
{

inline constexpr int boardRadius = 3;
inline constexpr int hexCount = 37;
inline constexpr int directionCount = 6;
inline constexpr int maximumStackHeight = 22;
inline constexpr int piecesPerPlayer = 11;
inline constexpr int maximumMovableStackHeight = 6;
inline constexpr int moveIdCount = hexCount * directionCount * 2;
inline constexpr int legalMoveWordCount = (moveIdCount + 63) / 64;

inline constexpr int encodedStackDepth = maximumMovableStackHeight;
inline constexpr int pieceCodeCount = 5; // empty, my normal, my kernel, their normal, their kernel
inline constexpr int heightBucketCount = maximumMovableStackHeight + 2; // 0..6 exactly, then 7-and-up

// Occurrences of the same position - same stacks, same side to move - that draw.
inline constexpr int repetitionLimit = 3;

// Moves without a landing capture after which the game is adjudicated.
inline constexpr int stalenessLimit = 80;

// Total moves after which the game is adjudicated.
inline constexpr int moveLimit = 250;

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

inline constexpr std::array<Direction, directionCount> directions{{
    {  1,  0 }, // 0 east
    {  1, -1 }, // 1 northeast
    {  0, -1 }, // 2 northwest
    { -1,  0 }, // 3 west
    { -1,  1 }, // 4 southwest
    {  0,  1 }, // 5 southeast
}};

inline constexpr uint8_t directionIdx(Direction dir)
{
    static constexpr auto table = [] -> std::array<uint8_t, 9>
    {
        std::array<uint8_t, 9> table{};
        table.fill(directionCount);
        for (uint8_t index = 0; index < directionCount; ++index)
            table[(directions[index].q + 1) * 3 + directions[index].r + 1] = index;
        return table;
    }();

    assert(dir.q >= -1 && dir.q <= 1 && dir.r >= -1 && dir.r <= 1);
    const uint8_t index = table[(dir.q + 1) * 3 + dir.r + 1];
    assert(index < directionCount);
    return index;
}

constexpr uint8_t oppositeDirection(uint8_t direction)
{
    assert(direction < directionCount);
    return static_cast<uint8_t>((direction + directionCount / 2) % directionCount);
}

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

constexpr Piece colorSpwaped(Piece piece)
{
    static constexpr std::array<Piece, 5> table = {
        Piece::empty,
        Piece::black,
        Piece::blackKernel,
        Piece::white,
        Piece::whiteKernel
    };
    return table[(uint8_t)piece];
}

class Hex
{
  public:
    static constexpr int heightShift = 44;
    static constexpr uint64_t heightMask = 0x1FULL << heightShift;

    constexpr Hex() = default;

    constexpr uint8_t height() const
    {
        return static_cast<uint8_t>((m_bits >> heightShift) & 0x1F);
    }

    constexpr bool isEmpty() const
    {
        return height() == 0;
    }

    // Depth zero is the bottom of the stack. Reading above the top returns empty.
    constexpr Piece pieceAt(uint8_t depth) const
    {
        if (depth >= height())
            return Piece::empty;
        return static_cast<Piece>(((m_bits >> (depth * 2)) & 0x3) + 1);
    }

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

constexpr Coordinate hexCoord(uint8_t hexIdx)
{
    static constexpr auto table = [] -> std::array<Coordinate, hexCount>
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
    assert(hexIdx < hexCount);
    return table[hexIdx];
}

constexpr std::optional<uint8_t> hexIndex(Coordinate coord)
{
    static constexpr auto coordinateIndices = [] -> std::array<std::optional<uint8_t>, 49>
    {
        std::array<std::optional<uint8_t>, 49> table{};
        for (std::optional<uint8_t>& index : table)
            index = std::nullopt;
        for (int8_t hexIdx = 0; hexIdx < hexCount; ++hexIdx)
        {
            const Coordinate coordinate = hexCoord(hexIdx);
            table[(coordinate.q + boardRadius) * 7 + (coordinate.r + boardRadius)] = hexIdx;
        }
        return table;
    }();

    if (coord.q < -boardRadius || coord.q > boardRadius || coord.r < -boardRadius || coord.r > boardRadius)
        return std::nullopt;
    return coordinateIndices[(coord.q + boardRadius) * 7 + (coord.r + boardRadius)];
}

constexpr std::optional<uint8_t> neighboringHex(uint8_t hexIdx, Direction dir)
{
    static constexpr auto table = [] -> std::array<std::array<std::optional<uint8_t>, directionCount>, hexCount>
    {
        std::array<std::array<std::optional<uint8_t>, directionCount>, hexCount> table{};
        for (uint8_t hexIdx = 0; hexIdx < hexCount; ++hexIdx) {
            for (uint8_t dirIdx = 0; dirIdx < directionCount; ++dirIdx) {
                table[hexIdx][dirIdx] = hexIndex(Coordinate{
                    .q = static_cast<int8_t>(hexCoord(hexIdx).q + directions[dirIdx].q),
                    .r = static_cast<int8_t>(hexCoord(hexIdx).r + directions[dirIdx].r)
                });
            }
        }
        return table;
    }();
    assert(hexIdx < hexCount);
    return table[hexIdx][directionIdx(dir)];
}

constexpr std::optional<uint8_t> destinationHex(uint8_t startHexIdx, Direction dir, uint8_t distance)
{
    assert(startHexIdx < hexCount);

    uint8_t currentHex = startHexIdx;
    for (uint8_t step = 0; step < distance; ++step)
    {
        std::optional<uint8_t> neighbor = neighboringHex(currentHex, dir);
        if (neighbor.has_value() == false)
            return std::nullopt;
        currentHex = neighbor.value();
    }
    return currentHex;
}

constexpr uint8_t rotatedHex(uint8_t hexIdx)
{
    static constexpr auto table = [] -> std::array<uint8_t, hexCount>
    {
        std::array<uint8_t, hexCount> table{};
        for (int hexIdx = 0; hexIdx < hexCount; ++hexIdx) {
            table[hexIdx] = hexIndex(Coordinate{
                static_cast<int8_t>(-hexCoord(hexIdx).q),
                static_cast<int8_t>(-hexCoord(hexIdx).r)
            }).value();
        }
        return table;
    }();
    assert(hexIdx < hexCount);
    return table[hexIdx];
}

struct Board
{
    Hex hexes[hexCount]{};
    uint64_t positionHash{};

    uint64_t legalMoveBits[legalMoveWordCount]{};
    uint16_t legalMoveCount{};

    uint8_t whiteKernelIndex{};
    uint8_t blackKernelIndex{};

    uint16_t plyCount{};
    uint16_t stalenessCount{};
    uint8_t repetitionCount{1};

    bool whiteToMove{true};

    constexpr Board() = default;

    // Uses the server's legal moves when my_side == current_turn; otherwise
    // generates moves for current_turn. A null legal_moves also generates them.
    Board(const arena_game_state_t&);

    static Board startingBoard()
    {
        static const char* boardString =
            "-3,1:W;-3,3:W;-2,-1:B;-2,1:W;-2,3:W;-1,-1:B;-1,1:W;-1,2:WK;-1,3:W;0,-3:B;0,-1:B;0,1:W;"
            "0,3:W;1,-3:B;1,-2:BK;1,-1:B;1,1:W;2,-3:B;2,-1:B;2,1:W;3,-3:B;3,-1:B";
        arena_game_state_t state {
            .board = boardString,
            .current_turn = arena_side_t::ARENA_SIDE_WHITE,
            .my_side = arena_side_t::ARENA_SIDE_WHITE,
            .legal_moves = nullptr,
            .legal_moves_count = 0
        };
        return Board(state);
    }

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

    constexpr void forEachLegal(std::invocable<uint16_t> auto&& callback) const
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

    mlx::core::array tensorEncoding() const;
};

struct Move
{
    uint8_t sourceCoord{};
    uint8_t direction{};
    bool splitsStack{};

    constexpr uint16_t id() const
    {
        assert(sourceCoord < hexCount);
        assert(direction < directionCount);
        return static_cast<uint16_t>((sourceCoord * directionCount + direction) * 2 + (splitsStack ? 1 : 0));
    }

    static constexpr Move fromId(uint16_t id)
    {
        assert(id < moveIdCount);
        return Move{
            .sourceCoord = static_cast<uint8_t>(id / (directionCount * 2)),
            .direction = static_cast<uint8_t>((id / 2) % directionCount),
            .splitsStack = (id & 1) != 0
        };
    }

    arena_move_t toArena(uint8_t height) const
    {
        static constexpr std::array<const char*, hexCount> arenaCoordinateStrings{{
            "-3,0", "-3,1", "-3,2", "-3,3", "-2,-1", "-2,0", "-2,1", "-2,2", "-2,3",
            "-1,-2", "-1,-1", "-1,0", "-1,1", "-1,2", "-1,3", "0,-3", "0,-2", "0,-1",
            "0,0", "0,1", "0,2", "0,3", "1,-3", "1,-2", "1,-1", "1,0", "1,1", "1,2",
            "2,-3", "2,-2", "2,-1", "2,0", "2,1", "3,-3", "3,-2", "3,-1", "3,0",
        }};

        assert(sourceCoord < hexCount);
        assert(direction < directionCount);
        assert(height > 0 && height <= maximumMovableStackHeight);
        assert(!splitsStack || height >= 2);
        const std::optional<uint8_t> destination =
            destinationHex(sourceCoord, directions[direction], height);
        assert(destination.has_value());

        return arena_move_t{
            .from_pos = arenaCoordinateStrings[sourceCoord],
            .to_pos = arenaCoordinateStrings[destination.value()],
            .side = nullptr,
            .splitting = splitsStack,
        };
    }

    constexpr auto operator<=>(const Move&) const = default;
};

enum class Outcome : uint8_t
{
    whiteWins,
    blackWins,
    draw,
};

using MoveResult = std::variant<Board, Outcome>;

// history contain `board` as its last element
MoveResult applyMove(const Board& board, Move move, std::span<const uint64_t> positionHistory = {});

} // namespace amoeba_bot
