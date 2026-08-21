#include "amoeba.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <span>
#include <utility>

namespace amoeba
{

namespace
{

// ---------------------------------------------------------------------------
// Zobrist table, generated at compile time so runs are reproducible.
// Indexed [hex][depth][piece]. Depth matters because stack order is part of
// the position - a split rewrites the depths of everything it touches.
// ---------------------------------------------------------------------------

constexpr uint64_t nextSplitmix64(uint64_t& state)
{
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

struct ZobristTable
{
    uint64_t pieces[hexCount][maximumStackHeight][5]{};
    uint64_t blackToMove{};
};

consteval ZobristTable makeZobrist()
{
    ZobristTable table{};
    uint64_t randomState = 0xA5A5A5A5DEADBEEFULL;

    for (int hex = 0; hex < hexCount; ++hex)
    {
        for (int depth = 0; depth < maximumStackHeight; ++depth)
        {
            // Piece::empty is never stored, so index zero remains unused.
            for (int piece = 1; piece <= 4; ++piece)
                table.pieces[hex][depth][piece] = nextSplitmix64(randomState);
        }
    }

    table.blackToMove = nextSplitmix64(randomState);
    return table;
}

constexpr ZobristTable zobristTable = makeZobrist();

constexpr uint64_t zobristValue(uint8_t hex, uint8_t depth, Piece piece)
{
    return zobristTable.pieces[hex][depth][static_cast<uint8_t>(piece)];
}

} // namespace

// ---------------------------------------------------------------------------

namespace
{

uint64_t computeHash(const Board& board)
{
    uint64_t hash = 0;
    for (uint8_t hex = 0; hex < hexCount; ++hex)
    {
        const Hex& stack = board.hexes[hex];
        for (uint8_t depth = 0; depth < stack.height(); ++depth)
            hash ^= zobristValue(hex, depth, stack.pieceAt(depth));
    }

    if (!board.whiteToMove)
        hash ^= zobristTable.blackToMove;
    return hash;
}

void refreshKernelPositions(Board& board)
{
    for (uint8_t hex = 0; hex < hexCount; ++hex)
    {
        const Hex& stack = board.hexes[hex];
        for (uint8_t depth = 0; depth < stack.height(); ++depth)
        {
            if (stack.pieceAt(depth) == Piece::whiteKernel)
                board.whiteKernelIndex = hex;
            if (stack.pieceAt(depth) == Piece::blackKernel)
                board.blackKernelIndex = hex;
        }
    }
}

// ---------------------------------------------------------------------------

bool isKernelAttacked(const Board& board, uint8_t kernelHex, bool attackerIsWhite)
{
    assert(kernelHex < hexCount);

    // Walk outward from the kernel. A stack `distance` hexes away attacks back
    // along the opposite direction.
    for (uint8_t outwardDirection = 0; outwardDirection < directionCount; ++outwardDirection)
    {
        int8_t attackerHex = static_cast<int8_t>(kernelHex);
        for (uint8_t distance = 1; distance <= maximumMovableStackHeight; ++distance)
        {
            attackerHex = neighboringHexes[attackerHex][outwardDirection];
            if (attackerHex < 0)
                break;

            const Hex& attacker = board.hexes[attackerHex];
            const uint8_t attackerHeight = attacker.height();
            if (attackerHeight == 0)
                continue;
            if (!board.isControlledBy(static_cast<uint8_t>(attackerHex), attackerIsWhite))
                continue;

            const uint8_t attackDirection = oppositeDirection(outwardDirection);

            // A whole-stack move lands on the kernel when height equals distance.
            if (attackerHeight == distance)
                return true;

            // A split drops the piece at depth `distance - 1` onto the kernel. The
            // entire stack still needs a valid endpoint even though the kernel is
            // encountered earlier on its path.
            const bool canSplitOverKernel =
                attackerHeight >= 2 && attackerHeight >= distance &&
                destinationHex(static_cast<uint8_t>(attackerHex), attackDirection, attackerHeight) >= 0;
            if (canSplitOverKernel)
            {
                const Piece landingPiece = attacker.pieceAt(static_cast<uint8_t>(distance - 1));
                if (isWhitePiece(landingPiece) == attackerIsWhite)
                    return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------

Board applyMoveWithoutDeterminingOutcome(const Board& board, Move move)
{
    assert(move.sourceHex < hexCount);
    assert(move.direction < directionCount);
    assert(!board.hexes[move.sourceHex].isEmpty());
    assert(board.controls(move.sourceHex));

    const uint8_t stackHeight = board.hexes[move.sourceHex].height();
    const int8_t destination = destinationHex(move.sourceHex, move.direction, stackHeight);
    assert(destination >= 0);
    assert(!move.splitsStack || stackHeight >= 2);

    Board result = board;
    result.clearLegal();
    Hex& sourceStack = result.hexes[move.sourceHex];

    // Only the last hex of the path counts: intermediate captures made during a
    // sow do not reset the staleness counter.
    const Piece previousTopPiece = board.hexes[destination].topPiece();
    const bool capturedStack = previousTopPiece != Piece::empty && isWhitePiece(previousTopPiece) != board.whiteToMove;

    Piece movedPieces[maximumStackHeight];
    for (uint8_t depth = 0; depth < stackHeight; ++depth)
    {
        movedPieces[depth] = sourceStack.pieceAt(depth);
        result.positionHash ^= zobristValue(move.sourceHex, depth, movedPieces[depth]);
    }
    sourceStack.clear();

    const auto placePiece = [&](uint8_t destinationHex, Piece piece)
    {
        Hex& destinationStack = result.hexes[destinationHex];
        result.positionHash ^= zobristValue(destinationHex, destinationStack.height(), piece);
        destinationStack.pushPiece(piece);
        if (piece == Piece::whiteKernel)
            result.whiteKernelIndex = destinationHex;
        if (piece == Piece::blackKernel)
            result.blackKernelIndex = destinationHex;
    };

    if (!move.splitsStack)
    {
        for (uint8_t depth = 0; depth < stackHeight; ++depth)
            placePiece(static_cast<uint8_t>(destination), movedPieces[depth]);
    }
    else
    {
        int8_t currentHex = static_cast<int8_t>(move.sourceHex);
        for (uint8_t depth = 0; depth < stackHeight; ++depth)
        {
            currentHex = neighboringHexes[currentHex][move.direction];
            placePiece(static_cast<uint8_t>(currentHex), movedPieces[depth]);
        }
    }

    result.whiteToMove = !result.whiteToMove;
    result.positionHash ^= zobristTable.blackToMove;
    ++result.plyCount;
    result.stalenessCount = capturedStack ? 0 : static_cast<uint16_t>(result.stalenessCount + 1);
    result.repetitionCount = 1;
    return result;
}

void generateLegalMoves(Board& board)
{
    board.clearLegal();

    const bool moverIsWhite = board.whiteToMove;

    for (uint8_t sourceHex = 0; sourceHex < hexCount; ++sourceHex)
    {
        if (!board.controls(sourceHex))
            continue;

        const uint8_t stackHeight = board.hexes[sourceHex].height();

        for (uint8_t direction = 0; direction < directionCount; ++direction)
        {
            // Both move types have the same endpoint. A stack taller than six can
            // never have a valid endpoint on this board and is permanently frozen.
            if (destinationHex(sourceHex, direction, stackHeight) < 0)
                continue;

            for (bool splitsStack : {false, true})
            {
                // Splitting a one-piece stack is identical to moving it intact.
                if (splitsStack && stackHeight < 2)
                    continue;

                const Move move{sourceHex, direction, splitsStack};
                const Board nextBoard = applyMoveWithoutDeterminingOutcome(board, move);

                // The move may have carried the mover's kernel somewhere else.
                const uint8_t moverKernel = moverIsWhite ? nextBoard.whiteKernelIndex : nextBoard.blackKernelIndex;

                // A split can expose an opposing piece above the mover's own
                // kernel. That is an immediate loss of control, not merely check.
                if (nextBoard.isControlledBy(moverKernel, !moverIsWhite))
                    continue;

                // The server requires the mover's kernel to be safe even if this
                // move also captures the opposing kernel.
                if (isKernelAttacked(nextBoard, moverKernel, !moverIsWhite))
                    continue;

                board.setLegal(move.id());
                ++board.legalMoveCount;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------

namespace
{

// Both adjudicated endings score the same way: most controlled stacks, then most
// enemy pieces held inside those stacks, then a draw.
Outcome adjudicate(const Board& board)
{
    constexpr int white = 0;
    constexpr int black = 1;
    int controlledStacks[2]{};
    int prisoners[2]{};

    for (uint8_t hex = 0; hex < hexCount; ++hex)
    {
        const Hex& stack = board.hexes[hex];
        const Piece controllingPiece = stack.topPiece();
        if (controllingPiece == Piece::empty)
            continue;

        const int controller = isWhitePiece(controllingPiece) ? white : black;
        ++controlledStacks[controller];
        for (uint8_t depth = 0; depth < stack.height(); ++depth)
        {
            if (isWhitePiece(stack.pieceAt(depth)) != isWhitePiece(controllingPiece))
                ++prisoners[controller];
        }
    }

    if (controlledStacks[white] != controlledStacks[black])
        return controlledStacks[white] > controlledStacks[black] ? Outcome::whiteWins : Outcome::blackWins;
    if (prisoners[white] != prisoners[black])
        return prisoners[white] > prisoners[black] ? Outcome::whiteWins : Outcome::blackWins;
    return Outcome::draw;
}

} // namespace

MoveResult applyMove(const Board& board, Move move, std::span<const uint64_t> positionHistory)
{
    assert(board.isLegal(move.id()));

    const bool moverIsWhite = board.whiteToMove;
    Board result = applyMoveWithoutDeterminingOutcome(board, move);

    // Kernel capture. Check both: a split can lose you your own kernel.
    const uint8_t opponentKernel = moverIsWhite ? result.blackKernelIndex : result.whiteKernelIndex;
    const uint8_t moverKernel = moverIsWhite ? result.whiteKernelIndex : result.blackKernelIndex;
    const bool opponentKernelCaptured = result.isControlledBy(opponentKernel, moverIsWhite);
    const bool moverKernelCaptured = result.isControlledBy(moverKernel, !moverIsWhite);

    if (opponentKernelCaptured || moverKernelCaptured)
    {
        // A legal move cannot leave the mover's kernel captured. Keep both checks
        // because this function's precondition is enforced only in debug builds.
        const bool winnerIsWhite = opponentKernelCaptured ? moverIsWhite : !moverIsWhite;
        return winnerIsWhite ? Outcome::whiteWins : Outcome::blackWins;
    }

    generateLegalMoves(result);

    // No legal moves for the side to move: they lose.
    if (result.legalMoveCount == 0)
    {
        return result.whiteToMove ? Outcome::blackWins : Outcome::whiteWins;
    }

    if (!positionHistory.empty())
    {
        int occurrenceCount = 1; // Include the newly created position.
        for (uint64_t previousHash : positionHistory)
        {
            if (previousHash == result.positionHash)
                ++occurrenceCount;
        }

        result.repetitionCount = static_cast<uint8_t>(occurrenceCount);
        if (occurrenceCount >= repetitionLimit)
        {
            return Outcome::draw;
        }
    }

    if (result.stalenessCount >= stalenessLimit || result.plyCount >= moveLimit)
        return adjudicate(result);

    return result;
}

// ---------------------------------------------------------------------------

Board createStartingBoard()
{
    return parseBoard("-3,1:W;-3,3:W;-2,-1:B;-2,1:W;-2,3:W;-1,-1:B;-1,1:W;-1,2:WK;-1,3:W;0,-3:B;0,-1:B;0,1:W;"
                      "0,3:W;1,-3:B;1,-2:BK;1,-1:B;1,1:W;2,-3:B;2,-1:B;2,1:W;3,-3:B;3,-1:B");
}

Board parseBoard(const std::string& serializedBoard, bool whiteToMove)
{
    Board board{};
    board.whiteToMove = whiteToMove;

    size_t entryStart = 0;
    while (entryStart < serializedBoard.size())
    {
        size_t entryEnd = serializedBoard.find(';', entryStart);
        if (entryEnd == std::string::npos)
            entryEnd = serializedBoard.size();

        const std::string entry = serializedBoard.substr(entryStart, entryEnd - entryStart);
        const size_t colon = entry.find(':');
        const size_t comma = entry.find(',');
        if (colon != std::string::npos && comma != std::string::npos && comma < colon)
        {
            const int q = std::atoi(entry.substr(0, comma).c_str());
            const int r = std::atoi(entry.substr(comma + 1, colon - comma - 1).c_str());
            const int8_t hex = hexIndex(q, r);
            if (hex >= 0)
            {
                // Stack letters run bottom to top: W, B, WK, BK.
                const std::string stackText = entry.substr(colon + 1);
                for (size_t character = 0; character < stackText.size(); ++character)
                {
                    const bool isKernel = character + 1 < stackText.size() && stackText[character + 1] == 'K';
                    Piece piece = Piece::empty;
                    if (stackText[character] == 'W')
                        piece = isKernel ? Piece::whiteKernel : Piece::white;
                    if (stackText[character] == 'B')
                        piece = isKernel ? Piece::blackKernel : Piece::black;

                    if (piece != Piece::empty)
                        board.hexes[hex].pushPiece(piece);
                    if (isKernel)
                        ++character;
                }
            }
        }
        entryStart = entryEnd + 1;
    }

    refreshKernelPositions(board);
    board.positionHash = computeHash(board);
    generateLegalMoves(board);
    return board;
}

// ===========================================================================
// Board -> network input
// ===========================================================================

namespace
{

// Swapping colours for Black lands exactly on the Piece enum's own numbering, so
// the perspective code is just the swapped enum value: 0 empty, 1 my normal,
// 2 my kernel, 3 their normal, 4 their kernel.
constexpr int perspectiveCode(Piece piece, bool whiteToMove)
{
    return static_cast<int>(whiteToMove ? piece : swappedPieceColors[static_cast<uint8_t>(piece)]);
}

} // namespace

void encodeBoard(const Board& board, std::span<float, encodedBoardSize> output)
{
    assert(board.repetitionCount >= 1);
    std::ranges::fill(output, 0.0f);

    const bool moverIsWhite = board.whiteToMove;
    const float countScale = 1.0f / piecesPerPlayer;

    int ownStackCount = 0;
    int opponentStackCount = 0;
    int ownPrisonerCount = 0;
    int opponentPrisonerCount = 0;

    for (int hexIdx = 0; hexIdx < hexCount; ++hexIdx)
    {
        const std::span<float, featuresPerHex> hexFeatures{output.data() + (hexIdx * featuresPerHex), featuresPerHex};

        const uint8_t absoluteHexIdx = moverIsWhite ? static_cast<uint8_t>(hexIdx) : rotatedHexes[hexIdx];
        const Hex& stack = board.hexes[absoluteHexIdx];
        const uint8_t stackHeight = stack.height();

        // pieceAt() reports empty past the top, so short stacks pad themselves.
        for (uint8_t depth = 0; depth < encodedStackDepth; ++depth)
        {
            const int pieceCode = perspectiveCode(stack.pieceAt(depth), moverIsWhite);
            hexFeatures[stackSlotsOffset + depth * pieceCodeCount + pieceCode] = 1.0f;
        }

        hexFeatures[stackHeightOffset + std::min<int>(stackHeight, maximumMovableStackHeight + 1)] = 1.0f;
        hexFeatures[topPieceOffset + perspectiveCode(stack.topPiece(), moverIsWhite)] = 1.0f;

        int ownPieces = 0;
        int opponentPieces = 0;
        for (uint8_t depth = 0; depth < stackHeight; ++depth)
        {
            const int pieceCode = perspectiveCode(stack.pieceAt(depth), moverIsWhite);
            if (pieceCode <= 2)
                ++ownPieces;
            else
                ++opponentPieces;

            if (pieceCode == 2)
                hexFeatures[kernelPresenceOffset] = 1.0f;
            if (pieceCode == 4)
                hexFeatures[kernelPresenceOffset + 1] = 1.0f;
        }
        hexFeatures[pieceCountsOffset] = static_cast<float>(ownPieces) * countScale;
        hexFeatures[pieceCountsOffset + 1] = static_cast<float>(opponentPieces) * countScale;

        if (stackHeight > 0)
        {
            if (perspectiveCode(stack.topPiece(), moverIsWhite) <= 2) {
                ++ownStackCount;
                ownPrisonerCount += opponentPieces;
            }
            else {
                ++opponentStackCount;
                opponentPrisonerCount += ownPieces;
            }
        }

        for (uint8_t relativeDirection = 0; relativeDirection < directionCount; ++relativeDirection)
        {
            const uint8_t absoluteDirection = moverIsWhite ? relativeDirection : oppositeDirection(relativeDirection);
            for (int splitsStack = 0; splitsStack < 2; ++splitsStack)
            {
                const Move move{absoluteHexIdx, absoluteDirection, splitsStack != 0};
                hexFeatures[legalMovesOffset + relativeDirection * 2 + splitsStack] =
                    board.isLegal(move.id()) ? 1.0f : 0.0f;
            }
        }
    }

    const std::span<float, globalFeatureCount> globalFeatures{output.data() + hexCount * featuresPerHex, globalFeatureCount};

    globalFeatures[globalMoveNumberIndex] = std::min<float>(static_cast<float>(board.plyCount), moveLimit) / moveLimit;
    globalFeatures[globalStalenessIndex] = std::min<float>(static_cast<float>(board.stalenessCount), stalenessLimit) / stalenessLimit;
    globalFeatures[globalRepetitionIndex] = std::min<float>(static_cast<float>(board.repetitionCount - 1), repetitionLimit - 1) / (repetitionLimit - 1);

    globalFeatures[globalOwnStackCountIndex] = static_cast<float>(ownStackCount) * countScale;
    globalFeatures[globalOpponentStackCountIndex] = static_cast<float>(opponentStackCount) * countScale;
    globalFeatures[globalOwnPrisonerCountIndex] = static_cast<float>(ownPrisonerCount) * countScale;
    globalFeatures[globalOpponentPrisonerCountIndex] = static_cast<float>(opponentPrisonerCount) * countScale;
    globalFeatures[globalInCheckIndex] = isKernelAttacked(board, board.ownKernelIndex(), !moverIsWhite) ? 1.0f : 0.0f;

    assert(std::ranges::all_of(output, [](float value) { return value >= 0.0f && value <= 1.0f; }));
}

} // namespace amoeba
