#include "amoeba/amoeba.hpp"

#include <cstdio>
#include <cstdlib>

namespace amoeba
{

namespace
{

// ---------------------------------------------------------------------------
// Zobrist table, generated at compile time so runs are reproducible.
// Indexed [hex][depth][piece]. Depth matters because stack order is part of
// the position - a split rewrites the depths of everything it touches.
// ---------------------------------------------------------------------------

constexpr uint64_t splitmix64(uint64_t& s)
{
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct ZobristTable
{
    uint64_t piece[kNumHexes][kMaxHeight][5]{};
    uint64_t blackToMove{};
};

consteval ZobristTable makeZobrist()
{
    ZobristTable t{};
    uint64_t s = 0xA5A5A5A5DEADBEEFULL;
    for (int h = 0; h < kNumHexes; ++h)
        for (int d = 0; d < kMaxHeight; ++d)
            for (int p = 1; p <= 4; ++p)  // index 0 (Empty) is never used
                t.piece[h][d][p] = splitmix64(s);
    t.blackToMove = splitmix64(s);
    return t;
}

constexpr ZobristTable kZobrist = makeZobrist();

constexpr uint64_t zob(uint8_t hex, uint8_t depth, Piece p)
{
    return kZobrist.piece[hex][depth][static_cast<uint8_t>(p)];
}

} // namespace

// ---------------------------------------------------------------------------

uint64_t computeHash(const Board& b)
{
    uint64_t h = 0;
    for (uint8_t i = 0; i < kNumHexes; ++i)
    {
        const Hex& hx = b.hexes[i];
        for (uint8_t d = 0; d < hx.height(); ++d) h ^= zob(i, d, hx.at(d));
    }
    if (!b.whiteToMove) h ^= kZobrist.blackToMove;
    return h;
}

bool controlledBy(const Board& b, uint8_t hex, bool white)
{
    const Piece t = b.hexes[hex].top();
    return t != Piece::Empty && isWhite(t) == white;
}

void refreshKernels(Board& b)
{
    for (uint8_t i = 0; i < kNumHexes; ++i)
    {
        const Hex& hx = b.hexes[i];
        for (uint8_t d = 0; d < hx.height(); ++d)
        {
            if (hx.at(d) == Piece::WK) b.whiteKernelHex = i;
            if (hx.at(d) == Piece::BK) b.blackKernelHex = i;
        }
    }
}

// ---------------------------------------------------------------------------

bool kernelAttacked(const Board& b, uint8_t kernelHex, bool attackerIsWhite)
{
    // Walk outward from the kernel. A stack sitting k hexes away in direction d
    // attacks the kernel by moving back along opposite(d).
    for (uint8_t d = 0; d < kNumDirs; ++d)
    {
        int8_t c = static_cast<int8_t>(kernelHex);
        for (uint8_t k = 1; k <= kMovableMax; ++k)
        {
            c = kNeighbour[c][d];
            if (c < 0) break;

            const Hex& hx = b.hexes[c];
            const uint8_t h = hx.height();
            if (h == 0) continue;
            if (!controlledBy(b, static_cast<uint8_t>(c), attackerIsWhite)) continue;

            const uint8_t back = opposite(d);

            // Simple move: travels exactly h, so it lands on the kernel iff h == k.
            // The stack arrives intact, so its top piece - the attacker's - ends
            // up on top.
            if (h == k) return true;

            // Split: drops pieces bottom-first along the path, so the piece that
            // lands on the kernel is the one at depth k-1. It must be the
            // attacker's, and the full path must stay on the board.
            if (h >= 2 && h >= k && ray(static_cast<uint8_t>(c), back, h) >= 0)
            {
                const Piece lands = hx.at(static_cast<uint8_t>(k - 1));
                if (lands != Piece::Empty && isWhite(lands) == attackerIsWhite) return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------

Board applyRaw(const Board& b, Move m)
{
    Board n = b;
    n.clearLegal();
    n.state = State::Ongoing;

    Hex& src = n.hexes[m.from];
    const uint8_t h = src.height();

    // Only the last hex of the path counts: intermediate captures made during a
    // sow do not reset the staleness counter.
    const uint8_t dest    = static_cast<uint8_t>(ray(m.from, m.dir, h));
    const Piece   landedOn = b.hexes[dest].top();
    const bool    seized   = landedOn != Piece::Empty && isWhite(landedOn) != b.whiteToMove;

    Piece buf[kMaxHeight];
    for (uint8_t i = 0; i < h; ++i)
    {
        buf[i] = src.at(i);
        n.hash ^= zob(m.from, i, buf[i]);   // lift the whole stack out of the hash
    }
    src.clear();

    auto place = [&](uint8_t to, Piece p) {
        Hex& dst = n.hexes[to];
        n.hash ^= zob(to, dst.height(), p);  // new depth, so hash in at that depth
        dst.push(p);
        if (p == Piece::WK) n.whiteKernelHex = to;
        if (p == Piece::BK) n.blackKernelHex = to;
    };

    if (!m.splitting)
    {
        for (uint8_t i = 0; i < h; ++i) place(dest, buf[i]);  // order preserved
    }
    else
    {
        int8_t cur = static_cast<int8_t>(m.from);
        for (uint8_t i = 0; i < h; ++i)
        {
            cur = kNeighbour[cur][m.dir];
            place(static_cast<uint8_t>(cur), buf[i]);  // bottom piece lands first
        }
    }

    n.whiteToMove = !n.whiteToMove;
    n.hash ^= kZobrist.blackToMove;
    ++n.ply;
    n.staleness = seized ? 0 : static_cast<uint16_t>(n.staleness + 1);
    n.repeats   = 1;
    return n;
}

// ---------------------------------------------------------------------------

void generateLegal(Board& b)
{
    b.clearLegal();

    const bool me = b.whiteToMove;

    for (uint8_t from = 0; from < kNumHexes; ++from)
    {
        if (!b.controls(from)) continue;
        const uint8_t h = b.hexes[from].height();

        for (uint8_t d = 0; d < kNumDirs; ++d)
        {
            // Both move types span exactly h hexes, so both need this endpoint.
            // Stacks taller than kMovableMax can never satisfy it - they are
            // frozen for the rest of the game.
            if (ray(from, d, h) < 0) continue;

            for (int sp = 0; sp < 2; ++sp)
            {
                // For h == 1 a split is the same move as a simple move.
                if (sp == 1 && h < 2) continue;

                const Move mv{from, d, sp == 1};
                const Board next = applyRaw(b, mv);

                // Read the kernel hexes from `next`, not `b` - the move may have
                // carried a kernel somewhere else.
                const uint8_t myKernel    = me ? next.whiteKernelHex : next.blackKernelHex;
                const uint8_t theirKernel = me ? next.blackKernelHex : next.whiteKernelHex;

                // Splitting scatters the stack bottom-first, and the lower
                // pieces may be the opponent's - so a move can hand them your
                // own kernel. Check that before anything else.
                if (controlledBy(next, myKernel, !me)) continue;

                // Capturing their kernel ends the game, so our own safety is moot.
                if (controlledBy(next, theirKernel, me))
                {
                    b.setLegal(mv.id());
                    ++b.moveCount;
                    continue;
                }

                if (kernelAttacked(next, myKernel, !me)) continue;

                b.setLegal(mv.id());
                ++b.moveCount;
            }
        }
    }
}

// ---------------------------------------------------------------------------

namespace
{

// Both adjudicated endings score the same way: most controlled stacks, then most
// enemy pieces held inside those stacks, then a draw.
State adjudicate(const Board& b)
{
    int stacks[2]{};     // [0] white, [1] black
    int prisoners[2]{};

    for (uint8_t i = 0; i < kNumHexes; ++i)
    {
        const Hex&  hx = b.hexes[i];
        const Piece t  = hx.top();
        if (t == Piece::Empty) continue;

        const int controller = isWhite(t) ? 0 : 1;
        ++stacks[controller];
        for (uint8_t d = 0; d < hx.height(); ++d)
            if (isWhite(hx.at(d)) != isWhite(t)) ++prisoners[controller];
    }

    if (stacks[0] != stacks[1])
        return stacks[0] > stacks[1] ? State::WhiteWins : State::BlackWins;
    if (prisoners[0] != prisoners[1])
        return prisoners[0] > prisoners[1] ? State::WhiteWins : State::BlackWins;
    return State::Draw;
}

} // namespace

Board apply(const Board& b, Move m, std::span<const uint64_t> history)
{
    const bool mover = b.whiteToMove;
    Board n = applyRaw(b, m);

    // Kernel capture. Check both: a split can lose you your own kernel.
    const bool theirsTaken = controlledBy(n, mover ? n.blackKernelHex : n.whiteKernelHex, mover);
    const bool oursTaken   = controlledBy(n, mover ? n.whiteKernelHex : n.blackKernelHex, !mover);

    if (theirsTaken || oursTaken)
    {
        // If somehow both, the mover wins - they completed their move. This is a
        // corner case worth confirming against the server.
        const bool winnerIsWhite = theirsTaken ? mover : !mover;
        n.state = winnerIsWhite ? State::WhiteWins : State::BlackWins;
        return n;
    }

    generateLegal(n);

    // No legal moves for the side to move: they lose.
    if (n.moveCount == 0)
    {
        n.state = n.whiteToMove ? State::BlackWins : State::WhiteWins;
        return n;
    }

    // Repetition. Needs the path, not just the position.
    if (!history.empty())
    {
        int seen = 1;  // n itself
        for (uint64_t past : history)
            if (past == n.hash) ++seen;
        n.repeats = static_cast<uint8_t>(seen);
        if (seen >= kRepetitionLimit)
        {
            n.state = State::Draw;
            return n;
        }
    }

    if (n.staleness >= kStalenessLimit || n.ply >= kMoveCap)
        n.state = adjudicate(n);

    return n;
}

// ---------------------------------------------------------------------------

Board fromString(const std::string& s, bool whiteToMove)
{
    Board b{};
    b.whiteToMove = whiteToMove;

    size_t i = 0;
    while (i < s.size())
    {
        size_t end = s.find(';', i);
        if (end == std::string::npos) end = s.size();

        const std::string entry = s.substr(i, end - i);
        const size_t colon = entry.find(':');
        const size_t comma = entry.find(',');
        if (colon != std::string::npos && comma != std::string::npos && comma < colon)
        {
            const int q = std::atoi(entry.substr(0, comma).c_str());
            const int r = std::atoi(entry.substr(comma + 1, colon - comma - 1).c_str());
            const int8_t idx = hexIndex(q, r);
            if (idx >= 0)
            {
                // Stack letters run bottom to top: W, B, WK, BK.
                const std::string stack = entry.substr(colon + 1);
                for (size_t c = 0; c < stack.size(); ++c)
                {
                    const bool kernel = (c + 1 < stack.size() && stack[c + 1] == 'K');
                    Piece p = Piece::Empty;
                    if (stack[c] == 'W') p = kernel ? Piece::WK : Piece::WN;
                    if (stack[c] == 'B') p = kernel ? Piece::BK : Piece::BN;
                    if (p != Piece::Empty) b.hexes[idx].push(p);
                    if (kernel) ++c;
                }
            }
        }
        i = end + 1;
    }

    refreshKernels(b);
    b.hash = computeHash(b);
    generateLegal(b);
    return b;
}

std::string toString(const Board& b)
{
    std::string out;
    for (uint8_t i = 0; i < kNumHexes; ++i)
    {
        const Hex& hx = b.hexes[i];
        if (hx.empty()) continue;
        if (!out.empty()) out += ';';

        char coord[16];
        std::snprintf(coord, sizeof coord, "%d,%d", kCoordinates[i].q, kCoordinates[i].r);
        out += coord;
        out += ':';

        for (uint8_t d = 0; d < hx.height(); ++d)
        {
            switch (hx.at(d))
            {
                case Piece::WN: out += "W";  break;
                case Piece::WK: out += "WK"; break;
                case Piece::BN: out += "B";  break;
                case Piece::BK: out += "BK"; break;
                default: break;
            }
        }
    }
    return out;
}

} // namespace amoeba
