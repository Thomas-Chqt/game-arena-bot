# Amoeba bot

## Goal

A competitive bot for **Amoeba** on Game Arena (`https://staging-game-arena.irvine.jp/docs#amoeba`,
operated by Irvine Systems). Approach: **AlphaZero** — neural network + MCTS, trained by self-play.

Target platform: **macOS, C++23, MLX** for the network. The rules engine is plain C++ with no MLX
dependency, and that separation must be preserved — it's what lets the engine be tested without
touching the ML side.

Assume the reader is a strong C++ programmer but new to machine learning. Explain ML concepts in
plain terms without assuming vocabulary; don't explain C++.

## Layout

```
CMakeLists.txt
games/amoeba/
  include/amoeba/amoeba.hpp   the engine
  src/amoeba.cpp
  test/random_test.cpp        live server parity harness
  test/CMakeLists.txt
amoeba_bot_1/                 STALE — does not compile against the current API
```

## Current state

The rules engine works. Benchmarked at **~296k plies/sec, ~3.7k full random games/sec**,
single-threaded, `-O2`, unoptimised.

### What's implemented

- `Hex` — a stack packed into one `uint64_t` (2 bits × 22 slots + 5-bit height). `sizeof(Hex) == 8`.
- `Board` — 368 bytes. Hexes, Zobrist hash, a 444-bit legality bitmask, cached kernel hexes,
  ply counter, side to move, and a `state` field.
- Compile-time tables: `kCoordinates`, `kIndexOf`, `kNeighbour`, `kFlipped`, `kDirections`.
- `applyRaw` — board mechanics only. `apply` — full move including terminal detection.
- `generateLegal` — pseudo-legal generation plus the kernel-safety filter.
- `kernelAttacked` — ray-walk attack detection, verified against brute force.
- `fromString` / `toString` — the server's board format.

### Previously verified, but the tests are gone

These claims are secondhand: the test suite that established them is **not in the repo**.

- Split semantics (bottom piece lands nearest, origin emptied, kernel cache follows it)
- Simple moves stacking onto occupants with order preserved
- `kernelAttacked` matching brute force over 3000 random positions
- 200 random games asserting hash consistency, piece conservation (always 22), and kernel cache
  on every ply
- Serialization round-trip

### `amoeba_bot_1` is stale

It targets a previous engine API — `game::Game`, `game::Position`, `game::Side`,
`Move{from, to, splitting}`, `parse_coord` / `encode_coord`, `board_string()`, `terminal_reason()`.
None of that exists now. A full `cmake --build` fails on it; build targets individually until it is
ported, deleted, or dropped from the root `CMakeLists.txt`.

## Game rules

Sources: the Game Arena docs, and the designer's original page
(`https://www.nakajim.net/masg/Amoeba.html` — Masahiro Nakajima, 2010, published by nestorgames).

- Radius-3 hex board, axial coordinates, 37 cells. Valid iff `|q| <= 3, |r| <= 3, |q+r| <= 3`.
- 11 pieces per side: 1 kernel + 10 normal. **22 pieces total, always** — pieces are never added to
  or removed from the board. Good invariant for tests.
- The top piece of a stack controls it; only the controller may move it.
- **Move distance always equals stack height.** A 2-stack moves exactly 2 hexes, jumping over
  anything between. Distance is therefore not stored in `Move`.
- **Simple move:** the whole stack travels, landing intact on top of any occupant.
- **Split (sow):** pieces drop one at a time **from the bottom**, along the path. A height-3 stack
  splitting east puts bottom→+1, middle→+2, top→+3, leaving the origin empty. Both move types span
  the same distance, so both need the same endpoint check.
- Splits are only distinct from simple moves at height ≥ 2.
- **Stacks of height ≥ 7 are frozen forever.** The longest straight line on the board is 6 hexes,
  and stacks only ever grow. This is why the encoder caps depth at 6 (see below).
- **Win by controlling the stack containing the opponent's kernel.** The kernel can be buried
  mid-stack; it doesn't need to be on top.
- No legal moves on your turn = you lose.
- **A split can lose you your own kernel** — the bottom pieces may be the opponent's, and one
  landing on your kernel's stack hands it to them. `generateLegal` filters these. Easy to miss.
- Game Arena adds a chess-like rule the original lacks: moves that leave your own kernel capturable
  are illegal. This is the expensive part of move generation.

## Unverified assumptions — resolve these first

None of these are confirmed. Training on a wrong one means training on a different game than the one
we compete in. The harness below exists to settle them.

- `kRepetitionLimit = 3` — the original rules say three occurrences. Game Arena only says "draw by
  repeated position", which might mean two. Marked in `amoeba.hpp`.
- `kMoveCap = 400` — invented. Game Arena mentions a move cap but not its value. Self-play needs
  some cap or early random games never terminate. Marked in `amoeba.hpp`.
- **The sow `to_pos` convention.** The SDK expresses moves as `(from_pos, to_pos)` strings. The
  harness assumes `to_pos` for a sow is the far endpoint at distance = height, the same hex a simple
  move would reach. If it is the first landing hex instead, the harness's step-count check fires
  immediately.
- **`splitting` is per *piece*, not per destination** (`arena_piece_moves_t.splitting`). A stack that
  can both move whole and sow must therefore appear as two entries with the same `pos`. If it appears
  only once, the server is not enumerating those as separate moves and the set sizes will differ
  structurally from ours.
- What happens if a single move simultaneously captures the opponent's kernel and loses your own.
  Currently the mover wins.
- **The real starting position.** The docs render it as a diagram that could not be read reliably.
  The harness prints it at game start.

## Next task: run the verification harness

**Do this before anything else.** A rules bug will silently poison training and be very hard to
diagnose later.

`games/amoeba/test/random_test.cpp` is written and builds clean, but **has never been run against the
server**.

```
cmake --build build.nosync --target amoeba_random_test
ARENA_DOMAIN=staging-game-arena.irvine.jp BOT1_ID=… BOT1_KEY=… ROOM_ID=… \
  ./build.nosync/games/amoeba/test/Debug/amoeba_random_test --seed 1
```

Run it repeatedly — a `splitting` bug only surfaces once a sow comes up, so a broken run can look
fine for many moves.

### Scope — keep it this way

One file, one binary, live server only, no test framework. It plays random legal moves and stops on
the first disagreement. A layered design (pure translation layer, recorded transcripts replayed
offline, a GTest suite) was proposed and rejected: the question being answered is "are my rules
right", and only the server can answer it. Offline tests would only re-assert our own reading of the
rules. Do not add layers, scripts, or scaffolding around it.

### What it checks each ply

- **Move-set parity.** Converts every server move to a local move id and diffs against `Board::legal`
  bit for bit, printing the symmetric difference as `local only:` / `server only:`. This is the point
  of the whole thing.
- **Geometry.** The `(from_pos, to_pos)` → `(from, dir)` conversion requires a straight ray *and*
  that the step count equals the stack height, so it tests the distance rule rather than assuming it.
- **Parse integrity.** Asserts 22 pieces after `fromString`, which silently drops entries it cannot
  read — a misread board format shows up as pieces vanishing rather than as an error.
- **`apply` correctness.** Between our turns the opponent plays exactly once, so it searches our legal
  moves for the one reproducing the server's new board. This is the only check on where split pieces
  actually land.
- **Terminal agreement.** Winner comparison. If the server ends a game the engine calls ongoing, it
  prints the ply and the repeat count of the final position — the evidence for `kMoveCap` and
  `kRepetitionLimit`.

One deliberate normalisation: a height-1 stack sows and moves identically and `generateLegal` only
emits the whole-stack form, so a server `splitting` flag on such a piece is folded to non-splitting
and counted. The count prints in the pass line.

## Game Arena SDK

Game Arena ships a **C SDK, `arena` 0.6**, installed at `~/.local` (`include/arena/arena.h`,
`libarena.a`). Every `CMakeLists.txt` finds it with
`find_package(arena 0.6 CONFIG REQUIRED HINTS $ENV{HOME}/.local)`. The Python `game_ai_arena_sdk` is
**not** installed and is not used. GoogleTest also sits at `~/.local` if it is ever wanted.

- `ARENA_DOMAIN` overrides the default `game-arena.irvine.jp`; set it to
  `staging-game-arena.irvine.jp`.
- Entry points: `arena_start` (matchmade), `arena_start_continuous`, `arena_start_practice` (takes a
  room id, or reads `ROOM_ID` when passed `NULL`).
- Credentials come from `BOT1_ID` and `BOT1_KEY`.
- Moves are `(from_pos, to_pos, splitting)` **strings**. Omitting `splitting` produces
  `INVALID_MOVE`, but only when a sow comes up.
- The SDK owns the strings in `arena_piece_moves_t` for the duration of the callback. Write those
  pointers straight back into `arena_move_t` rather than re-spelling the coordinates.
- `on_game_start` reports moves for *our* side even when the opponent is to move.

## After the harness passes

1. **MCTS** — Zobrist hashing is already in place for the transposition table.
2. **Encoder** (`Board` → float array). Design already settled:
   - Per-hex tokens, not fixed depth planes. Shape `[batch, 37, 6]` of piece codes, bottom to top,
     zero-padded.
   - Depth cap of **6**, justified by the frozen-stack argument above: order only matters for stacks
     that can still move. For deeper stacks add three scalars — total height, top piece, and whether
     a kernel is buried inside.
   - Plus per-hex extras: height, controller, kernel-present flag, kernel depth.
   - **Perspective flip:** always present the board from the mover's point of view. Swap colours
     (`kSwapColour`) *and* negate coordinates (`kFlipped`); directions map via `opposite()`. The start
     position is 180°-symmetric, so colour swap alone leaves it geometrically upside down. Keep
     `Board` itself in absolute colours — the flip belongs only in `encode()`.
3. **Network** — embedding table (5 rows), flatten depth, linear to 128, then ~4-6 self-attention
   blocks over the 37 tokens. Policy head: 12 logits per token (6 directions × 2 move types) → 444,
   masked with `Board::legal`. Value head: mean-pool → MLP → tanh. ~1-2M params.
4. **Training loop.**

## Things to get right in the ML code

- **Value sign.** The value head means "good for the mover", not "good for white". When labelling
  training examples with the game outcome, negate it for every position where the mover lost. Getting
  this backwards produces a bot that reliably plays badly, and everything still runs.
- **Policy under flip.** Coordinates are negated, so hex indices move and directions remap. The policy
  is in flipped space; map back before sending a move to the server.
- **Symmetry augmentation.** A hex board has 12 rotations/reflections, and colour swap doubles it —
  24× more training data for free. Highest-value thing available on a small compute budget. Every
  symmetry permutes both hexes and directions, so policy targets must be permuted identically.
- **Batch network evaluations.** Run 128+ self-play games concurrently and evaluate their search
  leaves in one call. Single-position inference wastes the GPU entirely.
- **MLX is lazy** — nothing computes until `mx::eval()`. Easy to build a huge graph by accident, and
  it makes naive timing measurements meaningless.
- MLX's C++ API is less travelled than the Python one; optimizers and layers may need writing by hand.
  If that turns into a time sink, doing the training loop in Python + MLX and keeping C++ for the
  engine is a reasonable fallback.

## Build

CMake + Ninja, build tree in `build.nosync/`, multi-config (`Debug` / `Release` / `RelWithDebInfo`).

```
cmake -S . -B build.nosync
cmake --build build.nosync --target amoeba_random_test
```

Build individual targets — a full `cmake --build` still fails on the stale `amoeba_bot_1`.

`forEachLegal` uses `__builtin_ctzll` (GCC/Clang). `std::countr_zero` from `<bit>` is the portable
equivalent if that ever matters.

`Board::moveCount` is maintained separately from the bitmask and can drift out of sync — consider
`std::popcount` over the seven words instead.
