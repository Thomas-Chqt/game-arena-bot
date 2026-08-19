# Amoeba bot

## Goal

A competitive bot for **Amoeba** on Game Arena (`https://staging-game-arena.irvine.jp/docs#amoeba`,
operated by Irvine Systems). Approach: **AlphaZero** — neural network + MCTS, trained by self-play.

Target platform: **macOS, C++23, MLX** for the network. The rules engine is plain C++ with no MLX
dependency, and that separation must be preserved — it's what lets the engine be tested without
touching the ML side.

Assume the reader is a strong C++ programmer but new to machine learning. Explain ML concepts in
plain terms without assuming vocabulary; don't explain C++.

## Game rules

`games/amoeba/amoeba-reference.md`

## Layout

```
games/amoeba/
  amoeba-reference.md           the rules, mirrored from the server engine
  include/amoeba/amoeba.hpp     rules engine
  include/amoeba/encode.hpp     Board -> network input
  src/amoeba.cpp
  src/encode.cpp
  test/encode_test.cpp          offline
  test/random_test.cpp          live server parity harness
amoeba_bot_1/
  mcts.hpp / mcts.cpp           PUCT search + rollout evaluator
  checkpoints.hpp / .cpp        the model directory and the `best` link
  network.hpp                   the seam with the network branch - declarations only
  play.cpp                      arena client        -> amoeba_bot_1
  train.cpp                     self-play + training -> amoeba_bot_1_train
```

**Two programs, on purpose.** `amoeba_bot_1` plays and never trains;
`amoeba_bot_1_train` trains and never talks to the server — it does not even link the arena SDK.
They share a checkpoint directory and nothing else, so both can run at once on the same machine and
either can be killed and restarted without the other noticing. A match costs almost nothing next to
self-play (one game is ~4 minutes of thinking, most of it waiting on the opponent's clock), so
continuous ranked play takes a couple of percent off training throughput. Coordinating them in one
process would buy the ability to pause training during a turn, which is not worth the coupling.

## Current state

The rules engine works. ~296k plies/sec, ~3.7k full random games/sec, single-threaded, `-O2`.

- `Hex` — a stack packed into one `uint64_t`. `Board` — 376 bytes, with a 444-bit legality bitmask
  that doubles as the policy mask.
- Compile-time tables: `kCoordinates`, `kIndexOf`, `kNeighbour`, `kFlipped`, `kDirections`.
- `generateLegal`, `kernelAttacked`, `applyRaw` / `apply`, `fromString` / `toString`.
- Terminal conditions agree with `amoeba-reference.md`: repetition on the 3rd occurrence, staleness
  80, move cap 250, adjudication by controlled stacks then by enemy pieces held inside them.
- `encode()`, its test, and `kFlippedMove` — the 444-entry permutation that maps a policy out of
  the mover's frame. `static_assert`ed to be its own inverse.
- `openingBoard()`, the position self-play starts every game from.
- PUCT search with a deadline, Dirichlet root noise and temperature sampling; a rollout evaluator.
- `amoeba_bot_1` (continuous arena play) and `amoeba_bot_1_train` (self-play, training, gating),
  both complete except for the network. `network.hpp` declares the eight symbols the model branch
  owes them; there is no `network.cpp`, so **both targets compile and fail to link**, which is the
  intended state until the model lands.

**`random_test` has still never been run against the server.** `amoeba-reference.md` is a document,
and the reference *Python* implementation is what was differentially tested — our C++ engine has not
been. 200 random games do land within noise of the reference's own outcome distribution (§7.3), which
is reassuring but is not the same thing. Run the harness before trusting a long training run.

## Encoder

### The principle

A network cannot compute. It has no loops or branches and cannot call `generateLegal`; it multiplies
and adds, and what it learns is "inputs shaped like this tend to give that answer". So **anything
computable exactly and cheaply in C++ goes into the input array** rather than being left for the net
to approximate from examples. Every decision below follows from that one rule.

### Layout

2191 floats: 37 blocks of 59, one per hex, then 8 globals. Every value is in `[0, 1]`, asserted in
debug builds — the first layer weighs all inputs against each other, so a feature with a much larger
range would drown out the rest.

```
per hex, 59 floats:
  [ 0:30]  6 stack slots x 5-way one-hot, bottom first
  [30:38]  height one-hot: 0,1,2,3,4,5,6,7+
  [38:43]  top piece one-hot
  [43:45]  my kernel / their kernel buried anywhere in this stack
  [45:47]  my piece count / their piece count, each /11
  [47:59]  6 directions x (can move, can sow), straight out of Board::legal

globals, 8 floats:
  ply/250   staleness/80   (repeats-1)/2
  my controlled stacks /11    theirs /11
  my prisoners /11            theirs /11      (enemy pieces inside stacks I control)
  am I in check
```

### Why it is shaped that way

- **Categories are one-hot** (N floats, one of them 1) rather than a single integer. `piece = 3`
  would be treated as a magnitude: a black piece as three times a white one, a kernel as the midpoint
  of two normals. None of that means anything.
- **Slots run bottom first.** A sow drops `stack[0]` one hex away, `stack[1]` two hexes away, and the
  attack rule reads `stack[d-1]` at distance `d`. So slot `k` has one fixed meaning on every hex:
  *the piece that lands `k+1` away if this stack sows*. Top-aligned, slot 0 would mean something
  different per height and the net would have to learn the offset first.
- **The depth cap of 6 is exact, not a heuristic.** A stack moves exactly its height, the longest
  straight run is 6 steps, and heights only ever grow — so height ≥ 7 is frozen forever and its
  internal order can never matter again.
- **Top piece is kept separately** because it is the one thing about a frozen stack that still
  matters: whoever tops it counts for the adjudication tiebreak, and someone can still land on it.
- **Height is one-hot, not a scalar.** Height selects a rule ("moves exactly 3", "cannot move"), it
  is not a quantity — height 4 is not "a bit more than" height 3, it has a different destination set.
- **The legality bits are the highest-value block.** Computing them needs the whole check filter, the
  most expensive code in the engine; the net cannot run it; and 293/300 random games end by running
  out of legal moves, so legality *is* the game. They are already sitting in `Board::legal`.
- **The globals are invisible in the piece layout.** Two identical boards at staleness 5 and 79 are
  completely different positions. This is why `Board` carries `staleness` and `repeats` at all.
- **11 is the divisor** for anything counting one side's material — each side has exactly 11 pieces.

### Perspective flip

Everything is written from the mover's point of view: for Black, swap colours (`kSwapColour`) *and*
rotate 180° (`kFlipped`), directions via `opposite()`. Both, because colour swap alone leaves the
board geometrically upside down. The start position is 180°-symmetric so the transform is exact, and
the net learns the game once instead of once per colour.

Convenient accident: `Piece` is numbered `Empty=0, WN=1, WK=2, BN=3, BK=4`, so after the colour swap
the enum value *is* the one-hot index — 1/2 are always "mine", 3/4 always "theirs".

`Board` stays in absolute colours. The flip lives only in `encode()`.

**The policy comes back in flipped space.** Map it back before naming a move: hex via `kFlipped`,
direction via `opposite()`. That is a fixed 444-entry permutation table, built once — needed both for
the move you send and for training targets.

`encode_test.cpp` exists for exactly this: it asserts `encode(position) == encode(mirrored position)`
bit for bit. Verified to have teeth — dropping the coordinate rotation, the colour swap, or the
direction remap each fails on the first position.

## Search (MCTS)

`amoeba_bot_1/mcts.hpp` / `mcts.cpp`. PUCT, the AlphaZero variant — no rollouts *inside* the search;
a new position's value comes from a `bot::Evaluator`.

### The pieces

- `Evaluator` — `evaluate(span<const Board*>, span<Evaluation>)`, where `Evaluation` is a 444-float
  policy plus a value in `[-1, 1]`. The span form is deliberate even though the search asks one at a
  time: a single-position MLX forward pass wastes the device, and widening the interface later would
  touch every implementation.
- `RolloutEvaluator` — uniform priors, value from one uniformly random playout to the end of the
  game. Knows nothing about Amoeba, but it makes the search playable before the network exists and it
  is the baseline the network has to beat.
- `Search::run(root, history) -> VisitCounts` — 444 visit counts. `bestMove()` is the argmax;
  normalised, the same array is the policy target for training.

### Invariants that are easy to break

- **Every stored value is from the point of view of the side to move at the node that owns it**, so
  the backup flips sign once per level. Inverting this yields a bot that actively seeks its own worst
  lines — it loses to random play, which is at least a loud symptom.
- **The statistics live on the edge, not the child node.** PUCT needs a prior and a visit count for
  moves that have never been played, i.e. before the resulting position exists. Measured branching is
  52 at the opening and 27 on average, so an 800-simulation search holds ~801 nodes and ~21,600
  edges; making every edge a node would mean 27× the `apply()` calls and 8.5 MB instead of 432 KB.
- **`m_path` is the repetition history**, maintained as one vector: the caller's game history as a
  fixed prefix, one hash pushed per node stepped into during the descent, truncated back to the
  prefix at the top of every simulation. At the `apply()` call its last element is the board being
  applied — exactly what that function's contract asks for.
- **No transposition table.** A node's terminal `state` is path-dependent (draw by repetition), and
  with one path per node it stays valid forever. Sharing statistics across paths would make a cached
  `Draw` verdict a lie on the other path.

### The board in the node

`Node` is 392 bytes because it holds a whole `Board`. The alternative is replaying `applyRaw` down
the descent path every simulation. Storing won because `apply()` — the expensive call, it runs
`generateLegal` — is then paid exactly once per node, and a node needs path-dependent state anyway.
The cost is cache: the descent reads only `visits`, `edgeCount` and `board.hash` out of those 392
bytes. Splitting hot (~24 B) from cold would fix it, and is not worth doing while the evaluator is
~99% of the time.

### Self-play knobs, now present

- **`Config::deadline`** — the search stops at the simulation count *or* the clock, whichever comes
  first, and always runs at least one simulation so the visit counts can never be all zero. Match
  play sets the count high and lets the deadline bind: the trainer may be holding the GPU, and a
  late turn is a forfeit while a short search is only a weaker move. Self-play does the reverse, so
  its data does not depend on how busy the machine was.
- **`Config::rootNoiseWeight`** — Dirichlet noise mixed into the root priors, 0.25 in self-play and
  0 everywhere else. Without it every self-play game from one network is the same game.
- **`sampleMove(counts, temperature, rng)`** — picks in proportion to `visits^(1/T)`. Self-play and
  the gate use it for the first 15 plies, then `bestMove`. It deliberately does not accept `T = 0`;
  call `bestMove` for the greedy phase.

### Still deliberately absent

- **Tree reuse between moves** — the subtree under the played move stays valid, and re-rooting is
  roughly 20-40% more effective simulations for free. `m_nodes` / `m_edges` are members so
  allocations are reused, but the tree is cleared on every `run()`.
- **Leaf batching with virtual loss** — see Trainer below. Cheaper to build before the network than
  to retrofit after, and it is what unlocks concurrent self-play games.

### Verification gap

A `strength_check` binary existed: rollout MCTS against uniformly random legal play, alternating
colours, seed `20260819`. It scored **97.5%** (39-0-1) at 200 simulations and **100%** (20-0-0) at
800. Both numbers mattered — the second showed strength rising with simulation count, which is the
property a broken tree does not have. It was deleted on request, so nothing offline checks the search
any more.

## Arena client

`amoeba_bot_1/play.cpp`. Four callbacks, same shape as the reference's random bot. With no
`ROOM_ID` it runs `arena_start_continuous`, which queues game after game and only returns on error.

- **Translation.** The engine names a move `(from, dir, splitting)`; the server names it
  `(pos, destination, splitting)`. `collectServerMoves()` turns the server's list into engine move
  ids, the search picks one, and `chooseMove()` finds it back in that list to recover the server's
  own strings.
- **The bot keeps its own board.** `arena_game_state_t` carries a position and nothing else — no ply
  count, no history — so `syncToServer()` works out which legal move the opponent played and applies
  it locally. Re-parsing the server's board each turn instead would silently reset `ply` and
  `staleness` to zero every ply, and would leave no hash history for the search to detect repetition
  with.
- **Two fallbacks, because forfeiting is worse than wrong bookkeeping.** If no legal move reaches the
  server's position, adopt that position and carry `ply` forward — that is a rules bug, and
  `amoeba_random_test` should have caught it first. If the search picks a move the server did not
  offer, play the server's first move.
- **The model is loaded once per game**, in `on_game_start`, from the `best` link. Between games is
  the only safe moment: one tree scoring its positions with two different networks is incoherent and
  would not show up in any log.
- **Every callback is wrapped in `guarded()`.** A C++ exception unwinding through the SDK's C frames
  is undefined behaviour. An unanswered turn times out and loses one game; a crash loses every game
  that would have followed.
- **Never run against the live server.** The credential check and both fallbacks are exercised;
  actual match play is not.

## Trainer

`amoeba_bot_1/train.cpp`. One generation is three steps: self-play with the champion into a replay
buffer, gradient steps on batches from that buffer, then a match against the champion. Everything is
tunable by env var (`AMOEBA_SELFPLAY_GAMES`, `AMOEBA_TRAIN_STEPS`, `AMOEBA_GATE_GAMES`, …).

- **The gate is the point of the file.** Training does not improve monotonically, so "newest model"
  and "strongest model" are different things, and AlphaZero bugs produce clean loss curves. A
  generation is promoted only if it scores ≥ 55% over 100 games against the incumbent. `play.cpp`
  reads only `best`; pointing it at the newest file instead would make its rating a random walk over
  checkpoint noise.
- **Gate games must not be deterministic.** Both sides play the argmax, so without the sampled
  opening the whole match is one game replayed 100 times and the score is 0% or 100%. Colours
  alternate, because White moving first is worth something.
- **The learner is carried across generations**, and is *not* reset to the champion after a failed
  gate — that would throw away a generation of training and make a two-generation plateau
  impossible to cross. Self-play uses the champion (the AlphaGo Zero arrangement), the learner is
  what gets gated.
- **`Sample` holds a `Board`, not its encoding.** Six times smaller in the buffer, and the 12×
  symmetry augmentation has to permute hexes and directions, which it cannot do to a flat array of
  floats. Its `policy` is in absolute move ids; flipping it into the mover's frame is the trainer's
  job, next to the `encode()` it has to agree with — one place, so the two flips cannot drift apart.
- **The replay buffer is a ring over the last 500k positions**, persisted to `replay.bin` beside the
  checkpoints so a crash does not cost a day of self-play. Written staging-then-rename, like the
  `best` link.
- **Self-play is one game at a time.** The first thing to fix once the network exists: `Evaluator`
  already takes a span of boards, so the shape to move to is many concurrent games collecting leaves
  into one batched forward pass.

## Code conventions

- **Never `using namespace`** — not in headers, not in `.cpp` files, not inside `main`. Qualify
  everything (`amoeba::Board`). No `namespace game = amoeba;` aliases either. `games/amoeba/test/*`
  predates this rule and still has `using namespace amoeba;`.
- **One short namespace per target**, named after it: `amoeba_bot_1/` is `namespace bot`. Everything
  in the target goes in it, including a file's internal helpers (anonymous namespace nested inside);
  `main` stays at global scope and just calls in. No per-feature namespace (`mcts`) beside it.
- **Private members are `m_`-prefixed**, not underscore-suffixed: `m_nodes`, not `nodes_`.
- **No new targets or modules without asking.** The search lives inside `amoeba_bot_1`, not in a
  `search/` library. Only `games/amoeba` earns its own module, because keeping it MLX-free is a real
  constraint.
- Comments explain *why*, never *what*. Newest standard-library facility that fits — `std::print`,
  ranges, `std::span`. No `--flag` parsing in tools; env vars are enough.
- **No README.** This file is the documentation.

## Network — not written yet

- **Attention over the 37 hex tokens, not convolution.** Amoeba's interactions are long-range by
  construction: a stack jumps over everything and lands at exactly its height. The neighbouring cell
  is nearly irrelevant; the cell 5 away on your line is what kills you. Conv nets need depth just to
  see distance 6. With 37 tokens, all-pairs attention is 1369 pairs — free.
- **Relative-position bias.** Precompute a `constexpr bucket[37][37]`: 0 if the two hexes share no
  line, else `1 + dir*6 + (dist-1)`. Learn one bias per bucket per head. That table *is* the attack
  relation from §6.1 of the reference — hand it over instead of making the net discover it.
- **Policy head:** 12 logits per token (6 directions × move/sow) → 444, which is already exactly
  `Move::id`. Mask with `Board::legal` (set illegal logits to −∞ before softmax).
- **Value head:** mean-pool → MLP → tanh. ~1-2M params total.

## Training — what is left

The orchestration in `train.cpp` is done; the gradient step is not. What the network branch owes it
is the eight symbols in `network.hpp`: `loadNetwork`, `newNetwork`, `~Network`, `Network::evaluate`,
`Network::save`, `Trainer::Trainer`, `Trainer::~Trainer`, `Trainer::step`. That header is where the
two branches are expected to meet and disagree exactly once.

- **Value sign.** Handled by `outcomeFor()`, and `train.cpp` signs every target with it. The value
  means "good for the side to move", not "good for White". Getting this backwards trains a bot that
  reliably plays badly while every loss curve looks healthy.
- **`Trainer::step` owns both flips.** It gets `Sample`s holding a `Board` and an absolute-move-id
  policy, and has to call `encode()` and permute the policy through `kFlippedMove` itself. Keeping
  the encode and the policy flip in one function is what stops them drifting apart.
- **Symmetry augmentation is 12×, not 24×.** The hex board's symmetry group is D6 — 6 rotations × a
  reflection — and the rules are direction-agnostic, so evaluation is invariant under all 12. Colour
  swap is *not* an extra factor: once positions are canonicalised to the mover's view, swapping
  colours just sends a position back where it came from. Every symmetry permutes hexes *and*
  directions, so policy targets must be permuted identically.
- **Batch the network evaluations.** Write MCTS as "descend to collect N leaves with virtual loss →
  one batched evaluation → back up all N", never one leaf per forward pass. At ~1-2M params over 37
  tokens, single-position inference is pure overhead. This shape is also what lets 128+ self-play
  games run concurrently, and it is painful to retrofit.
- **Thread the repetition history through search.** `apply` only detects repetition when given the
  hash history; without it MCTS walks into draws it cannot see.
- **MLX is lazy** — nothing computes until `mx::eval()`. Easy to build a huge graph by accident, and
  it makes naive timing meaningless. Homebrew has `mlx` 0.31.2 (`/opt/homebrew/include/mlx`) plus the
  Python bindings.
- The MLX C++ API has no real optimizer or layer library, so Adam and the attention blocks are
  hand-written. If that becomes a time sink, training in Python + MLX while keeping engine, encoder
  and MCTS in C++ is a legitimate architecture, not a retreat.

## Next steps

1. **Run `amoeba_random_test` against the server**, and play in a practice room. Neither has ever
   touched the live server, and everything downstream assumes they would pass. Worth doing before
   the model lands: if the engine and the server disagree about the rules, a bad rating later cannot
   be told apart from a bad network, and `syncToServer`'s resync fallback hides it.
2. **Network** in MLX, filling in `network.hpp`. Nothing else has to change.
3. **Concurrent self-play with batched leaf evaluation.** One game at a time wastes the device.
4. **Tree reuse between moves**, worth 20-40% more effective simulations.

Run the whole pipeline end to end at a deliberately tiny scale first — 2 attention blocks, 50 sims,
20 games a generation. "It runs and the loss goes down" proves almost nothing; AlphaZero bugs
produce clean training curves. **The real first milestone is generation 1 beating generation 0** —
that is what the gate in `train.cpp` reports, and it is the only honest signal.

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

## Build

CMake + Ninja, multi-config, build tree in `build.nosync/`.

```
cmake -S . -B build.nosync
cmake --build build.nosync
```

`amoeba_bot_1_core` (search + checkpoints) is shared; `play.cpp` and `train.cpp` each build one
executable on top of it. Only the player links the arena SDK. **Both fail to link** until
`network.cpp` arrives — the eight undefined symbols are the whole of the model branch's contract.

```
# ranked games back to back, for as long as the process lives
BOT1_ID=… BOT1_KEY=… ./build.nosync/amoeba_bot_1/Release/amoeba_bot_1

# one practice game
ROOM_ID=… BOT1_ID=… BOT1_KEY=… ./build.nosync/amoeba_bot_1/Release/amoeba_bot_1

# self-play and training, forever; no credentials, never touches the network
./build.nosync/amoeba_bot_1/Release/amoeba_bot_1_train
```

Both read `AMOEBA_CHECKPOINTS` (default `./checkpoints`). Run them side by side and the player picks
up each promoted generation at its next game start. The trainer writes `gen-NNNN.safetensors` per
generation plus `replay.bin`, and moves the `best` symlink only when a generation passes the gate.

Everything else the trainer takes is an env var with a default: `AMOEBA_SELFPLAY_GAMES` (200),
`AMOEBA_SELFPLAY_SIMULATIONS` (400), `AMOEBA_TRAIN_STEPS` (1000), `AMOEBA_BATCH` (512),
`AMOEBA_GATE_GAMES` (100), `AMOEBA_GATE_SIMULATIONS` (400), `AMOEBA_GATE_THRESHOLD` (0.55),
`AMOEBA_OPENING_PLIES` (15), `AMOEBA_BUFFER` (500000), `AMOEBA_SEED`.

Two test binaries, neither using a test framework:

- `amoeba_encode_test` — offline. Plays random games and checks `encode()` on every position.
  Just run it: `./build.nosync/games/amoeba/test/Debug/amoeba_encode_test`
- `amoeba_random_test` — needs a live server. Plays random legal moves against Game Arena and
  compares the legal-move set every ply.
  `ARENA_DOMAIN=staging-game-arena.irvine.jp BOT1_ID=… BOT1_KEY=… ROOM_ID=… ./build.nosync/games/amoeba/test/Debug/amoeba_random_test`
