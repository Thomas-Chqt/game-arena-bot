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
  include/amoeba/amoeba.hpp     rules engine, incl. startPosition()
  include/amoeba/encode.hpp     Board -> network input
  src/amoeba.cpp
  src/encode.cpp
  test/encode_test.cpp          offline
  test/random_test.cpp          live server parity harness
amoeba_bot_1/
  mcts.hpp / mcts.cpp           PUCT search + rollout evaluator
  network.hpp / network.cpp     parameters, forward pass, NetworkEvaluator
  training.hpp / training.cpp   Batch, the loss and Adam; no self-play yet
  main.cpp                      imitation-training driver (was the arena client)
```

## Current state

The rules engine works. ~296k plies/sec, ~3.7k full random games/sec, single-threaded, `-O2`.

- `Hex` — a stack packed into one `uint64_t`. `Board` — 376 bytes, with a 444-bit legality bitmask
  that doubles as the policy mask.
- Compile-time tables: `kCoordinates`, `kIndexOf`, `kNeighbour`, `kFlipped`, `kDirections`.
- `generateLegal`, `kernelAttacked`, `applyRaw` / `apply`, `fromString` / `toString`.
- Terminal conditions agree with `amoeba-reference.md`: repetition on the 3rd occurrence, staleness
  80, move cap 250, adjudication by controlled stacks then by enemy pieces held inside them.
- `encode()` and its test.
- PUCT search and a rollout evaluator, wired into `amoeba_bot_1` as a playable arena client.

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

**The policy comes back in flipped space.** `policyToAbsolute(whiteToMove)` in `encode.hpp` is the
444-entry permutation that maps it back — identity for White, `kPolicyUnflip` for Black. Needed both
for the move you send and for training targets. `kFlipped` and `opposite()` are both involutions, so
the permutation is its own inverse; a `static_assert` holds that.

`encode_test.cpp` covers all of this: `encode(position) == encode(mirrored position)` bit for bit,
plus `policyToAbsolute()` against the legality bits `encode()` wrote, on every position of 200 random
games. Verified to have teeth — dropping the coordinate rotation, the colour swap, or the direction
remap each fails on the first position, and so does breaking the policy permutation three ways.

**This is the failure mode to fear.** With the unflip disabled, the network's policy still sums to
exactly 1.0 over the legal moves with zero mass on illegal ones. It looks perfect and plays the board
rotated.

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

### Deliberately absent

- **Temperature sampling** — lives in `main.cpp`'s `chooseMove()` rather than in the search.
- **Tree reuse between moves** — the subtree under the played move stays valid, and re-rooting is
  roughly 20-40% more effective simulations for free. `m_nodes` / `m_edges` are members so
  allocations are reused, but the tree is cleared on every `run()`.
- **A search deadline** — 2000 fixed simulations measure **351 ms** worst case over the first 24
  plies, against a 5 s turn limit. Safe only while an evaluation is a rollout.

### Dirichlet root noise

`Config::rootNoise` / `noiseAlpha` / `noiseSeed`, **off by default** so competition keeps the
network's own priors. `prior = (1 - rootNoise) * network + rootNoise * Dirichlet(alpha)`, at the root
only, because the root is the position that becomes a training example — noise deeper in the tree just
spoils the search's judgement. A Dirichlet draw is independent `std::gamma_distribution` samples
normalised; `alpha < 1` makes each draw spiky, so a different random handful of moves is promoted each
game. AlphaZero scaled `alpha` as 10 / average legal moves, and Amoeba averages 27.

Verified on the failure it exists to prevent — a network and `selectEdge` are both deterministic:

```
no noise      : 1 distinct best move over 8 searches
Dirichlet 0.25: 8 distinct best moves over 8 searches
```

Without it, self-play from a given network replays one game forever, the network never sees a position
it does not already understand, and training stalls with every loss curve looking healthy.

### Leaf batching with virtual loss

`Search::run` is three phases: `collect()` descends `Config::batchSize` times, one batched
`evaluate()`, then `backUp()`. `addNode` split into `expand()` — build a node and its edges from an
`Evaluation` already in hand — and a single-board path used only for the root.

**Virtual loss** is what makes the descents diverge: on the way down, each edge is provisionally
recorded as having come back a loss (`valueSum -= 1`, `++visits`), which `backUp()` removes before
applying the real result. Without it all N descents follow one path and return the same leaf N times.

Measured, 200 simulations, 2 blocks at width 64, Metal:

| leaves per batch | ms/search |
|---|---|
| 1 | 336.4 |
| 4 | 105.8 |
| 16 | 42.2 |
| 64 | **21.5** |

**15.6×**, which puts the network *below* the rollout teacher's ~33 ms per search. Leaf batching is
what makes a network search cheaper than a rollout search, not merely competitive.

- **Duplicate leaves are kept, not deduplicated.** Two descents can still choose the same unexpanded
  edge; both are evaluated, one node ends up unreachable, and both back up the same correct value.
  Sharing one evaluation would need the backup to know about the pairing.
- **Terminal leaves stay in the batch** but take their value from the rules, not the network. Keeping
  the two arrays parallel is worth more than the handful of wasted evaluations — and it is why the
  policy mask is −1e9 rather than −∞.

### Verification

`MODE=match` in `main.cpp` is the strength check (see Training driver). Over 20 games at 200
simulations, alternating colours:

```
rollout vs random             : 100.0% +/- 0.0%   (20-0-0)
network, 1 leaf   vs random   :  97.5% +/- 3.5%   (19-0-1)
network, 64 leaves vs random  : 100.0% +/- 0.0%   (20-0-0)
network, 64 leaves vs rollout :  40.0% +/- 11.0%  (6-10-4)
```

**Before refactoring `run()` for batching, fingerprint its visit counts.** At `batchSize = 1` the
batched code reproduces the pre-refactor fingerprints bit for bit — virtual loss adds and removes
exactly 1.0 at magnitudes float32 represents exactly — which pins the restructure independently of
whether batching helps. Do this again before touching the descent; the search has no other offline
test, and ±11% over 20 games cannot resolve a small regression.

## Training driver

`amoeba_bot_1/main.cpp`. **This replaced the arena client**, which is intact in git at `ba8cceb` and
will need somewhere to live again. `CMakeLists.txt` still links `arena::arena` even though nothing
uses it now.

It trains the network to **imitate rollout MCTS** — not self-play. The teacher is `RolloutEvaluator`,
which never changes, so games are generated once and the only moving part is the network. That
ordering is deliberate: in a bootstrapping loop, a broken network, bad data and an unstable loop all
look identical, whereas here there is exactly one thing that can be wrong.

- **The policy target inherits the teacher's ceiling; the value target does not.** Move preferences
  come from the teacher's visit counts, but the value target is who actually won the game, which is
  ground truth however weak the teacher is.
- **Generation is threaded, one game per slot.** `std::jthread` in an inner scope so the destructors
  join before results are read, one `std::atomic<int>` handing out game indices so a thread finishing
  a short game picks up another, and no mutex on the data because each game writes its own slot.
  Measured **5.7×** on a 10-core M-series (6 of those are efficiency cores): 100 games at 200
  simulations in 21 s, 940% CPU. 1000 games is ~3.6 minutes for ~85k positions.
- **Seeded per game, not per thread**, which is what keeps a run reproducible from `SEED` regardless
  of how the threads interleave, and keeps samples in game order.
- **Held out by position after a shuffle**, and the held-out loss is reported beside the training loss.
  Worth it: on a 703-position smoke run the training policy loss fell 3.76 → 2.76 while the held-out
  loss got *worse* after step 100. Without that column it looked like learning.
- **Progress output must be flushed.** `std::println` leaves stdout block-buffered when it is not a
  terminal, so a redirected run showed an empty log for 100 s while the buffer filled. Everything goes
  through `report()`, which flushes. The smoke test missed this because a short run flushes on exit.
- **Games are not saved to disk**, so every hyperparameter experiment re-generates them. Worth fixing
  before doing much tuning.

Env vars, all optional: `GAMES SIMULATIONS SAMPLING_PLIES SEED BLOCKS WIDTH HEADS STEPS BATCH RATE
DECAY OUT`.

## Arena client — replaced, see above

`amoeba_bot_1/main.cpp`. Four callbacks, same shape as the reference's random bot.

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
- **Never run against the live server.** The credential check and both fallbacks are exercised;
  actual match play is not.

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

## Network

`amoeba_bot_1/network.hpp` / `network.cpp`. `Network` holds the parameters, `forward()` is the
prediction, `NetworkEvaluator` is the `bot::Evaluator` the search talks to. No training yet.

- **Parameters are one flat `std::vector<mlx::core::array>`**, because that is what
  `mlx::core::value_and_grad` consumes and the shape it returns gradients in. `Network::layout()` is
  the single definition of which tensors exist and in what order; `forward()` walks that order
  positionally, so each tensor needs its own named local — the order in which C++ evaluates two
  cursor reads inside one expression is unspecified.
- **6 blocks at width 128 with 8 heads is 1,219,965 parameters.** Reproducible from the seed alone;
  checkpoints round-trip exactly through safetensors, with the shape in the metadata so a checkpoint
  from a different architecture is rejected at load rather than at the first matmul.
- **`forward()` returns raw logits.** Masking is the caller's job, because inference wants
  probabilities over legal moves and training wants the logits.
- **`NetworkEvaluator` must hand back probabilities, not logits.** The search sums the priors of the
  legal moves and divides (`mcts.cpp:78-79`), so raw logits would give it negative priors.
- **Mask with a large finite penalty, not −∞.** A terminal position has no legal moves, `addNode`
  evaluates it anyway, and softmax over an all-−∞ row is NaN, which would spread into every parameter
  that touched it. −1e9 leaves a harmless uniform row that nothing reads.

### Measured latency, and what it means for the search

One position, 6 blocks at width 128, `-O2`:

| batch | GPU | CPU |
|---|---|---|
| 1 | 1.70 ms | **0.95 ms** |
| 8 | 0.25 ms/pos | 0.47 ms/pos |
| 64 | **0.16 ms/pos** | 0.44 ms/pos |
| 256 | 0.15 ms/pos | 0.45 ms/pos |

At 1.2M parameters over 37 tokens this is dispatch-bound, not compute-bound, so **CPU beats Metal at
batch 1** and Metal wins from batch 8 up. The search asks one position at a time, so 2000 simulations
would cost 3.4 s on GPU against a 5 s turn limit — the fixed simulation count in `main.cpp` is no
longer safe. Leaf batching with virtual loss is worth ~10× per position and is now the thing standing
between the network and a usable bot.

- **Attention over the 37 hex tokens, not convolution.** Amoeba's interactions are long-range by
  construction: a stack jumps over everything and lands at exactly its height. The neighbouring cell
  is nearly irrelevant; the cell 5 away on your line is what kills you. Conv nets need depth just to
  see distance 6. With 37 tokens, all-pairs attention is 1369 pairs — free.
- **Relative-position bias.** `kBucket[37][37]` in `network.hpp`: 0 if the two hexes share no line,
  else `1 + dir * kMovableMax + (dist - 1)`. One learned bias per bucket per head, 37 buckets. That
  table *is* the attack relation from §6.1 of the reference — handed over instead of discovered. The
  stride for direction is the number of *distance* buckets; both are 6, which makes the mistake
  invisible.
- **Policy head:** 12 logits per token → `[batch, 37, 12]`, which reshapes straight onto `Move::id`
  because the 12 run direction-major and splitting-minor.
- **Value head:** mean-pool over the 37 tokens → MLP → tanh.

## Training

`amoeba_bot_1/training.hpp` / `training.cpp` has `Batch`, `makeBatch()`, `loss()` and `Adam`. What is
missing is self-play — there is no data, so no weight has ever been adjusted by anything but a test.

- **`loss()` returns `{ total, policy, value }`.** `mlx::core::value_and_grad` differentiates the
  first element and hands the rest back for free, so the components cost nothing and are worth
  keeping: a value loss sitting flat while the policy loss falls means the value head is not learning,
  and one averaged number would hide that.
- **`makeBatch()` is where the policy target crosses into flipped space.** The search counts visits in
  absolute `Move::id` and `forward()` answers in the space `encode()` used, so the target is permuted
  with `policyToAbsolute` — the same table as inference, since it is its own inverse. Skip it and you
  train on scrambled labels, and the loss still falls.
- **`outcomes` reaches `makeBatch()` already signed** from the mover's point of view. Only self-play
  knows the game result, so that is deliberately the caller's business and `makeBatch()` does not
  second-guess it. See the value-sign note below.
- **Pass `weightDecay = 0` when overfitting one batch on purpose**, or the penalty holds the loss off
  zero and hides whether the gradients connect at all.
- **A sample with no visits throws.** That is a terminal position in the training set, i.e. a
  self-play bug, and dividing by zero visits would otherwise put NaN into every parameter.
- **`Adam::step` takes the rate as an argument**, not from its config, because it is expected to fall
  on a schedule. It returns new parameters rather than mutating: an `mlx::core::array` is a handle onto
  a graph, not a buffer. The caller must `mlx::core::eval()` the result every step or the graph grows
  until memory runs out; evaluating the parameters is enough, since they depend on both moment
  estimates.
- **Adam's moment estimates are not checkpointed.** Only `Network` persists. Between generations that
  is fine — the optimiser restarts anyway — but interrupting a run mid-generation loses its momentum
  and costs a few hundred wasted steps. Decide before the first long run, not after.

Measured on 8 positions with a uniform target, untrained network:

```
policy loss 5.1620   value loss 1.7809   total 6.9429
floor for a uniform target = mean log(legal moves) = 3.9174   -> above it, as it must be
gradients: 87 tensors, 0 of them all-zero
largest gradient magnitude 2.553e+00, smallest 1.521e-03   (1678x spread)
```

The floor is the check worth keeping: cross-entropy of *any* prediction against a uniform target over
N legal moves cannot go below `log(N)`, so a policy loss under that means the mask or the
normalisation is wrong. The 1678× gradient spread across tensors is exactly why plain
`w -= rate * gradient` will not do and Adam divides by `sqrt(variance)`.

### The overfit-one-batch check

32 distinct positions with invented one-hot targets, 2 blocks at width 64, no weight decay,
rate 1e-3:

```
  step       total      policy       value
     1    5.774683    4.062842    1.711841
   300    0.000365    0.000351    0.000014
   600    0.000115    0.000110    0.000005
```

A 50,000× reduction, so the gradients reach every one of the 87 tensors and Adam moves them. Run this
before trusting anything downstream; a broken gradient path plateaus instead of converging, and no
later measurement would isolate it. Step time stayed flat at 5.6 ms across 600 steps, which is also
the check that nothing is leaking graph.

**Duplicate positions floor the loss, and that is correct.** The first attempt plateaued at exactly
0.04332 no matter the learning rate. Two of the 32 samples were the same position — the harness
restarted from the opening after a game ended — carrying contradictory one-hot targets. The best
possible answer is 50/50, so each contributes `-log(0.5)`, and `2 x 0.693 / 32 = 0.04332` to four
decimals. Nothing was wrong with the loss or the optimiser.

This matters beyond the test: in real self-play the same position *will* recur with different visit
distributions, and the network correctly learns their average. **So the training loss will never
approach zero, and chasing that would be chasing a bug that does not exist.** The floor is the
entropy of the search's own disagreement with itself.

- **Value sign.** The value means "good for the side to move", not "good for White". Negate it for
  every position where the mover lost. Getting this backwards trains a bot that reliably plays badly
  while every loss curve looks healthy.
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
  it makes naive timing meaningless. Homebrew has `mlx` 0.32.0 (`/opt/homebrew/include/mlx`) plus the
  Python bindings.
- The MLX C++ API has no real optimizer or layer library, so Adam and the attention blocks are
  hand-written. If that becomes a time sink, training in Python + MLX while keeping engine, encoder
  and MCTS in C++ is a legitimate architecture, not a retreat.

## Next steps

1. **Run `amoeba_random_test` against the server**, and play `amoeba_bot_1` in a practice room.
   Neither has ever touched the live server, and everything downstream assumes they would pass.
2. **Loss, Adam and the overfit-one-batch test.** Nothing downstream is trustworthy until 32
   positions trained in isolation drive the loss to nearly zero — that is what proves the gradient
   path is connected. MLX has `value_and_grad`, so only the optimiser is hand-written.
3. **Does the student beat the teacher?** Put the trained network in MCTS against rollout MCTS. Search
   with a learned prior should beat search without one — rollout MCTS puts only 40 of 200 visits on its
   best move, so most of its thinking goes into moves that do not matter. This is the honest test of
   whether the architecture learns Amoeba, and it gates whether self-play is worth starting.
4. **Leaf batching with virtual loss** in the search. The latency table above makes this a
   correctness issue for the 5 s turn limit, not an optimisation.
5. **Self-play → training loop.** Needs Dirichlet root noise and a real search deadline added first —
   see Search above. Temperature sampling now exists in `main.cpp` rather than in the search.
6. **Head-to-head harness**, generation 2 vs generation 1.

There is still **no entry point** for any of this: `amoeba_bot_1`'s `main` plays one arena match and
needs credentials, so the forward-pass benchmark and the overfit test above were run from throwaway
programs outside the repo. Somewhere to run offline network checks is the next structural decision.

Run the whole pipeline end to end at a deliberately tiny scale first — 2 attention blocks, 50 sims,
200 games. "It runs and the loss goes down" proves almost nothing; AlphaZero bugs produce clean
training curves. **The real first milestone is generation 2 beating generation 1 over 100 games**, so
build the head-to-head harness early. It is the only honest signal.

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

CMake + Ninja, multi-config, build tree in `build.nosync/`. The generator is not the default, so
pass it when configuring from scratch — the `Debug/` and `Release/` path components below depend on
it.

```
cmake -S . -B build.nosync -G "Ninja Multi-Config"
cmake --build build.nosync                     # Debug
cmake --build build.nosync --config Release
```

`find_package(MLX 0.32 CONFIG REQUIRED)` has to sit *after* `project()`: MLX imports itself as a
shared library, and before `project()` CMake has not yet established that the platform can link
one. The `arena` call above it predates that and only works because its import is static.

```
BOT1_ID=… BOT1_KEY=… ./build.nosync/amoeba_bot_1/Release/amoeba_bot_1        # one ranked match
ROOM_ID=… BOT1_ID=… BOT1_KEY=… ./build.nosync/amoeba_bot_1/Release/amoeba_bot_1   # practice room
```

Two test binaries, neither using a test framework:

- `amoeba_encode_test` — offline. Plays random games and checks `encode()` on every position.
  Just run it: `./build.nosync/games/amoeba/test/Debug/amoeba_encode_test`
- `amoeba_random_test` — needs a live server. Plays random legal moves against Game Arena and
  compares the legal-move set every ply.
  `ARENA_DOMAIN=staging-game-arena.irvine.jp BOT1_ID=… BOT1_KEY=… ROOM_ID=… ./build.nosync/games/amoeba/test/Debug/amoeba_random_test`
