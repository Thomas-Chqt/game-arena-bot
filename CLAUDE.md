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

`amoeba/amoeba-reference.md`

## Layout

Everything lives in one flat directory and builds two executables. There is no library, no
per-feature subdirectory, and no test binary.

```
amoeba/
  amoeba-reference.md    the rules, mirrored from the server engine
  amoeba.hpp / .cpp      rules engine, startPosition(), and encode() with its policy permutation
  mcts.hpp / .cpp        fresh-tree sequential PUCT search and rollout evaluator
  network.hpp / .cpp     parameters, forward pass, NetworkEvaluator, Batch, loss(), Adam
  bot.cpp                arena client        -> amoeba_bot
  train.cpp              self-play + training -> amoeba_train
```

**Two programs, on purpose.** `amoeba_bot` plays and never trains; `amoeba_train` trains and never
talks to the server — it does not even link the arena SDK, which is what keeps that true. They share
one `.safetensors` file and nothing else, so both can run at once and either can be killed and
restarted without the other noticing. `amoeba_train` writes the file whenever a generation passes the
gate; `amoeba_bot` re-reads it at the start of each game.

## Current state

The rules engine works. ~296k plies/sec, ~3.7k full random games/sec, single-threaded, `-O2`.

- `Hex` — a stack packed into one `uint64_t`. `Board` — 376 bytes, with a 444-bit legality bitmask
  that doubles as the policy mask.
- Compile-time tables: `kCoordinates`, `kIndexOf`, `kNeighbour`, `kFlipped`, `kDirections`.
- `generateLegal`, `kernelAttacked`, `applyRaw` / `apply`, `fromString` / `toString`.
- Terminal conditions agree with `amoeba-reference.md`: repetition on the 3rd occurrence, staleness
  80, move cap 250, adjudication by controlled stacks then by enemy pieces held inside them.
- `encode()`, plus `policyToAbsolute()` — the 444-entry permutation that maps a policy out of the
  mover's frame, `static_assert`ed to be its own inverse.
- Fresh-tree PUCT search with Dirichlet root noise; a rollout evaluator. The
  application code is single-threaded.
- The network: attention over 37 hex tokens with a relative-position bias, `loss()` and `Adam`.
- `amoeba_bot` and `amoeba_train`, both built and both run. The full generation loop — self-play,
  training, gate, promotion, write back to the same file — has been run end to end at toy scale and
  works; `amoeba_bot` loads a checkpoint and reaches the server connect.

**Nothing has ever been run against the live server**, and there is no longer a parity harness: the
engine was checked against `amoeba-reference.md` as a document, not against the server that enforces
it. The reference *Python* implementation is what was differentially tested; our C++ engine has not
been. 200 random games land within noise of the reference's own outcome distribution (§7.3), which is
reassuring but is not the same thing. `syncToServer()`'s resync fallback in `bot.cpp` will paper over
a rules disagreement rather than reporting it, so a bad rating and a bad network look identical.

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

**The policy comes back in flipped space.** `policyToAbsolute(whiteToMove)` in `amoeba.hpp` is the
444-entry permutation that maps it back — identity for White, `kPolicyUnflip` for Black. Needed both
for the move you send and for training targets. `kFlipped` and `opposite()` are both involutions, so
the permutation is its own inverse; a `static_assert` holds that.

`encode_test.cpp` used to cover all of this — `encode(position) == encode(mirrored position)` bit for
bit, plus `policyToAbsolute()` against the legality bits `encode()` wrote, on every position of 200
random games — and was verified to have teeth: dropping the coordinate rotation, the colour swap or
the direction remap each failed on the first position, and so did breaking the policy permutation
three ways. **It was deleted on request**, so nothing checks the flip any more. This is the one
mapping whose failure is silent: a wrong permutation still gives a valid distribution over legal
moves, the loss still falls, and the bot simply plays as though the board were rotated.

**This is the failure mode to fear.** With the unflip disabled, the network's policy still sums to
exactly 1.0 over the legal moves with zero mass on illegal ones. It looks perfect and plays the board
rotated.

## Search (MCTS)

`amoeba/mcts.hpp` / `mcts.cpp`. PUCT, the AlphaZero variant — no rollouts *inside* the search;
a new position's policy and value come back from the caller as an `Evaluation`.

### The pieces

- `Evaluation` — a 444-float policy plus a value in `[-1, 1]`. It is the only network-facing type
  MCTS knows.
- `MCTS` — **owns no evaluator and calls none.** Its constructor fixes one root board and history;
  `pendingLeaf()` performs one descent and hands out its one unevaluated leaf, and `absorb()` expands
  and backs up that leaf. `visits()` is the 444 visit counts; `bestMove()` is the argmax, and
  normalised the same array is the policy target. A real move creates a new `MCTS` instance. The
  bot drives this handshake directly; the trainer gathers leaves from multiple searches first.

### Invariants that are easy to break

- **Every stored value is from the point of view of the side to move at the node that owns it**, so
  the backup flips sign once per level. Inverting this yields a bot that actively seeks its own worst
  lines — it loses to random play, which is at least a loud symptom.
- **The statistics live on the edge, not the child node.** PUCT needs a prior and a visit count for
  moves that have never been played, i.e. before the resulting position exists. Measured branching is
  52 at the opening and 27 on average, so an 800-simulation search holds ~801 nodes and ~21,600
  edges; making every edge a node would mean 27× the `apply()` calls and 8.5 MB instead of 432 KB.
- **Repetition history has fixed storage.** `MCTS` keeps the real-game prefix in an array bounded
  by the 250-ply move limit. A descent copies that prefix into a local fixed array and appends one
  hash per node. The edge trail and root-noise samples are fixed arrays too, so none of this scratch
  state allocates. Only the persistent node and edge vectors grow dynamically.
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

- **Temperature sampling** — lives in `train.cpp`'s `chooseMove()` rather than in the search.
- **Batching inside `MCTS`** — every descent observes the result of the previous one. The runner in
  `train.cpp` gathers one leaf from each independent game into a network batch.
- **Tree reuse across moves** — every position gets a fresh tree and the full configured simulation
  budget, which keeps a training target independent of searches performed for earlier positions.

### Dirichlet root noise

`MCTSConfig::rootNoise` / `noiseAlpha` / `noiseSeed`, **off by default** so competition keeps the
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

### Batching across games, not across guesses

The evaluator is the whole cost of the search and it is dispatch-bound, so it has to be fed whole
batches. `MCTS` deliberately does not make one batch from speculative descents. It returns one leaf
and waits for that result, so every selection sees completely current visit and value statistics.

`GameBatchRunner` keeps independent games in flight. Each game contributes at most one pending leaf
per round; the runner groups those leaves by evaluator, performs one network call per non-empty
group, distributes the answers, and then lets every search descend again. `CONCURRENT` and
`GATE_CONCURRENT` control this caller-owned batch width.

The online bot has only one game, so it evaluates one leaf per call. That is deliberately slower than
speculative batching, but it keeps batching and virtual-loss policy out of MCTS and establishes a
simple baseline to measure before adding any optimisation back.

### Fresh tree per move

After a real move, the bot and trainer construct a new `MCTS` with the resulting board and complete
game history. No nodes, priors, values or visits survive from the previous position. Consequently every
training sample receives exactly `MCTSConfig::simulations` fresh root visits,
and the trainer supplies a fresh reproducible noise seed for every self-play position.

### Verification

The cheap invariants are that the root visit counts sum to exactly `simulations`, every illegal move
has zero visits, a completed self-play position has a non-empty policy target, and noise-disabled
search is deterministic. The repository does not currently contain a dedicated search test target,
so a bounded miniature training run is the end-to-end check of the handshake and caller batching.

## Execution model

All application-owned work is single-threaded. Tree traversal, board encoding, policy scattering and
training-batch preparation use direct loops. MLX may manage its own device execution internally, but
the program does not create threads or overlap CPU traversal with a network call.

## Trainer

`amoeba/train.cpp` -> `amoeba_train`. **The `.safetensors` path is a required argument** and is
created from random weights if it does not exist. It loops forever, and **writes back over that same
file whenever a generation passes the gate** — which is the
file `amoeba_bot` reads, so a bot running alongside picks the improvement up at its next game.

One generation is three steps: play `GAMES` games of the current best against itself into a replay
buffer, train a candidate on batches from that buffer, then play the candidate against the current
best and promote only if it wins.

**`GameBatchRunner` is where the games are played**, both for self-play and for the gate. It keeps
games in flight, refills a slot as its game ends, and drives them in lockstep: one round gathers one
pending leaf per game, evaluates one batch per network, distributes the answers, and continues.

- **`CONCURRENT` is the maximum inference batch width.** Each active game contributes at most one
  position. In a gate the positions split between the two networks according to the side to move.
- **`GAMES` is above `CONCURRENT`, so a slot takes on another game instead of going idle.** What a
  game costs does not depend on which slot owns it, so refilling keeps network batches populated for
  most of the run.
- **The tail uses smaller batches.** Once only a few games remain, each still contributes one leaf
  and the program accepts the lower device utilisation rather than complicating MCTS.
- **The gate's field is smaller (`GATE_CONCURRENT`, 64)** because there the field size is also how long
  it takes to hear a verdict: `settled()` is only reconsulted as games come in, and 256 at once would
  spend most of `GATE_GAMES` before the first chance to stop. Once it does say stop, the games still in
  flight are abandoned rather than played out, so the tally is exactly what it was when it decided.
- **The gate uses one tree per game.** The caller selects the candidate or champion evaluator from
  the side to move; a fresh tree contains no statistics belonging to an earlier network.
- **Completion is serial.** The callback updates its tally directly and can stop the field without
  atomics or a lock. Per-game progress reports moves and evaluated positions.
- **Everything random about a game comes from its id**, so a run stays reproducible from `SEED`.

**The defaults are sized for a real overnight run**, not a smoke test: 6 blocks at width 128
(~1.2M parameters), 400 simulations, 512 self-play games 256 at a time, and a 200-game gate 64 at a
time. A generation is expected to take on the order of an hour; `GAMES` is the knob to turn down
first, because it is the only one that trades directly against how many generations a night holds.
To check the wiring quickly instead:

```
BLOCKS=2 WIDTH=64 HEADS=4 GAMES=8 CONCURRENT=8 SIMULATIONS=50 STEPS=100 GATE_GAMES=8 \
  GATE_CONCURRENT=8 ./build.nosync/amoeba_train
```

Env vars, all optional: `BLOCKS WIDTH HEADS` (read only when starting from scratch — otherwise the
shape comes out of the checkpoint), `SEED GAMES CONCURRENT SIMULATIONS SAMPLING_PLIES NOISE`,
`STEPS BATCH RATE DECAY BUFFER`, `GATE_GAMES GATE_CONCURRENT GATE GATE_SIMULATIONS`.

### Things it gets right that are easy to get wrong

- **The gate is the point.** A training loss can fall while the player gets worse, and AlphaZero bugs
  produce clean loss curves. Nothing is promoted on a curve, only on games won.
- **Gate games must not be deterministic.** Both sides take the argmax past `SAMPLING_PLIES`, so the
  early sampling in `chooseMove()` is the only thing making game N differ from game N-1. Without it
  the whole match is one game replayed `GATE_GAMES` times and the score is 0% or 100%. Colours
  alternate, because White moving first is worth something.
- **Game ids stay unique across generations.** The replay buffer spans generations and
  `splitByGame()` holds out whole games — if ids restarted each generation it would hold out one
  generation's game and train on a different game with the same id.
- **Hold out whole games, never positions.** Positions within a game are near-copies carrying the
  same outcome label, so a by-position split leaves a held-out position's own game in training and
  the value head scores by recognising the game. The symptom is a held-out loss that *improves* as
  you generate fewer games.
- **Training keeps the best step's weights**, not the last. Held-out loss turns back up well before
  training ends; at 300 games the value loss at step 600 is 43% worse than at step 100.
- **The candidate starts from the best weights**, not from scratch: each generation refines rather
  than relearns.
- **Root noise is self-play only.** The gate forces `rootNoise = 0` so a match takes the network's
  own opinion.
- **Progress output is flushed.** `std::println` block-buffers when stdout is not a terminal, so a
  redirected run showed an empty log for 100 s. Short runs hide this because exiting flushes.
- **MLX calls are serial.** Each non-empty evaluator batch completes before the runner distributes
  its results and resumes tree traversal. This is intentionally simple and makes the batching
  boundary visible in one place.

### Deliberate gaps

- **Games are never written to disk**, so a restart loses the replay buffer and has to regenerate it.
  Only the weights survive.
- **Adam's moment estimates are not checkpointed**, so a generation cannot be resumed part way.
- **The gate needs hundreds of games**, which is why the default is 200 — an error bar of ±3.5%,
  against ±8% at 40 where promotion is close to a coin flip. It costs as much as the self-play it
  judges, and that is the price of the only honest signal in the system. It prints the error bar
  every time; read it before believing a promotion.
- **The gate still stops at twenty games, but it pays for a field.** `settled()` is consulted on each
  completion, and games finish staggered, so twenty of the field of 64 are usually done long before
  the rest — but when it does stop, the other 44 are abandoned part-played. The wall clock is still
  far better than twenty sequential games; the wasted work is the price of the batch.

## Arena client

`amoeba/bot.cpp` -> `amoeba_bot`. **The `.safetensors` path is a required argument and must exist** —
this program never trains and never writes weights, so there is nothing sensible to do without them.
Credentials are `BOT1_ID` and `BOT1_KEY`, the same as the reference random
bot; `ROOM_ID` picks a practice room, and without it the bot runs `arena_start_continuous`, which
queues game after game and only returns on error. Four callbacks, same shape as the random bot.

- **Translation.** The engine names a move `(from, dir, splitting)`; the server names it
  `(pos, destination, splitting)`. `collectServerMoves()` turns the server's list into engine move
  ids, the search picks one, and `chooseMove()` finds it back in that list to recover the server's
  own strings.
- **The bot keeps its own board and history.** `arena_game_state_t` carries a position
  and nothing else — no ply count, no history — so `syncToServer()` works out which legal move the
  opponent played and applies it locally. Every applied move constructs a fresh search with the full
  accumulated hash history. Re-parsing the server's board each turn instead would silently reset
  `ply` and `staleness` and would leave no history for search repetition detection.
- **Two fallbacks, because forfeiting is worse than wrong bookkeeping.** If no legal move reaches the
  server's position, adopt that position and carry `ply` forward — that is a rules bug. If the search
  picks a move the server did not offer, play the server's first move.
- **The model is loaded once per game**, in `on_game_start`. Between games is the only safe moment:
  one tree scoring its positions with two different networks is incoherent and would not show up in
  any log. It is also how a promotion by a trainer running alongside gets picked up.
- **The bot evaluates one leaf per call.** It has no independent games from which to form a batch;
  single-game batching is deliberately absent from the baseline.
- **The search is count-bound, not deadline-bound.** `searchSimulationCount` is 1000 and there is no
  clock, so the move logged is always the one 1000 fresh simulations chose. Nothing
  enforces the server's 5 s — `on_move` times the whole reply, sync and translation included, and
  prints it next to the visit count and the root's legal-move count so a flat distribution can be
  told apart from a truncated one.
- **Every callback is wrapped in `guarded()`.** A C++ exception unwinding through the SDK's C frames
  is undefined behaviour. An unanswered turn times out and loses one game; a crash loses every game
  that would have followed.
- **Never run against the live server.** The credential check, the checkpoint lookup and both
  fallbacks are exercised; actual match play is not.

## Code conventions

- **Never `using namespace`** — not in headers, not in `.cpp` files, not inside `main`. Qualify
  everything (`amoeba::Board`). No `namespace game = amoeba;` aliases either.
- **Two namespaces, and only two.** `amoeba` is the rules engine and the encoder; `bot` is the search,
  the network and the two programs. Everything in `bot` goes in it, including a file's internal
  helpers (anonymous namespace nested inside); `main` stays at global scope and just calls in. No
  per-feature namespace (`mcts`, `training`) beside them.
- **Private members are `m_`-prefixed**, not underscore-suffixed: `m_nodes`, not `nodes_`.
- **No new targets, directories or modules without asking.** Everything is one flat `amoeba/`
  directory building two executables. New code goes in an existing file unless asked otherwise.
- Comments explain *why*, never *what*. Newest standard-library facility that fits — `std::print`,
  ranges, `std::span`. No `--flag` parsing in tools; env vars are enough.
- **No README.** This file is the documentation.

## Network

`amoeba/network.hpp` / `network.cpp`. `Network` holds the parameters, `forward()` is the prediction,
and `NetworkEvaluator` turns batches of boards into `Evaluation` values for the applications.

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

`NetworkEvaluator::evaluate` end to end — encode, mask, forward, softmax, scatter back — is a fixed
~2 ms per call above that, which is most of the cost at batch 1 and none of it by batch 256:

| batch | 1 | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| ms/position | 3.84 | 0.75 | 0.39 | 0.204 | 0.163 | 0.155 | 0.153 |
| positions/s | 260 | 1342 | 2559 | 4905 | 6118 | 6450 | 6537 |

This is the table the search should be read against, not `forward()` alone. It says a field of 256 is
worth 1.3× a field of 64 per position — which is exactly what the end-to-end self-play numbers above
fail to show, and why that row is still open.

At 1.2M parameters over 37 tokens this is dispatch-bound, not compute-bound, so **CPU beats Metal at
batch 1** and Metal wins from batch 8 up. It also flattens out early: 64 to 256 buys 6%, while 1 to 64
buys 10×. That is why self-play batches across games — a field of 256 costs the same per position as
a field of 64 but does four times as many games at once. Match play accepts batch-one inference to
keep the search itself sequential and easy to reason about.

`NetworkEvaluator::evaluate` and `makeTrainingBatch` encode and scatter their boards with direct
loops. Their spans remain batched interfaces; only the application-owned CPU parallelism was removed.

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

`amoeba/network.hpp` / `network.cpp` also carry `Batch`, `makeBatch()`, `loss()` and `Adam`. What is
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
- **Batch the network evaluations across games**, one leaf from each, not N guesses out of one
  search. At ~1-2M params over 37 tokens single-position inference is pure overhead, and the batch
  wants to be hundreds wide. This is what `Field` does.
- **Thread the repetition history through search.** `apply` only detects repetition when given the
  hash history; without it MCTS walks into draws it cannot see.
- **MLX is lazy** — nothing computes until `mx::eval()`. Easy to build a huge graph by accident, and
  it makes naive timing meaningless. Homebrew has `mlx` 0.32.0 (`/opt/homebrew/include/mlx`) plus the
  Python bindings.
- The MLX C++ API has no real optimizer or layer library, so Adam and the attention blocks are
  hand-written. If that becomes a time sink, training in Python + MLX while keeping engine, encoder
  and MCTS in C++ is a legitimate architecture, not a retreat.

## Next steps

1. **Measure the simple baseline.** Compare `CONCURRENT=64` and `CONCURRENT=256` on the same seeded
   games, and measure the bot's 1000 single-leaf simulations against the server's five-second limit.
   Use those results to decide which optimisation, if any, deserves to return.
2. **Add a permanent search fingerprint before optimising it again.** Over a whole game, the root's
   visit counts must sum to exactly `simulations`, illegal moves must have zero visits, and repeated
   noise-disabled runs must agree.
3. **Play a practice game against the live server.** Nothing here has ever touched it, and the parity
   harness that would have caught a rules disagreement is gone. If the engine and the server disagree,
   `syncToServer()` silently resyncs and a bad rating cannot be told apart from a bad network. Do this
   before spending days of compute.
4. **Run the loop overnight at the defaults** and count promotions in the morning. Several means the
   loop works and the rest is compute; zero means something is wrong and more compute will not fix
   it. That count, not a rating, is what the first night is for.
5. **Persist the replay buffer.** A restart currently throws away every game played and regenerates
   them, which is by far the most expensive thing the trainer does.
6. **Checkpoint Adam's moment estimates** so a generation can resume part way.

Run the whole pipeline end to end at a deliberately tiny scale first — that has been done once, at
1 block / width 32 / 4 games, and the loop promoted a generation. "It runs and the loss goes down"
proves almost nothing; AlphaZero bugs produce clean training curves. **The real milestone is a
generation beating the one before it over a few hundred gate games**, which is the only honest signal
in the whole system.

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

CMake + Ninja, build tree in `build.nosync/`.

```
cmake -S . -B build.nosync -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build.nosync
```

`find_package(MLX 0.32 CONFIG REQUIRED)` has to sit *after* `project()`: MLX imports itself as a
shared library, and before `project()` CMake has not yet established that the platform can link
one. The `arena` call above it predates that and only works because its import is static.

**Both programs require the `.safetensors` path.** Neither guesses: the trainer creates the file if it
is missing, the bot fails. Naming it is how you say which run you mean, and a directory scan made
resuming the wrong network — or playing rated games with one — a silent mistake. Point both at the same
file and the bot picks up each promotion at its next game.

```
# train forever at the overnight defaults, creating the file if it does not exist
./build.nosync/amoeba_train run7.safetensors

# ranked games back to back, for as long as the process lives
BOT1_ID=… BOT1_KEY=… ./build.nosync/amoeba_bot run7.safetensors

# one practice game
ROOM_ID=… BOT1_ID=… BOT1_KEY=… ARENA_DOMAIN=staging-game-arena.irvine.jp \
  ./build.nosync/amoeba_bot run7.safetensors
```

There are **no tests**. `amoeba_encode_test` and `amoeba_random_test` both existed and were both
deleted on request; git has them at `29dcd5e` if they are ever wanted back.
