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
  mcts.hpp / .cpp        PUCT search, rollout evaluator, the global thread pool
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
- PUCT search with tree reuse across moves, Dirichlet root noise and a deadline; a rollout
  evaluator. One `ThreadPool` for the whole process.
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
a new position's value comes from a `bot::Evaluator`.

### The pieces

- `Evaluator` — `evaluate(span<const Board*>, span<Evaluation>)`, where `Evaluation` is a 444-float
  policy plus a value in `[-1, 1]`. The span form is deliberate even though the search asks one at a
  time: a single-position MLX forward pass wastes the device, and widening the interface later would
  touch every implementation.
- `RolloutEvaluator` — uniform priors, value from one uniformly random playout to the end of the
  game. Knows nothing about Amoeba, but it makes the search playable before the network exists and it
  is the baseline the network has to beat.
- `Search` — **owns no evaluator and calls none.** `restart(board, history)` starts a tree,
  `advance(moveId, board, history)` re-roots an existing one, then `pendingLeaves()` hands out the
  boards it cannot go on without and `absorb()` takes the answers back. `visits()` is the 444 visit
  counts; `bestMove()` is the argmax, and normalised the same array is the policy target.
- `runSearch(search, evaluator)` — the loop those four calls make, for a caller with one game.
  `bot.cpp` uses it; the trainer does not, because the whole point is to gather the batch across
  games instead.

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

### The deadline

`Config::deadline` stops the search on the clock as well as on the simulation count, whichever comes
first, and always runs one batch so the visit counts can never be empty. **Nothing sets it any
more** — both programs leave it at its hour default, so neither's play depends on how busy the machine
was. `bot.cpp` runs to 800 simulations and reports how long that took; the server's 5 s is not
enforced, and the elapsed time in the log is what says whether it needs to be.

The mechanism stays because that is a measurement, not a guarantee, and the moment a turn does have
to be bounded this is where the bound goes. **The clock starts when the search does**, not when the
root is set: `bot.cpp` re-roots the moment it hears the opponent's move and only searches when it is
asked to, and the wait in between must not count against a turn.

### Deliberately absent

- **Temperature sampling** — lives in `train.cpp`'s `chooseMove()` rather than in the search.
- **Batching leaves across games *inside* `Search`** — the search knows nothing about other games and
  should not. `Field` in `train.cpp` is what gathers across them.

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

### Batching across games, not across guesses

The evaluator is the whole cost of the search and it is dispatch-bound, so it has to be fed whole
batches. There are two places a batch can come from.

**From one search.** Descend N times before evaluating anything. Nothing makes the N descents
diverge on their own, so each is provisionally recorded on the way down as having come back a loss
(*virtual loss*), removed at back-up. It works, but every descent after the first chooses on
statistics that are a round stale, and at 400 simulations with 64 leaves that is only six rounds of
"look, learn, redirect" per move.

**From many games.** Keep hundreds of games in flight and take one leaf from each. The batch is the
size of the field, every descent sees its own tree exactly as the last round left it, and no virtual
loss is needed — this is what `train.cpp`'s `Field` does, at `CONCURRENT` games and `LEAVES=1`.

Both are still in `Search`: `Config::batchSize` is leaves per search per round, and virtual loss
engages only above 1. **Match play is the one caller that has no choice** — one game, so its batch
can only come from inside its own search, and it sets `batchSize = 16`.

Measured at 6 blocks / width 128 / 200 simulations, both runs from the same checkpoint file:

| | games | wall clock | positions | positions/s | CPU |
|---|---|---|---|---|---|
| one game per thread, 64 leaves, virtual loss | 64 | 66 s | 2362 | 35.8 | 65% |
| field of 64, 1 leaf each | 64 | 41 s | 2677 | 65.3 | 25% |
| field of 256, 1 leaf each | 512 | 720 s | 25926 | 36.0 | 11% |

**1.82× at a field of 64**, and the CPU load falls because ten threads are no longer each pushing
their own small batch through MLX — one driver thread issues one call per network per round.

**The field of 256 is the unexplained result, and `CONCURRENT` should not be trusted until it is
settled.** It came out at baseline speed, not better, even though the evaluator on its own is
measurably cheaper per position at 256 than at 64 (table under "Network" below). The two rows are not
a controlled comparison — different games, and the field of 64 drains instead of refilling — so what
they disagree about is *evaluations per ply*, not milliseconds per evaluation. The clean experiment,
which has not been run: `GAMES=256` with `CONCURRENT=64` and then `CONCURRENT=256`. Same game ids,
same seeds, therefore the same games and the same evaluation count, so the wall clock is the batching
effect and nothing else. If 64 wins there, lower the default.

### Tree reuse

`advance()` re-roots on the move played and keeps that child's subtree; `keepSubtree()` copies it
into scratch vectors, renumbering as it goes, so a 120-ply game does not accumulate 120 searches
worth of dead nodes. It falls back to `restart()` when the move played was never expanded, which is
what happens when the opponent plays something the search never looked at.

`Config::simulations` counts simulations **through the root**, inherited ones included, so a re-rooted
tree simply arrives with part of its budget already spent. Measured against the same games played
with `restart()` every ply: **26.7% of evaluations saved at 200 simulations**, and 46-50% at toy
counts where the tree is mostly one line deep.

The reused subtree is not an approximation of the search you would have run — it *is* that search,
paused. A descent inside a subtree reads only that subtree's own statistics, so the previous search's
descents into it are exactly the first N descents a fresh search would make. Checked: with a
deterministic evaluator, games played with reuse and without are the same game, move for move.

- **Self-play keeps one tree per game, not per colour.** Both sides are the same network, so the tree
  is re-rooted every ply instead of every other one and carries what both sides found. The gate keeps
  two, because a tree's statistics are worth exactly what the network that produced them is.
- **Path-dependence survives.** The path from the new root down is the path it always was, with one
  more ply of it now living in the game history, so every node's repetition-dependent verdict stays
  true. This is why there is still no transposition table.
- **Fresh Dirichlet noise goes on each new root.** What the subtree inherited are the network's own
  priors — noise only ever went on the root above it — and the handful of moves the last search was
  told to promote should not go on being promoted.
- **Fresh visit counts are not what the log reports.** `bot.cpp` prints the simulations behind the
  move as the sum of the root's counts, and that sum is the full `kSimulations` — part of it inherited
  from an earlier turn rather than run just now. Still 800 simulations behind the move, just not 800
  new ones.

### Verification

These numbers came from a `MODE=match` driver that no longer exists — the gate in `train.cpp` only
plays network against network. Over 20 games at 200 simulations, alternating colours:

```
rollout vs random             : 100.0% +/- 0.0%   (20-0-0)
network, 1 leaf   vs random   :  97.5% +/- 3.5%   (19-0-1)
network, 64 leaves vs random  : 100.0% +/- 0.0%   (20-0-0)
network, 64 leaves vs rollout :  40.0% +/- 11.0%  (6-10-4)
```

Over **200** games, network at 64 leaves against rollout scores **44.0% ± 3.5%** (64-88-48). That is
1.7 standard errors below parity — suggestive that the network is slightly weaker than its teacher, not
conclusive. Whether the virtual loss was responsible would be worth rerunning now that self-play
does not use it: those matches were played at 64 leaves out of one search.

**Before restructuring the descent again, check the invariants rather than the strength.** ±11% over
20 games cannot resolve a small regression, and the search has no offline test in the tree. What does
have teeth, and was run on the tree-reuse change: the root's visit counts sum to exactly
`simulations` at every ply of a whole game, they are zero on every move not legal in the position
searched, and with a deterministic evaluator a game played with `advance()` matches one played with
`restart()` move for move. All three fail loudly on a broken renumbering or a lost simulation.

## The thread pool

`ThreadPool` in `mcts.hpp`, one for the whole process, reached through `ThreadPool::global()`. It is
`hardware_concurrency()` wide counting the caller, and its only operation is
`forEach(count, body)` — run every index somewhere, return when they have all run.

Everything here is a burst of identical independent work with nothing to own threads between bursts:
advance 256 games one round, encode 256 boards, scatter 256 policies back. `train.cpp` no longer
starts a thread per game.

- **The calling thread takes indices too**, so a one-thread pool is a plain loop and no work waits on
  a thread that does not exist.
- **A `forEach` from inside a body runs serially.** The outer round already holds every thread, so
  waiting for one here would deadlock; a `thread_local` flag catches it. This is what lets
  `NetworkEvaluator::evaluate` use the pool without knowing who called it.
- **An exception from a body is rethrown to the caller** after the round finishes. `makeBatch` throws
  on a sample with no visits, from inside a body, and that has to reach the trainer rather than
  terminate.

## Trainer

`amoeba/train.cpp` -> `amoeba_train`. **The `.safetensors` path is a required argument** and is
created from random weights if it does not exist. It loops forever, and **writes back over that same
file whenever a generation passes the gate** — which is the
file `amoeba_bot` reads, so a bot running alongside picks the improvement up at its next game.

One generation is three steps: play `GAMES` games of the current best against itself into a replay
buffer, train a candidate on batches from that buffer, then play the candidate against the current
best and promote only if it wins.

**`Field` is where the games are played**, both for self-play and for the gate. It keeps `slots` games
in flight, refills a slot as its game ends, and drives them all in lockstep: one round is one
simulation per game, gathered into one network call per network. The whole field runs on the global
pool — one `forEach` over the slots per round — and the driver thread does the MLX call. See
"Batching across games" above for why.

- **`CONCURRENT` is twice the batch size.** The field is walked in two halves that take turns, so a
  field of 256 sends batches of 128. That costs 5% per position (0.163 ms against 0.155) and buys the
  overlap below, which is worth ~1.3×. **The default of 256 measured no faster than the old code while 64 measured 1.82× faster** —
  see "Batching across games" for the experiment that would explain it.
- **`GAMES` is above `CONCURRENT`, so a slot takes on another game instead of going idle.** What a
  game costs does not depend on when it is played: all 512 build their own tree from nothing and all
  512 get their own cheap endgame, so refill buys games rather than adding overhead. Set `GAMES` for
  how much data a generation should hold, not for scheduling — the schedule barely moves the number,
  because the late plies of a long game inherit most of their 400 simulations and cost a fraction of
  an opening ply, which is what makes a draining field far less wasteful than a naive
  cost-per-ply estimate suggests.
- **The tail is the one thing a full field does not fix.** A game can only take one simulation per
  round, so once the field is down to a handful they are each paying 0.75 ms a position instead of
  0.155 with nobody to share the call. `Config::batchSize` is the lever — `256 / games still playing`
  rather than a fixed 1 would keep the batch full to the end, at the cost of virtual loss on those
  last simulations. Not done.
- **The gate's field is smaller (`GATE_CONCURRENT`, 64)** because there the field size is also how long
  it takes to hear a verdict: `settled()` is only reconsulted as games come in, and 256 at once would
  spend most of `GATE_GAMES` before the first chance to stop. Once it does say stop, the games still in
  flight are abandoned rather than played out, so the tally is exactly what it was when it decided.
- **The gate needs two trees per game and two batches per round**, one per network. Only one side is to
  move at a time, so each batch is about half the field.
- **`finished` is called under the field's lock**, once per game, so the callers' tallies need no
  atomics. It is also where the one line a game gets is printed: how long it took, how many moves it
  played, how many network calls it cost, and how many games are left. **There is no per-round
  progress line, on purpose** — the games are never on the same move, so a field-wide "round" can only
  be an average, and reporting one reads as though the games move in lockstep at the move level. They
  do not: every game gets one simulation per round, so a game whose tree is nearly full plays its next
  move in ~50 rounds while one in the opening needs 400.
- **Everything random about a game comes from its id**, never from which slot or thread picked it up, so
  a run stays reproducible from `SEED` however the field interleaves.

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
shape comes out of the checkpoint), `SEED GAMES CONCURRENT SIMULATIONS LEAVES SAMPLING_PLIES NOISE`,
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
- **MLX is called from one thread at a time**, though not always the same one: the device call is task
  0 of the same `forEach` round as the tree work, so whichever thread picks it up spends the round
  inside MLX while the others walk trees. It used to be called from every game thread at once, which
  worked but left ten threads each pushing a small batch.
- **The device call and the tree work overlap.** A round used to be every descent with the device idle
  (~15 ms) and then one network call with every core idle (~40 ms). Now half the field's batch is on
  the device while the other half's trees are walked, so neither waits: ~42 ms per 256 positions
  against 55, about **1.3×**. Each half is evaluated on one turn and stepped on the next, which is
  what keeps an answer from being left unabsorbed. `NetworkEvaluator`'s own encode is a nested
  `forEach` and so runs serially inside that task — 0.3 ms for 128 boards, measured, against a device
  call of twenty.
- **When one half empties, it degrades to the old behaviour rather than to something worse.** The
  turns still alternate, but the empty half's turn costs nothing, so the surviving half pays
  `evaluate + step` unoverlapped — exactly what every round used to cost.

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
- **The bot keeps its own board, and the tree follows it.** `arena_game_state_t` carries a position
  and nothing else — no ply count, no history — so `syncToServer()` works out which legal move the
  opponent played and applies it locally. `advance()` re-roots the tree on it at the same time: a
  reply the search already looked at is a subtree it gets to keep, so reuse pays on the opponent's
  moves as well as our own. The resync fallback has to `restart()` instead, since the board it adopts
  is not one the tree has. Re-parsing the server's board each turn instead would silently reset `ply` and
  `staleness` to zero every ply, and would leave no hash history for the search to detect repetition
  with.
- **Two fallbacks, because forfeiting is worse than wrong bookkeeping.** If no legal move reaches the
  server's position, adopt that position and carry `ply` forward — that is a rules bug. If the search
  picks a move the server did not offer, play the server's first move.
- **The model is loaded once per game**, in `on_game_start`. Between games is the only safe moment:
  one tree scoring its positions with two different networks is incoherent and would not show up in
  any log. It is also how a promotion by a trainer running alongside gets picked up.
- **`kLeaves` is 16, not 1.** One game means the batch can only come from inside the one search, so
  virtual loss earns its keep here and nowhere else.
- **The search is count-bound, not deadline-bound.** `kSimulations` is 800 and there is no clock, so
  the move logged is always the one 800 simulations chose. Those are simulations *through the root*,
  so a re-rooted tree reaches 800 with fewer new evaluations and the reply lands sooner. Nothing
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

`amoeba/network.hpp` / `network.cpp`. `Network` holds the parameters, `forward()` is the
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
a field of 64 but does four times as many games at once, and never leaves a game waiting on stale
statistics — and why match play still batches leaves inside its one search rather than sending
positions one at a time.

Both `NetworkEvaluator::evaluate` and `makeBatch` encode their boards on the global thread pool. One
slice per position, no sharing, and at 256 boards the encode is otherwise measurable next to the
forward pass it feeds.

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

1. **Settle `CONCURRENT`.** `GAMES=256` at `CONCURRENT=64` against `CONCURRENT=256` — same games,
   same evaluation count, so the wall clock is the field size and nothing else. One measurement has
   the 256-wide field running at old-code speed while the 64-wide one runs at 1.82×, and the
   evaluator benchmark says that should be impossible. Something between the two is unaccounted for,
   and it is worth an hour before it is worth a night of compute.
2. **Fingerprint the search before touching it again.** There is no offline test of `Search` in the
   tree, and the two things most likely to break silently are `keepSubtree()`'s renumbering and the
   `pendingLeaves` / `absorb` handshake. The checks worth rerunning are cheap and were run on this
   change: over a whole game, the root's visit counts must sum to exactly `simulations` and must be
   zero on every move that is not legal in the position searched, and with a deterministic evaluator a
   game played with `advance()` must match one played with `restart()` move for move.
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
