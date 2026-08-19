# Amoeba — Complete Reference

Everything needed to play Amoeba on Game Arena: the exact rules as the server
implements them, the wire/SDK API, and a verified reference move generator you
can search with.

Authoritative sources this document mirrors:

- Engine: `backend/game-server/app/game/amoeba.py`
- Wire protocol: `backend/game-server/app/websocket/{types,handlers,room}.py`
- Tunables: `backend/shared/config.py`
- Python SDK: `sdk/python/`, C SDK: `sdk/c/include/arena/arena.h`

---

## 1. One-minute summary

Amoeba is a two-player perfect-information game on a radius-3 hex board (37
cells). Pieces form **stacks**. Whoever owns the **top** piece of a stack
controls the whole stack and may move it. Each side has exactly one **kernel**
(`WK` / `BK`) — the king. You lose when your kernel is captured, and, in
practice, you lose when you run out of legal moves (Amoeba's checkmate).

A stack always travels **exactly as many hexes as it is tall**, in one of the six
hex directions, jumping over everything in between. There are two ways to do it:

- **move** — the whole stack lands on the destination hex, on top of whatever is
  there.
- **sow** (`splitting = true`, only for height ≥ 2) — the stack is dealt out one
  piece per hex along the path, **bottom piece first**, top piece landing on the
  final hex.

Nothing is ever removed from the board. "Capture" only means *taking control* by
putting your piece on top.

---

## 2. Board and coordinates

Axial hex coordinates `q,r`. A cell is on the board when
`max(|q|, |r|, |q+r|) <= 3`. That is 37 cells.

```
         0,-3   1,-3   2,-3   3,-3
      -1,-2  0,-2   1,-2   2,-2   3,-2
   -2,-1  -1,-1  0,-1   1,-1   2,-1   3,-1
-3,0   -2,0   -1,0   0,0    1,0    2,0    3,0
   -3,1   -2,1   -1,1   0,1    1,1    2,1
      -3,2   -2,2   -1,2   0,2    1,2
         -3,3   -2,3   -1,3   0,3
```

Coordinates are always strings of the form `"q,r"` on the wire: `"0,0"`,
`"1,-1"`, `"-2,1"`. No spaces, no padding.

### The six directions

| Name | `dq, dr` | Notes |
|------|----------|-------|
| `E`  | `+1, 0`  | right |
| `W`  | `-1, 0`  | left |
| `NE` | `+1, -1` | up-right (toward Black's home) |
| `NW` | `0, -1`  | up-left |
| `SE` | `0, +1`  | down-right (toward White's home) |
| `SW` | `-1, +1` | down-left |

Decreasing `r` goes "north". Black starts at low `r`, White at high `r`.

The three straight lines through a cell are `r` constant (E/W), `q` constant
(NW/SE), and `q+r` constant (NE/SW). Two cells are on a common line iff
`dr == 0`, `dq == 0`, or `dq == -dr`. Distance along that line is
`max(|dq|, |dr|)`.

The board is convex, so if a path's endpoint is on the board, every hex between
is too.

---

## 3. Pieces and the initial position

Four piece codes:

| Code | Owner | Role |
|------|-------|------|
| `W`  | White | Standard |
| `WK` | White | Kernel (White's king) |
| `B`  | Black | Standard |
| `BK` | Black | Kernel (Black's king) |

Ownership is the first letter. Each side starts with 10 standard pieces and 1
kernel — 22 pieces, all on separate cells, no stacks.

```
      B  B  B  B          r = -3   (0,-3) (1,-3) (2,-3) (3,-3)
    .  .  BK .  .         r = -2   BK at (1,-2)
  B  B  B  B  B  B        r = -1   (-2,-1)…(3,-1)
.  .  .  .  .  .  .       r =  0   empty
  W  W  W  W  W  W        r =  1   (-3,1)…(2,1)
    .  .  WK .  .         r =  2   WK at (-1,2)
      W  W  W  W          r =  3   (-3,3)…(0,3)
```

Opening board string:

```
-3,1:W;-3,3:W;-2,-1:B;-2,1:W;-2,3:W;-1,-1:B;-1,1:W;-1,2:WK;-1,3:W;0,-3:B;0,-1:B;0,1:W;0,3:W;1,-3:B;1,-2:BK;1,-1:B;1,1:W;2,-3:B;2,-1:B;2,1:W;3,-3:B;3,-1:B
```

**White moves first.**

---

## 4. Stacks

A cell holds an ordered list of pieces, **bottom first**. `0,0:WBW` means: `W`
on the bottom, `B` in the middle, `W` on top.

- The **top** piece's owner **controls** the stack. Only the controller may move
  it.
- Stack **height** = number of pieces. Height is the only thing that determines
  movement distance.
- Pieces are never removed, never flipped, never captured off the board. Control
  changes purely by who ends up on top.
- A kernel buried under friendly pieces is safe. A kernel with an enemy piece
  anywhere above it in the stack is *not* automatically lost — only the **top**
  piece matters.

---

## 5. Movement

Let a stack at `(q,r)` have height `h`, and a direction `(dq,dr)`. Define

```
path = [ (q + dq*k, r + dr*k) for k in 1..h ]
dest = path[-1]                       # exactly h hexes away
```

The move is geometrically available iff `dest` is on the board. (Convexity makes
every intermediate hex automatically on the board.) Intervening pieces never
block anything — stacks fly over them.

### 5.1 `move` (`splitting = false`)

The entire stack is lifted and dropped on `dest`, preserving internal order, on
top of whatever already sits there. The origin becomes empty.

```
before:  -3,0:WBW              (height 3, White controls: top is W)
action:  move E from -3,0
after:   0,0:WBW               ← landed 3 hexes east, origin now empty
```

If `dest` held `-3,0:X`, the result is `X` + `WBW` with `W` on top.

### 5.2 `sow` (`splitting = true`, requires `h >= 2`)

The stack is dealt out one piece per hex along `path`: `stack[0]` (the bottom
piece) on `path[0]`, `stack[1]` on `path[1]`, …, `stack[h-1]` (the top piece) on
`path[h-1] == dest`. Each dropped piece goes **on top** of whatever is already on
that hex. The origin becomes empty.

```
before:  -3,0:WBW              (height 3, White controls)
action:  sow E from -3,0
after:   -2,0:W ; -1,0:B ; 0,0:W
```

Note what happened: the `B` that White was sitting on got **handed back to
Black**, now controlling `-1,0` on its own. Sowing a mixed stack releases the
enemy pieces underneath you. This is the single most common way bots throw away
material.

### 5.3 Both actions share a destination

Critical: for a stack of height ≥ 2, `move` and `sow` in the same direction have
the **identical** `to_pos`. `from_pos` + `to_pos` alone do not identify a move.
The `splitting` flag is what disambiguates them. Omitting it is the #1 cause of
`INVALID_MOVE` in Amoeba.

---

## 6. Legality, check, and mate

### 6.1 Check

`side`'s kernel is **in check** iff there exists *any* geometrically available
opponent action (move or sow, ignoring whether it is itself legal) that results
in a board where the opponent controls a stack containing `side`'s kernel.

Concretely, an opponent stack at distance `d` on a common line, with height `h`,
whose destination is on the board, threatens your kernel when:

- `d == h` — the whole stack lands on your kernel's cell, opponent's top piece on
  top; or
- `d < h` and the opponent's `stack[d-1]` is an opponent-owned piece — a sow
  drops exactly that piece on your kernel's cell.

Kernels are ordinary attackers too: a lone `BK` threatens all six of its
neighbours.

### 6.2 Legal moves

A geometrically available action is **legal** iff the resulting position does
**not** leave your own kernel in check. This is filtered server-side; the
`legal_moves` you receive are already safe.

Consequences worth internalising:

- You must resolve check, exactly like chess.
- **Resolving check outranks winning.** A move that captures the enemy kernel is
  still illegal if it leaves your own kernel capturable. Verified:

  ```
  board:  0,0:WK ; 1,0:B ; -2,0:BK ; -3,0:W      White to move
  ```
  White's `-3,0 → -2,0` would capture `BK`, but the `B` on `1,0` still threatens
  `WK`, so that move is **not** in White's legal list. White's only legal moves
  are kernel escapes: `0,0 → 1,0`, `0,0 → 0,-1`, `0,0 → -1,1`.

- Because every legal move leaves your kernel un-capturable, `kernel-capture`
  is unreachable in a correctly played game. A mate arrives as `no-move`
  instead (see below). The terminal condition exists in the engine but never
  fires in practice — do not build your search around reaching it; build it
  around leaving the opponent with zero legal moves.

- A sow is often illegal even when the matching `move` is legal, because sowing
  drops enemy pieces near your own kernel. Never assume both variants exist.

---

## 7. Terminal conditions

Reported as a reason string (visible in replays / `end_reason`, not in the SDK
game-end callback).

| Reason | Trigger | Result |
|--------|---------|--------|
| `no-move` | The side to move has zero legal moves | The **other** side wins. This is checkmate — it is not a stalemate draw. |
| `kernel-capture` | Mover controls a stack containing the enemy kernel after their move | Mover wins. Unreachable in legal play (§6.2). |
| `draw-repetition` | The same position (stack layout **and** side to move) occurs for the 3rd time | Draw |
| `stack-control-staleness` | 80 consecutive moves without a landing capture | Adjudicated (below) |
| `stack-control-move-cap` | 250 total moves played | Adjudicated (below) |

Beyond the game rules, the room layer can also end a match by disconnect or
turn timeout (forfeit).

### 7.1 The staleness counter

Reset to 0 when the move's **final landing cell** held a stack whose top piece
belonged to the opponent (i.e. the landing seized an enemy-controlled stack).
Otherwise +1. Note the asymmetry: intermediate captures made during a sow do
**not** reset it — only the last hex counts. Threshold: `GAME_STALENESS_AMOEBA`
= 80.

### 7.2 Adjudication (tiebreak)

Applied for both `stack-control-*` reasons, in order:

1. Most **controlled stacks** (stacks whose top piece is yours) wins.
2. Tie → most **opponent-owned pieces sitting inside your controlled stacks**
   wins (you are holding more of their material prisoner).
3. Still tied → draw.

This gives you a concrete evaluation target for long games: maximise stacks you
top, and prefer topping stacks that are full of enemy pieces.

### 7.3 What random play actually produces

300 self-play games with uniformly random legal move selection:

| Reason | Games |
|--------|-------|
| `no-move` | 293 |
| `stack-control-move-cap` | 5 |
| `draw-repetition` | 1 |
| `stack-control-staleness` | 1 |
| `kernel-capture` | 0 |

Game length: min 6, median 75, mean 84, max 250 moves. White won 158, Black 141,
1 draw. Mate is by far the dominant outcome, and it can happen very early — a
6-move loss is possible.

---

## 8. Board string format

`state.board` is a single compact string:

```
"q,r:CODES" entries, joined by ";", sorted ascending by (q, r)
```

- Empty cells are simply absent.
- `CODES` is the stack concatenated **bottom to top**.
- Tokens are `W`, `B`, `WK`, `BK`. Parsing is unambiguous: read one letter; if
  the next character is `K`, it belongs to that token.
- Empty board → empty string.

Examples:

| String | Meaning |
|--------|---------|
| `0,0:W` | single White piece at the center |
| `0,0:WB` | `W` bottom, `B` top → **Black** controls, height 2 |
| `1,0:BKW` | Black's kernel with a White piece on top → White has captured it |
| `-1,2:WK;1,-2:BK` | just the two kernels |

---

## 9. The move API

### 9.1 What the server sends you

`legal_moves` is a list of grouped entries, one per `(origin, splitting)` pair:

```json
{ "pos": "0,0", "name": "Standard", "valid_moves": ["2,0", "2,-2", "0,-2"], "splitting": false }
```

- `pos` — origin coordinate.
- `valid_moves` — legal destination coordinates for that origin **with that
  `splitting` value**.
- `name` — `"Standard"` or `"Kernel"`, based on the top piece. Informational.
- `splitting` — `false` for whole-stack moves, `true` for sows. Always present
  for Amoeba (never `null`).

**A single origin can appear twice** — once with `splitting: false` and once with
`splitting: true` — usually with overlapping but *not identical* `valid_moves`.
Treat each entry as its own bucket. Real example for a height-2 Black stack:

```
{ "pos": "0,0", "valid_moves": ["2,0","2,-2","0,-2","-2,0","-2,2","0,2"], "splitting": false }
{ "pos": "0,0", "valid_moves": ["2,0","2,-2",        "-2,0","-2,2","0,2"], "splitting": true  }
```

`0,-2` is missing from the sow list because that sow would drop a White piece
next to Black's own kernel.

Entries are ordered by `(q, r, splitting)`. `legal_moves` is `[]` once the game
is over.

### 9.2 What you send back

```json
{ "from_pos": "0,0", "to_pos": "2,0", "splitting": true }
```

- `from_pos` — copy `piece.pos` verbatim.
- `to_pos` — one string from that entry's `valid_moves`.
- `splitting` — copy `piece.splitting` verbatim. **Do not infer it.**
- `side` — FlipFour only; leave unset for Amoeba.

Anything not matching a legal `(from, to, splitting)` triple is rejected with
`INVALID_MOVE`. Invalid moves do not reset your turn deadline.

### 9.3 Python SDK

```bash
pip install game-ai-arena-sdk
```

```python
import random
from game_ai_arena_sdk import Bot, GameType, Move, GameStateLoop, start

class MyBot(Bot):
    async def on_move(self, state: GameStateLoop) -> Move:
        piece = random.choice(state.legal_moves)
        dest = random.choice(piece.valid_moves)
        return Move(from_pos=piece.pos, to_pos=dest, splitting=piece.splitting)

bot = MyBot(bot_id="YOUR_BOT_ID", api_key="YOUR_API_KEY")
start(bot, GameType.AMOEBA)
```

Types (`sdk/python/types.py`):

```python
GameType.AMOEBA == "amoeba"
PlayerSide.WHITE == "white"; PlayerSide.BLACK == "black"

class GameStateLoop:      # passed to on_move / on_game_start
    board: str
    current_turn: PlayerSide
    my_side: PlayerSide
    legal_moves: list[PieceMovesInfo]

class GameStateEnd:       # passed to on_game_end
    board: str
    current_turn: PlayerSide
    my_side: PlayerSide
    winner: PlayerSide | None      # None = draw

class PieceMovesInfo:
    name: str
    pos: str
    valid_moves: list[str]
    splitting: bool | None

class Move:
    from_pos: str
    to_pos: str
    side: str | None = None        # FlipFour only
    splitting: bool | None = None  # Amoeba
```

Entry points:

| Function | Use |
|----------|-----|
| `start(bot, GameType.AMOEBA)` | one ranked match, blocking |
| `run(bot, GameType.AMOEBA)` | same, coroutine form |
| `start_practice(bot, room_id=...)` | one practice match, no ELO |
| `run_practice(bot, room_id=...)` | coroutine form |
| `start_continuous(bot)` | interactive continuous ranked runner |
| `run_continuous(bot, GameType.AMOEBA)` | headless continuous, returns `RunnerStatus` |
| `ContinuousRunner` | programmatic `run/pause/resume/status/stop/wait` |

Lifecycle hooks on `Bot` — only `on_move` is required:

```
on_ready() → on_queue_entry() → on_queue_exit() → on_match_found(game_id)
→ on_room_joined(room_id) → on_game_start(state) → on_move(state)*
→ on_game_end(winner, state) → on_match_finished(result)
on_disconnect(reason)   # your own connection only
```

`on_game_end`'s `winner` is `"white"`, `"black"`, or `None` for a draw. An
opponent disconnect surfaces as a normal `on_game_end` win, not as
`on_disconnect`.

### 9.4 C SDK

`sdk/c/include/arena/arena.h`, built from source.

```c
#include <arena/arena.h>

static void on_move(const arena_game_state_t *state,
                    arena_move_t *move_out, void *user_data) {
    const arena_piece_moves_t *p = &state->legal_moves[0];
    move_out->from_pos  = p->pos;
    move_out->to_pos    = p->valid_moves[0];
    move_out->side      = NULL;              /* FlipFour only */
    move_out->splitting = p->splitting;      /* copy it through */
}

int main(void) {
    arena_bot_config_t cfg = {
        .bot_id  = "YOUR_BOT_ID",
        .api_key = "YOUR_API_KEY",
        .callbacks = { .on_move = on_move },
    };
    return arena_start(&cfg, ARENA_GAME_AMOEBA);
}
```

`arena_piece_moves_t` carries `has_splitting` (whether the field was present on
the wire) alongside `splitting`. Entry points: `arena_start`,
`arena_start_continuous`, `arena_start_practice`.

### 9.5 Raw WebSocket protocol

Only needed to write another SDK. Endpoints derive from `ARENA_DOMAIN`
(default `game-arena.irvine.jp`): matchmaker `wss://HOST/matchmaking/ws`, game
room `wss://HOST/arena`. Every frame is `{"type": ..., "data": {...}}`.

Matchmaker:

```
S→C  connected
C→S  join_queue          {"game_type": "amoeba"}
S→C  queue_joined | queue_position | queue_left | error | pong
S→C  match_found         {"game_id", "your_side", "game_server_ws", "arena_admission_token", ...}
```

Connect to `game_server_ws` with header
`Authorization: Arena <arena_admission_token>` (short-lived, secret, never log
it), then:

```
C→S  join_room     {"room_id": "<game_id>"}
S→C  room_joined   {"room_id", "your_side"}
S→C  game_start    {"room_id", "game_type", "your_side", "board", "current_turn",
                    "legal_moves", "in_check",
                    "turn_timeout_enabled", "turn_timeout_seconds", "turn_timeout_grace_seconds"}
C→S  submit_move   {"from_pos", "to_pos", "splitting"}
S→C  move_made     {"room_id", "move", "move_number", "board", "current_turn",
                    "status", "legal_moves", "in_check", ...timeout metadata}
S→C  game_end      {"room_id", "winner", "board", "current_turn", "forfeit"}
S→C  opponent_disconnected {"room_id", "message"}
S→C  error         {"code", ...}      e.g. INVALID_MOVE, ROOM_NOT_FOUND, ROOM_CLOSED
```

`in_check` is Amoeba-specific and refers to **the side to move**, not to the
recipient. `legal_moves` in every message is computed per-recipient. The Python
and C SDKs hide `room_id`, `move_number`, `status`, `in_check`, and `forfeit`.

### 9.6 Timing

Server-authoritative; local countdowns are display only.

| Setting | Default |
|---------|---------|
| Room join deadline (ranked) | 30 s |
| First accepted move | 10 s |
| Subsequent moves | 5 s |
| Server grace | 0.5 s |
| Practice join window | 600 s, normally untimed play |

Timeout only produces a forfeit result once **both** sides have made one
accepted move; before that it closes the room as a setup error with no winner,
no ELO, no replay. Invalid moves do not reset the deadline — budget your search
against a **5 second** wall clock and always have a fallback move ready.

---

## 10. Reference implementation

Drop-in, dependency-free move generator. Differentially tested against the
server engine over 800 random positions (legal-move sets, check detection,
board-string round-trip) and 60 full random self-play games (board state after
every move) — zero mismatches.

```python
DIRECTIONS = {"E": (1, 0), "NE": (1, -1), "NW": (0, -1),
              "W": (-1, 0), "SW": (-1, 1), "SE": (0, 1)}
RADIUS = 3


def in_bounds(q, r):
    return max(abs(q), abs(r), abs(q + r)) <= RADIUS


def parse_board(text):
    board = {}
    for entry in filter(None, text.split(";")):
        coord, codes = entry.split(":")
        q, r = (int(n) for n in coord.split(","))
        stack, i = [], 0
        while i < len(codes):
            if codes[i + 1:i + 2] == "K":
                stack.append(codes[i:i + 2]); i += 2
            else:
                stack.append(codes[i]); i += 1
        board[(q, r)] = stack
    return board


def encode_board(board):
    return ";".join(f"{q},{r}:{''.join(s)}" for (q, r), s in sorted(board.items()) if s)


def owner(piece):
    return "white" if piece[0] == "W" else "black"


def candidate_moves(board, side):
    """Every geometrically available action, before the self-check filter."""
    for (q, r), stack in board.items():
        if not stack or owner(stack[-1]) != side:
            continue
        height = len(stack)
        for name, (dq, dr) in DIRECTIONS.items():
            path = [(q + dq * k, r + dr * k) for k in range(1, height + 1)]
            if not in_bounds(*path[-1]):
                continue
            yield ("move", (q, r), name, path)
            if height > 1:
                yield ("sow", (q, r), name, path)


def apply_move(board, action, origin, path):
    new = {c: s[:] for c, s in board.items()}
    moving = new.pop(origin)
    if action == "move":
        new.setdefault(path[-1], []).extend(moving)
    else:
        for piece, cell in zip(moving, path):
            new.setdefault(cell, []).append(piece)
    return new


def controls_enemy_kernel(board, side):
    kernel = "BK" if side == "white" else "WK"
    return any(s and kernel in s and owner(s[-1]) == side for s in board.values())


def in_check(board, side):
    foe = "black" if side == "white" else "white"
    return any(controls_enemy_kernel(apply_move(board, a, o, p), foe)
               for a, o, _, p in candidate_moves(board, foe))


def legal_moves(board, side):
    out = []
    for action, origin, direction, path in candidate_moves(board, side):
        if not in_check(apply_move(board, action, origin, path), side):
            out.append({"action": action, "from": origin, "direction": direction,
                        "to": path[-1], "path": path, "splitting": action == "sow"})
    return out
```

Wiring it to the SDK:

```python
async def on_move(self, state: GameStateLoop) -> Move:
    board = parse_board(state.board)
    best = max(legal_moves(board, state.my_side.value), key=self.evaluate)
    q, r = best["to"]
    return Move(from_pos=f'{best["from"][0]},{best["from"][1]}',
                to_pos=f"{q},{r}",
                splitting=best["splitting"])
```

`legal_moves` is the naive O(my_moves × their_moves) formulation — roughly 100 ×
100 board copies per call, fine for shallow search. For deeper search, replace
`in_check` with the direct line test in §6.1 (walk the three lines out of the
kernel's cell) and use copy-on-write stacks, which is what the server does.

---

## 11. Playing well

Mechanical consequences, in rough order of importance:

1. **Mate is the win condition.** `kernel-capture` never fires; you win by
   leaving the opponent with an empty legal-move list. Search for positions
   where the opponent's kernel is in check and every escape is also attacked.
   Random games mate in a median of 75 moves and sometimes in 6 — mate threats
   are live from the opening.
2. **Height is range, and range is a liability.** A tall stack must move
   *exactly* its height. As `h` grows, most directions run off the board and the
   stack's legal destinations collapse. A height-5 stack near an edge can be
   nearly immobile — and immobile pieces contribute to a `no-move` loss.
3. **Never sow a mixed stack carelessly.** Sowing hands every enemy piece
   beneath you back to the enemy, each one landing as a controlled height-1
   stack on a hex you chose. Prefer `move` when your stack is impure; prefer
   `sow` to spread your own material or to seed cells you want to control.
4. **Sowing is a multi-hex capture.** With a pure stack, a sow can seize control
   of up to `h` different cells in one action — the strongest tempo move in the
   game. Only the last one resets the staleness counter, though.
5. **Attack geometry.** You threaten a cell at distance `d` on a common line if
   `d == h` (whole stack lands) or `d < h` and your `stack[d-1]` is your own
   piece (sow drops it there). Tall stacks therefore threaten a *range* of
   distances, not one — but only through pieces they actually own at those
   depths. Track your own stack composition, not just its height.
6. **Bury your kernel, but keep it mobile.** Stacking your own pieces on top of
   your kernel makes the cell safe (only the top matters) — but the whole stack
   then moves at the taller distance, and if it gets trapped you may run out of
   moves.
7. **Kernels attack.** A lone kernel threatens all six neighbours; it is a
   perfectly good attacking piece, subject to never stepping into check.
8. **Endgame evaluation = adjudication rule.** If a game is heading toward move
   250 or 80 quiet moves, the winner is decided by controlled-stack count, then
   by opponent pieces held inside your stacks. Bias your evaluation toward
   controlling *many* stacks rather than one tall one.
9. **Watch repetition.** Three occurrences of the same (layout, side-to-move)
   is an immediate draw. If you are winning, break the cycle; if losing, seek it.

---

## 12. Constants

| Constant | Value | Env override |
|----------|-------|--------------|
| Board radius | 3 (37 cells) | — |
| Pieces per side | 10 standard + 1 kernel | — |
| First to move | White | — |
| Repetition draw | 3rd occurrence | — |
| Staleness threshold | 80 moves without a landing capture | `GAME_STALENESS_AMOEBA` |
| Move cap | 250 moves | `GAME_STALENESS_AMOEBA_MOVE_CAP` |
| Join deadline | 30 s | `GAME_ROOM_JOIN_TIMEOUT_SECONDS` |
| First move | 10 s | `GAME_INITIAL_MOVE_TIMEOUT_SECONDS` |
| Later moves | 5 s | `GAME_TURN_TIMEOUT_SECONDS` |
| Grace | 0.5 s | `GAME_TURN_TIMEOUT_GRACE_SECONDS` |
| Game type string | `amoeba` | — |
| Starting ELO | 1000 | — |

---

## 13. Failure checklist

| Symptom | Cause |
|---------|-------|
| `INVALID_MOVE` intermittently, fine for many moves | `splitting` omitted or hardcoded `false`; it only bites when a sow is chosen |
| `INVALID_MOVE` on a destination you computed | You paired a `to_pos` from the `splitting: false` entry with `splitting: true`, or vice versa |
| Move rejected as illegal though geometry is fine | It left your own kernel in check |
| Losing on time | Search exceeded 5 s; invalid moves do not reset the clock |
| `GAME_TYPE_MISMATCH` | Bot is registered for a different game than `GameType.AMOEBA` |
| Never matched | Opponent must be a *different user's* bot in the same queue |
| Immediate disconnect | Bad/regenerated API key, unverified account, inactive bot |
| Practice join fails | Room timed out; create a new one and use the new `room_id` |
