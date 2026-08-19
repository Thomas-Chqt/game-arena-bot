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
