cmake_minimum_required(VERSION 3.22)

set(GAME_ARENA_SDK_ROOT "$ENV{HOME}/.local" CACHE PATH "Prefix where the Game Arena C SDK is installed")
find_package(arena 0.6 CONFIG REQUIRED HINTS "${GAME_ARENA_SDK_ROOT}")

project(game_arena_bots VERSION 0.1.0 LANGUAGES CXX)

add_subdirectory(games/amoeba)
add_subdirectory(amoeba_bot_1)
