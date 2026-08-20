cmake_minimum_required(VERSION 3.22)

project(amoeba VERSION 0.1.0 LANGUAGES CXX)

set(GAME_ARENA_SDK_ROOT "$ENV{HOME}/.local" CACHE PATH "Prefix where the Game Arena C SDK is installed")
find_package(arena 0.1.0 CONFIG REQUIRED HINTS "${GAME_ARENA_SDK_ROOT}")

set(MLX_ROOT "/opt/homebrew" CACHE PATH "Prefix where MLX is installed")
find_package(MLX 0.32 CONFIG REQUIRED HINTS "${MLX_ROOT}")

# Both executables share the game engine, search, and network. Only the match bot
# links the Game Arena SDK; the trainer never communicates with the server.
set(AMOEBA_SOURCES
    amoeba/amoeba.cpp
    amoeba/mcts.cpp
    amoeba/network.cpp
)

add_executable(amoeba_bot amoeba/bot.cpp ${AMOEBA_SOURCES})
target_compile_features(amoeba_bot PRIVATE cxx_std_23)
target_link_libraries(amoeba_bot PRIVATE arena::arena mlx)

add_executable(amoeba_train amoeba/train.cpp ${AMOEBA_SOURCES})
target_compile_features(amoeba_train PRIVATE cxx_std_23)
target_link_libraries(amoeba_train PRIVATE mlx)
