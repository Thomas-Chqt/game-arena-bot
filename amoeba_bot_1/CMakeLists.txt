# Two programs out of one directory: amoeba_bot_1 plays on Game Arena,
# amoeba_bot_1_train does self-play and training. Only the player links the
# arena SDK - self-play never talks to the server, and keeping the SDK off the
# trainer is what enforces that.
#
# network.cpp does not exist yet; it arrives with the MLX model, and both
# targets fail to link until it does.
add_library(amoeba_bot_1_core STATIC mcts.cpp checkpoints.cpp)
target_compile_features(amoeba_bot_1_core PUBLIC cxx_std_23)
target_include_directories(amoeba_bot_1_core PUBLIC .)
target_link_libraries(amoeba_bot_1_core PUBLIC games::amoeba)

add_executable(amoeba_bot_1 play.cpp)
target_link_libraries(amoeba_bot_1 PRIVATE amoeba_bot_1_core arena::arena)

add_executable(amoeba_bot_1_train train.cpp)
target_link_libraries(amoeba_bot_1_train PRIVATE amoeba_bot_1_core)
