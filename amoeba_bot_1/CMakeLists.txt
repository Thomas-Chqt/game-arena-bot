add_executable(amoeba_bot_1 main.cpp)
target_compile_features(amoeba_bot_1 PRIVATE cxx_std_23)
target_link_libraries(amoeba_bot_1 PRIVATE games::amoeba arena::arena)
