add_executable(amoeba_random_test random_test.cpp)
target_compile_features(amoeba_random_test PRIVATE cxx_std_23)
target_link_libraries(amoeba_random_test PRIVATE games::amoeba arena::arena)
