cmake_minimum_required(VERSION 3.28)
project(prove2)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_FLAGS "-march=armv8-a+sve -g -Wall -Wextra")
set(CMAKE_CXX_FLAGS "-march=armv8-a+sve -g -Wall -Wextra -fno-exceptions")

#find_package(benchmark REQUIRED)

add_executable(test_exp test_exp.cpp
        Benchmark.cpp
        Benchmark.h)
add_executable(test_log test_log.cpp
        Benchmark.cpp
        Benchmark.h)
add_executable(test_tanh test_tanh.cpp
        Benchmark.cpp
        Benchmark.h)

#target_link_libraries(test_exp benchmark::benchmark)
add_executable(expa expa.cpp)
