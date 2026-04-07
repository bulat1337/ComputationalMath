cmake_minimum_required(VERSION 3.10)
project(solve_tests C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SOURCES
    solve.c
    tests.c
)

add_executable(tests ${SOURCES})

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tests PRIVATE -Wall -Wextra -Wno-gnu-folding-constant)
endif()

target_link_libraries(tests PRIVATE m)
