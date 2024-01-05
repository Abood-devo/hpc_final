cmake_minimum_required(VERSION 3.9.1)
project( hpc_project )

set(CMAKE_CXX_STANDARD 14)

# -fopenmp flag (enables OpenMP)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -pedantic -lm -O3 -funroll-loops -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable( main main.cpp )

# opencv lib
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( main PUBLIC ${OpenCV_LIBS} MPI::MPI_CXX )

# MPI
find_package(MPI REQUIRED)
# target_link_libraries(main PUBLIC MPI::MPI_CXX)