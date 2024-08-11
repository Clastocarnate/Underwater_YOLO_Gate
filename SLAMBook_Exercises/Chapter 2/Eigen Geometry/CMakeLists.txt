cmake_minimum_required(VERSION 2.8)
project( geometry )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

include_directories("/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")
add_executable(eigenGeometry eigenGeometry.cpp)