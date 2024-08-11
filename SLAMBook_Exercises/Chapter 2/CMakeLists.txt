# Specify the minimum required version of CMake that can be used to process this file.
# In this case, it's version 2.8 or higher.
cmake_minimum_required(VERSION 2.8)

# Define the name of the project. Here, the project is named "useEigen".
project(useEigen)

# Set the build type to "Release". This enables optimizations in the build process.
# This is typically used for production-ready builds where performance is crucial.
set(CMAKE_BUILD_TYPE "Release")

# Set the C++ compiler flags for the build. 
# In this case, the flag "-O3" is used, which tells the compiler to optimize the code for maximum speed.
# It's a highly aggressive optimization level that might increase compilation time but should improve runtime performance.
set(CMAKE_CXX_FLAGS "-O3")

# Specify the directory where the Eigen library's header files are located.
# This tells the compiler where to find the Eigen headers when compiling the code.
# The path provided here is specific to the system where Eigen is installed; it needs to match where Eigen is installed on your system.
include_directories("/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")

# Define an executable target named "eigenMatrix".
# This tells CMake to compile the source file "eigenMatrix.cpp" and link it into an executable named "eigenMatrix".
add_executable(eigenMatrix eigenMatrix.cpp)
