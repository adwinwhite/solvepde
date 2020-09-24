cmake_minimum_required(VERSION 3.5)

project(solvepde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(EIGEN_LIBRARY NAMES eigen)
add_executable(solvepde  PropagateGaussianWavePacket.cpp)
