cmake_minimum_required(VERSION 3.5)

project(solvepde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(EIGEN_LIBRARY NAMES eigen)
add_executable(PropagateGaussianWavePacket  PropagateGaussianWavePacket.cpp)
target_compile_options(PropagateGaussianWavePacket PRIVATE -O3 -fopenmp -ffast-math -march=native -mtune=native)
target_link_options(PropagateGaussianWavePacket PRIVATE -lgomp)

#find_package(Python3 COMPONENTS Development NumPy)
#target_include_directories(fft PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
#target_link_libraries(fft Python3::Python Python3::NumPy)
