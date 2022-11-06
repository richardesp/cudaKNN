cmake_minimum_required(VERSION 3.23)
project(cudaKNN CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cudaKNN main.cu include/Dataset.h include/Lock.h)

set_target_properties(cudaKNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
