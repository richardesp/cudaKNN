cmake_minimum_required(VERSION 3.23)
project(cudaKNN CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(cudaKNN main.cu include/Dataset.h include/cuda/Lock.h include/Point.h include/Label.h src/cuda/cudaKNN.cu)

set_target_properties(cudaKNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
