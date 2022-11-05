#include <iostream>
#include <cuda_runtime.h>
#include "include/Dataset.h"

#define TOTAL_BLOCKS 1
#define TOTAL_THREADS_AXIS_X 1024
#define TOTAL_THREADS_AXIS_Y 1024
#define TOTAL_THREADS_AXIS_Z 64

int main(int argc, char **argv) {
    int deviceCount = 0;

    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) { // Throw a runtime exception
        std::cerr << "No CUDA devices found" << std::endl;
        return EXIT_FAILURE;
    }

    int bestDevice = 0;
    cudaDeviceProp bestDeviceProp;

    for (int i = 0; i < deviceCount; i++) {

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Global memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Memory pitch: " << deviceProp.memPitch << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions: " << deviceProp.maxThreadsDim[0] << " " << deviceProp.maxThreadsDim[1]
                  << " " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Max grid dimensions: " << deviceProp.maxGridSize[0] << " " << deviceProp.maxGridSize[1] << " "
                  << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem << std::endl;
        std::cout << "  Texture alignment: " << deviceProp.textureAlignment << std::endl;
        std::cout << "  Concurrent copy and execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Kernel execution timeout: " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No")
                  << std::endl;

        if (i == 0 || bestDeviceProp.totalGlobalMem < deviceProp.totalGlobalMem) {
            bestDevice = i;
            bestDeviceProp = deviceProp;
        }

    }

    std::cout << "\n################################################\n" << std::endl;
    std::cout << "Best device found: " << bestDeviceProp.name << std::endl;

    Dataset dataset(argv[1]);

    long double total_points = dataset.getTotalPoints();

    dim3 gridDimension(TOTAL_BLOCKS, 1, 1);
    dim3 blockDimension(1, 1, 1);

    // Allocating the number of threads to be used in the kernel
    if (TOTAL_THREADS_AXIS_X > total_points) {
        blockDimension.x = total_points;

    } else {
        blockDimension.x = TOTAL_THREADS_AXIS_X;

        // It is desirable to have blocks left over
        int n_dim_axis_y = ceil(total_points / TOTAL_THREADS_AXIS_X);

        if (n_dim_axis_y > TOTAL_THREADS_AXIS_Y) {
            blockDimension.y = TOTAL_THREADS_AXIS_Y;
            int n_dim_axis_z = ceil(n_dim_axis_y / TOTAL_THREADS_AXIS_Y);

            if (n_dim_axis_z > TOTAL_THREADS_AXIS_Z) {
                blockDimension.z = TOTAL_THREADS_AXIS_Z;

                // We must increase the block size
                blockDimension.x = ceil(total_points / (TOTAL_THREADS_AXIS_Y * TOTAL_THREADS_AXIS_Z));
            } else {
                blockDimension.z = n_dim_axis_z;
            }
        } else {
            blockDimension.y = n_dim_axis_y;
        }
    }

    // Print block dimension
    std::cout << "Block dimension for the specified dataset: " << blockDimension.x << " " << blockDimension.y << " "
              << blockDimension.z
              << std::endl;

    return EXIT_SUCCESS;
}
