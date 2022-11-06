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

    if (deviceCount == 0) {
        fprintf(stderr, "There is no device supporting CUDA\n");
        return EXIT_FAILURE;
    }

    int bestDevice = 0;
    cudaDeviceProp bestDeviceProp;

    for (int i = 0; i < deviceCount; i++) {

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        fprintf(stdout, "\tdevice %d: %s\n", i, deviceProp.name);
        fprintf(stdout, "\tCompute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(stdout, "\tGlobal memory: %ld\n", deviceProp.totalGlobalMem);
        fprintf(stdout, "\tShared memory per block: %ld\n", deviceProp.sharedMemPerBlock);
        fprintf(stdout, "\tRegisters per block: %d\n", deviceProp.regsPerBlock);
        fprintf(stdout, "\tWarp size: %d\n", deviceProp.warpSize);
        fprintf(stdout, "\tMemory pitch: %ld\n", deviceProp.memPitch);
        fprintf(stdout, "\tMax threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(stdout, "\tMax threads dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
                deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        fprintf(stdout, "\tMax grid dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
        fprintf(stdout, "\tClock rate: %d\n", deviceProp.clockRate);
        fprintf(stdout, "\tTotal constant memory: %ld\n", deviceProp.totalConstMem);
        fprintf(stdout, "\tTexture alignment: %ld\n", deviceProp.textureAlignment);
        fprintf(stdout, "\tConcurrent copy and execution: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
        fprintf(stdout, "\tNumber of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        fprintf(stdout, "\tKernel execution timeout: %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        fprintf(stdout, "\tIntegrated: %s\n", deviceProp.integrated ? "Yes" : "No");

        if (i == 0 || bestDeviceProp.totalGlobalMem < deviceProp.totalGlobalMem) {
            bestDevice = i;
            bestDeviceProp = deviceProp;
        }

    }

    fprintf(stdout, "\n################################################\n");
    fprintf(stdout, "Best device: %s (%d)\n", bestDeviceProp.name, bestDevice);

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
    fprintf(stdout, "Block dimension: (%d, %d, %d)\n", blockDimension.x, blockDimension.y, blockDimension.z);

    return EXIT_SUCCESS;
}