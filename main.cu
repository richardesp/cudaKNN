#include <cstdio>
#include <cmath>
#include "include/Dataset.h"
#include "include/cuda/Lock.h"
#include "include/cuda/cudaKNN.cu"
#include "include/Point.h"


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

    cudaError_t cudaStatus = cudaSetDevice(bestDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    size_t *host_k = (size_t *) malloc(sizeof(size_t));
    size_t *dev_k = nullptr;
    *host_k = 5;

    Dataset dataset(argv[1]);

    size_t *host_totalPoints = (size_t *) malloc(sizeof(size_t));
    size_t *dev_totalPoints = nullptr;

    *host_totalPoints = dataset.getNPoints();

    if (*host_totalPoints > MAX_GRID_DIM_X) {
        fprintf(stderr, "The number of points is too big for the grid size\n");
        return EXIT_FAILURE;
    }

    Point *host_points = dataset.getPoints();
    Point *dev_points = nullptr;

    fprintf(stdout, "Total points: %ld.\nResulting points array for training:\n", *host_totalPoints);
    for (size_t i = 0; i < *host_totalPoints; ++i) {
        fprintf(stdout, "%zu: %f %f %f\n", host_points[i].getId(), host_points[i].getX(), host_points[i].getY(),
                host_points[i].getZ());
    }

    dim3 gpuBlocks(TOTAL_BLOCKS, 1, 1);
    dim3 gpuThreads(*host_totalPoints, 1, 1);

    fprintf(stdout, "################################################\n");
    fprintf(stdout, "GPU blocks: %d, GPU threads: %d\n", gpuBlocks.x, gpuThreads.x);

    // Allocating the points in the GPU
    cudaStatus = cudaMalloc((void **) &dev_points, *host_totalPoints * sizeof(Point));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Allocating the number of points in the GPU
    cudaStatus = cudaMalloc((void **) &dev_totalPoints, sizeof(size_t));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Allocating the K in the GPU
    cudaStatus = cudaMalloc((void **) &dev_k, sizeof(size_t));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the points to the GPU
    cudaStatus = cudaMemcpy(dev_points, host_points, *host_totalPoints * sizeof(Point), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the number of points to the GPU
    cudaStatus = cudaMemcpy(dev_totalPoints, host_totalPoints, sizeof(size_t), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the K to the GPU
    cudaStatus = cudaMemcpy(dev_k, host_k, sizeof(size_t), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    printf("Starting the kernel...\n");

    // Creating a lock
    Lock lock;

    // Starting the kernel
    cudaKNN<<<*host_totalPoints, 1>>>(dev_points, dev_totalPoints, dev_k, lock);


    return EXIT_SUCCESS;
}