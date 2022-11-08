#include <cstdio>
#include <cmath>
#include "include/Dataset.h"
#include "include/cuda/Lock.h"
#include "include/cuda/cudaKNN.cuh"
#include "include/Point.h"
#include "include/Label.h"


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
    *host_k = 3;

    Dataset dataset(argv[1]);

    size_t *host_totalPoints = (size_t *) malloc(sizeof(size_t));
    size_t *dev_totalPoints = nullptr;

    size_t *host_totalLabels = (size_t *) malloc(sizeof(size_t));
    size_t *dev_totalLabels = nullptr;

    *host_totalPoints = dataset.getNPoints();
    *host_totalLabels = dataset.getNLabels();

    if (*host_totalPoints > MAX_GRID_DIM_X) {
        fprintf(stderr, "The number of points is too big for the grid size\n");
        return EXIT_FAILURE;
    }

    DistanceType *host_distanceType = (DistanceType *) malloc(sizeof(DistanceType));
    DistanceType *dev_distanceType = nullptr;

    *host_distanceType = EUCLIDEAN; // This must be corrected

    Point *host_points = dataset.getPoints();
    Point *dev_points = nullptr;

    // N labels to predict
    Label *host_labels = (Label *) malloc(sizeof(Label) * *host_totalLabels);
    Label *dev_labels = nullptr;

    fprintf(stdout, "Total points: %ld.\nResulting points array for training:\n", *host_totalPoints);
    for (size_t i = 0; i < *host_totalPoints; ++i) {
        fprintf(stdout, "%zu: %f %f %f. Label: %zu\n", host_points[i].getId(), host_points[i].getX(),
                host_points[i].getY(),
                host_points[i].getZ(), host_points[i].getLabel());
    }

    fprintf(stdout, "Total labels to predict: \n");
    for (size_t i = 0; i < *host_totalLabels; ++i) {
        host_labels[i].frequency = 0;
        host_labels[i].label = i;
        fprintf(stdout, "%zu: %d.\n", i, host_labels[i].frequency);
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

    // Allocating the labels in the GPU
    cudaStatus = cudaMalloc((void **) &dev_labels, *host_totalLabels * sizeof(Label));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Allocating the distance type in the GPU
    cudaStatus = cudaMalloc((void **) &dev_distanceType, sizeof(DistanceType));

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

    // Allocating the number of labels in the GPU
    cudaStatus = cudaMalloc((void **) &dev_totalLabels, sizeof(size_t));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

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

    // Copying the number of labels to the GPU
    cudaStatus = cudaMemcpy(dev_totalLabels, host_totalLabels, sizeof(size_t), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the distance type to the GPU
    cudaStatus = cudaMemcpy(dev_distanceType, host_distanceType, sizeof(DistanceType), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the labels to the GPU
    cudaStatus = cudaMemcpy(dev_labels, host_labels, *host_totalLabels * sizeof(Label), cudaMemcpyHostToDevice);

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

    // Creating the distance array into the GPU memory
    double *dev_distances = nullptr;

    cudaStatus = cudaMalloc((void **) &dev_distances, *host_totalPoints * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Creating the query point
    Point *host_queryPoint = (Point *) malloc(sizeof(Point));
    Point *dev_queryPoint = nullptr;

    host_queryPoint->x = 2.5;
    host_queryPoint->y = 7.0;
    host_queryPoint->z = 0.0;

    // Allocating the query point in the GPU
    cudaStatus = cudaMalloc((void **) &dev_queryPoint, sizeof(Point));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    // Copying the query point to the GPU
    cudaStatus = cudaMemcpy(dev_queryPoint, host_queryPoint, sizeof(Point), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        return EXIT_FAILURE;
    }

    printf("Starting the kernel...\n");

    // Creating a lock
    Lock lock;

    // Creating the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Label *predictedLabel = (Label *) malloc(sizeof(Label));

    // Starting the kernel
    cudaEventRecord(start);
    knn::predict<<<*host_totalPoints, 1>>>(dev_points, dev_totalLabels, dev_totalPoints, dev_k, dev_distanceType, dev_distances,
            dev_queryPoint, dev_labels, lock);

    /*knn::cdp_simple_quicksort<<<*host_totalPoints, 1>>>(dev_distances, dev_points, 0,
            *host_totalPoints - 1, 1); REQUIRED */

    thrust::host_vector<NodeThrust> host_nodesVector(*host_totalPoints);

    // Memcpy the points array to the host
    cudaMemcpy(host_points, dev_points, *host_totalPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    for (size_t index = 0; index < *host_totalPoints; index++) {
        host_nodesVector[index].x = host_points[index].x;
        host_nodesVector[index].y = host_points[index].y;
        host_nodesVector[index].z = host_points[index].z;
        host_nodesVector[index].label = host_points[index].label;
        host_nodesVector[index].distance = host_points[index].distance;
    }

    thrust::device_vector<NodeThrust> dev_nodesVector = host_nodesVector;

    thrust::sort(dev_nodesVector.begin(), dev_nodesVector.end());

    host_nodesVector = dev_nodesVector;

    for (size_t index = 0; index < *host_k; ++index) {
        for (size_t label = 0; label < *host_totalLabels; ++label) {
            if (host_nodesVector[index].label == label) {
                host_labels[label].frequency = host_labels[label].frequency + 1;
                continue;
            }
        }
    }

    predictedLabel = thrust::max_element(host_labels, host_labels + *host_totalLabels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    printf("Kernel finished. Elapsed time: %f ms\n", miliseconds);
    fprintf(stdout, "################################################\n");

    // Printing the points
    /*
    fprintf(stdout, "Points:\n");
    for (size_t i = 0; i < *host_totalPoints; ++i) {
        fprintf(stdout, "%zu: (%f, %f, %f) --> Label: %zu. Distance to queryPoint: %f\n", i, host_nodesVector[i].x,
                host_nodesVector[i].y, host_nodesVector[i].z, host_nodesVector[i].label, host_nodesVector[i].distance);
    }
     */

    // Printing the labels
    printf("Predicted label: %zu\n", predictedLabel->label);

    return EXIT_SUCCESS;
}