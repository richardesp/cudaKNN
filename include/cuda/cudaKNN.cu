#include <cuda_runtime.h>

#define MAX_GRID_DIM_X 2147483647

#define EUCLIDEAN_DISTANCE(x1, y1, z1, x2, y2, z2) sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))

// We are going to use one block for apply mutual exclusion to the shared memory in the processing GPU
#define TOTAL_BLOCKS 1

__global__ void cudaKNN(Point *points, const size_t *totalPoints, const size_t *k, Lock lock) {
    size_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // If the index is greater than the total points, we are out of bounds
    if (threadIndex < *totalPoints) {
        lock.lock();
        printf("Thread index: %d\n", threadIndex);
        printf("Total points: %d\n", *totalPoints);
        printf("K: %d\n", *k);
        printf("x: %f\n", points[threadIndex].x);
        printf("y: %f\n", points[threadIndex].y);
        printf("z: %f\n", points[threadIndex].z);
        lock.unlock();
    }
}