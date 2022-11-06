#include <cuda_runtime.h>

#define MAX_GRID_DIM_X 2147483647

#define EUCLIDEAN_DISTANCE(x1, y1, z1, x2, y2, z2) sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))


// We are going to use one block for apply mutual exclusion to the shared memory in the processing GPU
#define TOTAL_BLOCKS 1

__global__ void
cudaKNNPredict(Point *points, const size_t *totalPoints, const size_t *k, DistanceType *distanceType,
               double *dev_distance,
               Point *queryPoint, Lock lock) {
    size_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // If the index is greater than the total points, we are out of bounds
    if (threadIndex < *totalPoints) {

        // Calculating the distance
        switch (*distanceType) {
            case EUCLIDEAN:
                dev_distance[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                               points[threadIndex].z, queryPoint->x, queryPoint->y,
                                                               queryPoint->z);
                break;
            case MANHATTAN:
                break;
            case CHEBYSHEV:
                break;
            default:
                dev_distance[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                               points[threadIndex].z, 0.0,
                                                               0.0, 0.0);
        }

        lock.lock();
        printf("Thread index: %d\n", threadIndex);
        printf("Total points: %d\n", *totalPoints);
        printf("K: %d\n", *k);
        printf("x: %f\n", points[threadIndex].x);
        printf("y: %f\n", points[threadIndex].y);
        printf("z: %f\n", points[threadIndex].z);
        printf("Distance type: %d\n", *distanceType);
        printf("Query point x,y,z: %f,%f,%f\n", queryPoint->x, queryPoint->y, queryPoint->z);
        printf("Distance: %f\n", dev_distance[threadIndex]);
        lock.unlock();

    }
}