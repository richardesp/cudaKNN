#include <cuda_runtime.h>

#define MAX_GRID_DIM_X 2147483647

#define EUCLIDEAN_DISTANCE(x1, y1, z1, x2, y2, z2) sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))


// We are going to use one block for apply mutual exclusion to the shared memory in the processing GPU
#define TOTAL_BLOCKS 1

__device__ void selection_sort(double *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        double min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            double val_j = data[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void
cudaKNNPredict(Point *points, const size_t *totalPoints, const size_t *k, DistanceType *distanceType,
               double *distances,
               Point *queryPoint, Lock lock) {
    size_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // If the index is greater than the total points, we are out of bounds
    if (threadIndex < *totalPoints) {

        // Calculating the distance
        switch (*distanceType) {
            case EUCLIDEAN:
                distances[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                            points[threadIndex].z, queryPoint->x, queryPoint->y,
                                                            queryPoint->z);
                break;
            case MANHATTAN:
                break;
            case CHEBYSHEV:
                break;
            default:
                distances[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                            points[threadIndex].z, 0.0,
                                                            0.0, 0.0);
        }

        // We need wait until all threads finish the distance calculation for sort the resulting array
        __syncthreads();
        selection_sort(distances, 0, *totalPoints - 1);

        lock.lock();
        printf("Thread index: %d\n", threadIndex);
        printf("Total points: %d\n", *totalPoints);
        printf("K: %d\n", *k);
        printf("x: %f\n", points[threadIndex].x);
        printf("y: %f\n", points[threadIndex].y);
        printf("z: %f\n", points[threadIndex].z);
        printf("Distance type: %d\n", *distanceType);
        printf("Query point x,y,z: %f,%f,%f\n", queryPoint->x, queryPoint->y, queryPoint->z);
        printf("Distance: %f\n", distances[threadIndex]);

        // Print the distances vector
        for (int i = 0; i < *totalPoints; ++i) {
            printf("Distance %d: %f\n", i, distances[i]);
        }

        lock.unlock();

    }
}