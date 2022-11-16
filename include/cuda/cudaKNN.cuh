#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../Label.h"
#include "../Point.h"
#include "Lock.h"

#define MAX_GRID_DIM_X 2147483647
#define MAX_THREAD_DIM_X 1024

#define EUCLIDEAN_DISTANCE(x1, y1, z1, x2, y2, z2) sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))

// We are going to use one block for apply mutual exclusion to the shared memory in the processing GPU
#define TOTAL_BLOCKS 1
#define MAX_DEPTH       16
#define INSERTION_SORT  32

namespace knn {

    __host__ void
    predict(Point *points, const size_t *totalLabels, const size_t *totalPoints, const size_t *k,
            DistanceType *distanceType, Point *queryPoints, const size_t *totalEvaluationPoints,
            Label *frequencyLabels, double *distances, Label *predictedLabels);

    __device__ void selection_sort(double *data, Point *points, int left, int right);

    __global__ void cdp_simple_quicksort(double *data, Point *points, int left, int right, int depth);

    __device__ void
    computeDistance(Point *points, const size_t *totalPoints, const size_t *k, DistanceType *distanceType,
                    double *distances,
                    Point *queryPoint);

    __host__ void
    findKBests(double *distances, Point *points, const size_t *totalPoints, const size_t *k, Label *frequencyLabels,
               const size_t *totalLabels);

    __global__ void
    predictOnePoint(Point *points, const size_t *totalLabels, const size_t *totalPoints, const size_t *k,
                    DistanceType *distanceType,
                    double *distances,
                    Point *queryPoint, Label *frequencyLabels);

    __global__ void
    predictAllPoints(Point *points, const size_t *totalLabels, const size_t *totalPoints,
                     const size_t *totalQueriesPoints, const size_t *k,
                     DistanceType *distanceType,
                     double *distances,
                     Point *queriesPoints, Label *frequencyLabels);
}