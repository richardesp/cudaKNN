#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../Label.h"

#define MAX_GRID_DIM_X 2147483647

#define EUCLIDEAN_DISTANCE(x1, y1, z1, x2, y2, z2) sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))

// We are going to use one block for apply mutual exclusion to the shared memory in the processing GPU
#define TOTAL_BLOCKS 1
#define MAX_DEPTH       16
#define INSERTION_SORT  32

namespace knn {

    __device__ void selection_sort(double *data, Point *points, int left, int right) {
        for (int i = left; i <= right; ++i) {
            double min_val = data[i];
            int min_idx = i;
            Point min_point = points[i];
            int min_point_idx = i;

            // Find the smallest value in the range [left, right].
            for (int j = i + 1; j <= right; ++j) {
                double val_j = data[j];
                Point val_point_j = points[j];
                if (val_j < min_val) {
                    min_idx = j;
                    min_val = val_j;
                    min_point_idx = j;
                    min_point = val_point_j;
                }
            }

            // Swap the values.
            if (i != min_idx) {
                data[min_idx] = data[i];
                data[i] = min_val;
                points[min_idx] = points[i];
                points[i] = min_point;
            }
        }
    }

    __global__ void cdp_simple_quicksort(double *data, Point *points, int left, int right, int depth) {
        //If we're too deep or there are few elements left, we use an insertion sort...
        if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
            selection_sort(data, points, left, right);
            return;
        }

        cudaStream_t s, s1;
        double *lptr = data + left;
        double *rptr = data + right;
        Point *lpointptr = points + left;
        Point *rpointptr = points + right;
        int pivot = data[(left + right) / 2];

        double lval;
        double rval;
        int lid, rid;
        double lx, rx;
        double ly, ry;
        double lz, rz;
        DistanceType ldistanceType, rdistanceType;


        double nright, nleft;

        // Do the partitioning.
        while (lptr <= rptr) {
            // Find the next left- and right-hand values to swap
            lval = *lptr;
            rval = *rptr;
            lid = lpointptr->id;
            rid = rpointptr->id;
            lx = lpointptr->x;
            rx = rpointptr->x;
            ly = lpointptr->y;
            ry = rpointptr->y;
            lz = lpointptr->z;
            rz = rpointptr->z;
            ldistanceType = lpointptr->distanceType;
            rdistanceType = rpointptr->distanceType;

            // Move the left pointer as long as the pointed element is smaller than the pivot.
            while (lval < pivot && lptr < data + right) {
                lptr++;
                lval = *lptr;
                lpointptr++;
                lid = lpointptr->id;
                lx = lpointptr->x;
                ly = lpointptr->y;
                lz = lpointptr->z;
                ldistanceType = lpointptr->distanceType;
            }

            // Move the right pointer as long as the pointed element is larger than the pivot.
            while (rval > pivot && rptr > data + left) {
                rptr--;
                rval = *rptr;
                rpointptr--;
                rid = rpointptr->id;
                rx = rpointptr->x;
                ry = rpointptr->y;
                rz = rpointptr->z;
                rdistanceType = rpointptr->distanceType;
            }

            // If the swap points are valid, do the swap!
            if (lptr <= rptr) {
                *lptr = rval;
                *rptr = lval;
                lptr++;
                rptr--;
                lpointptr->id = rid;
                lpointptr->x = rx;
                lpointptr->y = ry;
                lpointptr->z = rz;
                lpointptr->distanceType = rdistanceType;
                rpointptr->id = lid;
                rpointptr->x = lx;
                rpointptr->y = ly;
                rpointptr->z = lz;
                rpointptr->distanceType = ldistanceType;
                lpointptr++;
                rpointptr--;
            }
        }

        // Now the recursive part
        nright = rptr - data;
        nleft = lptr - data;

        // Launch a new block to sort the left part.
        if (left < (rptr - data)) {
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, points, left, nright, depth + 1);
            cudaStreamDestroy(s);
        }

        // Launch a new block to sort the right part.
        if ((lptr - data) < right) {
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, points, nleft, right, depth + 1);
            cudaStreamDestroy(s1);
        }
    }

    __device__ void
    computeDistance(Point *points, const size_t *totalPoints, const size_t *k, DistanceType *distanceType,
                    double *distances,
                    Point *queryPoint) {
        size_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

        // If the index is greater than the total points, we are out of bounds
        if (threadIndex < *totalPoints) {

            // Calculating the distance
            switch (*distanceType) {
                case EUCLIDEAN:
                    distances[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                                points[threadIndex].z, queryPoint->x, queryPoint->y,
                                                                queryPoint->z);
                    points[threadIndex].distance = distances[threadIndex];

                    break;
                case MANHATTAN:
                    break;
                case CHEBYSHEV:
                    break;
                default:
                    distances[threadIndex] = EUCLIDEAN_DISTANCE(points[threadIndex].x, points[threadIndex].y,
                                                                points[threadIndex].z, 0.0,
                                                                0.0, 0.0);
                    points[threadIndex].distance = distances[threadIndex];
            }
        }
    }

    __host__ void
    findKBests(double *distances, Point *points, const size_t *totalPoints, const size_t *k, Label *frequencyLabels,
               const size_t *totalLabels) {

    }

    __global__ void
    predict(Point *points, const size_t *totalLabels, const size_t *totalPoints, const size_t *k,
            DistanceType *distanceType,
            double *distances,
            Point *queryPoint, Label *frequencyLabels, Lock lock) {


        computeDistance(points, totalPoints, k, distanceType, distances, queryPoint);
        __syncthreads();
    }
}