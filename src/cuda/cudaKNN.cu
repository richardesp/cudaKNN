//
// Created by ricardo on 11/11/22.
//

# include "../../include/cuda/cudaKNN.cuh"

struct sort_double {
    __host__ __device__ bool operator()(const NodeThrust &a, const NodeThrust &b) {
        return a.distance < b.distance;
    }
};

__host__ void
knn::predict(Point *points, const size_t *totalLabels, const size_t *totalPoints, const size_t *k,
             DistanceType *distanceType, Point *queryPoints, const size_t *totalEvaluationPoints,
             Label *frequencyLabels, double *distances, Label *predictedLabels) {

    for (size_t index = 0; index < *totalEvaluationPoints; ++index) {

        knn::predictOnePoint<<<dim3(*totalPoints, 1,
                                    1), 1>>>(points, totalLabels, totalPoints, k, distanceType, distances,
                queryPoints, frequencyLabels);

        /*knn::cdp_simple_quicksort<<<*host_totalPoints, 1>>>(dev_distances, dev_points, 0,
                *host_totalPoints - 1, 1); REQUIRED */

        thrust::host_vector<NodeThrust> host_nodesVector(*totalPoints);

        for (size_t index = 0; index < *totalPoints; index++) {
            host_nodesVector[index].x = points[index].x;
            host_nodesVector[index].y = points[index].y;
            host_nodesVector[index].z = points[index].z;
            host_nodesVector[index].label = points[index].label;
            host_nodesVector[index].distance = points[index].distance;
        }

        thrust::sort(host_nodesVector.begin(), host_nodesVector.end(), sort_double());

        thrust::device_vector<NodeThrust> dev_nodesVector = host_nodesVector;

        host_nodesVector = dev_nodesVector;

        for (size_t index = 0; index < *k; ++index) {
            for (size_t label = 0; label < *totalLabels; ++label) {
                if (host_nodesVector[index].label == label) {
                    frequencyLabels[label].frequency = frequencyLabels[label].frequency + 1;
                    continue;
                }
            }
        }

        predictedLabels[index].label = reinterpret_cast<size_t>(thrust::max_element(frequencyLabels,
                                                                                    frequencyLabels + *totalLabels));
    }

}

__global__ void
knn::predictOnePoint(Point *points, const size_t *totalLabels, const size_t *totalPoints, const size_t *k,
                     DistanceType *distanceType,
                     double *distances,
                     Point *queryPoint, Label *frequencyLabels) {


    computeDistance(points, totalPoints, k, distanceType, distances, queryPoint);
    __syncthreads();
}

__device__ void
knn::computeDistance(Point *points, const size_t *totalPoints, const size_t *k, DistanceType *distanceType,
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

__global__ void knn::cdp_simple_quicksort(double *data, Point *points, int left, int right, int depth) {
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

__device__ void knn::selection_sort(double *data, Point *points, int left, int right) {
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

__global__ void
knn::predictAllPoints(Point *points, const size_t *totalLabels, const size_t *totalPoints,
                      const size_t *totalQueriesPoints, const size_t *k,
                      DistanceType *distanceType,
                      double *distances,
                      Point *queriesPoints, Label *frequencyLabels) {

    //size_t threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    //size_t threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}