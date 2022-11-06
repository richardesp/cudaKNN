//
// Created by ricardo on 6/11/22.
//

#ifndef CUDAKNN_LOCK_H
#define CUDAKNN_LOCK_H
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct Lock {
    int *mutex;

    Lock(void) {
        int state = 0;
        cudaMalloc((void **) &mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
        //printf("constructor, lock pointer: %p\n", mutex);
    }

    ~Lock(void) {
//cudaFree(mutex);
        //printf("destructor\n");
    }

    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
        __threadfence();
    }

    __device__ void unlock(void) {
        atomicExch(mutex, 0);
        __threadfence();
    }
};

#endif //CUDAKNN_LOCK_H
