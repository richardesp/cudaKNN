#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Global memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Memory pitch: " << deviceProp.memPitch << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions: " << deviceProp.maxThreadsDim[0] << " " << deviceProp.maxThreadsDim[1]
                  << " " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Max grid dimensions: " << deviceProp.maxGridSize[0] << " " << deviceProp.maxGridSize[1] << " "
                  << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem << std::endl;
        std::cout << "  Texture alignment: " << deviceProp.textureAlignment << std::endl;
        std::cout << "  Concurrent copy and execution: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Kernel execution timeout: " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No")
                  << std::endl;
    }
    return 0;
}
