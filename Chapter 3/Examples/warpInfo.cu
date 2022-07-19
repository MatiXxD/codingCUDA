#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) {                                                          \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


int main(int argc, char *argv[]) {

    int iDev = 0;
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp, iDev));

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("  Number of multiprocessors:                     %d\n",
           iProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %4.2f KB\n",
           iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n",
           iProp.sharedMemPerBlock / 1024.0);
    printf("  Total number of registers available per block: %d\n",
           iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n",
           iProp.maxThreadsPerMultiProcessor / 32);

    return EXIT_SUCCESS;

}