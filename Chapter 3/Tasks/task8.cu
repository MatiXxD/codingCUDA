// Compile with falg -rdc=true

#include <stdio.h>
#include <cuda_runtime.h>


__global__ void nestedHelloWorld(const unsigned int size, int depth, int minSize){

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Recursion=%d: Hello World from thread %d\n", depth, tid);

    if(size == minSize) return;

    unsigned int newSize = size >> 1;
    if(tid == 0 && newSize > 0){
        int tempGrid = (newSize + blockDim.x - 1) / blockDim.x;
        nestedHelloWorld<<<tempGrid, blockDim.x>>>(newSize, minSize, ++depth);
        printf("-------> nested execution depth: %d\n", depth);
    }

}


int main(int argc, char* argv[]){

    int size = 16;
    int blockSize = 8;

    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    printf("%s Execution Configuration: grid %d block %d\n", 
        argv[0], grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(size, 0, grid.x);
    cudaDeviceSynchronize();

    return 0;

}