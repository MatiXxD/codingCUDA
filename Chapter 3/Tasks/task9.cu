// Compile with flag -rdc=true

#include <stdio.h>
#include <cuda_runtime.h>


__global__ void nestedHelloWorld(const unsigned size, int depth, int maxDepth){

    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n",
        depth, tid, blockIdx.x);

    if(size == 1 || depth >= maxDepth) return;
    
    int nthreads = size >> 1;
    if(tid == 0 && nthreads > 0){
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++depth, maxDepth);
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

    nestedHelloWorld<<<grid, block>>>(size, 0, 1);
    cudaDeviceSynchronize();

    return 0;

}
