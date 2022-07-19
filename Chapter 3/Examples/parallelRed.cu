#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>


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


__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int* idata = g_idata + blockIdx.x * blockDim.x;                 // Get elements for every block 

    if(idx > n) return;                                             

    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if ((tid % (2 * stride)) == 0){                             // Take neighbored elements of array
            idata[tid] += idata[tid + stride];
            idata[tid + stride] = 0;                                // I include it to show that data with index tid+stride doesn't needed to further iterations
        }
        __syncthreads();
    }
            
    if (tid == 0) g_odata[blockIdx.x] = idata[0];                   // At the end sum will be in array with index 0 for each block

}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
            idata[index] += idata[index + stride];

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int* idata = g_idata + blockIdx.x * blockDim.x;                 // Get elements for every block 

    if(idx >= n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 2 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 2 * blockIdx.x * blockDim.x;                           // Get data for two blocks 

    if(idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];           // Each tread add element from the neighboring data block
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 4 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 4 * blockIdx.x * blockDim.x;                           // Get data for four blocks 

    if(idx + 3*blockDim.x < n) {                                                  // Each tread add element from the neighboring data blocks
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2*blockDim.x];
        g_idata[idx] += g_idata[idx + 3*blockDim.x];
    }     
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceUnrolling4v2(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 4 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 4 * blockIdx.x * blockDim.x;                           // Get data for four blocks 

    if(idx + 3*blockDim.x < n) {                                                  // Each tread add element from the neighboring data blocks
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2*blockDim.x];
        g_idata[idx] += g_idata[idx + 3*blockDim.x];
    }     
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {                // Now stride should be > 32. Last iteration will be complited in if statement
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if(tid < 32){                                                                 // Complete last iterations 
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceUnrolling4v3(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 4 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 4 * blockIdx.x * blockDim.x;                           // Get data for four blocks 

    if(idx + 3*blockDim.x < n) {                                                  // Each tread add element from the neighboring data blocks
        g_idata[idx] += g_idata[idx + blockDim.x];
        g_idata[idx] += g_idata[idx + 2*blockDim.x];
        g_idata[idx] += g_idata[idx + 3*blockDim.x];
    }     
    __syncthreads();

    if (blockDim.x>=1024 && tid < 512) idata[tid] += idata[tid + 512];
        __syncthreads();
    if (blockDim.x>=512 && tid < 256) idata[tid] += idata[tid + 256];
        __syncthreads();
    if (blockDim.x>=256 && tid < 128) idata[tid] += idata[tid + 128];
        __syncthreads();
    if (blockDim.x>=128 && tid < 64) idata[tid] += idata[tid + 64];
        __syncthreads();

    if(tid < 32){                                                                 // Complete last iterations 
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int recursiveReduce(int *data, int const size) {

    if (size == 1) return data[0];

    int const stride = size / 2;
    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);

}

int main(int argc, char **argv) {

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);
    bool bResult = false;

    // initialization
    int size = 1<<24; // total number of elements to reduce
    printf(" with array size %d ", size);

    // execution configuration
    int blocksize = 512;                                                            // initial block size
    if(argc > 1) 
        blocksize = atoi(argv[1]);                                                  // block size from command line argument
    dim3 block (blocksize,1);
    dim3 grid ((size+block.x-1)/block.x,1);
    printf("grid %d block %d\n",grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x*sizeof(int));
    int *tmp = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) 
        h_idata[i] = (int)(rand() & 0xFF);                                          // mask off high 2 bytes to force max number to 255
    memcpy (tmp, h_idata, bytes);
    size_t iStart,iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x*sizeof(int));

    // cpu reduction
    iStart = seconds ();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds () - iStart;
    printf("cpu reduce elapsed %d ms cpu_sum: %d\n",iElaps,cpu_sum);

    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceNeighbored elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceNeighboredLess elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 3: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu reduceInterleaved elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 4: reduceUnrolling2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceUnrolling2<<<grid.x / 2, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/2)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/2; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling2 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x/2,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 5: reduceUnrolling4
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceUnrolling4<<<grid.x / 4, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/4)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/4; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling4 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x/4,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 6: reduceUnrolling4v2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceUnrolling4v2<<<grid.x / 4, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/4)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/4; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling4v2 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x/4,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // kernel 7: reduceUnrolling4v3
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    reduceUnrolling4v3<<<grid.x / 4, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/4)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/4; i++) gpu_sum += h_odata[i];
    printf("gpu reduceUnrolling4v3 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x/4,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    /// free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset();

    if(bResult) puts("Arrays doesn't match.");

    return EXIT_SUCCESS;

}