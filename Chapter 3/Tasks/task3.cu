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


__global__ void unrolling8v1(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 8 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 8 * blockIdx.x * blockDim.x;                           // Get data for four blocks 

    if(idx + 7*blockDim.x < n){                                                   // Each tread add element from the neighboring data blocks
        g_idata[idx] += g_idata[idx + 1*blockDim.x];
        g_idata[idx] += g_idata[idx + 2*blockDim.x];
        g_idata[idx] += g_idata[idx + 3*blockDim.x];
        g_idata[idx] += g_idata[idx + 4*blockDim.x];
        g_idata[idx] += g_idata[idx + 5*blockDim.x];
        g_idata[idx] += g_idata[idx + 6*blockDim.x];
        g_idata[idx] += g_idata[idx + 7*blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

__global__ void unrolling8v2(int *g_idata, int *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 8 * blockIdx.x * blockDim.x;

    int* idata = g_idata + 8 * blockIdx.x * blockDim.x;                               // Get data for four blocks 

    int *ptr = g_idata + idx;           
    int tmp = 0;
    if(idx + 7*blockDim.x < n){
        for (int i = 0; i < 8; i++) {                                                 // Each tread add element from the neighboring data blocks     
            tmp += *ptr; 
            ptr += blockDim.x;
        }
        g_idata[idx] = tmp;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) 
            idata[tid] += idata[tid + stride];
        __syncthreads();
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

    // gpu unrolling8v1 
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    unrolling8v1<<<grid.x/8, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/8)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu unrolling8v1 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    if(gpu_sum != cpu_sum) bResult = true;

    // gpu unrolling8v2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    unrolling8v2<<<grid.x/8, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/8)*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu unrolling8v2 elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
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