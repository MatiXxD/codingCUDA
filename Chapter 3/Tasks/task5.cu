// Array with big size can be a little bit different because of float type

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


__global__ void gpuReduction(float *g_idata, float *g_odata, unsigned int n){

    unsigned int tid = threadIdx.x;                                 
    unsigned int idx = threadIdx.x + 8 * blockIdx.x * blockDim.x;

    float* idata = g_idata + 8 * blockIdx.x * blockDim.x;                           // Get data for four blocks 

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

double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int recursiveReduce(float *data, int const size) {

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
    int size = 1<<12; // total number of elements to reduce
    printf(" with array size %d ", size);

    // execution configuration
    int blocksize = 512;                                                            // initial block size
    if(argc > 1) 
        blocksize = atoi(argv[1]);                                                  // block size from command line argument
    dim3 block (blocksize,1);
    dim3 grid ((size+block.x-1)/block.x,1);
    printf("grid %d block %d\n",grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_idata = (float*)malloc(bytes);
    float *h_odata = (float*)malloc(grid.x*sizeof(float));
    float *tmp = (float*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) 
        h_idata[i] = (float)(rand() & 0xFF);                                          
    memcpy (tmp, h_idata, bytes);
    size_t iStart,iElaps;
    float gpu_sum = 0;

    // allocate device memory
    float *d_idata = NULL;
    float *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x*sizeof(float));

    // cpu reduction
    iStart = seconds ();
    float cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds () - iStart;
    printf("cpu reduce elapsed %d ms cpu_sum: %f\n",iElaps,cpu_sum);

    // gpu reduction 
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds ();
    gpuReduction<<<grid.x/8, block.x>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x/8)*sizeof(float), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu gpuReduction elapsed %d ms gpu_sum: %f <<<grid %d block %d>>>\n",
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