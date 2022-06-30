#include <stdio.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B , float *C, const int N){

    for(int idx = 0 ; idx < N ; idx++)
        C[idx] = A[idx] + B[idx];

}

__global__ void sumArraysOnDevice(float *A, float *B , float *C, const int N){

    for(int idx = 0 ; idx < N ; idx++)
        C[idx] = A[idx] + B[idx];

}

void initialData(float *ip, int size){

    time_t t;
    srand((unsigned int) time(&t));

    for(int i = 0 ; i < size ; i++)
        ip[i] =  (float)( rand() & 0xFF )/10.0f;

}

void printArray(float *arr, int size){

    int i;
    for(i = 0 ; i < size ; i++){
        printf("%7.2f", arr[i]);
        if((i+1) % 15 == 0) putchar('\n');
    }
    if((i+1) != 15) putchar('\n');

}

int main(int argc, char* argv[]){

    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    //Host memory allocation
    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    //Host memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    //Initialize host data
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    //Move initialized data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    //Get sum of arrays with host and print result
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    puts("Array from CPU:");
    printArray(h_C, nElem);
    //Clean host memory
    free(h_A);
    free(h_B);
    free(h_C);

    //Get sum of arrays with device and print result
    sumArraysOnDevice<<<1, 1>>>(d_A, d_B, d_C, nElem);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);            //Copy data form device to host
    cudaFree(d_A);                                                      //Clean device memory
    cudaFree(d_B);
    cudaFree(d_C);
    puts("\nArray from GPU:");                              
    printArray(gpuRef, nElem);                                          //Print copied form gpu data

    free(gpuRef);

    return 0;

}