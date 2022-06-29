 #include <stdio.h>

 __global__ void helloFromGPU(void){

    printf("Hello World from GPU thread %d\n", threadIdx.x);

 }

 int main(void){

    printf("Hello World from CPU!\n");

    helloFromGPU<<<5, 5>>>();
    cudaDeviceReset();
    //cudaDeviceSynchronize();                                             // Can be used instead of cudaDeviceReset()

    return 0;

 }
