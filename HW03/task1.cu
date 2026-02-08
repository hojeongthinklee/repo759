// task1.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void factorial_kernel(int *dA) {
    int tid = threadIdx.x;      // thread index: 0..7
    int val = tid + 1;          // compute factorial of 1..8

    int fact = 1;
    for (int i = 2; i <= val; ++i)
        fact *= i;

    dA[tid] = fact;             // store result
}

int main() {
    const int N = 8;

    int *dA;
    cudaMalloc(&dA, N * sizeof(int));

    factorial_kernel<<<1, N>>>(dA);
    cudaDeviceSynchronize();

    int hA[N];
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << hA[i] << "\n";

    cudaFree(dA);
    return 0;
}
