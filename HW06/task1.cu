#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include "mmul.h"

int main(int argc, char* argv[])
{
    if (argc != 3) return EXIT_FAILURE;

    int n = std::atoi(argv[1]);
    int n_tests = std::atoi(argv[2]);

    size_t bytes = n * n * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
        C[i] = dist(gen);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < n_tests; ++i)
        mmul(handle, A, B, C, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms / n_tests << std::endl;

    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}