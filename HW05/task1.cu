#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "matmul.cuh"

// Fill A and B on the device with small values (0..9) to avoid int overflow.
// This keeps int/float/double results in similar ranges.
template <typename T>
__global__ void fill_AB_kernel(T* A, T* B, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long N = (unsigned long long)n * (unsigned long long)n;
    if ((unsigned long long)idx >= N) return;

    unsigned int i = idx / n;
    unsigned int j = idx - i * n;

    int aval = (int)((i + j) % 10);           // 0..9
    int bval = (int)((i * 3u + j * 7u) % 10); // 0..9

    A[idx] = (T)aval;
    B[idx] = (T)bval;
}

template <typename T>
static void run_case(void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
                     unsigned int n, unsigned int block_dim)
{
    unsigned long long N = (unsigned long long)n * (unsigned long long)n;

    // Allocate device matrices only (avoid huge host allocations)
    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, (size_t)(N * sizeof(T)));
    cudaMalloc(&dB, (size_t)(N * sizeof(T)));
    cudaMalloc(&dC, (size_t)(N * sizeof(T)));

    // Fill A and B on device
    const unsigned int TPB = 256;
    unsigned int blocks = (unsigned int)((N + TPB - 1) / TPB);
    fill_AB_kernel<T><<<blocks, TPB>>>(dA, dB, n);
    cudaDeviceSynchronize();

    // Time only the matmul call (matmul_* includes cudaDeviceSynchronize)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_func(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back only C[0] and C[last] (avoid huge D2H copy)
    T first = 0;
    T last  = 0;
    cudaMemcpy(&first, dC, sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last,  dC + (N - 1), sizeof(T), cudaMemcpyDeviceToHost);

    // Required output: first element, last element, time (ms)
    std::cout << first << "\n";
    std::cout << last  << "\n";
    std::cout << ms    << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv)
{
    unsigned int n = std::strtoul(argv[1], nullptr, 10);
    unsigned int block_dim = std::strtoul(argv[2], nullptr, 10);

    run_case<int>(matmul_1, n, block_dim);
    run_case<float>(matmul_2, n, block_dim);
    run_case<double>(matmul_3, n, block_dim);

    return 0;
}