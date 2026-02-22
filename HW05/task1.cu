#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

// Fill A and B with small integer patterns that are safe for int
// and produce similar results across int/float/double.
template <typename T>
static void fill_matrices(std::vector<T>& A, std::vector<T>& B, unsigned int n) {
    const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);
    A.resize(N);
    B.resize(N);

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int aval = (i + j) % 10;          // 0..9
            int bval = (i * 3 + j * 7) % 10;  // 0..9
            A[i * n + j] = static_cast<T>(aval);
            B[i * n + j] = static_cast<T>(bval);
        }
    }
}

template <typename T>
static void run_case(void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
                     unsigned int n,
                     unsigned int block_dim)
{
    const size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::vector<T> hA, hB, hC(N);

    fill_matrices<T>(hA, hB, n);

    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, N * sizeof(T));
    cudaMalloc(&dB, N * sizeof(T));
    cudaMalloc(&dC, N * sizeof(T));

    cudaMemcpy(dA, hA.data(), N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, N * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_func(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC.data(), dC, N * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << hC[0] << "\n";
    std::cout << hC[N - 1] << "\n";
    std::cout << ms << "\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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