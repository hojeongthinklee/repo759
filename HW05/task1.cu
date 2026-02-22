#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

// Safe fill for float/double
template <typename T>
static void fill_matrix(std::vector<T>& M, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            M[i * n + j] =
                static_cast<T>((i % 7) * 0.1 + (j % 11) * 0.01);
}

// Safe fill for int (no overflow)
template <>
void fill_matrix<int>(std::vector<int>& M, unsigned int /*n*/) {
    for (size_t i = 0; i < M.size(); ++i)
        M[i] = 1;
}

template <typename T>
static void run_case(
    void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
    unsigned int n,
    unsigned int block_dim)
{
    const size_t N = static_cast<size_t>(n) * n;

    std::vector<T> hA(N), hB(N), hC(N);

    fill_matrix<T>(hA, n);
    fill_matrix<T>(hB, n);

    T *dA, *dB, *dC;
    cudaMalloc(&dA, N * sizeof(T));
    cudaMalloc(&dB, N * sizeof(T));
    cudaMalloc(&dC, N * sizeof(T));

    cudaMemcpy(dA, hA.data(), N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * sizeof(T), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_func(dA, dB, dC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
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