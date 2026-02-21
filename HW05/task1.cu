// task1.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

// CUDA error checking
static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Fill matrix (float/double)
template <typename T>
void fill_matrix(std::vector<T>& M, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            M[i * n + j] =
                static_cast<T>((i + 1) * 0.01 + (j + 2) * 0.02);
}

// Specialized fill for int
template <>
void fill_matrix<int>(std::vector<int>& M, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            M[i * n + j] =
                static_cast<int>((i + 1) + (j + 2));
}

// Run one matmul variant
template <typename T>
void run_case(void (*matmul_func)(
                  const T*, const T*, T*,
                  unsigned int, unsigned int),
              unsigned int n,
              unsigned int block_dim)
{
    size_t N = static_cast<size_t>(n) * n;

    std::vector<T> hA(N), hB(N), hC(N, 0);

    fill_matrix<T>(hA, n);
    fill_matrix<T>(hB, n);

    T *dA, *dB, *dC;
    cuda_check(cudaMalloc(&dA, N*sizeof(T)), "Malloc dA");
    cuda_check(cudaMalloc(&dB, N*sizeof(T)), "Malloc dB");
    cuda_check(cudaMalloc(&dC, N*sizeof(T)), "Malloc dC");

    cuda_check(cudaMemcpy(dA, hA.data(),
               N*sizeof(T), cudaMemcpyHostToDevice),
               "Memcpy A");
    cuda_check(cudaMemcpy(dB, hB.data(),
               N*sizeof(T), cudaMemcpyHostToDevice),
               "Memcpy B");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_func(dA, dB, dC, n, block_dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC.data(), dC,
               N*sizeof(T), cudaMemcpyDeviceToHost);

    // Required output format:
    std::cout << hC[0] << std::endl;
    std::cout << hC[N-1] << std::endl;
    std::cout << ms << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    unsigned int n =
        static_cast<unsigned int>(std::atoi(argv[1]));
    unsigned int block_dim =
        static_cast<unsigned int>(std::atoi(argv[2]));

    run_case<int>(matmul_1, n, block_dim);
    run_case<float>(matmul_2, n, block_dim);
    run_case<double>(matmul_3, n, block_dim);

    return 0;
}