// task1.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

// Simple CUDA error checking helper
static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Fill matrix with deterministic values (float and double version)
template <typename T>
void fill_matrix(std::vector<T>& M, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            M[i * n + j] = static_cast<T>((i + 1) * 0.01 + (j + 2) * 0.02);
        }
    }
}

// Specialized fill for int to avoid overflow
template <>
void fill_matrix<int>(std::vector<int>& M, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            M[i * n + j] = static_cast<int>((i + 1) + (j + 2));
        }
    }
}

// Run a single matmul variant (int, float, or double)
template <typename T>
void run_matmul(
    void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
    unsigned int n,
    unsigned int block_dim)
{
    size_t N = static_cast<size_t>(n) * static_cast<size_t>(n);

    // Allocate host matrices (row-major)
    std::vector<T> hA(N);
    std::vector<T> hB(N);
    std::vector<T> hC(N, static_cast<T>(0));

    fill_matrix<T>(hA, n);
    fill_matrix<T>(hB, n);

    // Allocate device memory
    T *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cuda_check(cudaMalloc(&dA, N * sizeof(T)), "cudaMalloc dA");
    cuda_check(cudaMalloc(&dB, N * sizeof(T)), "cudaMalloc dB");
    cuda_check(cudaMalloc(&dC, N * sizeof(T)), "cudaMalloc dC");

    // Copy input matrices to device
    cuda_check(cudaMemcpy(dA, hA.data(), N * sizeof(T), cudaMemcpyHostToDevice),
               "Memcpy A H2D");
    cuda_check(cudaMemcpy(dB, hB.data(), N * sizeof(T), cudaMemcpyHostToDevice),
               "Memcpy B H2D");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "EventCreate start");
    cuda_check(cudaEventCreate(&stop), "EventCreate stop");

    // Record start time
    cuda_check(cudaEventRecord(start), "EventRecord start");

    // Call matrix multiplication function
    // Note: matmul_* internally calls cudaDeviceSynchronize()
    matmul_func(dA, dB, dC, n, block_dim);

    // Record stop time
    cuda_check(cudaEventRecord(stop), "EventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "EventSync stop");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "EventElapsedTime");

    // Copy result back to host
    cuda_check(cudaMemcpy(hC.data(), dC, N * sizeof(T), cudaMemcpyDeviceToHost),
               "Memcpy C D2H");

    // Print required outputs:
    // 1) first element
    // 2) last element
    // 3) runtime in milliseconds
    std::cout << hC[0] << std::endl;
    std::cout << hC[N - 1] << std::endl;
    std::cout << elapsed_ms << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    // Expect two command-line arguments: n and block_dim
    if (argc < 3) {
        std::cerr << "Usage: ./task1 n block_dim" << std::endl;
        return 1;
    }

    unsigned int n = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    unsigned int block_dim =
        static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));

    if (n == 0 || block_dim == 0) {
        std::cerr << "n and block_dim must be positive integers." << std::endl;
        return 1;
    }

    // Run integer version
    run_matmul<int>(matmul_1, n, block_dim);

    // Run float version
    run_matmul<float>(matmul_2, n, block_dim);

    // Run double version
    run_matmul<double>(matmul_3, n, block_dim);

    return 0;
}