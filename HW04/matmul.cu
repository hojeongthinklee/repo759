// matmul.cu
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
  // 1D kernel configuration: each thread computes one element of C
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = n * n;
  if (idx >= total) return;

  size_t row = idx / n;
  size_t col = idx % n;

  float sum = 0.0f;
  // C[row, col] = sum_k A[row, k] * B[k, col]
  for (size_t k = 0; k < n; ++k) {
    sum += A[row * n + k] * B[k * n + col];
  }
  C[row * n + col] = sum;
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
  size_t total = n * n;
  unsigned int blocks = static_cast<unsigned int>((total + threads_per_block - 1) / threads_per_block);

  matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
  CUDA_CHECK(cudaGetLastError());
}
