// task1.cu
#include "matmul.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s n threads_per_block\n", argv[0]);
    return 1;
  }

  const size_t n = static_cast<size_t>(std::stoul(argv[1]));
  const unsigned int threads_per_block = static_cast<unsigned int>(std::stoul(argv[2]));
  const size_t N = n * n;

  // Host matrices (row-major)
  std::vector<float> hA(N), hB(N);

  // Fill with random numbers in [-1, 1]
  std::mt19937 rng(12345); // fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N; ++i) {
    hA[i] = dist(rng);
    hB[i] = dist(rng);
  }

  // Device memory
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  matmul(dA, dB, dC, n, threads_per_block);
  CUDA_CHECK(cudaEventRecord(stop));

  // Synchronize on stop event (also ensures kernel finished)
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy back the last element only
  float last = 0.0f;
  CUDA_CHECK(cudaMemcpy(&last, dC + (N - 1), sizeof(float), cudaMemcpyDeviceToHost));

  // Print last element then time (ms), matching example format
  std::printf("%.2f\n", last);
  std::printf("%.2f\n", ms);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}
