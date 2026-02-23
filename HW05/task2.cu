// task2.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include "reduce.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#endif

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: ./task2 N threads_per_block\n");
    return 1;
  }

  unsigned int N = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
  unsigned int threads_per_block =
      static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));

  if (N == 0 || threads_per_block == 0) {
    std::fprintf(stderr, "Error: N and threads_per_block must be positive.\n");
    return 1;
  }

  std::vector<float> h(N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (unsigned int i = 0; i < N; ++i) h[i] = dist(rng);

  float* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h.data(),
                        static_cast<size_t>(N) * sizeof(float),
                        cudaMemcpyHostToDevice));

  unsigned int blocks_first =
      (N + (threads_per_block * 2u) - 1u) / (threads_per_block * 2u);
  blocks_first = std::max(1u, blocks_first);

  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(blocks_first) * sizeof(float)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  reduce(&d_input, &d_output, N, threads_per_block);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("%f\n", result);
  std::printf("%f\n", ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}