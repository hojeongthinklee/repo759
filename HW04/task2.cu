// task2.cu
#include "stencil.cuh"
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
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s n R threads_per_block\n", argv[0]);
    return 1;
  }

  const unsigned int n = static_cast<unsigned int>(std::stoul(argv[1]));
  const unsigned int R = static_cast<unsigned int>(std::stoul(argv[2]));
  const unsigned int threads_per_block = static_cast<unsigned int>(std::stoul(argv[3]));

  const unsigned int mask_len = 2u * R + 1u;

  // Host arrays
  std::vector<float> hImage(n);
  std::vector<float> hMask(mask_len);

  // Fill with random numbers in [-1, 1]
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (unsigned int i = 0; i < n; ++i) hImage[i] = dist(rng);
  for (unsigned int i = 0; i < mask_len; ++i) hMask[i] = dist(rng);

  // Device arrays
  float *dImage = nullptr, *dMask = nullptr, *dOutput = nullptr;
  CUDA_CHECK(cudaMalloc(&dImage, static_cast<size_t>(n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dMask, static_cast<size_t>(mask_len) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOutput, static_cast<size_t>(n) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dImage, hImage.data(),
                        static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dMask, hMask.data(),
                        static_cast<size_t>(mask_len) * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Time only the stencil() call using CUDA events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  stencil(dImage, dMask, dOutput, n, R, threads_per_block);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy back the last element only
  float last = 0.0f;
  CUDA_CHECK(cudaMemcpy(&last, dOutput + (n - 1u),
                        sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Print last element then time (ms), matching assignment format
  std::printf("%.2f\n", last);
  std::printf("%.2f\n", ms);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dImage));
  CUDA_CHECK(cudaFree(dMask));
  CUDA_CHECK(cudaFree(dOutput));

  return 0;
}
