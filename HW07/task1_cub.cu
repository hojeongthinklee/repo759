#define CUB_STDERR

#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cub;

CachingDeviceAllocator g_allocator(true);

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " \
                << cudaGetErrorString(err__) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " n\n";
    return EXIT_FAILURE;
  }

  const size_t num_items = std::strtoull(argv[1], nullptr, 10);
  if (num_items == 0) {
    std::cerr << "n must be a positive integer\n";
    return EXIT_FAILURE;
  }

  // Host input: random floats in [-1.0, 1.0]
  std::vector<float> h_in(num_items);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < num_items; ++i) {
    h_in[i] = dist(rng);
  }

  // Device input allocation/copy pattern matches the reference example.
  float* d_in = nullptr;
  float* d_sum = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  CUDA_CHECK(g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_in),
                                        sizeof(float) * num_items));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(float) * num_items,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(g_allocator.DeviceAllocate(reinterpret_cast<void**>(&d_sum),
                                        sizeof(float)));

  // Dry run to get temporary storage size.
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                    d_sum, num_items));
  CUDA_CHECK(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                    d_sum, num_items));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  float gpu_sum = 0.0f;
  CUDA_CHECK(
      cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << std::setprecision(10) << gpu_sum << '\n';
  std::cout << std::fixed << std::setprecision(6) << elapsed_ms << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (d_in) CUDA_CHECK(g_allocator.DeviceFree(d_in));
  if (d_sum) CUDA_CHECK(g_allocator.DeviceFree(d_sum));
  if (d_temp_storage) CUDA_CHECK(g_allocator.DeviceFree(d_temp_storage));

  return 0;
}
