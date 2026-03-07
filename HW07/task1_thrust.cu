#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

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

  const size_t n = std::strtoull(argv[1], nullptr, 10);
  if (n == 0) {
    std::cerr << "n must be a positive integer\n";
    return EXIT_FAILURE;
  }

  thrust::host_vector<float> h_vec(n);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    h_vec[i] = dist(rng);
  }

  thrust::device_vector<float> d_vec = h_vec;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f,
                                thrust::plus<float>());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  std::cout << std::setprecision(10) << result << '\n';
  std::cout << std::fixed << std::setprecision(6) << elapsed_ms << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
