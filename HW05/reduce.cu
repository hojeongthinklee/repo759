// reduce.cu
// Implements Reduction #4: "First Add During Load" with dynamic shared memory.

#include <cuda_runtime.h>
#include <cstdio>
#include "reduce.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#endif

// Kernel 4: first add during global load
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int blockSize = blockDim.x;

  // Each block reduces 2*blockSize elements
  unsigned int start = blockIdx.x * (blockSize * 2);
  unsigned int i = start + tid;

  float sum = 0.0f;

  // First add during load (guard for out-of-range)
  if (i < n) sum = g_idata[i];
  if (i + blockSize < n) sum += g_idata[i + blockSize];

  sdata[tid] = sum;
  __syncthreads();

  // Standard shared-memory reduction
  for (unsigned int s = blockSize >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
  // Requirements:
  // - No part of sum computed on host.
  // - Call reduce_kernel repeatedly until full sum obtained.
  // - Final sum must be written to first element of *input array (device memory).
  // - End with cudaDeviceSynchronize() for timing purposes.

  if (N == 0) {
    // Nothing to do; still synchronize for timing consistency.
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }

  float *orig_input = *input;
  float *in = *input;
  float *out = *output;

  unsigned int n = N;

  while (n > 1) {
    unsigned int blocks =
        (n + (threads_per_block * 2u) - 1u) / (threads_per_block * 2u);

    size_t shmem_bytes = static_cast<size_t>(threads_per_block) * sizeof(float);

    reduce_kernel<<<blocks, threads_per_block, shmem_bytes>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());

    // Next stage input is previous stage output
    n = blocks;

    // Ping-pong buffers: swap in/out pointers
    float *tmp = in;
    in = out;
    out = tmp;
  }

  // Ensure result ends up in orig_input[0]
  if (in != orig_input) {
    CUDA_CHECK(cudaMemcpy(orig_input, in, sizeof(float), cudaMemcpyDeviceToDevice));
    in = orig_input;
  }

  *input = orig_input; // keep contract: sum in first element of *input

  CUDA_CHECK(cudaDeviceSynchronize());
}