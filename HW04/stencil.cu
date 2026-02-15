// stencil.cu
#include "stencil.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// Computes 1D convolution:
// output[i] = sum_{j=-R..R} image[i+j] * mask[j+R]
// Boundary condition: image[x] = 1 when x < 0 or x >= n
//
// Shared memory (dynamic only) stores:
//  - full mask (2R+1)
//  - image tile for the block including halo (blockDim.x + 2R)
//  - output values for the block (blockDim.x) before writing to global
__global__ void stencil_kernel(const float* image,
                               const float* mask,
                               float* output,
                               unsigned int n,
                               unsigned int R) {
  const unsigned int tid  = threadIdx.x;
  const unsigned int bdim = blockDim.x;
  const unsigned int gid  = blockIdx.x * bdim + tid;

  const unsigned int mask_len = 2u * R + 1u;
  const unsigned int tile_len = bdim + 2u * R;
  const unsigned int out_len  = bdim;

  // Dynamic shared memory layout:
  // [0 .. mask_len-1]                         -> sh_mask
  // [mask_len .. mask_len+tile_len-1]         -> sh_image (with halo)
  // [mask_len+tile_len .. +tile_len+out_len-1]-> sh_out
  extern __shared__ float sh[];
  float* sh_mask  = sh;
  float* sh_image = sh + mask_len;
  float* sh_out   = sh + mask_len + tile_len;

  // Load entire mask into shared memory
  for (unsigned int k = tid; k < mask_len; k += bdim) {
    sh_mask[k] = mask[k];
  }

  // Load image tile including halo into shared memory
  // sh_image[k] corresponds to global index: (block_start + k - R)
  const long block_start = static_cast<long>(blockIdx.x) * static_cast<long>(bdim);
  for (unsigned int k = tid; k < tile_len; k += bdim) {
    const long g = block_start + static_cast<long>(k) - static_cast<long>(R);
    if (g < 0 || g >= static_cast<long>(n)) {
      sh_image[k] = 1.0f; // boundary value
    } else {
      sh_image[k] = image[static_cast<unsigned int>(g)];
    }
  }

  __syncthreads();

  // Compute one output element per thread into shared memory first
  if (gid < n) {
    float sum = 0.0f;
    const unsigned int center = tid + R; // center index inside sh_image

    for (int j = -static_cast<int>(R); j <= static_cast<int>(R); ++j) {
      const unsigned int mj = static_cast<unsigned int>(j + static_cast<int>(R));
      const unsigned int ij = static_cast<unsigned int>(static_cast<int>(center) + j);
      sum += sh_image[ij] * sh_mask[mj];
    }
    sh_out[tid] = sum;
  }

  __syncthreads();

  // Write results from shared memory to global memory
  if (gid < n) {
    output[gid] = sh_out[tid];
  }
}

// Host wrapper: makes exactly one kernel launch
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block) {
  const unsigned int blocks = (n + threads_per_block - 1u) / threads_per_block;

  const unsigned int mask_len = 2u * R + 1u;
  const unsigned int tile_len = threads_per_block + 2u * R;
  const unsigned int out_len  = threads_per_block;

  const size_t shmem_bytes =
      static_cast<size_t>(mask_len + tile_len + out_len) * sizeof(float);

  stencil_kernel<<<blocks, threads_per_block, shmem_bytes>>>(image, mask, output, n, R);
  CUDA_CHECK(cudaGetLastError());
}
