// matmul.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "matmul.cuh"

// Simple CUDA error checking
static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n",
                     msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// Tiled matrix multiplication kernel using shared memory
template <typename T>
__global__ void matmul_kernel(const T* A,
                              const T* B,
                              T* C,
                              unsigned int n,
                              unsigned int block_dim)
{
    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Allocate dynamic shared memory
    extern __shared__ unsigned char shared_raw[];
    T* As = reinterpret_cast<T*>(shared_raw);
    T* Bs = As + block_dim * block_dim;

    // Global row and column of C this thread computes
    int row = by * block_dim + ty;
    int col = bx * block_dim + tx;

    T Cvalue = 0;

    // Number of tiles
    int numTiles = (n + block_dim - 1) / block_dim;

    for (int t = 0; t < numTiles; ++t) {

        int tiledColA = t * block_dim + tx;
        int tiledRowB = t * block_dim + ty;

        // Load A tile
        if (row < (int)n && tiledColA < (int)n)
            As[ty * block_dim + tx] = A[row * n + tiledColA];
        else
            As[ty * block_dim + tx] = 0;

        // Load B tile
        if (tiledRowB < (int)n && col < (int)n)
            Bs[ty * block_dim + tx] = B[tiledRowB * n + col];
        else
            Bs[ty * block_dim + tx] = 0;

        __syncthreads();

        // Compute partial product for this tile
        for (unsigned int k = 0; k < block_dim; ++k)
            Cvalue += As[ty * block_dim + k] *
                      Bs[k * block_dim + tx];

        __syncthreads();
    }

    // Write result
    if (row < (int)n && col < (int)n)
        C[row * n + col] = Cvalue;
}

// Kernel launcher wrapper
template <typename T>
void launch_matmul(const T* A,
                   const T* B,
                   T* C,
                   unsigned int n,
                   unsigned int block_dim)
{
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim,
              (n + block_dim - 1) / block_dim);

    size_t shared_bytes =
        2ull * block_dim * block_dim * sizeof(T);

    matmul_kernel<T><<<grid, block, shared_bytes>>>(
        A, B, C, n, block_dim);

    cuda_check(cudaGetLastError(), "Kernel launch");
    cuda_check(cudaDeviceSynchronize(), "Device sync");
}

// Required host wrappers
__host__ void matmul_1(const int* A,
                       const int* B,
                       int* C,
                       unsigned int n,
                       unsigned int block_dim)
{
    launch_matmul<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float* A,
                       const float* B,
                       float* C,
                       unsigned int n,
                       unsigned int block_dim)
{
    launch_matmul<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double* A,
                       const double* B,
                       double* C,
                       unsigned int n,
                       unsigned int block_dim)
{
    launch_matmul<double>(A, B, C, n, block_dim);
}