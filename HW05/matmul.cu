#include <cuda_runtime.h>
#include <cstdlib>
#include "matmul.cuh"

template <typename T>
__global__ void matmul_kernel(const T* A,
                              const T* B,
                              T* C,
                              unsigned int n)
{
    const unsigned int bd = blockDim.x;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int row = blockIdx.y * bd + ty;
    const unsigned int col = blockIdx.x * bd + tx;

    extern __shared__ unsigned char sh_raw[];
    T* As = reinterpret_cast<T*>(sh_raw);
    T* Bs = As + bd * bd;

    T acc = static_cast<T>(0);

    const unsigned int numTiles = (n + bd - 1) / bd;

    for (unsigned int t = 0; t < numTiles; ++t) {

        const unsigned int a_col = t * bd + tx;
        const unsigned int b_row = t * bd + ty;

        if (row < n && a_col < n)
            As[ty * bd + tx] = A[row * n + a_col];
        else
            As[ty * bd + tx] = static_cast<T>(0);

        if (b_row < n && col < n)
            Bs[ty * bd + tx] = B[b_row * n + col];
        else
            Bs[ty * bd + tx] = static_cast<T>(0);

        __syncthreads();

        for (unsigned int k = 0; k < bd; ++k)
            acc += As[ty * bd + k] * Bs[k * bd + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = acc;
}

template <typename T>
static void launch_matmul(const T* A,
                          const T* B,
                          T* C,
                          unsigned int n,
                          unsigned int block_dim)
{
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim,
              (n + block_dim - 1) / block_dim);

    size_t shmem_bytes =
        2ull * block_dim * block_dim * sizeof(T);

    matmul_kernel<T><<<grid, block, shmem_bytes>>>(
        A, B, C, n);

    cudaDeviceSynchronize();
}

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