#include <cuda_runtime.h>
#include "scan.cuh"

__global__
void hillis_steele(float* output,
                   const float* input,
                   unsigned int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
        sdata[tid] = input[gid];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset)
            val = sdata[tid - offset];

        __syncthreads();
        sdata[tid] += val;
        __syncthreads();
    }

    if (gid < n)
        output[gid] = sdata[tid];
}

__global__
void add_offsets(float* data,
                 const float* offsets,
                 unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n && blockIdx.x > 0)
        data[gid] += offsets[blockIdx.x - 1];
}

__host__
void scan(const float* input,
          float* output,
          unsigned int n,
          unsigned int threads_per_block)
{
    unsigned int blocks =
        (n + threads_per_block - 1) / threads_per_block;

    hillis_steele<<<blocks,
                    threads_per_block,
                    threads_per_block * sizeof(float)>>>(
        output, input, n);

    if (blocks > 1) {

        float* block_sums;
        cudaMalloc(&block_sums, blocks * sizeof(float));

        cudaMemcpy(block_sums,
                   output + (threads_per_block - 1),
                   blocks * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        float* scanned_block_sums;
        cudaMalloc(&scanned_block_sums,
                   blocks * sizeof(float));

        hillis_steele<<<1,
                        threads_per_block,
                        threads_per_block * sizeof(float)>>>(
            scanned_block_sums,
            block_sums,
            blocks);

        add_offsets<<<blocks, threads_per_block>>>(
            output,
            scanned_block_sums,
            n);

        cudaFree(block_sums);
        cudaFree(scanned_block_sums);
    }

    cudaDeviceSynchronize();
}