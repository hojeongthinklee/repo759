#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include "scan.cuh"

int main(int argc, char* argv[])
{
    if (argc != 3) return EXIT_FAILURE;

    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    float *input, *output;

    cudaMallocManaged(&input, n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (unsigned int i = 0; i < n; ++i)
        input[i] = dist(gen);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    scan(input, output, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << output[n - 1] << std::endl;
    std::cout << ms << std::endl;

    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}