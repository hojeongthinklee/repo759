#include "count.cuh"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: ./task2 n\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 500);

    thrust::host_vector<int> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // optional warm-up
    count(d_in, values, counts);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << values.back() << '\n';
    std::cout << counts.back() << '\n';
    std::cout << ms << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}