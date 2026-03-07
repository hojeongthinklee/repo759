#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <iostream>
#include <random>
#include <cstdlib>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: ./task1_thrust n\n";
        return 1;
    }

    int n = atoi(argv[1]);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    thrust::host_vector<float> h(n);

    for (int i = 0; i < n; i++)
        h[i] = dist(rng);

    thrust::device_vector<float> d = h;

    // -------- warmup --------
    thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());
    cudaDeviceSynchronize();
    // ------------------------

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float sum = thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << sum << std::endl;
    std::cout << ms << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}