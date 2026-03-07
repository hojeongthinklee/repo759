#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>

#include <iostream>
#include <random>
#include <vector>

using namespace cub;

CachingDeviceAllocator g_allocator(true);

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: ./task1_cub n\n";
        return 1;
    }

    int n = atoi(argv[1]);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_in(n);

    for (int i = 0; i < n; i++)
        h_in[i] = dist(rng);

    float *d_in = NULL;
    float *d_out = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float)*n);
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in.data(), sizeof(float)*n, cudaMemcpyHostToDevice);

    // temp storage size query
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // -------- warmup --------
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();
    // ------------------------

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float sum;
    cudaMemcpy(&sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << sum << std::endl;
    std::cout << ms << std::endl;

    g_allocator.DeviceFree(d_in);
    g_allocator.DeviceFree(d_out);
    g_allocator.DeviceFree(d_temp_storage);

    return 0;
}