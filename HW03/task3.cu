// task3.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <iomanip>
#include "vscale.cuh"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    unsigned int n = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    if (n == 0) return 1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distB(0.0f, 1.0f);

    std::vector<float> hA(n), hB(n);
    for (unsigned int i = 0; i < n; ++i) {
        hA[i] = distA(gen);
        hB[i] = distB(gen);
    }

    float *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, n *sizeof(float));
    cudaMalloc(&dB, n *sizeof(float));

    cudaMemcpy(dA, hA.data(), n *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), n *sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 512;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<blocks, threads>>>(dA, dB, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hB.data(), dB, n *sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(3) << ms << "\n";
    std::cout << hB.front() << "\n";
    std::cout << hB.back() << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    return 0;
}
