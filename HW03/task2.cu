#include <cuda_runtime.h>
#include <iostream>
#include <random>

__global__ void kernel(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int idx = y * blockDim.x + x;   // match expected output order
    dA[idx] = a * x + y;
}

int main() {
    constexpr int N = 16;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 20);
    int a = dist(gen);

    int *dA = nullptr;
    cudaMalloc(&dA, N * sizeof(int));

    kernel<<<2, 8>>>(dA, a);
    cudaDeviceSynchronize();

    int hA[N];
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (i) std::cout << ' ';
        std::cout << hA[i];
    }
    std::cout << '\n';

    cudaFree(dA);
    return 0;
}
