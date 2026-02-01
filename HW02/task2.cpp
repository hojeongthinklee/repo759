#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "convolution.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n m\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::size_t m = static_cast<std::size_t>(std::stoull(argv[2]));

    if (n == 0 || m == 0 || m % 2 == 0) {
        std::cerr << "Error: n must be positive and m must be a positive odd number.\n";
        return 1;
    }

    // Allocate arrays (row-major)
    float* image = new float[n * n];
    float* mask  = new float[m * m];
    float* output = new float[n * n];

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = dist_img(rng);
    }
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = dist_mask(rng);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    // No precision requirement
    std::cout << elapsed.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
