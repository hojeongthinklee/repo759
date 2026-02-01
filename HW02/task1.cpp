#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "scan.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task1 n\n";
        return 1;
    }

    // Read n from command line
    std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    if (n == 0) {
        std::cerr << "Error: n must be positive.\n";
        return 1;
    }

    // Allocate memory
    float *arr = new float[n];
    float *output = new float[n];

    // Generate random floats in [-1.0, 1.0]
    std::mt19937 gen(12345);  // fixed seed
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }

    // Time only the scan function
    auto start = std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print results (each on its own line)
    std::cout << elapsed.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n - 1] << "\n";


    // Free memory
    delete[] arr;
    delete[] output;

    return 0;
}
