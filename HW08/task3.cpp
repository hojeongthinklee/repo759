#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

#include "msort.h"

int main(int argc, char* argv[]) {

    if (argc != 4) return 1;

    const std::size_t n = std::stoull(argv[1]);
    const int t = std::stoi(argv[2]);
    const std::size_t threshold = std::stoull(argv[3]);

    omp_set_num_threads(t);

    std::vector<int> arr(n);

    // random input
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (std::size_t i = 0; i < n; ++i)
        arr[i] = dist(rng);

    const auto start = std::chrono::high_resolution_clock::now();

    msort(arr.data(), n, threshold);

    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << arr.front() << '\n';
    std::cout << arr.back() << '\n';
    std::cout << elapsed.count() << '\n';

    return 0;
}
