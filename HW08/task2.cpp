#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>

#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoull(argv[1]));
    const int t = std::stoi(argv[2]);

    if (n == 0 || t < 1 || t > 20) {
        return 1;
    }

    omp_set_num_threads(t);

    const std::size_t m = 3;

    std::vector<float> image(n * n);
    std::vector<float> output(n * n);
    std::vector<float> mask(m * m);

    // fill image
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            image[i * n + j] = static_cast<float>(i + j);
        }
    }

    // fill mask
    mask[0] = 0.0f; mask[1] = 1.0f; mask[2] = 0.0f;
    mask[3] = 1.0f; mask[4] = 4.0f; mask[5] = 1.0f;
    mask[6] = 0.0f; mask[7] = 1.0f; mask[8] = 0.0f;

    const auto start = std::chrono::high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), m);
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << output.front() << '\n';
    std::cout << output.back() << '\n';
    std::cout << elapsed.count() << '\n';

    return 0;
}
