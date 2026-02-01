#include <cstddef>
#include <iostream>

#include "convolution.h"

int main() {
    const std::size_t n = 4;
    const std::size_t m = 3;

    // Example input image f (row-major)
    const float f[n * n] = {
        1, 3, 4, 8,
        6, 5, 2, 4,
        3, 4, 6, 8,
        1, 4, 5, 2
    };

    // Example mask w (row-major)
    const float w[m * m] = {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
    };

    float g[n * n] = {0};

    // Call your implementation
    convolve(f, g, n, w, m);

    // Print g as a 4x4 matrix
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::cout << g[i * n + j];
            if (j + 1 < n) std::cout << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
