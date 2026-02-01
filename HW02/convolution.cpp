#include "convolution.h"

#include <cstddef>

// Boundary rule from the assignment:
// - inside image: use image value
// - exactly one index out of range: pad with 1 (edge, excluding corners)
// - both indices out of range: pad with 0 (corners)
static inline float padded_value(const float* image,
                                 std::size_t n,
                                 long long i,
                                 long long j) {
    const bool in_i = (0 <= i) && (i < static_cast<long long>(n));
    const bool in_j = (0 <= j) && (j < static_cast<long long>(n));

    if (in_i && in_j) {
        return image[static_cast<std::size_t>(i) * n +
                     static_cast<std::size_t>(j)];
    }
    if (in_i || in_j) {
        return 1.0f;
    }
    return 0.0f;
}

void convolve(const float *image,
              float *output,
              std::size_t n,
              const float *mask,
              std::size_t m) {
    if (n == 0 || m == 0) return;

    const long long r = static_cast<long long>((m - 1) / 2);

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {

            float sum = 0.0f;

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {

                    long long xi = static_cast<long long>(x) +
                                   static_cast<long long>(i) - r;
                    long long yj = static_cast<long long>(y) +
                                   static_cast<long long>(j) - r;

                    sum += mask[i * m + j] *
                           padded_value(image, n, xi, yj);
                }
            }

            output[x * n + y] = sum;
        }
    }
}
