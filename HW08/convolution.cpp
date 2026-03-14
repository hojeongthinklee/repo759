#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    const std::size_t r = m / 2;

    // parallel over pixels
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;

            for (std::size_t u = 0; u < m; ++u) {
                for (std::size_t v = 0; v < m; ++v) {
                    const int ii = static_cast<int>(i) + static_cast<int>(u) - static_cast<int>(r);
                    const int jj = static_cast<int>(j) + static_cast<int>(v) - static_cast<int>(r);

                    if (ii >= 0 && ii < static_cast<int>(n) &&
                        jj >= 0 && jj < static_cast<int>(n)) {
                        sum += image[static_cast<std::size_t>(ii) * n + static_cast<std::size_t>(jj)] *
                               mask[u * m + v];
                    }
                }
            }

            output[i * n + j] = sum;
        }
    }
}
