#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    // zero C
    #pragma omp parallel for
    for (std::size_t t = 0; t < n * n; ++t) {
        C[t] = 0.0f;
    }

    // parallel over i
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t ci = i * n;
        const std::size_t ai = i * n;

        for (std::size_t k = 0; k < n; ++k) {
            const float aik = A[ai + k];
            const std::size_t bk = k * n;

            for (std::size_t j = 0; j < n; ++j) {
                C[ci + j] += aik * B[bk + j];
            }
        }
    }
}
