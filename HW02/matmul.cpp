#include "matmul.h"
#include <cstddef>
#include <vector>

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t N = static_cast<std::size_t>(n);

    // zero C
    for (std::size_t t = 0; t < N * N; ++t) C[t] = 0.0;

    // i, j, k
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            for (unsigned int k = 0; k < n; ++k) {
                C[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j)] +=
                    A[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(k)] *
                    B[static_cast<std::size_t>(k) * N + static_cast<std::size_t>(j)];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t N = static_cast<std::size_t>(n);

    // zero C
    for (std::size_t t = 0; t < N * N; ++t) C[t] = 0.0;

    // i, k, j
    for (unsigned int i = 0; i < n; ++i) {
        const std::size_t ci = static_cast<std::size_t>(i) * N;
        const std::size_t ai = static_cast<std::size_t>(i) * N;

        for (unsigned int k = 0; k < n; ++k) {
            const double aik = A[ai + static_cast<std::size_t>(k)];
            const std::size_t bk = static_cast<std::size_t>(k) * N;

            for (unsigned int j = 0; j < n; ++j) {
                C[ci + static_cast<std::size_t>(j)] += aik * B[bk + static_cast<std::size_t>(j)];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t N = static_cast<std::size_t>(n);

    // zero C
    for (std::size_t t = 0; t < N * N; ++t) C[t] = 0.0;

    // j, k, i
    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int k = 0; k < n; ++k) {
            const double bkj =
                B[static_cast<std::size_t>(k) * N + static_cast<std::size_t>(j)];

            for (unsigned int i = 0; i < n; ++i) {
                C[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j)] +=
                    A[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(k)] * bkj;
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C,
           const unsigned int n) {
    const std::size_t N = static_cast<std::size_t>(n);

    // zero C
    for (std::size_t t = 0; t < N * N; ++t) C[t] = 0.0;

    // same loop order as mmul1: i, j, k
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            for (unsigned int k = 0; k < n; ++k) {
                C[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j)] +=
                    A[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(k)] *
                    B[static_cast<std::size_t>(k) * N + static_cast<std::size_t>(j)];
            }
        }
    }
}
