#include "matmul.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

static double ms(const std::chrono::high_resolution_clock::time_point& a,
                 const std::chrono::high_resolution_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    const unsigned int n = 1024; // >= 1000
    const std::size_t N = static_cast<std::size_t>(n);
    const std::size_t NN = N * N;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double* A = new double[NN];
    double* B = new double[NN];
    double* C = new double[NN];

    for (std::size_t t = 0; t < NN; ++t) {
        A[t] = dist(rng);
        B[t] = dist(rng);
    }

    std::vector<double> Av(NN), Bv(NN);
    for (std::size_t t = 0; t < NN; ++t) {
        Av[t] = A[t];
        Bv[t] = B[t];
    }

    std::cout << n << "\n";

    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul1(A, B, C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << ms(t0, t1) << "\n" << C[NN - 1] << "\n";
    }
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul2(A, B, C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << ms(t0, t1) << "\n" << C[NN - 1] << "\n";
    }
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul3(A, B, C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << ms(t0, t1) << "\n" << C[NN - 1] << "\n";
    }
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmul4(Av, Bv, C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << ms(t0, t1) << "\n" << C[NN - 1] << "\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
