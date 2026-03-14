#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>

#include "matmul.h"

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

    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n);

    // fill A and B
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A[i * n + j] = static_cast<float>(i + j);
            B[i * n + j] = static_cast<float>(i - j);
        }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    mmul(A.data(), B.data(), C.data(), n);
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << C.front() << '\n';
    std::cout << C.back() << '\n';
    std::cout << elapsed.count() << '\n';

    return 0;
}
