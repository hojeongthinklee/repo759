#include <omp.h>

#include <iostream>

long long factorial(int n) {
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int main() {
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        #pragma omp single
        {
            std::cout << "Number of threads: " << nthreads << '\n';
        }

        std::cout << "I am thread No. " << tid << '\n';

        #pragma omp for
        for (int i = 1; i <= 8; ++i) {
            std::cout << i << "!=" << factorial(i) << '\n';
        }
    }

    return 0;
}