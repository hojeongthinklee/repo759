#include "scan.h"

// Inclusive scan: output[i] = arr[0] + arr[1] + ... + arr[i]
void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) return;

    output[0] = arr[0];
    for (std::size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + arr[i];
    }
}