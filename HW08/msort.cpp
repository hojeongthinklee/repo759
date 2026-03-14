#include "msort.h"
#include <algorithm>
#include <vector>

// merge two sorted halves
static void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right) {
    std::vector<int> temp(right - left);

    std::size_t i = left;
    std::size_t j = mid;
    std::size_t k = 0;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while (i < mid) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];

    for (std::size_t t = 0; t < k; ++t) {
        arr[left + t] = temp[t];
    }
}

// recursive mergesort
static void msort_rec(int* arr, std::size_t left, std::size_t right, std::size_t threshold) {

    std::size_t size = right - left;

    // use serial sort for small arrays
    if (size <= threshold) {
        std::sort(arr + left, arr + right);
        return;
    }

    std::size_t mid = left + size / 2;

    // parallel tasks
    #pragma omp task shared(arr)
    msort_rec(arr, left, mid, threshold);

    #pragma omp task shared(arr)
    msort_rec(arr, mid, right, threshold);

    #pragma omp taskwait

    merge(arr, left, mid, right);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {

    #pragma omp parallel
    {
        #pragma omp single
        msort_rec(arr, 0, n, threshold);
    }
}
