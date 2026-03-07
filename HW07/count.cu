#include "count.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts)
{
    if (d_in.empty()) {
        values.clear();
        counts.clear();
        return;
    }

    // d_in is const, so work on a copy
    thrust::device_vector<int> sorted = d_in;
    thrust::sort(sorted.begin(), sorted.end());

    const int n = static_cast<int>(sorted.size());

    // Worst case: every element is unique
    values.resize(n);
    counts.resize(n);

    auto ones_begin = thrust::make_constant_iterator<int>(1);

    auto new_end = thrust::reduce_by_key(
        sorted.begin(),
        sorted.end(),
        ones_begin,
        values.begin(),
        counts.begin()
    );

    values.resize(new_end.first - values.begin());
    counts.resize(new_end.second - counts.begin());
}