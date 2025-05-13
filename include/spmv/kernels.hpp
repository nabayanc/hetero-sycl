#pragma once
#include <sycl/sycl.hpp>

namespace spmv {

sycl::event spmv_gpu(sycl::queue& q,
                     int nrows,
                     const int* row_ptr,
                     const int* col_idx,
                     const float* vals,
                     const float* x,
                     float* y);

sycl::event spmv_cpu(sycl::queue& q,
                     int nrows,
                     const int* row_ptr,
                     const int* col_idx,
                     const float* vals,
                     const float* x,
                     float* y);

} // namespace spmv
