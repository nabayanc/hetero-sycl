#include <sycl/sycl.hpp>

namespace spmv {

sycl::event
spmv_cpu(sycl::queue& q,
         int nrows,
         const int* row_ptr,
         const int* col_idx,
         const float* vals,
         const float* x,
         float* y)
{
  return q.parallel_for(
    sycl::range<1>(static_cast<size_t>(nrows)),
    [=](sycl::id<1> id) {
      int row = id[0];
      float sum = 0.0f;
      for(int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj)
        sum += vals[jj] * x[col_idx[jj]];
      y[row] = sum;
    });
}

} // namespace spmv
