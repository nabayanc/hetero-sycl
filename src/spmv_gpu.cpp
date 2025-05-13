#include <sycl/sycl.hpp>
#include "spmv/csr.hpp"

namespace spmv {

sycl::event
spmv_gpu(sycl::queue& q,
         int nrows,
         const int* row_ptr,
         const int* col_idx,
         const float* vals,
         const float* x,
         float* y)
{
  constexpr int WG_SZ = 128;
  constexpr int ROWS_PER_WG = 4;

  sycl::range<1> g(((nrows + ROWS_PER_WG - 1) / ROWS_PER_WG) * WG_SZ);
  sycl::range<1> l(WG_SZ);

  return q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<1>{g, l},
      [=](sycl::nd_item<1> it)
        [[sycl::reqd_sub_group_size(32)]] {

      const int wg_id = it.get_group(0);
      const int lid   = it.get_local_id(0);
      const int row0  = wg_id * ROWS_PER_WG;

      auto sg = it.get_sub_group();

      for(int r = 0; r < ROWS_PER_WG; ++r) {
        int row = row0 + r;
        if(row >= nrows) break;

        int start = row_ptr[row];
        int end   = row_ptr[row + 1];

        float sum = 0.0f;
        for(int jj = start + lid; jj < end; jj += WG_SZ)
          sum += vals[jj] * x[col_idx[jj]];

        for(int off = sg.get_local_range()[0] / 2; off > 0; off >>= 1)
          sum += sycl::shift_group_left(sg, sum, off);

        if(lid == 0) y[row] = sum;
      }
    });
  });
}

} // namespace spmv
