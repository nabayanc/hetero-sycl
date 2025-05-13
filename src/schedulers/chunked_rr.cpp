// src/schedulers/chunked_rr.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <vector>
#include <algorithm>

namespace spmv {

/**
 * @brief Chunk‐Round‐Robin scheduler: slice the rows into K small chunks
 * and assign them in round-robin order across the D devices.
 */
class ChunkedRRScheduler : public IScheduler {
public:
  std::vector<Part>
  make_plan(int nrows,
            const std::vector<sycl::queue>& queues) override {
    const int D = int(queues.size());
    // how many chunks per device (tuneable)
    const int CHUNKS_PER_DEVICE = 4;
    const int K = D * CHUNKS_PER_DEVICE;

    // compute chunk size (ceiling division)
    int chunk_size = (nrows + K - 1) / K;

    std::vector<Part> parts;
    parts.reserve(K);

    int row_begin = 0;
    int chunk_id  = 0;
    while (row_begin < nrows) {
      int row_end = std::min(nrows, row_begin + chunk_size);
      // assign chunk chunk_id → device (chunk_id % D)
      sycl::queue* q = const_cast<sycl::queue*>(&queues[chunk_id % D]);

      parts.push_back( Part{ row_begin, row_end, q } );

      row_begin = row_end;
      ++chunk_id;
    }

    return parts;
  }

  const char* name() const noexcept override {
    return "chunked_rr";
  }
};

// register under the name "chunked_rr"
REGISTER_SCHEDULER("chunked_rr", ChunkedRRScheduler);

} // namespace spmv
