// src/schedulers/locality_block.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <vector>
#include <algorithm>
#include <sycl/sycl.hpp>

namespace spmv {

/**
 * @brief Locality-Aware Block Scheduler
 *
 * Divides nrows into D contiguous blocks (one per device), just like
 * StaticBlockScheduler, but then realigns each block’s start/end to the
 * nearest multiple of `chunk_size` rows (except we force the last block
 * to end at nrows).  This ensures that each device reads whole “chunks”
 * of the CSR arrays, improving cache/TLB behavior.
 */
class LocalityAwareBlockScheduler : public IScheduler {
public:
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    const int D    = int(queues.size());
    const int base = nrows / D;
    const int rem  = nrows % D;

    // Tuneable: number of rows per alignment boundary
    const int chunk_size = 256;

    std::vector<Part> parts;
    parts.reserve(D);

    int prev_end = 0;
    for (int i = 0; i < D; ++i) {
      // 1) start/end of the “ideal” static block
      int rows = base + (i < rem ? 1 : 0);
      int start = prev_end;
      int end   = start + rows;

      // 2) realign to chunk boundaries
      if (i > 0) {
        // round start *up* to next multiple of chunk_size
        int aligned = ((start + chunk_size - 1) / chunk_size) * chunk_size;
        start = std::min(aligned, end);
      }
      if (i < D - 1) {
        // round end *down* to previous multiple of chunk_size
        int aligned = (end / chunk_size) * chunk_size;
        end = std::max(start, aligned);
      } else {
        // last block must reach the end
        end = nrows;
      }

      parts.push_back( Part{
        /*row_begin=*/ start,
        /*row_end  =*/ end,
        /*q        =*/ const_cast<sycl::queue*>(&queues[i])
      } );

      prev_end = end;
    }

    return parts;
  }

  const char* name() const noexcept override {
    return "locality_block";
  }
};

// register under the name "locality_block"
REGISTER_SCHEDULER("locality_block", LocalityAwareBlockScheduler);

} // namespace spmv
