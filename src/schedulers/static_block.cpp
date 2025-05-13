#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

namespace spmv {

/**
 * @brief A scheduler that divides the rows into D contiguous blocks,
 *        one per device, as evenly as possible.
 */
class StaticBlockScheduler : public IScheduler {
public:
  // Partition nrows across devices into contiguous blocks
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    const int D    = int(queues.size());
    const int base = nrows / D;
    const int rem  = nrows % D;

    std::vector<Part> parts;
    parts.reserve(D);

    int row_start = 0;
    for (int i = 0; i < D; ++i) {
      int rows    = base + (i < rem ? 1 : 0);
      int row_end = row_start + rows;

      parts.push_back( Part{
        /*row_begin=*/ row_start,
        /*row_end  =*/ row_end,
        /*q        =*/ const_cast<sycl::queue*>(&queues[i])
      } );

      row_start = row_end;
    }
    return parts;
  }

  // Name used in the factory
  const char* name() const noexcept override {
    return "static_block";
  }
};

// Register under the name "static_block"
REGISTER_SCHEDULER("static_block", StaticBlockScheduler);

} // namespace spmv
