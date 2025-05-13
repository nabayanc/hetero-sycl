// src/schedulers/workstealing.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

#include <vector>
#include <algorithm>
#include <limits>

namespace spmv {

/**
 * @brief Greedy “work‐stealing” scheduler:
 *   1) Chop the rows into fixed‐size chunks.
 *   2) Keep a running “load” total for each device.
 *   3) For each chunk, assign it to the device with the smallest current load.
 */
class WorkStealingScheduler : public IScheduler {
public:
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    const int D = int(queues.size());
    // tuneable chunk‐size (in rows)
    const int CHUNK = 256;

    // 1) build the list of row‐chunks
    struct Chunk { int begin, end; };
    std::vector<Chunk> chunks;
    for (int start = 0; start < nrows; start += CHUNK) {
      int end = std::min(nrows, start + CHUNK);
      chunks.push_back({start, end});
    }

    // 2) greedy assign each chunk to the least‐loaded device
    std::vector<int> load(D, 0);
    std::vector<Part> parts;
    parts.reserve(chunks.size());

    for (auto& c : chunks) {
      // find device with min load
      int best = 0;
      int min_load = std::numeric_limits<int>::max();
      for (int i = 0; i < D; ++i) {
        if (load[i] < min_load) {
          min_load = load[i];
          best = i;
        }
      }
      // assign chunk to queues[best]
      parts.push_back( Part{
        /*row_begin=*/ c.begin,
        /*row_end  =*/ c.end,
        /*q        =*/ const_cast<sycl::queue*>(&queues[best])
      });
      // update that device’s load by rows count
      load[best] += (c.end - c.begin);
    }

    return parts;
  }

  const char* name() const noexcept override {
    return "workstealing";
  }
};

// register under the name "workstealing"
REGISTER_SCHEDULER("workstealing", WorkStealingScheduler);

} // namespace spmv
