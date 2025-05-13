#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <numeric>

namespace spmv {

class StaticSplit final : public IScheduler {
public:
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& devs) override
  {
    std::vector<Part> parts;
    parts.reserve(devs.size());

    const float weights[] = {0.4f, 0.4f, 0.2f}; // GPU0, GPU1, CPU
    int row0 = 0;
    for(size_t i = 0; i < devs.size(); ++i) {
      int rows = static_cast<int>(nrows * weights[i]);
      parts.push_back({row0, row0 + rows, const_cast<sycl::queue*>(&devs[i])});
      row0 += rows;
    }
    parts.back().row_end = nrows;                // catch rounding
    return parts;
  }
  const char* name() const noexcept override { return "static_split"; }
};

REGISTER_SCHEDULER("static", StaticSplit);

} // namespace spmv
