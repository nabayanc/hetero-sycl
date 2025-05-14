// src/schedulers/static_split.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <numeric>
#include <vector>
#include <algorithm>

namespace spmv {

class StaticSplit final : public IScheduler {
public:
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& devs) override
  {
    std::vector<Part> parts;
    if (devs.empty() || nrows == 0) {
        return parts;
    }
    parts.reserve(devs.size());

    const float weights[] = {0.45f, 0.45f, 0.10f}; 
    const int num_defined_weights = sizeof(weights) / sizeof(float);

    int row0 = 0;
    int assigned_rows_total = 0;
    size_t num_devices_to_assign = std::min((size_t)num_defined_weights, devs.size());

    for(size_t i = 0; i < num_devices_to_assign; ++i) {
      int rows_for_this_device;
      if (i == num_devices_to_assign - 1) { // Last device being assigned gets the remainder
          rows_for_this_device = nrows - assigned_rows_total;
      } else {
          rows_for_this_device = static_cast<int>(nrows * weights[i]);
      }
      
      int row_end = row0 + rows_for_this_device;
      if (row_end > nrows) row_end = nrows; // Clamp
      if (rows_for_this_device < 0) rows_for_this_device = 0; // Ensure non-negative
      if (row_end < row0) row_end = row0; // Ensure non-decreasing

      parts.push_back({row0, row_end, const_cast<sycl::queue*>(&devs[i])});
      row0 = row_end;
      assigned_rows_total += (row_end - parts.back().row_begin); // Actual rows assigned
    }
    
    // If there are more devices than weights, assign empty parts to them
    for (size_t i = num_devices_to_assign; i < devs.size(); ++i) {
        parts.push_back({nrows, nrows, const_cast<sycl::queue*>(&devs[i])});
    }
    return parts;
  }
  const char* name() const noexcept override { return "static_split"; }
};

REGISTER_SCHEDULER("static_split", StaticSplit);
}