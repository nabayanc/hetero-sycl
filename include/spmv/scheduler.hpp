#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <vector>
#include <string>

#include "spmv/csr.hpp"  // for CSR

namespace spmv {

// A contiguous row‐range assigned to one device
struct Part {
  int           row_begin;
  int           row_end;
  sycl::queue*  q;        // non-owning
};

// Per‐device USM pointers (row_ptr, col_idx, vals, x, y)
struct DeviceMem {
  int*    rp;
  int*    ci;
  float*  v;
  float*  x;
  float*  y;
};

// Record for each launched chunk (used in dynamic scheduler)
struct DevRecord {
  int    iter;
  int    dev_idx;
  int    row_begin;
  int    row_end;
  double launch_ms;
  double kernel_ms;
};

class IScheduler {
public:
  virtual ~IScheduler() = default;

  /// Partition [0,nrows) across the given devices (static/feedback/etc)
  virtual std::vector<Part>
  make_plan(int nrows,
            const std::vector<sycl::queue>& devices) = 0;

  /// Human-readable name
  virtual const char* name() const noexcept = 0;

  /// Feedback hook: give last-iteration kernel times (ms)
  /// Default is no-op; overridden by feedback scheduler.
  virtual void update_times(const std::vector<double>& /*times*/) { }

  /// Return true if this scheduler implements its own dynamic path
  virtual bool is_dynamic() const noexcept { return false; }

  /// Dynamic work-stealing entrypoint.
  /// Only invoked if is_dynamic() returns true.
  virtual void execute_dynamic(
    const CSR&                     A,
    int                            iterations,
    std::vector<sycl::queue>& queues,
    const std::vector<DeviceMem>&   mems,
    std::vector<float>&             h_y,
    std::vector<DevRecord>&         dev_recs,
    double&                         avg_sched_ms,
    double&                         avg_kernel_ms,
    double&                         avg_copy_ms
  ) {
    (void)A; (void)iterations; (void)queues; (void)mems;
    (void)h_y; (void)dev_recs;
    (void)avg_sched_ms; (void)avg_kernel_ms; (void)avg_copy_ms;
  }
};

/* Factory  ------------------------------------------------------------- */
std::unique_ptr<IScheduler>
make_scheduler(const std::string& kind);

} // namespace spmv
