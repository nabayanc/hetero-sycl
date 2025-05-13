// src/schedulers/bandit_scheduler.cpp

#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <cmath>
#include <numeric>

namespace spmv {

class GradientBanditScheduler : public IScheduler {
public:
  /// learning rate for preferences, and baseline smoothing rate
  explicit GradientBanditScheduler(double alpha = 0.1, double beta = 0.1)
    : alpha_(alpha), beta_(beta), baseline_(0.0)
  {}

  const char* name() const noexcept override { return "bandit"; }
  bool is_dynamic() const noexcept override { return false; }

  // 1) On each iteration decide how many rows to give each device
  std::vector<Part>
  make_plan(int nrows,
            const std::vector<sycl::queue>& devices) override
  {
    int D = int(devices.size());
    // initialize H on first call
    if (H_.size() != (size_t)D) {
      H_.assign(D, 0.0);
      last_devices_ = devices;
    }

    // softmax preferences → probabilities p[i]
    double maxH = *std::max_element(H_.begin(), H_.end());
    std::vector<double> exps(D);
    double sumexp = 0.0;
    for (int i = 0; i < D; ++i) {
      exps[i] = std::exp(H_[i] - maxH);
      sumexp += exps[i];
    }

    std::vector<double> p(D);
    for (int i = 0; i < D; ++i)
      p[i] = exps[i] / sumexp;

    // partition rows ~ p[i]
    std::vector<Part> parts;
    parts.reserve(D);
    int offset = 0;
    for (int i = 0; i < D; ++i) {
      int rows = (i+1 == D)
               ? (nrows - offset)
               : int(std::round(p[i] * nrows));
      int end = std::min(nrows, offset + rows);
      parts.push_back({ offset, end, const_cast<sycl::queue*>(&devices[i]) });
      offset = end;
    }
    if (offset < nrows)
      parts.back().row_end = nrows;

    last_probs_   = p;
    last_parts_   = parts;
    last_devices_ = devices;
    return parts;
  }

  // 2) After each iteration we get per‐device kernel times
  //    we compute a single reward = total_rows / total_kernel_ms
  //    then update baseline & preferences by the gradient‐bandit rule.
  void update_times(const std::vector<double>& times) override {
    int D = int(times.size());
    // how many rows each device got
    int total_rows = 0;
    double total_time = 0.0;
    for (int i = 0; i < D; ++i) {
      int rows = 0;
      for (auto &p : last_parts_) {
        if (p.q == &last_devices_[i])
          rows += (p.row_end - p.row_begin);
      }
      total_rows += rows;
      total_time += times[i];
    }
    if (total_time <= 0.0) return;
    double R = double(total_rows) / total_time;  // rows per ms

    // update baseline with smoothing
    baseline_ += beta_ * (R - baseline_);

    // gradient‐bandit preference update
    for (int i = 0; i < D; ++i) {
      // ∂log p[i] / ∂H[j] = (1 - p[i])  if j==i;  = -p[j] else
      double grad = (R - baseline_);
      H_[i] += alpha_ * grad * (1.0 - last_probs_[i]);
      for (int j = 0; j < D; ++j) {
        if (j == i) continue;
        H_[j] -= alpha_ * grad * last_probs_[j];
      }
    }
  }

private:
  double                       alpha_, beta_;
  double                       baseline_;
  std::vector<double>          H_;
  std::vector<double>          last_probs_;
  std::vector<Part>            last_parts_;
  std::vector<sycl::queue>     last_devices_;
};

REGISTER_SCHEDULER("bandit", GradientBanditScheduler);

} // namespace spmv
