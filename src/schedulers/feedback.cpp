// src/schedulers/feedback.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace spmv {

class FeedbackScheduler : public IScheduler {
public:
  FeedbackScheduler() : inited_(false) {}

  std::vector<Part>
  make_plan(int nrows,
            const std::vector<sycl::queue>& queues) override {
    const int D = int(queues.size());
    std::vector<Part> parts;
    parts.reserve(D);

    if (!inited_) {
      // --- 1) First iteration: static block split ---
      int base = nrows / D;
      int rem  = nrows % D;
      prev_rows_.resize(D);
      prev_times_.assign(D, 1.0);  // start with uniform weights

      int start = 0;
      for (int i = 0; i < D; ++i) {
        int rows = base + (i < rem ? 1 : 0);
        prev_rows_[i] = rows;

        parts.push_back( Part{
          /*row_begin=*/ start,
          /*row_end  =*/ start + rows,
          /*q        =*/ const_cast<sycl::queue*>(&queues[i])
        });
        start += rows;
      }
      inited_ = true;
    } else {
      // --- 2) Subsequent iterations: feedback adjust ---
      // compute average time
      double avg = std::accumulate(prev_times_.begin(),
                                   prev_times_.end(), 0.0) / D;

      // target sizes ∝ 1 / time_i
      std::vector<double> target(D);
      for (int i = 0; i < D; ++i)
        target[i] = prev_rows_[i] * (avg / prev_times_[i]);

      // scale so sum(target) == nrows
      double sum_t = std::accumulate(target.begin(), target.end(), 0.0);
      double scale = double(nrows) / sum_t;
      for (auto& t : target) t *= scale;

      // floor to ints, track fractional parts
      std::vector<int> new_rows(D);
      std::vector<std::pair<double,int>> frac;
      frac.reserve(D);
      int assigned = 0;
      for (int i = 0; i < D; ++i) {
        int r = int(std::floor(target[i]));
        new_rows[i] = r;
        assigned   += r;
        frac.emplace_back(target[i] - r, i);
      }

      // distribute the remainder to largest fractions
      int rem = nrows - assigned;
      std::sort(frac.begin(), frac.end(),
                [](auto &a, auto &b){ return a.first > b.first; });
      for (int k = 0; k < rem; ++k)
        new_rows[ frac[k].second ]++;

      // clamp zeros
        for (int i = 0; i < D; ++i) {
            if (new_rows[i] == 0) {
            int j = int(std::distance(new_rows.begin(),
                        std::max_element(new_rows.begin(), new_rows.end())));
            new_rows[j]--;
            new_rows[i] = 1;
            }
        }

      prev_rows_ = new_rows;

      // build parts contiguously
      int start = 0;
      for (int i = 0; i < D; ++i) {
        int rows = prev_rows_[i];
        parts.push_back( Part{
          /*row_begin=*/ start,
          /*row_end  =*/ start + rows,
          /*q        =*/ const_cast<sycl::queue*>(&queues[i])
        });
        start += rows;
      }
    }

    return parts;
  }

  const char* name() const noexcept override {
    return "feedback";
  }

  /// Call this *after* each iteration, passing the measured
  /// kernel time for each device (size == D).
  void update_times(const std::vector<double>& times) override {
    prev_times_ = times;
  }

private:
  bool               inited_;
  std::vector<int>    prev_rows_;
  std::vector<double> prev_times_;
};

// register it
REGISTER_SCHEDULER("feedback", FeedbackScheduler);

} // namespace spmv
