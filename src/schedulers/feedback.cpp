// src/schedulers/feedback.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits> // Required for std::numeric_limits

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

    if (D == 0) { // Handle case with no devices
        return parts;
    }

    if (!inited_) {
      // --- 1) First iteration: static block split ---
      prev_rows_.resize(D);
      prev_times_.assign(D, 1.0);  // start with uniform weights (non-zero)

      int base = nrows / D;
      int rem  = nrows % D;
      int start_row_idx = 0;
      for (int i = 0; i < D; ++i) {
        int rows_for_device = base + (i < rem ? 1 : 0);
        prev_rows_[i] = rows_for_device;

        int end_row_idx = start_row_idx + rows_for_device;
        // Ensure end_row_idx does not exceed nrows, especially if nrows is small
        if (end_row_idx > nrows) end_row_idx = nrows;
        if (start_row_idx > end_row_idx) start_row_idx = end_row_idx; // Should not happen with correct logic


        parts.push_back( Part{
          /*row_begin=*/ start_row_idx,
          /*row_end  =*/ end_row_idx,
          /*q        =*/ const_cast<sycl::queue*>(&queues[i])
        });
        start_row_idx = end_row_idx;
      }
      // Ensure the last part covers up to nrows if rounding caused a mismatch
      if (!parts.empty() && parts.back().row_end < nrows && start_row_idx < nrows) {
          parts.back().row_end = nrows;
      }
      inited_ = true;
    } else {
      // --- 2) Subsequent iterations: feedback adjust ---
      double total_prev_rows = 0;
      for(int r : prev_rows_) total_prev_rows += r;
      if (total_prev_rows == 0 && nrows > 0) { // Safety if previous distribution was all zeros but now we have rows
          // Fallback to static split for one iteration to get new timings
          prev_rows_.assign(D, 0);
          prev_times_.assign(D, 1.0);
          int base = nrows / D;
          int rem  = nrows % D;
          int start_row_idx = 0;
          for (int i = 0; i < D; ++i) {
            prev_rows_[i] = base + (i < rem ? 1 : 0);
            parts.push_back( Part{ start_row_idx, start_row_idx + prev_rows_[i], const_cast<sycl::queue*>(&queues[i]) });
            start_row_idx += prev_rows_[i];
          }
          if (!parts.empty() && parts.back().row_end < nrows && start_row_idx < nrows) {
             parts.back().row_end = nrows;
          }
          return parts; // Return early after re-initializing with static split
      }


      // compute average time per row for each device, then use that to estimate work distribution.
      // If a device had 0 rows previously, its prev_times_ entry will be the epsilon from update_times.
      double sum_inverse_time_per_row = 0.0;
      std::vector<double> performance_metric(D);
      for(int i=0; i<D; ++i) {
          if (prev_rows_[i] > 0) {
              // prev_times_[i] is already sanitized by update_times to be non-zero
              performance_metric[i] = static_cast<double>(prev_rows_[i]) / prev_times_[i]; // rows per ms (speed)
          } else {
              // If it had no rows, its "speed" is effectively unknown or very low for this calculation.
              // To give it a chance to get work, assign a small baseline speed.
              // Or, rely on the fact that its prev_times_ is small epsilon, so speed would be 0 / epsilon = 0.
              // Let's give it a small chance by assigning a fraction of average speed if others exist.
              performance_metric[i] = 0; // Will be handled by giving it some rows later if it gets 0
          }
          sum_inverse_time_per_row += performance_metric[i];
      }
      
      // If all devices had zero performance (e.g. all prev_rows were 0, or all times were huge)
      // then fall back to equal distribution.
      if (sum_inverse_time_per_row < 1e-9 && nrows > 0) {
          int base = nrows / D;
          int rem  = nrows % D;
          int start_row_idx = 0;
          for (int i = 0; i < D; ++i) {
            prev_rows_[i] = base + (i < rem ? 1 : 0);
            parts.push_back( Part{ start_row_idx, start_row_idx + prev_rows_[i], const_cast<sycl::queue*>(&queues[i]) });
            start_row_idx += prev_rows_[i];
          }
          if (!parts.empty() && parts.back().row_end < nrows && start_row_idx < nrows) {
             parts.back().row_end = nrows;
          }
          return parts;
      }


      std::vector<double> target_row_fractions(D);
      for(int i=0; i<D; ++i) {
          if (sum_inverse_time_per_row > 1e-9) {
            target_row_fractions[i] = performance_metric[i] / sum_inverse_time_per_row;
          } else { // Should be caught by above, but as safeguard
            target_row_fractions[i] = 1.0 / D;
          }
      }

      std::vector<int> new_rows(D);
      int assigned_rows_total = 0;
      for (int i = 0; i < D; ++i) {
        new_rows[i] = static_cast<int>(std::round(target_row_fractions[i] * nrows));
        assigned_rows_total += new_rows[i];
      }

      // Adjust for rounding errors to ensure total rows == nrows
      int current_sum = 0;
      for(int r : new_rows) current_sum +=r;
      int diff = nrows - current_sum;
      
      // Distribute diff (can be positive or negative)
      // Prefer adding to/removing from devices with larger shares or based on some other heuristic
      // For simplicity, add/remove from devices cyclically or based on largest fractional parts (more complex)
      // Simple cyclic distribution of diff:
      if (diff != 0) {
          for (int i = 0; i < std::abs(diff); ++i) {
              int dev_to_adjust = i % D;
              if (diff > 0) {
                  new_rows[dev_to_adjust]++;
              } else if (new_rows[dev_to_adjust] > 0) { // Only decrement if rows > 0
                  new_rows[dev_to_adjust]--;
              }
          }
      }


      // Ensure no device gets 0 rows if nrows > 0, and at least one device has >0 rows
      // This 'clamp zeros' logic needs to be careful not to take from a device that also needs a row.
      bool all_zero = true;
      for(int r : new_rows) if(r > 0) all_zero = false;

      if (nrows > 0 && !all_zero) { // Only if there's work and not all devices are meant to be zero
          for (int i = 0; i < D; ++i) {
              if (new_rows[i] == 0) {
                  // Find a device with the most rows to steal one from
                  int max_rows = 0;
                  int donor_idx = -1;
                  for (int j = 0; j < D; ++j) {
                      if (new_rows[j] > max_rows) {
                          max_rows = new_rows[j];
                          donor_idx = j;
                      }
                  }
                  if (donor_idx != -1 && new_rows[donor_idx] > 1) { // Ensure donor has more than 1 to give
                      new_rows[donor_idx]--;
                      new_rows[i] = 1;
                  } else if (donor_idx != -1 && D == 1) { // Single device must do all work
                      new_rows[i] = nrows; // Should already be this
                  } else if (donor_idx == -1 && nrows > 0) { // All have 0 or 1, but sum != nrows
                      // This case is complex, might need fallback to static split if distribution is bad
                      // For now, if no clear donor, this device might stay at 0 if others also have few.
                      // If total sum matches nrows this shouldn't leave valid work for a 0-row device.
                  }
              }
          }
      }
      // Final check if sum of new_rows matches nrows after clamping. If not, adjust last device.
      current_sum = 0;
      for(int r : new_rows) current_sum +=r;
      if (current_sum != nrows && D > 0) {
          new_rows[D-1] += (nrows - current_sum);
          if (new_rows[D-1] < 0) new_rows[D-1] = 0; // Ensure non-negative
          // Re-check sum and redistribute if last device became 0 and took all work from others
          // This part can get complicated; simpler might be to ensure the total sum is right
          // and accept that some devices might get 0 if nrows is small.
      }


      prev_rows_ = new_rows;

      int start_row_idx = 0;
      for (int i = 0; i < D; ++i) {
        int rows_for_device = prev_rows_[i];
        if (rows_for_device < 0) rows_for_device = 0; // Defensive
        int end_row_idx = start_row_idx + rows_for_device;
        if (end_row_idx > nrows) end_row_idx = nrows;


        parts.push_back( Part{
          /*row_begin=*/ start_row_idx,
          /*row_end  =*/ end_row_idx,
          /*q        =*/ const_cast<sycl::queue*>(&queues[i])
        });
        start_row_idx = end_row_idx;
      }
       // Ensure the last part covers up to nrows if rounding caused a mismatch
      if (!parts.empty() && parts.back().row_end < nrows && start_row_idx < nrows) {
          parts.back().row_end = nrows;
      }
    }

    return parts;
  }

  const char* name() const noexcept override {
    return "feedback";
  }

  void update_times(const std::vector<double>& times) override {
    prev_times_.resize(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
        if (times[i] > 1e-9) { // Use a small positive threshold (e.g. 1ns)
            prev_times_[i] = times[i];
        } else {
            // If time is zero or extremely small, assign a tiny positive value.
            // This represents a very fast device but avoids division by zero.
            // The magnitude should be significantly smaller than typical non-zero kernel times.
            prev_times_[i] = 1e-6; // e.g., 1 microsecond (if times are in ms, this is 1ns)
        }
    }
  }

private:
  bool               inited_;
  std::vector<int>    prev_rows_;
  std::vector<double> prev_times_;
};

REGISTER_SCHEDULER("feedback", FeedbackScheduler);

} // namespace spmv