// src/schedulers/adaptive_load_balancing.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>   // For std::numeric_limits
#include <iostream> // For debugging print statements
#include <vector>

namespace spmv {

class AdaptiveLoadBalancingScheduler : public IScheduler {
public:
  // Constructor
  // chunk_size: the base size for row chunks when partitioning work
  // initial_speed_guess: initial guess for device speed in rows/ms for uninitialized devices
  // speed_alpha: smoothing factor for updating device speeds (0.0 to 1.0); 0.0 means ignore new data, 1.0 means only use new data
  explicit AdaptiveLoadBalancingScheduler(int chunk_size = 256, double initial_speed_guess = 1.0, double speed_alpha = 0.5)
    : chunk_size_(chunk_size), initial_speed_guess_(initial_speed_guess), speed_alpha_(speed_alpha), inited_(false) {}

  // Scheduler name
  const char* name() const noexcept override {
    return "adaptive_lb";
  }

  // Not a dynamic scheduler in the sense of taking over the execution loop
  bool is_dynamic() const noexcept override {
    return false;
  }

  // Plan generation: Determines how rows are partitioned and assigned to devices
  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    const int D = static_cast<int>(queues.size());
    std::vector<Part> parts;
    if (D == 0 || nrows == 0) {
        prev_rows_assigned_by_index_.assign(D, 0);
        return parts;
    }

    // Check if state needs re-initialization (first run or device count changed)
    if (!inited_ || device_speeds_.size() != (size_t)D) {
      // Reset/resize all state vectors to match the current number of devices
      prev_rows_assigned_by_index_.assign(D, 0);
      prev_times_.assign(D, 1e-9); // Initialize with a tiny non-zero time
      device_speeds_.assign(D, initial_speed_guess_); // Initialize with the initial guess
      inited_ = true; // Mark as initialized with this device count

      // --- First iteration plan: Static block split (large blocks) ---
      int base = nrows / D;
      int rem  = nrows % D;
      int current_row_start = 0;
      for (int i = 0; i < D; ++i) {
          int rows_for_device = base + (i < rem ? 1 : 0);
          int current_row_end = current_row_start + rows_for_device;
          if (current_row_end > nrows) current_row_end = nrows; // Clamp

          parts.push_back( Part{
              /*row_begin=*/ current_row_start,
              /*row_end  =*/ current_row_end, // This is a large block, not a chunk
              /*q        =*/ const_cast<sycl::queue*>(&queues[i])
          });
          // Store assigned rows for the *next* update_times call
          prev_rows_assigned_by_index_[i] = current_row_end - current_row_start;

          current_row_start = current_row_end;
      }
      // std::cerr << "AdaptiveLoadBalancingScheduler: Initial static plan generated for " << D << " devices." << std::endl;
      return parts; // <-- RETURN HERE AFTER THE FIRST PLAN

    } else {
      // --- Subsequent iterations plan: Feedback-based adaptive distribution into chunks ---

      // 1) Calculate Device Speeds from previous iteration's results (done in update_times)
      // device_speeds_ should already contain the updated speeds from the last iteration

      // 2) Calculate Target Rows per Device based on Proportional Speed
      // ... (Same logic as before) ...
      std::vector<int> target_rows_per_device(D);
      double total_estimated_speed = 0.0;
      for(int i = 0; i < D; ++i) total_estimated_speed += device_speeds_[i];

      // Fallback to equal split if all estimated speeds are zero or negligible
      if (total_estimated_speed < 1e-9 && nrows > 0) {
          // std::cerr << "AdaptiveLoadBalancingScheduler: Total estimated speed near zero, falling back to static block split for planning." << std::endl;
          device_speeds_.assign(D, 1.0);
          total_estimated_speed = static_cast<double>(D);
      } else if (nrows > 0) {
           bool any_positive_speed = false;
           for(int i=0; i<D; ++i) if(device_speeds_[i] > 1e-9) any_positive_speed = true;
           if (!any_positive_speed) {
               // std::cerr << "AdaptiveLoadBalancingScheduler: No positive estimated speed, falling back to static block split for planning." << std::endl;
               device_speeds_.assign(D, 1.0);
               total_estimated_speed = static_cast<double>(D);
           }
       }

      int assigned_rows_total_target = 0;
      for(int i = 0; i < D; ++i) {
           double target_fraction = (total_estimated_speed > 1e-9) ? (device_speeds_[i] / total_estimated_speed) : (1.0 / D);
           target_rows_per_device[i] = static_cast<int>(std::round(target_fraction * nrows));
           assigned_rows_total_target += target_rows_per_device[i];
      }

      // Adjust for rounding errors
      int diff = nrows - assigned_rows_total_target;
      if (diff != 0) {
           for (int i = 0; i < std::abs(diff); ++i) {
               int dev_to_adjust = i % D;
               if (diff > 0) target_rows_per_device[dev_to_adjust]++;
               else if (target_rows_per_device[dev_to_adjust] > 0) target_rows_per_device[dev_to_adjust]--;
           }
      }
      for(int i=0; i<D; ++i) if(target_rows_per_device[i] < 0) target_rows_per_device[i] = 0;


      // 3) Create Chunks and Assign to Devices (interleaved)
      parts.reserve(nrows / chunk_size_ + D);

      // Store the actual rows assigned in this plan for the *next* update_times call
      prev_rows_assigned_by_index_.assign(D, 0); // Reset for current plan's actual counts

      std::vector<int> row_start_offset_within_global_range(D, 0); // Track progress within each device's total assigned rows

      bool more_chunks_to_make = true;
      int chunks_made_count = 0;

      while(more_chunks_to_make) {
          more_chunks_to_make = false;
          for(int i=0; i<D; ++i) {
               int rows_remaining_to_assign_for_device = target_rows_per_device[i] - row_start_offset_within_global_range[i];

               if(rows_remaining_to_assign_for_device > 0) {
                   int current_chunk_rows = std::min(chunk_size_, rows_remaining_to_assign_for_device);

                   // Calculate the actual global row range for this chunk
                   int actual_global_start = 0;
                   // Need to find the sum of target_rows for devices before this one
                   for(int j=0; j<i; ++j) actual_global_start += target_rows_per_device[j];
                   actual_global_start += row_start_offset_within_global_range[i];
                   int actual_global_end = actual_global_start + current_chunk_rows;


                   parts.push_back(Part{
                       /*row_begin=*/ actual_global_start,
                       /*row_end  =*/ actual_global_end,
                       /*q        =*/ const_cast<sycl::queue*>(&queues[i])
                   });

                   row_start_offset_within_global_range[i] += current_chunk_rows;
                   prev_rows_assigned_by_index_[i] += current_chunk_rows; // Track actual assigned rows
                   chunks_made_count++;
                   more_chunks_to_make = true; // We made a chunk, so loop again
               }
          }
      }
      // std::cerr << "AdaptiveLoadBalancingScheduler: Adaptive plan generated with " << chunks_made_count << " chunks." << std::endl;
      // ... (optional debug prints) ...

    } // End else (subsequent iterations)

    return parts;
  }

  // Feedback hook: receive per-device kernel times (in ms) for the *last executed* plan
  void update_times(const std::vector<double>& times) override {
    const int D = static_cast<int>(times.size());
    // Only update if we have device data and we have previously made a plan for this number of devices
    // and the rows assigned match the size of the times vector.
    if (D == 0 || !inited_ || prev_rows_assigned_by_index_.size() != (size_t)D) {
        // This warning might be noisy if device config changes, but useful if times vec size is unexpected.
        // std::cerr << "AdaptiveLoadBalancingScheduler::update_times: Warning: Cannot update times. Device count mismatch or not initialized." << std::endl;
        return;
    }

     // Ensure prev_times_ vector is the correct size
    if (prev_times_.size() != (size_t)D) {
        prev_times_.resize(D, 1e-9);
         // std::cerr << "AdaptiveLoadBalancingScheduler::update_times: Resized prev_times_ to " << D << std::endl;
    }
    // Ensure device_speeds_ vector is the correct size
     if (device_speeds_.size() != (size_t)D) {
         device_speeds_.resize(D, initial_speed_guess_);
          // std::cerr << "AdaptiveLoadBalancingScheduler::update_times: Resized device_speeds_ to " << D << std::endl;
     }


    // Update previous times and speeds based on received kernel times
    // Sanitize times to avoid division by zero or extreme values. Use the smoothing factor.
    for (size_t i = 0; i < times.size(); ++i) {
        double current_kernel_time = (times[i] > 1e-9) ? times[i] : 1e-9; // Sanitize time (min 1ns)
        prev_times_[i] = current_kernel_time; // Store the time for the plan that just finished

        double current_speed_calc = 0.0;
        // Calculate speed (rows / ms) based on actual rows assigned in the *previous* plan
        // (stored in prev_rows_assigned_by_index_ from the last make_plan call)
        // and the measured kernel time for that device in the execution of that plan.
         if (prev_rows_assigned_by_index_[i] > 0) {
             current_speed_calc = static_cast<double>(prev_rows_assigned_by_index_[i]) / current_kernel_time; // rows/ms
             if (current_speed_calc < 1e-9) current_speed_calc = 1e-9; // Sanitize calculated speed
         } else {
             // Device was assigned 0 rows in the previous plan. Cannot calculate speed from this iter.
             // Keep its previous estimated speed.
             current_speed_calc = device_speeds_[i]; // Retain old speed estimate
         }

        // Apply smoothing to the device speed estimate:
        // New Speed = alpha * Current Iteration Speed + (1 - alpha) * Old Speed Estimate
        // This makes the speed estimate more stable over iterations.
        device_speeds_[i] = speed_alpha_ * current_speed_calc + (1.0 - speed_alpha_) * device_speeds_[i];

         // Ensure speed doesn't become zero or negative
         if (device_speeds_[i] < 1e-9) device_speeds_[i] = 1e-9;
    }
    // std::cerr << "AdaptiveLoadBalancingScheduler::update_times: Feedback received and state updated for " << D << " devices." << std::endl;
    // for(size_t i=0; i<D; ++i) {
    //      std::cerr << "  Device " << i << ": Recvd Time = " << times[i] << " ms, Prev Assigned Rows = " << prev_rows_assigned_by_index_[i] << ", New Est. Speed = " << device_speeds_[i] << " rows/ms" << std::endl;
    // }
  }


private:
  int                      chunk_size_;         // Base chunk size for partitioning rows
  double                   initial_speed_guess_;// Initial speed guess (rows/ms)
  double                   speed_alpha_;        // Smoothing factor for speed updates

  bool                     inited_;             // Flag for first-time initialization

  // State for feedback:
  std::vector<int>         prev_rows_assigned_by_index_; // Actual total rows assigned to device i in the *previous* completed plan
  std::vector<double>      prev_times_;                  // Total kernel time for device i in the *previous* executed iteration (from update_times)
  std::vector<double>      device_speeds_;      // Estimated speed of device i (rows/ms), used for planning the *next* iteration

};

// Register the scheduler with the factory
REGISTER_SCHEDULER("adaptive_lb", AdaptiveLoadBalancingScheduler);

} // namespace spmv