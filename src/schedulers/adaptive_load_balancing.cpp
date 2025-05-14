// src/schedulers/adaptive_load_balancing.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace spmv {

class AdaptiveLoadBalancingScheduler : public IScheduler {
public:
  AdaptiveLoadBalancingScheduler(double alpha = 0.5)
    : speed_alpha_(alpha), iteration_count_(0) {}

  const char* name() const noexcept override {
    return "adaptive_lb";
  }

  bool is_dynamic() const noexcept override {
    return false;
  }

  std::vector<Part>
  make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    const int D = static_cast<int>(queues.size());
    std::vector<Part> parts;
    
    if (D == 0 || nrows == 0) return parts;

    // Initialize state on first call
    if (device_speeds_.size() != (size_t)D) {
      device_speeds_.assign(D, 1.0);
      prev_rows_.assign(D, 0);
      std::cout << "AdaptiveLoadBalancingScheduler: Initialized for " << D << " devices" << std::endl;
    }

    // ALWAYS use static block for iteration 0
    if (iteration_count_ == 0) {
      std::cout << "AdaptiveLoadBalancingScheduler: STATIC BLOCK DISTRIBUTION (iteration " 
                << iteration_count_ << ")" << std::endl;
                
      // Simple static block distribution
      int base = nrows / D;
      int rem  = nrows % D;
      int row_start = 0;
      
      for (int i = 0; i < D; ++i) {
        int rows_for_device = base + (i < rem ? 1 : 0);
        int row_end = row_start + rows_for_device;
        
        std::cout << "  Device " << i << ": rows " << row_start << " to " << row_end 
                  << " (" << rows_for_device << " rows)" << std::endl;
                  
        parts.push_back(Part{
          row_start, row_end, const_cast<sycl::queue*>(&queues[i])
        });
        
        prev_rows_[i] = rows_for_device;
        row_start = row_end;
      }
    }
    // Use chunked distribution for iterations 1+
    else {
      std::cout << "AdaptiveLoadBalancingScheduler: CHUNKED DISTRIBUTION (iteration " 
                << iteration_count_ << ")" << std::endl;
      
      // Calculate total speed for proportional distribution
      double total_speed = std::accumulate(device_speeds_.begin(), device_speeds_.end(), 0.0);
      if (total_speed < 1e-9) {
        std::cout << "  Warning: Total speed near zero, equalizing device speeds" << std::endl;
        device_speeds_.assign(D, 1.0);
        total_speed = D;
      }
      
      // Print current device speeds
      std::cout << "  Device speeds:";
      for (int i = 0; i < D; ++i) {
        std::cout << " [" << i << "]=" << device_speeds_[i];
      }
      std::cout << std::endl;
      
      // Calculate target rows per device based on speed
      std::vector<int> target_rows(D);
      int assigned_rows = 0;
      for (int i = 0; i < D; ++i) {
        double fraction = device_speeds_[i] / total_speed;
        target_rows[i] = static_cast<int>(fraction * nrows);
        assigned_rows += target_rows[i];
      }
      
      // Adjust for rounding errors
      int diff = nrows - assigned_rows;
      if (diff != 0) {
        // Find the fastest device that can handle the difference
        int adjust_device = std::max_element(device_speeds_.begin(), device_speeds_.end()) 
                          - device_speeds_.begin();
        target_rows[adjust_device] += diff;
      }
      
      // Create chunks (use a larger chunk size for visibility in gantt chart)
      const int CHUNK_SIZE = 500;
      std::vector<std::pair<int, int>> chunks; // (device_index, chunk_size)
      
      for (int i = 0; i < D; ++i) {
        int remaining = target_rows[i];
        while (remaining > 0) {
          int current_chunk = std::min(CHUNK_SIZE, remaining);
          chunks.push_back({i, current_chunk});
          remaining -= current_chunk;
        }
      }
      
      // Shuffle chunks for better load balancing
      std::random_shuffle(chunks.begin(), chunks.end());
      
      // Assign chunks to devices
      int row_idx = 0;
      std::vector<int> actual_rows(D, 0);
      
      for (const auto& chunk : chunks) {
        int dev_idx = chunk.first;
        int chunk_size = chunk.second;
        
        parts.push_back(Part{
          row_idx, row_idx + chunk_size, const_cast<sycl::queue*>(&queues[dev_idx])
        });
        
        actual_rows[dev_idx] += chunk_size;
        row_idx += chunk_size;
      }
      
      // Print row distribution summary
      std::cout << "  Row distribution:";
      for (int i = 0; i < D; ++i) {
        std::cout << " [" << i << "]=" << actual_rows[i];
        prev_rows_[i] = actual_rows[i];
      }
      std::cout << std::endl;
    }
    
    // Increment iteration counter
    iteration_count_++;
    return parts;
  }

  void update_times(const std::vector<double>& times) override {
    const int D = static_cast<int>(times.size());
    if (D == 0 || device_speeds_.size() != (size_t)D) return;
    
    std::cout << "AdaptiveLoadBalancingScheduler: update_times() called for iteration " 
              << (iteration_count_ - 1) << std::endl;
    
    std::cout << "  Kernel times:";
    for (int i = 0; i < D; ++i) {
      std::cout << " [" << i << "]=" << times[i] << "ms";
    }
    std::cout << std::endl;
    
    std::cout << "  Previous rows:";
    for (int i = 0; i < D; ++i) {
      std::cout << " [" << i << "]=" << prev_rows_[i];
    }
    std::cout << std::endl;
    
    // Update device speeds based on performance
    std::cout << "  New speeds:";
    for (int i = 0; i < D; ++i) {
      if (prev_rows_[i] > 0 && times[i] > 1e-9) {
        double current_speed = static_cast<double>(prev_rows_[i]) / times[i];
        device_speeds_[i] = speed_alpha_ * current_speed + (1.0 - speed_alpha_) * device_speeds_[i];
      }
      std::cout << " [" << i << "]=" << device_speeds_[i];
    }
    std::cout << std::endl;
  }

private:
  double speed_alpha_;
  int iteration_count_;
  std::vector<double> device_speeds_;
  std::vector<int> prev_rows_;
};

REGISTER_SCHEDULER("adaptive_lb", AdaptiveLoadBalancingScheduler);

} // namespace spmv