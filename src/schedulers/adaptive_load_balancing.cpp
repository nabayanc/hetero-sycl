// src/schedulers/adaptive_load_balancing.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include "spmv/csr.hpp"

#include <sycl/sycl.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>

using namespace std::chrono;

namespace spmv {

class TrueAdaptiveLoadBalancingScheduler : public IScheduler {
private:
  struct DeviceModel {
    double perf_estimate;    // Estimated performance (rows/ms)
    double alpha;            // Smoothing factor
    double confidence;       // Confidence in estimates (0-1)
    int total_rows_processed;// Cumulative rows processed
    
    DeviceModel() : perf_estimate(1.0), alpha(0.2), confidence(0.0), 
                   total_rows_processed(0) {}
  };
  
  // Workload characteristics for prioritization
  struct Chunk {
    int start, end;
    double complexity;  // Estimated complexity (e.g., nnz in range)
    
    bool operator<(const Chunk& other) const {
      return complexity < other.complexity;  // Higher complexity has higher priority
    }
  };
  
  int initial_chunk_size_;
  std::vector<DeviceModel> device_models_;
  double total_runtime_ms_ = 0.0;
  int total_iterations_ = 0;
  
public:
  TrueAdaptiveLoadBalancingScheduler(int initial_chunk_size = 1000) 
    : initial_chunk_size_(initial_chunk_size) {}

  std::vector<Part> make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    // For compatibility with non-dynamic execution path
    const int D = int(queues.size());
    if (D == 0) return {};
    
    // Initialize device models if needed
    if (device_models_.size() != (size_t)D) {
      device_models_.resize(D);
    }
    
    // For the first iteration or if called directly, use performance-weighted distribution
    std::vector<double> weights(D);
    double total_weight = 0.0;
    
    for (int i = 0; i < D; ++i) {
      // Higher weight for devices with better performance estimates
      weights[i] = device_models_[i].perf_estimate * 
                  (0.5 + 0.5 * device_models_[i].confidence);
      total_weight += weights[i];
    }
    
    // Handle case where we have no performance history
    if (total_weight <= 0.0) {
      // Default to equal weights
      for (int i = 0; i < D; ++i) {
        weights[i] = 1.0;
      }
      total_weight = D;
    }
    
    std::vector<Part> parts;
    int start = 0;
    
    for (int i = 0; i < D; ++i) {
      double fraction = weights[i] / total_weight;
      int rows = i == D-1 ? 
                 (nrows - start) : // Last device gets remainder
                 std::max(1, static_cast<int>(nrows * fraction));
                 
      int end = std::min(nrows, start + rows);
      parts.push_back({start, end, const_cast<sycl::queue*>(&queues[i])});
      start = end;
    }
    
    return parts;
  }

  const char* name() const noexcept override { return "true_adaptive_lb"; }
  
  bool is_dynamic() const noexcept override { return true; }

  void execute_dynamic(
      const CSR &A,
      int iterations,
      std::vector<sycl::queue>& queues,
      const std::vector<DeviceMem>& mems,
      std::vector<float>& h_y,
      std::vector<DevRecord>& dev_recs,
      double& avg_sched_ms,
      double& avg_kernel_ms,
      double& avg_copy_ms) override
  {
    int D = int(queues.size());
    if (D == 0) return;
    
    // Initialize or resize device models if needed
    if (device_models_.size() != (size_t)D) {
      device_models_.resize(D);
    }
    
    std::vector<double> sched_times, kernel_times, copy_times;
    sched_times.reserve(iterations);
    kernel_times.reserve(iterations);
    copy_times.reserve(iterations);

    // For each iteration
    for (int it = 0; it < iterations; ++it) {
      // Reset output vector on all devices
      for (int di = 0; di < D; ++di) {
        queues[di].memcpy(mems[di].y, h_y.data(), A.nrows * sizeof(float));
      }
      for (auto& q : queues) q.wait();

      auto t0 = high_resolution_clock::now();
      
      // Calculate per-row complexities (based on non-zeros)
      std::vector<double> row_complexity(A.nrows);
      for (int r = 0; r < A.nrows; ++r) {
        row_complexity[r] = A.row_ptr[r+1] - A.row_ptr[r]; // nnz in this row
      }
      
      // Determine adaptive chunk size based on matrix size and iteration
      int base_chunk_size;
      if (total_iterations_ == 0) {
        // First time, use static initial size
        base_chunk_size = initial_chunk_size_;
      } else {
        // Adapt chunk size based on matrix and previous runtime
        double avg_time_per_row = total_runtime_ms_ / total_iterations_ / A.nrows;
        // Target chunk execution time ~5ms for good balance without too much overhead
        base_chunk_size = std::max(100, std::min(5000, 
            static_cast<int>(5.0 / avg_time_per_row)));
      }
      
      // Create prioritized chunks with varying sizes based on complexity
      std::priority_queue<Chunk> chunk_queue;
      
      for (int start = 0; start < A.nrows;) {
        // Determine chunk size - more complex regions get smaller chunks
        double local_complexity = 0;
        for (int r = start; r < std::min(A.nrows, start + 100); ++r) {
          local_complexity += row_complexity[r];
        }
        local_complexity /= std::min(100, A.nrows - start);
        
        // Adjust chunk size based on local complexity
        int adaptive_size = static_cast<int>(base_chunk_size / 
            std::sqrt(1.0 + local_complexity / 100.0));
        adaptive_size = std::max(50, std::min(base_chunk_size * 2, adaptive_size));
        
        int end = std::min(A.nrows, start + adaptive_size);
        
        // Calculate total complexity for this chunk
        double chunk_total_complexity = 0;
        for (int r = start; r < end; ++r) {
          chunk_total_complexity += row_complexity[r];
        }
        
        chunk_queue.push({start, end, chunk_total_complexity});
        start = end;
      }
      
      // Shared scheduling structures
      std::mutex sched_mutex;
      std::vector<std::queue<Chunk>> device_queues(D);
      std::vector<std::atomic<bool>> device_active(D);
      std::vector<std::atomic<int>> device_completed_rows(D);
      std::vector<double> device_completed_complexity(D, 0.0);
      std::atomic<int> total_rows_completed{0};
      std::atomic<bool> all_chunks_assigned{false};
      std::atomic<bool> all_done{false};
      
      // Initial distribution of chunks based on device performance model
      {
        std::lock_guard<std::mutex> lock(sched_mutex);
        
        // Calculate target fractions based on performance estimates
        std::vector<double> target_fractions(D);
        double total_perf = 0.0;
        
        for (int di = 0; di < D; ++di) {
          // Consider both performance and confidence
          double weighted_perf = device_models_[di].perf_estimate * 
                               (0.5 + 0.5 * device_models_[di].confidence);
          target_fractions[di] = weighted_perf;
          total_perf += weighted_perf;
        }
        
        if (total_perf <= 0.0) {
          // If no history, distribute evenly
          for (int di = 0; di < D; ++di) {
            target_fractions[di] = 1.0 / D;
          }
        } else {
          // Normalize to fractions
          for (int di = 0; di < D; ++di) {
            target_fractions[di] /= total_perf;
          }
        }
        
        // Assign initial chunks to each device based on target fractions
        std::vector<int> rows_assigned(D, 0);
        int total_assigned = 0;
        
        // First, assign a minimum amount to each device
        int min_rows_per_device = std::min(A.nrows / (2*D), 1000);
        
        while (!chunk_queue.empty() && total_assigned < A.nrows) {
          // Find the device that's furthest below its target
          int target_device = 0;
          double max_deficit = -1.0;
          
          for (int di = 0; di < D; ++di) {
            double target_rows = target_fractions[di] * A.nrows;
            double deficit = (target_rows - rows_assigned[di]) / target_rows;
            
            if (deficit > max_deficit) {
              max_deficit = deficit;
              target_device = di;
            }
          }
          
          // Assign the most complex chunk to this device
          Chunk c = chunk_queue.top();
          chunk_queue.pop();
          
          device_queues[target_device].push(c);
          rows_assigned[target_device] += (c.end - c.start);
          total_assigned += (c.end - c.start);
          
          device_active[target_device] = true;
        }
        
        all_chunks_assigned = true;
      }
      
      size_t rec0 = dev_recs.size();
      
      // Worker threads - one per device
      std::vector<std::thread> workers;
      for (int di = 0; di < D; ++di) {
        workers.emplace_back([&, di]() {
          auto &q = queues[di];
          auto const &m = mems[di];
          
          int local_rows_completed = 0;
          double local_complexity_completed = 0.0;
          auto device_start_time = high_resolution_clock::now();
          
          while (!all_done.load()) {
            Chunk chunk;
            bool has_chunk = false;
            
            // Try to get a chunk from own queue
            {
              std::lock_guard<std::mutex> lock(sched_mutex);
              if (!device_queues[di].empty()) {
                chunk = device_queues[di].front();
                device_queues[di].pop();
                has_chunk = true;
              }
            }
            
            if (!has_chunk) {
              if (all_chunks_assigned.load() && total_rows_completed.load() >= A.nrows) {
                // All work is done
                break;
              }
              
              // Dynamic load balancing - try to steal or request more work
              if (all_chunks_assigned.load()) {
                // Try work stealing from other devices
                std::lock_guard<std::mutex> lock(sched_mutex);
                
                // Find device with the most work left
                int steal_from = -1;
                int max_remaining = 0;
                
                for (int odi = 0; odi < D; ++odi) {
                  if (odi == di) continue;
                  
                  int queue_size = device_queues[odi].size();
                  if (queue_size > max_remaining) {
                    max_remaining = queue_size;
                    steal_from = odi;
                  }
                }
                
                if (steal_from >= 0 && !device_queues[steal_from].empty()) {
                  chunk = device_queues[steal_from].front();
                  device_queues[steal_from].pop();
                  has_chunk = true;
                }
              }
              
              if (!has_chunk) {
                // Short wait before trying again
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
              }
            }
            
            // Process the chunk
            int rb = chunk.start;
            int re = chunk.end;
            double chunk_complexity = chunk.complexity;
            
            auto tl = high_resolution_clock::now();
            sycl::event e = q.submit([&](sycl::handler &cgh) {
              cgh.parallel_for(
                sycl::range<1>(re - rb),
                [=](sycl::id<1> idx) {
                  int r = rb + idx[0];
                  float sum = 0.0f;
                  for (int k = m.rp[r]; k < m.rp[r+1]; ++k)
                    sum += m.v[k] * m.x[m.ci[k]];
                  m.y[r] = sum;
                });
            });
            e.wait();
            
            uint64_t ts = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t te = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            double ktime = double(te - ts) * 1e-6;
            double ltime = duration<double,std::milli>(tl - t0).count();
            
            // Update completion status
            local_rows_completed += (re - rb);
            local_complexity_completed += chunk_complexity;
            device_completed_rows[di] += (re - rb);
            
            {
              std::lock_guard<std::mutex> lock(sched_mutex);
              device_completed_complexity[di] += chunk_complexity;
            }
            
            total_rows_completed += (re - rb);
            
            // Record execution details
            {
              std::lock_guard<std::mutex> lock(sched_mutex);
              dev_recs.push_back({it, di, rb, re, ltime, ktime});
            }
            
            // Check for completion
            if (total_rows_completed.load() >= A.nrows) {
              all_done.store(true);
              break;
            }
          }
          
          // Calculate performance metrics for this device
          auto device_end_time = high_resolution_clock::now();
          double device_runtime_ms = duration<double,std::milli>(device_end_time - device_start_time).count();
          
          if (local_rows_completed > 0 && device_runtime_ms > 0) {
            std::lock_guard<std::mutex> lock(sched_mutex);
            
            // Calculate performance (rows/ms)
            double current_perf = local_rows_completed / device_runtime_ms;
            
            // Calculate complexity-weighted performance
            double complexity_perf = local_complexity_completed / device_runtime_ms;
            
            // Update device model with exponential smoothing
            DeviceModel& model = device_models_[di];
            
            // Increase confidence with more data
            model.total_rows_processed += local_rows_completed;
            model.confidence = std::min(1.0, model.total_rows_processed / 10000.0);
            
            // Update performance estimate
            if (model.perf_estimate <= 0.0) {
              model.perf_estimate = current_perf; // Initial value
            } else {
              model.perf_estimate = model.alpha * current_perf + 
                                   (1.0 - model.alpha) * model.perf_estimate;
            }
            
            // Adaptively adjust alpha based on consistency
            double perf_diff = std::abs(current_perf - model.perf_estimate) / model.perf_estimate;
            if (perf_diff > 0.2) {
              // Performance changed significantly, be more adaptive
              model.alpha = std::min(0.5, model.alpha * 1.5);
            } else {
              // Performance is stable, reduce adaptivity
              model.alpha = std::max(0.05, model.alpha * 0.9);
            }
          }
        });
      }
      
      // Wait for all worker threads to complete
      for (auto& w : workers) {
        w.join();
      }
      
      auto t1 = high_resolution_clock::now();
      double iter_sched_time = duration<double,std::milli>(t1 - t0).count();
      sched_times.push_back(iter_sched_time);
      
      // Update total runtime statistics
      total_runtime_ms_ += iter_sched_time;
      total_iterations_++;
      
      // Copy results back to host
      auto t2 = high_resolution_clock::now();
      // Merge results from all devices
      for (const auto& part : make_plan(A.nrows, queues)) {
        int dev_idx = -1;
        for (int di = 0; di < D; ++di) {
          if (&queues[di] == part.q) {
            dev_idx = di;
            break;
          }
        }
        
        if (dev_idx >= 0 && part.row_end > part.row_begin) {
          queues[dev_idx].memcpy(
            h_y.data() + part.row_begin,
            mems[dev_idx].y + part.row_begin,
            (part.row_end - part.row_begin) * sizeof(float)).wait();
        }
      }
      auto t3 = high_resolution_clock::now();
      copy_times.push_back(duration<double,std::milli>(t3 - t2).count());
      
      // Calculate total kernel time
      double sumk = 0.0;
      for (size_t r = rec0; r < dev_recs.size(); ++r) {
        sumk += dev_recs[r].kernel_ms;
      }
      kernel_times.push_back(sumk);
      
      // Print performance report for this iteration
      std::cout << "Iteration " << it << " performance report:" << std::endl;
      for (int di = 0; di < D; ++di) {
        std::cout << "  Device " << di << ": " 
                  << device_completed_rows[di].load() << " rows, "
                  << "Perf est: " << device_models_[di].perf_estimate 
                  << " rows/ms, Alpha: " << device_models_[di].alpha 
                  << ", Confidence: " << device_models_[di].confidence << std::endl;
      }
    }
    
    // Calculate averages
    avg_sched_ms = std::accumulate(sched_times.begin(), sched_times.end(), 0.0) / iterations;
    avg_kernel_ms = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / iterations;
    avg_copy_ms = std::accumulate(copy_times.begin(), copy_times.end(), 0.0) / iterations;
  }
};

REGISTER_SCHEDULER("true_adaptive_lb", TrueAdaptiveLoadBalancingScheduler);

} // namespace spmv