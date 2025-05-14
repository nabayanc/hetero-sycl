// src/schedulers/workstealing.cpp
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

using namespace std::chrono;

namespace spmv {

class WorkStealingScheduler : public IScheduler {
private:
  int chunk_size_;

public:
  WorkStealingScheduler(int chunk_size = 500) : chunk_size_(chunk_size) {}

  std::vector<Part> make_plan(int nrows, const std::vector<sycl::queue>& queues) override {
    // For compatibility with non-dynamic execution path
    // Create a basic initial distribution
    const int D = int(queues.size());
    if (D == 0) return {};

    std::vector<Part> parts;
    int base = nrows / D;
    int remainder = nrows % D;

    int start = 0;
    for (int i = 0; i < D; ++i) {
      int size = base + (i < remainder ? 1 : 0);
      parts.push_back({start, start + size, const_cast<sycl::queue*>(&queues[i])});
      start += size;
    }
    return parts;
  }

  const char* name() const noexcept override { return "workstealing"; }
  
  // Mark this scheduler as dynamic to use the execute_dynamic path
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
    
    std::vector<double> sched_times, kernel_times, copy_times;
    sched_times.reserve(iterations);
    kernel_times.reserve(iterations);
    copy_times.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
      // Reset output vector on all devices
      for (int di = 0; di < D; ++di) {
        queues[di].memcpy(mems[di].y, h_y.data(), A.nrows * sizeof(float));
      }
      for (auto& q : queues) q.wait();

      auto t0 = high_resolution_clock::now();
      
      // Create work chunks
      std::vector<std::pair<int, int>> chunks;
      for (int start = 0; start < A.nrows; start += chunk_size_) {
        int end = std::min(A.nrows, start + chunk_size_);
        chunks.push_back({start, end});
      }
      
      // Shared structures with thread-safe access
      std::mutex chunk_mutex;
      std::vector<std::queue<std::pair<int, int>>> device_queues(D);
      std::vector<std::atomic<bool>> device_busy(D);
      std::vector<std::atomic<int>> chunks_completed(D);
      std::atomic<int> total_chunks_completed{0};
      std::atomic<bool> all_done{false};
      
      // Initial assignment - give each device some chunks to start with
      {
        std::lock_guard<std::mutex> lock(chunk_mutex);
        int assigned_chunks = 0;
        int chunks_per_device = std::max(1, int(chunks.size() / D)); // At least 1 chunk per device
        
        for (int di = 0; di < D; ++di) {
          for (int c = 0; c < chunks_per_device && assigned_chunks < chunks.size(); ++c) {
            device_queues[di].push(chunks[assigned_chunks++]);
          }
          device_busy[di] = !device_queues[di].empty();
          chunks_completed[di] = 0;
        }
        
        // Push any remaining chunks to the queue of the first device
        while (assigned_chunks < chunks.size()) {
          device_queues[0].push(chunks[assigned_chunks++]);
        }
      }
      
      size_t rec0 = dev_recs.size();
      
      // Worker threads - one per device
      std::vector<std::thread> workers;
      for (int di = 0; di < D; ++di) {
        workers.emplace_back([&, di]() {
          auto &q = queues[di];
          auto const &m = mems[di];
          
          while (!all_done.load()) {
            std::pair<int, int> chunk;
            bool has_chunk = false;
            
            // Try to get a chunk from own queue
            {
              std::lock_guard<std::mutex> lock(chunk_mutex);
              if (!device_queues[di].empty()) {
                chunk = device_queues[di].front();
                device_queues[di].pop();
                has_chunk = true;
              }
            }
            
            // If no chunk in own queue, try to steal
            if (!has_chunk) {
              // Try to steal from other devices
              for (int other_di = 0; other_di < D && !has_chunk; ++other_di) {
                if (other_di == di) continue;
                
                if (device_busy[other_di].load()) {
                  std::lock_guard<std::mutex> lock(chunk_mutex);
                  if (!device_queues[other_di].empty()) {
                    chunk = device_queues[other_di].front();
                    device_queues[other_di].pop();
                    has_chunk = true;
                  }
                }
              }
              
              // If still no chunk, check if we're done
              if (!has_chunk) {
                if (total_chunks_completed.load() >= chunks.size()) {
                  // All chunks are completed
                  break;
                }
                
                // Short wait before trying again
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
              }
            }
            
            // Process the chunk
            int rb = chunk.first;
            int re = chunk.second;
            device_busy[di] = true;
            
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
            chunks_completed[di]++;
            total_chunks_completed++;
            
            // Record execution details
            {
              std::lock_guard<std::mutex> lock(chunk_mutex);
              dev_recs.push_back({it, di, rb, re, ltime, ktime});
            }
            
            // If we've completed all chunks, set the done flag
            if (total_chunks_completed.load() >= chunks.size()) {
              all_done.store(true);
            }
            
            // Mark device as potentially available for stealing from
            device_busy[di] = !device_queues[di].empty();
          }
        });
      }
      
      // Wait for all worker threads to complete
      for (auto& w : workers) {
        w.join();
      }
      
      auto t1 = high_resolution_clock::now();
      sched_times.push_back(duration<double,std::milli>(t1 - t0).count());
      
      // Copy results back to host
      auto t2 = high_resolution_clock::now();
      for (int di = 0; di < D; ++di) {
        queues[di].memcpy(h_y.data(), mems[di].y, A.nrows * sizeof(float)).wait();
      }
      auto t3 = high_resolution_clock::now();
      copy_times.push_back(duration<double,std::milli>(t3 - t2).count());
      
      // Calculate total kernel time
      double sumk = 0.0;
      for (size_t r = rec0; r < dev_recs.size(); ++r) {
        sumk += dev_recs[r].kernel_ms;
      }
      kernel_times.push_back(sumk);
    }
    
    // Calculate averages
    avg_sched_ms = std::accumulate(sched_times.begin(), sched_times.end(), 0.0) / iterations;
    avg_kernel_ms = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / iterations;
    avg_copy_ms = std::accumulate(copy_times.begin(), copy_times.end(), 0.0) / iterations;
  }
};

REGISTER_SCHEDULER("workstealing", WorkStealingScheduler);

} // namespace spmv