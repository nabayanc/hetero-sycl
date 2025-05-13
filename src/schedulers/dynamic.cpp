#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include "spmv/csr.hpp"

#include <sycl/sycl.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdint>

using namespace std::chrono;
using namespace std;

namespace spmv {

class DynamicScheduler : public IScheduler {
public:
  DynamicScheduler(int chunk_size = 256) : chunk_size_(chunk_size) {}

  std::vector<Part> make_plan(int, const std::vector<sycl::queue>&) override {
    return {};
  }
  const char* name() const noexcept override { return "dynamic"; }
  bool is_dynamic() const noexcept override { return true; }

  void execute_dynamic(
      const CSR &A,
      int iterations,
      std::vector<sycl::queue>& queues,               // now non-const
      const std::vector<DeviceMem>& mems,
      std::vector<float>& h_y,
      std::vector<DevRecord>& dev_recs,
      double& avg_sched_ms,
      double& avg_kernel_ms,
      double& avg_copy_ms) override
  {
    int D = int(queues.size());
    std::vector<double> sched_times, kernel_times, copy_times;
    sched_times.reserve(iterations);
    kernel_times.reserve(iterations);
    copy_times.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
      // reset y
      for (int di = 0; di < D; ++di)
        queues[di].memcpy(mems[di].y,
                          h_y.data(),
                          A.nrows * sizeof(float),
                          sycl::event{});
      for (auto& q : queues) q.wait();

      std::atomic<int> next_row{0};
      std::mutex rec_mtx;
      size_t rec0 = dev_recs.size();

      auto t0 = high_resolution_clock::now();

      // one host-thread per device
      std::vector<std::thread> workers;
      workers.reserve(D);
      for (int di = 0; di < D; ++di) {
        workers.emplace_back([&,di]() {
          auto &q = queues[di];
          auto const &m = mems[di];
          while (true) {
            int rb = next_row.fetch_add(chunk_size_);
            if (rb >= A.nrows) break;
            int re = std::min(A.nrows, rb + chunk_size_);

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

            std::lock_guard<std::mutex> lk(rec_mtx);
            dev_recs.push_back({it, di, rb, re, ltime, ktime});
          }
        });
      }
      for (auto& w : workers) w.join();
      auto t1 = high_resolution_clock::now();
      sched_times.push_back(duration<double,std::milli>(t1 - t0).count());

      // copy-back
      auto t2 = high_resolution_clock::now();
      for (int di = 0; di < D; ++di)
        queues[di].memcpy(h_y.data(),
                          mems[di].y,
                          A.nrows * sizeof(float),
                          sycl::event{});
      for (auto& q : queues) q.wait();
      auto t3 = high_resolution_clock::now();
      copy_times.push_back(duration<double,std::milli>(t3 - t1).count());

      // sum kernel times
      double sumk = 0.0;
      for (size_t r = rec0; r < dev_recs.size(); ++r)
        sumk += dev_recs[r].kernel_ms;
      kernel_times.push_back(sumk);
    }

    avg_sched_ms  = std::accumulate(sched_times.begin(),  sched_times.end(),  0.0) / iterations;
    avg_kernel_ms = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / iterations;
    avg_copy_ms   = std::accumulate(copy_times.begin(),   copy_times.end(),   0.0) / iterations;
  }

private:
  int chunk_size_;
};

REGISTER_SCHEDULER("dynamic", DynamicScheduler);

} // namespace spmv
