// apps/spmv_cli/main.cpp
#include "spmv/csr.hpp"
#include "spmv/scheduler.hpp"
#include "spmv/kernels.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>   // for unique_ptr, dynamic_cast
#include <iomanip>  // For std::fixed and std::setprecision

using namespace std::chrono;
using namespace std;
using spmv::spmv_cpu;
using spmv::spmv_gpu;

// Helper to generate a unique short name for a device for CSV header
std::string get_device_short_name(const sycl::device& dev, int dev_idx) {
    std::string type = dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "DEV");
    // Keep it simple for now, can be enhanced to get more specific names if needed
    // and sanitize them for CSV header.
    return type + std::to_string(dev_idx);
}


int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: spmv_cli <matrix.mtx> <summary.csv> <devices.csv> <iterations> <scheduler>\n";
    return 1;
  }
  const std::string matrix_path = argv[1];
  const std::string summary_csv  = argv[2];
  const std::string devices_csv  = argv[3];
  const int         iterations   = std::stoi(argv[4]);
  const std::string sched_name   = argv[5];

  // 1) Load CSR once
  auto t0_load = high_resolution_clock::now();
  auto A  = spmv::CSR::load_mm(matrix_path);
  auto t1_load = high_resolution_clock::now();
  if (A.empty()) {
    std::cerr << "Failed to load CSR\n";
    return 1;
  }

  // 2) Discover devices & build profiling‐enabled queues
  std::vector<sycl::device> discovered_devs;
  for (auto& plt : sycl::platform::get_platforms()) {
    for (auto& d : plt.get_devices()) {
      if (d.is_gpu() || d.is_cpu()) {
        discovered_devs.push_back(d);
      }
    }
  }

  // Filter to ensure we have a consistent set, e.g., up to 2 GPUs and 1 CPU
  // This part might need adjustment based on desired device selection logic.
  // For now, let's try to use all discovered CPU/GPU devices.
  std::vector<sycl::device> devs;
  // Prioritize GPUs then CPUs, can be made more sophisticated
  for(const auto& d : discovered_devs) {
    if (d.is_gpu()) devs.push_back(d);
  }
  for(const auto& d : discovered_devs) {
    if (d.is_cpu()) devs.push_back(d);
  }

  if (devs.empty()) {
      std::cerr << "No suitable SYCL devices (GPU or CPU) found.\n";
      return 1;
  }
  // Example: Limit to a certain number of devices if necessary
  // const size_t MAX_DEVICES = 3; // Example limit
  // if (devs.size() > MAX_DEVICES) {
  //     devs.resize(MAX_DEVICES);
  // }


  std::cout << "Using " << devs.size() << " devices:" << std::endl;
  std::vector<std::string> device_names_for_header;
  for(size_t i = 0; i < devs.size(); ++i) {
    std::cout << "  - " << devs[i].get_info<sycl::info::device::name>() << std::endl;
    device_names_for_header.push_back(get_device_short_name(devs[i], i));
  }


  std::vector<sycl::queue> queues;
  for (auto& d : devs) {
    queues.emplace_back(d,
      sycl::property_list{
        sycl::property::queue::in_order(),
        sycl::property::queue::enable_profiling()
      });
  }

  // 3) Prepare host vectors
  std::vector<int>   h_row_ptr = A.row_ptr;
  std::vector<int>   h_col_idx = A.col_idx;
  std::vector<float> h_vals    = A.vals;
  std::vector<float> h_x(A.ncols, 1.0f);
  std::vector<float> h_y(A.nrows, 0.0f); // h_y is used as the reference for reset and final check

  // 4) Allocate USM and copy CSR + x,y once
  std::vector<spmv::DeviceMem> mems(devs.size());
  auto t2_copy_init = high_resolution_clock::now();
  for (size_t i = 0; i < devs.size(); ++i) {
    auto& q = queues[i];
    auto& m = mems[i];
    m.rp = sycl::malloc_device<int>(h_row_ptr.size(), q);
    m.ci = sycl::malloc_device<int>(h_col_idx.size(), q);
    m.v  = sycl::malloc_device<float>(h_vals.size(),    q);
    m.x  = sycl::malloc_device<float>(h_x.size(),       q);
    m.y  = sycl::malloc_device<float>(h_y.size(),       q); // Each device gets its own y buffer

    q.memcpy(m.rp, h_row_ptr.data(), h_row_ptr.size()*sizeof(int));
    q.memcpy(m.ci, h_col_idx.data(), h_col_idx.size()*sizeof(int));
    q.memcpy(m.v,  h_vals.data(),    h_vals.size()*sizeof(float));
    q.memcpy(m.x,  h_x.data(),       h_x.size()*sizeof(float));
    // y is typically initialized to 0, so we might copy h_y (filled with 0s) or use q.fill()
    q.memcpy(m.y,  h_y.data(),       h_y.size()*sizeof(float));
  }
  for (auto& q : queues) q.wait();
  auto t3_copy_init = high_resolution_clock::now();

  // 5) Prepare accumulators
  std::vector<double> sched_times_all_iters, kernel_times_all_iters, copyback_times_all_iters, total_times_all_iters;
  std::vector<spmv::DevRecord> dev_recs; // Remains the store for all chunk-level records

  sched_times_all_iters.reserve(iterations);
  kernel_times_all_iters.reserve(iterations);
  copyback_times_all_iters.reserve(iterations);
  total_times_all_iters.reserve(iterations);

  auto scheduler = spmv::make_scheduler(sched_name);

  double avg_sched_dyn = 0.0, avg_kernel_dyn = 0.0, avg_copy_dyn = 0.0; // Used by dynamic scheduler
  std::vector<float> h_y_result(A.nrows, 0.0f); // Buffer to copy result back for verification/use


  // 6) Run
  if (scheduler->is_dynamic()) {
    scheduler->execute_dynamic(
      A, iterations,
      queues,
      mems,
      h_y_result, // Dynamic scheduler writes its final result here
      dev_recs,
      avg_sched_dyn,  // This is avg wall-time of the parallel execution phase per iteration
      avg_kernel_dyn, // This is avg sum of SYCL event kernel times per iteration
      avg_copy_dyn    // This is avg time for final data copy-back per iteration
    );
  } else {
    // Static/Feedback Path
    for (int it = 0; it < iterations; ++it) {
      // Reset device y buffers to 0.0f for each iteration
      for (size_t i = 0; i < devs.size(); ++i) {
          // queues[i].memcpy(mems[i].y, h_y.data(), h_y.size() * sizeof(float)); // h_y contains 0.0f
          // Or, more explicitly:
          std::vector<float> zeros(A.nrows, 0.0f);
          queues[i].memcpy(mems[i].y, zeros.data(), zeros.size() * sizeof(float));
      }
      for (auto& q : queues) q.wait();

      auto t4_sched_start = high_resolution_clock::now();
      auto parts = scheduler->make_plan(A.nrows, queues);
      auto t5_sched_end = high_resolution_clock::now();
      sched_times_all_iters.push_back(duration<double, milli>(t5_sched_end - t4_sched_start).count());

      std::vector<sycl::event> events;
      events.reserve(parts.size());
      size_t rec0_iter = dev_recs.size(); // Starting index in dev_recs for this iteration

      auto t5_kernel_phase_start = high_resolution_clock::now(); // Start timing kernel phase
      for (size_t pi = 0; pi < parts.size(); ++pi) {
        auto const& p = parts[pi];
        if (p.row_end <= p.row_begin) continue; // Skip empty parts

        auto t_launch = high_resolution_clock::now(); // This is a bit redundant if t5_kernel_phase_start is used for overall

        int di = -1;
        for(size_t i = 0; i < queues.size(); ++i) {
            if (&queues[i] == p.q) {
                di = static_cast<int>(i);
                break;
            }
        }
        if (di == -1) {
            std::cerr << "Error: Could not find device index for a partition." << std::endl;
            return 1; // Or handle error appropriately
        }
        auto const& m = mems[di];

        sycl::event e = p.q->get_device().is_gpu()
          ? spmv_gpu(*p.q,
                     p.row_end - p.row_begin,
                     m.rp   + p.row_begin, // Pointer to the start of row_ptr for this partition
                     m.ci,
                     m.v,
                     m.x,
                     m.y   + p.row_begin) // Pointer to the start of y for this partition
          : spmv_cpu(*p.q,
                     p.row_end - p.row_begin,
                     m.rp   + p.row_begin,
                     m.ci,
                     m.v,
                     m.x,
                     m.y   + p.row_begin);
        
        events.push_back(e);
        // launch_ms in DevRecord is time from kernel_phase_start to launch submit time
        dev_recs.push_back({
          it, di,
          p.row_begin,
          p.row_end,
          duration<double,milli>(high_resolution_clock::now() - t5_kernel_phase_start).count(), // time since kernel phase start
          0.0 // kernel_ms will be filled later
        });
      }

      sycl::event::wait_and_throw(events); // Wait for all kernels in this iteration
      auto t6_kernel_phase_end = high_resolution_clock::now();
      kernel_times_all_iters.push_back(duration<double, milli>(t6_kernel_phase_end - t5_kernel_phase_start).count());

      for (size_t ei = 0; ei < events.size(); ++ei) {
        auto& e = events[ei];
        uint64_t s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        dev_recs[rec0_iter + ei].kernel_ms = double(t - s) * 1e-6;
      }

      auto t6_copyback_start = high_resolution_clock::now();
      // Copy back y from device to host for this iteration (typically only one part contributes to final h_y_result)
      // For SpMV, each part computes a section of y. We need to assemble it.
      std::fill(h_y_result.begin(), h_y_result.end(), 0.0f); // Clear host result buffer
      for (size_t pi = 0; pi < parts.size(); ++pi) {
        auto const& p = parts[pi];
        if (p.row_end <= p.row_begin) continue;

        int di = -1;
        for(size_t i = 0; i < queues.size(); ++i) {
            if (&queues[i] == p.q) {
                di = static_cast<int>(i);
                break;
            }
        }
        // Copy the relevant part of y from mems[di].y to h_y_result
        queues[di].memcpy(
          h_y_result.data() + p.row_begin,
          mems[di].y + p.row_begin,
          (p.row_end - p.row_begin) * sizeof(float)
        ).wait(); // Wait for this copy
      }
      auto t7_copyback_end = high_resolution_clock::now();
      copyback_times_all_iters.push_back(duration<double, milli>(t7_copyback_end - t6_copyback_start).count());
      
      total_times_all_iters.push_back(
          sched_times_all_iters.back() +
          kernel_times_all_iters.back() +
          copyback_times_all_iters.back());

      if (sched_name == "feedback") {
        std::vector<double> times_feedback(devs.size(), 0.0);
        for (size_t r_idx = rec0_iter; r_idx < dev_recs.size(); ++r_idx) {
          if (dev_recs[r_idx].iter == it) { // Ensure it's from the current iteration
            times_feedback[dev_recs[r_idx].dev_idx] += dev_recs[r_idx].kernel_ms;
          }
        }
        scheduler->update_times(times_feedback);
      }
    }
  }

  // 7) Calculate averages and other summary metrics
  double load_ms = duration<double, milli>(t1_load - t0_load).count();
  double copy_init_ms = duration<double, milli>(t3_copy_init - t2_copy_init).count();

  double avg_sched_ms_summary, avg_kernel_ms_summary, avg_copyback_ms_summary, avg_total_ms_summary;

  std::vector<double> avg_device_utilization_pct(devs.size(), 0.0);
  std::vector<double> total_kernel_ms_per_device(devs.size(), 0.0);
  double total_kernel_phase_duration_all_iters = 0.0;

  if (scheduler->is_dynamic()) {
    avg_sched_ms_summary = avg_sched_dyn; // This is avg wall-time of parallel execution phase
    avg_kernel_ms_summary = avg_kernel_dyn; // This is avg sum of SYCL event kernel times
    avg_copyback_ms_summary = avg_copy_dyn;
    avg_total_ms_summary = avg_sched_ms_summary + avg_copyback_ms_summary; // Total is trickier for dynamic, sum of phases

    // Calculate utilization for dynamic
    for(const auto& rec : dev_recs) {
        if(rec.dev_idx >= 0 && rec.dev_idx < devs.size()) {
            total_kernel_ms_per_device[rec.dev_idx] += rec.kernel_ms;
        }
    }
    total_kernel_phase_duration_all_iters = avg_sched_dyn * iterations; // Total time spent in the parallel execution phase
    if (total_kernel_phase_duration_all_iters > 0) {
        for(size_t i=0; i < devs.size(); ++i) {
            avg_device_utilization_pct[i] = (total_kernel_ms_per_device[i] / total_kernel_phase_duration_all_iters) * 100.0;
        }
    }

  } else { // Static/Feedback
    avg_sched_ms_summary = std::accumulate(sched_times_all_iters.begin(), sched_times_all_iters.end(), 0.0) / iterations;
    avg_kernel_ms_summary = std::accumulate(kernel_times_all_iters.begin(), kernel_times_all_iters.end(), 0.0) / iterations;
    avg_copyback_ms_summary = std::accumulate(copyback_times_all_iters.begin(), copyback_times_all_iters.end(), 0.0) / iterations;
    avg_total_ms_summary = std::accumulate(total_times_all_iters.begin(), total_times_all_iters.end(), 0.0) / iterations;

    // Calculate utilization for static/feedback
    for(const auto& rec : dev_recs) {
        if(rec.dev_idx >= 0 && rec.dev_idx < devs.size()) {
            total_kernel_ms_per_device[rec.dev_idx] += rec.kernel_ms;
        }
    }
    total_kernel_phase_duration_all_iters = std::accumulate(kernel_times_all_iters.begin(), kernel_times_all_iters.end(), 0.0);
    if (total_kernel_phase_duration_all_iters > 0) {
        for(size_t i=0; i < devs.size(); ++i) {
            avg_device_utilization_pct[i] = (total_kernel_ms_per_device[i] / total_kernel_phase_duration_all_iters) * 100.0;
        }
    }
  }
  
  float y0_check = h_y_result.empty() ? 0.0f : h_y_result[0];


  // 8) Write summary CSV
  {
    std::ofstream os(summary_csv);
    os << std::fixed << std::setprecision(6); // Set precision for floating point numbers
    os << "matrix,scheduler,iterations,load_ms,copy_init_ms,avg_sched_ms,avg_kernel_ms,avg_copyback_ms,avg_total_ms,y0";
    for(size_t i = 0; i < device_names_for_header.size(); ++i) {
        os << ",avg_util_" << device_names_for_header[i] << "_pct";
    }
    os << "\n";

    os << matrix_path << ","
       << sched_name   << ","
       << iterations   << ","
       << load_ms      << ","
       << copy_init_ms << ","
       << avg_sched_ms_summary    << "," // For static, this is make_plan. For dynamic, this is the parallel exec phase wall time.
       << avg_kernel_ms_summary   << "," // For static, critical path of kernels. For dynamic, sum of actual kernel times.
       << avg_copyback_ms_summary << ","
       << avg_total_ms_summary    << ","
       << y0_check;
    for(size_t i = 0; i < avg_device_utilization_pct.size(); ++i) {
        os << "," << avg_device_utilization_pct[i];
    }
    os << "\n";
  }

  // 9) Write devices CSV (no changes needed here, dev_recs is populated as before)
  {
    std::ofstream od(devices_csv);
    od << std::fixed << std::setprecision(6);
    od << "iteration,dev_idx,row_begin,row_end,launch_ms,kernel_ms\n";
    for (auto& r : dev_recs) {
      od << r.iter      << ","
         << r.dev_idx   << ","
         << r.row_begin << ","
         << r.row_end   << ","
         << r.launch_ms << "," // For static, this is time from kernel phase start to launch. For dynamic, from iter start.
         << r.kernel_ms << "\n";
    }
  }

  std::cout << "Completed " << iterations
            << " runs using scheduler '" << sched_name << "'\n"
            << "  summary → " << summary_csv << "\n"
            << "  devices → " << devices_csv << "\n";
  return 0;
}