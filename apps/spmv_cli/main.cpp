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
#include <memory>
#include <iomanip>
#include <stdexcept> // For std::runtime_error for unknown args
#include <map> // For parsing optional args

// Helper to parse simple key-value command line arguments like --key value
std::map<std::string, std::string> parse_optional_args(int argc, char** argv, int start_index) {
    std::map<std::string, std::string> optional_args;
    for (int i = start_index; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) { // Check if it starts with --
            std::string key = arg.substr(2);
            if (i + 1 < argc) {
                std::string next_arg = argv[i+1];
                if (next_arg.rfind("--", 0) != 0) { // Next arg is not another key
                    optional_args[key] = next_arg;
                    i++; // Consume value
                } else {
                    optional_args[key] = "true"; // Flag-like argument
                }
            } else {
                optional_args[key] = "true"; // Flag-like argument at the end
            }
        }
    }
    return optional_args;
}


using namespace std::chrono;
using namespace std;
using spmv::spmv_cpu;
using spmv::spmv_gpu;

std::string get_device_label(const sycl::device& dev, int dev_idx) {
    std::string type = dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "DEV");
    return "dev" + std::to_string(dev_idx) + "_" + type;
}


int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Usage: spmv_cli <matrix.mtx> <summary.csv> <devices.csv> <timed_iterations> <scheduler_name> [--warmup_iterations N]\n";
    return 1;
  }
  const std::string matrix_path = argv[1];
  const std::string summary_csv_path  = argv[2];
  const std::string devices_csv_path  = argv[3];
  const int         timed_iterations   = std::stoi(argv[4]);
  const std::string sched_name   = argv[5];

  int warmup_iterations = 0;
  std::map<std::string, std::string> optional_args = parse_optional_args(argc, argv, 6);
  if (optional_args.count("warmup_iterations")) {
      try {
        warmup_iterations = std::stoi(optional_args["warmup_iterations"]);
      } catch (const std::exception& e) {
        std::cerr << "Error parsing --warmup_iterations: " << e.what() << std::endl;
        return 1;
      }
  }

  auto t0_load = high_resolution_clock::now();
  auto A  = spmv::CSR::load_mm(matrix_path);
  auto t1_load = high_resolution_clock::now();
  if (A.empty()) { std::cerr << "Failed to load CSR\n"; return 1; }
  double load_ms = duration<double, milli>(t1_load - t0_load).count();

  auto t_sycl_setup_start = high_resolution_clock::now();
  std::vector<sycl::device> discovered_devs;
  for (auto& plt : sycl::platform::get_platforms()) {
    for (auto& d : plt.get_devices()) {
      if (d.is_gpu() || d.is_cpu()) {
        discovered_devs.push_back(d);
      }
    }
  }
  std::vector<sycl::device> devs; // Final list of devices to use
  // Prioritize GPUs, then sort by name for consistent ordering if multiple of same type
  std::vector<sycl::device> gpus, cpus;
  for(const auto& d : discovered_devs) {
    if (d.is_gpu()) gpus.push_back(d);
    else if (d.is_cpu()) cpus.push_back(d);
  }
  std::sort(gpus.begin(), gpus.end(), [](const sycl::device& a, const sycl::device& b){
      return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
  });
  std::sort(cpus.begin(), cpus.end(), [](const sycl::device& a, const sycl::device& b){
      return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
  });
  for(const auto& d : gpus) devs.push_back(d);
  for(const auto& d : cpus) devs.push_back(d);

  if (devs.empty()) { std::cerr << "No suitable SYCL devices found.\n"; return 1; }
  
  std::vector<std::string> device_labels_for_header;
  std::cout << "Using " << devs.size() << " devices for scheduler '" << sched_name << "':" << std::endl;
  for(size_t i = 0; i < devs.size(); ++i) {
    std::string label = get_device_label(devs[i], i);
    std::cout << "  - " << label << ": " << devs[i].get_info<sycl::info::device::name>() << std::endl;
    device_labels_for_header.push_back(label);
  }

  std::vector<sycl::queue> queues;
  for (auto& d : devs) {
    queues.emplace_back(d, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
  }
  auto t_sycl_setup_end = high_resolution_clock::now();
  double sycl_setup_ms = duration<double, milli>(t_sycl_setup_end - t_sycl_setup_start).count();

  std::vector<float> h_x(A.ncols, 1.0f);
  std::vector<float> h_y_zeros(A.nrows, 0.0f);
  std::vector<float> h_y_result_buffer(A.nrows, 0.0f); // For final result and y0_check

  auto t_alloc_start = high_resolution_clock::now();
  std::vector<spmv::DeviceMem> mems(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) {
    auto& q = queues[i]; auto& m = mems[i];
    m.rp = sycl::malloc_device<int>(A.row_ptr.size(), q);
    m.ci = sycl::malloc_device<int>(A.col_idx.size(), q);
    m.v  = sycl::malloc_device<float>(A.vals.size(),    q);
    m.x  = sycl::malloc_device<float>(h_x.size(),       q);
    m.y  = sycl::malloc_device<float>(A.nrows,       q);
  }
  for (auto& q : queues) q.wait();
  auto t_alloc_end = high_resolution_clock::now();
  double usm_alloc_ms = duration<double, milli>(t_alloc_end - t_alloc_start).count();

  std::vector<sycl::event> initial_transfer_events_vec;
  initial_transfer_events_vec.reserve(devs.size() * 4);
  auto t_initial_h2d_start = high_resolution_clock::now();
  for (size_t i = 0; i < devs.size(); ++i) {
    auto& q = queues[i]; auto& m = mems[i];
    initial_transfer_events_vec.push_back(q.memcpy(m.rp, A.row_ptr.data(), A.row_ptr.size()*sizeof(int)));
    initial_transfer_events_vec.push_back(q.memcpy(m.ci, A.col_idx.data(), A.col_idx.size()*sizeof(int)));
    initial_transfer_events_vec.push_back(q.memcpy(m.v,  A.vals.data(),    A.vals.size()*sizeof(float)));
    initial_transfer_events_vec.push_back(q.memcpy(m.x,  h_x.data(),       h_x.size()*sizeof(float)));
  }
  sycl::event::wait_and_throw(initial_transfer_events_vec);
  auto t_initial_h2d_end = high_resolution_clock::now();
  double initial_h2d_wall_ms = duration<double, milli>(t_initial_h2d_end - t_initial_h2d_start).count();
  double sum_initial_h2d_event_ms = 0;
  for(auto& e : initial_transfer_events_vec) {
    sum_initial_h2d_event_ms += (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
  }
  
  std::vector<double> iter_y_reset_ms, iter_sched_ms, iter_kernel_submission_ms, iter_kernel_sync_ms, iter_overall_kernel_phase_ms, iter_copyback_ms, iter_total_ms;
  std::vector<std::vector<double>> iter_device_event_ms(devs.size()); // [dev_idx][iter_event_sum]

  std::vector<spmv::DevRecord> all_dev_recs; // For devices.csv, accumulates over all timed iterations
  auto scheduler = spmv::make_scheduler(sched_name);

  double dyn_avg_overall_kernel_phase_ms = 0.0, dyn_total_avg_sycl_kernel_event_ms = 0.0, dyn_avg_copyback_ms = 0.0;
  std::vector<double> dyn_avg_kernel_event_ms_per_device(devs.size(), 0.0);


  if (scheduler->is_dynamic()) {
      std::vector<spmv::DevRecord> dynamic_run_dev_recs; // Temp for one dynamic run call
      scheduler->execute_dynamic(A, timed_iterations, queues, mems, h_y_result_buffer, dynamic_run_dev_recs,
                                 dyn_avg_overall_kernel_phase_ms, dyn_total_avg_sycl_kernel_event_ms, dyn_avg_copyback_ms);
      all_dev_recs = dynamic_run_dev_recs; // Assuming execute_dynamic populates it for all iters
      // For dynamic, 'sched_ms' is part of overall_kernel_phase, submission/sync are not distinct like static.
      // y_reset is handled internally by execute_dynamic's loop or not applicable if y is stateful.
      // For now, we'll report 0 for those if dynamic.
      for (int i = 0; i < timed_iterations; ++i) {
          iter_y_reset_ms.push_back(0.0);
          iter_sched_ms.push_back(0.0);
          iter_kernel_submission_ms.push_back(0.0);
          iter_kernel_sync_ms.push_back(0.0);
          iter_overall_kernel_phase_ms.push_back(dyn_avg_overall_kernel_phase_ms); // Using the average reported
          iter_copyback_ms.push_back(dyn_avg_copyback_ms); // Using the average reported
          iter_total_ms.push_back(dyn_avg_overall_kernel_phase_ms + dyn_avg_copyback_ms);
          
          // Estimate per-device per-iter contributions for dynamic based on its output
          // This is an approximation as execute_dynamic gives averages.
          // We need to parse dynamic_run_dev_recs if we want per-iter per-device data.
          // For simplicity, let's calculate overall avg per device from dynamic_run_dev_recs
          // dyn_avg_kernel_event_ms_per_device will be calculated later from all_dev_recs
      }
  } else { // Static / Feedback Schedulers
    for (int run_idx = 0; run_idx < warmup_iterations + timed_iterations; ++run_idx) {
      bool is_warmup = run_idx < warmup_iterations;
      
      auto t_iter_start = high_resolution_clock::now();

      auto t_y_reset_s = high_resolution_clock::now();
      for (size_t i = 0; i < devs.size(); ++i) {
        queues[i].memcpy(mems[i].y, h_y_zeros.data(), A.nrows * sizeof(float)).wait();
      }
      auto t_y_reset_e = high_resolution_clock::now();
      if (!is_warmup) iter_y_reset_ms.push_back(duration<double, milli>(t_y_reset_e - t_y_reset_s).count());

      auto t_sched_s = high_resolution_clock::now();
      auto parts = scheduler->make_plan(A.nrows, queues);
      auto t_sched_e = high_resolution_clock::now();
      if (!is_warmup) iter_sched_ms.push_back(duration<double, milli>(t_sched_e - t_sched_s).count());

      std::vector<sycl::event> current_iter_kernel_events;
      current_iter_kernel_events.reserve(parts.size());
      size_t current_iter_dev_recs_start_idx = all_dev_recs.size();

      auto t_kernel_sub_s = high_resolution_clock::now();
      for (size_t pi = 0; pi < parts.size(); ++pi) {
        auto const& p = parts[pi];
        if (p.row_end <= p.row_begin) continue;
        int di = -1;
        for(size_t i=0; i<queues.size(); ++i) if(&queues[i] == p.q) {di=i; break;}
        if (di == -1) throw std::runtime_error("Partition queue not found in device list");
        auto const& m = mems[di];
        sycl::event e = p.q->get_device().is_gpu() ?
          spmv_gpu(*p.q, p.row_end-p.row_begin, m.rp+p.row_begin, m.ci, m.v, m.x, m.y+p.row_begin) :
          spmv_cpu(*p.q, p.row_end-p.row_begin, m.rp+p.row_begin, m.ci, m.v, m.x, m.y+p.row_begin);
        current_iter_kernel_events.push_back(e);
        if (!is_warmup) {
          all_dev_recs.push_back({run_idx - warmup_iterations, di, p.row_begin, p.row_end, 
                                duration<double,milli>(high_resolution_clock::now() - t_kernel_sub_s).count(), 0.0});
        }
      }
      auto t_kernel_sub_e = high_resolution_clock::now();
      if (!is_warmup) iter_kernel_submission_ms.push_back(duration<double, milli>(t_kernel_sub_e - t_kernel_sub_s).count());
      
      sycl::event::wait_and_throw(current_iter_kernel_events);
      auto t_kernel_sync_e = high_resolution_clock::now();
      if (!is_warmup) {
        iter_kernel_sync_ms.push_back(duration<double, milli>(t_kernel_sync_e - t_kernel_sub_e).count());
        iter_overall_kernel_phase_ms.push_back(duration<double, milli>(t_kernel_sync_e - t_kernel_sub_s).count());
        
        std::vector<double> iter_dev_event_sum(devs.size(), 0.0);
        for (size_t ei = 0; ei < current_iter_kernel_events.size(); ++ei) {
          auto& ev = current_iter_kernel_events[ei];
          double kernel_event_time = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() - 
                                     ev.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
          all_dev_recs[current_iter_dev_recs_start_idx + ei].kernel_ms = kernel_event_time;
          iter_dev_event_sum[all_dev_recs[current_iter_dev_recs_start_idx + ei].dev_idx] += kernel_event_time;
        }
        for(size_t i=0; i<devs.size(); ++i) iter_device_event_ms[i].push_back(iter_dev_event_sum[i]);
      }
      
      if (sched_name == "feedback" && !current_iter_kernel_events.empty()) {
          std::vector<double> times_feedback(devs.size(), 0.0);
          // This part requires careful indexing if multiple parts map to same device
          // The logic from exp01 for feedback was summing per device from dev_recs of that iteration
          // For spmv_cli, we can use iter_dev_event_sum from just above if is_warmup is false
          // or recalculate if needed for warmup feedback too
          if(!is_warmup && iter_device_event_ms[0].size() > 0) { // check if we have data for current timed iter
            for(size_t i=0; i<devs.size(); ++i) times_feedback[i] = iter_device_event_ms[i].back();
          } else { // Fallback for warmup or if data not ready: recalculate based on current events
            std::fill(times_feedback.begin(), times_feedback.end(), 0.0);
            for (size_t ei = 0; ei < current_iter_kernel_events.size(); ++ei) {
                auto& ev = current_iter_kernel_events[ei];
                double kt = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() - 
                             ev.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
                // Find device index for this event's queue (parts[ei].q)
                int event_dev_idx = -1;
                for(size_t q_idx=0; q_idx < queues.size(); ++q_idx) if(parts[ei].q == &queues[q_idx]) event_dev_idx = q_idx;
                if(event_dev_idx != -1) times_feedback[event_dev_idx] += kt;
            }
          }
          scheduler->update_times(times_feedback);
      }

      auto t_copyback_s = high_resolution_clock::now();
      std::fill(h_y_result_buffer.begin(), h_y_result_buffer.end(), 0.0f);
      for (size_t pi = 0; pi < parts.size(); ++pi) {
        auto const& p = parts[pi];
        if (p.row_end <= p.row_begin) continue;
        int di = -1;
        for(size_t i=0; i<queues.size(); ++i) if(&queues[i] == p.q) {di=i; break;}
        queues[di].memcpy(h_y_result_buffer.data() + p.row_begin, mems[di].y + p.row_begin, (p.row_end - p.row_begin) * sizeof(float)).wait();
      }
      auto t_copyback_e = high_resolution_clock::now();
      if (!is_warmup) iter_copyback_ms.push_back(duration<double, milli>(t_copyback_e - t_copyback_s).count());
      
      if (!is_warmup) iter_total_ms.push_back(duration<double, milli>(high_resolution_clock::now() - t_iter_start).count());
    }
  }

  auto calculate_avg = [&](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };

  double avg_y_reset_ms = calculate_avg(iter_y_reset_ms);
  double avg_sched_ms = calculate_avg(iter_sched_ms);
  double avg_kernel_submission_ms = calculate_avg(iter_kernel_submission_ms);
  double avg_kernel_sync_ms = calculate_avg(iter_kernel_sync_ms);
  double avg_overall_kernel_phase_ms = calculate_avg(iter_overall_kernel_phase_ms);
  double avg_copyback_ms = calculate_avg(iter_copyback_ms);
  double avg_total_iter_ms = calculate_avg(iter_total_ms);
  
  std::vector<double> avg_kernel_event_ms_devX_type(devs.size(), 0.0);
  double total_avg_sycl_kernel_event_ms = 0;

  if (scheduler->is_dynamic()) {
      avg_overall_kernel_phase_ms = dyn_avg_overall_kernel_phase_ms;
      total_avg_sycl_kernel_event_ms = dyn_total_avg_sycl_kernel_event_ms; // This is already a sum for dynamic
      avg_copyback_ms = dyn_avg_copyback_ms;
      avg_total_iter_ms = avg_overall_kernel_phase_ms + avg_copyback_ms; // Approximation for dynamic total
      // Calculate avg_kernel_event_ms_devX_type from all_dev_recs for dynamic
      std::vector<double> sum_event_ms_per_device(devs.size(), 0.0);
      std::vector<int> count_event_ms_per_device(devs.size(), 0); // This isn't quite right for averaging per-iteration sums
                                                                   // For dynamic, all_dev_recs holds all chunks.
                                                                   // We need a better way to get per-device contribution to total_avg_sycl_kernel_event_ms
                                                                   // If dyn_total_avg_sycl_kernel_event_ms is the sum of SYCL times for ONE iter,
                                                                   // and we need to break it down, this requires more info from execute_dynamic or parsing dev_recs.
                                                                   // For now, use an approximation or leave some device specific breakdown as N/A for dynamic if too complex.
      // Let's assume all_dev_recs for dynamic contains per-chunk kernel_ms for all timed_iterations
      // Sum them up and divide by timed_iterations to get average sum for that device.
      for(const auto& rec : all_dev_recs) {
          if(rec.dev_idx >= 0 && rec.dev_idx < devs.size()){
              sum_event_ms_per_device[rec.dev_idx] += rec.kernel_ms;
          }
      }
      for(size_t i=0; i < devs.size(); ++i) {
          if(timed_iterations > 0) avg_kernel_event_ms_devX_type[i] = sum_event_ms_per_device[i] / timed_iterations;
      }
      // Note: total_avg_sycl_kernel_event_ms from execute_dynamic is likely the sum across devices for ONE iteration.
      // The CSV asks for sum of *averages*. This implies we average each device's contribution first, then sum.
      // So, for dynamic, total_avg_sycl_kernel_event_ms should be sum of avg_kernel_event_ms_devX_type.
      total_avg_sycl_kernel_event_ms = std::accumulate(avg_kernel_event_ms_devX_type.begin(), avg_kernel_event_ms_devX_type.end(), 0.0);


  } else { // Static
      for(size_t i=0; i<devs.size(); ++i) {
        avg_kernel_event_ms_devX_type[i] = calculate_avg(iter_device_event_ms[i]);
        total_avg_sycl_kernel_event_ms += avg_kernel_event_ms_devX_type[i];
      }
  }
  
  std::vector<double> avg_util_devX_type_pct(devs.size(), 0.0);
  if (avg_overall_kernel_phase_ms > 1e-9) {
    for(size_t i=0; i<devs.size(); ++i) {
      avg_util_devX_type_pct[i] = std::min(100.0, (avg_kernel_event_ms_devX_type[i] / avg_overall_kernel_phase_ms) * 100.0);
    }
  }
  
  float y0_check = h_y_result_buffer.empty() ? 0.0f : h_y_result_buffer[0];

  std::ofstream os(summary_csv_path);
  os << std::fixed << std::setprecision(6);
  os << "matrix_path,scheduler_name,num_rows,num_cols,num_nonzeros,warmup_iterations,timed_iterations,"
     << "load_ms,sycl_setup_ms,usm_alloc_ms,initial_h2d_wall_ms,initial_h2d_event_sum_ms,"
     << "avg_y_reset_ms,avg_sched_ms,avg_kernel_submission_ms,avg_kernel_sync_ms,avg_overall_kernel_phase_wall_ms,"
     << "avg_copyback_ms,avg_total_iter_ms,total_avg_sycl_kernel_event_ms,y0_check";
  for(const auto& label : device_labels_for_header) {
      os << ",avg_kernel_event_ms_" << label << ",avg_util_" << label << "_pct";
  }
  os << "\n";

  os << matrix_path << "," << sched_name << "," << A.nrows << "," << A.ncols << "," << A.nnz << ","
     << warmup_iterations << "," << timed_iterations << ","
     << load_ms << "," << sycl_setup_ms << "," << usm_alloc_ms << "," << initial_h2d_wall_ms << "," << sum_initial_h2d_event_ms << ","
     << avg_y_reset_ms << "," << avg_sched_ms << "," << avg_kernel_submission_ms << "," << avg_kernel_sync_ms << "," << avg_overall_kernel_phase_ms << ","
     << avg_copyback_ms << "," << avg_total_iter_ms << "," << total_avg_sycl_kernel_event_ms << "," << y0_check;
  for(size_t i=0; i<devs.size(); ++i) {
      os << "," << avg_kernel_event_ms_devX_type[i] << "," << avg_util_devX_type_pct[i];
  }
  os << "\n";
  os.close();

  std::ofstream od(devices_csv_path);
  od << std::fixed << std::setprecision(6);
  od << "iteration,dev_idx,row_begin,row_end,launch_ms_rel_submission_start,kernel_ms\n"; // launch_ms is relative to kernel submission phase start for static
  for (const auto& r : all_dev_recs) {
    od << r.iter << "," << r.dev_idx << "," << r.row_begin << "," << r.row_end << "," << r.launch_ms << "," << r.kernel_ms << "\n";
  }
  od.close();

  std::cout << "Completed " << warmup_iterations << " warmup, " << timed_iterations << " timed runs for scheduler '" << sched_name << "' on matrix " << matrix_path << "\n"
            << "  Summary  -> " << summary_csv_path << "\n"
            << "  Devices  -> " << devices_csv_path << "\n";

  for (size_t i = 0; i < devs.size(); ++i) {
    sycl::free(mems[i].rp, queues[i]); sycl::free(mems[i].ci, queues[i]);
    sycl::free(mems[i].v, queues[i]); sycl::free(mems[i].x, queues[i]);
    sycl::free(mems[i].y, queues[i]);
  }
  return 0;
}