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
#include <stdexcept> // For std::runtime_error
#include <map>       // For parsing optional args
#include <thread>    // For std::thread (for multi-threaded dispatch)
#include <mutex>     // For std::mutex (for multi-threaded dispatch)

// Helper to parse simple key-value command line arguments like --key value
// and flags like --flag
std::map<std::string, std::string> parse_optional_args(int argc, char** argv, int start_index) {
    std::map<std::string, std::string> optional_args;
    for (int i = start_index; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) { // Check if it starts with --
            std::string key = arg.substr(2);
            if (i + 1 < argc) {
                std::string next_arg = argv[i+1];
                if (next_arg.rfind("--", 0) != 0) { // Next arg is not another key, so it's a value
                    optional_args[key] = next_arg;
                    i++; // Consume value
                } else {
                    optional_args[key] = "true"; // Flag-like argument (no explicit value)
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
// Note: spmv_cpu and spmv_gpu are already in spmv namespace, no need for using declarations here
// using spmv::spmv_cpu;
// using spmv::spmv_gpu;


struct DetailedDevRecord {
    int iter;
    int dev_idx; 
    std::string device_name_label; 
    int row_begin;
    int row_end;
    double host_dispatch_offset_ms; 
    double kernel_duration_ms;      
};

struct EventDispatchContext {
    sycl::event event;
    int timed_iter_idx;
    int dev_experimental_idx; 
    std::string dev_name_label_str;
    int p_row_begin;
    int p_row_end;
    double dispatch_offset_ms_val;
};


std::string get_device_label(const sycl::device& dev, int dev_idx_experimental) {
    std::string type = dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "DEV");
    return "dev" + std::to_string(dev_idx_experimental) + "_" + type;
}

auto ms_duration_fn = [](high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
};


int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Usage: spmv_cli <matrix.mtx> <summary.csv> <devices.csv> <timed_iterations> <scheduler_name> [--warmup_iterations N] [--dispatch_mode single|multi]\n";
    return 1;
  }
  const std::string matrix_path      = argv[1];
  const std::string summary_csv_path = argv[2];
  const std::string devices_csv_path = argv[3];
  const int         timed_iterations = std::stoi(argv[4]);
  const std::string sched_name       = argv[5];

  int warmup_iterations = 0;
  std::string dispatch_mode_str = "single"; 

  std::map<std::string, std::string> optional_args = parse_optional_args(argc, argv, 6);
  if (optional_args.count("warmup_iterations")) {
      try {
        warmup_iterations = std::stoi(optional_args["warmup_iterations"]);
      } catch (const std::exception& e) {
        std::cerr << "Error parsing --warmup_iterations: " << e.what() << std::endl;
        return 1;
      }
  }
  if (optional_args.count("dispatch_mode")) {
      dispatch_mode_str = optional_args["dispatch_mode"];
      if (dispatch_mode_str != "single" && dispatch_mode_str != "multi") {
          std::cerr << "Error: Invalid --dispatch_mode. Must be 'single' or 'multi'." << std::endl;
          return 1;
      }
  }

  bool use_multithreaded_dispatch = (dispatch_mode_str == "multi");

  cout << "Running spmv_cli with:" << endl;
  cout << "  Matrix: " << matrix_path << endl;
  cout << "  Scheduler: " << sched_name << endl;
  cout << "  Warmup Iterations: " << warmup_iterations << endl;
  cout << "  Timed Iterations: " << timed_iterations << endl;
  cout << "  Dispatch Mode: " << dispatch_mode_str << endl;


  auto t0_load = high_resolution_clock::now();
  auto A  = spmv::CSR::load_mm(matrix_path);
  auto t1_load = high_resolution_clock::now();
  if (A.empty()) { std::cerr << "Failed to load CSR " << matrix_path << "\n"; return 1; }
  double load_ms = ms_duration_fn(t0_load, t1_load);
  cout << "Matrix " << matrix_path << " loaded: " << A.nrows << "x" << A.ncols << ", " << A.nnz << " non-zeros. (Load time: " << load_ms << " ms)" << endl;


  auto t_sycl_setup_start = high_resolution_clock::now();
  std::vector<sycl::device> discovered_devs_all;
  for (auto& plt : sycl::platform::get_platforms()) {
    for (auto& d : plt.get_devices()) {
      if (d.is_gpu() || d.is_cpu()) { 
        discovered_devs_all.push_back(d);
      }
    }
  }

  std::vector<sycl::device> gpus, cpus;
  for(const auto& d : discovered_devs_all) {
    if (d.is_gpu()) gpus.push_back(d);
    else if (d.is_cpu()) cpus.push_back(d);
  }
  std::sort(gpus.begin(), gpus.end(), [](const sycl::device& a, const sycl::device& b){
      return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
  });
  std::sort(cpus.begin(), cpus.end(), [](const sycl::device& a, const sycl::device& b){
      return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
  });

  std::vector<sycl::device> final_selected_devices; 
  for(const auto& d : gpus) final_selected_devices.push_back(d);
  for(const auto& d : cpus) final_selected_devices.push_back(d);

  if (final_selected_devices.empty()) { std::cerr << "No suitable SYCL devices found.\n"; return 1; }
  
  std::vector<std::string> device_name_labels_for_header;
  
  std::cout << "Using " << final_selected_devices.size() << " devices for scheduler '" << sched_name << "':" << std::endl;
  for(size_t i = 0; i < final_selected_devices.size(); ++i) {
    std::string label = get_device_label(final_selected_devices[i], i);
    std::cout << "  - [" << i << "] " << label << ": " << final_selected_devices[i].get_info<sycl::info::device::name>() << std::endl;
    device_name_labels_for_header.push_back(label);
  }

  std::vector<sycl::queue> queues; // Ensure this is mutable
  for (auto& d : final_selected_devices) {
    queues.emplace_back(d, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
  }
  auto t_sycl_setup_end = high_resolution_clock::now();
  double sycl_setup_ms = ms_duration_fn(t_sycl_setup_start, t_sycl_setup_end);

  std::vector<float> h_x(A.ncols, 1.0f);
  std::vector<float> h_y_zeros(A.nrows, 0.0f);
  std::vector<float> h_y_result_buffer(A.nrows, 0.0f); 

  auto t_alloc_start = high_resolution_clock::now();
  std::vector<spmv::DeviceMem> mems(final_selected_devices.size());
  for (size_t i = 0; i < final_selected_devices.size(); ++i) {
    auto& q_ref = queues[i]; auto& m_ref = mems[i]; // Use references to mutable queues
    m_ref.rp = sycl::malloc_device<int>(A.row_ptr.size(), q_ref);
    m_ref.ci = sycl::malloc_device<int>(A.col_idx.size(), q_ref);
    m_ref.v  = sycl::malloc_device<float>(A.vals.size(),    q_ref);
    m_ref.x  = sycl::malloc_device<float>(h_x.size(),       q_ref);
    m_ref.y  = sycl::malloc_device<float>(A.nrows,          q_ref);
  }
  for (auto& q_ref : queues) q_ref.wait(); // Use references
  auto t_alloc_end = high_resolution_clock::now();
  double usm_alloc_ms = ms_duration_fn(t_alloc_start, t_alloc_end);

  std::vector<sycl::event> initial_transfer_events_vec;
  initial_transfer_events_vec.reserve(final_selected_devices.size() * 4);
  auto t_initial_h2d_start = high_resolution_clock::now();
  for (size_t i = 0; i < final_selected_devices.size(); ++i) {
    auto& q_ref = queues[i]; auto& m_ref = mems[i]; // Use references
    initial_transfer_events_vec.push_back(q_ref.memcpy(m_ref.rp, A.row_ptr.data(), A.row_ptr.size()*sizeof(int)));
    initial_transfer_events_vec.push_back(q_ref.memcpy(m_ref.ci, A.col_idx.data(), A.col_idx.size()*sizeof(int)));
    initial_transfer_events_vec.push_back(q_ref.memcpy(m_ref.v,  A.vals.data(),    A.vals.size()*sizeof(float)));
    initial_transfer_events_vec.push_back(q_ref.memcpy(m_ref.x,  h_x.data(),       h_x.size()*sizeof(float)));
  }
  sycl::event::wait_and_throw(initial_transfer_events_vec);
  auto t_initial_h2d_end = high_resolution_clock::now();
  double initial_h2d_wall_ms = ms_duration_fn(t_initial_h2d_start, t_initial_h2d_end);
  double sum_initial_h2d_event_ms = 0;
  for(auto& e : initial_transfer_events_vec) {
    sum_initial_h2d_event_ms += (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
  }
  
  std::vector<double> iter_y_reset_ms, iter_sched_ms, iter_kernel_submission_ms, iter_kernel_sync_ms, iter_overall_kernel_phase_ms, iter_copyback_ms, iter_total_ms;
  std::vector<std::vector<double>> iter_device_sum_kernel_event_ms(final_selected_devices.size()); 

  std::vector<DetailedDevRecord> all_detailed_dev_recs; 
  auto scheduler = spmv::make_scheduler(sched_name);

  double dyn_avg_overall_kernel_phase_ms = 0.0, dyn_total_avg_sycl_kernel_event_ms = 0.0, dyn_avg_copyback_ms = 0.0;

  if (scheduler->is_dynamic()) {
      std::cout << "Scheduler is dynamic. Using execute_dynamic path." << std::endl;
      std::vector<spmv::DevRecord> dynamic_run_dev_recs_raw; 
      // Pass mutable queues to execute_dynamic
      scheduler->execute_dynamic(A, timed_iterations, queues, mems, h_y_result_buffer, dynamic_run_dev_recs_raw,
                                 dyn_avg_overall_kernel_phase_ms, dyn_total_avg_sycl_kernel_event_ms, dyn_avg_copyback_ms);
      
      for(const auto& raw_rec : dynamic_run_dev_recs_raw) {
          all_detailed_dev_recs.push_back({
              raw_rec.iter,
              raw_rec.dev_idx, 
              get_device_label(queues[raw_rec.dev_idx].get_device(), raw_rec.dev_idx), 
              raw_rec.row_begin, raw_rec.row_end,
              raw_rec.launch_ms, 
              raw_rec.kernel_ms
          });
      }
      for (int i = 0; i < timed_iterations; ++i) {
          iter_y_reset_ms.push_back(0.0); 
          iter_sched_ms.push_back(0.0); 
          iter_kernel_submission_ms.push_back(0.0); 
          iter_kernel_sync_ms.push_back(0.0); 
          iter_overall_kernel_phase_ms.push_back(dyn_avg_overall_kernel_phase_ms);
          iter_copyback_ms.push_back(dyn_avg_copyback_ms);
          iter_total_ms.push_back(dyn_avg_overall_kernel_phase_ms + dyn_avg_copyback_ms);
      }
  } else { 
    cout << "Scheduler is static/feedback. Using iterative plan/execute path with dispatch: " << dispatch_mode_str << endl;
    for (int run_idx = 0; run_idx < warmup_iterations + timed_iterations; ++run_idx) {
      bool is_warmup = run_idx < warmup_iterations;
      if (!is_warmup && run_idx == warmup_iterations) std::cout << "Warmup complete, starting timed runs." << std::endl;
      
      auto t_iter_start_wall = high_resolution_clock::now();

      auto t_y_reset_s_wall = high_resolution_clock::now();
      std::vector<sycl::event> y_reset_events;
      y_reset_events.reserve(final_selected_devices.size());
      for (size_t i = 0; i < final_selected_devices.size(); ++i) {
        y_reset_events.push_back(queues[i].memcpy(mems[i].y, h_y_zeros.data(), A.nrows * sizeof(float)));
      }
      sycl::event::wait_and_throw(y_reset_events);
      auto t_y_reset_e_wall = high_resolution_clock::now();
      if (!is_warmup) iter_y_reset_ms.push_back(ms_duration_fn(t_y_reset_s_wall, t_y_reset_e_wall));

      auto t_sched_s_wall = high_resolution_clock::now();
      auto parts = scheduler->make_plan(A.nrows, queues); 
      auto t_sched_e_wall = high_resolution_clock::now();
      if (!is_warmup) iter_sched_ms.push_back(ms_duration_fn(t_sched_s_wall, t_sched_e_wall));
      
      auto t_kernel_submission_phase_start_wall = high_resolution_clock::now();
      std::vector<sycl::event> current_iter_kernel_exec_events;
      std::vector<EventDispatchContext> current_iter_event_contexts; 
      std::mutex events_mutex, contexts_mutex; 

      if (use_multithreaded_dispatch && final_selected_devices.size() > 1) {
          std::vector<std::thread> submission_workers;
          submission_workers.reserve(final_selected_devices.size());

          std::map<int, std::vector<spmv::Part>> parts_by_dev_original_idx;
          for(const auto& p : parts) {
              int original_dev_idx = -1;
              for(size_t q_idx = 0; q_idx < queues.size(); ++q_idx) {
                  if (p.q == &queues[q_idx]) {
                      original_dev_idx = q_idx;
                      break;
                  }
              }
              if (original_dev_idx != -1) {
                  parts_by_dev_original_idx[original_dev_idx].push_back(p);
              } else {
                  std::cerr << "Warning: Could not map partition queue to an original device queue in multi-threaded dispatch." << std::endl;
              }
          }
          
          for(int dev_original_idx_loop = 0; dev_original_idx_loop < final_selected_devices.size(); ++dev_original_idx_loop) {
              if (parts_by_dev_original_idx.count(dev_original_idx_loop) == 0 || parts_by_dev_original_idx[dev_original_idx_loop].empty()) {
                  continue; 
              }
              // Important: Capture dev_original_idx_loop by value for the lambda
              submission_workers.emplace_back([&, dev_original_idx = dev_original_idx_loop, t_kernel_submission_phase_start_wall, is_warmup, run_idx, warmup_iterations]() {
                  // No need to capture parts_by_dev_original_idx by reference if already captured by &
                  std::vector<sycl::event> thread_local_events;
                  std::vector<EventDispatchContext> thread_local_contexts;
                  
                  // Access parts_by_dev_original_idx from the outer scope (captured by &)
                  const auto& dev_specific_parts = parts_by_dev_original_idx.at(dev_original_idx);
                  
                  sycl::queue& q_ref_lambda = queues[dev_original_idx]; // Mutable reference
                  const auto& m_ref_lambda = mems[dev_original_idx];  
                  std::string current_dev_name_label = get_device_label(q_ref_lambda.get_device(), dev_original_idx);

                  for (const auto& p_part : dev_specific_parts) {
                      if (p_part.row_end <= p_part.row_begin) continue;
                      
                      double current_dispatch_offset_ms = ms_duration_fn(t_kernel_submission_phase_start_wall, high_resolution_clock::now());
                      sycl::event e = q_ref_lambda.get_device().is_gpu() ?
                        spmv::spmv_gpu(q_ref_lambda, p_part.row_end - p_part.row_begin, m_ref_lambda.rp + p_part.row_begin, m_ref_lambda.ci, m_ref_lambda.v, m_ref_lambda.x, m_ref_lambda.y + p_part.row_begin) :
                        spmv::spmv_cpu(q_ref_lambda, p_part.row_end - p_part.row_begin, m_ref_lambda.rp + p_part.row_begin, m_ref_lambda.ci, m_ref_lambda.v, m_ref_lambda.x, m_ref_lambda.y + p_part.row_begin);
                      
                      thread_local_events.push_back(e);
                      if (!is_warmup) {
                          thread_local_contexts.push_back({
                              e, run_idx - warmup_iterations, dev_original_idx, current_dev_name_label,
                              p_part.row_begin, p_part.row_end, current_dispatch_offset_ms
                          });
                      }
                  }

                  if(!thread_local_events.empty()){
                      std::lock_guard<std::mutex> lock_ev(events_mutex);
                      current_iter_kernel_exec_events.insert(current_iter_kernel_exec_events.end(), thread_local_events.begin(), thread_local_events.end());
                  }
                  if(!thread_local_contexts.empty()){
                      std::lock_guard<std::mutex> lock_ctx(contexts_mutex);
                      current_iter_event_contexts.insert(current_iter_event_contexts.end(), thread_local_contexts.begin(), thread_local_contexts.end());
                  }
              });
          }
          for(auto& worker : submission_workers) worker.join();

      } else { 
          for (size_t pi = 0; pi < parts.size(); ++pi) {
            auto const& p = parts[pi];
            if (p.row_end <= p.row_begin) continue;
            
            int dev_original_idx = -1; 
            for(size_t i=0; i<queues.size(); ++i) if(&queues[i] == p.q) {dev_original_idx=i; break;}
            if (dev_original_idx == -1) throw std::runtime_error("Partition queue not found in device list for single-threaded dispatch");
            
            sycl::queue& current_q_ref = queues[dev_original_idx]; // Mutable reference
            const auto& current_m_ref = mems[dev_original_idx];
            std::string current_dev_name_label = get_device_label(current_q_ref.get_device(), dev_original_idx);

            double current_dispatch_offset_ms = ms_duration_fn(t_kernel_submission_phase_start_wall, high_resolution_clock::now());
            sycl::event e = current_q_ref.get_device().is_gpu() ?
              spmv::spmv_gpu(current_q_ref, p.row_end-p.row_begin, current_m_ref.rp+p.row_begin, current_m_ref.ci, current_m_ref.v, current_m_ref.x, current_m_ref.y+p.row_begin) :
              spmv::spmv_cpu(current_q_ref, p.row_end-p.row_begin, current_m_ref.rp+p.row_begin, current_m_ref.ci, current_m_ref.v, current_m_ref.x, current_m_ref.y+p.row_begin);
            
            current_iter_kernel_exec_events.push_back(e);
            if (!is_warmup) {
                 current_iter_event_contexts.push_back({ 
                    e, run_idx - warmup_iterations, dev_original_idx, current_dev_name_label,
                    p.row_begin, p.row_end, current_dispatch_offset_ms
                });
            }
          }
      }
      auto t_kernel_submission_phase_end_wall = high_resolution_clock::now();
      if (!is_warmup) iter_kernel_submission_ms.push_back(ms_duration_fn(t_kernel_submission_phase_start_wall, t_kernel_submission_phase_end_wall));
      
      sycl::event::wait_and_throw(current_iter_kernel_exec_events);
      auto t_kernel_sync_phase_end_wall = high_resolution_clock::now();
      if (!is_warmup) {
        iter_kernel_sync_ms.push_back(ms_duration_fn(t_kernel_submission_phase_end_wall, t_kernel_sync_phase_end_wall));
        iter_overall_kernel_phase_ms.push_back(ms_duration_fn(t_kernel_submission_phase_start_wall, t_kernel_sync_phase_end_wall));
        
        std::vector<double> current_iter_dev_total_kernel_event_time(final_selected_devices.size(), 0.0);
        for(const auto& ctx : current_iter_event_contexts) {
            double kernel_event_time_ms = 0.0;
            try {
                 uint64_t cmd_start_ns = ctx.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                 uint64_t cmd_end_ns   = ctx.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                 kernel_event_time_ms = static_cast<double>(cmd_end_ns - cmd_start_ns) * 1e-6;
            } catch (const sycl::exception& e) {
                std::cerr << "Warning: Profiling info error iter " << ctx.timed_iter_idx 
                          << " dev " << ctx.dev_name_label_str << ": " << e.what() << std::endl;
            }
            current_iter_dev_total_kernel_event_time[ctx.dev_experimental_idx] += kernel_event_time_ms;
            all_detailed_dev_recs.push_back({
                ctx.timed_iter_idx, ctx.dev_experimental_idx, ctx.dev_name_label_str,
                ctx.p_row_begin, ctx.p_row_end,
                ctx.dispatch_offset_ms_val, kernel_event_time_ms
            });
        }
        for(size_t i=0; i<final_selected_devices.size(); ++i) iter_device_sum_kernel_event_ms[i].push_back(current_iter_dev_total_kernel_event_time[i]);
      }
      
  if ((sched_name == "feedback" || sched_name == "adaptive_lb" || sched_name == "bandit") 
      && !current_iter_kernel_exec_events.empty()) {
          std::vector<double> times_for_feedback(final_selected_devices.size(), 0.0);
          for(const auto& ctx : current_iter_event_contexts) {
              if (ctx.dev_experimental_idx >=0 && ctx.dev_experimental_idx < final_selected_devices.size()) {
                  try {
                    uint64_t cmd_s = ctx.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    uint64_t cmd_e = ctx.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    times_for_feedback[ctx.dev_experimental_idx] += static_cast<double>(cmd_e - cmd_s) * 1e-6;
                  } catch(...){} 
              }
          }
          scheduler->update_times(times_for_feedback);
      }

      auto t_copyback_s_wall = high_resolution_clock::now();
      std::fill(h_y_result_buffer.begin(), h_y_result_buffer.end(), 0.0f);
      std::vector<sycl::event> copy_back_events;
      copy_back_events.reserve(parts.size()); 

      for (const auto& p : parts) {
          if (p.row_end <= p.row_begin) continue;
          int dev_original_idx = -1;
          for(size_t i=0; i<queues.size(); ++i) if(&queues[i] == p.q) {dev_original_idx=i; break;}
          if(dev_original_idx != -1) {
            copy_back_events.push_back(queues[dev_original_idx].memcpy(h_y_result_buffer.data() + p.row_begin, 
                                                                       mems[dev_original_idx].y + p.row_begin, 
                                                                      (p.row_end - p.row_begin) * sizeof(float)));
          }
      }
      sycl::event::wait_and_throw(copy_back_events);
      auto t_copyback_e_wall = high_resolution_clock::now();
      if (!is_warmup) iter_copyback_ms.push_back(ms_duration_fn(t_copyback_s_wall, t_copyback_e_wall));
      
      if (!is_warmup) iter_total_ms.push_back(ms_duration_fn(t_iter_start_wall, high_resolution_clock::now()));
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
  double avg_overall_kernel_phase_ms_calc = calculate_avg(iter_overall_kernel_phase_ms);
  double avg_copyback_ms_calc = calculate_avg(iter_copyback_ms);
  double avg_total_iter_ms = calculate_avg(iter_total_ms);
  
  std::vector<double> avg_kernel_event_ms_per_device(final_selected_devices.size(), 0.0);
  double total_sum_avg_sycl_kernel_event_ms = 0;

  if (scheduler->is_dynamic()) {
      avg_overall_kernel_phase_ms_calc = dyn_avg_overall_kernel_phase_ms; 
      avg_copyback_ms_calc = dyn_avg_copyback_ms; 
      std::map<int, double> sum_kernel_ms_map; 
      for(const auto& rec : all_detailed_dev_recs) {
          sum_kernel_ms_map[rec.dev_idx] += rec.kernel_duration_ms;
      }
      for(size_t i = 0; i < final_selected_devices.size(); ++i) {
          if (timed_iterations > 0) {
              // Check if device 'i' had any work in dynamic mode
              if (sum_kernel_ms_map.count(i)) { 
                 avg_kernel_event_ms_per_device[i] = sum_kernel_ms_map[i] / timed_iterations;
              } else {
                 avg_kernel_event_ms_per_device[i] = 0.0;
              }
          }
          total_sum_avg_sycl_kernel_event_ms += avg_kernel_event_ms_per_device[i];
      }

  } else { 
      for(size_t i=0; i<final_selected_devices.size(); ++i) {
        avg_kernel_event_ms_per_device[i] = calculate_avg(iter_device_sum_kernel_event_ms[i]);
        total_sum_avg_sycl_kernel_event_ms += avg_kernel_event_ms_per_device[i];
      }
  }
  
  std::vector<double> avg_device_utilization_pct(final_selected_devices.size(), 0.0);
  if (avg_overall_kernel_phase_ms_calc > 1e-9) {
    for(size_t i=0; i<final_selected_devices.size(); ++i) {
      avg_device_utilization_pct[i] = std::min(100.0, (avg_kernel_event_ms_per_device[i] / avg_overall_kernel_phase_ms_calc) * 100.0);
    }
  }
  
  float y0_check = h_y_result_buffer.empty() ? 0.0f : h_y_result_buffer[0];

  std::ofstream os(summary_csv_path);
  os << std::fixed << std::setprecision(6);
  os << "matrix_path,scheduler_name,dispatch_mode,num_rows,num_cols,num_nonzeros,warmup_iterations,timed_iterations,"
     << "load_ms,sycl_setup_ms,usm_alloc_ms,initial_h2d_wall_ms,initial_h2d_event_sum_ms,"
     << "avg_y_reset_ms,avg_sched_ms,avg_kernel_submission_ms,avg_kernel_sync_ms,avg_overall_kernel_phase_wall_ms,"
     << "avg_d2h_ms,avg_total_iter_ms,total_avg_sycl_kernel_event_ms,y0_check"; 
  for(const auto& label : device_name_labels_for_header) { 
      os << ",avg_kernel_event_ms_" << label << ",avg_util_" << label << "_pct";
  }
  os << "\n";

  os << matrix_path << "," << sched_name << "," << dispatch_mode_str << "," << A.nrows << "," << A.ncols << "," << A.nnz << ","
     << warmup_iterations << "," << timed_iterations << ","
     << load_ms << "," << sycl_setup_ms << "," << usm_alloc_ms << "," << initial_h2d_wall_ms << "," << sum_initial_h2d_event_ms << ","
     << avg_y_reset_ms << "," << avg_sched_ms << "," << avg_kernel_submission_ms << "," << avg_kernel_sync_ms << "," << avg_overall_kernel_phase_ms_calc << ","
     << avg_copyback_ms_calc << "," << avg_total_iter_ms << "," << total_sum_avg_sycl_kernel_event_ms << "," << y0_check;
  for(size_t i=0; i<final_selected_devices.size(); ++i) {
      os << "," << avg_kernel_event_ms_per_device[i] << "," << avg_device_utilization_pct[i];
  }
  os << "\n";
  os.close();
  cout << "Summary results written to " << summary_csv_path << endl;


  std::ofstream od(devices_csv_path);
  od << std::fixed << std::setprecision(6);
  od << "timed_iteration_idx,device_exp_idx,device_name_label,row_begin,row_end,host_dispatch_offset_ms,kernel_duration_ms\n";
  for (const auto& r : all_detailed_dev_recs) {
    od << r.iter << ","                
       << r.dev_idx << ","             
       << r.device_name_label << ","  
       << r.row_begin << ","
       << r.row_end << ","
       << r.host_dispatch_offset_ms << ","
       << r.kernel_duration_ms << "\n";
  }
  od.close();
  cout << "Device details written to " << devices_csv_path << endl;


  std::cout << "Completed " << warmup_iterations << " warmup, " << timed_iterations << " timed runs for scheduler '" << sched_name 
            << "' on matrix " << matrix_path << " with dispatch '" << dispatch_mode_str << "'\n";

  for (size_t i = 0; i < final_selected_devices.size(); ++i) {
    sycl::free(mems[i].rp, queues[i]); sycl::free(mems[i].ci, queues[i]);
    sycl::free(mems[i].v, queues[i]); sycl::free(mems[i].x, queues[i]);
    sycl::free(mems[i].y, queues[i]);
  }
  cout << "USM memory freed." << endl;
  return 0;
}