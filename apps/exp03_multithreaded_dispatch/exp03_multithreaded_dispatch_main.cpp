// apps/exp01_baseline_spmv/exp01_baseline_spmv_main.cpp
#include "spmv/csr.hpp"
#include "spmv/kernels.hpp" // For spmv_cpu, spmv_gpu
#include <sycl/sycl.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <map>
#include <thread> // For std::thread
#include <mutex>  // For std::mutex

// Helper struct for USM memory per device
struct DeviceUsmMem {
    sycl::queue q;
    int* rp = nullptr;
    int* ci = nullptr;
    float* v = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    std::string device_name_full;
    std::string device_name_label; // e.g., "dev0_GPU"
    int device_exp_idx = 0;
};

// Helper struct for a work partition
struct WorkPartition {
    int row_begin;
    int row_end;
    DeviceUsmMem* device_mem;
};

// REVISED: Struct for detailed kernel dispatch and execution records
struct KernelDispatchDetail {
    int timed_iteration_idx;
    int device_experimental_idx;
    std::string device_name_label;
    int part_row_begin;
    int part_row_end;
    double host_dispatch_offset_ms; // Offset from start of iteration's kernel submission phase
    double kernel_duration_ms;      // Actual device execution time from SYCL event
};

// REVISED: Struct to temporarily hold event and its context
struct EventDispatchContext {
    sycl::event event;
    int timed_iter_idx;
    int dev_exp_idx;
    std::string dev_name_label;
    int p_row_begin;
    int p_row_end;
    double dispatch_offset_ms_val; // Store the calculated host dispatch offset
};

// --- Timing Helper ---
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
auto ms_duration = [](TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
};

// --- Main Application Logic ---
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix.mtx> <output.csv> <num_warmup_runs> <num_timed_runs> <config_type> <device_indices_str_placeholder>" << std::endl;
        return 1;
    }

    std::string matrix_path = argv[1];
    std::string output_csv_path = argv[2];
    int num_warmup_runs = std::stoi(argv[3]);
    int num_timed_runs = std::stoi(argv[4]);
    std::string config_type = argv[5];
    // std::string device_indices_str = argv[6]; // Placeholder

    TimePoint t_load_start = Clock::now();
    spmv::CSR A = spmv::CSR::load_mm(matrix_path);
    TimePoint t_load_end = Clock::now();
    double load_ms = ms_duration(t_load_start, t_load_end);
    if (A.empty()) { std::cerr << "Error: Failed to load matrix " << matrix_path << std::endl; return 1; }
    std::cout << "Matrix " << matrix_path << " loaded: " << A.nrows << "x" << A.ncols << ", " << A.nnz << " non-zeros." << std::endl;

    TimePoint t_sycl_setup_start = Clock::now();
    std::vector<sycl::device> all_cpu_devs, all_gpu_devs;
    for (auto& p : sycl::platform::get_platforms()) {
        for (auto& d : p.get_devices()) {
            if (d.is_cpu() && all_cpu_devs.empty()) all_cpu_devs.push_back(d);
            else if (d.is_gpu()) all_gpu_devs.push_back(d);
        }
    }
    std::sort(all_gpu_devs.begin(), all_gpu_devs.end(), [](const sycl::device& a, const sycl::device& b){
        return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
    });

    std::vector<DeviceUsmMem> selected_devices_mem;
    std::vector<sycl::device> current_experiment_devices;

    if (config_type == "cpu1") {
        if (all_cpu_devs.empty()) { std::cerr << "No CPU found for cpu1." << std::endl; return 1; }
        current_experiment_devices.push_back(all_cpu_devs[0]);
    } else if (config_type == "gpu1") {
        if (all_gpu_devs.empty()) { std::cerr << "No GPU found for gpu1." << std::endl; return 1; }
        current_experiment_devices.push_back(all_gpu_devs[0]);
    } else if (config_type == "gpu2_split") {
        if (all_gpu_devs.size() < 2) { std::cerr << "Need >= 2 GPUs for gpu2_split." << std::endl; return 1; }
        current_experiment_devices.push_back(all_gpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[1]);
    } else if (config_type == "cpu1gpu2_split") {
        if (all_cpu_devs.empty() || all_gpu_devs.size() < 2) { std::cerr << "Need 1 CPU and >= 2 GPUs for cpu1gpu2_split." << std::endl; return 1; }
        current_experiment_devices.push_back(all_cpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[1]);
    } else { std::cerr << "Unknown config_type: " << config_type << std::endl; return 1;}
    
    int dev_exp_idx_counter = 0;
    for(const auto& dev : current_experiment_devices) {
        selected_devices_mem.emplace_back();
        DeviceUsmMem& mem = selected_devices_mem.back();
        mem.q = sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        mem.device_name_full = mem.q.get_device().get_info<sycl::info::device::name>();
        mem.device_exp_idx = dev_exp_idx_counter++;
        std::string type_str = mem.q.get_device().is_gpu() ? "GPU" : (mem.q.get_device().is_cpu() ? "CPU" : "DEV");
        mem.device_name_label = "dev" + std::to_string(mem.device_exp_idx) + "_" + type_str;
        std::cout << "Using device [" << mem.device_exp_idx << "] " << mem.device_name_label << ": " << mem.device_name_full << std::endl;
    }
    TimePoint t_sycl_setup_end = Clock::now();
    double sycl_setup_ms = ms_duration(t_sycl_setup_start, t_sycl_setup_end);

    TimePoint t_alloc_start = Clock::now();
    for (DeviceUsmMem& mem : selected_devices_mem) {
        mem.rp = sycl::malloc_device<int>(A.row_ptr.size(), mem.q);
        mem.ci = sycl::malloc_device<int>(A.col_idx.size(), mem.q);
        mem.v  = sycl::malloc_device<float>(A.vals.size(), mem.q);
        mem.x  = sycl::malloc_device<float>(A.ncols, mem.q);
        mem.y  = sycl::malloc_device<float>(A.nrows, mem.q);
    }
    for (DeviceUsmMem& mem : selected_devices_mem) { mem.q.wait(); }
    TimePoint t_alloc_end = Clock::now();
    double usm_alloc_ms = ms_duration(t_alloc_start, t_alloc_end);

    std::vector<float> h_x(A.ncols, 1.0f);
    std::vector<float> h_y_zeros(A.nrows, 0.0f);
    std::vector<sycl::event> initial_transfer_events_vec;
    TimePoint t_initial_h2d_start = Clock::now();
    for (DeviceUsmMem& mem : selected_devices_mem) {
        initial_transfer_events_vec.push_back(mem.q.memcpy(mem.rp, A.row_ptr.data(), A.row_ptr.size() * sizeof(int)));
        initial_transfer_events_vec.push_back(mem.q.memcpy(mem.ci, A.col_idx.data(), A.col_idx.size() * sizeof(int)));
        initial_transfer_events_vec.push_back(mem.q.memcpy(mem.v, A.vals.data(), A.vals.size() * sizeof(float)));
        initial_transfer_events_vec.push_back(mem.q.memcpy(mem.x, h_x.data(), h_x.size() * sizeof(float)));
    }
    sycl::event::wait_and_throw(initial_transfer_events_vec);
    TimePoint t_initial_h2d_end = Clock::now();
    double initial_h2d_ms = ms_duration(t_initial_h2d_start, t_initial_h2d_end);
    double sum_initial_h2d_event_ms = 0;
    for(auto& e : initial_transfer_events_vec) {
        sum_initial_h2d_event_ms += (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
    }

    std::vector<WorkPartition> partitions;
    int D_actual = selected_devices_mem.size();
    if (D_actual > 0) {
        if (config_type == "cpu1" || config_type == "gpu1") {
            if (D_actual >= 1) partitions.push_back({0, A.nrows, &selected_devices_mem[0]});
        } else if (config_type == "gpu2_split") {
            if (D_actual >= 2) {
                int split_point = A.nrows / 2;
                partitions.push_back({0, split_point, &selected_devices_mem[0]});
                partitions.push_back({split_point, A.nrows, &selected_devices_mem[1]});
            } else if (D_actual == 1) partitions.push_back({0, A.nrows, &selected_devices_mem[0]});
        } else if (config_type == "cpu1gpu2_split") {
            if (D_actual >= 3) {
                int s1 = A.nrows / 3; int s2 = 2 * (A.nrows / 3);
                partitions.push_back({0, s1, &selected_devices_mem[0]});      // CPU
                partitions.push_back({s1, s2, &selected_devices_mem[1]});    // GPU0
                partitions.push_back({s2, A.nrows, &selected_devices_mem[2]});// GPU1
            } // Add fallbacks for D_actual < 3 if necessary
        }
        if (!partitions.empty() && partitions.back().row_end < A.nrows) partitions.back().row_end = A.nrows;
        else if (partitions.empty() && D_actual > 0 && A.nrows > 0) partitions.push_back({0, A.nrows, &selected_devices_mem[0]});
    }

    std::vector<double> y_reset_times_ms_all_iters, kernel_submission_wall_times_ms_all_iters, kernel_sync_wall_times_ms_all_iters, d2h_result_times_ms_all_iters, total_iteration_wall_times_ms_all_iters, overall_kernel_phase_wall_times_ms_all_iters;
    std::vector<std::vector<double>> per_device_sycl_kernel_event_times_ms(selected_devices_mem.size());
    std::vector<KernelDispatchDetail> all_kernel_dispatch_details;

    std::cout << "Starting " << num_warmup_runs << " warmup and " << num_timed_runs << " timed runs for config " << config_type << "..." << std::endl;
    for (int iter = 0; iter < num_warmup_runs + num_timed_runs; ++iter) {
        bool is_warmup = iter < num_warmup_runs;
        if (!is_warmup && iter == num_warmup_runs) std::cout << "Warmup complete, starting timed runs for " << config_type << "." << std::endl;

        TimePoint iter_total_start = Clock::now();
        std::vector<sycl::event> y_reset_events;
        TimePoint t_y_reset_start = Clock::now();
        for (DeviceUsmMem& mem : selected_devices_mem) {
            y_reset_events.push_back(mem.q.memcpy(mem.y, h_y_zeros.data(), A.nrows * sizeof(float)));
        }
        sycl::event::wait_and_throw(y_reset_events);
        TimePoint t_y_reset_end = Clock::now();
        if (!is_warmup) y_reset_times_ms_all_iters.push_back(ms_duration(t_y_reset_start, t_y_reset_end));
        
        // --- Start of Block to Insert/Replace for Kernel Submission ---
        TimePoint iteration_dispatch_phase_start_tp = Clock::now(); // Host ref for this iter's dispatch
        std::vector<sycl::event> kernel_exec_events_this_iter_all_threads; // Collect all events
        std::vector<EventDispatchContext> event_contexts_this_iter_all_threads; // Collect all contexts
        std::mutex events_mutex, contexts_mutex; // Mutexes for shared vectors

        // Check if multi-threading is beneficial/applicable for this configuration
        bool use_multithreaded_dispatch = false;
        if (selected_devices_mem.size() > 1 && 
            (config_type == "gpu2_split" || config_type == "cpu1gpu2_split")) {
            use_multithreaded_dispatch = true;
        }

        if (use_multithreaded_dispatch) {
            // --- MULTI-THREADED DISPATCH for multi-device configurations ---
            std::vector<std::thread> submission_workers;
            
            // Group partitions by their assigned device's experimental index.
            // This ensures each thread handles all work for one specific device.
            std::map<int, std::vector<WorkPartition>> parts_per_device_idx;
            for(const auto& p : partitions) { // 'partitions' is the vector of WorkPartition
                if (p.device_mem) { // Ensure device_mem is not null
                    parts_per_device_idx[p.device_mem->device_exp_idx].push_back(p);
                }
            }
            
            for(const auto& pair : parts_per_device_idx) {
                int current_dev_exp_idx = pair.first;
                const std::vector<WorkPartition>& device_specific_partitions = pair.second;
                
                DeviceUsmMem* mem_ptr_for_thread = nullptr;
                for(auto& s_mem : selected_devices_mem){ 
                    if(s_mem.device_exp_idx == current_dev_exp_idx) {
                        mem_ptr_for_thread = &s_mem; 
                        break;
                    } 
                }

                if(!mem_ptr_for_thread || device_specific_partitions.empty()) {
                    // Should not happen if partitions were correctly made for selected_devices_mem
                    continue; 
                }

                submission_workers.emplace_back([&, current_dev_exp_idx, mem_ptr_for_thread, device_specific_partitions, iteration_dispatch_phase_start_tp, is_warmup, iter, num_warmup_runs]() {
                    std::vector<sycl::event> thread_local_events;
                    std::vector<EventDispatchContext> thread_local_contexts;

                    for (const auto& p_part : device_specific_partitions) {
                        if (p_part.row_end <= p_part.row_begin) continue;
                        
                        // Record host dispatch offset for this specific part
                        // Clock::now() is taken by the thread just before its submission attempt
                        double current_dispatch_offset_ms = ms_duration(iteration_dispatch_phase_start_tp, Clock::now());
                        sycl::event e;
                        if (mem_ptr_for_thread->q.get_device().is_cpu()) {
                            e = spmv::spmv_cpu(mem_ptr_for_thread->q, (p_part.row_end - p_part.row_begin), mem_ptr_for_thread->rp + p_part.row_begin, mem_ptr_for_thread->ci, mem_ptr_for_thread->v, mem_ptr_for_thread->x, mem_ptr_for_thread->y + p_part.row_begin);
                        } else { // GPU
                            e = spmv::spmv_gpu(mem_ptr_for_thread->q, (p_part.row_end - p_part.row_begin), mem_ptr_for_thread->rp + p_part.row_begin, mem_ptr_for_thread->ci, mem_ptr_for_thread->v, mem_ptr_for_thread->x, mem_ptr_for_thread->y + p_part.row_begin);
                        }
                        thread_local_events.push_back(e);
                        if (!is_warmup) {
                            thread_local_contexts.push_back({e, iter - num_warmup_runs, mem_ptr_for_thread->device_exp_idx, mem_ptr_for_thread->device_name_label, p_part.row_begin, p_part.row_end, current_dispatch_offset_ms});
                        }
                    } // End loop over parts for this device/thread

                    if(!thread_local_events.empty()){
                        std::lock_guard<std::mutex> lock_ev(events_mutex);
                        kernel_exec_events_this_iter_all_threads.insert(kernel_exec_events_this_iter_all_threads.end(), thread_local_events.begin(), thread_local_events.end());
                    }
                    if(!thread_local_contexts.empty()){
                        std::lock_guard<std::mutex> lock_ctx(contexts_mutex);
                        event_contexts_this_iter_all_threads.insert(event_contexts_this_iter_all_threads.end(), thread_local_contexts.begin(), thread_local_contexts.end());
                    }
                }); // End of lambda for submission_workers.emplace_back
            } // End loop for creating worker threads

            for (auto& worker : submission_workers) { // Wait for all host threads to finish submitting
                worker.join(); 
            }

        } else {
            // --- SINGLE-THREADED DISPATCH for single-device configurations (cpu1, gpu1) ---
            // This is the original loop from exp01_baseline_spmv_main.cpp
            for (const auto& p : partitions) {
                if (p.row_end <= p.row_begin) continue;
                DeviceUsmMem* mem_ptr = p.device_mem;
                double current_dispatch_offset_ms = ms_duration(iteration_dispatch_phase_start_tp, Clock::now());
                sycl::event e;
                if (mem_ptr->q.get_device().is_cpu()) {
                    e = spmv::spmv_cpu(mem_ptr->q, (p.row_end - p.row_begin), mem_ptr->rp + p.row_begin, mem_ptr->ci, mem_ptr->v, mem_ptr->x, mem_ptr->y + p.row_begin);
                } else {
                    e = spmv::spmv_gpu(mem_ptr->q, (p.row_end - p.row_begin), mem_ptr->rp + p.row_begin, mem_ptr->ci, mem_ptr->v, mem_ptr->x, mem_ptr->y + p.row_begin);
                }
                kernel_exec_events_this_iter_all_threads.push_back(e);
                if (!is_warmup) {
                    event_contexts_this_iter_all_threads.push_back({e, iter - num_warmup_runs, mem_ptr->device_exp_idx, mem_ptr->device_name_label, p.row_begin, p.row_end, current_dispatch_offset_ms});
                }
            }
        }
        TimePoint actual_kernel_submission_phase_end_tp = Clock::now(); // Host is done submitting
        
        sycl::event::wait_and_throw(kernel_exec_events_this_iter_all_threads); // Wait for device kernels
        TimePoint actual_kernel_sync_phase_end_tp = Clock::now();

        if (!is_warmup) {
            kernel_submission_wall_times_ms_all_iters.push_back(ms_duration(iteration_dispatch_phase_start_tp, actual_kernel_submission_phase_end_tp));
            kernel_sync_wall_times_ms_all_iters.push_back(ms_duration(actual_kernel_submission_phase_end_tp, actual_kernel_sync_phase_end_tp));
            overall_kernel_phase_wall_times_ms_all_iters.push_back(ms_duration(iteration_dispatch_phase_start_tp, actual_kernel_sync_phase_end_tp));

            std::vector<double> current_iter_device_total_kernel_event_time(selected_devices_mem.size(), 0.0);
            // Sort contexts by device_exp_idx and then row_begin if strict order is needed for profiling display,
            // but for populating all_kernel_dispatch_details, the order of processing contexts doesn't strictly matter.
            // If event_contexts_this_iter_all_threads might be populated out of partition order by threads,
            // sorting might be good practice before processing, though not strictly necessary for sum.
            // std::sort(event_contexts_this_iter_all_threads.begin(), ... ); // Optional sort

            for(const auto& ctx : event_contexts_this_iter_all_threads) {
                double kernel_event_time_ms = 0.0;
                // uint64_t cmd_start_ns = 0; // Not strictly needed for host_dispatch_offset plot
                try {
                     uint64_t cmd_s = ctx.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                     uint64_t cmd_e   = ctx.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                     kernel_event_time_ms = static_cast<double>(cmd_e - cmd_s) * 1e-6;
                } catch (const sycl::exception& e) {
                    std::cerr << "Warning: Profiling info error in iter " << ctx.timed_iter_idx 
                              << " for device " << ctx.dev_name_label << " (exp_idx " << ctx.dev_exp_idx << "). SYCL what(): " << e.what() << std::endl;
                    kernel_event_time_ms = 0.0; // Assign a default/error value
                }
                
                current_iter_device_total_kernel_event_time[ctx.dev_exp_idx] += kernel_event_time_ms;
                
                all_kernel_dispatch_details.push_back({
                    ctx.timed_iter_idx, ctx.dev_exp_idx, ctx.dev_name_label,
                    ctx.p_row_begin, ctx.p_row_end,
                    ctx.dispatch_offset_ms_val, // This is the crucial host_dispatch_offset_ms
                    kernel_event_time_ms
                });
            }
            for(size_t d_idx=0; d_idx < selected_devices_mem.size(); ++d_idx) {
                per_device_sycl_kernel_event_times_ms[d_idx].push_back(current_iter_device_total_kernel_event_time[d_idx]);
            }
        }
        // --- End of Block to Insert/Replace ---
        
        std::vector<float> h_y_result_this_iter(A.nrows); 
        std::vector<sycl::event> d2h_events;
        TimePoint t_d2h_start = Clock::now();
        for (const auto& p : partitions) { 
            if (p.row_end <= p.row_begin) continue;
            d2h_events.push_back(p.device_mem->q.memcpy(h_y_result_this_iter.data() + p.row_begin, p.device_mem->y + p.row_begin, (p.row_end - p.row_begin) * sizeof(float)));
        }
        sycl::event::wait_and_throw(d2h_events);
        TimePoint t_d2h_end = Clock::now();
        if (!is_warmup) d2h_result_times_ms_all_iters.push_back(ms_duration(t_d2h_start, t_d2h_end));
        
        TimePoint iter_total_end = Clock::now();
        if(!is_warmup) total_iteration_wall_times_ms_all_iters.push_back(ms_duration(iter_total_start, iter_total_end));
    }

    auto calculate_avg = [](const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    double avg_y_reset_ms = calculate_avg(y_reset_times_ms_all_iters);
    double avg_kernel_submission_ms = calculate_avg(kernel_submission_wall_times_ms_all_iters);
    double avg_kernel_sync_ms = calculate_avg(kernel_sync_wall_times_ms_all_iters);
    double avg_d2h_ms = calculate_avg(d2h_result_times_ms_all_iters);
    double avg_total_iter_ms = calculate_avg(total_iteration_wall_times_ms_all_iters);
    double avg_overall_kernel_phase_wall_ms = calculate_avg(overall_kernel_phase_wall_times_ms_all_iters);
    std::vector<double> avg_per_device_sycl_kernel_event_ms(selected_devices_mem.size());
    double total_sum_avg_sycl_kernel_event_ms = 0;
    for(size_t d_idx=0; d_idx < selected_devices_mem.size(); ++d_idx) {
        avg_per_device_sycl_kernel_event_ms[d_idx] = calculate_avg(per_device_sycl_kernel_event_times_ms[d_idx]);
        total_sum_avg_sycl_kernel_event_ms += avg_per_device_sycl_kernel_event_ms[d_idx];
    }
    std::vector<double> avg_device_utilization_pct(selected_devices_mem.size(), 0.0);
    if (avg_overall_kernel_phase_wall_ms > 1e-9) {
        for(size_t i=0; i < selected_devices_mem.size(); ++i) {
            avg_device_utilization_pct[i] = std::min(100.0, (avg_per_device_sycl_kernel_event_ms[i] / avg_overall_kernel_phase_wall_ms) * 100.0);
        }
    }

    std::ofstream out_summary_file(output_csv_path);
    out_summary_file << std::fixed << std::setprecision(6);
    out_summary_file << "matrix_path,config_type,num_rows,num_cols,num_nonzeros,warmup_runs,timed_runs,"
             << "load_ms,sycl_setup_ms,usm_alloc_ms,initial_h2d_wall_ms,initial_h2d_event_sum_ms,"
             << "avg_y_reset_ms,avg_sched_ms,avg_kernel_submission_ms,avg_kernel_sync_ms,avg_overall_kernel_phase_wall_ms,"
             << "avg_d2h_ms,avg_total_iter_ms,total_avg_sycl_kernel_event_ms";
    for(const auto& mem_info : selected_devices_mem) {
        out_summary_file << ",avg_kernel_event_ms_" << mem_info.device_name_label;
        out_summary_file << ",avg_util_" << mem_info.device_name_label << "_pct";
    }
    out_summary_file << std::endl;
    out_summary_file << matrix_path << "," << config_type << "," << A.nrows << "," << A.ncols << "," << A.nnz << ","
             << num_warmup_runs << "," << num_timed_runs << ","
             << load_ms << "," << sycl_setup_ms << "," << usm_alloc_ms << "," << initial_h2d_ms << "," << sum_initial_h2d_event_ms << ","
             << avg_y_reset_ms << "," << 0.0 << "," // avg_sched_ms for exp01 is effectively 0
             << avg_kernel_submission_ms << "," << avg_kernel_sync_ms << "," << avg_overall_kernel_phase_wall_ms << ","
             << avg_d2h_ms << "," << avg_total_iter_ms << "," << total_sum_avg_sycl_kernel_event_ms;
    for(size_t d_idx=0; d_idx < selected_devices_mem.size(); ++d_idx) {
        out_summary_file << "," << avg_per_device_sycl_kernel_event_ms[d_idx];
        out_summary_file << "," << avg_device_utilization_pct[d_idx];
    }
    out_summary_file << std::endl;
    out_summary_file.close();
    std::cout << "Summary results written to " << output_csv_path << std::endl;

    if (!all_kernel_dispatch_details.empty()) {
        std::string devices_csv_filename = output_csv_path;
        size_t pos_results = devices_csv_filename.find("results_");
        if (pos_results != std::string::npos) {
            devices_csv_filename.replace(pos_results, std::string("results_").length(), "devices_");
        } else {
            size_t last_slash = devices_csv_filename.find_last_of("/\\");
            if (last_slash == std::string::npos) devices_csv_filename = "devices_" + devices_csv_filename;
            else devices_csv_filename.insert(last_slash + 1, "devices_");
        }
        std::ofstream out_devices_file(devices_csv_filename);
        out_devices_file << std::fixed << std::setprecision(6);
        out_devices_file << "timed_iteration_idx,device_exp_idx,device_name_label,row_begin,row_end,host_dispatch_offset_ms,kernel_duration_ms\n";
        for (const auto& detail : all_kernel_dispatch_details) {
            out_devices_file << detail.timed_iteration_idx << ","
                             << detail.device_experimental_idx << ","
                             << detail.device_name_label << ","
                             << detail.part_row_begin << ","
                             << detail.part_row_end << ","
                             << detail.host_dispatch_offset_ms << ","
                             << detail.kernel_duration_ms << "\n";
        }
        out_devices_file.close();
        std::cout << "Device dispatch details written to " << devices_csv_filename << std::endl;
    }

    for (DeviceUsmMem& mem : selected_devices_mem) {
        sycl::free(mem.rp, mem.q); sycl::free(mem.ci, mem.q);
        sycl::free(mem.v, mem.q); sycl::free(mem.x, mem.q);
        sycl::free(mem.y, mem.q);
    }
    std::cout << "USM memory freed." << std::endl;
    return 0;
}