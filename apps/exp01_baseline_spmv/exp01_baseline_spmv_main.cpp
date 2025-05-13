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
#include <iomanip> // For std::fixed and std::setprecision
#include <algorithm> // For std::min, std::max

// Helper struct for USM memory per device
struct DeviceUsmMem {
    sycl::queue q; // Store the queue along with memory for context
    int* rp = nullptr;
    int* ci = nullptr;
    float* v = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    // Store device name for reporting
    std::string device_name; 
    // Store a simple index (0, 1, 2...) for which device this is in the current experiment
    int device_exp_idx = 0; 
};

// Helper struct for a work partition
struct WorkPartition {
    int row_begin;
    int row_end;
    DeviceUsmMem* device_mem; // Pointer to the DeviceUsmMem to use
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
                  << " <matrix.mtx> <output.csv> <num_warmup_runs> <num_timed_runs> <config_type> <device_indices_str>" << std::endl;
        std::cerr << "  config_type: cpu1, gpu1, gpu2_split, cpu1gpu2_split" << std::endl;
        std::cerr << "  device_indices_str: Comma-separated indices (e.g., '0' for first GPU, '0,1' for first two GPUs, 'c0,g0,g1' for cpu0,gpu0,gpu1 )" << std::endl;
        return 1;
    }

    std::string matrix_path = argv[1];
    std::string output_csv_path = argv[2];
    int num_warmup_runs = std::stoi(argv[3]);
    int num_timed_runs = std::stoi(argv[4]);
    std::string config_type = argv[5];
    std::string device_indices_str = argv[6]; // New argument to specify which devices

    // --- Phase 1: CSR Matrix Loading ---
    TimePoint t_load_start = Clock::now();
    spmv::CSR A = spmv::CSR::load_mm(matrix_path);
    TimePoint t_load_end = Clock::now();
    double load_ms = ms_duration(t_load_start, t_load_end);

    if (A.empty()) {
        std::cerr << "Error: Failed to load or empty CSR matrix from " << matrix_path << std::endl;
        return 1;
    }
    std::cout << "Matrix " << matrix_path << " loaded: " << A.nrows << "x" << A.ncols << ", " << A.nnz << " non-zeros." << std::endl;

    // --- Phase 2: SYCL Device Discovery and Setup ---
    TimePoint t_sycl_setup_start = Clock::now();
    std::vector<sycl::device> all_cpu_devs;
    std::vector<sycl::device> all_gpu_devs;
    for (auto& p : sycl::platform::get_platforms()) {
        for (auto& d : p.get_devices()) {
            if (d.is_cpu() && !all_cpu_devs.size()) { // Take first CPU for simplicity
                all_cpu_devs.push_back(d);
            } else if (d.is_gpu()) {
                all_gpu_devs.push_back(d);
            }
        }
    }
     // Sort GPUs by some criteria if necessary, e.g., name, to make selection deterministic
    std::sort(all_gpu_devs.begin(), all_gpu_devs.end(), [](const sycl::device& a, const sycl::device& b){
        return a.get_info<sycl::info::device::name>() < b.get_info<sycl::info::device::name>();
    });


    std::vector<DeviceUsmMem> selected_devices_mem;
    std::vector<sycl::device> current_experiment_devices; // For creating queues

    // Parse device_indices_str and select devices
    // Example: "c0" -> 1st CPU, "g0" -> 1st GPU, "g0,g1" -> 1st and 2nd GPUs
    // This part needs careful implementation based on device_indices_str format
    // For simplicity in this example, we'll map config_type to device selections.
    // A more robust approach would parse device_indices_str to pick specific device instances.

    if (config_type == "cpu1") {
        if (all_cpu_devs.empty()) { std::cerr << "No CPU found for cpu1 config." << std::endl; return 1; }
        current_experiment_devices.push_back(all_cpu_devs[0]);
    } else if (config_type == "gpu1") {
        if (all_gpu_devs.empty()) { std::cerr << "No GPU found for gpu1 config." << std::endl; return 1; }
        current_experiment_devices.push_back(all_gpu_devs[0]);
    } else if (config_type == "gpu2_split") {
        if (all_gpu_devs.size() < 2) { std::cerr << "Need at least 2 GPUs for gpu2_split config. Found " << all_gpu_devs.size() << std::endl; return 1; }
        current_experiment_devices.push_back(all_gpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[1]);
    } else if (config_type == "cpu1gpu2_split") {
        if (all_cpu_devs.empty() || all_gpu_devs.size() < 2) { std::cerr << "Need 1 CPU and 2 GPUs for cpu1gpu2_split. Found " << all_cpu_devs.size() << " CPUs, " << all_gpu_devs.size() << " GPUs." << std::endl; return 1; }
        current_experiment_devices.push_back(all_cpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[0]);
        current_experiment_devices.push_back(all_gpu_devs[1]);
    } else {
        std::cerr << "Unknown config_type: " << config_type << std::endl;
        return 1;
    }
    
    int dev_exp_idx_counter = 0;
    for(const auto& dev : current_experiment_devices) {
        selected_devices_mem.emplace_back();
        DeviceUsmMem& mem = selected_devices_mem.back();
        mem.q = sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        mem.device_name = mem.q.get_device().get_info<sycl::info::device::name>();
        mem.device_exp_idx = dev_exp_idx_counter++;
        std::cout << "Using device [" << mem.device_exp_idx << "]: " << mem.device_name << std::endl;
    }
    TimePoint t_sycl_setup_end = Clock::now();
    double sycl_setup_ms = ms_duration(t_sycl_setup_start, t_sycl_setup_end);


    // --- Phase 3: USM Allocation ---
    TimePoint t_alloc_start = Clock::now();
    for (DeviceUsmMem& mem : selected_devices_mem) {
        mem.rp = sycl::malloc_device<int>(A.row_ptr.size(), mem.q);
        mem.ci = sycl::malloc_device<int>(A.col_idx.size(), mem.q);
        mem.v  = sycl::malloc_device<float>(A.vals.size(), mem.q);
        mem.x  = sycl::malloc_device<float>(A.ncols, mem.q);
        mem.y  = sycl::malloc_device<float>(A.nrows, mem.q);
    }
    // Wait for allocations if necessary (malloc_device is host-synchronous by default)
    for (DeviceUsmMem& mem : selected_devices_mem) { mem.q.wait(); }
    TimePoint t_alloc_end = Clock::now();
    double usm_alloc_ms = ms_duration(t_alloc_start, t_alloc_end);

    // --- Phase 4: Initial Host-to-Device Data Transfers (Matrix A, vector x) ---
    std::vector<float> h_x(A.ncols, 1.0f); // Host vector x
    std::vector<float> h_y_zeros(A.nrows, 0.0f); // Host vector y (for reset and result)
    std::vector<sycl::event> initial_transfer_events;

    TimePoint t_initial_h2d_start = Clock::now();
    for (DeviceUsmMem& mem : selected_devices_mem) {
        initial_transfer_events.push_back(mem.q.memcpy(mem.rp, A.row_ptr.data(), A.row_ptr.size() * sizeof(int)));
        initial_transfer_events.push_back(mem.q.memcpy(mem.ci, A.col_idx.data(), A.col_idx.size() * sizeof(int)));
        initial_transfer_events.push_back(mem.q.memcpy(mem.v, A.vals.data(), A.vals.size() * sizeof(float)));
        initial_transfer_events.push_back(mem.q.memcpy(mem.x, h_x.data(), h_x.size() * sizeof(float)));
        // y will be reset per iteration
    }
    sycl::event::wait_and_throw(initial_transfer_events);
    TimePoint t_initial_h2d_end = Clock::now();
    double initial_h2d_ms = ms_duration(t_initial_h2d_start, t_initial_h2d_end);
    double sum_initial_h2d_event_ms = 0;
    for(auto& e : initial_transfer_events) {
        sum_initial_h2d_event_ms += (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
    }


    // --- Phase 5: Work Partitioning ---
    std::vector<WorkPartition> partitions;
    int D = selected_devices_mem.size();
    if (D > 0) {
        if (config_type == "cpu1" || config_type == "gpu1") {
            partitions.push_back({0, A.nrows, &selected_devices_mem[0]});
        } else if (config_type == "gpu2_split" && D == 2) {
            int split_point = A.nrows / 2;
            partitions.push_back({0, split_point, &selected_devices_mem[0]});
            partitions.push_back({split_point, A.nrows, &selected_devices_mem[1]});
        } else if (config_type == "cpu1gpu2_split" && D == 3) {
            int s1 = A.nrows / 3;
            int s2 = 2 * (A.nrows / 3);
            partitions.push_back({0, s1, &selected_devices_mem[0]}); // CPU
            partitions.push_back({s1, s2, &selected_devices_mem[1]}); // GPU0
            partitions.push_back({s2, A.nrows, &selected_devices_mem[2]}); // GPU1
        } else {
             // Should have been caught earlier, but as a safeguard
            std::cerr << "Partitioning logic not defined for config_type " << config_type << " with " << D << " devices." << std::endl;
            return 1;
        }
         // Ensure last partition reaches nrows
        if (!partitions.empty()) partitions.back().row_end = A.nrows;
    }


    // --- Phase 6: Warmup and Timed Runs ---
    std::vector<double> y_reset_times_ms;
    std::vector<double> kernel_submission_wall_times_ms;
    std::vector<double> kernel_sync_wall_times_ms;
    std::vector<double> d2h_result_times_ms;
    std::vector<double> total_iteration_wall_times_ms;
    
    // NEW: To store wall time of the entire kernel phase (submission + sync) for each timed iter
    std::vector<double> overall_kernel_phase_wall_times_ms; 

    std::vector<std::vector<double>> per_device_sycl_kernel_event_times_ms(D); // D is num selected devices

    std::cout << "Starting " << num_warmup_runs << " warmup runs and " << num_timed_runs << " timed runs..." << std::endl;
    for (int iter = 0; iter < num_warmup_runs + num_timed_runs; ++iter) {
        bool is_warmup = iter < num_warmup_runs;
        if (!is_warmup && iter == num_warmup_runs) {
            std::cout << "Warmup complete, starting timed runs." << std::endl;
        }

        TimePoint iter_total_start = Clock::now();
        std::vector<sycl::event> temp_events; // Used for y_reset and d2h

        // 6.1 Reset Y vector on all devices
        TimePoint t_y_reset_start = Clock::now();
        for (DeviceUsmMem& mem : selected_devices_mem) {
            temp_events.push_back(mem.q.memcpy(mem.y, h_y_zeros.data(), A.nrows * sizeof(float)));
        }
        sycl::event::wait_and_throw(temp_events);
        TimePoint t_y_reset_end = Clock::now();
        if (!is_warmup) y_reset_times_ms.push_back(ms_duration(t_y_reset_start, t_y_reset_end));
        temp_events.clear();

        // 6.2 Kernel Dispatch & Synchronization
        TimePoint t_kernel_submission_start = Clock::now(); // Start of kernel phase
        std::vector<sycl::event> kernel_exec_events;
        for (const auto& p : partitions) {
            if (p.row_end <= p.row_begin) continue;
            DeviceUsmMem* mem_ptr = p.device_mem;
            sycl::event e;
            if (mem_ptr->q.get_device().is_cpu()) {
                e = spmv::spmv_cpu(mem_ptr->q, (p.row_end - p.row_begin),
                                   mem_ptr->rp + p.row_begin, mem_ptr->ci, mem_ptr->v,
                                   mem_ptr->x, mem_ptr->y + p.row_begin);
            } else { // GPU
                e = spmv::spmv_gpu(mem_ptr->q, (p.row_end - p.row_begin),
                                   mem_ptr->rp + p.row_begin, mem_ptr->ci, mem_ptr->v,
                                   mem_ptr->x, mem_ptr->y + p.row_begin);
            }
            kernel_exec_events.push_back(e);
        }
        TimePoint t_kernel_submission_end = Clock::now(); // End of submission part of phase
        
        sycl::event::wait_and_throw(kernel_exec_events);
        TimePoint t_kernel_sync_end = Clock::now(); // End of kernel phase (sync complete)

        if (!is_warmup) {
            kernel_submission_wall_times_ms.push_back(ms_duration(t_kernel_submission_start, t_kernel_submission_end));
            kernel_sync_wall_times_ms.push_back(ms_duration(t_kernel_submission_end, t_kernel_sync_end)); // Sync time is after submission ends
            
            // NEW: Store overall kernel phase wall time for this iteration
            overall_kernel_phase_wall_times_ms.push_back(ms_duration(t_kernel_submission_start, t_kernel_sync_end));

            // Collect SYCL kernel event times
            std::vector<double> current_iter_device_kernel_times(D, 0.0);
            for(size_t i=0; i < kernel_exec_events.size(); ++i) {
                const auto& p = partitions[i]; 
                if (p.row_end <= p.row_begin) continue;
                DeviceUsmMem* mem_ptr = p.device_mem;
                double kernel_event_time = (kernel_exec_events[i].get_profiling_info<sycl::info::event_profiling::command_end>() - 
                                           kernel_exec_events[i].get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
                current_iter_device_kernel_times[mem_ptr->device_exp_idx] += kernel_event_time;
            }
            for(int d_idx=0; d_idx < D; ++d_idx) {
                per_device_sycl_kernel_event_times_ms[d_idx].push_back(current_iter_device_kernel_times[d_idx]);
            }
        }

        // 6.4 D2H Result Transfer
        std::vector<float> h_y_result(A.nrows); 
        TimePoint t_d2h_start = Clock::now();
        for (const auto& p : partitions) {
            if (p.row_end <= p.row_begin) continue;
            DeviceUsmMem* mem_ptr = p.device_mem;
            temp_events.push_back(mem_ptr->q.memcpy(h_y_result.data() + p.row_begin, 
                                                  mem_ptr->y + p.row_begin, 
                                                  (p.row_end - p.row_begin) * sizeof(float)));
        }
        sycl::event::wait_and_throw(temp_events);
        TimePoint t_d2h_end = Clock::now();
        if (!is_warmup) d2h_result_times_ms.push_back(ms_duration(t_d2h_start, t_d2h_end));
        temp_events.clear();
        
        TimePoint iter_total_end = Clock::now();
        if(!is_warmup) total_iteration_wall_times_ms.push_back(ms_duration(iter_total_start, iter_total_end));
    }

    // --- Phase 7: Calculate Averages (for timed runs) ---
    auto calculate_avg = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    double avg_y_reset_ms = calculate_avg(y_reset_times_ms);
    double avg_kernel_submission_ms = calculate_avg(kernel_submission_wall_times_ms);
    double avg_kernel_sync_ms = calculate_avg(kernel_sync_wall_times_ms); // Time spent just in wait_and_throw
    double avg_d2h_ms = calculate_avg(d2h_result_times_ms);
    double avg_total_iter_ms = calculate_avg(total_iteration_wall_times_ms);
    
    // NEW: Average for overall kernel phase wall time
    double avg_overall_kernel_phase_wall_ms = calculate_avg(overall_kernel_phase_wall_times_ms);

    std::vector<double> avg_per_device_sycl_kernel_event_ms(D);
    double total_avg_sycl_kernel_event_ms = 0;
    for(int d_idx=0; d_idx < D; ++d_idx) {
        avg_per_device_sycl_kernel_event_ms[d_idx] = calculate_avg(per_device_sycl_kernel_event_times_ms[d_idx]);
        total_avg_sycl_kernel_event_ms += avg_per_device_sycl_kernel_event_ms[d_idx];
    }

    // --- Phase 8: Write Results to CSV ---
    std::ofstream out_file(output_csv_path);
    out_file << std::fixed << std::setprecision(6);
    // Header
    out_file << "matrix_path,config_type,num_rows,num_cols,num_nonzeros,warmup_runs,timed_runs,"
             << "load_ms,sycl_setup_ms,usm_alloc_ms,initial_h2d_wall_ms,initial_h2d_event_sum_ms,"
             << "avg_y_reset_ms,avg_kernel_submission_ms,avg_kernel_sync_ms,avg_overall_kernel_phase_wall_ms," // ADDED avg_overall_kernel_phase_wall_ms
             << "avg_d2h_ms,avg_total_iter_ms,"
             << "total_avg_sycl_kernel_event_ms";
    for(int d_idx=0; d_idx < D; ++d_idx) {
        std::string dev_label = "dev" + std::to_string(selected_devices_mem[d_idx].device_exp_idx) + "_" +
                               (selected_devices_mem[d_idx].q.get_device().is_cpu() ? "cpu" : "gpu");
        out_file << ",avg_kernel_event_ms_" << dev_label;
        out_file << ",avg_util_" << dev_label << "_pct"; // NEW UTILIZATION COLUMN
    }
    out_file << std::endl;

    // Data
    out_file << matrix_path << "," << config_type << "," << A.nrows << "," << A.ncols << "," << A.nnz << ","
             << num_warmup_runs << "," << num_timed_runs << ","
             << load_ms << "," << sycl_setup_ms << "," << usm_alloc_ms << "," << initial_h2d_ms << "," << sum_initial_h2d_event_ms << ","
             << avg_y_reset_ms << "," << avg_kernel_submission_ms << "," << avg_kernel_sync_ms << "," << avg_overall_kernel_phase_wall_ms << "," // ADDED avg_overall_kernel_phase_wall_ms
             << avg_d2h_ms << "," << avg_total_iter_ms << ","
             << total_avg_sycl_kernel_event_ms;
    for(int d_idx=0; d_idx < D; ++d_idx) {
        out_file << "," << avg_per_device_sycl_kernel_event_ms[d_idx];
        double util_pct = 0.0;
        if (avg_overall_kernel_phase_wall_ms > 1e-9) { // Avoid division by zero or tiny numbers
             // Cap utilization at 100% if event time slightly exceeds wall time due to precision,
             // though ideally, event time should be <= wall time for the phase.
            util_pct = std::min(100.0, (avg_per_device_sycl_kernel_event_ms[d_idx] / avg_overall_kernel_phase_wall_ms * 100.0));
        }
        out_file << "," << util_pct; // NEW UTILIZATION DATA
    }
    out_file << std::endl;
    out_file.close();
    std::cout << "Results written to " << output_csv_path << std::endl;

    // --- Phase 9: Free USM ---
    for (DeviceUsmMem& mem : selected_devices_mem) {
        if(mem.rp) sycl::free(mem.rp, mem.q);
        if(mem.ci) sycl::free(mem.ci, mem.q);
        if(mem.v) sycl::free(mem.v, mem.q);
        if(mem.x) sycl::free(mem.x, mem.q);
        if(mem.y) sycl::free(mem.y, mem.q);
        mem.q.wait(); // Ensure all freeing is done
    }
    std::cout << "USM memory freed." << std::endl;

    return 0;
}