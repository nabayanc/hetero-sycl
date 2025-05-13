#include "spmv/csr.hpp"
#include "spmv/scheduler.hpp"
#include "spmv/kernels.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <iomanip> // For std::fixed and std::setprecision
#include <limits>  // For std::numeric_limits

using namespace std::chrono;
using spmv::spmv_cpu;
using spmv::spmv_gpu;

// Local DeviceMem struct
struct DeviceMem {
  int* rp;
  int* ci;
  float* v;
  float* x;
  float* y;
};

// Function to apply Jacobi preconditioner (M_inv * input_vec) on the host
static void apply_jacobi_preconditioner_host(
    const std::vector<float>& diag_A_inv,
    const std::vector<float>& input_vec,
    std::vector<float>& output_vec) {
    if (diag_A_inv.size() != input_vec.size() || input_vec.size() != output_vec.size()) {
        throw std::runtime_error("Vector size mismatch in apply_jacobi_preconditioner_host");
    }
    for (size_t i = 0; i < input_vec.size(); ++i) {
        output_vec[i] = diag_A_inv[i] * input_vec[i];
    }
}


static std::vector<double>
spmvAll(std::unique_ptr<spmv::IScheduler>& sched,
        std::vector<sycl::queue> &queues,       
        const spmv::CSR &A,
        const std::vector<struct DeviceMem> &mems,
        const std::vector<float> &x_host, // This is the vector to be multiplied by A
        std::vector<float> &y_host) {     // This will store A*x_host
  
  for (size_t i = 0; i < queues.size(); ++i) {
    queues[i].memcpy(mems[i].x, x_host.data(), A.ncols * sizeof(float));
  }
  for (auto &q : queues) q.wait();

  auto parts = sched->make_plan(A.nrows, queues);

  std::vector<sycl::event> kernel_events;
  kernel_events.reserve(parts.size());

  for (size_t i = 0; i < parts.size(); ++i) {
    auto &p = parts[i];
    if (p.row_end <= p.row_begin) continue; 

    int di = -1;
    for(size_t q_idx = 0; q_idx < queues.size(); ++q_idx) {
        if (p.q == &queues[q_idx]) {
            di = static_cast<int>(q_idx);
            break;
        }
    }
    if (di == -1) {
        std::cerr << "Error: Could not map part's queue to an original queue in spmvAll." << std::endl;
        return std::vector<double>(queues.size(), 0.0);
    }
    
    const auto& current_mem = mems[di];
    int rows_for_part = p.row_end - p.row_begin;

    kernel_events.push_back(
      p.q->get_device().is_gpu()
        ? spmv_gpu(*p.q, rows_for_part,
                   current_mem.rp + p.row_begin, 
                   current_mem.ci, current_mem.v, current_mem.x, // x here is from x_host
                   current_mem.y + p.row_begin)
        : spmv_cpu(*p.q, rows_for_part,
                   current_mem.rp + p.row_begin, 
                   current_mem.ci, current_mem.v, current_mem.x, // x here is from x_host
                   current_mem.y + p.row_begin)
    );
  }
  
  sycl::event::wait_and_throw(kernel_events);

  std::vector<double> times_per_device(queues.size(), 0.0);
  for (size_t i = 0; i < kernel_events.size(); ++i) {
    auto& event = kernel_events[i];
    auto& part = parts[i]; 
    if (part.row_end <= part.row_begin) continue;

    double kernel_duration_ms = 0.0;
    try {
        kernel_duration_ms = (event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                              event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6; 
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL profiling exception: " << e.what() << std::endl;
    }
    
    int device_idx = -1;
    for (size_t q_idx = 0; q_idx < queues.size(); ++q_idx) {
      if (part.q == &queues[q_idx]) {
        device_idx = q_idx;
        break;
      }
    }
    if (device_idx != -1) {
      times_per_device[device_idx] += kernel_duration_ms;
    }
  }

  std::fill(y_host.begin(), y_host.end(), 0.0f); 
  for (size_t i = 0; i < parts.size(); ++i) {
    auto const& p = parts[i];
    if (p.row_end <= p.row_begin) continue;

    int di = -1;
    for(size_t q_idx = 0; q_idx < queues.size(); ++q_idx) {
        if (p.q == &queues[q_idx]) {
            di = static_cast<int>(q_idx);
            break;
        }
    }
     if (di == -1) {
        std::cerr << "Error: Could not map part's queue for copy-back." << std::endl;
        continue; 
    }

    queues[di].memcpy(y_host.data() + p.row_begin,
                      mems[di].y + p.row_begin,
                      (p.row_end - p.row_begin) * sizeof(float)
                     ).wait(); 
  }

  return times_per_device;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: bicgstab_cli <matrix.mtx> <scheduler> "
                 "<tol> <maxiter> <output_x.mtx>\n";
    return 1;
  }
  std::string matrix_path = argv[1];
  std::string sched_name  = argv[2];
  float       tol         = std::stof(argv[3]);
  int         maxiter     = std::stoi(argv[4]);
  std::string out_x       = argv[5];

  auto A = spmv::CSR::load_mm(matrix_path);
  if (A.empty()) {
    std::cerr << "Failed to load CSR\n"; return 1;
  }
  int N = A.nrows;
  if (N != A.ncols) {
      std::cerr << "Warning: Matrix is not square (" << N << "x" << A.ncols << "). BiCGSTAB is typically for square matrices." << std::endl;
  }


  // --- Jacobi Preconditioner: Extract Inverse Diagonal ---
  std::vector<float> diag_A_inv(N);
  const float tiny_threshold = std::numeric_limits<float>::epsilon() * 100; 
  std::vector<bool> diag_found(N, false);

  for (int r = 0; r < N; ++r) {
      // Declare intnz with type int
      for (int intnz = A.row_ptr[r]; intnz < A.row_ptr[r+1]; ++intnz) {
          if (A.col_idx[intnz] == r) { 
              if (std::abs(A.vals[intnz]) > tiny_threshold) {
                  diag_A_inv[r] = 1.0f / A.vals[intnz];
              } else {
                  diag_A_inv[r] = 1.0f; 
                  std::cout << "Warning: Diagonal element A(" << r << "," << r << ") is near zero. Setting M_inv("<<r<<","<<r<<") = 1." << std::endl;
              }
              diag_found[r] = true;
              break; 
          }
      }
      if (!diag_found[r]) { 
          diag_A_inv[r] = 1.0f; 
           std::cout << "Warning: Diagonal element A(" << r << "," << r << ") not found (assuming zero). Setting M_inv("<<r<<","<<r<<") = 1." << std::endl;
      }
  }
  std::cout << "Jacobi preconditioner (inverse diagonal) computed." << std::endl;
  // --- End Jacobi Preconditioner Setup ---


  std::vector<sycl::device> devs;
  for (auto &p : sycl::platform::get_platforms()) {
    for (auto &d : p.get_devices()) {
      if (d.is_gpu() || d.is_cpu()) {
        devs.push_back(d);
        std::cout << "Found device: " << d.get_info<sycl::info::device::name>() << std::endl;
      }
    }
  }
   if (devs.empty()) {
      std::cerr << "No suitable SYCL devices (GPU or CPU) found.\n";
      return 1;
  }
  std::cout << "Using " << devs.size() << " devices for BiCGSTAB." << std::endl;

  std::vector<sycl::queue> queues; 
  for (auto &d : devs) {
    queues.emplace_back(d, sycl::property_list{
                               sycl::property::queue::in_order(),
                               sycl::property::queue::enable_profiling() 
                           });
  }

  std::unique_ptr<spmv::IScheduler> scheduler;
  try {
      scheduler = spmv::make_scheduler(sched_name);
  } catch (const std::runtime_error& e) {
      std::cerr << "Failed to create scheduler: " << e.what() << std::endl;
      return 1;
  }
  std::cout << "Using scheduler: " << scheduler->name() << std::endl;

  std::vector<DeviceMem> mems(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) {
    auto &q = queues[i]; 
    mems[i].rp = sycl::malloc_device<int  >(A.row_ptr.size(), q);
    mems[i].ci = sycl::malloc_device<int  >(A.col_idx.size(), q);
    mems[i].v  = sycl::malloc_device<float>(A.vals.size(),    q); 
    mems[i].x  = sycl::malloc_device<float>(N,                q); 
    mems[i].y  = sycl::malloc_device<float>(N,                q); 

    q.memcpy(mems[i].rp, A.row_ptr.data(), A.row_ptr.size()*sizeof(int));
    q.memcpy(mems[i].ci, A.col_idx.data(), A.col_idx.size()*sizeof(int));
    q.memcpy(mems[i].v,  A.vals.data(),    A.vals.size()*sizeof(float));
  }
  for (auto &q : queues) q.wait();

  std::vector<float> x(N, 0.0f), r(N), r_hat(N), p_vec(N), v_vec(N), s_vec(N), t_vec(N);
  std::vector<float> b(N);
  std::fill(b.begin(), b.end(), 1.0f); 

  std::vector<float> p_hat_vec(N), s_hat_vec(N); 

  r = b;

  r_hat = r; 
  float rho_prev = 1.0f, alpha = 1.0f, omega = 1.0f; 
  std::fill(p_vec.begin(), p_vec.end(), 0.0f);
  std::fill(v_vec.begin(), v_vec.end(), 0.0f);


  int iter = 0;
  double total_spmv_time_ms = 0.0;
  double total_feedback_update_time_ms = 0.0;
  double total_precond_time_ms = 0.0;


  std::cout << std::fixed << std::setprecision(6); 

  float resid_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0f));
  float initial_resid_norm = resid_norm; 
  if (initial_resid_norm == 0.0f) initial_resid_norm = 1.0f; 

  std::cout << "Iter 0, abs_residual_norm = " << resid_norm 
            << ", rel_residual_norm = " << resid_norm / initial_resid_norm << "\n";

  float rho_curr = std::inner_product(r_hat.begin(), r_hat.end(), r.begin(), 0.0f);


  while (resid_norm > tol && (resid_norm / initial_resid_norm) > tol && iter < maxiter) { 
    iter++;
    
    if (std::abs(rho_prev) < tiny_threshold * 100) { 
        std::cerr << "Error: rho_prev (rho_{i-1}) is close to zero at iter " << iter <<", breakdown in BiCGSTAB." << std::endl;
        break;
    }
    
    float beta = (rho_curr / rho_prev) * (alpha / omega);

    for (int i = 0; i < N; ++i) {
      p_vec[i] = r[i] + beta * (p_vec[i] - omega * v_vec[i]);
    }

    auto precond_start_time_p = high_resolution_clock::now();
    apply_jacobi_preconditioner_host(diag_A_inv, p_vec, p_hat_vec);
    auto precond_end_time_p = high_resolution_clock::now();
    total_precond_time_ms += duration<double, std::milli>(precond_end_time_p - precond_start_time_p).count();
    
    auto spmv_start_time_v = high_resolution_clock::now();
    std::vector<double> kernel_times_v = spmvAll(scheduler, queues, A, mems, p_hat_vec, v_vec);
    auto spmv_end_time_v = high_resolution_clock::now();
    total_spmv_time_ms += duration<double, std::milli>(spmv_end_time_v - spmv_start_time_v).count();

    if (!scheduler->is_dynamic() && (scheduler->name() == std::string("feedback") || scheduler->name() == std::string("bandit")) ) { 
        auto feedback_start_time = high_resolution_clock::now();
        scheduler->update_times(kernel_times_v);
        auto feedback_end_time = high_resolution_clock::now();
        total_feedback_update_time_ms += duration<double, std::milli>(feedback_end_time - feedback_start_time).count();
    }
    
    float r_hat_dot_v = std::inner_product(r_hat.begin(), r_hat.end(), v_vec.begin(), 0.0f);
    if (std::abs(r_hat_dot_v) < tiny_threshold * 100) { 
        std::cerr << "Error: r_hat_dot_v (denominator for alpha) is close to zero at iter " << iter << ", breakdown in BiCGSTAB." << std::endl;
        break;
    }
    alpha = rho_curr / r_hat_dot_v;

    for (int i = 0; i < N; ++i) {
      s_vec[i] = r[i] - alpha * v_vec[i];
    }
    
    float s_norm = std::sqrt(std::inner_product(s_vec.begin(), s_vec.end(), s_vec.begin(), 0.0f));
    if (s_norm < tol || (s_norm / initial_resid_norm) < tol) {
        for (int i = 0; i < N; ++i) x[i] += alpha * p_hat_vec[i]; 
        resid_norm = s_norm;
        std::cout << "Iter " << iter << ", abs_residual_norm = " << resid_norm 
                  << ", rel_residual_norm = " << resid_norm / initial_resid_norm 
                  << " (Converged on s_vec)" << "\n";
        break; 
    }

    auto precond_start_time_s = high_resolution_clock::now();
    apply_jacobi_preconditioner_host(diag_A_inv, s_vec, s_hat_vec);
    auto precond_end_time_s = high_resolution_clock::now();
    total_precond_time_ms += duration<double, std::milli>(precond_end_time_s - precond_start_time_s).count();

    auto spmv_start_time_t = high_resolution_clock::now();
    std::vector<double> kernel_times_t = spmvAll(scheduler, queues, A, mems, s_hat_vec, t_vec);
    auto spmv_end_time_t = high_resolution_clock::now();
    total_spmv_time_ms += duration<double, std::milli>(spmv_end_time_t - spmv_start_time_t).count();

    if (!scheduler->is_dynamic() && (scheduler->name() == std::string("feedback") || scheduler->name() == std::string("bandit")) ) { 
        auto feedback_start_time = high_resolution_clock::now();
        scheduler->update_times(kernel_times_t);
        auto feedback_end_time = high_resolution_clock::now();
        total_feedback_update_time_ms += duration<double, std::milli>(feedback_end_time - feedback_start_time).count();
    }

    float t_vec_dot_s_vec = std::inner_product(t_vec.begin(), t_vec.end(), s_vec.begin(), 0.0f); 
    float t_vec_dot_t_vec = std::inner_product(t_vec.begin(), t_vec.end(), t_vec.begin(), 0.0f);

    if (std::abs(t_vec_dot_t_vec) < tiny_threshold * 100) { 
        std::cerr << "Warning: t_dot_t (denominator for omega) is close to zero at iter " << iter << ". Setting omega = 0." << std::endl;
        omega = 0.0f; 
    } else {
        omega = t_vec_dot_s_vec / t_vec_dot_t_vec;
    }

    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p_hat_vec[i] + omega * s_hat_vec[i]; 
    }

    for (int i = 0; i < N; ++i) {
      r[i] = s_vec[i] - omega * t_vec[i];
    }

    rho_prev = rho_curr; 
    rho_curr = std::inner_product(r_hat.begin(), r_hat.end(), r.begin(), 0.0f); 

    resid_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0f));
    std::cout << "Iter " << iter << ", abs_residual_norm = " << resid_norm 
              << ", rel_residual_norm = " << resid_norm / initial_resid_norm 
              << "\n";
     
     if (std::abs(omega) < tiny_threshold * 100 && std::abs(t_vec_dot_t_vec) >= tiny_threshold*100 ) { 
        std::cerr << "Warning: omega is close to zero at iter " << iter << "." << std::endl;
    }
  }

  std::cout << "\nBiCGSTAB finished in " << iter << " iterations." << std::endl;
  std::cout << "Final absolute residual norm: " << resid_norm << std::endl;
  std::cout << "Final relative residual norm: " << resid_norm / initial_resid_norm << std::endl;
  std::cout << "Total time spent in SpMV calls (wall time, approx): " << total_spmv_time_ms << " ms" << std::endl;
  std::cout << "Total time spent in Jacobi preconditioner application (host): " << total_precond_time_ms << " ms" << std::endl;

  if (!scheduler->is_dynamic() && (scheduler->name() == std::string("feedback") || scheduler->name() == std::string("bandit"))) {
    std::cout << "Total time spent in scheduler update_times calls: " << total_feedback_update_time_ms << " ms" << std::endl;
  }

  std::ofstream out_file(out_x);
  if (!out_file) {
      std::cerr << "Error: Could not open output file " << out_x << std::endl;
      for (size_t i = 0; i < devs.size(); ++i) {
        sycl::free(mems[i].rp, queues[i]);
        sycl::free(mems[i].ci, queues[i]);
        sycl::free(mems[i].v,  queues[i]);
        sycl::free(mems[i].x,  queues[i]);
        sycl::free(mems[i].y,  queues[i]);
      }
      return 1;
  }
  out_file << "%%MatrixMarket matrix array real general\n";
  out_file << N << " " << 1 << "\n";
  out_file << std::fixed << std::setprecision(17); 
  for (int i = 0; i < N; ++i) {
    out_file << x[i] << "\n";
  }
  out_file.close();
  std::cout << "Solution vector x written to " << out_x << std::endl;

  for (size_t i = 0; i < devs.size(); ++i) {
    sycl::free(mems[i].rp, queues[i]);
    sycl::free(mems[i].ci, queues[i]);
    sycl::free(mems[i].v,  queues[i]);
    sycl::free(mems[i].x,  queues[i]);
    sycl::free(mems[i].y,  queues[i]);
  }
  std::cout << "Device memory freed." << std::endl;

  return 0;
}