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

using namespace std::chrono;
using spmv::spmv_cpu;
using spmv::spmv_gpu;

// Partition + launch A*x into y_host
static void spmvAll(const std::string &sched_name,
                    const std::vector<sycl::queue> &queues,
                    const spmv::CSR &A,
                    const std::vector<struct DeviceMem> &mems,
                    const std::vector<float> &x_host,
                    std::vector<float> &y_host) {
  // copy x → device
  for (size_t i = 0; i < queues.size(); ++i) {
    queues[i].memcpy(mems[i].x, x_host.data(), A.ncols * sizeof(float));
  }
  for (auto &q : queues) q.wait();

  // schedule & partition
  auto sched = spmv::make_scheduler(sched_name);
  auto parts = sched->make_plan(A.nrows, queues);

  // launch
  std::vector<sycl::event> evs;
  evs.reserve(parts.size());
  for (size_t i = 0; i < parts.size(); ++i) {
    auto &p = parts[i];
    int rows = p.row_end - p.row_begin;
    int* rp  = mems[i].rp + p.row_begin;
    float* y = mems[i].y  + p.row_begin;

    evs.push_back(
      p.q->get_device().is_gpu()
        ? spmv_gpu(*p.q, rows, rp,
                   mems[i].ci, mems[i].v, mems[i].x, y)
        : spmv_cpu(*p.q, rows, rp,
                   mems[i].ci, mems[i].v, mems[i].x, y)
    );
  }
  sycl::event::wait(evs);

  // copy y → host
  for (size_t i = 0; i < queues.size(); ++i) {
    auto &p = sched->make_plan(A.nrows, queues)[i];
    queues[i].memcpy(y_host.data() + p.row_begin,
                     mems[i].y + p.row_begin,
                     (p.row_end - p.row_begin) * sizeof(float));
  }
  for (auto &q : queues) q.wait();
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

  // 1) Load A
  auto A = spmv::CSR::load_mm(matrix_path);
  if (A.empty()) {
    std::cerr << "Failed to load CSR\n"; return 1;
  }
  int N = A.nrows;

  // 2) Discover devices & queues
  std::vector<sycl::device> devs;
  for (auto &p : sycl::platform::get_platforms())
    for (auto &d : p.get_devices())
      if (d.is_gpu() || d.is_cpu())
        devs.push_back(d);
  std::vector<sycl::queue> queues;
  for (auto &d : devs)
    queues.emplace_back(d, sycl::property::queue::in_order());

  // 3) Allocate CSR and x,y USM per device
  struct DeviceMem { int* rp; int* ci; float* v; float* x; float* y; };
  std::vector<DeviceMem> mems(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) {
    auto &q = queues[i];
    mems[i].rp = sycl::malloc_device<int  >(A.row_ptr.size(), q);
    mems[i].ci = sycl::malloc_device<int  >(A.col_idx.size(), q);
    mems[i].v  = sycl::malloc_device<float>(A.vals.size(),    q);
    mems[i].x  = sycl::malloc_device<float>(N,                q);
    mems[i].y  = sycl::malloc_device<float>(N,                q);

    // copy CSR once
    q.memcpy(mems[i].rp, A.row_ptr.data(), A.row_ptr.size()*sizeof(int));
    q.memcpy(mems[i].ci, A.col_idx.data(), A.col_idx.size()*sizeof(int));
    q.memcpy(mems[i].v,  A.vals.data(),    A.vals.size()*sizeof(float));
  }
  for (auto &q : queues) q.wait();

  // 4) Host vectors
  std::vector<float> x(N, 0.0f), r(N), r_hat(N), p(N), v(N), s(N), t(N), b(N, 1.0f);

  // r0 = b - A*x (x=0) → r = b
  r = b;
  r_hat = r;
  float rho_old = 1.0f, alpha = 1.0f, omega = 1.0f;

  // 5) Main BiCGSTAB loop
  int iter = 0;
  float resid = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0f));
  std::cout << "Iter 0, residual = " << resid << "\n";

  while (resid > tol && iter < maxiter) {
    ++iter;
    float rho_new = std::inner_product(r_hat.begin(), r_hat.end(), r.begin(), 0.0f);
    float beta    = (rho_new/rho_old) * (alpha/omega);

    // p = r + beta*(p - omega*v)
    for (int i = 0; i < N; ++i)
      p[i] = r[i] + beta*(p[i] - omega*v[i]);

    // v = A*p
    spmvAll(sched_name, queues, A, mems, p, v);

    alpha = rho_new / std::inner_product(r_hat.begin(), r_hat.end(), v.begin(), 0.0f);

    // s = r - alpha*v
    for (int i = 0; i < N; ++i)
      s[i] = r[i] - alpha*v[i];

    // t = A*s
    spmvAll(sched_name, queues, A, mems, s, t);

    float dot_ts = std::inner_product(t.begin(), t.end(), s.begin(), 0.0f);
    float dot_tt = std::inner_product(t.begin(), t.end(), t.begin(), 0.0f);
    omega = dot_ts / dot_tt;

    // x += alpha*p + omega*s
    for (int i = 0; i < N; ++i)
      x[i] += alpha*p[i] + omega*s[i];

    // r = s - omega*t
    for (int i = 0; i < N; ++i)
      r[i] = s[i] - omega*t[i];

    resid = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0f));
    std::cout << "Iter " << iter << ", residual = " << resid << "\n";

    rho_old = rho_new;
  }

  // 6) Output solution x in Matrix-Market COO as a vector
  std::ofstream out(out_x);
  out << "%%MatrixMarket matrix array real general\n";
  out << N << " " << 1 << "\n";
  for (int i = 0; i < N; ++i)
    out << x[i] << "\n";

  return 0;
}
