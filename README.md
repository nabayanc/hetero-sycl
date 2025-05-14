# SpMV-Hetero-SYCL

A high-performance implementation of sparse matrix-vector multiplication (SpMV) targeting heterogeneous systems with multiple CPUs and GPUs. The project uses SYCL to enable code portability across diverse hardware architectures, with a focus on efficient workload distribution through various scheduling strategies.

## Table of Contents

- [Overview](#overview)
- [Codebase Structure](#codebase-structure)
- [Build Requirements](#build-requirements)
- [Build Instructions](#build-instructions)
- [Running Experiments](#running-experiments)
- [Creating Custom Schedulers](#creating-custom-schedulers)
- [Setting Up Custom Experiments](#setting-up-custom-experiments)

## Overview

SpMV-Hetero-SYCL provides a framework for executing sparse matrix-vector operations (y = Ax) across heterogeneous computing resources using the SYCL programming model. The project includes:

- Optimized SpMV implementations for both CPU and GPU
- Multiple scheduling algorithms to distribute work across devices
- Comprehensive profiling and benchmarking tools
- Support for various sparse matrix formats (CSR)
- Performance visualization tools

This implementation focuses on exploring different work distribution strategies while minimizing data transfers between host and device memory.

## Codebase Structure

```
./
├── apps/                          # Application executables
│   ├── createcsr/                 # Tool to create CSR matrices
│   ├── exp01_baseline_spmv/       # Baseline SpMV experiments
│   ├── exp03_multithreaded_dispatch/ # Multi-threaded dispatch experiments
│   ├── spmv_bicgstab/             # BiCGSTAB solver using SpMV
│   └── spmv_cli/                  # CLI for SpMV with various schedulers
├── include/
│   └── spmv/                      # Header files
│       ├── csr.hpp                # CSR matrix format
│       ├── kernels.hpp            # SpMV kernel declarations
│       ├── scheduler_registry.hpp # Scheduler registration
│       └── scheduler.hpp          # Scheduler interface
├── src/
│   ├── schedulers/                # Scheduler implementations
│   │   ├── adaptive_load_balancing.cpp
│   │   ├── bandit_scheduler.cpp
│   │   ├── chunked_rr.cpp
│   │   ├── dynamic.cpp
│   │   ├── feedback.cpp
│   │   ├── locality_block.cpp
│   │   ├── static_block.cpp
│   │   ├── static_split.cpp
│   │   └── workstealing.cpp
│   ├── spmv_cpu.cpp               # CPU kernel implementation
│   ├── spmv_gpu.cpp               # GPU kernel implementation
│   └── spmv_csr.cpp               # CSR matrix implementation
├── tests/                         # Test code
│   ├── test_devices.cpp           # Lists available SYCL devices
├── *.sh                           # Various experiment scripts
├── *.py                           # Plotting and analysis scripts
├── CMakeLists.txt                 # Main CMake configuration
└── Makefile                       # Convenience Make wrapper
```

## Build Requirements

### Essential Dependencies

1. **DPC++ Compiler**: Intel's SYCL implementation with extensions for heterogeneous computing
   - See [Intel's oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) for the official distribution
   - You can also use a custom build of LLVM with SYCL support

2. **CMake**: Version 3.23 or higher
   - Used for build configuration

3. **Ninja**: Build system
   - Recommended for faster parallel builds

4. **Python 3**: For analysis and visualization scripts
   - Required packages: pandas, matplotlib, numpy

### Optional Dependencies

- **CUDA Toolkit**: For NVIDIA GPU support
   - Required if targeting NVIDIA GPUs through SYCL's CUDA backend

## Build Instructions

### 1. Set up the DPC++ Compiler

Ensure that your DPC++ compiler is properly installed and available in your PATH, or set the `DPCPP` environment variable to point to your compiler:

```bash
# Example if using Intel oneAPI DPC++
source /opt/intel/oneapi/setvars.sh

# Or set the DPCPP variable directly
export DPCPP=/path/to/your/dpcpp/compiler
```

### 2. Configure the Build

Using the provided Makefile (which calls CMake internally):

```bash
# Configure and build using the Makefile
make build
```

Or manually with CMake:

```bash
# Create and navigate to build directory
mkdir -p build && cd build

# Configure with CMake
cmake -GNinja .. -DCMAKE_CXX_COMPILER=$DPCPP

# Build all targets
ninja
```

### 3. Verify Build

To verify that the build completed successfully and to see available devices:

```bash
# Run the device test (from the build directory)
./tests/test_devices
```

This should output all available SYCL devices (CPUs, GPUs, etc.) on your system.

## Running Experiments

The project includes several scripts for running various experiments and benchmarks.

### 1. Generate Test Matrices

Before running experiments, you need test matrices:

```bash
# Generate test matrices of various sizes and densities
./generate_data.sh
```

This creates matrices in the `DATASETS` directory, which will be used by the experiment scripts.

### 2. Run Specific Experiments

Several experiment scripts are provided:

```bash
# Baseline SpMV experiment (fixed work distribution)
./run_exp01_baselines.sh

# Feedback scheduler experiment with multi-threading
./run_exp02.sh

# Multi-threaded dispatch experiment
./run_exp03.sh

# Experiment with various other schedulers
./run_exp04.sh

# Run BiCGSTAB solver experiments
./experiments_bicgstab.sh

# Run a comprehensive suite of experiments
./run_experiment_suite.sh
```

Each script will:
1. Create output directories for results and plots
2. Run the specified experiment(s) on the generated test matrices
3. Generate plots and analysis of the results

### 3. Controlling Device Selection

To control which devices are used in the experiments, set the `SYCL_DEVICE_FILTER` environment variable:

```bash
# Example: Use only NVIDIA GPUs and CPU
export SYCL_DEVICE_FILTER="cuda:gpu,opencl:cpu"

# Example: Use specific GPU devices
export SYCL_DEVICE_FILTER="level_zero:gpu:0,level_zero:gpu:1,opencl:cpu:0"
```

### 4. Viewing Results

After running experiments, results can be found in:
- CSV data: `EXPERIMENTS/[experiment_name]/results/`
- Plots: `EXPERIMENTS/[experiment_name]/plots/`

## Creating Custom Schedulers

To implement your own scheduler, follow these steps:

1. **Create a new C++ file** in `src/schedulers/` (e.g., `my_scheduler.cpp`)

2. **Implement your scheduler class** that inherits from `spmv::IScheduler`:

```cpp
// src/schedulers/my_scheduler.cpp
#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"

namespace spmv {

class MyScheduler : public IScheduler {
public:
  // Partition work across devices
  std::vector<Part> make_plan(int nrows, 
                             const std::vector<sycl::queue>& queues) override {
    // Your work distribution logic goes here
    std::vector<Part> parts;
    // ... fill parts vector ...
    return parts;
  }

  // Return scheduler name
  const char* name() const noexcept override {
    return "my_scheduler";
  }
  
  // Optional: Implement feedback mechanism
  void update_times(const std::vector<double>& times) override {
    // Process performance feedback from previous iteration
  }
};

// Register your scheduler under a name
REGISTER_SCHEDULER("my_scheduler", MyScheduler);

} // namespace spmv
```

3. **Add your scheduler file to the build system** by updating `src/CMakeLists.txt` if necessary.

4. **Rebuild the project**:

```bash
cd build && ninja
```

5. **Use your scheduler** with the existing CLI tool:

```bash
./build/apps/spmv_cli/spmv_cli <matrix.mtx> <summary.csv> <devices.csv> <timed_iterations> my_scheduler
```

## Setting Up Custom Experiments

To create a custom experiment script:

1. **Create a new shell script** (e.g., `run_my_experiment.sh`)

2. **Define experiment parameters**:

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="build"
SPMV_CLI_EXE="${BUILD_DIR}/apps/spmv_cli/spmv_cli"

DATASETS_DIR="DATASETS"
RESULTS_DIR="EXPERIMENTS/my_experiment/results"
PLOTS_DIR="EXPERIMENTS/my_experiment/plots"

# Matrices to test
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

# Schedulers to test
SCHEDULERS=("my_scheduler" "static_block" "workstealing")

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# Create output directories
mkdir -p "${RESULTS_DIR}" "${PLOTS_DIR}"
```

3. **Implement the experiment loop**:

```bash
# --- Run Experiments ---
for matrix_file in "${MATRICES[@]}"; do
    matrix_basename=$(basename "${matrix_file}" .mtx)
    
    for scheduler in "${SCHEDULERS[@]}"; do
        summary_csv="${RESULTS_DIR}/results_${matrix_basename}_${scheduler}.csv"
        devices_csv="${RESULTS_DIR}/devices_${matrix_basename}_${scheduler}.csv"
        
        if [[ -f "${summary_csv}" ]]; then
            echo "Results exist for ${matrix_basename} with ${scheduler}, skipping."
            continue
        fi
        
        echo "Testing ${matrix_basename} with ${scheduler}..."
        
        "${SPMV_CLI_EXE}" \
            "${matrix_file}" \
            "${summary_csv}" \
            "${devices_csv}" \
            "${NUM_TIMED_RUNS}" \
            "${scheduler}" \
            --warmup_iterations "${NUM_WARMUP_RUNS}"
    done
done
```

4. **Add plotting**:

```bash
# --- Plot Results ---
PYTHON_EXE="python3"
PLOT_SCRIPT="./plot_exp04_gantt.py"  # Choose appropriate plotting script

"${PYTHON_EXE}" "${PLOT_SCRIPT}" \
    --results_dir "${RESULTS_DIR}" \
    --plots_dir "${PLOTS_DIR}" \
    --experiment_name "My Experiment" \
    --iteration_to_plot 9 \
    --font_scale 1.2
```

5. **Make the script executable and run it**:

```bash
chmod +x run_my_experiment.sh
./run_my_experiment.sh
```

## Contact

For questions or feedback, please contact mailto:nabayanc@vt.edu.
