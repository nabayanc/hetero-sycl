#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="${BUILD_DIR:-build}" # Use existing BUILD_DIR or default to 'build'
SPMV_CLI_EXE="${BUILD_DIR}/apps/spmv_cli/spmv_cli"

DATASETS_DIR="DATASETS"
# Root directory for storing all experiment data (CSVs)
EXPERIMENTS_DATA_ROOT="EXPERIMENTS_SUITE"
# Root directory for storing all plots
EXPERIMENTS_PLOT_ROOT="PLOTS_SUITE"

# Matrices for the experiment suite
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

# Schedulers to test (must match registration names in C++ code)
# Ensure 'static_split' is registered if you modified it for 45-45-10
# SCHEDULERS=(
#     "static_block"
#     "static_split" # This will use the 45-45-10 if static_split.cpp is modified as such
#     "chunked_rr"
#     "locality_block"
#     "feedback"
#     "workstealing"
#     "dynamic"
#     "bandit"
# )

SCHEDULERS=("static_split")

NUM_WARMUP_RUNS=3 # Reduced for quicker testing, adjust as needed
NUM_TIMED_RUNS=5  # Reduced for quicker testing

# --- Device Selection ---
# IMPORTANT: Set this environment variable to select devices if specific ordering or subset is needed.
# For 'static_split' (45-45-10 for GPU0, GPU1, CPU0), you would need:
# export SYCL_DEVICE_FILTER="level_zero:gpu:0,level_zero:gpu:1,opencl:cpu:0" (Example)
# For other schedulers, spmv_cli will use all detected GPUs then all CPUs by default.
# If a scheduler (like static_split) is designed for a fixed number of devices (e.g., 3),
# ensure the SYCL_DEVICE_FILTER provides exactly that many, or spmv_cli's device usage
# combined with the scheduler's make_plan needs to be robust.
echo "INFO: spmv_cli by default uses all detected GPUs then all CPUs. Specific schedulers might behave best with specific device counts/types controlled by SYCL_DEVICE_FILTER."

# --- Preparation ---
if [[ ! -x "${SPMV_CLI_EXE}" ]]; then
  echo "Error: Executable ${SPMV_CLI_EXE} not found. Build the project first."
  exit 1
fi
echo "Using executable: ${SPMV_CLI_EXE}"
echo "Warmup runs: ${NUM_WARMUP_RUNS}, Timed runs: ${NUM_TIMED_RUNS}"

# --- Run Experiments ---
for scheduler_name in "${SCHEDULERS[@]}"; do
    echo "====================================================================="
    echo "Starting experiments for SCHEDULER: ${scheduler_name}"
    echo "====================================================================="

    current_scheduler_results_dir="${EXPERIMENTS_DATA_ROOT}/${scheduler_name}/results"
    current_scheduler_plots_dir="${EXPERIMENTS_PLOT_ROOT}/${scheduler_name}"

    mkdir -p "${current_scheduler_results_dir}"
    mkdir -p "${current_scheduler_plots_dir}"

    for matrix_file in "${MATRICES[@]}"; do
        if [[ ! -f "${matrix_file}" ]]; then
            echo "Warning: Matrix file ${matrix_file} not found. Skipping."
            continue
        fi
        matrix_basename=$(basename "${matrix_file}" .mtx)
        echo "  Processing matrix: ${matrix_basename} with scheduler: ${scheduler_name}"

        summary_csv="${current_scheduler_results_dir}/summary_${matrix_basename}.csv"
        devices_csv="${current_scheduler_results_dir}/devices_${matrix_basename}.csv"
        
        if [[ -f "${summary_csv}" ]]; then
            echo "    Result file ${summary_csv} already exists. Skipping."
            continue
        fi

        # Set SYCL_DEVICE_FILTER here if specific to this scheduler_name and matrix combination
        # e.g., if [[ "${scheduler_name}" == "static_split" ]]; then
        #          export SYCL_DEVICE_FILTER="level_zero:gpu:0,level_zero:gpu:1,opencl:cpu:0"
        #      else
        #          unset SYCL_DEVICE_FILTER # Or set to a default like "all" if supported
        #      fi
        
        # Ensure spmv_cli is called with the correct arguments
        "${SPMV_CLI_EXE}" \
            "${matrix_file}" \
            "${summary_csv}" \
            "${devices_csv}" \
            "${NUM_TIMED_RUNS}" \
            "${scheduler_name}" \
            --warmup_iterations "${NUM_WARMUP_RUNS}"
        
        if [[ $? -eq 0 ]]; then
            echo "    Successfully completed. Results in ${summary_csv}"
        else
            echo "    ERROR running spmv_cli for ${matrix_basename} with scheduler ${scheduler_name}."
        fi
        echo "    ---------------------------------"
    done

    # After processing all matrices for the current scheduler, call the plotting script
    PLOT_SCRIPT_PATH="./plot_scheduler_gantt.py" # Assuming it's in the root
    PYTHON_EXE="${PYTHON_EXE:-python3}"

    if [[ -f "${PLOT_SCRIPT_PATH}" ]]; then
        echo "  Generating plots for scheduler: ${scheduler_name}..."
        "${PYTHON_EXE}" "${PLOT_SCRIPT_PATH}" \
            --scheduler_results_dir "${current_scheduler_results_dir}" \
            --scheduler_plots_dir "${current_scheduler_plots_dir}" \
            --scheduler_name "${scheduler_name}"
        echo "  Plots for ${scheduler_name} should be in ${current_scheduler_plots_dir}"
    else
        echo "  Warning: Plotting script ${PLOT_SCRIPT_PATH} not found. Skipping plotting for ${scheduler_name}."
    fi
    echo ""
done

echo "====================================================================="
echo "All experiment suite runs completed."
echo "Results are in subdirectories under: ${EXPERIMENTS_DATA_ROOT}"
echo "Plots are in subdirectories under: ${EXPERIMENTS_PLOT_ROOT}"
echo "====================================================================="