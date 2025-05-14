#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="build" # Assuming your build directory
SPMV_CLI_EXE="${BUILD_DIR}/apps/spmv_cli/spmv_cli" # Path to the modified spmv_cli

DATASETS_DIR="DATASETS" # Directory containing your .mtx files
EXPERIMENTS_ROOT_DIR="EXPERIMENTS"
EXPERIMENT_NAME="02_feedback_detailed_profiling" # Used for subdirectory naming
RESULTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/results"
PLOTS_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/plots"

# Matrices for the experiment (adjust as needed)
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    # Add more matrices if desired
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

SCHEDULER_TO_TEST="feedback"
DISPATCH_MODES=("single" "multi")

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# Plotting script configuration
PYTHON_EXE="${PYTHON_EXE:-python3}" # Use system python3 or specify an alternative
PLOT_SCRIPT_PATH="./plot_exp02_gantt.py" # Script from exp01
ITERATION_TO_PLOT=0 # Which timed iteration index to plot (0 for the first)
FONT_SCALE_PLOT=1.2

# --- Preparation ---
mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${PLOTS_DIR}"

if [[ ! -x "${SPMV_CLI_EXE}" ]]; then
  echo "Error: SPMV_CLI Executable ${SPMV_CLI_EXE} not found. Build the project first."
  # Attempt to build spmv_cli specifically
  echo "Attempting to build spmv_cli..."
  (cd "${BUILD_DIR}" && ninja spmv_cli) || (cd .. && cmake --build "${BUILD_DIR}" --target spmv_cli)
  if [[ ! -x "${SPMV_CLI_EXE}" ]]; then
    echo "Error: Build failed or executable spmv_cli still not found at ${SPMV_CLI_EXE}."
    exit 1
  fi
fi
echo "Using SPMV_CLI executable: ${SPMV_CLI_EXE}"

if [[ ! -f "${PLOT_SCRIPT_PATH}" ]]; then
    echo "Error: Plotting script ${PLOT_SCRIPT_PATH} not found. Make sure it's in the current directory or adjust the path."
    exit 1
fi
echo "Using plotting script: ${PLOT_SCRIPT_PATH}"
echo "IMPORTANT: Ensure SYCL_DEVICE_FILTER is set if you need specific device ordering (e.g., 2 GPUs then 1 CPU)."
echo "Example: export SYCL_DEVICE_FILTER=\"level_zero:gpu:0,level_zero:gpu:1,opencl:cpu:0\""
echo ""

# --- Run Experiments ---
for matrix_file in "${MATRICES[@]}"; do
    if [[ ! -f "${matrix_file}" ]]; then
        echo "Warning: Matrix file ${matrix_file} not found. Skipping."
        continue
    fi
    matrix_basename=$(basename "${matrix_file}" .mtx)
    echo "====================================================================="
    echo "Processing matrix: ${matrix_basename} with scheduler: ${SCHEDULER_TO_TEST}"
    echo "====================================================================="

    for dispatch_mode in "${DISPATCH_MODES[@]}"; do
        echo "  Dispatch Mode: ${dispatch_mode}"

        summary_csv="${RESULTS_BASE_DIR}/results_${matrix_basename}_${SCHEDULER_TO_TEST}_${dispatch_mode}.csv"
        devices_csv="${RESULTS_BASE_DIR}/devices_${matrix_basename}_${SCHEDULER_TO_TEST}_${dispatch_mode}.csv"

        if [[ -f "${summary_csv}" ]]; then
            echo "    Result file ${summary_csv} already exists. Skipping."
        else
            echo "    Running SpMV CLI..."
            echo "      Matrix: ${matrix_file}"
            echo "      Summary CSV: ${summary_csv}"
            echo "      Devices CSV: ${devices_csv}"
            echo "      Timed Iterations: ${NUM_TIMED_RUNS}"
            echo "      Scheduler: ${SCHEDULER_TO_TEST}"
            echo "      Warmup Iterations: ${NUM_WARMUP_RUNS}"
            echo "      Dispatch Mode: ${dispatch_mode}"

            "${SPMV_CLI_EXE}" \
                "${matrix_file}" \
                "${summary_csv}" \
                "${devices_csv}" \
                "${NUM_TIMED_RUNS}" \
                "${SCHEDULER_TO_TEST}" \
                --warmup_iterations "${NUM_WARMUP_RUNS}" \
                --dispatch_mode "${dispatch_mode}"

            if [[ $? -eq 0 ]]; then
                echo "    Successfully completed. Results in ${summary_csv} and ${devices_csv}"
            else
                echo "    ERROR running spmv_cli for ${matrix_basename} with scheduler ${SCHEDULER_TO_TEST} and dispatch ${dispatch_mode}."
            fi
        fi
        echo "    ---------------------------------"
    done # End dispatch_mode loop
done # End matrix loop

echo "All SpMV CLI runs for Experiment 02 completed."
echo "Results are in: ${RESULTS_BASE_DIR}"

# --- Plotting Phase ---
echo ""
echo "Starting plotting phase for Experiment 02..."

# The plot_exp01_gantt.py script iterates through files in results_dir,
# so we just need to point it to the correct directory.
# It will find all "results_*.csv" files and their corresponding "devices_*.csv"
if [[ $(find "${RESULTS_BASE_DIR}" -name 'results_*.csv' -print -quit) ]]; then
    "${PYTHON_EXE}" "${PLOT_SCRIPT_PATH}" \
        --results_dir "${RESULTS_BASE_DIR}" \
        --plots_dir "${PLOTS_DIR}" \
        --experiment_name "${EXPERIMENT_NAME}" \
        --iteration_to_plot "${ITERATION_TO_PLOT}" \
        --font_scale "${FONT_SCALE_PLOT}"
    echo "Plotting complete. Plots should be in ${PLOTS_DIR}"
else
    echo "No result files found in ${RESULTS_BASE_DIR}. Skipping plotting."
fi

echo ""
echo "Experiment 02 script finished."