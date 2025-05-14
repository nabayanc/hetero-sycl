#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="build"
MAIN_EXE_NAME="exp03_multithreaded_dispatch_main" # Changed executable name
EXE_PATH="${BUILD_DIR}/apps/exp03_multithreaded_dispatch/${MAIN_EXE_NAME}"

DATASETS_DIR="DATASETS"
EXPERIMENTS_ROOT_DIR="EXPERIMENTS"
EXPERIMENT_NAME="03_multithreaded_dispatch" # Changed experiment name for output dirs
RESULTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/results"
PLOTS_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/plots"

# Matrices for the experiment (can be the same as Exp01)
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

# Config types are the same, the C++ app handles the dispatch change internally
CONFIGS_TYPES=("cpu1" "gpu1" "gpu2_split" "cpu1gpu2_split") 

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# Plotting script configuration (reuses the Exp01 plotter)
PYTHON_EXE="${PYTHON_EXE:-python3}"
PLOT_SCRIPT_PATH="./plot_exp01_gantt.py" # Reusing the same plotter logic
ITERATION_TO_PLOT=9 
FONT_SCALE_PLOT=1.25

# --- Preparation ---
mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${PLOTS_DIR}"

if [[ ! -x "${EXE_PATH}" ]]; then
  echo "Error: Executable ${EXE_PATH} not found. Build the project first."
  echo "Attempting to build..."
  # Adjust build command if your main CMake builds all targets, or specify this one
  (cd "${BUILD_DIR}" && ninja "${MAIN_EXE_NAME}") || (cd .. && cmake --build "${BUILD_DIR}" --target "${MAIN_EXE_NAME}")
  if [[ ! -x "${EXE_PATH}" ]]; then
    echo "Error: Build failed or executable ${MAIN_EXE_NAME} still not found at ${EXE_PATH}."
    exit 1
  fi
fi
echo "Using executable: ${EXE_PATH}"

# --- Run Experiments ---
for matrix_file in "${MATRICES[@]}"; do
    if [[ ! -f "${matrix_file}" ]]; then
        echo "Warning: Matrix file ${matrix_file} not found. Skipping."
        continue
    fi
    matrix_basename=$(basename "${matrix_file}" .mtx)
    echo "Processing matrix: ${matrix_basename} for Exp03"

    for config_key in "${CONFIGS_TYPES[@]}"; do
        device_select_str="auto" 

        echo "  Config: ${config_key}"
        
        output_csv="${RESULTS_BASE_DIR}/results_${matrix_basename}_${config_key}.csv"

        if [[ -f "${output_csv}" ]]; then
            echo "    Result file ${output_csv} already exists. Skipping."
            continue
        fi

        "${EXE_PATH}" \
            "${matrix_file}" \
            "${output_csv}" \
            "${NUM_WARMUP_RUNS}" \
            "${NUM_TIMED_RUNS}" \
            "${config_key}" \
            "${device_select_str}"
        
        if [[ $? -eq 0 ]]; then
            echo "    Successfully completed. Results in ${output_csv}"
        else
            echo "    ERROR running experiment for ${matrix_basename} with config ${config_key}."
        fi
        echo "    ---------------------------------"
    done
    echo "---------------------------------------------------------------------"
done

echo "All Exp03 experiments completed."
echo "Results are in: ${RESULTS_BASE_DIR}"

# --- Plotting Phase ---
echo ""
echo "Starting Exp03 plotting phase..."
if [[ -f "${PLOT_SCRIPT_PATH}" ]]; then
    echo "Using plotting script: ${PLOT_SCRIPT_PATH}"
    "${PYTHON_EXE}" "${PLOT_SCRIPT_PATH}" \
        --results_dir "${RESULTS_BASE_DIR}" \
        --plots_dir "${PLOTS_DIR}" \
        --experiment_name "${EXPERIMENT_NAME}" \
        --iteration_to_plot "${ITERATION_TO_PLOT}" \
        --font_scale "${FONT_SCALE_PLOT}"
    echo "Plotting complete. Plots should be in ${PLOTS_DIR}"
else
    echo "Warning: Plotting script ${PLOT_SCRIPT_PATH} not found. Skipping plotting."
fi

echo ""
echo "Experiment script for Exp03 finished."