#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="build" # Assuming your build directory
MAIN_EXE_NAME="exp01_baseline_spmv_main"
EXE_PATH="${BUILD_DIR}/apps/exp01_baseline_spmv/${MAIN_EXE_NAME}"

DATASETS_DIR="DATASETS"
EXPERIMENTS_ROOT_DIR="EXPERIMENTS"
EXPERIMENT_NAME="01_baseline_spmv" # Used for subdirectory naming and titles
RESULTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/results"
PLOTS_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/plots" # Plots will go here

# Matrices for the experiment
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

CONFIGS_TYPES=("cpu1" "gpu1" "gpu2_split" "cpu1gpu2_split")

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# Plotting script configuration
PYTHON_EXE="${PYTHON_EXE:-python3}" # Use system python3 or specify an alternative
PLOT_SCRIPT_PATH="./plot_exp01_gantt.py" # Assuming it's in the root of the repo
ITERATION_TO_PLOT=9 # Which timed iteration index to plot for the Gantt chart (0 for the first)
FONT_SCALE_PLOT=1.2 # Font scaling for plots

# --- Preparation ---
mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${PLOTS_DIR}" # Create plots directory

if [[ ! -x "${EXE_PATH}" ]]; then
  echo "Error: Executable ${EXE_PATH} not found. Build the project first."
  echo "Attempting to build..."
  (cd "${BUILD_DIR}" && ninja "${MAIN_EXE_NAME}") || (cd .. && cmake --build "${BUILD_DIR}" --target "${MAIN_EXE_NAME}")
  if [[ ! -x "${EXE_PATH}" ]]; then
    echo "Error: Build failed or executable still not found."
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
    echo "Processing matrix: ${matrix_basename}"

    for config_key in "${CONFIGS_TYPES[@]}"; do
        device_select_str="auto" # C++ will infer from config_key for now

        echo "  Config: ${config_key}"
        
        # Summary CSV filename (C++ app writes this)
        output_csv="${RESULTS_BASE_DIR}/results_${matrix_basename}_${config_key}.csv"
        # Devices CSV filename (C++ app also writes this, using a derived name)
        # The C++ code constructs it as: devices_ + (basename of output_csv without "results_")
        # e.g., devices_matrix_config.csv

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

echo "All baseline experiments completed."
echo "Results are in: ${RESULTS_BASE_DIR}"

# --- Plotting Phase ---
echo ""
echo "Starting plotting phase..."
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
echo "Experiment script finished."