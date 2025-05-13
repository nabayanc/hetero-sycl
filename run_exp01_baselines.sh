#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BUILD_DIR="build" # Assuming your build directory
MAIN_EXE_NAME="exp01_baseline_spmv_main"
EXE_PATH="${BUILD_DIR}/apps/exp01_baseline_spmv/${MAIN_EXE_NAME}"

DATASETS_DIR="DATASETS"
EXPERIMENTS_ROOT_DIR="EXPERIMENTS"
EXPERIMENT_NAME="01_baseline_spmv"
RESULTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/results"

# Matrices for the experiment (ensure these exist in DATASETS_DIR with this naming)
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

# Experiment configurations and corresponding device selection string
# For device_indices_str: "c0" = first CPU, "g0"=first GPU, "g1"=second GPU etc.
# This mapping needs to be handled by the C++ app's device selection logic.
# The C++ example provided uses config_type to infer devices, a more robust C++
# would parse a detailed device_indices_str.
# For simplicity, the C++ code currently maps config_type to device counts.
# If you want to select specific devices (e.g. GPU with specific ID), the C++ needs more complex device parsing.

CONFIGS_TYPES=("cpu1" "gpu1" "gpu2_split" "cpu1gpu2_split") 
# Device indices are now handled by the C++ based on config_type for this example
# If your C++ app takes a device_indices_str, you'd have:
# declare -A CONFIG_DEVICE_INDICES
# CONFIG_DEVICE_INDICES["cpu1"]="c0"
# CONFIG_DEVICE_INDICES["gpu1"]="g0"
# CONFIG_DEVICE_INDICES["gpu2_split"]="g0,g1" # Assumes C++ can parse this
# CONFIG_DEVICE_INDICES["cpu1gpu2_split"]="c0,g0,g1"

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# --- Preparation ---
mkdir -p "${RESULTS_BASE_DIR}"

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
        # device_select_str="${CONFIG_DEVICE_INDICES[${config_key}]}" # Use if C++ takes detailed device string
        device_select_str="auto" # C++ will infer from config_key for now

        echo "  Config: ${config_key}"
        
        output_csv="${RESULTS_BASE_DIR}/results_${matrix_basename}_${config_key}.csv"

        # Check if result already exists
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
            "${device_select_str}" # Pass device selection string
        
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