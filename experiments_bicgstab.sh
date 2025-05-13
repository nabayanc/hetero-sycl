#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────

BUILD_DIR="build" # Assuming your build directory is named 'build' at the root of the project

# --- Executables ---
CREATECSR_EXE="${BUILD_DIR}/createcsr"
BICGSTAB_CLI_EXE="${BUILD_DIR}/apps/spmv_bicgstab/bicgstab" # Verify this executable name

# --- Directories ---
# Base directory for all BiCGSTAB related benchmarks
BICGSTAB_BENCHMARK_BASE_DIR="benchmarks_bicgstab"

# Directory for diagonally dominant datasets
DIAG_DOM_DATA_DIR="${BICGSTAB_BENCHMARK_BASE_DIR}/datasets_diag_dominant"

# Base directory for BiCGSTAB results (logs, solutions)
RESULT_DIR="${BICGSTAB_BENCHMARK_BASE_DIR}/results_pc_diag"

# --- BiCGSTAB Parameters ---
TOLERANCE="1e-6"
MAX_ITERATIONS_BICGSTAB="1000" # Max iterations for BiCGSTAB solver

# --- Experiment Parameters ---
# For diagonally dominant matrices, rows must equal cols.
# These sizes will be used for square matrices (NxN).
SIZES=(1000 5000 10000) # Reduced for quicker testing, adjust as needed
DENSITIES=(0.01 0.05)  # Density for off-diagonal elements.
# SIZES=(100 500 1000 5000 10000) # Example sizes
# DENSITIES=(0.001 0.01 0.05 0.1) # Example densities

SCHEDULERS=(static static_block chunked_rr locality_block feedback workstealing dynamic bandit)


# ─── PREP ─────────────────────────────────────────────────────────────────

# Check if executables exist
if [[ ! -x "${CREATECSR_EXE}" ]]; then
  echo >&2 "Error: ${CREATECSR_EXE} not found or not executable."
  echo >&2 "Please build the project first (e.g., using make or ninja)."
  exit 1
fi

if [[ ! -x "${BICGSTAB_CLI_EXE}" ]]; then
  echo >&2 "Error: ${BICGSTAB_CLI_EXE} not found or not executable."
  echo >&2 "Please build the project first (e.g., using make or ninja)."
  exit 1
fi

# Create necessary directories
mkdir -p "${DIAG_DOM_DATA_DIR}"
mkdir -p "${RESULT_DIR}" # This will be the parent for scheduler-specific result directories

echo "Starting BiCGSTAB experiments with DIAGONALLY DOMINANT matrices..."
echo "Build directory: ${BUILD_DIR}"
echo "CreateCSR executable: ${CREATECSR_EXE}"
echo "BiCGSTAB CLI executable: ${BICGSTAB_CLI_EXE}"
echo "Diagonally Dominant Data directory: ${DIAG_DOM_DATA_DIR}"
echo "Results base directory: ${RESULT_DIR}"
echo "Tolerance: ${TOLERANCE}, Max Iterations: ${MAX_ITERATIONS_BICGSTAB}"
echo "Matrix Sizes (NxN): ${SIZES[*]}"
echo "Densities (for off-diagonals): ${DENSITIES[*]}"
echo "Schedulers: ${SCHEDULERS[*]}"
echo "---------------------------------------------------------------------"


# ─── EXPERIMENTS ────────────────────────────────────────────────────────────

for SIZE in "${SIZES[@]}"; do
  for DENSITY in "${DENSITIES[@]}"; do
    # Create a string representation of density for filenames (e.g., 0.01 -> 0_01)
    DSTR=$(printf "%s" "$DENSITY" | sed 's/\./_/')
    # Filename indicates it's diagonally dominant
    MTX_FILENAME="csr${SIZE}_diagdom_${DSTR}.mtx"
    MTX_FILEPATH="${DIAG_DOM_DATA_DIR}/${MTX_FILENAME}"

    # 1) Generate diagonally dominant matrix (must be square)
    # The createcsr tool with --diag_dominant will enforce rows == cols.
    # We set cols = SIZE here explicitly.
    if [[ ! -f "${MTX_FILEPATH}" ]]; then
      echo "==> Generating Diagonally Dominant Matrix: ${MTX_FILEPATH}"
      echo "    Size: ${SIZE}x${SIZE}, Off-diagonal Density: ${DENSITY}"
      "${CREATECSR_EXE}" \
        --rows    "${SIZE}" \
        --cols    "${SIZE}" \
        --density "${DENSITY}" \
        --output  "${MTX_FILEPATH}" \
        --diag_dominant # Use the flag to make it diagonally dominant
    else
      echo "==> Using existing Diagonally Dominant Matrix: ${MTX_FILEPATH}"
    fi

    # 2) Run bicgstab_cli for each scheduler using the diagonally dominant matrix
    for SCHED in "${SCHEDULERS[@]}"; do
      SCHED_OUTPUT_DIR="${RESULT_DIR}/${SCHED}"
      mkdir -p "${SCHED_OUTPUT_DIR}" # Create scheduler-specific subdirectory

      # Define output filenames for this specific run
      # Prefix solution and log files to indicate they used a diagonally dominant matrix
      SOLUTION_X_FILENAME="solution_${MTX_FILENAME}"
      LOG_FILENAME="log_${MTX_FILENAME%.mtx}_${SCHED}.txt"
      
      SOLUTION_X_FILEPATH="${SCHED_OUTPUT_DIR}/${SOLUTION_X_FILENAME}"
      LOG_FILEPATH="${SCHED_OUTPUT_DIR}/${LOG_FILENAME}"

      # Skip if log file already exists and is non-empty
      if [[ -s "${LOG_FILEPATH}" ]]; then
        echo "==> [${SCHED}] Already benchmarked for ${MTX_FILENAME}, skipping. Log: ${LOG_FILEPATH}"
        continue
      fi

      echo "==> [${SCHED}] Running BiCGSTAB for Diagonally Dominant ${MTX_FILENAME}"
      echo "    Matrix: ${MTX_FILEPATH}"
      echo "    Scheduler: ${SCHED}"
      echo "    Tolerance: ${TOLERANCE}, Max Iterations: ${MAX_ITERATIONS_BICGSTAB}"
      echo "    Solution output: ${SOLUTION_X_FILEPATH}"
      echo "    Log output: ${LOG_FILEPATH}"

      # Run the bicgstab_cli command
      # Redirect stdout and stderr to the log file
      if "${BICGSTAB_CLI_EXE}" \
          "${MTX_FILEPATH}" \
          "${SCHED}" \
          "${TOLERANCE}" \
          "${MAX_ITERATIONS_BICGSTAB}" \
          "${SOLUTION_X_FILEPATH}" \
          > "${LOG_FILEPATH}" 2>&1; then
        echo "    SUCCESS: BiCGSTAB completed for [${SCHED}] on ${MTX_FILENAME}."
      else
        echo "    ERROR: BiCGSTAB failed for [${SCHED}] on ${MTX_FILENAME}. Check log: ${LOG_FILEPATH}"
      fi
      echo "    -------------------------------------------------"
    done # End scheduler loop
  done # End density loop
done # End size loop

echo "====================================================================="
echo "All BiCGSTAB experiments with diagonally dominant matrices completed."
echo "Datasets are in: ${DIAG_DOM_DATA_DIR}"
echo "Results are in subdirectories under: ${RESULT_DIR}"
echo "====================================================================="
