#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Assuming your build directory is at the root of the project and named 'build'
BUILD_DIR="${BUILD_DIR:-build}"
SPMV_CLI_EXE="${BUILD_DIR}/apps/spmv_cli/spmv_cli"

# Directory containing your generated .mtx files (from generate_data.sh)
DATASETS_DIR="DATASETS"

# Root directory for storing all experiment data (CSVs) for this suite
EXPERIMENTS_ROOT_DIR="EXPERIMENTS"
EXPERIMENT_NAME="04_all_other_schedulers" # Name for this experiment suite
# Base directories for results and plots - scheduler subdirectories will be created here
RESULTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/results"
PLOTS_BASE_DIR="${EXPERIMENTS_ROOT_DIR}/${EXPERIMENT_NAME}/plots"


# Matrices for the experiment (adjust paths/names if yours are different)
MATRICES=(
    "${DATASETS_DIR}/csr_r25000_c25000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r25000_c25000_d0_001.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_01.mtx"
    "${DATASETS_DIR}/csr_r75000_c75000_d0_001.mtx"
)

# Schedulers to test (excluding 'static_split')
# These names must match the strings used in REGISTER_SCHEDULER macro in src/schedulers/
SCHEDULERS=(
    # "static_block"
    # "chunked_rr"
    # "locality_block"
    # "feedback"
    # "workstealing"
    # "dynamic"
    # "bandit"
    "adaptive_lb"
)

# Dispatch modes to test for spmv_cli
DISPATCH_MODES=("single" "multi") # Note: 'dynamic' scheduler's behavior is not affected by this flag.

NUM_WARMUP_RUNS=5
NUM_TIMED_RUNS=10

# Plotting script configuration
PYTHON_EXE="${PYTHON_EXE:-python3}" # Use system python3 or specify an alternative
PLOT_SCRIPT_PATH="./plot_exp02_gantt.py" # Reusing the modified Exp02 plotter
ITERATION_TO_PLOT=0 # Which timed iteration index to plot (0 for the first, 9 for the last of 10 timed runs)
FONT_SCALE_PLOT=1.2

# --- Preparation ---
echo "Setting up directories for Experiment 04..."
# Create the top-level results and plots directories
mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${PLOTS_BASE_DIR}"
echo "Results base directory: ${RESULTS_BASE_DIR}"
echo "Plots base directory: ${PLOTS_BASE_DIR}"
echo ""

# Check if spmv_cli executable exists
if [[ ! -x "${SPMV_CLI_EXE}" ]]; then
  echo "Error: SPMV_CLI Executable ${SPMV_CLI_EXE} not found."
  echo "Please build the project first (e.g., by running cmake --build . --target spmv_cli in your build directory)."
  exit 1
fi
echo "Using SPMV_CLI executable: ${SPMV_CLI_EXE}"
echo "Warmup runs: ${NUM_WARMUP_RUNS}, Timed runs: ${NUM_TIMED_RUNS}"
echo ""

# Verify existence of matrices
echo "Verifying existence of matrix files:"
for matrix_file in "${MATRICES[@]}"; do
    if [[ ! -f "${matrix_file}" ]]; then
        echo "Error: Matrix file ${matrix_file} not found."
        echo "Please generate matrices using generate_data.sh first."
        exit 1
    fi
    echo "  Found: ${matrix_file}"
done
echo ""

# --- Run Experiments ---
echo "Starting Experiment 04 runs..."

for scheduler_name in "${SCHEDULERS[@]}"; do
    echo "====================================================================="
    echo "Running experiments for SCHEDULER: ${scheduler_name}"
    echo "====================================================================="

    # Create scheduler-specific results and plots directories
    current_scheduler_results_dir="${RESULTS_BASE_DIR}/${scheduler_name}"
    current_scheduler_plots_dir="${PLOTS_BASE_DIR}/${scheduler_name}"
    mkdir -p "${current_scheduler_results_dir}"
    mkdir -p "${current_scheduler_plots_dir}"
    echo "  Scheduler results directory: ${current_scheduler_results_dir}"
    echo "  Scheduler plots directory: ${current_scheduler_plots_dir}"
    echo "  ---"

    for matrix_file in "${MATRICES[@]}"; do
        matrix_basename=$(basename "${matrix_file}" .mtx)
        echo "  Processing matrix: ${matrix_basename}"

        for dispatch_mode in "${DISPATCH_MODES[@]}"; do
            # Note: dispatch_mode has no effect on the 'dynamic' scheduler.
             if [[ "${scheduler_name}" == "dynamic" ]]; then
                 # Only run/plot dynamic once per matrix since dispatch_mode doesn't matter
                 if [[ "${dispatch_mode}" == "multi" ]]; then
                    echo "    Skipping dispatch mode 'multi' for 'dynamic' scheduler (behavior is same as 'single')."
                    continue
                 fi
                 echo "    Dispatch Mode: ${dispatch_mode} (Note: 'dynamic' scheduler is always multi-threaded internally)"
             else
                 echo "    Dispatch Mode: ${dispatch_mode}"
             fi


            # Define output filenames within the scheduler's results directory
            summary_csv="${current_scheduler_results_dir}/results_${matrix_basename}_${scheduler_name}_${dispatch_mode}.csv"
            devices_csv="${current_scheduler_results_dir}/devices_${matrix_basename}_${scheduler_name}_${dispatch_mode}.csv"

            # Skip if result file already exists
            if [[ -f "${summary_csv}" ]]; then
                echo "      Result file ${summary_csv} already exists. Skipping run."
            else
                echo "      Running SpMV CLI..."
                echo "        Matrix: ${matrix_file}"
                echo "        Summary CSV: ${summary_csv}"
                echo "        Devices CSV: ${devices_csv}"
                echo "        Timed Iterations: ${NUM_TIMED_RUNS}"
                echo "        Scheduler: ${scheduler_name}"
                echo "        Warmup Iterations: ${NUM_WARMUP_RUNS}"
                echo "        Dispatch Mode: ${dispatch_mode}"

                # Execute spmv_cli
                "${SPMV_CLI_EXE}" \
                    "${matrix_file}" \
                    "${summary_csv}" \
                    "${devices_csv}" \
                    "${NUM_TIMED_RUNS}" \
                    "${scheduler_name}" \
                    --warmup_iterations "${NUM_WARMUP_RUNS}" \
                    --dispatch_mode "${dispatch_mode}"

                if [[ $? -eq 0 ]]; then
                    echo "      Successfully completed. Results in ${summary_csv} and ${devices_csv}"
                else
                    echo "      ERROR running spmv_cli for ${matrix_basename} with scheduler ${scheduler_name} and dispatch ${dispatch_mode}."
                    echo "      Check logs or console output for details."
                fi
            fi # End if [[ -f "${summary_csv}" ]]
            echo "      ---"
        done # End dispatch_mode loop
        echo "    ---------------------------------"
    done # End matrix loop
    echo "---------------------------------------------------------------------"

    # After running all matrices/dispatch_modes for the current scheduler, trigger plotting for THIS scheduler's results
    PLOT_SCRIPT_PATH="./plot_exp04_gantt.py" # Ensure this path is correct
    PYTHON_EXE="${PYTHON_EXE:-python3}"

    # Verify plotting script exists before calling
    if [[ ! -f "${PLOT_SCRIPT_PATH}" ]]; then
        echo "Error: Plotting script ${PLOT_SCRIPT_PATH} not found."
        echo "Ensure plot_exp04_gantt.py is in the current directory or adjust PLOT_SCRIPT_PATH."
        # Decide whether to exit here or continue with the next scheduler's runs
        # For now, let's print error and continue, allowing runs to complete even if plotting fails.
        continue
    fi

    echo "  Generating plots for scheduler: ${scheduler_name}..."
    # Point the plotting script to the current scheduler's results and plots directories
    if [[ $(find "${current_scheduler_results_dir}" -name 'results_*.csv' -print -quit) ]]; then
        "${PYTHON_EXE}" "${PLOT_SCRIPT_PATH}" \
            --results_dir "${current_scheduler_results_dir}" \
            --plots_dir "${current_scheduler_plots_dir}" \
            --experiment_name "${EXPERIMENT_NAME}_${scheduler_name}" \
            --iteration_to_plot "${ITERATION_TO_PLOT}" \
            --font_scale "${FONT_SCALE_PLOT}"
        echo "  Plots for ${scheduler_name} should be in ${current_scheduler_plots_dir}"
    else
        echo "  No result files found in ${current_scheduler_results_dir} to plot for scheduler ${scheduler_name}. Skipping plotting."
    fi
    echo "" # Add a newline after plotting for a scheduler
done # End scheduler_name loop

echo "====================================================================="
echo "All Experiment 04 runs completed."
echo "Results are in subdirectories under: ${RESULTS_BASE_DIR}"
echo "Plots are in subdirectories under: ${PLOTS_BASE_DIR}"
echo "====================================================================="