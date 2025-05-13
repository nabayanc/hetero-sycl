import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re # For parsing column names

# --- Configuration ---
EXPERIMENT_ACRONYM = "exp01" # Used for subdirectories if needed, matching C++ app structure
EXPERIMENT_NAME = "01_baseline_spmv"
RESULTS_DIR = f"EXPERIMENTS/{EXPERIMENT_NAME}/results/"
PLOTS_BASE_DIR = "PLOTS"
PLOTS_DIR = f"{PLOTS_BASE_DIR}/{EXPERIMENT_NAME}/"

# Ensure the plot directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_work_splits(config_type, num_rows, num_devices_in_config):
    """
    Determines the number of rows processed by each device based on config_type.
    Returns a list of row counts for each device in the order they appear in CSV columns.
    """
    splits = []
    if num_devices_in_config == 0:
        return []

    if config_type == "cpu1" or config_type == "gpu1":
        if num_devices_in_config == 1:
            splits = [num_rows]
    elif config_type == "gpu2_split":
        if num_devices_in_config == 2:
            rows_per_device = num_rows // 2
            splits = [rows_per_device, num_rows - rows_per_device] # GPU0, GPU1
    elif config_type == "cpu1gpu2_split":
        if num_devices_in_config == 3:
            rows_per_device = num_rows // 3
            splits = [rows_per_device, rows_per_device, num_rows - 2 * rows_per_device] # CPU0, GPU0, GPU1
    
    if not splits or len(splits) != num_devices_in_config:
        # Fallback or error: if logic doesn't match, distribute somewhat evenly or error
        print(f"Warning: Could not determine exact work split for {config_type} with {num_devices_in_config} devices. Distributing rows evenly.")
        if num_devices_in_config > 0:
            base_rows = num_rows // num_devices_in_config
            remainder = num_rows % num_devices_in_config
            splits = [base_rows + (1 if i < remainder else 0) for i in range(num_devices_in_config)]
        else:
            return []
            
    return splits


def plot_gantt_chart(csv_filepath):
    """
    Generates and saves a Gantt chart from a single CSV results file.
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print(f"Warning: CSV file {csv_filepath} is empty. Skipping.")
            return
    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file {csv_filepath} is empty or malformed. Skipping.")
        return
    except Exception as e:
        print(f"Error reading CSV {csv_filepath}: {e}. Skipping.")
        return

    # Assuming one data row per CSV file for these baseline summaries
    if len(df) > 1:
        print(f"Warning: CSV file {csv_filepath} has multiple rows. Using the first row for plotting.")
    data = df.iloc[0]

    matrix_path = data.get('matrix_path', 'UnknownMatrix')
    config_type = data.get('config_type', 'UnknownConfig')
    num_rows_total = data.get('num_rows', 0)
    avg_overall_kernel_phase_wall_ms = data.get('avg_overall_kernel_phase_wall_ms', 0)

    # Dynamically find device kernel time columns and parse device info
    device_kernel_times = [] # List of (device_label, kernel_time_ms)
    
    # Regex to find columns like 'avg_kernel_event_ms_dev0_cpu' or 'avg_util_dev0_gpu_pct'
    # We want the kernel event times, not utilization for bar length
    kernel_time_pattern = re.compile(r"avg_kernel_event_ms_(dev\d+_(?:cpu|gpu))$")

    for col in df.columns:
        match = kernel_time_pattern.match(col)
        if match:
            device_label_suffix = match.group(1) # e.g., "dev0_cpu"
            kernel_time = data.get(col, 0)
            # Prepend "Device " and capitalize type for better legend
            parts = device_label_suffix.split('_') # devX, type
            device_display_label = f"Device {parts[0][3:]} ({parts[1].upper()})" # "Device 0 (CPU)"
            device_kernel_times.append({'label': device_display_label, 'time': kernel_time, 'id_suffix': device_label_suffix})

    if not device_kernel_times:
        print(f"No device kernel time columns found in {csv_filepath} (e.g., 'avg_kernel_event_ms_devX_type'). Skipping plot.")
        return

    # Sort devices for consistent plotting order (e.g., dev0, dev1, ...)
    device_kernel_times.sort(key=lambda x: x['id_suffix'])
    
    num_devices_in_config = len(device_kernel_times)
    work_splits_rows = get_work_splits(config_type, num_rows_total, num_devices_in_config)

    if len(work_splits_rows) != num_devices_in_config:
        print(f"Error: Mismatch between detected devices ({num_devices_in_config}) and work split calculation ({len(work_splits_rows)}) for {config_type}. Skipping plot for {csv_filepath}")
        return

    device_labels = [d['label'] for d in device_kernel_times]
    kernel_times_ms = [d['time'] for d in device_kernel_times]

    # --- Bar Heights (proportional to work split) ---
    # Scale heights to be visually reasonable, max height of 0.8 for a single bar
    # If multiple bars, they are stacked by default in y_pos
    if num_rows_total > 0 and all(s >= 0 for s in work_splits_rows):
        # Normalize work_splits_rows to sum to a factor that looks good (e.g., 0.8 * num_devices)
        # or make height proportional to its fraction of total rows, scaled by a factor
        max_work_split_for_height = max(work_splits_rows) if any(s > 0 for s in work_splits_rows) else 1
        bar_heights = [ (ws / max_work_split_for_height) * 0.7 if max_work_split_for_height > 0 else 0.1 for ws in work_splits_rows ]
    else:
        bar_heights = [0.7] * num_devices_in_config # Default height if no row data

    y_pos = range(len(device_labels))

    fig, ax = plt.subplots(figsize=(12, 2 + num_devices_in_config * 0.8)) # Adjust fig size based on num devices

    colors = plt.cm.get_cmap('viridis', num_devices_in_config) # Use a colormap

    for i in range(num_devices_in_config):
        ax.barh(y_pos[i], kernel_times_ms[i], height=bar_heights[i], left=0, 
                label=f"{device_labels[i]}: {kernel_times_ms[i]:.2f} ms\n({work_splits_rows[i]} rows)",
                color=colors(i / num_devices_in_config if num_devices_in_config > 1 else 0.5), 
                edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(device_labels)
    ax.invert_yaxis()  # Devices listed top-to-bottom

    ax.set_xlabel("Average Kernel Execution Time (ms) - SYCL Event Time")
    ax.set_ylabel("Device")
    
    matrix_basename = os.path.basename(matrix_path).replace('.mtx', '')
    title = (f"SpMV Kernel Runtimes: {config_type} on {matrix_basename}\n"
             f"Avg. Overall Kernel Phase Wall Time (Host): {avg_overall_kernel_phase_wall_ms:.2f} ms")
    ax.set_title(title, fontsize=12)

    # Set x-axis limit slightly beyond the max of overall phase or longest kernel
    max_time_to_plot = avg_overall_kernel_phase_wall_ms
    if kernel_times_ms:
         max_time_to_plot = max(max_time_to_plot, max(kernel_times_ms))
    ax.set_xlim(0, max_time_to_plot * 1.1 if max_time_to_plot > 0 else 1)


    ax.legend(title="Device Contributions", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend

    # --- Save Plot ---
    plot_filename = f"gantt_{matrix_basename}_{config_type}.png"
    full_plot_path = os.path.join(PLOTS_DIR, plot_filename)
    try:
        plt.savefig(full_plot_path)
        print(f"Saved plot: {full_plot_path}")
    except Exception as e:
        print(f"Error saving plot {full_plot_path}: {e}")
    plt.close(fig)


# --- Main Script Execution ---
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}. Ensure baseline experiments have run and produced output.")
    else:
        print(f"Found {len(csv_files)} CSV files to process in {RESULTS_DIR}.")

    for csv_file in sorted(csv_files): # Sort for consistent processing order
        print(f"\nProcessing {csv_file}...")
        plot_gantt_chart(csv_file)
    
    print("\nPython plotting script finished.")