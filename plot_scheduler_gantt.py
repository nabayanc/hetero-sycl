#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import re
import argparse
import numpy as np

def generate_gantt_chart_for_iteration(
    summary_csv_path,
    devices_csv_path,
    plots_dir,
    scheduler_name_for_title,
    iteration_to_plot=0 # Default to the first timed iteration (index 0)
):
    try:
        summary_df = pd.read_csv(summary_csv_path)
        if summary_df.empty:
            print(f"  Skipping due to empty summary CSV: {summary_csv_path}")
            return
        summary_data = summary_df.iloc[0]
    except Exception as e:
        print(f"  Error reading summary CSV {summary_csv_path}: {e}. Skipping.")
        return

    try:
        devices_df = pd.read_csv(devices_csv_path)
        if devices_df.empty:
            print(f"  Skipping due to empty devices CSV: {devices_csv_path}")
            return
    except Exception as e:
        print(f"  Error reading devices CSV {devices_csv_path}: {e}. Skipping.")
        return

    # Filter devices_df for the selected iteration
    iter_df = devices_df[devices_df['iteration'] == iteration_to_plot]
    if iter_df.empty:
        print(f"  No data found for iteration {iteration_to_plot} in {devices_csv_path}. Skipping.")
        return

    matrix_path = summary_data.get('matrix_path', 'UnknownMatrix')
    matrix_total_rows = summary_data.get('num_rows', 0)
    avg_overall_kernel_phase_wall_ms = summary_data.get('avg_overall_kernel_phase_wall_ms', 0)
    
    # Get unique device indices from the selected iteration's data
    # And ensure labels are consistent from summary if possible
    
    device_labels_from_summary = []
    kernel_time_pattern = re.compile(r"avg_kernel_event_ms_(dev\d+_(?:GPU|CPU|DEV))$")
    for col in summary_df.columns:
        match = kernel_time_pattern.match(col)
        if match:
            device_labels_from_summary.append(match.group(1))
    device_labels_from_summary.sort() # Ensure consistent order like dev0_CPU, dev0_GPU ...
    
    # Map dev_idx from devices.csv to the sorted labels from summary.csv
    # This assumes dev_idx in devices.csv corresponds to the order in device_labels_from_summary
    # (e.g. dev_idx 0 -> dev0_CPU/GPU, dev_idx 1 -> dev1_CPU/GPU)
    # This mapping needs to be robust. For now, assume dev_idx maps to sorted order.
    
    unique_dev_indices = sorted(iter_df['dev_idx'].unique())
    device_labels_map = {}
    if len(unique_dev_indices) <= len(device_labels_from_summary):
        for i, dev_idx in enumerate(unique_dev_indices):
            device_labels_map[dev_idx] = device_labels_from_summary[i]
    else: # Fallback if mismatch
        for dev_idx in unique_dev_indices:
            # Attempt to find a matching device type (this is a heuristic)
            # This part needs spmv_cli to output device type along with dev_idx in devices.csv
            # For now, let's just use dev_idx if labels are missing
            device_labels_map[dev_idx] = f"Device {dev_idx}"


    # Calculate work per device for the selected iteration for bar thickness
    work_per_device = {} # dev_idx -> rows
    for dev_idx in unique_dev_indices:
        work_per_device[dev_idx] = iter_df[iter_df['dev_idx'] == dev_idx].apply(
            lambda row: row['row_end'] - row['row_begin'], axis=1
        ).sum()

    if matrix_total_rows == 0: # Avoid division by zero if matrix has no rows
        print(f"  Matrix {matrix_path} has 0 rows. Bar heights will be default.")
        work_fractions = {dev_idx: 0.1 for dev_idx in unique_dev_indices} # Default small height
    else:
        work_fractions = {
            dev_idx: (work / matrix_total_rows if matrix_total_rows > 0 else 0)
            for dev_idx, work in work_per_device.items()
        }

    # --- Plotting ---
    num_devices_in_plot = len(unique_dev_indices)
    fig, ax = plt.subplots(figsize=(14, max(5, 2 + num_devices_in_plot * 1.0))) # Wider for timeline

    # Define colors for devices
    # Use a standard colormap, or define your own
    # cmap = plt.cm.get_cmap('viridis', num_devices_in_plot if num_devices_in_plot > 0 else 1)
    # Use tab10 for more distinct colors if num_devices is small
    colors = plt.cm.get_cmap('tab10', max(10, num_devices_in_plot))


    # Determine y-positions for device lanes
    y_ticks_pos = []
    y_tick_labels = []
    
    # Max bar height (e.g., 0.8 of the lane space)
    # Min bar height (e.g., 0.1 for devices with very little work)
    MAX_TOTAL_THICKNESS_UNITS = 0.8 
    MIN_THICKNESS_UNIT_PER_DEVICE = 0.1
    
    current_y = 0
    for i, dev_idx in enumerate(unique_dev_indices):
        # Calculate bar height based on work fraction
        # Height should be a fraction of the available space for one device lane
        thickness_fraction = work_fractions.get(dev_idx, 0)
        bar_render_height = MIN_THICKNESS_UNIT_PER_DEVICE + thickness_fraction * (MAX_TOTAL_THICKNESS_UNITS - MIN_THICKNESS_UNIT_PER_DEVICE)
        bar_render_height = max(MIN_THICKNESS_UNIT_PER_DEVICE, min(bar_render_height, MAX_TOTAL_THICKNESS_UNITS))


        # Plot all chunks for this device in its lane
        device_chunks = iter_df[iter_df['dev_idx'] == dev_idx]
        for _, row_chunk in device_chunks.iterrows():
            start_time = row_chunk['launch_ms_rel_submission_start']
            duration = row_chunk['kernel_ms']
            ax.barh(
                current_y,          # Y-position of the bar's center
                duration,           # Width of the bar
                height=bar_render_height, # Calculated thickness
                left=start_time,    # X-position of the left edge
                color=colors(i % colors.N), # Cycle through colormap
                edgecolor='black',
                alpha=0.7
            )
        
        y_ticks_pos.append(current_y)
        y_tick_labels.append(device_labels_map.get(dev_idx, f"Device {dev_idx}"))
        current_y += 1 # Move to the next lane

    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis() # Devices listed top-to-bottom

    ax.set_xlabel(f"Time within Kernel Phase (ms) - Iteration {iteration_to_plot}")
    ax.set_ylabel("Device")

    matrix_basename = os.path.basename(matrix_path).replace('.mtx', '')
    title = (
        f"SpMV Gantt Chart: {scheduler_name_for_title} on {matrix_basename} (Iteration {iteration_to_plot})\n"
        f"Host Avg. Overall Kernel Phase: {avg_overall_kernel_phase_wall_ms:.2f} ms (from summary)"
    )
    ax.set_title(title, fontsize=11)

    # Determine x-axis limit from the max end time of any chunk in this iteration,
    # or from the host's average overall kernel phase time for context.
    max_chunk_end_time = 0
    if not iter_df.empty:
        max_chunk_end_time = (iter_df['launch_ms_rel_submission_start'] + iter_df['kernel_ms']).max()
    
    plot_xlim = max(max_chunk_end_time, avg_overall_kernel_phase_wall_ms) * 1.05
    if plot_xlim == 0: plot_xlim = 1 # Avoid xlim error if all times are zero
    ax.set_xlim(0, plot_xlim)

    # Add a vertical line for avg_overall_kernel_phase_wall_ms for context
    if avg_overall_kernel_phase_wall_ms > 0:
        ax.axvline(avg_overall_kernel_phase_wall_ms, color='gray', linestyle='--', linewidth=1,
                   label=f'Host Avg. Kernel Phase End ({avg_overall_kernel_phase_wall_ms:.2f} ms)')
        ax.legend(fontsize='small', bbox_to_anchor=(1.01, 1), loc='upper left')


    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.82, 0.96]) # Adjust for legend and title

    plot_filename = f"gantt_iter{iteration_to_plot}_{matrix_basename}.png"
    full_plot_path = os.path.join(plots_dir, plot_filename)
    try:
        plt.savefig(full_plot_path, dpi=150)
        print(f"  Saved plot: {full_plot_path}")
    except Exception as e:
        print(f"  Error saving plot {full_plot_path}: {e}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gantt charts for SpMV scheduler performance from a single iteration.")
    parser.add_argument('--scheduler_results_dir', type=str, required=True,
                        help="Directory containing summary_*.csv and devices_*.csv files for a scheduler.")
    parser.add_argument('--scheduler_plots_dir', type=str, required=True,
                        help="Directory where plots for this scheduler will be saved.")
    parser.add_argument('--scheduler_name', type=str, required=True,
                        help="Name of the scheduler (for plot titles).")
    parser.add_argument('--iteration_to_plot', type=int, default=4,
                        help="Index of the timed iteration to plot (default: 0 for the first timed iteration).")
    args = parser.parse_args()

    if not os.path.isdir(args.scheduler_results_dir):
        print(f"Error: Results directory not found: {args.scheduler_results_dir}")
        exit(1)
        
    os.makedirs(args.scheduler_plots_dir, exist_ok=True)

    # Find matching summary and devices files
    summary_files = glob.glob(os.path.join(args.scheduler_results_dir, "summary_*.csv"))

    if not summary_files:
        print(f"No summary CSV files found in {args.scheduler_results_dir}.")
    else:
        print(f"Found {len(summary_files)} summary CSV files to process for scheduler '{args.scheduler_name}'.")

    for summary_csv_file in sorted(summary_files):
        # Construct corresponding devices.csv filename
        # summary_csr_r25000_c25000_d0_01.csv -> devices_csr_r25000_c25000_d0_01.csv
        devices_csv_file = summary_csv_file.replace("summary_", "devices_")
        
        if not os.path.exists(devices_csv_file):
            print(f"  Warning: Corresponding devices file not found for {summary_csv_file} (expected: {devices_csv_file}). Skipping plot.")
            continue
        
        print(f"Processing {summary_csv_file} and {devices_csv_file}...")
        generate_gantt_chart_for_iteration(
            summary_csv_file,
            devices_csv_file,
            args.scheduler_plots_dir,
            args.scheduler_name,
            args.iteration_to_plot
        )
    
    print(f"\nPlotting script finished for scheduler '{args.scheduler_name}'.")