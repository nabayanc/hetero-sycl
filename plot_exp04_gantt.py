#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
import numpy as np

# Consistent Color Scheme (can be reused from plot_exp01_gantt.py)
PREDEFINED_DEVICE_COLORS = {
    "dev0_GPU": "#1f77b4", # Blue
    "dev1_GPU": "#ff7f0e", # Orange
    "dev2_GPU": "#2ca02c", # Green (used for a third device if it's a GPU)
    "dev0_CPU": "#d62728", # Red
    "dev1_CPU": "#9467bd", # Purple
    "dev2_CPU": "#8c564b", # Brown (if it's a CPU)
    # Add more if you have more devices or specific labels
}
# Fallback color cycle
DEFAULT_COLOR_CYCLE = plt.cm.get_cmap('tab20').colors

def get_device_color(device_label, color_index):
    """
    Determines a consistent color for a device label.
    Uses PREDEFINED_DEVICE_COLORS if a match is found, otherwise cycles through DEFAULT_COLOR_CYCLE.
    """
    if device_label in PREDEFINED_DEVICE_COLORS:
        return PREDEFINED_DEVICE_COLORS[device_label]

    # Fallback logic based on common patterns can be added here if needed
    # For now, simple cycle
    return DEFAULT_COLOR_CYCLE[color_index % len(DEFAULT_COLOR_CYCLE)]


def plot_scheduler_experiment_gantt(
    summary_csv_filepath,
    devices_csv_filepath,
    plots_dir_arg,
    experiment_name_prefix_for_title, # e.g., "Experiment04_static_block"
    iteration_to_plot=0,
    font_scale=1.3
):
    try:
        summary_df = pd.read_csv(summary_csv_filepath)
        if summary_df.empty:
            print(f"  Skipping due to empty summary CSV: {summary_csv_filepath}")
            return
        summary_data = summary_df.iloc[0]
    except Exception as e:
        print(f"  Error reading summary CSV {summary_csv_filepath}: {e}. Skipping.")
        return

    try:
        devices_df = pd.read_csv(devices_csv_filepath)
        if devices_df.empty:
            print(f"  Skipping due to empty devices CSV: {devices_csv_filepath}")
            return
    except Exception as e:
        print(f"  Error reading devices CSV {devices_csv_filepath}: {e}. Skipping.")
        return

    # Filter devices_df for the selected iteration
    # Check for 'timed_iteration_idx' (Exp01/03/02) or 'iteration' (suite script)
    iteration_col_name = 'timed_iteration_idx'
    if iteration_col_name not in devices_df.columns and 'iteration' in devices_df.columns:
         iteration_col_name = 'iteration'
    elif iteration_col_name not in devices_df.columns:
         print(f"  Error: Could not find iteration column ('timed_iteration_idx' or 'iteration') in {devices_csv_filepath}. Skipping plot.")
         return

    iter_df = devices_df[devices_df[iteration_col_name] == iteration_to_plot].copy()
    if iter_df.empty:
        original_iter_to_plot = iteration_to_plot
        # Fallback: try iteration 0 if the requested one isn't found
        iteration_to_plot = 0
        iter_df = devices_df[devices_df[iteration_col_name] == iteration_to_plot].copy()
        if iter_df.empty:
            print(f"  No data for iter {original_iter_to_plot} or 0 in {devices_csv_filepath}. Skipping plot.")
            return
        else:
            print(f"  No data for iter {original_iter_to_plot}, using iter 0 for {summary_csv_filepath}.")
            # Update the variable used in the title later
            actual_iteration_plotted = 0
    else:
        actual_iteration_plotted = iteration_to_plot


    critical_cols = ['host_dispatch_offset_ms', 'kernel_duration_ms', 'device_name_label', 'row_begin', 'row_end', 'device_exp_idx']
    if not all(col in iter_df.columns for col in critical_cols):
        missing_cols = [col for col in critical_cols if col not in iter_df.columns]
        print(f"  Devices CSV {devices_csv_filepath} missing critical columns: {missing_cols}. Skipping.")
        return

    iter_df['host_dispatch_offset_ms'] = pd.to_numeric(iter_df['host_dispatch_offset_ms'], errors='coerce')
    iter_df['kernel_duration_ms'] = pd.to_numeric(iter_df['kernel_duration_ms'], errors='coerce')
    iter_df.dropna(subset=['host_dispatch_offset_ms', 'kernel_duration_ms'], inplace=True)
    if iter_df.empty:
        print(f"  No valid numeric kernel timing data for iter {actual_iteration_plotted} after cleaning in {devices_csv_filepath}. Skipping.")
        return

    matrix_path = summary_data.get('matrix_path', 'UnknownMatrix')
    # Get dispatch_mode and scheduler_name from summary_data
    dispatch_mode = summary_data.get('dispatch_mode', 'UnknownDispatch')
    scheduler_name = summary_data.get('scheduler_name', 'UnknownScheduler')
    matrix_total_rows = summary_data.get('num_rows', 0)

    # Get average host times from summary for context in the title
    avg_kernel_submission_ms_summary = summary_data.get('avg_kernel_submission_ms', 0)
    avg_kernel_sync_ms_summary = summary_data.get('avg_kernel_sync_ms', 0)
    avg_overall_kernel_phase_wall_ms_summary = summary_data.get('avg_overall_kernel_phase_wall_ms', 0)
    if avg_overall_kernel_phase_wall_ms_summary < 1e-6 and (avg_kernel_submission_ms_summary > 0 or avg_kernel_sync_ms_summary > 0) :
        avg_overall_kernel_phase_wall_ms_summary = avg_kernel_submission_ms_summary + avg_kernel_sync_ms_summary


    # Sort devices by device_exp_idx for consistent lane ordering
    # Also ensure 'device_name_label' is used for the y-axis labels if available
    unique_device_exp_indices = sorted(iter_df['device_exp_idx'].unique())
    # Map device_exp_idx to device_name_label, using the first instance found in the data
    device_label_map = {}
    for dev_idx in unique_device_exp_indices:
         # Find the first row for this device index to get its label
         label = iter_df[iter_df['device_exp_idx'] == dev_idx]['device_name_label'].iloc[0] if not iter_df[iter_df['device_exp_idx'] == dev_idx].empty else f"Device {dev_idx}"
         device_label_map[dev_idx] = label

    unique_device_labels_in_plot_order = [device_label_map[idx] for idx in unique_device_exp_indices]

    # Calculate actual kernel phase wall time for this specific iteration
    actual_kernel_phase_end_time_this_iter = (iter_df['host_dispatch_offset_ms'] + iter_df['kernel_duration_ms']).max()


    work_per_device_in_iter = {}
    for dev_idx in unique_device_exp_indices:
         dev_label = device_label_map[dev_idx]
         work_per_device_in_iter[dev_label] = (iter_df[iter_df['device_exp_idx'] == dev_idx].apply(
             lambda row: row['row_end'] - row['row_begin'], axis=1
         ).sum())

    if matrix_total_rows == 0:
         print(f"  Matrix {matrix_path} has 0 rows. Bar heights will be default.")
         work_fractions = {label: 0.05 for label in unique_device_labels_in_plot_order} # Default small height
    else:
        work_fractions = {
            label: (work / matrix_total_rows if matrix_total_rows > 0 else 0)
            for label, work in work_per_device_in_iter.items()
        }

    # --- Plotting ---
    num_devices_in_plot = len(unique_device_exp_indices)
    if num_devices_in_plot == 0:
        print(f"  No devices with kernel data for iteration {actual_iteration_plotted} in {devices_csv_filepath}. Skipping plot.")
        return

    fig_height = max(5 * font_scale, (1.5 + num_devices_in_plot * 1.1) * font_scale)
    fig, ax = plt.subplots(figsize=(17 * font_scale, fig_height))

    y_ticks_pos = []
    y_tick_labels = []
    current_y_lane_center = 0
    lane_spacing = 1.0
    max_kernel_activity_end_time_ms = 0.0

    MAX_BAR_THICKNESS = 0.75 * font_scale
    MIN_BAR_THICKNESS = 0.20 * font_scale
    legend_handles = {}

    # Use device_exp_idx for consistent color assignment
    device_color_index_map = {dev_idx: i for i, dev_idx in enumerate(unique_device_exp_indices)}

    for i, dev_idx in enumerate(unique_device_exp_indices): # Iterate in sorted index order
        device_label = device_label_map.get(dev_idx, f"Device {dev_idx}")
        device_lane_y_val = current_y_lane_center
        y_ticks_pos.append(device_lane_y_val)
        y_tick_labels.append(device_label)

        thickness_fraction = work_fractions.get(device_label, 0.0)
        bar_render_height = MIN_BAR_THICKNESS + thickness_fraction * (MAX_BAR_THICKNESS - MIN_BAR_THICKNESS)
        bar_render_height = np.clip(bar_render_height, MIN_BAR_THICKNESS, MAX_BAR_THICKNESS)

        color_idx = device_color_index_map.get(dev_idx, i) # Get consistent index for color
        device_color = get_device_color(device_label, color_idx)
        if device_label not in legend_handles:
            legend_handles[device_label] = plt.Rectangle((0,0),1,1,color=device_color, alpha=0.85)

        # Plot background "envelope" based on the *actual* end time for this iteration
        if actual_kernel_phase_end_time_this_iter > 1e-4:
             ax.barh(device_lane_y_val, actual_kernel_phase_end_time_this_iter,
                    height=bar_render_height * 1.1,
                    left=0, color='whitesmoke', edgecolor='gainsboro', alpha=0.5, zorder=1)


        device_chunks = iter_df[iter_df['device_exp_idx'] == dev_idx].copy()
        # Sort chunks by host_dispatch_offset_ms for cleaner visualization within a lane
        device_chunks = device_chunks.sort_values(by='host_dispatch_offset_ms')

        for _, row_chunk in device_chunks.iterrows():
            host_dispatch_offset_ms_val = row_chunk['host_dispatch_offset_ms']
            kernel_duration_ms_val = row_chunk['kernel_duration_ms']
            if kernel_duration_ms_val < 1e-7: continue # Skip negligible kernels

            ax.barh(
                device_lane_y_val, kernel_duration_ms_val, height=bar_render_height,
                left=host_dispatch_offset_ms_val,
                color=device_color, edgecolor='black', alpha=0.85, zorder=2
            )
            max_kernel_activity_end_time_ms = max(max_kernel_activity_end_time_ms, host_dispatch_offset_ms_val + kernel_duration_ms_val)
        current_y_lane_center += lane_spacing

    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=11 * font_scale)
    ax.invert_yaxis()

    ax.set_xlabel(f"Time Since Host Kernel Dispatch Phase Start (ms) - Timed Iteration {actual_iteration_plotted}", fontsize=13 * font_scale)
    ax.set_ylabel("Device", fontsize=13 * font_scale)
    ax.tick_params(axis='both', which='major', labelsize=11 * font_scale)

    matrix_basename = os.path.basename(matrix_path).replace('.mtx', '')
    # Title uses experiment_name_prefix, scheduler, dispatch_mode, matrix
    title_dispatch_mode = dispatch_mode.replace("_", "-").capitalize()
    title = (f"{experiment_name_prefix_for_title}: {scheduler_name.capitalize()} Scheduler - {title_dispatch_mode} Dispatch\n"
             f"Matrix: {matrix_basename} (Iteration {actual_iteration_plotted})\n"
             f"Avg. Host Kernel Phase (Submit+Sync) from Summary: {avg_overall_kernel_phase_wall_ms_summary:.3f}ms. "
             f"(Avg. Submit Loop: {avg_kernel_submission_ms_summary:.3f}ms, Avg. Sync Wait: {avg_kernel_sync_ms_summary:.3f}ms)")
    ax.set_title(title, fontsize=12 * font_scale, pad=20 * font_scale, loc='center')

    # Determine x-axis limit - use the max end time observed in the current iteration's kernel data
    plot_xlim_val = max_kernel_activity_end_time_ms * 1.10 # Add 10% padding
    if plot_xlim_val <= 1e-3: plot_xlim_val = 1.0 # Ensure a reasonable range if all times are near zero
    ax.set_xlim(0, plot_xlim_val)

    if legend_handles:
        legend_labels_list = list(legend_handles.keys())
        legend_handles_list = list(legend_handles.values())

        # Add proxy artist for the background envelope to the legend
        if actual_kernel_phase_end_time_this_iter > 1e-4:
            legend_handles_list.append(plt.Rectangle((0,0),1,1,color='whitesmoke', alpha=0.5, ec='gainsboro'))
            legend_labels_list.append(f"Kernel Phase Envelope (Up to Max Kernel End Time: {actual_kernel_phase_end_time_this_iter:.3f}ms)")


        ax.legend(legend_handles_list, legend_labels_list, fontsize=10 * font_scale,
                  bbox_to_anchor=(1.01, 1), loc='upper left', title="Legend", title_fontsize=11*font_scale)

    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.78 if legend_handles else 0.95, 0.90 if "\n" in title else 0.92]) # Adjust rect for longer title
    if legend_handles:
        # Further adjust layout if legend is present to prevent overlap with title
        plt.tight_layout(rect=[0, 0, 1 - (0.22 * font_scale), 0.90 if "\n" in title else 0.92]) # Adjusted


    # Construct plot filename based on extracted info
    plot_filename = f"gantt_{scheduler_name}_{dispatch_mode}_{matrix_basename}_iter{actual_iteration_plotted}.png"
    full_plot_path = os.path.join(plots_dir_arg, plot_filename)
    try:
        plt.savefig(full_plot_path, dpi=200)
        print(f"  Saved Gantt plot: {full_plot_path}")
    except Exception as e: print(f"  Error saving plot {full_plot_path}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kernel-Centric Gantt charts for SpMV scheduler experiments.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing summary (results_*) and devices (devices_*) CSV files for a specific scheduler.")
    parser.add_argument('--plots_dir', type=str, required=True, help="Directory where plots will be saved.")
    parser.add_argument('--experiment_name', type=str, default="Experiment", help="Experiment name prefix for plot titles (e.g., 'Experiment04_static_block').")
    parser.add_argument('--iteration_to_plot', type=int, default=9, help="Index of the timed iteration to plot (default: 9).")
    parser.add_argument('--font_scale', type=float, default=1.2, help="Scaling factor for font sizes and plot elements.")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    # Regex to find files and extract matrix basename, scheduler name, and dispatch mode
    # This pattern matches results_matrixbasename_schedulername_dispatchmode.csv
    file_pattern = re.compile(r"results_(?P<matrix_basename>.+?)_(?P<scheduler_name>.+?)_(?P<dispatch_mode>single|multi)\.csv")

    # Search within the provided --results_dir
    summary_csv_files = glob.glob(os.path.join(args.results_dir, "results_*.csv"))

    if not summary_csv_files:
        print(f"No summary CSV files found in {args.results_dir}.")
    else:
        print(f"Found {len(summary_csv_files)} summary CSV files to process.")

    files_to_plot = []

    for summary_csv_file in sorted(summary_csv_files):
        base_name = os.path.basename(summary_csv_file)
        match = file_pattern.match(base_name)

        if match:
            matrix_basename = match.group("matrix_basename")
            scheduler_name = match.group("scheduler_name") # Extracted from filename
            dispatch_mode = match.group("dispatch_mode")   # Extracted from filename

            # Construct corresponding devices.csv filename
            devices_csv_file = os.path.join(args.results_dir, f"devices_{matrix_basename}_{scheduler_name}_{dispatch_mode}.csv")

            if not os.path.exists(devices_csv_file):
                print(f"  Warning: Devices file not found for {summary_csv_file} (expected: {devices_csv_file}). Skipping.")
                continue

            files_to_plot.append({
                "summary_path": summary_csv_file,
                "devices_path": devices_csv_file,
                "scheduler_name_from_file": scheduler_name, # Pass extracted name
                "dispatch_mode_from_file": dispatch_mode # Pass extracted mode
            })
        else:
            print(f"  Skipping file (does not match expected pattern): {summary_csv_file}")

    # Plot each found combination
    for plot_info in files_to_plot:
        print(f"Processing files: {os.path.basename(plot_info['summary_path'])}")

        # Pass the extracted scheduler and dispatch names to the plotting function
        plot_scheduler_experiment_gantt(
            plot_info['summary_path'],
            plot_info['devices_path'],
            args.plots_dir,
            args.experiment_name, # This is the overall experiment prefix passed from run_exp04.sh
            args.iteration_to_plot,
            args.font_scale
        )

    if not files_to_plot:
        print("No valid summary/devices file pairs found to plot.")
    print("\nScheduler Gantt plotting script finished.")