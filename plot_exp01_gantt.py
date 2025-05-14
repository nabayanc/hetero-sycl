#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
import numpy as np

# Consistent Color Scheme
PREDEFINED_DEVICE_COLORS = {
    "dev0_GPU": "#1f77b4", # Blue
    "dev1_GPU": "#ff7f0e", # Orange
    "dev2_GPU": "#2ca02c", # Green (used for cpu1gpu2_split's second GPU)
    "dev0_CPU": "#d62728", # Red
    "dev1_CPU": "#9467bd", # Purple (if a second CPU)
    # For host phases (if we were to plot them separately again, not used for device bars)
    # "Host_Y-Reset": "#8c564b",
    # "Host_Kernel-Submit": "#e377c2",
    # "Host_Kernel-Sync": "#7f7f7f",
    # "Host_D2H-Copy": "#bcbd22",
    # "Host_Scheduling": "#17becf"
}
DEFAULT_COLOR_CYCLE = plt.cm.get_cmap('tab20').colors # Using a perceptually uniform colormap

def get_device_color(device_label, color_index):
    """
    Determines a consistent color for a device label.
    Uses PREDEFINED_DEVICE_COLORS if a match is found, otherwise cycles through DEFAULT_COLOR_CYCLE.
    """
    # Try direct match first (case sensitive)
    if device_label in PREDEFINED_DEVICE_COLORS:
        return PREDEFINED_DEVICE_COLORS[device_label]

    # Fallback logic based on common patterns in device_label
    # This helps assign somewhat consistent colors if exact labels change slightly
    # but device types and indices are present.
    lower_label = device_label.lower()
    if "gpu" in lower_label:
        if "dev0" in lower_label: return PREDEFINED_DEVICE_COLORS.get("dev0_GPU", DEFAULT_COLOR_CYCLE[0])
        if "dev1" in lower_label: return PREDEFINED_DEVICE_COLORS.get("dev1_GPU", DEFAULT_COLOR_CYCLE[1])
        if "dev2" in lower_label: return PREDEFINED_DEVICE_COLORS.get("dev2_GPU", DEFAULT_COLOR_CYCLE[2])
        # Generic GPU color from cycle if specific index not matched
        return DEFAULT_COLOR_CYCLE[color_index % 3] # Cycle through first few for GPUs
    if "cpu" in lower_label:
        if "dev0" in lower_label: return PREDEFINED_DEVICE_COLORS.get("dev0_CPU", DEFAULT_COLOR_CYCLE[3])
        # Generic CPU color from cycle, offset to avoid clash with initial GPU colors
        return DEFAULT_COLOR_CYCLE[(3 + color_index) % len(DEFAULT_COLOR_CYCLE)]

    # Absolute fallback: cycle through default colors
    return DEFAULT_COLOR_CYCLE[color_index % len(DEFAULT_COLOR_CYCLE)]


def plot_kernel_centric_gantt(
    summary_csv_filepath,
    devices_csv_filepath,
    plots_dir_arg,
    experiment_name_for_title,
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

    iter_df = devices_df[devices_df['timed_iteration_idx'] == iteration_to_plot].copy()
    if iter_df.empty:
        original_iter_to_plot = iteration_to_plot
        iteration_to_plot = 0
        iter_df = devices_df[devices_df['timed_iteration_idx'] == iteration_to_plot].copy()
        if iter_df.empty:
            print(f"  No data for iter {original_iter_to_plot} or 0 in {devices_csv_filepath}. Skipping plot.")
            return
        else:
            print(f"  No data for iter {original_iter_to_plot}, using iter 0 for {summary_csv_filepath}.")

    critical_cols = ['host_dispatch_offset_ms', 'kernel_duration_ms', 'device_name_label', 'row_begin', 'row_end']
    if not all(col in iter_df.columns for col in critical_cols):
        missing_cols = [col for col in critical_cols if col not in iter_df.columns]
        print(f"  Devices CSV {devices_csv_filepath} missing critical columns: {missing_cols}. Skipping.")
        return

    iter_df['host_dispatch_offset_ms'] = pd.to_numeric(iter_df['host_dispatch_offset_ms'], errors='coerce')
    iter_df['kernel_duration_ms'] = pd.to_numeric(iter_df['kernel_duration_ms'], errors='coerce')
    iter_df.dropna(subset=['host_dispatch_offset_ms', 'kernel_duration_ms'], inplace=True)
    if iter_df.empty:
        print(f"  No valid numeric kernel timing data for iter {iteration_to_plot} after cleaning. Skipping.")
        return

    matrix_path = summary_data.get('matrix_path', 'UnknownMatrix')
    config_type = summary_data.get('config_type', 'UnknownConfig') # For Exp1/Exp03
    scheduler_name_from_summary = summary_data.get('scheduler_name', '') # For Exp02 (spmv_cli)
    matrix_total_rows = summary_data.get('num_rows', 0)

    # --- Calculate actual wall times for the plotted iteration ---
    # This requires recalculating from the dispatch offsets and kernel durations
    # The 'overall_kernel_phase_wall_ms' from summary is the average, we need the actual for this iter.
    # Find the max end time (dispatch offset + duration) across all kernels in this iteration
    actual_kernel_phase_end_time_this_iter = (iter_df['host_dispatch_offset_ms'] + iter_df['kernel_duration_ms']).max()

    unique_device_labels_in_iter = sorted(iter_df['device_name_label'].unique())

    work_per_device_in_iter = {
        label: (iter_df[iter_df['device_name_label'] == label]['row_end'] - iter_df[iter_df['device_name_label'] == label]['row_begin']).sum()
        for label in unique_device_labels_in_iter
    }
    work_fractions = {
        label: (work / matrix_total_rows if matrix_total_rows > 0 else 0.05)
        for label, work in work_per_device_in_iter.items()
    }

    num_devices_in_plot = len(unique_device_labels_in_iter)
    if num_devices_in_plot == 0:
        print(f"  No devices with kernel data for iteration {iteration_to_plot}. Skipping plot.")
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

    for i, device_label in enumerate(unique_device_labels_in_iter):
        device_lane_y_val = current_y_lane_center
        y_ticks_pos.append(device_lane_y_val)
        y_tick_labels.append(device_label)

        thickness_fraction = work_fractions.get(device_label, 0.0)
        bar_render_height = MIN_BAR_THICKNESS + thickness_fraction * (MAX_BAR_THICKNESS - MIN_BAR_THICKNESS)
        bar_render_height = np.clip(bar_render_height, MIN_BAR_THICKNESS, MAX_BAR_THICKNESS)

        color_idx = i # Simple index for color cycling
        device_color = get_device_color(device_label, color_idx) # Call the get_color function
        if device_label not in legend_handles:
            legend_handles[device_label] = plt.Rectangle((0,0),1,1,color=device_color, alpha=0.85)

        # Plot background "envelope" based on the *actual* end time for this iteration
        # Use the max kernel end time as a proxy for the overall kernel phase end
        if actual_kernel_phase_end_time_this_iter > 1e-4:
             ax.barh(device_lane_y_val, actual_kernel_phase_end_time_this_iter,
                    height=bar_render_height * 1.1, # Slightly larger than kernel bar
                    left=0, color='whitesmoke', edgecolor='gainsboro', alpha=0.5, zorder=1)


        device_chunks = iter_df[iter_df['device_name_label'] == device_label]
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

    ax.set_xlabel(f"Time Since Host Kernel Dispatch Phase Start (ms) - Timed Iteration {iteration_to_plot}", fontsize=13 * font_scale)
    ax.set_ylabel("Device", fontsize=13 * font_scale)
    ax.tick_params(axis='both', which='major', labelsize=11 * font_scale)

    matrix_basename = os.path.basename(matrix_path).replace('.mtx', '')
    # Determine display_config for title (Exp1/Exp03 use config_type, Exp02 uses scheduler_name)
    display_config = config_type if config_type and pd.notna(config_type) else scheduler_name_from_summary
    if not display_config: display_config = "N/A" # Fallback if neither is available

    # Get average host times from summary for context, but plot uses iteration-specific kernel times
    avg_kernel_submission_ms_summary = summary_data.get('avg_kernel_submission_ms', 0)
    avg_kernel_sync_ms_summary = summary_data.get('avg_kernel_sync_ms', 0)
    avg_overall_kernel_phase_wall_ms_summary = summary_data.get('avg_overall_kernel_phase_wall_ms', 0)
    if avg_overall_kernel_phase_wall_ms_summary < 1e-6 and (avg_kernel_submission_ms_summary > 0 or avg_kernel_sync_ms_summary > 0) :
        avg_overall_kernel_phase_wall_ms_summary = avg_kernel_submission_ms_summary + avg_kernel_sync_ms_summary


    title = (f"{experiment_name_for_title}: Kernel Activity Timeline - {display_config} on {matrix_basename} (Iter {iteration_to_plot})\n"
             f"Avg. Host Kernel Phase (Submit+Sync) from Summary: {avg_overall_kernel_phase_wall_ms_summary:.3f}ms. "
             f"(Avg. Submit Loop: {avg_kernel_submission_ms_summary:.3f}ms, Avg. Sync Wait: {avg_kernel_sync_ms_summary:.3f}ms)")
    ax.set_title(title, fontsize=12 * font_scale, pad=15 * font_scale, loc='center')

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
    plt.tight_layout(rect=[0, 0, 0.78 if legend_handles else 0.95, 0.92]) # Adjust rect for legend
    if legend_handles:
         # Further adjust layout if legend is present to prevent overlap with title
         plt.tight_layout(rect=[0, 0, 1 - (0.22 * font_scale), 0.90 if "\n" in title else 0.92])


    plot_filename = f"gantt_kernel_centric_iter{iteration_to_plot}_{matrix_basename}_{display_config.replace(' ','_')}.png"
    full_plot_path = os.path.join(plots_dir_arg, plot_filename)
    try:
        plt.savefig(full_plot_path, dpi=200)
        print(f"  Saved kernel-centric Gantt plot: {full_plot_path}")
    except Exception as e: print(f"  Error saving plot {full_plot_path}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kernel-Centric Gantt charts for SpMV experiments.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing summary (results_*) and devices (devices_*) CSV files.")
    parser.add_argument('--plots_dir', type=str, required=True, help="Directory where plots will be saved.")
    parser.add_argument('--experiment_name', type=str, default="Experiment", help="Experiment name prefix for plot titles (e.g., 'Exp1', 'Exp03').")
    parser.add_argument('--iteration_to_plot', type=int, default=0, help="Index of the timed iteration to plot (default: 0).")
    parser.add_argument('--font_scale', type=float, default=1.3, help="Scaling factor for font sizes and plot elements.")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    # Look for files generated by exp01 and exp03 main executables
    # Pattern: results_matrixbasename_configtype.csv
    summary_csv_files = glob.glob(os.path.join(args.results_dir, "results_*.csv"))

    if not summary_csv_files: print(f"No summary CSV files found in {args.results_dir}.")
    else: print(f"Found {len(summary_csv_files)} summary CSV files for kernel-centric Gantt plots.")

    files_to_plot = []
    file_pattern = re.compile(r"results_(?P<matrix_basename>.+?)_(?P<config_type>cpu1|gpu1|gpu2_split|cpu1gpu2_split)\.csv")

    for summary_csv_file in sorted(summary_csv_files):
        base_name = os.path.basename(summary_csv_file)
        match = file_pattern.match(base_name)

        if match:
            matrix_basename = match.group("matrix_basename")
            config_type = match.group("config_type")

            devices_file_base = f"devices_{matrix_basename}_{config_type}.csv"
            devices_csv_file = os.path.join(args.results_dir, devices_file_base)

            if not os.path.exists(devices_csv_file):
                 # Check the alternative naming convention used by the C++ apps if the direct replacement doesn't work
                alt_devices_file_base = "devices_" + base_name[len("results_"):]
                alt_devices_csv_file = os.path.join(args.results_dir, alt_devices_file_base)
                if os.path.exists(alt_devices_csv_file):
                    devices_csv_file = alt_devices_csv_file # Use the alternative name
                else:
                    print(f"  Warning: Devices file not found for {summary_csv_file} (expected: {devices_csv_file} or {alt_devices_csv_file}). Skipping.")
                    continue

            files_to_plot.append({
                "summary_path": summary_csv_file,
                "devices_path": devices_csv_file
            })
        else:
             print(f"  Skipping file (does not match expected pattern for Exp01/03): {summary_csv_file}")


    for file_info in files_to_plot:
        print(f"Processing {file_info['summary_path']} and {file_info['devices_path']}...")
        plot_kernel_centric_gantt(
            file_info['summary_path'],
            file_info['devices_path'],
            args.plots_dir,
            args.experiment_name,
            args.iteration_to_plot,
            args.font_scale
        )

    if not files_to_plot:
         print("No valid summary/devices file pairs found to plot for Experiment 01/03 pattern.")
    print("\nKernel-centric Gantt plotting script finished.")