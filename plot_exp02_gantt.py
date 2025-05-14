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


def plot_feedback_experiment_gantt(
    summary_csv_filepath,
    devices_csv_filepath,
    plots_dir_arg,
    experiment_name_for_title, # e.g., "Experiment 02 Feedback"
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

    # Expected columns from the modified spmv_cli output for devices.csv:
    # timed_iteration_idx, device_exp_idx, device_name_label, row_begin, row_end, host_dispatch_offset_ms, kernel_duration_ms
    critical_cols = ['host_dispatch_offset_ms', 'kernel_duration_ms', 'device_name_label', 'row_begin', 'row_end', 'device_exp_idx']
    if not all(col in iter_df.columns for col in critical_cols):
        missing_cols = [col for col in critical_cols if col not in iter_df.columns]
        print(f"  Devices CSV {devices_csv_filepath} missing critical columns: {missing_cols}. Skipping.")
        return
    
    iter_df['host_dispatch_offset_ms'] = pd.to_numeric(iter_df['host_dispatch_offset_ms'], errors='coerce')
    iter_df['kernel_duration_ms'] = pd.to_numeric(iter_df['kernel_duration_ms'], errors='coerce')
    iter_df.dropna(subset=['host_dispatch_offset_ms', 'kernel_duration_ms'], inplace=True)
    if iter_df.empty:
        print(f"  No valid numeric kernel timing data for iter {iteration_to_plot} after cleaning in {devices_csv_filepath}. Skipping.")
        return

    matrix_path = summary_data.get('matrix_path', 'UnknownMatrix')
    # Get dispatch_mode from summary_data (new column in spmv_cli output)
    dispatch_mode = summary_data.get('dispatch_mode', 'UnknownDispatch')
    scheduler_name = summary_data.get('scheduler_name', 'UnknownScheduler') # Should be 'feedback'
    matrix_total_rows = summary_data.get('num_rows', 0)
    
    avg_kernel_submission_ms_summary = summary_data.get('avg_kernel_submission_ms', 0)
    avg_kernel_sync_ms_summary = summary_data.get('avg_kernel_sync_ms', 0)
    avg_overall_kernel_phase_wall_ms_summary = summary_data.get('avg_overall_kernel_phase_wall_ms', 0)
    if avg_overall_kernel_phase_wall_ms_summary < 1e-6 and (avg_kernel_submission_ms_summary > 0 or avg_kernel_sync_ms_summary > 0) :
        avg_overall_kernel_phase_wall_ms_summary = avg_kernel_submission_ms_summary + avg_kernel_sync_ms_summary

    # Sort by device_exp_idx to ensure consistent lane ordering if labels are not perfectly ordered initially
    # iter_df.sort_values(by='device_exp_idx', inplace=True) # device_name_label should be used for grouping
    unique_device_labels_in_iter = sorted(iter_df['device_name_label'].unique(), key=lambda x: int(re.search(r'dev(\d+)_', x).group(1)))


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
        print(f"  No devices with kernel data for iteration {iteration_to_plot} in {devices_csv_filepath}. Skipping plot.")
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

    # Use device_exp_idx for consistent color assignment if labels are tricky
    device_exp_idx_map = {label: idx for idx, label in enumerate(unique_device_labels_in_iter)}

    for device_label in unique_device_labels_in_iter: # Iterate in sorted order
        device_lane_y_val = current_y_lane_center
        y_ticks_pos.append(device_lane_y_val)
        y_tick_labels.append(device_label)

        thickness_fraction = work_fractions.get(device_label, 0.0)
        bar_render_height = MIN_BAR_THICKNESS + thickness_fraction * (MAX_BAR_THICKNESS - MIN_BAR_THICKNESS)
        bar_render_height = np.clip(bar_render_height, MIN_BAR_THICKNESS, MAX_BAR_THICKNESS)
        
        color_idx = device_exp_idx_map.get(device_label, 0) # Get consistent index for color
        device_color = get_device_color(device_label, color_idx)
        if device_label not in legend_handles: 
            legend_handles[device_label] = plt.Rectangle((0,0),1,1,color=device_color, alpha=0.85)

        if avg_overall_kernel_phase_wall_ms_summary > 1e-4:
             ax.barh(device_lane_y_val, avg_overall_kernel_phase_wall_ms_summary, 
                    height=bar_render_height * 1.1, 
                    left=0, color='whitesmoke', edgecolor='gainsboro', alpha=0.5, zorder=1)

        device_chunks = iter_df[iter_df['device_name_label'] == device_label]
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
    # Title uses experiment_name, scheduler, dispatch_mode, matrix
    title_dispatch_mode = dispatch_mode.replace("_", "-").capitalize()
    title = (f"{experiment_name_for_title}: {scheduler_name.capitalize()} Scheduler - {title_dispatch_mode} Dispatch\n"
             f"Matrix: {matrix_basename} (Iteration {iteration_to_plot})\n"
             f"Avg. Host Kernel Phase (Submit+Sync): {avg_overall_kernel_phase_wall_ms_summary:.3f}ms. "
             f"(Submit Loop: {avg_kernel_submission_ms_summary:.3f}ms, Sync Wait: {avg_kernel_sync_ms_summary:.3f}ms)")
    ax.set_title(title, fontsize=12 * font_scale, pad=20 * font_scale, loc='center')


    plot_xlim_val = max(max_kernel_activity_end_time_ms, avg_overall_kernel_phase_wall_ms_summary) * 1.10
    if plot_xlim_val <= 1e-3: plot_xlim_val = 1.0 
    ax.set_xlim(0, plot_xlim_val)
    
    if legend_handles:
        legend_labels_list = list(legend_handles.keys())
        legend_handles_list = list(legend_handles.values())
        
        legend_handles_list.append(plt.Rectangle((0,0),1,1,color='whitesmoke', alpha=0.5, ec='gainsboro'))
        legend_labels_list.append("Host Kernel Phase Envelope (Avg.)")

        ax.legend(legend_handles_list, legend_labels_list, fontsize=10 * font_scale, 
                  bbox_to_anchor=(1.01, 1), loc='upper left', title="Legend", title_fontsize=11*font_scale)

    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.78 if legend_handles else 0.95, 0.90 if "\n" in title else 0.92]) # Adjust rect for longer title
    
    plot_filename = f"gantt_{experiment_name_for_title.lower().replace(' ','_')}_{matrix_basename}_{scheduler_name}_{dispatch_mode}_iter{iteration_to_plot}.png"
    full_plot_path = os.path.join(plots_dir_arg, plot_filename)
    try:
        plt.savefig(full_plot_path, dpi=200)
        print(f"  Saved Gantt plot: {full_plot_path}")
    except Exception as e: print(f"  Error saving plot {full_plot_path}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kernel-Centric Gantt charts for SpMV Experiment 02 (Feedback Scheduler).")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing summary (results_*) and devices (devices_*) CSV files.")
    parser.add_argument('--plots_dir', type=str, required=True, help="Directory where plots will be saved.")
    parser.add_argument('--experiment_name', type=str, default="Experiment02_Feedback", help="Experiment name prefix for plot titles (e.g., 'Experiment02_Feedback').")
    parser.add_argument('--iteration_to_plot', type=int, default=0, help="Index of the timed iteration to plot (default: 0).")
    parser.add_argument('--font_scale', type=float, default=1.2, help="Scaling factor for font sizes and plot elements.") # Adjusted default
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    # Regex to find files and extract matrix basename and dispatch mode
    # Example filename: results_csr_r25000_c25000_d0_01_feedback_single.csv
    file_pattern = re.compile(r"results_(?P<matrix_basename>csr_r\d+_c\d+_d\S+?)_feedback_(?P<dispatch_mode>single|multi)\.csv")
    
    summary_csv_files = glob.glob(os.path.join(args.results_dir, "results_*_feedback_*.csv"))

    if not summary_csv_files: 
        print(f"No summary CSV files matching '*_feedback_*.csv' found in {args.results_dir}.")
    else: 
        print(f"Found {len(summary_csv_files)} summary CSV files for Feedback scheduler Experiment 02.")

    processed_matrices = set()
    summary_files_to_plot = []

    for summary_csv_file in sorted(summary_csv_files):
        base_name = os.path.basename(summary_csv_file)
        match = file_pattern.match(base_name)
        
        if match:
            matrix_basename = match.group("matrix_basename")
            dispatch_mode = match.group("dispatch_mode")
            
            # Construct corresponding devices.csv filename
            # devices_csr_r25000_c25000_d0_01_feedback_single.csv
            devices_csv_file = os.path.join(args.results_dir, f"devices_{matrix_basename}_feedback_{dispatch_mode}.csv")
            
            if not os.path.exists(devices_csv_file):
                print(f"  Warning: Devices file not found for {summary_csv_file} (expected: {devices_csv_file}). Skipping.")
                continue
            
            summary_files_to_plot.append({
                "summary_path": summary_csv_file,
                "devices_path": devices_csv_file,
                "matrix": matrix_basename,
                "dispatch": dispatch_mode
            })
        else:
            print(f"  Skipping file (does not match expected pattern): {summary_csv_file}")

    # Plot each found combination
    for plot_info in summary_files_to_plot:
        print(f"Processing Matrix: {plot_info['matrix']}, Dispatch: {plot_info['dispatch']}")
        print(f"  Summary: {plot_info['summary_path']}")
        print(f"  Devices: {plot_info['devices_path']}")
        
        plot_feedback_experiment_gantt( 
            plot_info['summary_path'], 
            plot_info['devices_path'], 
            args.plots_dir,
            args.experiment_name, # This is the overall experiment name, title will add matrix/dispatch
            args.iteration_to_plot, 
            args.font_scale
        )
    
    if not summary_files_to_plot:
        print("No valid summary/devices file pairs found to plot.")
    print("\nFeedback scheduler Gantt plotting script finished.")