import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# Directory for plots
os.makedirs('benchmarks/plots', exist_ok=True)

# Regex to extract params from filename
pattern = re.compile(r'devices_(\d+)_(\S+)_(\w+)\.csv')

for path in glob.glob('benchmarks/results/devices_*.csv'):
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        continue
    size, dstr, sched = m.groups()
    
    df = pd.read_csv(path)
    df['chunk_size'] = df['row_end'] - df['row_begin']
    max_chunk   = df['chunk_size'].max()
    band_height = max_chunk * 1.5  # spacing between device lanes

    devices = sorted(df['dev_idx'].unique())
    cmap    = plt.get_cmap('tab20', len(devices))

    for it in sorted(df['iteration'].unique()):
        out_fp = f"benchmarks/plots/{size}_{dstr}_{sched}_iter{it}.png"
        # skip if we've already made this plot
        if os.path.exists(out_fp):
            continue

        sub = df[df['iteration'] == it]
        fig, ax = plt.subplots(figsize=(8, 6))

        # draw one bar per chunk
        for idx, dev in enumerate(devices):
            dev_df = sub[sub['dev_idx'] == dev]
            baseline = (len(devices) - 1 - idx) * band_height
            for _, row in dev_df.iterrows():
                start  = row['launch_ms']
                width  = row['kernel_ms']
                height = row['chunk_size']
                bottom = baseline - height / 2
                ax.barh(
                  bottom,
                  width,
                  height=height,
                  left=start,
                  color=cmap(idx),
                  edgecolor='black',
                  alpha=0.8
                )

        yticks = [(len(devices) - 1 - i) * band_height for i in range(len(devices))]
        ylabels = [f"Dev {dev}" for dev in devices]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.set_xlabel('Time since scheduler run start (ms)')
        ax.set_ylabel('Device')
        ax.set_title(f"{sched} — matrix {size}_{dstr} — iter {it}")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_fp)
        plt.close(fig)
