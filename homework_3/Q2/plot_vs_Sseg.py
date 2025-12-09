#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def plot_vs_Sseg(csv_file='results_vs_Sseg.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    # 按 S_seg 排序，方便看趋势
    df = df.sort_values('S_seg')

    N_vals = df['N'].values
    S_seg = df['S_seg'].values
    no_stream = df['no_stream_ms'].values
    stream = df['stream_ms'].values
    speedup = df['speedup'].values

    # 假设整张表 N 都一样，取第一个就行
    if len(np.unique(N_vals)) == 1:
        title_suffix = f' (N={int(N_vals[0]):,})'
    else:
        title_suffix = ''

    # -------- 图 1：不同 S_seg 下的时间对比 --------
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(S_seg))

    ax.plot(x, no_stream, marker='o', label='No-stream GPU time (ms)')
    ax.plot(x, stream, marker='s', label='4-stream GPU time (ms)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s):,}' for s in S_seg], rotation=30, ha='right')

    ax.set_xlabel('Segment size S_seg', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('GPU Vector Add: Time vs Segment Size' + title_suffix, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('time_vs_Sseg.png', dpi=300)
    print('Saved time_vs_Sseg.png')

    # -------- 图 2：不同 S_seg 下的 speedup --------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(x, speedup, marker='^', label='Speedup (no-stream / streams)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(s):,}' for s in S_seg], rotation=30, ha='right')

    ax2.set_xlabel('Segment size S_seg', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs Segment Size' + title_suffix, fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('speedup_vs_Sseg.png', dpi=300)
    print('Saved speedup_vs_Sseg.png')

    plt.show()

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results_vs_Sseg.csv'
    plot_vs_Sseg(csv_file)
