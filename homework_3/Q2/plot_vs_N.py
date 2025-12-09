#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def plot_vs_N(csv_file='results_vs_N.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    # 按 N 排序，防止顺序乱
    df = df.sort_values('N')

    N = df['N'].values
    no_stream = df['no_stream_ms'].values
    stream = df['stream_ms'].values
    speedup = df['speedup'].values

    # -------- 图 1：不同 N 下的时间对比（无 stream vs 有 stream） --------
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(N))

    ax.plot(x, no_stream, marker='o', label='No-stream GPU time (ms)')
    ax.plot(x, stream, marker='s', label='4-stream GPU time (ms)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n):,}' for n in N], rotation=30, ha='right')

    ax.set_xlabel('Vector length N', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('GPU Vector Add: Time vs N (No-stream vs 4-stream)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('time_vs_N.png', dpi=300)
    print('Saved time_vs_N.png')

    # -------- 图 2：不同 N 下的加速比 --------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(x, speedup, marker='^', label='Speedup (no-stream / streams)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(n):,}' for n in N], rotation=30, ha='right')

    ax2.set_xlabel('Vector length N', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs N (no-stream / 4-stream)', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('speedup_vs_N.png', dpi=300)
    print('Saved speedup_vs_N.png')

    plt.show()

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results_vs_N.csv'
    plot_vs_N(csv_file)
