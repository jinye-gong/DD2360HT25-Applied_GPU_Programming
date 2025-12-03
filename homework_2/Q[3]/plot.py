#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_dual_axis(csv_file='timing_results.csv'):
    df = pd.read_csv(csv_file)

    N = df['N'].astype(str)
    cpu = df['CPU_ms']
    naive = df['Naive_ms']
    t8 = df['Tile8_ms']
    t16 = df['Tile16_ms']
    t32 = df['Tile32_ms']

    x = np.arange(len(N))
    width = 0.1  # 每根柱子宽度小一点，避免挤在一起

    fig, ax_cpu = plt.subplots(figsize=(10, 6))
    ax_gpu = ax_cpu.twinx()  # 右侧 y 轴

    bar_cpu  = ax_cpu.bar(x - 2*width, cpu,   width, label='CPU',          color='tab:blue')

    bar_naive = ax_gpu.bar(x - width,   naive, width, label='Naive gemm()', color='tab:orange')
    bar_t8    = ax_gpu.bar(x,           t8,    width, label='tiled 8×8',    color='tab:green')
    bar_t16   = ax_gpu.bar(x + width,   t16,   width, label='tiled 16×16',  color='tab:red')
    bar_t32   = ax_gpu.bar(x + 2*width, t32,   width, label='tiled 32×32',  color='tab:purple')


    ax_cpu.set_xlabel('Matrix size N (N×N)')
    ax_cpu.set_ylabel('CPU runtime (ms)')
    ax_gpu.set_ylabel('GPU runtime (ms)')

    ax_cpu.set_xticks(x)
    ax_cpu.set_xticklabels(N, rotation=45)

    ax_cpu.set_title('CPU vs GPU runtimes for different matrix sizes')

    # 把两个轴的 legend 合在一起
    handles = [bar_cpu, bar_naive, bar_t8, bar_t16, bar_t32]
    labels = [h.get_label() for h in handles]
    ax_cpu.legend(handles, labels, loc='upper left')

    ax_cpu.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('runtime.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_dual_axis()

