#!/usr/bin/env python3


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def plot_stacked_bar_chart(csv_file='timing_results.csv'):
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run batch_test.sh first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    df = df.sort_values('N')
    
    N_values = df['N'].values
    timeH2D = df['timeH2D'].values
    timeKernel = df['timeKernel'].values
    timeD2H = df['timeD2H'].values
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(N_values))
    width = 0.6
    
    p1 = ax.bar(x, timeH2D, width, label='Host to Device', color='#3498db')
    p2 = ax.bar(x, timeKernel, width, bottom=timeH2D, label='CUDA Kernel', color='#2ecc71')
    p3 = ax.bar(x, timeD2H, width, bottom=timeH2D + timeKernel, label='Device to Host', color='#e74c3c')
    
    ax.set_xlabel('Vector Length (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Timing Breakdown for Vector Addition (Stacked Bar Chart)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n):,}' for n in N_values], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = 'timing_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    output_file_pdf = 'timing_breakdown.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Plot saved to {output_file_pdf}")
    
    plt.show()
    
    print("\n=== Summary Statistics ===")
    print(f"Total tests: {len(N_values)}")
    print(f"Vector length range: {N_values.min():,} to {N_values.max():,}")
    print(f"\nAverage times:")
    print(f"  Host to Device: {timeH2D.mean():.4f} ms")
    print(f"  CUDA Kernel:    {timeKernel.mean():.4f} ms")
    print(f"  Device to Host: {timeD2H.mean():.4f} ms")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'timing_results.csv'
    plot_stacked_bar_chart(csv_file)




