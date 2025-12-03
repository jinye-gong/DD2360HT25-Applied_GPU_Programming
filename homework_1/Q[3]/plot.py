#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

def plot_stacked_bar_chart(csv_file='timing_results_float.csv'):
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run batch_test.sh first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    data_type = df['DataType'].iloc[0] if 'DataType' in df.columns else 'float'
    
    df['MatrixSize'] = df.apply(
        lambda row: f"A({int(row['numARows'])}x{int(row['numAColumns'])}) * B({int(row['numBRows'])}x{int(row['numBColumns'])})",
        axis=1
    )
    
    df = df.sort_values('totalTime')
    
    matrix_labels = df['MatrixSize'].values
    timeH2D = df['timeH2D'].values
    timeKernel = df['timeKernel'].values
    timeD2H = df['timeD2H'].values
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(matrix_labels))
    width = 0.7
    
    p1 = ax.bar(x, timeH2D, width, label='Host to Device', color='#3498db')
    p2 = ax.bar(x, timeKernel, width, bottom=timeH2D, label='CUDA Kernel', color='#2ecc71')
    p3 = ax.bar(x, timeD2H, width, bottom=timeH2D + timeKernel, label='Device to Host', color='#e74c3c')
    
    ax.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Timing Breakdown for Matrix Multiplication (DataType: {data_type})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    base_name = csv_file.replace('.csv', '')
    output_file = f'{base_name}_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    output_file_pdf = f'{base_name}_breakdown.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Plot saved to {output_file_pdf}")
    
    plt.show()
    
    print("\n=== Summary Statistics ===")
    print(f"DataType: {data_type}")
    print(f"Total tests: {len(matrix_labels)}")
    print(f"\nAverage times:")
    print(f"  Host to Device: {timeH2D.mean():.4f} ms")
    print(f"  CUDA Kernel:    {timeKernel.mean():.4f} ms")
    print(f"  Device to Host: {timeD2H.mean():.4f} ms")
    print(f"\nTotal time range: {df['totalTime'].min():.4f} ms to {df['totalTime'].max():.4f} ms")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'timing_results_float.csv'
    plot_stacked_bar_chart(csv_file)


