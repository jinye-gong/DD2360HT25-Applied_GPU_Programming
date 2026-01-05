#!/usr/bin/env python3
"""
Script to plot results from Tensor Core experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

def safe_float(value, default=0.0):
    """Safely convert value to float, handling empty strings and invalid values"""
    if not value or value.strip() == '' or value == 'CPU:':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def plot_performance_comparison(csv_file):
    """Plot performance comparison for different matrix sizes"""
    sizes = []
    cpu_times = []
    gemm_times = []
    tiled_8_times = []
    tiled_16_times = []
    tiled_32_times = []
    wmma_times = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['size'] or row['size'].strip() == '':
                    continue
                sizes.append(int(row['size']))
                cpu_times.append(safe_float(row['cpu_time']))
                gemm_times.append(safe_float(row['gemm_time']))
                tiled_8_times.append(safe_float(row['tiled_8_time']))
                tiled_16_times.append(safe_float(row['tiled_16_time']))
                tiled_32_times.append(safe_float(row['tiled_32_time']))
                wmma_times.append(safe_float(row['wmma_time']))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    
    x = np.arange(len(sizes))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - 2.5*width, cpu_times, width, label='CPU', alpha=0.8)
    bars2 = ax.bar(x - 1.5*width, gemm_times, width, label='gemm()', alpha=0.8)
    bars3 = ax.bar(x - 0.5*width, tiled_8_times, width, label='tiled_gemm (8x8)', alpha=0.8)
    bars4 = ax.bar(x + 0.5*width, tiled_16_times, width, label='tiled_gemm (16x16)', alpha=0.8)
    bars5 = ax.bar(x + 1.5*width, tiled_32_times, width, label='tiled_gemm (32x32)', alpha=0.8)
    bars6 = ax.bar(x + 2.5*width, wmma_times, width, label='wmma_gemm (Tensor Core)', alpha=0.8)
    
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title('Performance Comparison: CPU vs CUDA Kernels', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300)
    print(f"Saved plot: results/performance_comparison.png")
    plt.close()

def plot_wmma_accuracy(csv_file):
    """Plot WMMA accuracy (error) vs matrix size"""
    sizes = []
    max_errors = []
    avg_errors = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['size'] or row['size'].strip() == '':
                    continue
                # Only plot rows that have valid error data
                max_err = safe_float(row['wmma_max_error'], None)
                avg_err = safe_float(row['wmma_avg_error'], None)
                if max_err is not None and avg_err is not None:
                    sizes.append(int(row['size']))
                    max_errors.append(max_err)
                    avg_errors.append(avg_err)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    
    if len(sizes) == 0:
        print("Warning: No valid WMMA error data found. Skipping accuracy plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, max_errors, 'r-o', label='Max Error', linewidth=2, markersize=8)
    ax.plot(sizes, avg_errors, 'b-s', label='Average Error', linewidth=2, markersize=8)
    
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
    ax.set_ylabel('Error vs CPU Reference', fontsize=12)
    ax.set_title('WMMA (Tensor Core) Accuracy Analysis', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/wmma_accuracy.png', dpi=300)
    print(f"Saved plot: results/wmma_accuracy.png")
    plt.close()

def main():
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found. Please run experiments first.")
        sys.exit(1)
    
    print("Generating plots...")
    
    # Plot performance comparison
    perf_file = os.path.join(results_dir, 'performance_comparison.csv')
    if os.path.exists(perf_file):
        plot_performance_comparison(perf_file)
        plot_wmma_accuracy(perf_file)
    else:
        print(f"Warning: {perf_file} not found. Skipping plots.")
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()

