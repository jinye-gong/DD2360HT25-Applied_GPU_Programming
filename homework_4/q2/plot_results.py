#!/usr/bin/env python3
"""
Script to plot results from the heat equation experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

def plot_flops_analysis(csv_file):
    """Plot FLOPS vs dimX"""
    dimX = []
    flops = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dimX.append(int(row['dimX']))
                flops.append(float(row['FLOPS']))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(dimX, flops, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('dimX (Grid Size)', fontsize=12)
    plt.ylabel('FLOPS (Floating Point Operations per Second)', fontsize=12)
    plt.title('SpMV Performance: FLOPS vs Grid Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/flops_vs_dimX.png', dpi=300)
    print(f"Saved plot: results/flops_vs_dimX.png")
    plt.close()

def plot_error_vs_nsteps(csv_file):
    """Plot relative error vs number of steps"""
    nsteps = []
    errors = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nsteps.append(int(row['nsteps']))
                errors.append(float(row['relative_error']))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(nsteps, errors, 'r-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Time Steps (nsteps)', fontsize=12)
    plt.ylabel('Relative Error', fontsize=12)
    plt.title('Convergence Analysis: Relative Error vs Number of Steps (dimX=1024)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/error_vs_nsteps.png', dpi=300)
    print(f"Saved plot: results/error_vs_nsteps.png")
    plt.close()

def plot_prefetch_comparison(csv_file):
    """Plot prefetching performance comparison"""
    prefetch_enabled = []
    prefetch_disabled = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['prefetch_enabled'] == 'true':
                    prefetch_enabled.append(float(row['time_us']))
                else:
                    prefetch_disabled.append(float(row['time_us']))
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return
    
    if len(prefetch_enabled) == 0 or len(prefetch_disabled) == 0:
        print("Warning: Insufficient data for prefetch comparison")
        return
    
    categories = ['With Prefetch', 'Without Prefetch']
    times = [prefetch_enabled[0], prefetch_disabled[0]]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, times, color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.ylabel('Iteration Time (microseconds)', fontsize=12)
    plt.title('Performance Comparison: With vs Without Prefetching', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.0f}μs', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calculate and display speedup
    speedup = times[1] / times[0]
    plt.text(0.5, max(times) * 0.9, f'Speedup: {speedup:.2f}x', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/prefetch_comparison.png', dpi=300)
    print(f"Saved plot: results/prefetch_comparison.png")
    plt.close()

def main():
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found. Please run experiments first.")
        sys.exit(1)
    
    print("Generating plots...")
    
    # Plot FLOPS analysis
    flops_file = os.path.join(results_dir, 'flops_analysis.csv')
    if os.path.exists(flops_file):
        plot_flops_analysis(flops_file)
    else:
        print(f"Warning: {flops_file} not found. Skipping FLOPS plot.")
    
    # Plot error vs nsteps
    error_file = os.path.join(results_dir, 'error_vs_nsteps.csv')
    if os.path.exists(error_file):
        plot_error_vs_nsteps(error_file)
    else:
        print(f"Warning: {error_file} not found. Skipping error plot.")
    
    # Plot prefetching comparison
    prefetch_file = os.path.join(results_dir, 'prefetch_comparison.csv')
    if os.path.exists(prefetch_file):
        plot_prefetch_comparison(prefetch_file)
    else:
        print(f"Warning: {prefetch_file} not found. Skipping prefetch comparison plot.")
    
    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()

