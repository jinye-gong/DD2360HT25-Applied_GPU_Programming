#!/usr/bin/env python3
"""
Script to plot timing data from the reduction program.
This creates a bar chart showing CPU vs GPU time for different array lengths.
"""

import matplotlib.pyplot as plt
import csv
import sys
import os

def load_timing_csv(filename):
    """Load timing data from CSV file."""
    array_lengths = []
    cpu_times = []
    gpu_times = []
    speedups = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                array_lengths.append(int(row['array_length']))
                cpu_times.append(float(row['cpu_time_ms']))
                gpu_times.append(float(row['gpu_time_ms']))
                speedups.append(float(row['speedup']))
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run generate_timing_data.py first.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing column in CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return array_lengths, cpu_times, gpu_times, speedups

def plot_timing_bar_chart(array_lengths, cpu_times, gpu_times, speedups, filename):
    """Plot a bar chart comparing CPU and GPU times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = range(len(array_lengths))
    width = 0.35
    
    # Bar chart
    bars1 = ax1.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Array Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('CPU vs GPU Time for Reduction', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in array_lengths], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Speedup line chart
    ax2.plot(array_lengths, [1.0] * len(array_lengths), 'k--', alpha=0.5, label='1x (no speedup)')
    ax2.plot(array_lengths, speedups, 'o-', color='#F18F01', linewidth=2, markersize=8, label='Speedup')
    
    ax2.set_xlabel('Array Length', fontsize=12)
    ax2.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
    ax2.set_title('GPU Speedup vs Array Length', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on speedup points
    for i, (length, speedup) in enumerate(zip(array_lengths, speedups)):
        ax2.annotate(f'{speedup:.2f}x', 
                    (length, speedup),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()

def main():
    csv_filename = 'reduction_timing.csv'
    
    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found.", file=sys.stderr)
        print("Please run generate_timing_data.py first to generate the timing data.", file=sys.stderr)
        sys.exit(1)
    
    array_lengths, cpu_times, gpu_times, speedups = load_timing_csv(csv_filename)
    
    png_filename = 'reduction_timing.png'
    plot_timing_bar_chart(array_lengths, cpu_times, gpu_times, speedups, png_filename)
    
    # Also create a summary
    print("\nSummary:")
    print(f"{'Array Length':<15} {'CPU Time (ms)':<15} {'GPU Time (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for length, cpu, gpu, speedup in zip(array_lengths, cpu_times, gpu_times, speedups):
        print(f"{length:<15} {cpu:<15.3f} {gpu:<15.3f} {speedup:<10.2f}x")

if __name__ == '__main__':
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        main()
    except ImportError:
        print("Error: matplotlib is not installed. Install it with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

