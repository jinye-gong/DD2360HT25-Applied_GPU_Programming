#!/usr/bin/env python3
"""
Script to generate histogram data from the histogram program output.
Run this after executing the histogram program to create CSV files for plotting.
"""

import subprocess
import sys
import re
import csv
import os

def run_histogram(length, distribution_type):
    """Run the histogram program and capture output."""
    env = os.environ.copy()
    env['OUTPUT_CSV'] = '1'  # Enable CSV output
    cmd = ['./histogram', str(length), str(distribution_type)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running histogram: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return None

def parse_histogram_output(output):
    """Parse histogram output to extract bin counts from CSV section."""
    lines = output.split('\n')
    bins = []
    counts = []
    
    # Look for CSV_OUTPUT_START marker
    in_csv_section = False
    for line in lines:
        if '# CSV_OUTPUT_START' in line:
            in_csv_section = True
            continue
        if '# CSV_OUTPUT_END' in line:
            break
        if in_csv_section:
            # Skip header line
            if line.strip() == 'bin,count' or line.strip().startswith('bin'):
                continue
            # Parse CSV line
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    bins.append(int(parts[0]))
                    counts.append(int(parts[1]))
                except ValueError:
                    continue
    
    return bins, counts

def save_to_csv(bins, counts, filename):
    """Save histogram data to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin', 'count'])
        for bin_idx, count in zip(bins, counts):
            writer.writerow([bin_idx, count])

def main():
    array_lengths = [1024, 10240, 102400, 1024000]
    distribution_types = [0, 1]  # 0 = uniform, 1 = normal
    dist_names = ['uniform', 'normal']
    
    print("Generating histogram data for plotting...")
    print("This will run the histogram program for each configuration.\n")
    
    for length in array_lengths:
        for dist_type, dist_name in zip(distribution_types, dist_names):
            print(f"Running histogram for N={length}, distribution={dist_name}...")
            output = run_histogram(length, dist_type)
            
            if output:
                bins, counts = parse_histogram_output(output)
                if bins and counts:
                    filename = f'histogram_{length}_{dist_name}.csv'
                    save_to_csv(bins, counts, filename)
                    print(f"  ✓ Saved {len(bins)} bins to {filename}")
                else:
                    print(f"  ✗ Warning: Could not parse histogram data from output")
                    # Print first few lines of output for debugging
                    output_lines = output.split('\n')[:10]
                    print(f"  First lines of output:")
                    for line in output_lines:
                        print(f"    {line}")
            else:
                print(f"  ✗ Failed to run histogram program")
    
    print("\nDone! Now run plot_histograms.py to generate the plots.")

if __name__ == '__main__':
    main()

