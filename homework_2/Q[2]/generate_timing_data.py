#!/usr/bin/env python3
"""
Script to generate timing data from the reduction program for different array lengths.
Run this to collect CPU and GPU timing data for plotting.
"""

import subprocess
import sys
import os
import csv
import re

def run_reduction(length):
    """Run the reduction program and capture output."""
    env = os.environ.copy()
    env['OUTPUT_CSV'] = '1'  # Enable CSV output
    cmd = ['./reduction', str(length)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running reduction: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return None

def parse_timing_output(output):
    """Parse reduction output to extract timing data from CSV section."""
    lines = output.split('\n')
    
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
            if line.strip() == 'array_length,cpu_time_ms,gpu_time_ms,speedup' or line.strip().startswith('array_length'):
                continue
            # Parse CSV line
            parts = line.strip().split(',')
            if len(parts) == 4:
                try:
                    return {
                        'array_length': int(parts[0]),
                        'cpu_time_ms': float(parts[1]),
                        'gpu_time_ms': float(parts[2]),
                        'speedup': float(parts[3])
                    }
                except ValueError:
                    continue
    
    return None

def main():
    # Array lengths: starting from 512, increasing at x2 rate for 10 runs
    array_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    
    print("Generating timing data for reduction program...")
    print("This will run the reduction program for each array length.\n")
    
    results = []
    
    for length in array_lengths:
        print(f"Running reduction for N={length}...", end=' ', flush=True)
        output = run_reduction(length)
        
        if output:
            data = parse_timing_output(output)
            if data:
                results.append(data)
                print(f"✓ CPU: {data['cpu_time_ms']:.3f}ms, GPU: {data['gpu_time_ms']:.3f}ms, Speedup: {data['speedup']:.2f}x")
            else:
                print(f"✗ Warning: Could not parse timing data from output")
        else:
            print(f"✗ Failed to run reduction program")
    
    # Save to CSV file
    if results:
        csv_filename = 'reduction_timing.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['array_length', 'cpu_time_ms', 'gpu_time_ms', 'speedup'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Saved timing data to {csv_filename}")
        print(f"  Total runs: {len(results)}")
    else:
        print("\n✗ No timing data collected")
        sys.exit(1)
    
    print("\nDone! Now run plot_timing.py to generate the bar chart.")

if __name__ == '__main__':
    main()


