#!/bin/bash

# Script to run experiments for Heat Equation Solver
# This script runs all the experiments mentioned in the assignment

EXECUTABLE="./hw4-heat"
OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Heat Equation Solver - Experiment Runner"
echo "=========================================="
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first using 'make'"
    exit 1
fi

# Experiment 1: Different dimX values and FLOPS calculation
echo "=========================================="
echo "Experiment 1: FLOPS Analysis (varying dimX)"
echo "=========================================="
echo "dimX,nzv,time_us,iterations,FLOPS" > $OUTPUT_DIR/flops_analysis.csv

# Test different dimX values
for dimX in 256 512 1024 2048 4096 8192; do
    echo "Running with dimX=$dimX..."
    
    # Calculate number of non-zero values
    nzv=$((3 * dimX - 6))
    
    # Run and capture output
    output=$($EXECUTABLE $dimX 1000 2>&1)
    
    # Extract timing and iteration information
    iteration_time=$(echo "$output" | grep "ITERATION_TIME_US:" | awk '{print $2}')
    actual_iterations=$(echo "$output" | grep "ACTUAL_ITERATIONS:" | awk '{print $2}')
    
    if [ -z "$iteration_time" ] || [ -z "$actual_iterations" ]; then
        echo "  Warning: Could not extract timing information for dimX=$dimX"
        continue
    fi
    
    # Calculate FLOPS for SpMV
    # For a tridiagonal matrix SpMV: approximately 2*nzv operations per iteration
    # (each non-zero requires 1 multiply and 1 add)
    total_ops=$(echo "scale=0; 2 * $nzv * $actual_iterations" | bc)
    flops=$(echo "scale=2; $total_ops * 1000000 / $iteration_time" | bc)
    
    echo "$dimX,$nzv,$iteration_time,$actual_iterations,$flops" >> $OUTPUT_DIR/flops_analysis.csv
    echo "  dimX=$dimX, nzv=$nzv, time=${iteration_time}us, iterations=$actual_iterations, FLOPS=$flops"
done

echo ""
echo "Results saved to $OUTPUT_DIR/flops_analysis.csv"
echo ""

# Experiment 2: Vary nsteps with dimX=1024
echo "=========================================="
echo "Experiment 2: Relative Error vs nsteps (dimX=1024)"
echo "=========================================="
echo "nsteps,relative_error" > $OUTPUT_DIR/error_vs_nsteps.csv

dimX=1024
for nsteps in 100 200 500 1000 2000 5000 10000; do
    echo "Running with dimX=$dimX, nsteps=$nsteps..."
    
    # Extract relative error from output
    output=$($EXECUTABLE $dimX $nsteps 2>&1)
    error=$(echo "$output" | grep "relative error" | awk '{print $NF}')
    
    if [ ! -z "$error" ]; then
        echo "$nsteps,$error" >> $OUTPUT_DIR/error_vs_nsteps.csv
        echo "  nsteps=$nsteps, error=$error"
    else
        echo "  Warning: Could not extract error for nsteps=$nsteps"
    fi
done

echo ""
echo "Results saved to $OUTPUT_DIR/error_vs_nsteps.csv"
echo ""

# Experiment 3: Performance with and without prefetching
echo "=========================================="
echo "Experiment 3: Prefetching Performance Comparison"
echo "=========================================="
echo "dimX,prefetch_enabled,time_us,speedup" > $OUTPUT_DIR/prefetch_comparison.csv

dimX=1024
nsteps=1000

echo "Testing with dimX=$dimX, nsteps=$nsteps..."

# With prefetching (default)
echo "  Running with prefetching enabled..."
output_prefetch=$($EXECUTABLE $dimX $nsteps 2>&1)
time_prefetch=$(echo "$output_prefetch" | grep "ITERATION_TIME_US:" | awk '{print $2}')

# Without prefetching
echo "  Running with prefetching disabled..."
output_no_prefetch=$($EXECUTABLE $dimX $nsteps --no-prefetch 2>&1)
time_no_prefetch=$(echo "$output_no_prefetch" | grep "ITERATION_TIME_US:" | awk '{print $2}')

if [ ! -z "$time_prefetch" ] && [ ! -z "$time_no_prefetch" ]; then
    speedup=$(echo "scale=2; $time_no_prefetch / $time_prefetch" | bc)
    echo "$dimX,true,$time_prefetch,1.00" >> $OUTPUT_DIR/prefetch_comparison.csv
    echo "$dimX,false,$time_no_prefetch,$speedup" >> $OUTPUT_DIR/prefetch_comparison.csv
    echo "  With prefetch:    ${time_prefetch}us"
    echo "  Without prefetch: ${time_no_prefetch}us"
    echo "  Speedup: ${speedup}x"
else
    echo "  Warning: Could not extract timing information"
fi

echo ""
echo "Results saved to $OUTPUT_DIR/prefetch_comparison.csv"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "Results are in the $OUTPUT_DIR/ directory"
echo ""
echo "To generate plots, run: python3 plot_results.py"
echo "=========================================="
