#!/bin/bash

# Script to run Tensor Core experiments
# This script runs experiments for different matrix sizes

EXECUTABLE="./vecMult"
OUTPUT_DIR="results"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Tensor Core Matrix Multiplication - Experiments"
echo "=========================================="
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first using 'make'"
    exit 1
fi

# Experiment 1: Test with 1024x2048 x 2048x1024
echo "=========================================="
echo "Experiment 1: Matrix A(1024x2048) x B(2048x1024)"
echo "=========================================="
$EXECUTABLE 1024 2048 2048 1024 > $OUTPUT_DIR/exp1_1024x2048.txt 2>&1
cat $OUTPUT_DIR/exp1_1024x2048.txt
echo ""

# Experiment 2: Test with different square sizes
echo "=========================================="
echo "Experiment 2: Square matrices (A x B where A=B)"
echo "=========================================="
echo "size,cpu_time,gemm_time,tiled_8_time,tiled_16_time,tiled_32_time,wmma_time,wmma_max_error,wmma_avg_error" > $OUTPUT_DIR/performance_comparison.csv

for size in 512 1024 2048 4096 8192; do
    echo "Running with size=$size..."
    output=$($EXECUTABLE $size $size $size $size 2>&1)
    
    # Extract timing information
    # Note: timing is usually 2 lines after the result header
    cpu_time=$(echo "$output" | grep "CPU reference result" -A 2 | grep "timing" | awk '{print $2}' || echo "")
    gemm_time=$(echo "$output" | grep "CUDA gemm result" -A 3 | grep "timing" | awk '{print $2}' || echo "")
    tiled_8_time=$(echo "$output" | grep "tile \[8, 8\]" -A 3 | grep "timing" | awk '{print $2}' || echo "")
    tiled_16_time=$(echo "$output" | grep "tile \[16, 16\]" -A 3 | grep "timing" | awk '{print $2}' || echo "")
    tiled_32_time=$(echo "$output" | grep "tile \[32, 32\]" -A 3 | grep "timing" | awk '{print $2}' || echo "")
    wmma_time=$(echo "$output" | grep "WMMA kernel result" -A 6 | grep "timing" | awk '{print $2}' || echo "")
    wmma_max_error=$(echo "$output" | grep "WMMA kernel result" -A 6 | grep "max error vs CPU" | awk '{print $5}' || echo "")
    wmma_avg_error=$(echo "$output" | grep "WMMA kernel result" -A 6 | grep "avg error vs CPU" | awk '{print $5}' || echo "")
    
    echo "$size,$cpu_time,$gemm_time,$tiled_8_time,$tiled_16_time,$tiled_32_time,$wmma_time,$wmma_max_error,$wmma_avg_error" >> $OUTPUT_DIR/performance_comparison.csv
    echo "  Completed size=$size"
done

echo ""
echo "Results saved to $OUTPUT_DIR/performance_comparison.csv"
echo ""

# Experiment 3: Test with different thread block sizes for WMMA
echo "=========================================="
echo "Experiment 3: WMMA with different thread block configurations"
echo "=========================================="
echo "Note: This requires modifying the code to test different block sizes"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "Results are in the $OUTPUT_DIR/ directory"
echo ""
echo "To generate plots, run: python3 plot_results.py"
echo "=========================================="

