#!/bin/bash


OUTPUT_FILE="timing_results.csv"

echo "N,timeH2D,timeKernel,timeD2H,totalTime" > $OUTPUT_FILE

VECTOR_LENGTHS=(512 1024 2048 4096 8192 16384 32768 65536 131072 263149 524288 1048576 2097152 4194304 8388608 16777216)

echo "Running batch tests..."
for N in "${VECTOR_LENGTHS[@]}"; do
    echo "Testing N = $N"
    ./vecAdd $N 2>&1 | grep "CSV_OUTPUT" | sed 's/CSV_OUTPUT,//' >> $OUTPUT_FILE
done

echo "Results saved to $OUTPUT_FILE"




