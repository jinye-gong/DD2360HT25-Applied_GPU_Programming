#!/bin/bash

DATA_TYPE=${1:-float}
OUTPUT_FILE="timing_results_${DATA_TYPE}.csv"

echo "numARows,numAColumns,numBRows,numBColumns,timeH2D,timeKernel,timeD2H,totalTime,DataType" > $OUTPUT_FILE


MATRIX_SIZES=(
    "32 64 64 8"
    "64 128 128 16"
    "128 256 256 32"
    "256 512 512 64"
    "512 1024 1024 128"
    "1024 2048 2048 256"
    "2048 4096 4096 2048"
    "8192 1024 1024 8192"
    "2048 8192 8192 2048"

)

echo "Running batch tests with DataType=$DATA_TYPE..."
echo "Building with DataType=$DATA_TYPE..."

make clean
make DATA_TYPE=$DATA_TYPE

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Testing matrices..."
for matrix_size in "${MATRIX_SIZES[@]}"; do
    echo "Testing: A(${matrix_size})"
    ./vecMult $matrix_size 2>&1 | grep "CSV_OUTPUT" | sed 's/CSV_OUTPUT,//' >> $OUTPUT_FILE
done

echo "Results saved to $OUTPUT_FILE"

