#!/bin/bash

# 要测试的矩阵尺寸（方阵：A=NxN, B=NxN）
SIZES=(64 128 192 256 384 512 640 768 896 1024)

EXE=./vecMult
CSV=timing_results.csv

# 写 CSV 表头
echo "N,CPU_ms,Naive_ms,Tile8_ms,Tile16_ms,Tile32_ms" > "$CSV"

for N in "${SIZES[@]}"; do
    echo "Running N = $N ..."
    
    # 运行一次程序，把输出全部存到 OUT 变量
    OUT=$($EXE $N $N $N $N)

    # 从 OUT 里用 grep + awk 抠 timing 数字
    CPU=$(echo "$OUT"   | grep "CPU reference result"          -A1 | tail -n1 | awk '{print $2}')
    NAIVE=$(echo "$OUT" | grep "CUDA gemm result"               -A2 | grep "timing" | awk '{print $2}')
    T8=$(echo "$OUT"    | grep "CUDA tiled_gemm with tile \[8, 8\] result"   -A2 | grep "timing" | awk '{print $2}')
    T16=$(echo "$OUT"   | grep "CUDA tiled_gemm with tile \[16, 16\] result" -A2 | grep "timing" | awk '{print $2}')
    T32=$(echo "$OUT"   | grep "CUDA tiled_gemm with tile \[32, 32\] result" -A2 | grep "timing" | awk '{print $2}')

    # 追加一行到 CSV
    echo "$N,$CPU,$NAIVE,$T8,$T16,$T32" >> "$CSV"
done

echo "Done. CSV saved to $CSV"

