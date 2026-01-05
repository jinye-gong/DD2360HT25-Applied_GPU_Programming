#!/usr/bin/env bash
set -euo pipefail

BIN="./con"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="run_${TS}.log"
CSVFILE="run_${TS}.csv"

# 要扫的 N 和 blockSize（按需改）
Ns=(1024 2048 4096 8192 16384 32768 65536 131072)
BSs=(32 64 128 256 512 1024)

if [[ ! -x "$BIN" ]]; then
  echo "Error: cannot execute $BIN"
  echo "Build it first, e.g.: nvcc hw4-convolution.cu -o con"
  exit 1
fi

echo "Running sweep: BIN=$BIN" | tee "$LOGFILE"
echo "Ns:  ${Ns[*]}" | tee -a "$LOGFILE"
echo "BSs: ${BSs[*]}" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"
echo "CSV: $CSVFILE" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# CSV header（逗号分隔，Excel 直接打开）
echo "N,blockSize,basic_avg_ms,tiled_avg_ms,speedup_basic_over_tiled" > "$CSVFILE"

for N in "${Ns[@]}"; do
  for BS in "${BSs[@]}"; do
    echo "" | tee -a "$LOGFILE"
    echo "==================== N=$N, blockSize=$BS ====================" | tee -a "$LOGFILE"

    out="$("$BIN" "$N" "$BS" 2>&1 | tee -a "$LOGFILE")"

    # 抓 RESULT 行：RESULT,N,blockSize,basic_avg_ms,tiled_avg_ms
    result_line="$(echo "$out" | awk -F',' '/^RESULT,/{print $0}' | tail -n 1)"

    if [[ -z "$result_line" ]]; then
      echo "Warning: RESULT line not found for N=$N BS=$BS" | tee -a "$LOGFILE"
      echo "$N,$BS,NA,NA,NA" >> "$CSVFILE"
      continue
    fi

    # 解析字段
    # shellcheck disable=SC2206
    IFS=',' read -r tag rN rBS basic_ms tiled_ms <<< "$result_line"

    # 计算 speedup = basic/tiled
    speedup="$(awk -v b="$basic_ms" -v t="$tiled_ms" 'BEGIN{ if(t>0) printf("%.6f", b/t); else print "NA"}')"

    echo "$rN,$rBS,$basic_ms,$tiled_ms,$speedup" | tee -a "$CSVFILE"
  done
done

echo "" | tee -a "$LOGFILE"
echo "Done. Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "Done. CSV saved to: $CSVFILE" | tee -a "$LOGFILE"
