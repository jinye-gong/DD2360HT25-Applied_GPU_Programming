#!/usr/bin/env bash

N=10000000   # 1e7

S_segs=(
  10000
  50000
  100000
  200000
  500000
  1000000
  2000000
)

OUT_CSV="results_vs_Sseg.csv"

echo "Running batch test over S_seg with fixed N=${N}"
echo "Results will be saved to ${OUT_CSV}"

echo "N,S_seg,no_stream_ms,stream_ms,speedup" > "${OUT_CSV}"

for S_SEG in "${S_segs[@]}"; do
echo "-----------------------------------------"
echo "Running: N = ${N}, S_seg = ${S_SEG}"

OUTPUT=$(./vecAdd_streams "${N}" "${S_SEG}")

echo "${OUTPUT}"

no_stream_ms=$(echo "${OUTPUT}" | awk '/No-stream GPU time/ {print $4}')
stream_ms=$(echo "${OUTPUT}"    | awk '/4-stream GPU time/ {print $5}')
speedup=$(echo "${OUTPUT}"      | awk '/Speedup \(no-stream \/ streams\)/ {print $6}')

echo "${N},${S_SEG},${no_stream_ms},${stream_ms},${speedup}" >> "${OUT_CSV}"
done

echo "Done. See ${OUT_CSV} for results."
