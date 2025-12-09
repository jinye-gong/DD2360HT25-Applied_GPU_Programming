#!/usr/bin/env bash

Ns=(
  1000000
  2000000
  4000000
  8000000
  16000000
  32000000
  64000000
  128000000
  256000000

)

S_SEG=$((1 << 20))

OUT_CSV="results_vs_N.csv"

echo "Running batch test over N..."
echo "Results will be saved to ${OUT_CSV}"

echo "N,S_seg,no_stream_ms,stream_ms,speedup" > "${OUT_CSV}"

for N in "${Ns[@]}"; do
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
