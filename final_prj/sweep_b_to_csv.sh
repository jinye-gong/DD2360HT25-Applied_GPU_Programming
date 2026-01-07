#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Sweep batch size (B) for 3 CUDA GPT-2 training implementations
# and collect throughput into a CSV.
#
# Run from your repo root so llmc/ headers + data/model files are found.
#
# Examples:
#   chmod +x sweep_b_to_csv.sh
#   ./sweep_b_to_csv.sh
#
#   B_LIST="1 2 4 8 16 32 64" T=256 REPEATS=3 ./sweep_b_to_csv.sh
#
#   # use custom data
#   TRAIN_DATA=dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
#   VAL_DATA=dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
#   ./sweep_b_to_csv.sh
#
# Output:
#   results_batch_sweep.csv
#   logs/*.log
# ============================================================

# ---- Sources (3 implementations)
SRC_BASE=${SRC_BASE:-train_gpt2_fp32.cu}
SRC_U1=${SRC_U1:-train_gpt2_fp32_update1.cu}
SRC_U2=${SRC_U2:-train_gpt2_fp32_update2.cu}

# ---- Binaries
BIN_BASE=${BIN_BASE:-build/train_base}
BIN_U1=${BIN_U1:-build/train_u1}
BIN_U2=${BIN_U2:-build/train_u2}

# ---- Data/model assets
TRAIN_DATA=${TRAIN_DATA:-dev/data/tinyshakespeare/tiny_shakespeare_train.bin}
VAL_DATA=${VAL_DATA:-dev/data/tinyshakespeare/tiny_shakespeare_val.bin}
CKPT=${CKPT:-gpt2_124M.bin}
TOKENIZER=${TOKENIZER:-gpt2_tokenizer.bin}

# ---- Sweep params
T=${T:-256}
B_LIST=${B_LIST:-"16 32"}
REPEATS=${REPEATS:-1}

# ---- Keep validation/sampling overhead minimal (cannot fully disable due to code)
# step 0 always evaluates val, and the final step always evaluates val + samples.
# These are NOT included in iteration timing, but they do add wall-time.
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-1000000000}
VAL_MAX_STEPS=${VAL_MAX_STEPS:-1}
SAMPLE_EVERY=${SAMPLE_EVERY:-1000000000}
GEN_T=${GEN_T:-1}

# ---- Build options
NVCC=${NVCC:-nvcc}
CXXFLAGS=${CXXFLAGS:-"-O3 -std=c++17"}
INCLUDES=${INCLUDES:-"-I."}
LIBS=${LIBS:-"-lcublas"}
GENCODE=${GENCODE:-""}  # e.g. "-gencode arch=compute_80,code=sm_80"

# ---- Output
CSV=${CSV:-results_batch_sweep.csv}

mkdir -p build logs

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[error] missing command: $1" >&2; exit 1; }
}

need_file() {
  [[ -f "$1" ]] || { echo "[error] missing file: $1" >&2; exit 1; }
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf "%s" "$s"
}

# Extract value column from the code's ASCII table.
# Example line: | num_layers L          | 12                                               |
# Usage: table_value "num_layers L" logfile

table_value() {
  local key="$1"
  local file="$2"
  local line
  line=$(grep -m1 "| ${key}" "$file" || true)
  [[ -n "$line" ]] || { echo "[error] key not found in log: ${key}" >&2; return 1; }
  trim "$(echo "$line" | awk -F'|' '{print $3}')"
}

build_one() {
  local src="$1"
  local bin="$2"
  echo "[build] $src -> $bin"
  need_file "$src"
  "$NVCC" $CXXFLAGS $GENCODE $INCLUDES "$src" -o "$bin" $LIBS
}

mean_ms() {
  python3 - "$@" <<'PY'
import sys
xs = [float(x) for x in sys.argv[1:]]
print(sum(xs)/len(xs))
PY
}

# Compute tokens/s and (approx) TFLOPs/s from config + timing
# FLOPs model: GEMM-dominant approximation
compute_metrics() {
  # args: B T L C Vp avg_ms
  python3 - "$@" <<'PY'
import sys
B  = int(sys.argv[1])
T  = int(sys.argv[2])
L  = int(sys.argv[3])
C  = int(sys.argv[4])
Vp = int(sys.argv[5])
avg_ms = float(sys.argv[6])

avg_s = avg_ms / 1000.0

tok_s = (B*T) / avg_s

# per-layer FLOPs (approx): linear projections + attention matmuls
flops_layer = 24*B*T*(C*C) + 4*B*(T*T)*C
# final vocab projection
flops_head  = 2*B*T*C*Vp
# training ~= 3x forward
flops_iter  = 3*(L*flops_layer + flops_head)

tflops_s = flops_iter / avg_s / 1e12
print(f"{tok_s:.2f} {tflops_s:.2f}")
PY
}

run_one() {
  local impl="$1"
  local bin="$2"
  local B="$3"

  local ms_list=()
  local first_log=""

  for rep in $(seq 1 "$REPEATS"); do
    local log="logs/${impl}_B${B}_T${T}_r${rep}.log"
    echo "[run] impl=${impl} B=${B} T=${T} rep=${rep}"

    "$bin" \
      -i "$TRAIN_DATA" -j "$VAL_DATA" \
      -b "$B" -t "$T" \
      -v "$VAL_LOSS_EVERY" -m "$VAL_MAX_STEPS" \
      -s "$SAMPLE_EVERY" -g "$GEN_T" \
      >"$log" 2>&1

    [[ -n "$first_log" ]] || first_log="$log"

    local avg_ms
    avg_ms=$(grep -m1 "total average iteration time:" "$log" | awk '{print $(NF-1)}' || true)
    [[ -n "$avg_ms" ]] || { echo "[error] cannot parse avg_ms from $log" >&2; tail -n 60 "$log" >&2; exit 1; }
    ms_list+=("$avg_ms")
  done

  local avg_ms
  avg_ms=$(mean_ms "${ms_list[@]}")

  # model config from the printed table
  local device L C Vp TF32
  device=$(table_value "device" "$first_log")
  TF32=$(table_value "TF32" "$first_log" || echo "unknown")
  L=$(table_value "num_layers L" "$first_log")
  C=$(table_value "channels C" "$first_log")
  Vp=$(table_value "padded_vocab_size Vp" "$first_log")

  local tok_s tflops
  read -r tok_s tflops < <(compute_metrics "$B" "$T" "$L" "$C" "$Vp" "$avg_ms")

  # CSV row
  echo "${impl},${B},${T},${avg_ms},${tok_s},${tflops},${L},${C},${Vp},${TF32},${device}"
}

main() {
  need_cmd "$NVCC"
  need_cmd python3

  # if user set CKPT/TOKENIZER to non-default paths, symlink to expected names
  need_file "$CKPT"
  need_file "$TOKENIZER"
  need_file "$TRAIN_DATA"
  need_file "$VAL_DATA"

  if [[ "$CKPT" != "gpt2_124M.bin" ]]; then
    ln -sf "$CKPT" gpt2_124M.bin
  fi
  if [[ "$TOKENIZER" != "gpt2_tokenizer.bin" ]]; then
    ln -sf "$TOKENIZER" gpt2_tokenizer.bin
  fi

  build_one "$SRC_BASE" "$BIN_BASE"
  build_one "$SRC_U1" "$BIN_U1"
  build_one "$SRC_U2" "$BIN_U2"

  echo "impl,B,T,avg_ms,tokens_per_s,approx_TFLOPs_s,L,C,Vp,TF32,device" > "$CSV"

  for B in $B_LIST; do
    run_one "base"   "$BIN_BASE" "$B" >> "$CSV"
    run_one "update1" "$BIN_U1"   "$B" >> "$CSV"
    run_one "update2" "$BIN_U2"   "$B" >> "$CSV"
  done

  echo
  echo "Saved CSV: $CSV"
  if command -v column >/dev/null 2>&1; then
    echo
    column -t -s, "$CSV"
  fi
}

main "$@"
