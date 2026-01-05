# CUDA **Tensor Core Matrix Multiplication (WMMA)**

This project implements GEMM on GPU and adds a **WMMA** kernel to use **Tensor Cores** (FP16 input, FP32 accumulation).

## Files

- `vecMult.cu` - Main CUDA program (CPU reference, `gemm`, `tiled_gemm`, `wmma_gemm`)
- `Makefile` - Build configuration
- `run_experiments.sh` - Runs experiments and saves results to `results/`
- `plot_results.py` - Generates plots from CSV results

## Building

```bash
make clean
make
```

### Windows (CMD)

From `assignment4\bonus`:

```bat
build.bat
```

If your GPU is not `sm_75`, override the architecture first (examples):

```bat
set SM=86
build.bat
```

## Running

```bash
# Usage: ./vecMult <numARows> <numAColumns> <numBRows> <numBColumns>
./vecMult 1024 2048 2048 1024

# Default (no args): 512x512
./vecMult

# Run batch experiments (writes results/*.csv and results/*.txt)
./run_experiments.sh

# Plot results (creates PNGs in results/)
python3 plot_results.py
```

### WMMA threads-per-block (warps-per-block)

The WMMA kernel launch is parameterized as:

- `blockDim = (32, warpsPerBlock)`  →  `threadsPerBlock = 32 * warpsPerBlock`
- Each warp computes one 16×16 output tile; warps are stacked along tile-rows (Y).

You can change it via:

```bash
./vecMult 8192 8192 8192 8192 --wmma-warps 1
./vecMult 8192 8192 8192 8192 --wmma-warps 2
./vecMult 8192 8192 8192 8192 --wmma-warps 4
./vecMult 8192 8192 8192 8192 --wmma-warps 8
```

On Windows:

```bat
.\vecMult.exe 8192 8192 8192 8192 --wmma-warps 4
```

## Profiling Tensor Cores (Nsight Compute / ncu)

`nvprof` is deprecated (and not supported on newer GPUs). Use Nsight Compute CLI (`ncu`).

### Quick run (full set)

```bash
ncu --set full --kernel-name regex:wmma_gemm --target-processes all -- ./vecMult 8192 8192 8192 8192 --wmma-warps 4
```

### Collect explicit Tensor Core metrics (recommended for the report)

Metric availability varies by GPU + Nsight Compute version. If a metric is missing on your machine, list candidates with:

```bash
ncu --query-metrics | grep -i tensor
ncu --query-metrics | grep -i fp16
```

Example command (HMMA/Tensor + FP16 pipe + overall throughput):

```bash
ncu --kernel-name regex:wmma_gemm --target-processes all --csv --metrics ^
sm__inst_executed_pipe_tensor_op_hmma.sum,^
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,^
smsp__pipe_tensor_active.avg.pct_of_peak_sustained_active,^
smsp__pipe_fp16_active.avg.pct_of_peak_sustained_active,^
sm__throughput.avg.pct_of_peak_sustained_elapsed ^
-- ./vecMult 8192 8192 8192 8192 --wmma-warps 4
```

Notes:
- `sm__inst_executed_pipe_tensor_op_hmma.sum`: executed HMMA (Tensor Core) instructions.
- `sm__pipe_tensor_op_hmma_cycles_active...` / `smsp__pipe_tensor_active...`: Tensor pipe activity (“Tensor Active”-style).
- `smsp__pipe_fp16_active...`: a good proxy for “half_precision_fu_utilization”-style utilization in NCU terminology.


