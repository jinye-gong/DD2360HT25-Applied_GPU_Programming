# CUDA **NVIDIA Libraries and Unified Memory (Heat Equation)**

This project implements a 1D heat equation solver using **cuSPARSE (SpMV)**, **cuBLAS (AXPY/norm)**, and **Unified Memory** (optional UM prefetch).

## Files

- `hw4-heat-template.cu` - Main CUDA program (cuBLAS + cuSPARSE + Unified Memory)
- `Makefile` - Build configuration
- `run_experiments.sh` - Runs all required experiments and saves CSVs to `results/`
- `plot_results.py` - Generates plots from CSV results

## Building

```bash
make clean
make
```

## Running

```bash
# Usage: ./hw4-heat <dimX> <nsteps> [--no-prefetch]
./hw4-heat 1024 1000

# Disable Unified Memory prefetch (performance comparison)
./hw4-heat 1024 1000 --no-prefetch

# Run batch experiments (FLOPS vs dimX, error vs nsteps, prefetch comparison)
./run_experiments.sh

# Plot results (creates PNGs in results/)
python3 plot_results.py
```

