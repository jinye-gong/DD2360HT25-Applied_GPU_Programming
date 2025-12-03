# **Tiled Matrix Multiplication**

This project implements a 2D Dense Matrix Multiplication with tiled version.

## Files

- `vecMult.cu` - Main CUDA program.
- `Makefile` - Build configuration.
- `batch_test.sh` - Bash script to generate data.
- `plot.py` - Python program to plot the chart.

## Building

```bash
make
```

## Running

```bash
./vecMult <numARows> <numAColumns> <numBRows> <numBColumns>
```

