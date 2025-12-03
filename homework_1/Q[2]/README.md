# CUDA Vector Addition

This project implements a CUDA-based vector addition program.

## Files

- `vecAdd.cu` - Main CUDA program with all required comments and analysis output
- `Makefile` - Build configuration
- `plot.py` - Python program to plot the stacked bar diagram.
- `batch_test.sh` - Bash script to run a batch of experiments and get the results saved  to a csv file.

## Building

```bash
make clean
make
```

## Running

```bash
# Default N=512
./vecAdd

# Custom vector length
./vecAdd 263149

# Run batch experiments
./batch_test.sh

# Plot the stacked bar according to the result given by `batch_test.sh`
python plot.py
```
