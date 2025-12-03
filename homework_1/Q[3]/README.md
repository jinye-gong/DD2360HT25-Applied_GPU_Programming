# CUDA Matrix Multiplication

This project implements a CUDA-based matrix multiplication program.

## Files

- `vecMult.cu` - Main CUDA program with DataType support (float/double) and analysis output
- `Makefile` - Build configuration with DataType support
- `plot.py` - Python program to plot the stacked bar diagram.
- `batch_test.sh` - Bash script to run a batch of experiments and get the results saved  to a csv file.

## Building

```bash
# Build with float (default)
make clean
make

# Build with double
make clean
make DATA_TYPE=double

# Or use targets
make float
make double
```

## Running

```bash
# Default: A(128×256) × B(256×32)
./vecMult

# Custom matrix dimensions: A(numARows×numAColumns) × B(numBRows×numBColumns)
./vecMult 128 256 256 32
./vecMult 1024 8191 8191 8197

# Run batch experiments
./batch_test.sh

# Plot the stacked bar according to the result given by `batch_test.sh`
python plot.py
```

## Requirements

- CUDA toolkit
- GCC compiler
- NVIDIA GPU (for execution)
- NVIDIA Nsight Compute (for Questions 3 and 6)

## DataType Support

The program supports both `float` and `double` DataTypes:

- **Default:** `float` (4 bytes)
- **Compile with double:** `make DATA_TYPE=double` or `make double`
- **Runtime:** The DataType is determined at compile time
