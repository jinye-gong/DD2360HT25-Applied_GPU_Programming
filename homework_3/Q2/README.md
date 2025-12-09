# CUDA **CUDA Streams**

This project implements a CUDA Streams program.

## Files

- `vecAdd_streams.cu`  - Main CUDA program with all required comments and analysis output
- `Makefile` - Build configuration
- `plot_vs_N.py plot_vs_Sseg.py` - Python program to plot the stacked bar diagram.
- `batch_test_N.sh batch_test_Sseg.sh ` - Bash script to run a batch of experiments and get the results saved  to a csv file.

## Building

```bash
make clean
make
```

## Running

```bash
# Default  N = 1 << 24 	S_seg = 1 << 20 	nStreams = 4
./vecAdd_streams

# Custom vector length and segment
./vecAdd_streams 200000 20000

# Run batch experiments
./batch_test_N.sh
./batch_test_Sseg.sh


# Plot the stacked bar according to the result given by `batch_test.sh`
python plot_vs_N.py
python plot_vs_Sseg.py
```
