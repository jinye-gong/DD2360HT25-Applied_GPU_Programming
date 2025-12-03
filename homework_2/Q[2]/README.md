# **Reduction **

This project implements a reduction of 1D list of signed 32-bit floats.

## Files

- `hw2_reduction_template.cu` - Main CUDA program.
- `Makefile` - Build configuration.
- `generate_timing_data.py` - Python program to generate data.
- `plot_timing.py` - Python program to plot the chart.

## Building

```bash
make
```

## Running

```bash
./reduction <input_length>
```

Where:
- `input_length`: Number of elements in the input array (signed 32-bit floats)

Example:
```bash
./reduction 1024
./reduction 262144
```

## Generating Timing Data for Plotting

To generate CSV files with timing data for different array lengths (Question 6):

```bash
python3 generate_timing_data.py
```

This will create `reduction_timing.csv` with timing data for:
- Array lengths: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144

## Plotting Timing Results

After generating the CSV file, create bar charts:

```bash
python3 plot_timing.py
```

This will generate `reduction_timing.png` showing:
- Bar chart comparing CPU vs GPU time for each array length
- Line chart showing speedup vs array length
