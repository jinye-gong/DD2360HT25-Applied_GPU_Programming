# **Shared Memory and Atomics**

This project contains the histogram implementation using CUDA with shared memory optimization.

## Files

- `hw2_histogram_template.cu` - Main CUDA program.
- `Makefile` - Build configuration.
- `generate_histogram_data.py` - Python program to generate histogram data with different array lengths and distribution types.
- `plot_histograms.py` - Python program to plot the bar chart.

## Building

```bash
make
```

## Running

```bash
./histogram <input_length> [distribution_type]
```

Where:
- `input_length`: Number of elements in the input array
- `distribution_type`: 0 for uniform distribution, 1 for normal distribution (default: 0)

Example:
```bash
./histogram 1024 0    # Uniform distribution
./histogram 1024 1    # Normal distribution
```

## Generating Histogram Data for Plotting

To generate CSV files with histogram data for the required array lengths:

```bash
python3 generate_histogram_data.py
```

This will create CSV files for:
- Array lengths: 1024, 10240, 102400, 1024000
- Distributions: uniform and normal

## Plotting Histograms

After generating the CSV files, create bar charts:

```bash
python3 plot_histograms.py
```

This will generate PNG files for each histogram configuration.
