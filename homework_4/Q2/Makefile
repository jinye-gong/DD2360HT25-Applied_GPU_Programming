# Makefile for CUDA Heat Equation Solver with cuBLAS and cuSPARSE

# CUDA installation path (adjust if needed)
CUDA_INSTALL_PATH ?= /usr/local/cuda

# Compiler - try to find nvcc in PATH first, then use CUDA_INSTALL_PATH
NVCC = $(shell which nvcc 2>/dev/null || echo $(CUDA_INSTALL_PATH)/bin/nvcc)

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_75 -std=c++11

# Libraries
LIBS = -lcublas -lcusparse

# Target executable
TARGET = hw4-heat
SOURCE = hw4-heat-template.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all clean

