# Makefile for Tensor Core Matrix Multiplication

# CUDA compiler
NVCC = $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Compiler flags for Tensor Core support
# sm_75 = Turing (T4), sm_80 = Ampere, sm_86 = Ampere (consumer)
NVCCFLAGS = -O3 -arch=sm_75 -std=c++14

# Target executable
TARGET = vecMult
SOURCE = vecMult.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all clean

