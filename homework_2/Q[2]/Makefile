NVCC = nvcc

CCBIN ?= gcc


NVCC_FLAGS = -arch=sm_50 -O2 -std=c++11 -ccbin $(CCBIN) -U_FORTIFY_SOURCE

NVCC_FLAGS += -Wno-deprecated-gpu-targets

TARGET = reduction

SOURCES = hw2_reduction_template.cu

OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

rebuild: clean all

info:
	@echo "NVCC: $(NVCC)"
	@echo "Host Compiler: $(CCBIN)"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	@$(NVCC) --version

.PHONY: all clean rebuild info

