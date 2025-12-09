NVCC = nvcc

CCBIN ?= gcc

NVCC_FLAGS = -arch=sm_80 -O2 -std=c++11 -ccbin $(CCBIN) -U_FORTIFY_SOURCE
NVCC_FLAGS += -Wno-deprecated-gpu-targets

# 同时编译两个目标
TARGETS =vecAdd_streams

SOURCES = vecAdd_streams.cu
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGETS)

# 通用链接规则：vecAdd_streams 由 vecAdd_streams.o 链接
%: %.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# .cu -> .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJECTS)

rebuild: clean all

info:
	@echo "NVCC: $(NVCC)"
	@echo "Host Compiler: $(CCBIN)"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	@$(NVCC) --version

.PHONY: all clean rebuild info
