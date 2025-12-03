NVCC = nvcc


CCBIN ?= gcc


DATA_TYPE ?= float


NVCC_FLAGS = -arch=sm_50 -O2 -std=c++11 -ccbin $(CCBIN) -U_FORTIFY_SOURCE -DDATA_TYPE=$(DATA_TYPE)


NVCC_FLAGS += -Wno-deprecated-gpu-targets


TARGET = vecMult


SOURCES = vecMult.cu


OBJECTS = $(SOURCES:.cu=.o)


all: $(TARGET)


$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJECTS)


%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


clean:
	rm -f $(TARGET) $(OBJECTS)


rebuild: clean all


float: DATA_TYPE=float
float: $(TARGET)


double: DATA_TYPE=double
double: $(TARGET)


info:
	@echo "NVCC: $(NVCC)"
	@echo "Host Compiler: $(CCBIN)"
	@echo "DataType: $(DATA_TYPE)"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	@$(NVCC) --version

.PHONY: all clean rebuild info float double
