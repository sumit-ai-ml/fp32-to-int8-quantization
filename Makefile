# Makefile for fp32_to_int8_cuda.cu
# Uses conda cuda_build env (conda create -n cuda_build -c nvidia/label/cuda-12.8.0 cuda-toolkit)

CUDA_HOME := $(HOME)/miniconda3/envs/cuda_build
NVCC      := $(CUDA_HOME)/bin/nvcc
CUDA_INC  := $(CUDA_HOME)/targets/x86_64-linux/include
CUDA_LIB  := $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_LIB2 := $(CUDA_HOME)/lib

# GPU architecture (RTX A3000 = sm_86)
ARCH := sm_86

TARGET := fp32_to_int8_cuda
SRC    := fp32_to_int8_cuda.cu

NVCC_FLAGS := -O3 -arch=$(ARCH) \
              -I$(CUDA_INC) \
              -L$(CUDA_LIB) -L$(CUDA_LIB2) \
              -lcublas -lcublasLt -lcudart

RUNTIME_LIBS := $(CUDA_LIB):$(CUDA_LIB2)

.PHONY: all clean run info benchmark plots dashboard test

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	LD_LIBRARY_PATH=$(RUNTIME_LIBS):$$LD_LIBRARY_PATH ./$(TARGET)

clean:
	rm -f $(TARGET)

benchmark: $(TARGET)
	python run_all_benchmarks.py

plots: benchmark
	python generate_plots.py

dashboard: benchmark
	python generate_dashboard.py

test:
	python -m pytest test_benchmarks.py -v

info:
	@echo "NVCC:      $(NVCC)"
	@echo "CUDA_INC:  $(CUDA_INC)"
	@echo "CUDA_LIB:  $(CUDA_LIB)"
	@echo "CUDA_LIB2: $(CUDA_LIB2)"
	@echo "ARCH:      $(ARCH)"
