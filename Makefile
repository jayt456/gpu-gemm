NVCC      := nvcc
ARCH      ?= native          # override: make ARCH=sm_75 (T4) or sm_80 (A100)
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDFLAGS   := -lcublas

all: bin/gemm

bin/gemm: src/main.cu src/kernels.cuh src/common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) src/main.cu -o bin/gemm $(LDFLAGS)

run: bin/gemm
	./bin/gemm

clean:
	rm -rf bin