.PHONY: all build run

# Path to your DPC++ clang++
DPCPP ?= $(HOME)/IRIS-GNN-WS/sycl-implementations/talos/dpcpp-cpu-cuda/bin/clang++

# Configure only once; subsequent `make` just runs ninja
build:
	@if [ ! -d build ]; then \
	  cmake -B build -GNinja -DCMAKE_CXX_COMPILER=$(DPCPP); \
	fi
	ninja -C build

all: build

run: build
	ONEAPI_DEVICE_SELECTOR="cuda:0,cuda:1,host" \
	 ./build/apps/spmv_cli/spmv_cli assets/matrix.mtx
