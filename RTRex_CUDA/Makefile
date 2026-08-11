# RTRExtractor CUDA — Makefile
# Requires: CUDA toolkit (nvcc) + Escape library from RTRex_Sequential

ESCAPE_HOME := ../RTRex_Sequential

# Compilers
NVCC := nvcc
CXX  := g++

# Include paths
INCLUDES := -I. -I$(ESCAPE_HOME)

# CUDA architecture — configurable via CUDA_ARCH
# sm_75 = Turing (T4, RTX 20xx)
# sm_80 = Ampere (A100)
# sm_86 = Ampere (RTX 30xx)
# sm_89 = Ada Lovelace (RTX 40xx)
CUDA_ARCH ?= sm_86

# Flags
NVCC_FLAGS := -arch=$(CUDA_ARCH) -O3 -std=c++14 -lineinfo \
              --expt-relaxed-constexpr
CXX_FLAGS   := -Wall -std=c++11 -O3

# Link flags
LDFLAGS := -L$(ESCAPE_HOME) -lescape -lm -lcudart

# Source files
CU_SRCS := main.cu gpu_graph.cu gpu_triangle_count.cu gpu_clean.cu gpu_extract.cu
CU_OBJS := $(CU_SRCS:.cu=.o)

# Target
TARGET := RTRex_CUDA

# Escape library dependency
LIBESCAPE := $(ESCAPE_HOME)/libescape.a

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(CU_OBJS) $(LIBESCAPE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CU_OBJS) $(LDFLAGS)

%.o: %.cu gpu_common.cuh gpu_graph.cuh gpu_triangle_count.cuh gpu_clean.cuh gpu_extract.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(LIBESCAPE):
	$(MAKE) -C $(ESCAPE_HOME) libescape.a

clean:
	rm -f $(CU_OBJS) $(TARGET)
