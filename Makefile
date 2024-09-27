NVCC_FLAGS = --threads=0 -G -g --use_fast_math -std=c++17 -O3
NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml
NVCC_LDLIBS = 
NVCC_INCLUDES = 
NVCC_CUDNN =
# By default we don't build with cudnn because it increases compile time from a few seconds to ~minute
USE_CUDNN ?= 0

BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))

# Adding Git Submodules
USE_TINY_CUDA_NN ?= 1
TINY_CUDA_NN_DIRS = dependencies/tiny-cuda-nn

# Get your GPU's compute capability
GPU_COMPUTE_CAPABILITY = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]

NVCC := $(shell which nvcc)

ifeq ($(shell dpkg -l | grep -q nccl && echo "exists"), exists)
    $(info ✓ NCCL found, OK to train with multiple GPUs)
    NVCC_FLAGS += -DMULTI_GPU
    NVCC_LDLIBS += -lnccl
else
    $(info ✗ NCCL is not found, disabling multi-GPU support)
    $(info ---> On Linux you can try install NCCL with `sudo apt install libnccl2 libnccl-dev`)
endif

ifeq ($(USE_CUDNN), 1)
    NVCC_FLAGS += -DUSE_CUDNN
    NVCC_LDLIBS += -lcudnn
endif

ifeq ($(USE_TINY_CUDA_NN), 1)
    NVCC_INCLUDES += -I$(TINY_CUDA_NN_DIRS)/include -I$(TINY_CUDA_NN_DIRS)/dependencies 
    NVCC_INCLUDES += -I$(TINY_CUDA_NN_DIRS)/dependencies/cutlass/include 
    NVCC_INCLUDES += -I$(TINY_CUDA_NN_DIRS)/dependencies/cutlass/tools/util/include 
    NVCC_INCLUDES += -I$(TINY_CUDA_NN_DIRS)/dependencies/fmt/include
    NVCC_FLAGS += -DTCNN_MIN_GPU_ARCH=$(GPU_COMPUTE_CAPABILITY) --extended-lambda --expt-relaxed-constexpr
endif

matrix_mul_stack: matrix_mul_stack.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $(BUILD_DIR)/$@

matrix_mul_heap: matrix_mul_heap.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $(BUILD_DIR)/$@

matrix_mul_multiThread: matrix_mul_multiThread.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $(BUILD_DIR)/$@

img_hashgrid_tcnn: img_hashgrid_tcnn.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)