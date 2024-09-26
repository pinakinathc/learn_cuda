This repo is for me to learn C/C++/CUDA/cuBLAS

Few stuff:

- My objective is to avoid writing custom cuda kernels. Instead use cuBLAS as much as possible.
- Only some stuff will be custom. Like Softsplat.
- It is unlikely I will entertain PRs (unless I have made some mistakes). If you find this repo useful and want to build a community project -- Fork it and build on your fork.

Thanks to @karpathy and ChatGPT team.

How to build: `make <target>`

Example:

```
root@fc4d3d9c5810:/workspace/learn_cuda# make matrix_mul_stack
✓ NCCL found, OK to train with multiple GPUs
/usr/local/cuda/bin/nvcc --threads=0 -G -g --use_fast_math -std=c++17 -O3 --generate-code arch=compute_75,code=[compute_75,sm_75] -DMULTI_GPU matrix_mul_stack.cu -lcublas -lcublasLt -lnvidia-ml  -lnccl -o build/matrix_mul_stack
root@fc4d3d9c5810:/workspace/learn_cuda# 
```

This means you have successfully build a simple matrix multiplication using cuBLAS. Check your compiled files here:

```
root@fc4d3d9c5810:/workspace/learn_cuda# ls build/
matrix_mul_stack
root@fc4d3d9c5810:/workspace/learn_cuda#
```

Now that you have build, it is time to check/run your code.

```
root@fc4d3d9c5810:/workspace/learn_cuda# ./build/matrix_mul_stack 
Matrix C:
22.0000         28.0000 
49.0000         64.0000 
root@fc4d3d9c5810:/workspace/learn_cuda# 
```