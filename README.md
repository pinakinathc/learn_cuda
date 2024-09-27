This repo is for me to learn C/C++/CUDA/cuBLAS

Few stuff:

- My objective is to avoid writing custom cuda kernels. Instead use cuBLAS as much as possible.
- Only some stuff will be custom. Like Softsplat.
- It is unlikely I will entertain PRs (unless I have made some mistakes). If you find this repo useful and want to build a community project -- Fork it and build on your fork.

Thanks to @karpathy and ChatGPT team.

### Environment Setup via Docker:

How to run Docker guide (written by me, you can also find other good sources): [https://www.pinakinathc.me/containers-tutorial/](https://www.pinakinathc.me/containers-tutorial/)

Make sure that NVIDIA Docker runtime is properly installed and configured on your machine since we will use `--gpus all` flag. This requires the `nvidia-container-toolkit` to be installed.

```
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
sketchx@sketchx2:~$ docker image ls
REPOSITORY         TAG                               IMAGE ID       CREATED         SIZE
extension-script   latest                            4006e7a901a1   4 months ago    4.45GB
nvidia/cuda        12.4.1-cudnn-devel-ubuntu22.04    edd3b6bf59a6   5 months ago    8.29GB
nvidia/cuda        11.7.1-cudnn8-devel-ubuntu22.04   1256fa5b1b7d   10 months ago   7.69GB
nvidia/cuda        11.8.0-cudnn8-devel-ubuntu22.04   d0117ee15b5f   10 months ago   9.74GB
ubuntu             xenial                            b6f507652425   3 years ago     135MB
sketchx@sketchx2:~$ 

sketchx@sketchx2:~$ export IMAGE_ID=edd3b6bf59a6
sketchx@sketchx2:~$ export PATH_TO_CODEBASE=$PWD
sketchx@sketchx2:~$ export CONTAINER_NAME=test_docker

sketchx@sketchx2:~$ docker run -it -v $PATH_TO_CODEBASE:/workspace --gpus all --name test_docker $IMAGE_ID
root@fc4d3d9c5810:~#
root@fc4d3d9c5810:~# exit
sketchx@sketchx2:~$ 
```

Restart container

```
sketchx@sketchx2:~$ docker container ls -a
CONTAINER ID   IMAGE                                        COMMAND                  CREATED        STATUS                     PORTS     NAMES
fc4d3d9c5810   nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04   "/opt/nvidia/nvidia_…"   2 months ago   Exited (0) 3 months ago              test_docker
852f155a2386   1256fa5b1b7d                                 "/opt/nvidia/nvidia_…"   3 months ago   Exited (255) 6 weeks ago             pinaki
6baf694fedd9   1256fa5b1b7d                                 "/opt/nvidia/nvidia_…"   3 months ago   Up 4 weeks                           frosty_turing
c90273dd8359   1256fa5b1b7d                                 "/opt/nvidia/nvidia_…"   3 months ago   Exited (0) 3 months ago              reverent_wright
sketchx@sketchx2:~$ 

sketchx@sketchx2:~$ docker start fc4d3d9c5810
sketchx@sketchx2:~$ docker container ls
CONTAINER ID   IMAGE                                        COMMAND                  CREATED        STATUS        PORTS     NAMES
fc4d3d9c5810   nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04   "/opt/nvidia/nvidia_…"   2 months ago   Up 32 hours             test_docker
6baf694fedd9   1256fa5b1b7d                                 "/opt/nvidia/nvidia_…"   3 months ago   Up 4 weeks              frosty_turing

sketchx@sketchx2:~$ docker attach fc4d3d9c5810
root@fc4d3d9c5810:~# 
root@fc4d3d9c5810:~# cd /workspace
root@fc4d3d9c5810:~# git clone https://github.com/pinakinathc/learn_cuda.git
root@fc4d3d9c5810:~# cd /workspace/learn_cuda
```

### How to build: 

`make <target>`

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

### Git Submodules

We also add a few third-party dependency projects (e.g., tiny-cuda-nn). Check `.gitmodules` for more information.

```
git submodule update --init --recursive
```

Example:

For tiny-cuda-nn do the following:

```
root@fc4d3d9c5810:/workspace/learn_cuda# git submodule add https://github.com/NVlabs/tiny-cuda-nn.git dependencies/tiny-cuda-nn
Cloning into '/workspace/learn_cuda/dependencies/tiny-cuda-nn'...
remote: Enumerating objects: 4098, done.
remote: Counting objects: 100% (1304/1304), done.
remote: Compressing objects: 100% (207/207), done.
remote: Total 4098 (delta 1121), reused 1153 (delta 1076), pack-reused 2794 (from 1)
Receiving objects: 100% (4098/4098), 19.61 MiB | 45.63 MiB/s, done.
Resolving deltas: 100% (2630/2630), done.
root@fc4d3d9c5810:/workspace/learn_cuda# git submodule update --init --recursive
Submodule 'dependencies/cutlass' (https://github.com/NVIDIA/cutlass) registered for path 'dependencies/tiny-cuda-nn/dependencies/cutlass'
Submodule 'dependencies/fmt' (https://github.com/fmtlib/fmt) registered for path 'dependencies/tiny-cuda-nn/dependencies/fmt'
Cloning into '/workspace/learn_cuda/dependencies/tiny-cuda-nn/dependencies/cutlass'...
Cloning into '/workspace/learn_cuda/dependencies/tiny-cuda-nn/dependencies/fmt'...
Submodule path 'dependencies/tiny-cuda-nn/dependencies/cutlass': checked out '1eb6355182a5124639ce9d3ff165732a94ed9a70'
Submodule path 'dependencies/tiny-cuda-nn/dependencies/fmt': checked out 'b0c8263cb26ea178d3a5df1b984e1a61ef578950'
root@fc4d3d9c5810:/workspace/learn_cuda# 
```