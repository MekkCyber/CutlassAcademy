# CUTLASS Academy

## What is CUTLASS?

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is a collection of CUDA C++ templates and abstractions for implementing high-performance matrix-multiplication and related computations at all levels and scales within CUDA. CUTLASS provides:

- Threadblock-level abstractions for matrix multiply-accumulate operations
- Warp-level primitives for matrix multiply-accumulate operations
- Epilogue components for various activation functions and tensor operations
- Utilities for efficiently loading and storing tensors in memory

CUTLASS is designed to deliver high performance for deep learning and HPC applications, with a focus on matrix multiplication operations that are fundamental to neural networks.

## What is CUTE?

CUTE (CUDA Template Library for Tensors) is a modern C++ library built on top of CUTLASS that provides a more flexible and composable approach to tensor operations. CUTE introduces:

- A unified tensor abstraction that works across different hardware levels
- Powerful layout mapping capabilities for tensors
- Composable building blocks for tensor algorithms
- A more intuitive programming model for complex tensor operations

CUTE was introduced in CUTLASS 3.0 and represents a significant evolution in NVIDIA's approach to tensor computing.

## How do CUTLASS, CUTE, and CUDA relate?

- **CUDA** is the base programming model and platform for NVIDIA GPUs. It provides the fundamental parallel computing architecture and programming interface.
- **CUTLASS** is a library built on top of CUDA that provides optimized implementations of matrix operations.
- **CUTE** is a higher-level abstraction built on top of CUTLASS that simplifies tensor programming.

## Key Differences

| Feature | CUDA | CUTLASS | CUTE |
|---------|------|---------|------|
| Level of Abstraction | Low-level GPU programming | Matrix operation templates | High-level tensor abstractions |
| Focus | General GPU computing | Matrix multiplication primitives | Flexible tensor operations |
| Programming Model | Explicit thread/block management | Threadblock/warp abstractions | Layout-focused tensor abstractions |
| Optimization Control | Manual | Template-based | Layout-driven |


# Resources

## Documentation

### CUTLASS Docs
- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS Wiki Documentation](https://github.com/NVIDIA/cutlass/wiki/Documentation) (great markdowns with examples and images)
- [CUTLASS Documentation](https://nvidia.github.io/cutlass/index.html)

### CUTE Docs
- [CUTE Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [CUTE Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)

### GTC
- [CUTLASS: CUDA TEMPLATE LIBRARY FOR DENSE LINEAR ALGEBRA AT ALL LEVELS AND SCALES (GTC 2018)](./s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)

- [PROGRAMMING TENSOR CORES: NATIVE VOLTA TENSOR CORES WITH CUTLASS (GTC 2019)](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)

- [Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100 (GTC 2020)](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)

- [Accelerating Convolution with Tensor Cores in CUTLASS (GTC 2021)](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/)

- [Accelerating Backward Data Gradient by Increasing Tensor Core Utilization in CUTLASS (GTC 2022)](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)

- [CUTLASS: Python API, Enhancements, and NVIDIA Hopper (GTC 2022)](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/)

- [Developing Optimal CUDA Kernels on Hopper Tensor Cores (GTC 2023)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)

- [CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores (GTC 2024)](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

GTC 2025 Coming soon (Cutlass in python coming soon !)

### Articles
**PyTorch**
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)  

**Nvidia**
- [Implementing High Performance Matrix Multiplication Using CUTLASS v2.8](https://developer.nvidia.com/blog/implementing-high-performance-matrix-multiplication-using-cutlass-v2-8/)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) 

**Colfax**
- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/)
- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)  

**Miscellaneous**
- [Build and Develop CUTLASS CUDA Kernels](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/) (How to create a CUDA Docker container for CUTLASS kernel development)
- [learn-cutlass](https://gty111.github.io/2023/03/21/learn-cutlass-1/)

### Videos
- [Lecture 15: CUTLASS (GPU MODE)](https://www.youtube.com/watch?v=G6q719ck7ww&ab_channel=GPUMODE)
- [CUTLASS: A CUDA C++ Template Library for Accelerating Deep Learning Computations (The Linux Foundation)](https://www.youtube.com/watch?v=PWWOGrLZtZg&ab_channel=TheLinuxFoundation)
- [Lecture 36: CUTLASS and Flash Attention 3 (GPU MODE)](https://www.youtube.com/watch?v=JwUcZwPOCpA&t=2831s&ab_channel=GPUMODE)
- [GTC 2024 : CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

### Repos using CUTLASS/CUTE
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)















