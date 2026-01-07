# DD2360 HT25 – GPU Projects

This repo contains five GPU programming projects for the course **DD2360 HT25**.

## Projects

### 1. Assignment I – GPU Architecture & CUDA Basics

Folder: `HW1/`

- Simple CUDA **vector addition** (CPU vs GPU)
- **2D dense matrix multiplication** in CUDA
- Basic **profiling** with Nsight (occupancy, timing)
- Small **Rodinia** benchmarks comparing CUDA vs OpenMP

### 2. Assignment II – Shared Memory, Atomics & Tiled GEMM

Folder: `HW2/`

- CUDA **histogram** using shared memory + atomics
- Parallel **reduction (sum)** in CUDA
- **Tiled matrix multiplication (GEMM)** with shared memory
- Performance comparison (different sizes, tile sizes, CPU vs GPU)

### 3. Assignment III – Thread Divergence & CUDA Streams

Folder: `HW3/`

- **Q1 – Thread divergence**
  - 2D CUDA kernel on images of different sizes with `blockDim = (64,16)`.
  - Compute total number of warps and how many of them diverge because of the boundary check.
  - Briefly explain which warps mix in-bounds / out-of-bounds threads.

- **Q2 – Vector addition with CUDA streams**
  - Baseline `vecAdd`: one H2D memcpy + one kernel + one D2H memcpy (no streams).
  - Streamed `vecAdd`: split the vector into segments and process them with 4 CUDA streams using async H2D / kernel / D2H to overlap copy and compute.
  - Measure total time, show small–medium speedup (~1.3× at best) vs. baseline, and study how different segment sizes affect performance (too small → overhead, too large → less overlap).

### 4. Assignment IV – CUDA Optimization, NVIDIA Libraries & Tensor Cores

Folder: `assignment_4/`

- **Q1 – 1D Convolution (Tiled Kernel)**
  - Implemented a basic 1D convolution and an optimized **shared-memory tiled kernel**.
  - Each block loads a tile of input data (with halo) into shared memory, then computes outputs after synchronization.
  - Tiling reduces redundant global memory reads and improves performance.
  - Tile size must balance warp alignment, synchronization overhead, and occupancy; **128–256** performs best in practice.
- **Q2 – NVIDIA Libraries & Unified Memory**
  - Solved the 1D heat equation using **Unified Memory**, **cuSPARSE (SpMV)**, and **cuBLAS (AXPY, NRM2)**.
  - FLOPS increase with problem size; small sizes are limited by launch overhead and low parallelism.
  - Increasing iteration steps reduces relative error and improves convergence.
  - **Unified Memory prefetching** reduces page migration overhead and gives a small speedup (~6%).
- **Bonus – Tensor Cores with WMMA**
  - Implemented GEMM using **WMMA** to utilize NVIDIA Tensor Cores.
  - Tensor Cores are used in all configurations; increasing warps per block improves utilization with diminishing returns.
  - WMMA achieves **~1.2×–1.35× speedup** over naive CUDA GEMM.
  - Performance improves significantly, at the cost of reduced numerical accuracy due to FP16 inputs.

### Final Project – Accelerating GPT-2 Training with Tensor Cores + Mixed Precision

- Built on **Andrej Karpathy’s llm.c**, a minimal CUDA GPT-2 training framework, we benchmark and optimize the GEMM (matrix multiplication) path.

We compare three implementations:

- **base**: handwritten FP32 CUDA kernels (no explicit Tensor Core usage)
- **update1**: replace all training GEMMs with **cuBLAS SGEMM**, allowing cuBLAS to use highly optimized Tensor Core/TF32 paths on supported GPUs (still FP32 I/O)
- **update2**: optimize only the first fully-connected (FC) layer using **cublasGemmEx** with selective mixed precision: FP16 inputs/weights + FP32 accumulation/output

Profiling shows the training workload is almost entirely **compute-bound**: about **99.5% of GPU time** is spent in kernel execution, and **matmul_forward_kernel4 accounts for ~70.5%**, so we focus on GEMM optimization.

Results: throughput increases with batch size; **update1 improves throughput by ~20%** across all batch sizes, while **update2 is ~5% faster than base** but **10–15% slower than update1**.

Quality check: compared to base, the final weights from update1/update2 are very close (**cosine similarity > 0.99995**, **rel_L2 ≈ 0.9%**, **max_abs < 0.084**), indicating only minor numerical differences.



## Repository Structure

```text
.
├── HW1/            # Assignment I code and local README
├── HW2/            # Assignment II code and local README
├── HW3/            # Assignment III code and local README
├── HW4/            # Assignment IV code and local README
├── final_prj/      # final project code and local README
└── README.md       # This file
```



## Academic integrity

All files in this repository are intended **only** for study and course examination at KTH. 

Please do **not** use this repository for plagiarism, and do **not** redistribute the solutions in ways that violate KTH’s rules on collaboration and academic honesty.

---





本仓库包含 DD2360 HT25 课程的五个 GPU 编程项目。

## 项目简介

### 1. 作业一 – GPU 架构与 CUDA 基础

目录：`HW1/`

- CUDA **向量加法**（CPU 对比 GPU）
- CUDA 实现的 **二维稠密矩阵乘法**
- 使用 Nsight 进行基本 **性能分析**（占用率、时间）
- 简单 **Rodinia 基准测试**，比较 CUDA 与 OpenMP

### 2. 作业二 – 共享内存、原子操作与 Tiled GEMM

目录：`HW2/`

- 使用共享内存和原子操作的 CUDA **直方图**
- CUDA **并行归约（求和）**
- 使用共享内存分块的 **矩阵乘法（Tiled GEMM）**
- 对不同规模和 tile 大小进行 **性能对比**（CPU vs GPU）

### 3. 作业三 – 线程发散与 CUDA Streams

目录：`HW3/`

- **问题 1：线程发射效率与发散**
  - 在不同尺寸的图像上，用 `blockDim = (64,16)` 的 2D kernel。
  - 计算网格中 warp 总数，以及因为边界判断导致发散的 warp 数。
  - 简要说明哪些 warp 中同时包含“在图像内/外”的线程，从而产生控制流发散。

- **问题 2：使用 CUDA Streams 的向量加法**
  - 基线版本：单次 H2D 拷贝 + 单次 kernel + 单次 D2H（无 stream）。
  - Streams 版本：将向量分段，用 4 个 CUDA streams 做异步 H2D / kernel / D2H，实现拷贝与计算重叠。
  - 比较总时间并给出加速比（最佳约 1.3×），同时分析不同 segment size 对性能的影响（太小开销大，太大重叠不足）。

### 4.  作业四 – CUDA 优化、NVIDIA 库与 Tensor Core

目录：`HW4/`

- **Q1 – 一维卷积（1D Convolution）**
  - 实现了基础版本和基于 **共享内存的 tiled kernel**。
  - 每个线程块先将输入数据（包含 halo）加载到 shared memory，再进行卷积计算。
  - Tiled 方法减少了全局内存的重复读取，提高了内存访问效率。
  - tile size 需要在 warp 对齐、同步开销和 occupancy 之间权衡，实验中 **128–256** 表现最好。
  
- **Q2 – NVIDIA 库与统一内存（Unified Memory）**
  - 使用 **Unified Memory（cudaMallocManaged）**，结合 **cuSPARSE（SpMV）** 和 **cuBLAS（AXPY、NRM2）** 求解一维热传导方程。
  - 随着问题规模增大，FLOPS 明显提升；小规模时受启动开销和并行度不足限制。
  - 随着迭代步数增加，解逐渐收敛，**相对误差单调下降**。
  - 使用 Unified Memory 预取（prefetch）可减少页面迁移开销，带来约 **6% 的性能提升**。
  
- **Bonus – Tensor Core（WMMA）**
  - 使用 **WMMA** 实现矩阵乘法，成功利用 NVIDIA Tensor Cores。
  - 不同线程块配置下均能使用 Tensor Core，增加每个 block 的 warp 数可提高利用率，但收益逐渐减小。
  - WMMA 相比普通 CUDA GEMM 有 **约 1.2×–1.35× 的加速**。
  - 由于使用 FP16 输入，性能提升的同时会带来一定的数值精度损失。
  
  

### 最终项目 – Tensor Core + Mixed Precision 加速 GPT-2 训练
目录：`final_prj/`

- 基于 **Karpathy 的 llm.c** 最小 CUDA GPT-2 训练框架，对 GEMM（矩阵乘）路径做加速对照实验
- 对比三种实现：
  - **base**：手写 FP32 CUDA kernel（不显式使用 Tensor Core）
  - **update1**：将训练中的 GEMM 全部替换为 **cuBLAS SGEMM**，在支持 GPU 上自动使用 Tensor Core/TF32 等高优化路径（仍保持 FP32 I/O）
  - **update2**：仅优化第一层（首个 FC），使用 **cublasGemmEx** 做选择性混合精度：FP16 输入/权重 + FP32 累加/输出
- Profiling 结果表明训练几乎是 **纯计算瓶颈**：约 **99.5% GPU 时间在 kernel 执行**，其中 **matmul_forward_kernel4 约占 70.5%**，因此重点优化 GEMM
- 性能结果：吞吐随 batch size 增大而上升；**update1 约提升 ~20% throughput**，**update2 约比 base 快 ~5%**，但比 update1 低 10–15%
- 质量检查：与 base 相比，update1/update2 最终权重差异很小（**cosine similarity > 0.99995**，**rel_L2 ≈ 0.9%**，**max_abs < 0.084**）





## 目录结构

```
.
├── HW1/            # 作业一代码及本地说明
├── HW2/            # 作业二代码及本地说明
├── HW3/            # 作业三代码及本地说明
├── HW4/            # 作业四代码及本地说明
├── final_prj/      # 最终项目代码及本地说明
└── README.md       # 本说明文件
```



## 使用说明与学术诚信

- 本仓库仅用于 DD2360 HT25 课程学习与作业展示。
- 请遵守 KTH 的学术诚信政策，不要将本仓库内容用于任何形式的抄袭或违规共享。
