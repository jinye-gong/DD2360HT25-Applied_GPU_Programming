# DD2360 HT25 – Assignment II: Shared Memory, Atomics and Tiled GEMM

This repository contains the code and data for **DD2360 HT25 – Assignment II**.  
The written report is in: `DD2360HT25_HW2_Group2.pdf`. 

Group contribution (as stated in the report):  
- **Shitong Guo** – 50% (Q[1], Q[2])  
- **Jinye Gong** – 50% (Q[2], Q[3]) 

## What we did

### Q[1] – Shared Memory and Atomics (Histogram)

- Implemented a CUDA histogram kernel using **per-block shared-memory histograms** plus a **global merge**.  
- Optimizations:  
  - Per-block shared-memory privatization + tree-style reduction.  
  - Parallel shared-memory initialization to avoid serial zeroing.  
  - Conditional global atomics (`if (s_bins[i] > 0)`) to skip empty bins. 
- Analyzed **global memory traffic**, number of **atomic operations**, and **shared memory usage**  
  (4096 bins → 16 KB shared memory per block). 
- Compared performance for **uniform vs normal distributions** and plotted histograms for array sizes  from 1,024 up to 1,024,000 elements, explaining why normal distribution causes more contention  
  (many values map to the same bins). 
- Measured GPU speedup over CPU and profiled **Achieved Occupancy** (~77.7% for N = 1,024,000). 

### Q[2] – Reduction

- Implemented a CUDA **sum reduction** using **block-level shared-memory tree reduction**.  
- Main ideas:  
  - Coalesced global loads, padding out-of-bounds threads with zero.  
  - One `atomicAdd` per block to accumulate partial sums globally. 
- Derived global memory traffic (N reads + a few global writes) and counted atomic ops  
  (equal to number of blocks). 
- Measured timing for N from 512 to 262,144:  
  - GPU slower for tiny N (launch + transfer overhead).  
  - Break-even around a few thousand elements.  
  - For large N, GPU achieves **10–70× speedup** over CPU. 
- Profiled shared memory usage (1 KB per block) and **Achieved Occupancy ~87%** at N = 262,144.  

### Q[3] – Tiled Matrix Multiplication (Tiled GEMM)

- Extended the matrix multiplication code with a **tiled GEMM kernel** using shared memory for tiles. 
- Derived formulas for total **global memory reads** for both naive and tiled GEMM and compared them. 
- Experiments:  
  - For **1024×1024 × 1024×1024** and **513×8192 × 8192×1023**,  
    all CUDA kernels match the CPU reference with small numerical error. 
  - Compared tile sizes **8×8**, **16×16**, **32×32**:  
    16×16 typically gives the best trade-off between data reuse, occupancy and overhead;  
    tiled kernels can be slightly slower than naive GEMM when cache & coalescing already work well. 
- Collected Nsight profiling data (block size, dynamic shared memory per block, Achieved Occupancy)  
  for different tile sizes and matrix shapes.
- Ran 8–10 different matrix sizes and plotted CPU vs GPU runtimes (naive GEMM + 3 tiled kernels).  
  Observed CPU runtime grows ~O(N³) and becomes seconds, while all CUDA kernels stay in ms,  
  giving 2–3 orders of magnitude speedup. 

## Repository structure

```text
.
├── DD2360HT25_HW2_Group2.pdf   # Main report (submitted to Canvas)
├── Q1/                         # Histogram with shared memory & atomics
│   ├── hw2_histogram_template.cu
│   ├── generate_histogram_data.py
│   ├── plot_histograms.py
│   ├── Makefile
│   └── README.md
├── Q2/                         # Parallel reduction
│   ├── hw2_reduction_template.cu
│   ├── generate_timing_data.py
│   ├── plot_timing.py
│   ├── Makefile
│   └── README.md
├── Q3/                         # Tiled matrix multiplication (GEMM)
│   ├── vecMult.cu
│   ├── batch_test.sh
│   ├── plot.py
│   ├── Makefile
│   └── README.md
└── README.md                   # This file
```

## Academic integrity

All files in this repository are intended **only** for study and course examination at KTH. 

Please do **not** use this repository for plagiarism, and do **not** redistribute the solutions in ways that violate KTH’s rules on collaboration and academic honesty.



# DD2360 HT25 – 作业二：共享内存、原子操作与 Tiled GEMM

本仓库包含 **DD2360 HT25 作业二** 的代码和数据。
 完整书面报告见：`DD2360HT25_HW2_Group2.pdf`。

报告中给出的分工如下：

- **Shitong Guo** – 50%（Q[1], Q[2]）
- **Jinye Gong** – 50%（Q[2], Q[3]）

## 做了什么

### Q[1] – 共享内存与原子操作（直方图）

- 实现了使用 **每块私有共享内存直方图 + 全局合并** 的 CUDA kernel。
- 主要优化：
  - 采用共享内存私有化 + 块内树形归约，减少全局原子操作。
  - 使用并行方式对共享内存直方图进行初始化，避免串行清零。
  - 合并阶段只对 `s_bins[i] > 0` 的 bin 进行全局 atomic 更新，减少无用写入。

分析了 **全局内存读写次数**、**原子操作数量** 以及 **共享内存占用**（4096 个 bin → 每块使用约 16 KB 共享内存）。

对比 **均匀分布** 与 **正态分布** 的输入数据，绘制了从 1,024 到 1,024,000 的直方图，解释了正态分布导致 bin 高度集中、原子操作竞争加剧、执行变慢的原因。

测量 GPU 相对 CPU 的加速比，并用 Nsight 分析 N = 1,024,000 时的**Achieved Occupancy（约 77.7%）**。

### Q[2] – 并行归约（Reduction）

- 实现了基于 **块内共享内存树形归约** 的 CUDA 求和 kernel。
- 思路：
  - 采用连续（coalesced）读访问，并对越界线程进行 0 填充。
  - 每个线程块只进行一次 `atomicAdd`，将块内部分和累加到全局结果中。

推导了全局内存访问量（N 次读 + 少量写）以及原子操作数量
 （等于线程块数量）。

对输入规模从 512 到 262,144 的运行时间进行了扫测：

- 小 N 时，GPU 被启动/拷贝开销主导，速度不如 CPU。
- 在几千规模附近达到收支平衡。
- 大 N 下，GPU 可获得 **10–70×** 的加速。

统计了共享内存使用（每块约 1 KB）和 **Achieved Occupancy ~87%**（N = 262,144）。

### Q[3] – 分块矩阵乘法（Tiled Matrix Multiplication）

- 在矩阵乘法中加入 **共享内存 tiling**，实现 `tiled_gemm()` kernel。

推导了 naive GEMM 与 tiled GEMM 的 **全局内存读取次数公式** 并进行对比。

实验：

- 对 **1024×1024 × 1024×1024** 和 **513×8192 × 8192×1023** 等矩阵， 验证所有 CUDA kernel 与 CPU 结果在数值上高度一致。

比较 tile 大小 **8×8、16×16、32×32**：
 16×16 通常效果最好；当 naive kernel 已有良好的 cache 与访存整合时， tiling 的额外共享内存与同步开销可能抵消其收益，使其略慢于 naive 版本。

记录了不同 tile 和矩阵尺寸下的 Nsight profiling 数据（线程块大小、动态共享内存、Achieved Occupancy）。

对 8–10 种矩阵规模进行了测试，绘制了 **CPU vs GPU（naive + 3 种 tiled）** 的运行时间柱状图，展示 CPU 时间随 N³ 急剧增长，而所有 CUDA kernel 仍维持在毫秒级，整体加速可达 2–3 个数量级。

## 文件结构

项目的大致目录结构如下：

```text
.
├── DD2360HT25_HW2_Group2.pdf   # 主报告（提交到 Canvas）
├── Q1/                         # 直方图：共享内存 + 原子操作
│   ├── hw2_histogram_template.cu
│   ├── generate_histogram_data.py
│   ├── plot_histograms.py
│   ├── Makefile
│   └── README.md
├── Q2/                         # 并行归约
│   ├── hw2_reduction_template.cu
│   ├── generate_timing_data.py
│   ├── plot_timing.py
│   ├── Makefile
│   └── README.md
├── Q3/                         # 分块矩阵乘法（Tiled GEMM）
│   ├── vecMult.cu
│   ├── batch_test.sh
│   ├── plot.py
│   ├── Makefile
│   └── README.md
└── README.md                   # 本说明文件
```

## 使用说明与学术诚信

- 本仓库仅用于 DD2360 HT25 课程学习与作业展示。
- 请遵守 KTH 的学术诚信政策，不要将本仓库内容用于任何形式的抄袭或违规共享。
