# DD2360 HT25 – Assignment I: GPU Architecture and CUDA Basics

This repository contains the code and data for **DD2360 HT25 – Assignment I**.  
The written report is in: `DD2360HT25_HW1_Group2.pdf`.

## What we did

### Q1 – Reflection on GPU-accelerated Computing
- Summarized three key architectural differences between GPUs and CPUs  
  (throughput-oriented vs latency-oriented design, simpler compute units, massive hardware threads/SIMD).
- Checked the latest Top500 list, counted how many of the top-10 systems use GPUs,  
  and listed their names, GPU vendors, and GPU models.
- Computed performance-per-watt (GFLOPS/W) for the top-10 supercomputers  
  and compared their energy efficiency.

### Q2 – Vector Addition in CUDA
- Implemented a `vectorAdd` CUDA program (`vecAdd.cu`) with a CPU reference version.
- Added comments to mark all key CUDA steps in the code (memory allocation, copies, kernel launch, etc.).
- For different vector sizes (including N = 512 and N = 263149), analyzed:
  - Number of FLOPs and global memory accesses
  - Number of threads and thread blocks launched
  - Achieved Occupancy using Nvidia Nsight
- Used CPU timers to separate H2D, kernel, and D2H time, and plotted stacked bar charts  
  to show how the time breakdown changes with N.

### Q3 – 2D Dense Matrix Multiplication
- Implemented 2D grid/block-based matrix multiplication in CUDA (`vecMult.cu`).
- Derived formulas for the number of FLOPs and global memory reads in the kernel.
- For different matrix sizes (e.g., 128×256 × 256×32, 1024×8191 × 8191×8197):
  - Computed the number of blocks and threads used
  - Distinguished between useful threads and threads that do no work
  - Measured Achieved Occupancy with Nsight
- For multiple matrix sizes, plotted stacked bar charts of H2D / kernel / D2H time  
  and compared performance when using `double` vs `float`.

### Q4 – Rodinia Benchmarks: GPU vs CPU
- Selected a few Rodinia benchmarks (e.g., BFS, hotspot) and compiled both OpenMP and CUDA versions.
- Ran CPU (OpenMP) and GPU (CUDA) with the same inputs, recorded execution times, and computed speedups.  
  Results were stored in CSV files such as:
  - `BFS_openmp_vs_cuda.csv`
  - `hotspot_openmp_vs_cuda.csv`
- Analyzed why, for some problem sizes, GPU does not always deliver large end-to-end speedups  
  (startup overheads, data transfer cost, irregular workloads, etc.).

## Repository structure

A simple overview of the folder layout:

```text
.
├── DD2360HT25_HW1_Group2.pdf      # Main report (submitted to Canvas)
├── Q1/                            # Notes / scripts for Question 1 (if any)
│   └── README.md
├── Q2/                            # Vector addition (CUDA)
│   ├── vecAdd.cu
│   ├── Makefile
│   └── README.md
├── Q3/                            # 2D dense matrix multiplication (CUDA)
│   ├── vecMult.cu
│   ├── Makefile
│   └── README.md
├── Q4/                            # Rodinia CUDA & OpenMP benchmarks
│   ├── BFS_openmp_vs_cuda.csv
│   ├── hotspot_openmp_vs_cuda.csv
│   └── README.md
└── README.md                      # This file
```



## Academic integrity

All files in this repository are intended **only** for study and course examination at KTH. 

Please do **not** use this repository for plagiarism, and do **not** redistribute the solutions in ways that violate KTH’s rules on collaboration and academic honesty.



------

本仓库包含 **DD2360 HT25 作业一** 的代码和数据。
 完整书面报告见：`DD2360HT25_HW1_Group2.pdf`。

## 做了什么

### Q1 – GPU 加速计算的思考

- 总结了 GPU 与 CPU 的三大架构差异
   （吞吐导向 vs 延迟导向、运算单元更简单、大量硬件线程和 SIMD）。
- 查阅最新 Top500 榜单，统计前 10 台超算中使用 GPU 的系统数量，
   并列出其名称、GPU 厂商以及具体型号。
- 计算前 10 台超算的性能/功耗比（GFLOPS/W），并比较它们的能效差异。

### Q2 – CUDA 向量加法

- 实现了带 CPU 基准版本的 `vectorAdd` 程序（`vecAdd.cu`）。
- 在代码中标注了所有关键 CUDA 步骤（内存分配、数据拷贝、kernel 调用等）。
- 对不同向量长度（包括 N = 512 和 N = 263149）进行了分析：
  - 计算 FLOPs 数量与全局内存访问次数
  - 计算启动的线程数与线程块数
  - 使用 Nvidia Nsight 采集 Achieved Occupancy
- 使用 CPU 计时器分别测量 H2D、kernel 和 D2H 的时间，
   对多组 N 绘制堆叠柱状图，分析时间占比随向量长度变化的趋势。

### Q3 – 二维稠密矩阵乘法

- 在 CUDA 中实现基于二维网格 / 线程块的矩阵乘法程序（`vecMult.cu`）。
- 推导了矩阵乘法 kernel 中 FLOPs 数量和全局内存读取次数的公式。
- 针对不同矩阵尺寸（如 128×256 × 256×32、1024×8191 × 8191×8197）：
  - 计算使用的线程块个数和线程总数
  - 区分真正参与计算的有效线程与不做计算的“浪费”线程
  - 使用 Nsight 测量 Achieved Occupancy
- 对多种矩阵规模绘制 H2D / kernel / D2H 时间的堆叠柱状图，
   并比较使用 `double` 与 `float` 两种数据类型时的性能差异。

### Q4 – Rodinia 基准测试：GPU vs CPU

- 从 Rodinia 基准套件中选择多个基准程序（如 BFS、hotspot），
   编译并运行其 OpenMP 和 CUDA 版本。
- 使用相同输入数据，对比 CPU（OpenMP）和 GPU（CUDA）的运行时间，
   计算加速比，并将结果保存在 CSV 文件中，例如：
  - `BFS_openmp_vs_cuda.csv`
  - `hotspot_openmp_vs_cuda.csv`
- 分析在给定数据规模下，GPU 未必总能带来显著端到端加速的原因，
   包括启动开销、数据传输开销以及工作负载不规则等因素。

## 文件结构

项目的大致目录结构如下：

```text
.
├── DD2360HT25_HW1_Group2.pdf      # 主报告（提交到 Canvas）
├── Q1/                    
│   └── README.md
├── Q2/                            # 向量加法（CUDA）
│   ├── vecAdd.cu
│   ├── Makefile
│   └── README.md
├── Q3/                            # 二维稠密矩阵乘法（CUDA）
│   ├── vecMult.cu
│   ├── Makefile
│   └── README.md
├── Q4/                            # Rodinia 的 CUDA & OpenMP 基准测试
│   ├── BFS_openmp_vs_cuda.csv
│   ├── hotspot_openmp_vs_cuda.csv
│   └── README.md
└── README.md                      # 本说明文件
```

## 使用说明与学术诚信

- 本仓库仅用于 DD2360 HT25 课程学习与作业展示。
- 请遵守 KTH 的学术诚信政策，不要将本仓库内容用于任何形式的抄袭或违规共享。
