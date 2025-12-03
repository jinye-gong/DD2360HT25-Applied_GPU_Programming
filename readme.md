# DD2360 HT25 – GPU Projects

This repo contains two GPU programming projects for the course **DD2360 HT25**.

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

## Repository Structure

```text
.
├── HW1/            # Assignment I code and local README
├── HW2/            # Assignment II code and local README
└── README.md       # This file
```



## Academic integrity

All files in this repository are intended **only** for study and course examination at KTH. 

Please do **not** use this repository for plagiarism, and do **not** redistribute the solutions in ways that violate KTH’s rules on collaboration and academic honesty.

---





本仓库包含 DD2360 HT25 课程的两个 GPU 编程项目。

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

## 目录结构

```
.
├── HW1/            # 作业一代码及本地说明
├── HW2/            # 作业二代码及本地说明
└── README.md       # 本说明文件
```



## 使用说明与学术诚信

- 本仓库仅用于 DD2360 HT25 课程学习与作业展示。
- 请遵守 KTH 的学术诚信政策，不要将本仓库内容用于任何形式的抄袭或违规共享。
