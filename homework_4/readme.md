# DD2360 HT25 – HW4 (Group 2)

## What we did
- **Q1 (1D Convolution):** Implemented a basic kernel and a shared-memory tiled kernel, and compared performance/memory access behavior.
- **Q2 (1D Heat Equation):** Implemented the iterative solver using **Unified Memory**, **cuSPARSE SpMV**, and **cuBLAS** vector ops; evaluated performance and accuracy, including **prefetch vs no-prefetch**.
- **Bonus (Tensor Cores):** Implemented a **WMMA/Tensor Core** GEMM variant and compared speed/accuracy; profiled with Nsight tools.

# DD2360 HT25 – HW4（第2组）

## 我们做了什么
- **Q1（一维卷积）：** 实现了基础版本和基于共享内存的分块（tiled）版本，并对性能/内存访问行为进行对比分析。
- **Q2（一维热传导）：** 使用**统一内存（Unified Memory）**、**cuSPARSE 的 SpMV** 和 **cuBLAS 向量操作**实现迭代求解器；评估性能与精度，并比较**预取（prefetch）与不预取**的差异。
- **Bonus（Tensor Cores）：** 实现了基于 **WMMA/Tensor Core** 的 GEMM 版本，对比速度与精度，并使用 Nsight 工具进行 profiling。
