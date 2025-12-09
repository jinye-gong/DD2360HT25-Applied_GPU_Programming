# DD2360 HT25 – Assignment III: Thread Divergence & CUDA Streams

This repo contains the code and report for **DD2360 HT25 – Assignment III**.  

Main report: `DD2360HT25_HW3_Group2.pdf`. 



## What we did

### Q[1] – Thread Scheduling & Divergence

- Analyzed a 2D CUDA kernel over images of size `X×Y` with `blockDim = (64,16)`.   
- For three cases (X=800,Y=600; X=600,Y=800; X=600,Y=899) we:
  - computed grid size, total number of warps;
  - counted how many warps suffer control divergence due to the boundary check  
    `if (Row < m && Col < n)`.   

### Q[2] – Vector Addition with CUDA Streams

- Implemented two GPU versions of `vecAdd`:   
  - **Non-streamed**: one H2D copy of inputs, one kernel launch, one D2H copy.  
  - **4-stream version**: split the vector into segments and use 4 CUDA streams with async H2D / kernel / D2H to overlap copy and compute.  
- Measured total GPU time vs vector length and computed speedup of 4-stream over non-streamed.   
- Used `nvprof` / Nsight timeline to show overlap of communication and computation. 
- For a fixed large N, varied segment size and plotted how it affects runtime and speedup (very small segments hurt, medium segments best, very large segments reduce overlap). 

---

# DD2360 HT25 – 作业三：线程发散与 CUDA Streams

本仓库包含 **DD2360 HT25 作业三** 的代码与报告。  

主报告：`DD2360HT25_HW3_Group2.pdf`。

## 做了什么

### Q[1] – 线程调度与控制流发散

- 针对使用 `blockDim = (64,16)` 的 2D CUDA kernel，在三种图像尺寸  
  `X=800,Y=600`、`X=600,Y=800`、`X=600,Y=899` 下：  
  - 计算网格大小与 **warp 总数**；  
  - 统计因边界判断 `if (Row < m && Col < n)` 产生控制流发散的 **warp 个数**。   

### Q[2] – 使用 CUDA Streams 的向量加法

- 实现并比较两个 GPU 版本的 `vecAdd`：  
  - **无 stream 版本**：一次 H2D 拷贝输入，一次 kernel，一次 D2H 拷贝输出；  
  - **4-stream 版本**：将向量按段切分，用 4 个 CUDA streams 做异步 H2D / kernel / D2H，实现拷贝与计算重叠。  
- 测量不同向量长度下的总 GPU 时间，并计算 4-stream 相对无 stream 的加速比。  
- 使用 `nvprof` / Nsight 生成时间线，对比无 stream 串行执行和 4-stream 的重叠执行。
- 在固定大 N 情况下，改变 segment size，画出时间和加速比随段大小变化的曲线，观察小段/大段对性能的影响。

