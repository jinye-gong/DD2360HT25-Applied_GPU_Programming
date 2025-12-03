#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

// 允许通过编译选项 -DDATA_TYPE=double 切换精度
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

typedef DATA_TYPE DataType;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


// CPU reference implementation
__host__ void vectorMultCPU(DataType *a, DataType *b, DataType *c, int numARows, int numAColumns, int numBRows, int numBColumns) {
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            c[i * numBColumns + j] = (DataType)0.0;
            for (int k = 0; k < numAColumns; k++) {
                c[i * numBColumns + j] += a[i * numAColumns + k] * b[k * numBColumns + j];
            }
        }
    }
}



// --- 核心：Basic 2D Matrix Multiplication Kernel ---
__global__ void matrixMultGPU(DataType *a, DataType *b, DataType *c, int M, int K, int N) {
    // 1. 计算当前线程处理的 C 矩阵坐标 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 边界检查：确保不越界 (这对处理 8191 这种奇数尺寸至关重要)
    if (row < M && col < N) {
        DataType sum = (DataType)0.0;
        // 3. 计算点积：遍历 A 的行和 B 的列
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void tiled_gemm( DataType *A, DataType *B, DataType *C,
                            int M, int K, int N,
                            int tileX, int tileY){
    // 1. 计算当前线程处理的 C 矩阵坐标 (row, col)
    int row = blockIdx.y * tileY + threadIdx.y; // C 的行 [0, M)
    int col = blockIdx.x * tileX + threadIdx.x; // C 的列 [0, N)

    extern __shared__ DataType shmem[];
    DataType* As = shmem;                 
    DataType* Bs = shmem + tileY * tileX; 

    DataType sum = (DataType)0.0;
                                
    int numTilesK = (K + tileX - 1) / tileX;

    for (int t = 0; t < numTilesK; ++t) {
        int kA = t * tileX + threadIdx.x; // A 的列索引 in [0, K)
        int kB = t * tileX + threadIdx.y; // B 的行索引 in [0, K)

        // --- 把 A 的一个 tile (tileY x tileX) 搬到 shared memory ---
        if (row < M && kA < K) {
            // A(row, kA) => A[row * K + kA]
            As[threadIdx.y * tileX + threadIdx.x] =
                A[row * K + kA];
        } else {
            As[threadIdx.y * tileX + threadIdx.x] = (DataType)0.0;
        }

        // --- 把 B 的一个 tile (tileX x tileY) 搬到 shared memory（转置风格）---
        if (kB < K && col < N) {
            // B(kB, col) => B[kB * N + col]
            Bs[threadIdx.y * tileX + threadIdx.x] =
                B[kB * N + col];
        } else {
            Bs[threadIdx.y * tileX + threadIdx.x] = (DataType)0.0;
        }

        __syncthreads();

        // 在共享内存中做部分乘加累积
        for (int k = 0; k < tileX; ++k) {
            // As: [threadIdx.y, k]
            DataType a_val = As[threadIdx.y * tileX + k];      
            // Bs: [k, threadIdx.x]
            DataType b_val = Bs[k * tileX + threadIdx.x];      
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    // 写回 C(row, col) => C[row * N + col]
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

}





int main(int argc, char *argv[]) {
    // 默认尺寸，可以通过命令行覆盖
    int numARows = 512;    // M
    int numAColumns = 512; // K
    int numBRows = 512;    // K
    int numBColumns = 512; // N

    if (argc >= 5) {
        numARows = atoi(argv[1]);
        numAColumns = atoi(argv[2]);
        numBRows = atoi(argv[3]);
        numBColumns = atoi(argv[4]);
    }

    // 检查维度匹配
    if (numAColumns != numBRows) {
        fprintf(stderr, "Error: Matrix dimension mismatch! K dim must match.\n");
        return 1;
    }

    printf("Output matrix dim: %d x %d\n\n", numARows, numBColumns);

    // 计算数据大小
    size_t sizeA = numARows * numAColumns * sizeof(DataType);
    size_t sizeB = numBRows * numBColumns * sizeof(DataType);
    size_t sizeC = numARows * numBColumns * sizeof(DataType);

    // Host 内存分配
    DataType *h_a = (DataType *)malloc(sizeA);
    DataType *h_b = (DataType *)malloc(sizeB);
    DataType *h_c = (DataType *)malloc(sizeC);//GPU
    DataType *h_c_ref = (DataType *)malloc(sizeC);//CPU
    DataType *h_c_tiled = (DataType *)malloc(sizeC); // tiled_gemm GPU result


    // 初始化数据
    srand(2024);
    for (int i = 0; i < numARows * numAColumns; i++) h_a[i] = (DataType)(rand() % 100) / 10.0;
    for (int i = 0; i < numBRows * numBColumns; i++) h_b[i] = (DataType)(rand() % 100) / 10.0;

    // Device 内存分配
    DataType *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sizeA));
    CHECK(cudaMalloc(&d_b, sizeB));
    CHECK(cudaMalloc(&d_c, sizeC));

     // Compute CPU reference
    double start = cpuSecond();
    vectorMultCPU(h_a, h_b, h_c_ref, numARows, numAColumns, numBRows, numBColumns);
    double cpuTime = (cpuSecond() - start) * 1000.0; // ms
    printf("CPU reference result:\n");
    printf("timing: %.3f ms\n\n", cpuTime);

    // --- 1. 计时 H2D ---
    start = cpuSecond();
    CHECK(cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice));
    double timeH2D = (cpuSecond() - start) * 1000.0; // ms

    // --- Naive Kernel 配置 ---
    // 使用 16x16 或 32x32 的 Block
    int dim = 32; 
    dim3 block(dim, dim);
    // 向上取整计算 Grid 大小
    dim3 grid((numBColumns + block.x - 1) / block.x, 
              (numARows + block.y - 1) / block.y);


    // --- 2. 计时 Naive Kernel ---
    // Warmup
    matrixMultGPU<<<grid, block>>>(d_a, d_b, d_c, numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize();

    start = cpuSecond();
    matrixMultGPU<<<grid, block>>>(d_a, d_b, d_c, numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize(); // 必须同步以获取准确时间
    double timeKernel = (cpuSecond() - start) * 1000.0; // ms
    CHECK(cudaGetLastError());


    // --- 3. 计时 D2H ---
    start = cpuSecond();
    CHECK(cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost));
    double timeD2H = (cpuSecond() - start) * 1000.0; // ms




    //--- Compare Naive with CPU reference ---
    DataType maxError = (DataType)0.0;
    
    for (int i = 0; i < numARows * numBColumns; i++) {
        DataType error = (DataType)fabs(h_c[i] - h_c_ref[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    
    printf("CUDA gemm result:\n");
    printf("max error vs CPU: %f\n", maxError);
    printf("timing: %.3f ms\n\n", timeKernel);

    // ==== Sweep different tile sizes for tiled_gemm ====
    int tileSizes[] = {8, 16, 32};
    int numTileConfigs = sizeof(tileSizes) / sizeof(tileSizes[0]);

    for (int t = 0; t < numTileConfigs; ++t) {
        int tileX = tileSizes[t];
        int tileY = tileSizes[t];  // 方块 tile

        dim3 blockT(tileX, tileY);
        dim3 gridT((numBColumns + tileX - 1) / tileX,
                   (numARows   + tileY - 1) / tileY);

        size_t sharedBytes = 2 * tileX * tileY * sizeof(DataType);


        // warmup
        tiled_gemm<<<gridT, blockT, sharedBytes>>>(
            d_a, d_b, d_c,
            numARows, numAColumns, numBColumns,
            tileX, tileY
        );
        CHECK(cudaDeviceSynchronize());

        // 计时 tiled_gemm kernel
        start = cpuSecond();
        tiled_gemm<<<gridT, blockT, sharedBytes>>>(
            d_a, d_b, d_c,
            numARows, numAColumns, numBColumns,
            tileX, tileY
        );
        CHECK(cudaDeviceSynchronize());
        double timeTiled = (cpuSecond() - start) * 1000.0; // ms

        // 拷结果回来
        CHECK(cudaMemcpy(h_c_tiled, d_c, sizeC, cudaMemcpyDeviceToHost));

        // 和 CPU reference 比误差
        DataType maxErrT = (DataType)0.0;
        for (int i = 0; i < numARows * numBColumns; ++i) {
            DataType err = (DataType)fabs(h_c_tiled[i] - h_c_ref[i]);
            if (err > maxErrT) maxErrT = err;
        }

        printf("CUDA tiled_gemm with tile [%d, %d] result:\n", tileX, tileY);
        printf("max error vs CPU: %f\n", maxErrT);
        printf("timing: %.3f ms\n\n", timeTiled);
    }

    // 清理（放在循环外）
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_c_ref); free(h_c_tiled);

    return 0;
}

