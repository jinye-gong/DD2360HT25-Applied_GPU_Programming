#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",      \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                         \
    }                                                    \
} while (0)

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

typedef DATA_TYPE DataType;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


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

__global__ void convertToHalf(__half *out, DataType *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half((float)in[idx]);
    }
}

__global__ void convertFromHalf(DataType *out, __half *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (DataType)__half2float(in[idx]);
    }
}

__global__ void convertFloatToDataType(DataType *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (DataType)in[idx];
    }
}



__global__ void matrixMultGPU(DataType *a, DataType *b, DataType *c, int M, int K, int N) {
    // 1. 计算当前线程处理的 C 矩阵坐标 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        DataType sum = (DataType)0.0;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void tiled_gemm( DataType *A, DataType *B, DataType *C,
                            int M, int K, int N,
                            int tileX, int tileY){
    int row = blockIdx.y * tileY + threadIdx.y; 
    int col = blockIdx.x * tileX + threadIdx.x; 

    extern __shared__ DataType shmem[];
    DataType* As = shmem;                 
    DataType* Bs = shmem + tileY * tileX; 

    DataType sum = (DataType)0.0;
                                
    int numTilesK = (K + tileX - 1) / tileX;

    for (int t = 0; t < numTilesK; ++t) {
        int kA = t * tileX + threadIdx.x; 
        int kB = t * tileX + threadIdx.y; 


        if (row < M && kA < K) {
            As[threadIdx.y * tileX + threadIdx.x] =
                A[row * K + kA];
        } else {
            As[threadIdx.y * tileX + threadIdx.x] = (DataType)0.0;
        }

        if (kB < K && col < N) {
            Bs[threadIdx.y * tileX + threadIdx.x] =
                B[kB * N + col];
        } else {
            Bs[threadIdx.y * tileX + threadIdx.x] = (DataType)0.0;
        }

        __syncthreads();

        for (int k = 0; k < tileX; ++k) {
            DataType a_val = As[threadIdx.y * tileX + k];      
            DataType b_val = Bs[k * tileX + threadIdx.x];      
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

}


__global__ void wmma_gemm(__half *A, __half *B, float *C, 
                          int M, int K, int N) {
    // WMMA tile dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    __shared__ __half shmemA[4][WMMA_M * WMMA_K];
    __shared__ __half shmemB[4][WMMA_K * WMMA_N];
    

    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    
    int warpRow = blockIdx.y * blockDim.y + warpId;
    int warpCol = blockIdx.x;
    
    int cRow = warpRow * WMMA_M;
    int cCol = warpCol * WMMA_N;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int numTilesK = (K + WMMA_K - 1) / WMMA_K;
    
    for (int t = 0; t < numTilesK; ++t) {
        int aRow = cRow;
        int aCol = t * WMMA_K;
        int bRow = t * WMMA_K;
        int bCol = cCol;
        

        for (int i = laneId; i < WMMA_M * WMMA_K; i += 32) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int globalRow = aRow + row;
            int globalCol = aCol + col;
            if (globalRow < M && globalCol < K) {
                shmemA[warpId][row * WMMA_K + col] = A[globalRow * K + globalCol];
            } else {
                shmemA[warpId][row * WMMA_K + col] = __float2half(0.0f);
            }
        }
        

        for (int i = laneId; i < WMMA_K * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int globalRow = bRow + row;
            int globalCol = bCol + col;
            if (globalRow < K && globalCol < N) {
                shmemB[warpId][col * WMMA_K + row] = B[globalRow * N + globalCol];
            } else {
                shmemB[warpId][col * WMMA_K + row] = __float2half(0.0f);
            }
        }
        
        __syncwarp();
        
        wmma::load_matrix_sync(a_frag, shmemA[warpId], WMMA_K);
        wmma::load_matrix_sync(b_frag, shmemB[warpId], WMMA_N);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    if (cRow < M && cCol < N) {
        if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
            wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
        } else {
            float temp[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(temp, c_frag, WMMA_N, wmma::mem_row_major);
            for (int i = 0; i < WMMA_M; i++) {
                for (int j = 0; j < WMMA_N; j++) {
                    int row = cRow + i;
                    int col = cCol + j;
                    if (row < M && col < N) {
                        C[row * N + col] = temp[i * WMMA_N + j];
                    }
                }
            }
        }
    }
}





int main(int argc, char *argv[]) {
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

    if (numAColumns != numBRows) {
        fprintf(stderr, "Error: Matrix dimension mismatch! K dim must match.\n");
        return 1;
    }

    printf("Output matrix dim: %d x %d\n\n", numARows, numBColumns);

    size_t sizeA = numARows * numAColumns * sizeof(DataType);
    size_t sizeB = numBRows * numBColumns * sizeof(DataType);
    size_t sizeC = numARows * numBColumns * sizeof(DataType);

    DataType *h_a = (DataType *)malloc(sizeA);
    DataType *h_b = (DataType *)malloc(sizeB);
    DataType *h_c = (DataType *)malloc(sizeC);//GPU
    DataType *h_c_ref = (DataType *)malloc(sizeC);//CPU
    DataType *h_c_tiled = (DataType *)malloc(sizeC); // tiled_gemm GPU result
    DataType *h_c_wmma = (DataType *)malloc(sizeC); // wmma_gemm GPU result


    srand(2024);
    for (int i = 0; i < numARows * numAColumns; i++) h_a[i] = (DataType)(rand() % 100) / 10.0;
    for (int i = 0; i < numBRows * numBColumns; i++) h_b[i] = (DataType)(rand() % 100) / 10.0;

    DataType *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sizeA));
    CHECK(cudaMalloc(&d_b, sizeB));
    CHECK(cudaMalloc(&d_c, sizeC));

    double start = cpuSecond();
    vectorMultCPU(h_a, h_b, h_c_ref, numARows, numAColumns, numBRows, numBColumns);
    double cpuTime = (cpuSecond() - start) * 1000.0; // ms
    printf("CPU reference result:\n");
    printf("timing: %.3f ms\n\n", cpuTime);

    start = cpuSecond();
    CHECK(cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice));
    double timeH2D = (cpuSecond() - start) * 1000.0; // ms


    int dim = 32; 
    dim3 block(dim, dim);
    dim3 grid((numBColumns + block.x - 1) / block.x, 
              (numARows + block.y - 1) / block.y);



    matrixMultGPU<<<grid, block>>>(d_a, d_b, d_c, numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize();

    start = cpuSecond();
    matrixMultGPU<<<grid, block>>>(d_a, d_b, d_c, numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize(); 
    double timeKernel = (cpuSecond() - start) * 1000.0; // ms
    CHECK(cudaGetLastError());


    start = cpuSecond();
    CHECK(cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost));
    double timeD2H = (cpuSecond() - start) * 1000.0; // ms


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

    int tileSizes[] = {8, 16, 32};
    int numTileConfigs = sizeof(tileSizes) / sizeof(tileSizes[0]);

    for (int t = 0; t < numTileConfigs; ++t) {
        int tileX = tileSizes[t];
        int tileY = tileSizes[t];  // 方块 tile

        dim3 blockT(tileX, tileY);
        dim3 gridT((numBColumns + tileX - 1) / tileX,
                   (numARows   + tileY - 1) / tileY);

        size_t sharedBytes = 2 * tileX * tileY * sizeof(DataType);


        tiled_gemm<<<gridT, blockT, sharedBytes>>>(
            d_a, d_b, d_c,
            numARows, numAColumns, numBColumns,
            tileX, tileY
        );
        CHECK(cudaDeviceSynchronize());

        start = cpuSecond();
        tiled_gemm<<<gridT, blockT, sharedBytes>>>(
            d_a, d_b, d_c,
            numARows, numAColumns, numBColumns,
            tileX, tileY
        );
        CHECK(cudaDeviceSynchronize());
        double timeTiled = (cpuSecond() - start) * 1000.0; // ms

        CHECK(cudaMemcpy(h_c_tiled, d_c, sizeC, cudaMemcpyDeviceToHost));

        DataType maxErrT = (DataType)0.0;
        for (int i = 0; i < numARows * numBColumns; ++i) {
            DataType err = (DataType)fabs(h_c_tiled[i] - h_c_ref[i]);
            if (err > maxErrT) maxErrT = err;
        }

        printf("CUDA tiled_gemm with tile [%d, %d] result:\n", tileX, tileY);
        printf("max error vs CPU: %f\n", maxErrT);
        printf("timing: %.3f ms\n\n", timeTiled);
    }

    // ==== WMMA (Tensor Core) Kernel ====
    printf("=== WMMA (Tensor Core) Kernel ===\n");
    
    // Allocate half precision matrices on device
    size_t sizeA_half = numARows * numAColumns * sizeof(__half);
    size_t sizeB_half = numBRows * numBColumns * sizeof(__half);
    size_t sizeC_float = numARows * numBColumns * sizeof(float);
    
    __half *d_a_half, *d_b_half;
    float *d_c_float;
    CHECK(cudaMalloc(&d_a_half, sizeA_half));
    CHECK(cudaMalloc(&d_b_half, sizeB_half));
    CHECK(cudaMalloc(&d_c_float, sizeC_float));
    
    int threadsPerBlock = 256;
    int blocksA = (numARows * numAColumns + threadsPerBlock - 1) / threadsPerBlock;
    int blocksB = (numBRows * numBColumns + threadsPerBlock - 1) / threadsPerBlock;
    
    convertToHalf<<<blocksA, threadsPerBlock>>>(d_a_half, d_a, numARows * numAColumns);
    convertToHalf<<<blocksB, threadsPerBlock>>>(d_b_half, d_b, numBRows * numBColumns);
    CHECK(cudaDeviceSynchronize());
    

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    dim3 blockWmma(32, 4);  // 32 threads per warp, 4 warps per block
    dim3 gridWmma((numBColumns + WMMA_N - 1) / WMMA_N,
                  (numARows + WMMA_M - 1) / WMMA_M);
    
    // Warmup
    wmma_gemm<<<gridWmma, blockWmma>>>(d_a_half, d_b_half, d_c_float, 
                                        numARows, numAColumns, numBColumns);
    CHECK(cudaDeviceSynchronize());
    
    // Time WMMA kernel
    start = cpuSecond();
    wmma_gemm<<<gridWmma, blockWmma>>>(d_a_half, d_b_half, d_c_float, 
                                        numARows, numAColumns, numBColumns);
    CHECK(cudaDeviceSynchronize());
    double timeWmma = (cpuSecond() - start) * 1000.0; // ms
    CHECK(cudaGetLastError());
    
    // Convert result from float to DataType
    DataType *d_c_wmma;
    CHECK(cudaMalloc(&d_c_wmma, sizeC));
    int blocksC = (numARows * numBColumns + threadsPerBlock - 1) / threadsPerBlock;
    convertFloatToDataType<<<blocksC, threadsPerBlock>>>(d_c_wmma, d_c_float, numARows * numBColumns);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_c_wmma, d_c_wmma, sizeC, cudaMemcpyDeviceToHost));
    
    // Compare with CPU reference
    DataType maxErrWmma = (DataType)0.0;
    DataType totalErrWmma = (DataType)0.0;
    for (int i = 0; i < numARows * numBColumns; ++i) {
        DataType err = (DataType)fabs((double)h_c_wmma[i] - (double)h_c_ref[i]);
        if (err > maxErrWmma) maxErrWmma = err;
        totalErrWmma += err;
    }
    DataType avgErrWmma = totalErrWmma / (numARows * numBColumns);
    
    // Calculate fragment information
    int fragmentsPerWarp = 1; // Each warp computes one output tile
    int totalWarps = gridWmma.x * gridWmma.y * blockWmma.y;
    int totalFragments = totalWarps * fragmentsPerWarp;
    
    printf("WMMA kernel result:\n");
    printf("Fragment dimensions: %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Total fragments used: %d\n", totalFragments);
    printf("Total warps: %d\n", totalWarps);
    printf("max error vs CPU: %e\n", (double)maxErrWmma);
    printf("avg error vs CPU: %e\n", (double)avgErrWmma);
    printf("timing: %.3f ms\n\n", timeWmma);
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_a_half); cudaFree(d_b_half); cudaFree(d_c_float); cudaFree(d_c_wmma);
    free(h_a); free(h_b); free(h_c); free(h_c_ref); free(h_c_tiled); free(h_c_wmma);

    return 0;
}

