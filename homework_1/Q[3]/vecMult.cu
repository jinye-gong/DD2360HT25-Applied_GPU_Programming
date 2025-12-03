#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

// DataType: Change this to double or float
// Use -DDATA_TYPE=double or -DDATA_TYPE=float at compile time
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

typedef DATA_TYPE DataType;

// CPU timer function using gettimeofday()
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


__global__ void vectorMultGPU(DataType *a, DataType *b, DataType *c, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numARows && col < numBColumns) {
        DataType sum = (DataType)0.0;
        for (int k = 0; k < numAColumns; k++) {
            sum += a[row * numAColumns + k] * b[k * numBColumns + col];
        }
        c[row * numBColumns + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    int numARows = 128;
    int numAColumns = 256;
    int numBRows = 256;
    int numBColumns = 32;
    
    if (argc >= 5) {
        numARows = atoi(argv[1]);
        numAColumns = atoi(argv[2]);
        numBRows = atoi(argv[3]);
        numBColumns = atoi(argv[4]);
    }
      
    printf("Matrix A dimensions: %d x %d\n", numARows, numAColumns);
    printf("Matrix B dimensions: %d x %d\n", numBRows, numBColumns);
    printf("Matrix C dimensions: %d x %d\n", numARows, numBColumns);
    
    // Print DataType being used
    printf("Using DataType: %s (%zu bytes)\n", 
           sizeof(DataType) == sizeof(double) ? "double" : "float", 
           sizeof(DataType));
    
    // Size in bytes
    size_t sizeA = numARows * numAColumns * sizeof(DataType);
    size_t sizeB = numBRows * numBColumns * sizeof(DataType);
    size_t sizeC = numARows * numBColumns * sizeof(DataType);
    
    //@@ 1. Allocate in host memory.
    DataType *h_a = (DataType *)malloc(sizeA);
    DataType *h_b = (DataType *)malloc(sizeB);
    DataType *h_c = (DataType *)malloc(sizeC);        // GPU result
    DataType *h_c_ref = (DataType *)malloc(sizeC);    // CPU reference result
    
    //@@ 3. Initialize host memory.
    srand(time(NULL));
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            h_a[i * numAColumns + j] = (DataType)(rand() % 100) / 10.0;
        }
    }
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            h_b[i * numBColumns + j] = (DataType)(rand() % 100) / 10.0;
        }
    }
    for (int i = 0; i < numARows * numBColumns; i++) {
        h_c[i] = (DataType)0.0;
        h_c_ref[i] = (DataType)0.0;
    }
    
    //@@ 2. Allocate in device memory.
    DataType *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, sizeA));
    CHECK(cudaMalloc((void **)&d_b, sizeB));
    CHECK(cudaMalloc((void **)&d_c, sizeC));
    
    // Time: Copy from host to device
    double iStart = cpuSecond();
    //@@ 4. Copy from host memory to device memory.
    CHECK(cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice));
    double iElaps = cpuSecond() - iStart;
    double timeH2D = iElaps * 1000.0;  // Convert to milliseconds
    printf("Host to Device time: %f ms\n", timeH2D);
    
    //@@ 5. Initialize thread block and thread grid.
    // Use 2D thread blocks: 16x16 = 256 threads per block
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numBColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numARows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("Thread block size: %d x %d = %d threads\n", 
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
    printf("Grid size: %d x %d = %d blocks\n", 
           blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.x * blocksPerGrid.y);
    printf("Total threads: %d\n", blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y);
    printf("Threads that compute results: %d\n", numARows * numBColumns);
    
    //@@ 6. Invoke the CUDA kernel.
    iStart = cpuSecond();
    vectorMultGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();  // Wait for all GPU threads to complete
    iElaps = cpuSecond() - iStart;
    double timeKernel = iElaps * 1000.0;  // Convert to milliseconds
    printf("Kernel execution time: %f ms\n", timeKernel);
    
    // Time: Copy from device to host
    iStart = cpuSecond();
    //@@ 7. Copy results from GPU to CPU.
    CHECK(cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    double timeD2H = iElaps * 1000.0;  // Convert to milliseconds
    printf("Device to Host time: %f ms\n", timeD2H);
    
    printf("Total GPU time: %f ms\n", timeH2D + timeKernel + timeD2H);
    
    
    // Compute CPU reference
    iStart = cpuSecond();
    vectorMultCPU(h_a, h_b, h_c_ref, numARows, numAColumns, numBRows, numBColumns);
    iElaps = cpuSecond() - iStart;
    double cpuTime = iElaps * 1000.0;  // Convert to milliseconds
    printf("CPU execution time: %f ms\n", cpuTime);
    
    //@@ 8. Compare the results with the CPU reference result.
    DataType maxError = (DataType)0.0;
    DataType tolerance = (DataType)1e-3;  // Floating point tolerance for matrix multiplication
    
    for (int i = 0; i < numARows * numBColumns; i++) {
        DataType error = (DataType)fabs(h_c[i] - h_c_ref[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    
    printf("Max error: %f\n", maxError);
    if (maxError < tolerance) {
        printf("Test PASSED - Results match within tolerance!\n");
    } else {
        printf("Test FAILED - Results differ by more than tolerance!\n");
    }
    
    //@@ 10. Free device memory.
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    
    //@@ 9. Free host memory.
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    
    return 0;
}
