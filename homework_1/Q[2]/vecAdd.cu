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

// CPU timer function using gettimeofday()
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// CPU reference implementation
__host__ void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process elements within array bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    // Get vector length from command line or use default
    int N = 512;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    printf("Vector length: %d\n", N);
    
    size_t size = N * sizeof(float);
    
    //@@ 1. Allocate in host memory.
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);    
    float *h_c_ref = (float *)malloc(size);   
    
    //@@ 3. Initialize host memory.
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) / 10.0f;
        h_b[i] = (float)(rand() % 100) / 10.0f;
        h_c[i] = 0.0f;
        h_c_ref[i] = 0.0f;
    }
    
    //@@ 2. Allocate in device memory.
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));
    
    // Time: Copy from host to device
    double iStart = cpuSecond();
    //@@ 4. Copy from host memory to device memory.
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    double iElaps = cpuSecond() - iStart;
    double timeH2D = iElaps * 1000.0;  // Convert to milliseconds
    printf("Host to Device time: %f ms\n", timeH2D);
    
    //@@ 5. Initialize thread block and thread grid.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division
    
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
    
    //@@ 6. Invoke the CUDA kernel.
    iStart = cpuSecond();
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();  // Wait for all GPU threads to complete
    iElaps = cpuSecond() - iStart;
    double timeKernel = iElaps * 1000.0;  // Convert to milliseconds
    printf("Kernel execution time: %f ms\n", timeKernel);
    
    // Time: Copy from device to host
    iStart = cpuSecond();
    //@@ 7. Copy results from GPU to CPU.
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    double timeD2H = iElaps * 1000.0;  // Convert to milliseconds
    printf("Device to Host time: %f ms\n", timeD2H);
    
    printf("Total GPU time: %f ms\n", timeH2D + timeKernel + timeD2H);
        
    // Compute CPU reference
    iStart = cpuSecond();
    vectorAddCPU(h_a, h_b, h_c_ref, N);
    iElaps = cpuSecond() - iStart;
    double cpuTime = iElaps * 1000.0;  // Convert to milliseconds
    printf("CPU execution time: %f ms\n", cpuTime);
    
    //@@ 8. Compare the results with the CPU reference result.
    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < N; i++) {
        float error = fabsf(h_c[i] - h_c_ref[i]);
        if (error > maxError) {
            maxError = error;
        }
        if (error > 1e-5) {
            errorCount++;
        }
    }
    printf("Max error: %f\n", maxError);
    if (errorCount > 0) {
        printf("Warning: %d elements have error > 1e-5\n", errorCount);
    } else {
        printf("Results match CPU reference (within tolerance)\n");
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
