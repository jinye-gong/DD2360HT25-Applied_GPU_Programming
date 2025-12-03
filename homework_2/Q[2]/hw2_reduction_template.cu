#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void reduction_kernel(float *input, float *output, int n) {
  
  //@@ Insert code below to compute reduction using shared memory and atomics
  
  // Shared memory for block-level reduction
  __shared__ float sdata[THREADS_PER_BLOCK];
  
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Load data into shared memory
  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();
  
  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result for this block to global memory using atomic operation
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}

// CPU reference implementation
__host__ float reduction_cpu(float *input, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += input[i];
  }
  return sum;
}

// Get time in milliseconds
__host__ double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
  
  int inputLength;

  //@@ Insert code below to read in inputLength from args
  if (argc < 2) {
    printf("Usage: %s <input_length>\n", argv[0]);
    return 1;
  }
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  /*@ add other needed data allocation on CPU and GPU here */
  float *hostInput;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  float cpuResult;
  
  // Allocate host memory
  hostInput = (float *)malloc(inputLength * sizeof(float));
  hostOutput = (float *)malloc(sizeof(float));
  
  if (!hostInput || !hostOutput) {
    printf("Error: Failed to allocate host memory\n");
    return 1;
  }

  
  //@@ Insert code below to initialize the input array with random values on CPU
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = dis(gen);
  }


  //@@ Insert code below to create reference result in CPU and add a timer
  double cpu_start = get_time_ms();
  cpuResult = reduction_cpu(hostInput, inputLength);
  double cpu_end = get_time_ms();
  double cpu_time = cpu_end - cpu_start;
  printf("CPU time: %.3f ms\n", cpu_time);
  printf("CPU result: %.6f\n", cpuResult);


  //@@ Insert code to copy data from CPU to the GPU
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, sizeof(float));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: Failed to allocate device memory: %s\n", cudaGetErrorString(err));
    return 1;
  }
  
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  
  // Initialize output to zero
  cudaMemset(deviceOutput, 0, sizeof(float));


  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(THREADS_PER_BLOCK);
  dim3 gridSize((inputLength + blockSize.x - 1) / blockSize.x);


  //@@ Launch the GPU Kernel here and add a timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);
  reduction_kernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, inputLength);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: Reduction kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  printf("GPU time: %.3f ms\n", gpu_time_ms);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  float gpuResult = hostOutput[0];
  printf("GPU result: %.6f\n", gpuResult);
  
  float diff = fabsf(cpuResult - gpuResult);
  float tolerance = 1e-3f; // Tolerance for floating point comparison
  
  if (diff < tolerance) {
    printf("Results match! (Difference: %.6f)\n", diff);
  } else {
    printf("Results do not match! (Difference: %.6f)\n", diff);
  }
  
  printf("Speedup: %.2fx\n", cpu_time / gpu_time_ms);
  
  // Output timing data in CSV format for plotting (if requested via environment variable)
  if (getenv("OUTPUT_CSV")) {
    printf("\n# CSV_OUTPUT_START\n");
    printf("array_length,cpu_time_ms,gpu_time_ms,speedup\n");
    printf("%d,%.6f,%.6f,%.2f\n", inputLength, cpu_time, gpu_time_ms, cpu_time / gpu_time_ms);
    printf("# CSV_OUTPUT_END\n");
  }


  //@@ Free memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  free(hostInput);
  free(hostOutput);


  return 0;
}

