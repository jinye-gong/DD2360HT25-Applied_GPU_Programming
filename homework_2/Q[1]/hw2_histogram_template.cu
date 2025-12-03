
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_BINS 4096
#define THREADS_PER_BLOCK 256
#define MAX_BIN_COUNT 127

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

    //@@ Insert code below to compute histogram of input using shared memory and atomics
    
    // Shared memory for per-block histogram
    extern __shared__ unsigned int s_bins[];
    
    // Initialize shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory bins to zero
    for (int i = tid; i < num_bins; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();
    
    // Process input elements assigned to this thread
    if (idx < num_elements) {
        unsigned int bin = input[idx] % num_bins;
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();
    
    // Merge shared memory histogram into global memory
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (s_bins[i] > 0) {
            atomicAdd(&bins[i], s_bins[i]);
        }
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    //@@ Insert code below to clean up bins that saturate at 127
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_bins) {
        if (bins[idx] > MAX_BIN_COUNT) {
            bins[idx] = MAX_BIN_COUNT;
        }
    }
}


// CPU reference implementation
__host__ void histogram_cpu(unsigned int *input, unsigned int *bins, 
                   unsigned int num_elements, unsigned int num_bins) {
    // Initialize bins
    for (unsigned int i = 0; i < num_bins; i++) {
        bins[i] = 0;
    }
    
    // Compute histogram
    for (unsigned int i = 0; i < num_elements; i++) {
        unsigned int bin = input[i] % num_bins;
        bins[bin]++;
    }
    
    // Apply saturation
    for (unsigned int i = 0; i < num_bins; i++) {
        if (bins[i] > MAX_BIN_COUNT) {
            bins[i] = MAX_BIN_COUNT;
        }
    }
}

// Get time in milliseconds
__host__ double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
  
  int inputLength;
  int distribution_type = 0; // 0 = uniform, 1 = normal
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc < 2) {
    printf("Usage: %s <input_length> [distribution_type: 0=uniform, 1=normal]\n", argv[0]);
    return 1;
  }
  inputLength = atoi(argv[1]);
  if (argc >= 3) {
    distribution_type = atoi(argv[2]);
  }

  printf("The input length is %d\n", inputLength);
  printf("Distribution type: %s\n", distribution_type == 0 ? "uniform" : "normal");
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  
  if (!hostInput || !hostBins || !resultRef) {
    printf("Error: Failed to allocate host memory\n");
    return 1;
  }

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::random_device rd;
  std::mt19937 gen(rd());
  
  if (distribution_type == 0) {
    // Uniform distribution
    std::uniform_int_distribution<unsigned int> dis(0, NUM_BINS - 1);
    for (int i = 0; i < inputLength; i++) {
      hostInput[i] = dis(gen);
    }
  } else {
    // Normal distribution (centered around NUM_BINS/2)
    std::normal_distribution<double> dis(NUM_BINS / 2.0, NUM_BINS / 6.0);
    for (int i = 0; i < inputLength; i++) {
      double val = dis(gen);
      // Clamp to [0, NUM_BINS-1]
      if (val < 0) val = 0;
      if (val >= NUM_BINS) val = NUM_BINS - 1;
      hostInput[i] = (unsigned int)val;
    }
  }


  //@@ Insert code below to create reference result in CPU
  double cpu_start = get_time_ms();
  histogram_cpu(hostInput, resultRef, inputLength, NUM_BINS);
  double cpu_end = get_time_ms();
  double cpu_time = cpu_end - cpu_start;
  printf("CPU time: %.3f ms\n", cpu_time);


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: Failed to allocate device memory: %s\n", cudaGetErrorString(err));
    return 1;
  }


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);


  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(THREADS_PER_BLOCK);
  dim3 gridSize((inputLength + blockSize.x - 1) / blockSize.x);
  size_t sharedMemSize = NUM_BINS * sizeof(unsigned int);


  //@@ Launch the GPU Kernel here
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);
  histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
      deviceInput, deviceBins, inputLength, NUM_BINS);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: Histogram kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }


  //@@ Initialize the second grid and block dimensions here
  dim3 convertBlockSize(THREADS_PER_BLOCK);
  dim3 convertGridSize((NUM_BINS + convertBlockSize.x - 1) / convertBlockSize.x);


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<convertGridSize, convertBlockSize>>>(deviceBins, NUM_BINS);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: Convert kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float gpu_time_ms;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);
  printf("GPU time: %.3f ms\n", gpu_time_ms);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  bool match = true;
  int mismatch_count = 0;
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      if (mismatch_count < 10) {
        printf("Mismatch at bin %d: GPU=%u, CPU=%u\n", i, hostBins[i], resultRef[i]);
      }
      mismatch_count++;
      match = false;
    }
  }
  
  if (match) {
    printf("Results match!\n");
  } else {
    printf("Results do not match! Total mismatches: %d\n", mismatch_count);
  }
  
  // Print some statistics
  unsigned int max_bin = 0, max_count = 0;
  unsigned int total_count = 0;
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] > max_count) {
      max_count = hostBins[i];
      max_bin = i;
    }
    total_count += hostBins[i];
  }
  printf("Max bin: %u with count: %u\n", max_bin, max_count);
  printf("Total count: %u (expected: %d)\n", total_count, inputLength);
  
  // Output histogram data in CSV format for plotting
  if (getenv("OUTPUT_CSV")) {
    printf("\n# CSV_OUTPUT_START\n");
    printf("bin,count\n");
    for (int i = 0; i < NUM_BINS; i++) {
      printf("%d,%u\n", i, hostBins[i]);
    }
    printf("# CSV_OUTPUT_END\n");
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);


  return 0;
}

