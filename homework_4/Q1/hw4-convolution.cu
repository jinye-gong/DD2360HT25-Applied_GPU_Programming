#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = (stmt);                                      \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          printf("CUDA error: %s\n", cudaGetErrorString(err));       \
          exit(EXIT_FAILURE);                                        \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start() { gettimeofday(&t_start, 0); }

void cputimer_stop(const char* info) {
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH / 2)

__global__ void convolution_1D_basic(float *N, float *M, float *P, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float p_value = 0.0f;
    #pragma unroll
    for (int j = -RADIUS; j <= RADIUS; j++) {
        int n_index = i + j;
        if (n_index >= 0 && n_index < n) {
            p_value += N[n_index] * M[j + RADIUS];
        }
    }
    P[i] = p_value;
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int n)
{
    // dynamic shared memory: size = blockDim.x + 2*RADIUS
    extern __shared__ float input_tile[];

    int tx   = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    int gid  = base + tx;

    int sh_idx = tx + RADIUS;

    // center
    input_tile[sh_idx] = (gid < n) ? N[gid] : 0.0f;

    // left halo
    if (tx < RADIUS) {
        int left = gid - RADIUS;
        input_tile[tx] = (left >= 0) ? N[left] : 0.0f;
    }

    // right halo
    if (tx >= blockDim.x - RADIUS) {
        int right = gid + RADIUS;
        input_tile[sh_idx + RADIUS] = (right < n) ? N[right] : 0.0f;
    }

    __syncthreads();

    if (gid < n) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = -RADIUS; k <= RADIUS; k++) {
            sum += input_tile[sh_idx + k] * M[k + RADIUS];
        }
        P[gid] = sum;
    }
}

int main(int argc, char *argv[])
{
  if (argc < 2) {
    printf("Usage: %s N [BLOCK_SIZE]\n", argv[0]);
    return 0;
  }

  int N = atoi(argv[1]);
  if (N <= 0) {
    printf("N must be positive.\n");
    return 0;
  }

  int blockSize = (argc >= 3) ? atoi(argv[2]) : 256;
  if (blockSize <= 0 || blockSize > 1024) {
    printf("BLOCK_SIZE must be in (0, 1024].\n");
    return 0;
  }
  if (blockSize % 32 != 0) {
    printf("Warning: blockSize=%d is not a multiple of 32 (warp size).\n", blockSize);
  }

  // Host arrays
  float *hostN = nullptr;
  float *hostM = nullptr;
  float *hostP_basic = nullptr;
  float *hostP_tiled = nullptr;

  cputimer_start();
  hostN = (float*)malloc(N * sizeof(float));
  hostM = (float*)malloc(MASK_WIDTH * sizeof(float));
  hostP_basic = (float*)malloc(N * sizeof(float));
  hostP_tiled = (float*)malloc(N * sizeof(float));
  if (!hostN || !hostM || !hostP_basic || !hostP_tiled) {
      printf("ERROR: Host malloc failed!\n");
      free(hostN); free(hostM); free(hostP_basic); free(hostP_tiled);
      return 0;
  }
  cputimer_stop("Allocated host memory");

  // Device arrays
  float *deviceN = nullptr;
  float *deviceM = nullptr;
  float *deviceP = nullptr;

  cputimer_start();
  gpuCheck(cudaMalloc((void**)&deviceN, N * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceP, N * sizeof(float)));
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  cputimer_stop("Allocated device memory");

  // Init host data
  cputimer_start();
  srand(0);
  for (int i = 0; i < N; i++) hostN[i] = (float)rand() / (float)RAND_MAX;

  hostM[0] = -0.25f;
  hostM[1] = 0.5f;
  hostM[2] = 1.0f;
  hostM[3] = 0.5f;
  hostM[4] = 0.25f;

  for (int i = 0; i < N; i++) {
      hostP_basic[i] = 0.0f;
      hostP_tiled[i] = 0.0f;
  }
  cputimer_stop("Initialized host arrays");

  // Copy to device
  cputimer_start();
  gpuCheck(cudaMemcpy(deviceN, hostN, N * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  cputimer_stop("Copying data to the GPU.");

  dim3 block(blockSize);
  dim3 grid((N + block.x - 1) / block.x);
  printf("N=%d, blockSize=%d, grid=%d\n", N, blockSize, (int)grid.x);

  // CUDA events
  const int repeat = 100;
  cudaEvent_t eStart, eStop;
  gpuCheck(cudaEventCreate(&eStart));
  gpuCheck(cudaEventCreate(&eStop));

  // ---------------- Basic: warm-up ----------------
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  convolution_1D_basic<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());

  // ---------------- Basic: timed runs ----------------
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  gpuCheck(cudaEventRecord(eStart));
  for (int it = 0; it < repeat; it++) {
      convolution_1D_basic<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  }
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaEventRecord(eStop));
  gpuCheck(cudaEventSynchronize(eStop));

  float ms_basic = 0.0f;
  gpuCheck(cudaEventElapsedTime(&ms_basic, eStart, eStop));
  printf("Kernel time (basic): total %.6f ms, avg %.6f ms over %d runs\n",
         ms_basic, ms_basic / repeat, repeat);

  // Copy basic output
  cputimer_start();
  gpuCheck(cudaMemcpy(hostP_basic, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  int k = (N < 10) ? N : 10;
  printf("P_basic (first %d): ", k);
  for (int i = 0; i < k; i++) printf("%.4f ", hostP_basic[i]);
  printf("\n");
  cputimer_stop("Copying output P(basic) to the CPU and print out the results");

  // ---------------- Tiled: warm-up ----------------
  size_t shmem_bytes = (blockSize + 2 * RADIUS) * sizeof(float);

  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  convolution_1D_tiled<<<grid, block, shmem_bytes>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());

  // ---------------- Tiled: timed runs ----------------
  gpuCheck(cudaMemset(deviceP, 0, N * sizeof(float)));
  gpuCheck(cudaEventRecord(eStart));
  for (int it = 0; it < repeat; it++) {
      convolution_1D_tiled<<<grid, block, shmem_bytes>>>(deviceN, deviceM, deviceP, N);
  }
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaEventRecord(eStop));
  gpuCheck(cudaEventSynchronize(eStop));

  float ms_tiled = 0.0f;
  gpuCheck(cudaEventElapsedTime(&ms_tiled, eStart, eStop));
  printf("Kernel time (tiled): total %.6f ms, avg %.6f ms over %d runs\n",
         ms_tiled, ms_tiled / repeat, repeat);

  // Copy tiled output
  cputimer_start();
  gpuCheck(cudaMemcpy(hostP_tiled, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  int k2 = (N < 10) ? N : 10;
  printf("P_tiled (first %d): ", k2);
  for (int i = 0; i < k2; i++) printf("%.4f ", hostP_tiled[i]);
  printf("\n");
  cputimer_stop("Copying output P(tiled) to the CPU and print out the results");

  // Speedup and validation
  if (ms_tiled > 0.0f) {
      printf("Speedup (avg basic/tiled): %.3fx\n", (ms_basic / repeat) / (ms_tiled / repeat));
  }

  float max_abs_err = 0.0f;
  for (int i = 0; i < N; i++) {
      float diff = fabsf(hostP_basic[i] - hostP_tiled[i]);
      if (diff > max_abs_err) max_abs_err = diff;
  }
  printf("Validation: max abs error = %.8f\n", max_abs_err);
  printf("RESULT,%d,%d,%.6f,%.6f\n", N, blockSize, ms_basic/repeat, ms_tiled/repeat);


  // Cleanup
  cputimer_start();
  gpuCheck(cudaEventDestroy(eStart));
  gpuCheck(cudaEventDestroy(eStop));

  gpuCheck(cudaFree(deviceN));
  gpuCheck(cudaFree(deviceM));
  gpuCheck(cudaFree(deviceP));

  free(hostN);
  free(hostM);
  free(hostP_basic);
  free(hostP_tiled);
  cputimer_stop("Free memory resources");

  return 0;
}
