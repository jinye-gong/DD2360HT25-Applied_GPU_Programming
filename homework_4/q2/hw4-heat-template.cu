#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <cusparse.h>
#include <cublas_v2.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)


static std::chrono::high_resolution_clock::time_point t_start;
void cputimer_start() {
  t_start = std::chrono::high_resolution_clock::now();
}
void cputimer_stop(const char* info) {
  const auto t_end = std::chrono::high_resolution_clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
  std::printf("Timing - %s. \t\tElasped %lld microseconds \n", info, (long long)us);
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX,
    double alpha) {
  // Stencil from the finete difference discretization of the equation
  double stencil[] = { 1, -2, 1 };
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

__global__ void initExactSolution(double* exact, int n, double left, double right) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (n <= 1) {
    exact[0] = left;
    return;
  }
  const double t = (double)i / (double)(n - 1);
  exact[i] = left + t * (right - left);
}

int main(int argc, char **argv) {
  int device = 0;            // Device to be used
  int dimX;                  // Dimension of the metal rod
  int nsteps;                // Number of time steps to perform
  double alpha = 0.4;        // Diffusion coefficient
  double* temp;              // Array to store the final time step
  double* A;                 // Sparse matrix A values in the CSR format
  int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
  int* AColIndx;             // Sparse matrix A col values in the CSR format
  int nzv;                   // Number of non zero values in the sparse matrix
  double* tmp;               // Temporal array of dimX for computations
  size_t bufferSize = 0;     // Buffer size needed by some routines
  void* buffer = nullptr;    // Buffer used by some routines in the libraries
  int concurrentAccessQ;     // Check if concurrent access flag is set
  double zero = 0;           // Zero constant
  double one = 1;            // One constant
  double norm;               // Variable for norm values
  double error;              // Variable for storing the relative error
  double tempLeft = 200.;    // Left heat source applied to the rod
  double tempRight = 300.;   // Right heat source applied to the rod
  cublasHandle_t cublasHandle;      // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseMatDescr_t Adescriptor;   // Mat descriptor needed by cuSPARSE
  cusparseSpMatDescr_t A_spmat = nullptr;  // cuSPARSE sparse matrix descriptor (generic API)
  cusparseDnVecDescr_t temp_vec = nullptr; // cuSPARSE dense vector descriptors
  cusparseDnVecDescr_t tmp_vec = nullptr;
  cudaStream_t stream = 0;
  auto prefetchToHost = [&](const void* p, size_t bytes) {
    gpuCheck(cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, stream));
  };
  auto prefetchToDevice = [&](const void* p, size_t bytes) {
    gpuCheck(cudaMemPrefetchAsync(p, bytes, device, stream));
  };

  // Read the arguments from the command line
  if (argc < 3) {
    std::printf("Usage: %s <dimX> <nsteps> [--no-prefetch]\n", argv[0]);
    return 1;
  }
  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);
  bool enablePrefetch = true;
  if (argc >= 4 && std::string(argv[3]) == "--no-prefetch") {
    enablePrefetch = false;
  }

  // Print input arguments
  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  //@@ Insert the code to allocate the temp, tmp and the sparse matrix
  //@@ arrays using Unified Memory
  cputimer_start();
  gpuCheck(cudaMallocManaged(&temp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&tmp, sizeof(double) * dimX));
  gpuCheck(cudaMallocManaged(&A, sizeof(double) * nzv));
  gpuCheck(cudaMallocManaged(&ARowPtr, sizeof(int) * (dimX + 1)));
  gpuCheck(cudaMallocManaged(&AColIndx, sizeof(int) * nzv));

  cputimer_stop("Allocating device memory");

  // Check if concurrentAccessQ is non zero in order to prefetch memory
  if (concurrentAccessQ && enablePrefetch) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
    prefetchToHost(temp, sizeof(double) * dimX);
    prefetchToHost(tmp, sizeof(double) * dimX);
    prefetchToHost(A, sizeof(double) * nzv);
    prefetchToHost(ARowPtr, sizeof(int) * (dimX + 1));
    prefetchToHost(AColIndx, sizeof(int) * nzv);
    gpuCheck(cudaStreamSynchronize(stream));

    cputimer_stop("Prefetching GPU memory to the host");
  }

  // Initialize the sparse matrix
  cputimer_start();
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
  cputimer_stop("Initializing the sparse matrix on the host");

  //Initiliaze the boundary conditions for the heat equation
  cputimer_start();
  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;
  cputimer_stop("Initializing memory on the host");

  if (concurrentAccessQ && enablePrefetch) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
    prefetchToDevice(temp, sizeof(double) * dimX);
    prefetchToDevice(tmp, sizeof(double) * dimX);
    prefetchToDevice(A, sizeof(double) * nzv);
    prefetchToDevice(ARowPtr, sizeof(int) * (dimX + 1));
    prefetchToDevice(AColIndx, sizeof(int) * nzv);
    gpuCheck(cudaStreamSynchronize(stream));

    cputimer_stop("Prefetching GPU memory to the device");
  }

  //@@ Insert code to create the cuBLAS handle
  cublasCheck(cublasCreate(&cublasHandle));

  //@@ Insert code to create the cuSPARSE handle
  cusparseCheck(cusparseCreate(&cusparseHandle));
  // Use host pointers for alpha/beta scalars passed to cuSPARSE routines
  cusparseCheck(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

  //@@ Insert code to set the cuBLAS pointer mode to CUBLAS_POINTER_MODE_HOST
  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

  //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
  cusparseCheck(cusparseCreateMatDescr(&Adescriptor));
  cusparseCheck(cusparseSetMatType(Adescriptor, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseCheck(cusparseSetMatIndexBase(Adescriptor, CUSPARSE_INDEX_BASE_ZERO));

  // Create generic SpMV descriptors (CSR + dense vectors)
  cusparseCheck(cusparseCreateCsr(&A_spmat,
                                 dimX, dimX, nzv,
                                 ARowPtr, AColIndx, A,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&temp_vec, dimX, temp, CUDA_R_64F));
  cusparseCheck(cusparseCreateDnVec(&tmp_vec, dimX, tmp, CUDA_R_64F));

  //@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per
  //@@ vector (SMPV) CSR routine of cuSPARSE
  {
    const double alphaSpmv = one;
    const double betaSpmv = zero;
    cusparseCheck(cusparseSpMV_bufferSize(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaSpmv,
        A_spmat,
        temp_vec,
        &betaSpmv,
        tmp_vec,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));
  }


  //@@ Insert code to allocate the buffer needed by cuSPARSE
  if (bufferSize > 0) {
    gpuCheck(cudaMalloc(&buffer, bufferSize));
  }


  // Perform the time step iterations
  auto iter_start = std::chrono::high_resolution_clock::now();
  int actual_iterations = 0;
  for (int it = 0; it < nsteps; ++it) {
    //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
    //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
    //@@ tmp = 1 * A * temp + 0 * tmp
    {
      const double alphaSpmv = one;
      const double betaSpmv = zero;
      cusparseCheck(cusparseSpMV(
          cusparseHandle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alphaSpmv,
          A_spmat,
          temp_vec,
          &betaSpmv,
          tmp_vec,
          CUDA_R_64F,
          CUSPARSE_SPMV_ALG_DEFAULT,
          buffer));
    }

    //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
    //@@ This calculation corresponds to: temp = alpha * tmp + temp
    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));

    //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
    //@@ This calculation corresponds to: ||temp||
    cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));

    actual_iterations++;
    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4)
      break;
  }
  auto iter_end = std::chrono::high_resolution_clock::now();
  const auto iteration_time_us = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();
  std::printf("Timing - Time step iterations. \t\tElasped %lld microseconds \n", (long long)iteration_time_us);
  
  // Output timing information for FLOPS calculation (parseable format)
  printf("ITERATION_TIME_US: %lld\n", (long long)iteration_time_us);
  printf("ACTUAL_ITERATIONS: %d\n", actual_iterations);

  // Calculate the exact solution on the GPU: linear ramp between boundaries
  {
    const int threads = 256;
    const int blocks = (dimX + threads - 1) / threads;
    initExactSolution<<<blocks, threads, 0, stream>>>(tmp, dimX, tempLeft, tempRight);
    gpuCheck(cudaGetLastError());
    gpuCheck(cudaStreamSynchronize(stream));
  }

  // Calculate the relative approximation error:
  one = -1;
  //@@ Insert the code to call cublas api to compute the difference between the exact solution
  //@@ and the approximation
  //@@ This calculation corresponds to: tmp = -temp + tmp
  cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));

  //@@ Insert the code to call cublas api to compute the norm of the absolute error
  //@@ This calculation corresponds to: || tmp ||
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));

  error = norm;
  //@@ Insert the code to call cublas api to compute the norm of temp
  //@@ This calculation corresponds to: || temp ||
  cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

  // Calculate the relative error
  error = error / norm;
  printf("The relative error of the approximation is %f\n", error);

  //@@ Insert the code to destroy the mat descriptor
  if (tmp_vec) cusparseCheck(cusparseDestroyDnVec(tmp_vec));
  if (temp_vec) cusparseCheck(cusparseDestroyDnVec(temp_vec));
  if (A_spmat) cusparseCheck(cusparseDestroySpMat(A_spmat));
  cusparseCheck(cusparseDestroyMatDescr(Adescriptor));

  //@@ Insert the code to destroy the cuSPARSE handle
  cusparseCheck(cusparseDestroy(cusparseHandle));

  //@@ Insert the code to destroy the cuBLAS handle
  cublasCheck(cublasDestroy(cublasHandle));


  //@@ Insert the code for deallocating memory
  if (buffer) gpuCheck(cudaFree(buffer));
  gpuCheck(cudaFree(temp));
  gpuCheck(cudaFree(tmp));
  gpuCheck(cudaFree(A));
  gpuCheck(cudaFree(ARowPtr));
  gpuCheck(cudaFree(AColIndx));



  return 0;
}
