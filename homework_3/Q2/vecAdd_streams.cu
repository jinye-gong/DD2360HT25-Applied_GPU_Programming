#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

typedef float DataType;


#define CHECK(call) do {                                          \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",          \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                             \
    }                                                             \
} while (0)

// CPU timer function using gettimeofday()
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// ======== CPU reference implementation ========
void vecAddCPU(const DataType *in1, const DataType *in2,
               DataType *out, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = in1[i] + in2[i];
    }
}


// CUDA kernel for vector addition
__global__ void vecAdd(DataType*  in1, DataType*  in2,
                       DataType*  out, int len){
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) { 
	out[i] = in1[i] + in2[i]; 
	} 
}



// ======== Non-stream baseline (one big copy + one kernel) ========
double run_no_stream(const DataType *h_in1,
                     const DataType *h_in2,
                     DataType *h_out,
                    int N)
{
    size_t bytes = N * sizeof(DataType);

    DataType *d_in1 = nullptr;
    DataType *d_in2 = nullptr;
    DataType *d_out = nullptr;

    CHECK(cudaMalloc((void**)&d_in1, bytes));
    CHECK(cudaMalloc((void**)&d_in2, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    double t0 = cpuSecond();

    // H2D (synchronous)
    CHECK(cudaMemcpy(d_in1, h_in1, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in2, h_in2, bytes, cudaMemcpyHostToDevice));

    // Kernel
    vecAdd<<<blocks, threads>>>(d_in1, d_in2, d_out, N);
    CHECK(cudaGetLastError());

    // D2H (synchronous)
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());
    double t1 = cpuSecond();

    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));

    return (t1 - t0) * 1000.0; // ms
}


// ======== Stream version: 4 streams + segmentation + async copies ========
double run_with_streams(const DataType *h_in1,
                        const DataType *h_in2,
                        DataType *h_out,
                        int N,
                        int S_seg,
                        int nStreams)
{
    size_t bytes = N * sizeof(DataType);

    DataType *d_in1 = nullptr;
    DataType *d_in2 = nullptr;
    DataType *d_out = nullptr;

    CHECK(cudaMalloc((void**)&d_in1, bytes));
    CHECK(cudaMalloc((void**)&d_in2, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    cudaStream_t *streams = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0; i < nStreams; ++i) {
        CHECK(cudaStreamCreate(&streams[i]));
    }

    int threads = 256;

    int nSegments = (N + S_seg - 1) / S_seg; // ceil(N / S_seg)

    double t0 = cpuSecond();

    for (int seg = 0; seg < nSegments; ++seg) {
        int streamId = seg % nStreams;
        cudaStream_t s = streams[streamId];

        int offset = seg * S_seg;
        int len = S_seg;
        if (offset + len > N) {
            len = N - offset;
        }

        size_t segBytes = len * sizeof(DataType);

        // Async H2D
        CHECK(cudaMemcpyAsync(d_in1 + offset, h_in1 + offset, segBytes,
                            cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(d_in2 + offset, h_in2 + offset, segBytes,
                            cudaMemcpyHostToDevice, s));

        // Kernel on this segment
        int blocks = (len + threads - 1) / threads;
        vecAdd<<<blocks, threads, 0, s>>>(d_in1 + offset,
                                        d_in2 + offset,
                                        d_out + offset,
                                        len);
        CHECK(cudaGetLastError());

        // Async D2H
        CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, segBytes,
                            cudaMemcpyDeviceToHost, s));
    }

    // Wait all streams
    for (int i = 0; i < nStreams; ++i) {
        CHECK(cudaStreamSynchronize(streams[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    CHECK(cudaDeviceSynchronize());
    double t1 = cpuSecond();

    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));

    return (t1 - t0) * 1000.0; // ms
}



int main(int argc, char **argv)
{
    // ======== Parse arguments ========
    int N = 1 << 24;      // default vector length
    int S_seg = 1 << 20;  // default segment size
    int nStreams = 4;     // fixed at 4 streams (as required)

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        S_seg = atoi(argv[2]);
    }

    printf("Vector length N   = %d\n", N);
    printf("Segment size S_seg= %d\n", S_seg);
    printf("Number of streams = %d\n", nStreams);

    size_t bytes = N * sizeof(DataType);

    // ======== Allocate host pinned memory ========
    // Pinned memory is important for real async & overlap
    DataType *h_in1, *h_in2, *h_out, *h_ref;
    CHECK(cudaMallocHost((void**)&h_in1, bytes)); // pinned
    CHECK(cudaMallocHost((void**)&h_in2, bytes)); // pinned
    CHECK(cudaMallocHost((void**)&h_out, bytes)); // pinned
    CHECK(cudaMallocHost((void**)&h_ref, bytes)); // pinned

    // ======== Initialize input data ========
    for (int i = 0; i < N; ++i) {
        h_in1[i] = static_cast<DataType>((rand() % 100) / 10.0f);
        h_in2[i] = static_cast<DataType>((rand() % 100) / 10.0f);
        h_out[i] = 0.0f;
        h_ref[i] = 0.0f;
    }

    // ======== CPU reference ========
    double t_cpu0 = cpuSecond();
    vecAddCPU(h_in1, h_in2, h_ref, N);
    double t_cpu1 = cpuSecond();
    double cpu_ms = (t_cpu1 - t_cpu0) * 1000.0;
    printf("CPU time          = %.3f ms\n", cpu_ms);

    // ======== Non-stream version ========
    double t_no_stream = run_no_stream(h_in1, h_in2, h_out, N);

    // check correctness
    double maxError = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(h_out[i] - h_ref[i]);
        if (err > maxError) maxError = err;
    }

    printf("No-stream GPU time= %.3f ms, max error = %e\n",
        t_no_stream, maxError);

    // ======== Stream version ========
    // reset output buffer
    for (int i = 0; i < N; ++i) {
        h_out[i] = 0.0f;
    }

    double t_stream = run_with_streams(h_in1, h_in2, h_out, N, S_seg, nStreams);

    // check correctness again
    maxError = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(h_out[i] - h_ref[i]);
        if (err > maxError) maxError = err;
    }

    printf("4-stream GPU time = %.3f ms, max error = %e\n",
        t_stream, maxError);

    printf("Speedup (no-stream / streams) = %.3f x\n",
        t_no_stream / t_stream);

    // ======== Free host memory ========
    CHECK(cudaFreeHost(h_in1));
    CHECK(cudaFreeHost(h_in2));
    CHECK(cudaFreeHost(h_out));
    CHECK(cudaFreeHost(h_ref));

    return 0;
}
