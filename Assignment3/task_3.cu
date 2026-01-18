#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

__global__ void coalesced(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * 2.0f;
    }
}

__global__ void non_coalesced(const float* a, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int access = (idx * stride) % n;
        out[idx] = a[access] * 2.0f;
    }
}

static float run_kernel_coalesced(const float* d_a, float* d_out, int n, int block) {
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    coalesced<<<grid, block>>>(d_a, d_out, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static float run_kernel_non_coalesced(const float* d_a, float* d_out, int n, int block, int stride) {
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    non_coalesced<<<grid, block>>>(d_a, d_out, n, stride);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1'000'000;
    const int block = 256;
    const int stride = 32; // intentionally breaks contiguous access

    std::vector<float> h_a(N), h_out(N);
    for (int i = 0; i < N; i++) h_a[i] = 0.1f * static_cast<float>(i);

    float *d_a = nullptr, *d_out = nullptr;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc d_out");
    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    // Warm-up
    coalesced<<<(N + block - 1) / block, block>>>(d_a, d_out, N);
    cudaDeviceSynchronize();

    float t_coal = run_kernel_coalesced(d_a, d_out, N, block);
    float t_non = run_kernel_non_coalesced(d_a, d_out, N, block, stride);

    // Validate (coalesced case)
    checkCuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
    for (int i : {0, 1, 2, 123, 999999}) {
        int access = (i * stride) % N;
        float expected = h_a[access] * 2.0f;
        if (std::fabs(h_out[i] - expected) > 1e-3f) {
            std::cerr << "Validation failed at i=" << i << ": got=" << h_out[i]
                      << " expected=" << expected << std::endl;
            return 2;
        }
    }

    std::cout << "N = " << N << ", block = " << block << ", stride = " << stride << "\n";
    std::cout << "Coalesced access time: " << t_coal << " ms\n";
    std::cout << "Non-coalesced access time: " << t_non << " ms\n";

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
