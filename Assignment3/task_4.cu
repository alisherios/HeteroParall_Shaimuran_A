#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static float time_add(const float* d_a, const float* d_b, float* d_c, int n, int block) {
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    add_arrays<<<grid, block>>>(d_a, d_b, d_c, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "eventRecord stop");

    checkCuda(cudaEventSynchronize(stop), "eventSynchronize stop");
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "eventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1'000'000;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 0.1f * static_cast<float>(i);
        h_b[i] = 1.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, N * sizeof(float)), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, N * sizeof(float)), "cudaMalloc d_c");

    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    // Candidate configurations
    const int blocks[] = {64, 128, 192, 256, 320, 384, 512, 768, 1024};

    float best_t = std::numeric_limits<float>::infinity();
    int best_block = -1;

    std::cout << "N = " << N << "\n";
    std::cout << "Testing block sizes (ms):\n";

    for (int b : blocks) {
        // Some GPUs may not support 1024 threads/block for all kernels; catch launch error.
        float t = time_add(d_a, d_b, d_c, N, b);
        std::cout << "  block = " << b << " -> " << t << " ms\n";
        if (t < best_t) {
            best_t = t;
            best_block = b;
        }
    }

    // Compare an explicitly "non-optimal" configuration vs the best found one
    int non_opt = 64;
    float t_non = time_add(d_a, d_b, d_c, N, non_opt);
    float t_opt = time_add(d_a, d_b, d_c, N, best_block);

    // Validate (using optimized run result)
    checkCuda(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H c");
    for (int i : {0, 1, 2, 123, 999999}) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-3f) {
            std::cerr << "Validation failed at i=" << i << ": got=" << h_c[i]
                      << " expected=" << expected << std::endl;
            return 2;
        }
    }

    std::cout << "\nBest block size: " << best_block << " (" << best_t << " ms)\n";
    std::cout << "\nComparison:\n";
    std::cout << "  Non-optimal (block=" << non_opt << ") : " << t_non << " ms\n";
    std::cout << "  Optimized   (block=" << best_block << ") : " << t_opt << " ms\n";
    if (t_opt > 0.0f) {
        std::cout << "  Speedup: " << (t_non / t_opt) << "x\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
