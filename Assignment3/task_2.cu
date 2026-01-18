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

__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static float run_add(const float* d_a, const float* d_b, float* d_c, int n, int block) {
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");

    checkCuda(cudaEventRecord(start), "eventRecord start");
    add_arrays<<<grid, block>>>(d_a, d_b, d_c, n);
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

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = 0.25f * static_cast<float>(i);
        h_b[i] = 1.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, N * sizeof(float)), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, N * sizeof(float)), "cudaMalloc d_c");

    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    int blocks[] = {128, 256, 512};

    std::cout << "N = " << N << "\n";
    for (int block : blocks) {
        float t = run_add(d_a, d_b, d_c, N, block);
        std::cout << "Block size = " << block << " -> time: " << t << " ms\n";
    }

    // Validate with the last run configuration (512)
    checkCuda(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H c");
    for (int i : {0, 1, 2, 123, 999999}) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-3f) {
            std::cerr << "Validation failed at i=" << i << ": got=" << h_c[i]
                      << " expected=" << expected << std::endl;
            return 2;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
