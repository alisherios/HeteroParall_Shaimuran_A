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

__global__ void mul_global(float* a, float k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= k;
    }
}

__global__ void mul_shared(float* a, float k, int n) {
    extern __shared__ float buf[]; // size = blockDim.x * sizeof(float)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        buf[tid] = a[idx];
        __syncthreads();

        buf[tid] *= k;
        __syncthreads();

        a[idx] = buf[tid];
    }
}

static float run_mul_global(float* d_a, float k, int n, int block) {
    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    checkCuda(cudaEventRecord(start), "event record start");
    mul_global<<<grid, block>>>(d_a, k, n);
    checkCuda(cudaEventRecord(stop), "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static float run_mul_shared(float* d_a, float k, int n, int block) {
    int grid = (n + block - 1) / block;
    size_t shmem = static_cast<size_t>(block) * sizeof(float);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    checkCuda(cudaEventRecord(start), "event record start");
    mul_shared<<<grid, block, shmem>>>(d_a, k, n);
    checkCuda(cudaEventRecord(stop), "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1'000'000;
    const float k = 2.0f;
    const int block = 256;

    std::vector<float> h_a(N);
    for (int i = 0; i < N; ++i) h_a[i] = 0.1f * static_cast<float>(i);

    float* d_a = nullptr;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");

    // --- Global memory version ---
    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    float t_global = run_mul_global(d_a, k, N, block);

    std::vector<float> h_out_global(N);
    checkCuda(cudaMemcpy(h_out_global.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    // --- Shared memory version ---
    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    float t_shared = run_mul_shared(d_a, k, N, block);

    std::vector<float> h_out_shared(N);
    checkCuda(cudaMemcpy(h_out_shared.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    // Simple validation
    auto check_sample = [&](const std::vector<float>& v, const char* tag) {
        for (int i : {0, 1, 2, 123, 999999}) {
            float expected = (0.1f * static_cast<float>(i)) * k;
            if (std::fabs(v[i] - expected) > 1e-3f) {
                std::cerr << "Validation failed (" << tag << ") at i=" << i
                          << ": got=" << v[i] << " expected=" << expected << std::endl;
                std::exit(2);
            }
        }
    };

    check_sample(h_out_global, "global");
    check_sample(h_out_shared, "shared");

    std::cout << "N = " << N << ", block = " << block << "\n";
    std::cout << "Global memory time: " << t_global << " ms\n";
    std::cout << "Shared memory time: " << t_shared << " ms\n";

    cudaFree(d_a);
    return 0;
}
