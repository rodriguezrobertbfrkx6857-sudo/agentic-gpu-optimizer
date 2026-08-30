// Educational CUDA source used by the optimizer contract.
// The Python fallback in reference.py is the executable path on this CPU-only host.

#include <cuda_runtime.h>

__global__ void reduction_baseline(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(output, input[index]);
}

__global__ void reduction_precision_candidate(const float* input, double* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(output, static_cast<double>(input[index]));
}

