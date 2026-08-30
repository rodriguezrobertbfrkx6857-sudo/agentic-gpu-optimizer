// Educational CUDA source used by the optimizer contract.
// The Python fallback in reference.py is the executable path on this CPU-only host.

#include <cuda_runtime.h>

__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) output[col * rows + row] = input[row * cols + col];
}

__global__ void transpose_tiled_padded(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33];
    const int col = blockIdx.x * 32 + threadIdx.x;
    const int row = blockIdx.y * 32 + threadIdx.y;
    if (row < rows && col < cols) tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    __syncthreads();
    const int output_col = blockIdx.y * 32 + threadIdx.x;
    const int output_row = blockIdx.x * 32 + threadIdx.y;
    if (output_row < cols && output_col < rows) output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
}

