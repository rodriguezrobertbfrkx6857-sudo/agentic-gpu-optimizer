#include "runner_common.cuh"

namespace {

__global__ void transpose_naive_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) output[col * rows + row] = input[row * cols + col];
}

void launch_transpose_naive(const float* input, float* output, int rows, int cols,
                            cudaStream_t stream) {
    constexpr int tile = 32;
    const dim3 block(tile, tile);
    const dim3 grid((cols + tile - 1) / tile, (rows + tile - 1) / tile);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    OPTIMIZER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = optimizer_transpose::parse_options(argc, argv);
        return optimizer_transpose::run_workload(options, "baseline", launch_transpose_naive);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
