#include "runner_common.cuh"

namespace {

__global__ void reduction_baseline_kernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(output, input[index]);
}

void launch_reduction_baseline(const float* input, float* output, int count, cudaStream_t stream) {
    constexpr int block_size = 256;
    OPTIMIZER_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    const int blocks = (count + block_size - 1) / block_size;
    reduction_baseline_kernel<<<blocks, block_size, 0, stream>>>(input, output, count);
    OPTIMIZER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = optimizer_reduction::parse_options(argc, argv);
        return optimizer_reduction::run_workload(options, "baseline", launch_reduction_baseline);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
