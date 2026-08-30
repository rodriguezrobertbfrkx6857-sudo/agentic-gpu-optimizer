#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace optimizer_transpose {

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                                 std::to_string(line) + ": " + cudaGetErrorString(status));
    }
}

#define OPTIMIZER_CUDA_CHECK(expression) \
    ::optimizer_transpose::check_cuda((expression), #expression, __FILE__, __LINE__)

struct Options {
    std::string variant;
    int size = 192;
    int warmup = 3;
    int iterations = 8;
    bool correctness = false;
    bool benchmark = false;
};

inline Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--variant" && index + 1 < argc) {
            options.variant = argv[++index];
        } else if (argument == "--size" && index + 1 < argc) {
            options.size = std::stoi(argv[++index]);
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = std::stoi(argv[++index]);
        } else if (argument == "--iterations" && index + 1 < argc) {
            options.iterations = std::stoi(argv[++index]);
        } else if (argument == "--mode" && index + 1 < argc) {
            const std::string mode = argv[++index];
            if (mode == "correctness") options.correctness = true;
            else if (mode == "benchmark") options.benchmark = true;
            else throw std::invalid_argument("unknown mode: " + mode);
        } else if (argument == "--correctness") {
            options.correctness = true;
        } else if (argument == "--benchmark") {
            options.benchmark = true;
        } else if (argument == "--help") {
            std::cout << "usage: workload --variant NAME --size N [--correctness|--benchmark] "
                         "[--warmup N --iterations N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
    }
    if (options.variant.empty()) throw std::invalid_argument("--variant is required");
    if (options.size <= 0 || options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument("size must be positive, warmup non-negative, iterations positive");
    }
    if (options.correctness == options.benchmark) {
        throw std::invalid_argument("select exactly one of --correctness or --benchmark");
    }
    return options;
}

inline std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\') escaped += "\\\\";
        else if (character == '"') escaped += "\\\"";
        else escaped += character;
    }
    return escaped;
}

template <typename Launch>
int run_workload(const Options& options, const char* expected_variant, Launch launch) {
    if (options.variant != expected_variant) {
        throw std::invalid_argument("binary expects variant " + std::string(expected_variant));
    }
    const std::size_t size = static_cast<std::size_t>(options.size);
    const std::size_t elements = size * size;
    const std::size_t bytes = elements * sizeof(float);
    std::vector<float> host_input(elements);
    std::vector<float> host_expected(elements);
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t col = 0; col < size; ++col) {
            const float value = std::sin(static_cast<float>(row * size + col) * 0.001f);
            host_input[row * size + col] = value;
            host_expected[col * size + row] = value;
        }
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    OPTIMIZER_CUDA_CHECK(cudaMalloc(&device_input, bytes));
    OPTIMIZER_CUDA_CHECK(cudaMalloc(&device_output, bytes));
    OPTIMIZER_CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    launch(device_input, device_output, options.size, options.size, nullptr);
    OPTIMIZER_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(elements);
    OPTIMIZER_CUDA_CHECK(cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    for (std::size_t index = 0; index < elements; ++index) {
        max_abs_error = std::max(max_abs_error,
                                 std::fabs(static_cast<double>(actual[index]) - host_expected[index]));
    }
    const bool correct = max_abs_error == 0.0;

    cudaDeviceProp properties{};
    int device = 0;
    OPTIMIZER_CUDA_CHECK(cudaGetDevice(&device));
    OPTIMIZER_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::cout << std::setprecision(9);
    if (options.correctness) {
        std::cout << "{\"variant\":\"" << expected_variant << "\",\"size\":" << options.size
                  << ",\"shape\":[" << options.size << "," << options.size
                  << "],\"dtype\":\"float32\",\"device\":\""
                  << json_escape(properties.name)
                  << "\",\"backend\":\"cuda\",\"hardware_mode\":\"cuda\""
                  << ",\"correctness_pass\":" << (correct ? "true" : "false")
                  << ",\"max_abs_error\":" << max_abs_error
                  << ",\"status\":\"" << (correct ? "CORRECTNESS_PASS" : "CORRECTNESS_FAIL")
                  << "\"}\n";
    } else {
        for (int index = 0; index < options.warmup; ++index) {
            launch(device_input, device_output, options.size, options.size, nullptr);
        }
        OPTIMIZER_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start{};
        cudaEvent_t stop{};
        OPTIMIZER_CUDA_CHECK(cudaEventCreate(&start));
        OPTIMIZER_CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(options.iterations));
        for (int index = 0; index < options.iterations; ++index) {
            OPTIMIZER_CUDA_CHECK(cudaEventRecord(start));
            launch(device_input, device_output, options.size, options.size, nullptr);
            OPTIMIZER_CUDA_CHECK(cudaEventRecord(stop));
            OPTIMIZER_CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            OPTIMIZER_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            samples.push_back(milliseconds);
        }
        OPTIMIZER_CUDA_CHECK(cudaEventDestroy(start));
        OPTIMIZER_CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(samples.begin(), samples.end());
        double mean = 0.0;
        for (float sample : samples) mean += sample;
        mean /= static_cast<double>(samples.size());
        double variance = 0.0;
        for (float sample : samples) {
            const double delta = sample - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(samples.size());
        const double median = samples.size() % 2 == 0
                                  ? (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]) * 0.5
                                  : samples[samples.size() / 2];
        const double bandwidth = (2.0 * static_cast<double>(bytes)) / (median * 1.0e6);
        std::cout << "{\"variant\":\"" << expected_variant << "\",\"size\":" << options.size
                  << ",\"shape\":[" << options.size << "," << options.size
                  << "],\"dtype\":\"float32\",\"device\":\""
                  << json_escape(properties.name)
                  << "\",\"backend\":\"cuda\",\"hardware_mode\":\"cuda\""
                  << ",\"warmup\":" << options.warmup << ",\"iterations\":" << options.iterations
                  << ",\"median_ms\":" << median << ",\"mean_ms\":" << mean
                  << ",\"min_ms\":" << samples.front() << ",\"std_ms\":" << std::sqrt(variance)
                  << ",\"effective_bandwidth_gbps\":" << bandwidth
                  << ",\"correctness_pass\":" << (correct ? "true" : "false")
                  << ",\"max_abs_error\":" << max_abs_error
                  << ",\"status\":\"BENCHMARKED_CUDA\"}\n";
    }
    OPTIMIZER_CUDA_CHECK(cudaFree(device_input));
    OPTIMIZER_CUDA_CHECK(cudaFree(device_output));
    return correct ? 0 : 1;
}

}  // namespace optimizer_transpose
