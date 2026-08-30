from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np


def baseline(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.empty((cols, rows), dtype=matrix.dtype)
    for row in range(rows):
        for col in range(cols):
            output[col, row] = matrix[row, col]
    return output


def candidate(matrix: np.ndarray) -> np.ndarray:
    # CPU analogue of the CUDA 32x33 shared-memory tile. The benchmark still
    # measures this implementation honestly as a CPU reference, not as GPU data.
    rows, cols = matrix.shape
    tile = 32
    output = np.empty((cols, rows), dtype=matrix.dtype)
    for row_start in range(0, rows, tile):
        for col_start in range(0, cols, tile):
            row_end = min(row_start + tile, rows)
            col_end = min(col_start + tile, cols)
            output[col_start:col_end, row_start:row_end] = matrix[
                row_start:row_end, col_start:col_end
            ].T
    return output


def run(variant: str, size: int, warmup: int, iterations: int, benchmark: bool) -> dict:
    matrix = np.arange(size * size, dtype=np.float32).reshape(size, size)
    expected = matrix.T.copy()
    function = baseline if variant == "baseline" else candidate if variant == "transpose_tiled_padded" else None
    if function is None:
        raise ValueError(f"unknown transpose variant: {variant}")
    output = function(matrix)
    error = float(np.max(np.abs(output - expected)))
    result = {
        "correctness_pass": bool(np.array_equal(output, expected)),
        "max_abs_error": error,
        "variant": variant,
        "size": size,
        "status": "CORRECTNESS_ONLY_CPU",
        "backend": "numpy_cpu_reference",
        "hardware_mode": "cpu_only",
        "device": platform.processor() or platform.machine(),
        "dtype": str(matrix.dtype),
        "warmup": warmup,
        "iterations": iterations,
        "timer": "perf_counter_ns",
        "cuda_validated": False,
    }
    if benchmark:
        for _ in range(warmup):
            function(matrix)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            function(matrix)
            samples.append((time.perf_counter_ns() - start) / 1.0e6)
        values = np.asarray(samples, dtype=np.float64)
        result.update({"status": "BENCHMARKED_CPU_ONLY", "median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)), "min_ms": float(np.min(values)), "std_ms": float(np.std(values))})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--mode", choices=("correctness", "benchmark"))
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.mode and (args.correctness or args.benchmark):
        parser.error("use --mode or an explicit --correctness/--benchmark flag, not both")
    if args.mode == "correctness":
        args.correctness = True
    elif args.mode == "benchmark":
        args.benchmark = True
    if args.correctness == args.benchmark:
        parser.error("select exactly one of --mode correctness|benchmark")
    print(json.dumps(run(args.variant, args.size, args.warmup, args.iterations, args.benchmark)))


if __name__ == "__main__":
    main()
