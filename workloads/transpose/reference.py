from __future__ import annotations

import argparse
import json
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
    return matrix.T.copy()


def run(variant: str, size: int, warmup: int, iterations: int, benchmark: bool) -> dict:
    matrix = np.arange(size * size, dtype=np.float32).reshape(size, size)
    expected = matrix.T.copy()
    function = baseline if variant == "baseline" else candidate if variant == "transpose_tiled_padded" else None
    if function is None:
        raise ValueError(f"unknown transpose variant: {variant}")
    output = function(matrix)
    error = float(np.max(np.abs(output - expected)))
    result = {"correctness_pass": bool(np.array_equal(output, expected)), "max_abs_error": error, "variant": variant, "size": size}
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
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.variant, args.size, args.warmup, args.iterations, args.benchmark)))


if __name__ == "__main__":
    main()

