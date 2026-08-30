from __future__ import annotations

import argparse
import json
import time

import numpy as np


def baseline(values: np.ndarray) -> np.float32:
    return np.add.reduce(values, dtype=np.float32)


def candidate(values: np.ndarray) -> np.float64:
    # A precision-oriented candidate is intentionally measured by the gate; it is not assumed to be faster.
    return np.add.reduce(values.astype(np.float64), dtype=np.float64)


def run(variant: str, size: int, warmup: int, iterations: int, benchmark: bool) -> dict:
    values = np.sin(np.arange(size, dtype=np.float32) * np.float32(0.001))
    expected = float(np.sum(values, dtype=np.float64))
    function = baseline if variant == "baseline" else candidate if variant == "reduction_precision_candidate" else None
    if function is None:
        raise ValueError(f"unknown reduction variant: {variant}")
    actual = float(function(values))
    error = abs(actual - expected)
    result = {"correctness_pass": bool(np.isclose(actual, expected, rtol=2.0e-5, atol=2.0e-3)), "max_abs_error": error, "variant": variant, "size": size}
    if benchmark:
        for _ in range(warmup):
            function(values)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            function(values)
            samples.append((time.perf_counter_ns() - start) / 1.0e6)
        numbers = np.asarray(samples, dtype=np.float64)
        result.update({"status": "BENCHMARKED_CPU_ONLY", "median_ms": float(np.median(numbers)), "mean_ms": float(np.mean(numbers)), "min_ms": float(np.min(numbers)), "std_ms": float(np.std(numbers))})
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
