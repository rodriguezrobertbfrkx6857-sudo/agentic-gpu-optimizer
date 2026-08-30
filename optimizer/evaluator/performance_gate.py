from __future__ import annotations


def evaluate_performance(baseline: dict, candidate: dict, min_improvement: float, max_cv: float) -> dict:
    baseline_median = float(baseline.get("median_ms") or 0.0)
    candidate_median = float(candidate.get("median_ms") or 0.0)
    cuda_validated = bool(
        baseline.get("cuda_validated", False)
        and candidate.get("cuda_validated", False)
        and baseline.get("hardware_mode") == "cuda"
        and candidate.get("hardware_mode") == "cuda"
    )
    if baseline_median <= 0.0 or candidate_median <= 0.0:
        return {
            "status": "FAIL",
            "reason": "missing or non-positive median timing",
            "cuda_validated": cuda_validated,
        }
    improvement = 1.0 - candidate_median / baseline_median
    speedup = baseline_median / candidate_median
    baseline_mean = float(baseline.get("mean_ms") or 0.0)
    candidate_mean = float(candidate.get("mean_ms") or 0.0)
    baseline_cv = float(baseline.get("std_ms") or 0.0) / baseline_mean if baseline_mean > 0 else float("inf")
    candidate_cv = float(candidate.get("std_ms") or 0.0) / candidate_mean if candidate_mean > 0 else float("inf")
    stable = baseline_cv <= max_cv and candidate_cv <= max_cv
    status = "PASS" if improvement >= min_improvement and stable else "INCONCLUSIVE"
    if improvement < 0.0:
        status = "FAIL"
    return {
        "status": status,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "improvement": improvement,
        "speedup": speedup,
        "baseline_cv": baseline_cv,
        "candidate_cv": candidate_cv,
        "stable": stable,
        "threshold": min_improvement,
        "max_cv": max_cv,
        "cuda_validated": cuda_validated,
        "baseline_status": baseline.get("status"),
        "candidate_status": candidate.get("status"),
        "baseline_backend": baseline.get("backend"),
        "candidate_backend": candidate.get("backend"),
    }
