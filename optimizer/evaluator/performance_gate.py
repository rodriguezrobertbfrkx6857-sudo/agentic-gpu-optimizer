from __future__ import annotations


def evaluate_performance(baseline: dict, candidate: dict, min_improvement: float, max_cv: float) -> dict:
    baseline_median = float(baseline.get("median_ms", 0.0))
    candidate_median = float(candidate.get("median_ms", 0.0))
    if baseline_median <= 0.0 or candidate_median <= 0.0:
        return {"status": "FAIL", "reason": "missing or non-positive median timing"}
    improvement = 1.0 - candidate_median / baseline_median
    speedup = baseline_median / candidate_median
    baseline_cv = float(baseline.get("std_ms", 0.0)) / float(baseline.get("mean_ms", 1.0))
    candidate_cv = float(candidate.get("std_ms", 0.0)) / float(candidate.get("mean_ms", 1.0))
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
    }

