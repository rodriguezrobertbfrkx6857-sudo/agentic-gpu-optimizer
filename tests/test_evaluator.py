from optimizer.evaluator.decision import decide
from optimizer.config import GateConfig


class Config:
    gate = GateConfig()


def test_accepts_correct_and_stable_improvement():
    correctness = {"correctness_pass": True, "return_code": 0, "cuda_validated": True}
    baseline = {
        "median_ms": 10.0,
        "mean_ms": 10.0,
        "std_ms": 0.1,
        "status": "BENCHMARKED_CUDA",
        "backend": "cuda",
        "hardware_mode": "cuda",
        "cuda_validated": True,
    }
    candidate = {
        "median_ms": 9.0,
        "mean_ms": 9.0,
        "std_ms": 0.1,
        "status": "BENCHMARKED_CUDA",
        "backend": "cuda",
        "hardware_mode": "cuda",
        "cuda_validated": True,
    }
    result = decide(Config(), correctness, baseline, candidate)
    assert result["decision"] == "ACCEPT"


def test_cpu_improvement_is_inconclusive_for_cuda_acceptance():
    correctness = {"correctness_pass": True, "return_code": 0}
    baseline = {
        "median_ms": 10.0,
        "mean_ms": 10.0,
        "std_ms": 0.1,
        "status": "BENCHMARKED_CPU_ONLY",
        "backend": "numpy_cpu_reference",
        "hardware_mode": "cpu_only",
    }
    candidate = {
        "median_ms": 9.0,
        "mean_ms": 9.0,
        "std_ms": 0.1,
        "status": "BENCHMARKED_CPU_ONLY",
        "backend": "numpy_cpu_reference",
        "hardware_mode": "cpu_only",
    }
    result = decide(Config(), correctness, baseline, candidate)
    assert result["decision"] == "INCONCLUSIVE"


def test_rejects_regression_after_correctness_pass():
    correctness = {"correctness_pass": True, "return_code": 0}
    baseline = {"median_ms": 10.0, "mean_ms": 10.0, "std_ms": 0.1}
    candidate = {"median_ms": 11.0, "mean_ms": 11.0, "std_ms": 0.1}
    result = decide(Config(), correctness, baseline, candidate)
    assert result["decision"] == "REJECT"


def test_rejects_correctness_failure_before_performance():
    correctness = {"correctness_pass": False, "return_code": 1}
    baseline = {"median_ms": 10.0, "mean_ms": 10.0, "std_ms": 0.1}
    candidate = {"median_ms": 1.0, "mean_ms": 1.0, "std_ms": 0.1}
    result = decide(Config(), correctness, baseline, candidate)
    assert result["decision"] == "REJECT"
