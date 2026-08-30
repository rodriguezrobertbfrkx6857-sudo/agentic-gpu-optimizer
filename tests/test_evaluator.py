from optimizer.evaluator.decision import decide
from optimizer.config import GateConfig


class Config:
    gate = GateConfig()


def test_accepts_correct_and_stable_improvement():
    correctness = {"correctness_pass": True, "return_code": 0}
    baseline = {"median_ms": 10.0, "mean_ms": 10.0, "std_ms": 0.1}
    candidate = {"median_ms": 9.0, "mean_ms": 9.0, "std_ms": 0.1}
    result = decide(Config(), correctness, baseline, candidate)
    assert result["decision"] == "ACCEPT"


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

