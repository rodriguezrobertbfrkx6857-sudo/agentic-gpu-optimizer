from __future__ import annotations

from .correctness_gate import evaluate_correctness
from .performance_gate import evaluate_performance


def decide(config, correctness: dict, baseline_benchmark: dict, candidate_benchmark: dict) -> dict:
    correctness_gate = evaluate_correctness(correctness)
    performance_gate = evaluate_performance(
        baseline_benchmark,
        candidate_benchmark,
        config.gate.min_improvement,
        config.gate.max_coefficient_of_variation,
    )
    if not correctness_gate["passed"]:
        decision = "REJECT"
        reason = "correctness gate failed"
    elif performance_gate["status"] == "PASS" and not performance_gate["cuda_validated"]:
        decision = "INCONCLUSIVE"
        reason = "CPU fallback evidence is insufficient for a CUDA acceptance decision"
    elif performance_gate["status"] == "PASS":
        decision = "ACCEPT"
        reason = "correctness passed and performance threshold/stability gates passed"
    elif performance_gate["status"] == "FAIL":
        decision = "REJECT"
        reason = "candidate regressed or benchmark metrics were invalid"
    else:
        decision = "REJECT" if config.gate.low_improvement_policy == "reject" else "INCONCLUSIVE"
        reason = "candidate was correct but did not clear the configured performance/stability gate"
    return {
        "decision": decision,
        "reason": reason,
        "correctness_gate": correctness_gate,
        "performance_gate": performance_gate,
    }
