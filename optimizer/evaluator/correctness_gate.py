from __future__ import annotations


def evaluate_correctness(payload: dict) -> dict:
    passed = bool(payload.get("correctness_pass", False)) and payload.get("return_code", 1) == 0
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "evidence": payload}

