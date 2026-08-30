from __future__ import annotations

import json
import subprocess
import sys

from optimizer.config import WorkloadConfig


def run_correctness(config: WorkloadConfig, variant: str) -> dict:
    command = [sys.executable, str(config.fallback_script), "--variant", variant, "--size", str(config.input_size), "--correctness"]
    result = subprocess.run(command, capture_output=True, text=True)
    payload = _last_json(result.stdout)
    payload.update({"variant": variant, "command": ["python", config.fallback_script.name, *command[2:]], "return_code": result.returncode})
    if result.returncode != 0:
        payload.update({"correctness_pass": False, "stderr": result.stderr.strip()})
    return payload


def _last_json(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"correctness_pass": False, "error": "runner did not emit a JSON result"}
