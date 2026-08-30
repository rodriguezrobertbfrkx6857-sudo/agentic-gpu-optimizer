from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from optimizer.config import WorkloadConfig
from optimizer.runner.build import build_workload


_COMMAND_TIMEOUT_SECONDS = 600
_OUTPUT_LIMIT = 4000


def _last_json(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"status": "FAIL", "error": "runner did not emit a JSON result"}


def _safe_output(value: str | bytes | None, workspace: Path, source_root: Path) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value[-_OUTPUT_LIMIT:]
    for path in (workspace, source_root):
        value = value.replace(str(path), "<workspace>")
        value = value.replace(str(path).replace("\\", "/"), "<workspace>")
    return value


def _fallback_path(config: WorkloadConfig, workspace: Path) -> Path:
    return workspace / config.fallback_script.relative_to(config.workload_dir)


def _not_run(status: str, variant: str, build_result: dict) -> dict:
    return {
        "status": status,
        "variant": variant,
        "return_code": 1,
        "execution_mode": build_result.get("mode"),
        "cuda_validated": False,
        "build": build_result,
    }


def run_benchmark(
    config: WorkloadConfig,
    variant: str,
    build_result: dict | None = None,
    workspace: str | Path | None = None,
) -> dict:
    workspace_path = Path(workspace).resolve() if workspace is not None else config.workload_dir
    build_result = build_result or build_workload(config, variant, workspace_path)
    if build_result.get("status") != "PASS":
        return _not_run("NOT_RUN_BUILD_FAILED", variant, build_result)

    if build_result.get("mode") == "cuda_nvcc":
        if not build_result.get("executable"):
            return _not_run("NOT_RUN_MISSING_CUDA_EXECUTABLE", variant, build_result)
        executable = workspace_path / Path(str(build_result["executable"]))
        if not executable.is_file():
            return _not_run("NOT_RUN_MISSING_CUDA_EXECUTABLE", variant, build_result)
        command = [
            str(executable),
            "--variant",
            variant,
            "--size",
            str(config.input_size),
            "--warmup",
            str(config.warmup),
            "--iterations",
            str(config.iterations),
            "--mode",
            "benchmark",
        ]
        cwd = executable.parent
        display_command = [executable.name, *command[1:]]
    else:
        fallback = _fallback_path(config, workspace_path)
        command = [
            sys.executable,
            str(fallback),
            "--variant",
            variant,
            "--size",
            str(config.input_size),
            "--warmup",
            str(config.warmup),
            "--iterations",
            str(config.iterations),
            "--mode",
            "benchmark",
        ]
        cwd = fallback.parent
        display_command = ["python", fallback.name, *command[2:]]

    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
        return_code = result.returncode
        stdout = _safe_output(result.stdout, workspace_path, config.workload_dir)
        stderr = _safe_output(result.stderr, workspace_path, config.workload_dir)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return_code = None
        stdout = _safe_output(exc.stdout, workspace_path, config.workload_dir)
        stderr = _safe_output(exc.stderr, workspace_path, config.workload_dir)
        timed_out = True
    duration_ms = (time.perf_counter() - started) * 1000.0
    payload = _last_json(stdout)
    payload.update(
        {
            "variant": variant,
            "command": display_command,
            "return_code": return_code,
            "duration_ms": duration_ms,
            "execution_mode": build_result.get("mode"),
            "cuda_validated": bool(build_result.get("cuda_validated", False) and return_code == 0),
            "stdout": stdout,
            "stderr": stderr,
            "timeout_seconds": _COMMAND_TIMEOUT_SECONDS,
        }
    )
    if timed_out:
        payload.update({"status": "BENCHMARK_TIMEOUT"})
    elif return_code != 0:
        payload.update({"status": "FAIL"})
    return payload
