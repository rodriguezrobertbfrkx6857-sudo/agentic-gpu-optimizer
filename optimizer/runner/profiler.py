from __future__ import annotations

import subprocess
import time
from pathlib import Path
import shutil

from optimizer.config import WorkloadConfig
from optimizer.environment import collect_environment


_COMMAND_TIMEOUT_SECONDS = 600
_OUTPUT_LIMIT = 4000


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-_OUTPUT_LIMIT:]


def _tool_version(executable: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def profile_workload(
    config: WorkloadConfig,
    build_result: dict | None = None,
    workspace: str | Path | None = None,
    output_dir: str | Path | None = None,
    variant: str | None = None,
) -> dict:
    environment = collect_environment()
    if (
        environment["hardware_mode"] != "cuda"
        or build_result is None
        or build_result.get("mode") != "cuda_nvcc"
        or workspace is None
        or not build_result.get("executable")
    ):
        return {
            "available": False,
            "status": "NOT_AVAILABLE",
            "hardware_mode": environment["hardware_mode"],
            "note": "Nsight profiling was not run because a validated CUDA executable is unavailable.",
            "workload": config.name,
        }

    executable = Path(workspace).resolve() / str(build_result["executable"])
    destination = Path(output_dir or Path(workspace).resolve() / "profile").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected_variant = variant or config.baseline_variant
    arguments = ["--variant", selected_variant, "--size", str(config.input_size), "--mode", "correctness"]
    ncu = shutil.which("ncu")
    nsys = shutil.which("nsys")
    if not ncu and not nsys:
        return {
            "available": False,
            "status": "NOT_AVAILABLE",
            "hardware_mode": "cuda",
            "note": "Neither ncu nor nsys is installed on the CUDA host.",
            "workload": config.name,
        }
    if ncu:
        report = destination / "ncu_report"
        actual_command = [
            ncu,
            "--set",
            "full",
            "--target-processes",
            "all",
            "--export",
            str(report),
            str(executable),
            *arguments,
        ]
        tool = "ncu"
        report_name = "profile/ncu_report.ncu-rep"
        display_command = ["ncu", "--set", "full", "--target-processes", "all", "--export", report_name, executable.name, *arguments]
    else:
        assert nsys is not None
        report = destination / "nsys_report"
        actual_command = [
            nsys,
            "profile",
            "--force-overwrite=true",
            "--output",
            str(report),
            str(executable),
            *arguments,
        ]
        tool = "nsys"
        report_name = "profile/nsys_report"
        display_command = ["nsys", "profile", "--force-overwrite=true", "--output", report_name, executable.name, *arguments]

    started = time.perf_counter()
    try:
        result = subprocess.run(
            actual_command,
            cwd=executable.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "status": "PROFILE_TIMEOUT",
            "hardware_mode": "cuda",
            "tool": tool,
            "tool_version": _tool_version(ncu or nsys or tool),
            "command": display_command,
            "return_code": None,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "stdout": _tail(exc.stdout),
            "stderr": _tail(exc.stderr),
            "timeout_seconds": _COMMAND_TIMEOUT_SECONDS,
            "workload": config.name,
        }
    duration_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        return {
            "available": True,
            "status": "PROFILE_FAILED",
            "hardware_mode": "cuda",
            "tool": tool,
            "tool_version": _tool_version(ncu or nsys or tool),
            "command": display_command,
            "return_code": result.returncode,
            "duration_ms": duration_ms,
            "stdout": _tail(result.stdout),
            "stderr": _tail(result.stderr),
            "timeout_seconds": _COMMAND_TIMEOUT_SECONDS,
            "detail": (result.stderr or result.stdout).strip()[-4000:],
            "workload": config.name,
        }
    reports = list(destination.glob(report.name + "*"))
    if not reports and report.is_file():
        reports = [report]
    if not reports:
        return {
            "available": True,
            "status": "PROFILE_FAILED",
            "hardware_mode": "cuda",
            "tool": tool,
            "tool_version": _tool_version(ncu or nsys or tool),
            "command": display_command,
            "return_code": result.returncode,
            "duration_ms": duration_ms,
            "stdout": _tail(result.stdout),
            "stderr": _tail(result.stderr),
            "timeout_seconds": _COMMAND_TIMEOUT_SECONDS,
            "detail": "profiler returned success but no report file was found",
            "workload": config.name,
        }
    return {
        "available": True,
        "status": "PROFILE_CAPTURED",
        "hardware_mode": "cuda",
        "tool": tool,
        "tool_version": _tool_version(ncu or nsys or tool),
        "command": display_command,
        "return_code": result.returncode,
        "duration_ms": duration_ms,
        "stdout": _tail(result.stdout),
        "stderr": _tail(result.stderr),
        "timeout_seconds": _COMMAND_TIMEOUT_SECONDS,
        "report": report_name,
        "workload": config.name,
        "note": "Raw profiler output is kept in the isolated run directory; counters must be interpreted from that export.",
    }
