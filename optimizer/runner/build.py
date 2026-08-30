from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from optimizer.config import WorkloadConfig


@dataclass(frozen=True)
class BuildResult:
    status: str
    mode: str
    command: list[str]
    source: str
    detail: str


def build_workload(config: WorkloadConfig, variant: str) -> dict:
    if not config.fallback_script.exists():
        return asdict(BuildResult("FAIL", "none", [], config.fallback_script.name, "fallback script is missing"))
    py_compile = subprocess.run(
        [sys.executable, "-m", "py_compile", str(config.fallback_script)],
        capture_output=True,
        text=True,
    )
    if py_compile.returncode != 0:
        return asdict(BuildResult("FAIL", "cpu_fallback", ["python", "-m", "py_compile", config.fallback_script.name], config.fallback_script.name, py_compile.stderr.strip()))
    nvcc = shutil.which("nvcc")
    if nvcc and config.source_file.exists():
        return asdict(BuildResult(
            "PASS",
            "cpu_fallback_with_cuda_source_present",
            [sys.executable, "-m", "py_compile", str(config.fallback_script)],
            config.source_file.name,
            f"CUDA compiler detected; workload source is retained for target-specific build. Variant={variant}",
        ))
    return asdict(BuildResult(
        "PASS",
        "cpu_fallback",
        ["python", "-m", "py_compile", config.fallback_script.name],
        config.fallback_script.name,
        f"CUDA compiler unavailable; executed reference runner build check. Variant={variant}",
    ))
