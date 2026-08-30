from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else None


def collect_environment() -> dict:
    torch_info: dict = {"installed": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "compiled_cuda": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        pass
    nvidia = shutil.which("nvidia-smi")
    nvidia_status = {"available": False, "query": None}
    if nvidia:
        try:
            result = subprocess.run([nvidia, "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
            nvidia_status = {"available": result.returncode == 0 and bool(result.stdout.strip()), "query": result.stdout.strip() or None}
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "schema_version": 1,
        "operating_system": {"system": platform.system(), "release": platform.release(), "version": platform.version(), "machine": platform.machine()},
        "python": {"version": platform.python_version(), "executable": "redacted"},
        "toolchain": {name: _version(name) for name in ("nvcc", "cmake", "git", "gh", "ncu", "nsys")},
        "torch": torch_info,
        "nvidia": nvidia_status,
        "hardware_mode": "cuda" if nvidia_status["available"] and shutil.which("nvcc") else "cpu_only",
    }


def write_environment(path: str | Path) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = collect_environment()
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data

