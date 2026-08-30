from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return result.returncode, (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, repr(exc)


def _version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    code, output = _run([executable, "--version"])
    if code != 0 and not output:
        return None
    return output.splitlines()[0] if output else "available"


def _ram_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("available_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _display_adapters() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    code, output = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
        ]
    )
    if code != 0 or not output:
        return []
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return []
    items = value if isinstance(value, list) else [value]
    return [
        {
            "name": item.get("Name"),
            "adapter_ram_bytes": item.get("AdapterRAM"),
            "driver_version": item.get("DriverVersion"),
        }
        for item in items
        if isinstance(item, dict)
    ]


def _torch_info() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch

        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "compiled_cuda": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "capabilities": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # pragma: no cover - external runtime dependent
        return {"installed": True, "import_error": repr(exc)}


def _nvidia() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {"available": False, "executable": bool(executable), "devices": []}
    if executable is None:
        result["reason"] = "nvidia-smi was not found"
        return result
    code, output = _run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        result["reason"] = output or "nvidia-smi returned a non-zero status"
        return result
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            result["devices"].append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_total_mib": fields[2],
                    "compute_capability": fields[3],
                }
            )
    result["available"] = bool(result["devices"])
    if not result["devices"]:
        result["reason"] = "nvidia-smi returned no CUDA-capable devices"
    return result


def collect_environment() -> dict[str, Any]:
    nvidia = _nvidia()
    torch_info = _torch_info()
    cuda_workflow_available = bool(nvidia["available"] and shutil.which("nvcc"))
    return {
        "schema_version": 2,
        "captured_by": "optimizer.environment.collect_environment",
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "name": platform.processor() or platform.machine(),
            "logical_processors": os.cpu_count(),
        },
        "ram_bytes": _ram_bytes(),
        "display_adapters": _display_adapters(),
        "python": {"version": platform.python_version()},
        "toolchain": {
            "nvcc": _version("nvcc"),
            "cmake": _version("cmake"),
            "cxx": _version("cl") or _version("g++") or _version("gcc"),
            "git": _version("git"),
            "gh": _version("gh"),
            "ncu": _version("ncu"),
            "nsys": _version("nsys"),
        },
        "torch": torch_info,
        "nvidia": nvidia,
        "cuda_workflow_available": cuda_workflow_available,
        "github_cli_authenticated": bool(
            shutil.which("gh") and _run([shutil.which("gh") or "gh", "auth", "status"])[0] == 0
        ),
        "hardware_mode": "cuda" if cuda_workflow_available else "cpu_only",
    }


def write_environment(path: str | Path) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = collect_environment()
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
