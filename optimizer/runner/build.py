from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from optimizer.config import WorkloadConfig
from optimizer.environment import collect_environment


_COMMAND_TIMEOUT_SECONDS = 600
_OUTPUT_LIMIT = 4000


@dataclass(frozen=True)
class BuildResult:
    status: str
    mode: str
    command: list[str]
    source: str
    executable: str | None
    cuda_validated: bool
    hardware_mode: str
    detail: str
    return_code: int | None = None
    duration_ms: float = 0.0
    compiler_version: str | None = None
    source_sha256: str | None = None
    output_sha256: str | None = None
    stdout: str = ""
    stderr: str = ""
    timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-_OUTPUT_LIMIT:]


def _safe_output(value: str | bytes | None, workspace: Path, source_root: Path) -> str:
    text = _tail(value)
    for path in (workspace, source_root):
        text = text.replace(str(path), "<workspace>")
        text = text.replace(str(path).replace("\\", "/"), "<workspace>")
    return text


def _relative_source(config: WorkloadConfig, variant: str) -> Path:
    if variant == config.baseline_variant:
        return config.source_file.relative_to(config.workload_dir)
    if variant == config.candidate.variant:
        return config.candidate_source_file.relative_to(config.workload_dir)
    raise ValueError(f"unknown workload variant: {variant}")


def _result(
    *,
    status: str,
    mode: str,
    command: list[str],
    source: str,
    executable: str | None,
    cuda_validated: bool,
    hardware_mode: str,
    detail: str,
    return_code: int | None = None,
    duration_ms: float = 0.0,
    compiler_version: str | None = None,
    source_sha256: str | None = None,
    output_sha256: str | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict:
    return asdict(
        BuildResult(
            status=status,
            mode=mode,
            command=command,
            source=source,
            executable=executable,
            cuda_validated=cuda_validated,
            hardware_mode=hardware_mode,
            detail=detail,
            return_code=return_code,
            duration_ms=duration_ms,
            compiler_version=compiler_version,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            stdout=stdout,
            stderr=stderr,
            timeout_seconds=60 if mode == "cpu_reference" else _COMMAND_TIMEOUT_SECONDS,
        )
    )


def build_workload(
    config: WorkloadConfig,
    variant: str,
    workspace: str | Path | None = None,
) -> dict:
    environment = collect_environment()
    hardware_mode = str(environment["hardware_mode"])
    workspace_path = Path(workspace).resolve() if workspace is not None else config.workload_dir
    try:
        relative_source = _relative_source(config, variant)
    except ValueError as exc:
        return _result(
            status="FAIL",
            mode="none",
            command=[],
            source="unknown",
            executable=None,
            cuda_validated=False,
            hardware_mode=hardware_mode,
            detail=str(exc),
        )
    source = workspace_path / relative_source
    source_hash = _sha256(source)

    if hardware_mode == "cuda":
        nvcc = shutil.which("nvcc")
        compiler_version = environment["toolchain"].get("nvcc")
        if nvcc is None:
            return _result(
                status="FAIL",
                mode="cuda_nvcc",
                command=[],
                source=relative_source.as_posix(),
                executable=None,
                cuda_validated=False,
                hardware_mode=hardware_mode,
                detail="nvcc disappeared after environment audit",
                compiler_version=compiler_version,
                source_sha256=source_hash,
            )
        if not source.is_file():
            return _result(
                status="FAIL",
                mode="cuda_nvcc",
                command=[],
                source=relative_source.as_posix(),
                executable=None,
                cuda_validated=False,
                hardware_mode=hardware_mode,
                detail="CUDA source is missing",
                compiler_version=compiler_version,
                source_sha256=source_hash,
            )
        build_dir = workspace_path / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        executable = build_dir / (f"{variant}.exe" if os.name == "nt" else variant)
        actual_command = [
            nvcc,
            "-O3",
            "-std=c++17",
            "-lineinfo",
            source.name,
            "-o",
            str(executable),
        ]
        display_command = [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-lineinfo",
            source.name,
            "-o",
            executable.name,
        ]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                actual_command,
                cwd=source.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
            return_code = result.returncode
            stdout = _safe_output(result.stdout, workspace_path, config.workload_dir)
            stderr = _safe_output(result.stderr, workspace_path, config.workload_dir)
            detail = (stderr or stdout).strip() or "nvcc completed"
        except subprocess.TimeoutExpired as exc:
            return _result(
                status="FAIL",
                mode="cuda_nvcc",
                command=display_command,
                source=relative_source.as_posix(),
                executable=None,
                cuda_validated=False,
                hardware_mode=hardware_mode,
                detail="nvcc build timed out",
                return_code=None,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                compiler_version=compiler_version,
                source_sha256=source_hash,
                stdout=_safe_output(exc.stdout, workspace_path, config.workload_dir),
                stderr=_safe_output(exc.stderr, workspace_path, config.workload_dir),
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if return_code != 0:
            return _result(
                status="FAIL",
                mode="cuda_nvcc",
                command=display_command,
                source=relative_source.as_posix(),
                executable=None,
                cuda_validated=False,
                hardware_mode=hardware_mode,
                detail=detail[-_OUTPUT_LIMIT:],
                return_code=return_code,
                duration_ms=duration_ms,
                compiler_version=compiler_version,
                source_sha256=source_hash,
                stdout=stdout,
                stderr=stderr,
            )
        if not executable.is_file():
            return _result(
                status="FAIL",
                mode="cuda_nvcc",
                command=display_command,
                source=relative_source.as_posix(),
                executable=None,
                cuda_validated=False,
                hardware_mode=hardware_mode,
                detail="nvcc returned success but the executable is missing",
                return_code=return_code,
                duration_ms=duration_ms,
                compiler_version=compiler_version,
                source_sha256=source_hash,
                stdout=stdout,
                stderr=stderr,
            )
        return _result(
            status="PASS",
            mode="cuda_nvcc",
            command=display_command,
            source=relative_source.as_posix(),
            executable=f"build/{executable.name}",
            cuda_validated=True,
            hardware_mode=hardware_mode,
            detail="nvcc compiled the workload source successfully",
            return_code=return_code,
            duration_ms=duration_ms,
            compiler_version=compiler_version,
            source_sha256=source_hash,
            output_sha256=_sha256(executable),
            stdout=stdout,
            stderr=stderr,
        )

    fallback = workspace_path / config.fallback_script.relative_to(config.workload_dir)
    source_hash = _sha256(fallback)
    display_command = ["python", "-m", "py_compile", fallback.name]
    if not fallback.is_file():
        return _result(
            status="FAIL",
            mode="cpu_reference",
            command=display_command,
            source=config.fallback_script.name,
            executable=None,
            cuda_validated=False,
            hardware_mode=hardware_mode,
            detail="fallback script is missing",
            source_sha256=source_hash,
        )
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(fallback)],
            cwd=fallback.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        return_code = result.returncode
        stdout = _safe_output(result.stdout, workspace_path, config.workload_dir)
        stderr = _safe_output(result.stderr, workspace_path, config.workload_dir)
    except subprocess.TimeoutExpired as exc:
        return _result(
            status="FAIL",
            mode="cpu_reference",
            command=display_command,
            source=config.fallback_script.name,
            executable=None,
            cuda_validated=False,
            hardware_mode=hardware_mode,
            detail="Python fallback compile check timed out",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            source_sha256=source_hash,
            stdout=_safe_output(exc.stdout, workspace_path, config.workload_dir),
            stderr=_safe_output(exc.stderr, workspace_path, config.workload_dir),
        )
    duration_ms = (time.perf_counter() - started) * 1000.0
    if return_code != 0:
        return _result(
            status="FAIL",
            mode="cpu_reference",
            command=display_command,
            source=config.fallback_script.name,
            executable=None,
            cuda_validated=False,
            hardware_mode=hardware_mode,
            detail=(stderr or stdout).strip()[-_OUTPUT_LIMIT:],
            return_code=return_code,
            duration_ms=duration_ms,
            source_sha256=source_hash,
            stdout=stdout,
            stderr=stderr,
        )
    return _result(
        status="PASS",
        mode="cpu_reference",
        command=display_command,
        source=config.fallback_script.name,
        executable=None,
        cuda_validated=False,
        hardware_mode=hardware_mode,
        detail=f"CUDA validation unavailable; compiled Python fallback only for variant={variant}",
        return_code=return_code,
        duration_ms=duration_ms,
        source_sha256=source_hash,
        stdout=stdout,
        stderr=stderr,
    )
