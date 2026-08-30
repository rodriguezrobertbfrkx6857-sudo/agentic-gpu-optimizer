from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from optimizer.config import WorkloadConfig
from optimizer.providers.base import Candidate


_PATCH_PATH = re.compile(r"^(?:---|\+\+\+)\s+([^\t\s]+)")
_DIFF_PATHS = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_OUTPUT_LIMIT = 4000


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-_OUTPUT_LIMIT:]


def prepare_workspace(config: WorkloadConfig, target: str | Path) -> Path:
    """Copy one workload into an isolated run directory without copying build products."""
    destination = Path(target).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        config.workload_dir,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "build", "*.exe", "*.o"),
    )
    return destination


def _relative_patch_path(raw: str) -> str | None:
    if raw == "/dev/null":
        return None
    normalized = raw.replace("\\", "/")
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    path = Path(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe patch path: {raw}")
    return path.as_posix()


def _validate_patch(candidate: Candidate, config: WorkloadConfig) -> list[str]:
    if not candidate.patch.strip():
        raise ValueError("candidate patch is empty")
    allowed = {
        config.source_file.relative_to(config.workload_dir).as_posix(),
        config.candidate_source_file.relative_to(config.workload_dir).as_posix(),
    }
    paths: list[str] = []
    for line in candidate.patch.splitlines():
        diff_match = _DIFF_PATHS.match(line)
        patch_match = _PATCH_PATH.match(line)
        raw_paths = (
            diff_match.groups()
            if diff_match
            else (patch_match.group(1),)
            if patch_match
            else ()
        )
        for raw_path in raw_paths:
            path = _relative_patch_path(raw_path)
            if path is not None:
                paths.append(path)
                if path not in allowed:
                    raise ValueError(f"patch touches a path outside the workload contract: {path}")
    if not paths:
        raise ValueError("candidate patch contains no file paths")
    if not any(path == config.candidate_source_file.relative_to(config.workload_dir).as_posix() for path in paths):
        raise ValueError("candidate patch does not add or modify the configured candidate source")
    return sorted(set(paths))


def apply_candidate(candidate: Candidate, run_dir: str | Path, config: WorkloadConfig) -> dict:
    """Apply a provider's real unified diff inside a fresh, run-local workspace."""
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = run_root / "candidate_workspace"
    patch_path = run_root / "candidate.patch"
    try:
        changed_paths = _validate_patch(candidate, config)
        prepare_workspace(config, workspace)
        patch_path.write_text(candidate.patch, encoding="utf-8")
        initialized = subprocess.run(
            ["git", "init", "--quiet"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if initialized.returncode != 0:
            return {
                "status": "REJECTED",
                "mode": "isolated_unified_diff",
                "candidate": candidate.name,
                "variant": candidate.variant,
                "patch_artifact": "candidate.patch",
                "workspace": "candidate_workspace",
                "changed_paths": changed_paths,
                "init_return_code": initialized.returncode,
                "init_stdout": _tail(initialized.stdout),
                "init_stderr": _tail(initialized.stderr),
                "detail": (initialized.stderr or initialized.stdout).strip(),
            }
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check.returncode != 0:
            return {
                "status": "REJECTED",
                "mode": "isolated_unified_diff",
                "candidate": candidate.name,
                "variant": candidate.variant,
                "patch_artifact": "candidate.patch",
                "workspace": "candidate_workspace",
                "changed_paths": changed_paths,
                "repository_initialized": True,
                "check_command": ["git", "apply", "--check", "candidate.patch"],
                "check_return_code": check.returncode,
                "check_stdout": _tail(check.stdout),
                "check_stderr": _tail(check.stderr),
                "detail": (check.stderr or check.stdout).strip(),
            }
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if applied.returncode != 0:
            return {
                "status": "REJECTED",
                "mode": "isolated_unified_diff",
                "candidate": candidate.name,
                "variant": candidate.variant,
                "patch_artifact": "candidate.patch",
                "workspace": "candidate_workspace",
                "changed_paths": changed_paths,
                "repository_initialized": True,
                "apply_command": ["git", "apply", "candidate.patch"],
                "apply_return_code": applied.returncode,
                "apply_stdout": _tail(applied.stdout),
                "apply_stderr": _tail(applied.stderr),
                "detail": (applied.stderr or applied.stdout).strip(),
            }
        candidate_source = workspace / config.candidate_source_file.relative_to(config.workload_dir)
        if not candidate_source.is_file():
            return {
                "status": "REJECTED",
                "mode": "isolated_unified_diff",
                "candidate": candidate.name,
                "variant": candidate.variant,
                "patch_artifact": "candidate.patch",
                "workspace": "candidate_workspace",
                "changed_paths": changed_paths,
                "repository_initialized": True,
                "apply_return_code": applied.returncode,
                "apply_stdout": _tail(applied.stdout),
                "apply_stderr": _tail(applied.stderr),
                "detail": "candidate source is missing after patch application",
            }
        return {
            "status": "APPLIED",
            "mode": "isolated_unified_diff",
            "candidate": candidate.name,
            "variant": candidate.variant,
            "patch_artifact": "candidate.patch",
            "workspace": "candidate_workspace",
            "changed_paths": changed_paths,
            "repository_initialized": True,
            "check_command": ["git", "apply", "--check", "candidate.patch"],
            "check_return_code": check.returncode,
            "check_stdout": _tail(check.stdout),
            "check_stderr": _tail(check.stderr),
            "apply_command": ["git", "apply", "candidate.patch"],
            "apply_return_code": applied.returncode,
            "apply_stdout": _tail(applied.stdout),
            "apply_stderr": _tail(applied.stderr),
            "note": "The provider patch was applied and verified inside an isolated workspace.",
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "status": "REJECTED",
            "mode": "isolated_unified_diff",
            "candidate": candidate.name,
            "variant": candidate.variant,
            "patch_artifact": "candidate.patch",
            "workspace": "candidate_workspace",
            "detail": str(exc),
        }
