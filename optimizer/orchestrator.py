from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from optimizer.config import load_config
from optimizer.environment import collect_environment
from optimizer.evaluator.decision import decide
from optimizer.prompts import render_optimization_prompt
from optimizer.providers.manual import ManualProvider
from optimizer.runner.apply_candidate import apply_candidate, prepare_workspace
from optimizer.runner.benchmark import run_benchmark
from optimizer.runner.build import build_workload
from optimizer.runner.correctness import run_correctness
from optimizer.runner.profiler import profile_workload


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(config, baseline_workspace: Path, candidate_workspace: Path | None) -> dict:
    baseline_source = baseline_workspace / config.source_file.relative_to(config.workload_dir)
    candidate_source = (
        candidate_workspace / config.candidate_source_file.relative_to(config.workload_dir)
        if candidate_workspace is not None
        else None
    )
    return {
        "baseline": {
            "path": config.source_file.relative_to(config.workload_dir).as_posix(),
            "sha256": _sha256(baseline_source),
        },
        "candidate": {
            "path": config.candidate_source_file.relative_to(config.workload_dir).as_posix(),
            "sha256": _sha256(candidate_source) if candidate_source is not None else None,
        },
    }


def _skip_benchmark(reason: str, variant: str) -> dict:
    return {
        "status": reason,
        "variant": variant,
        "return_code": 1,
        "duration_ms": 0.0,
        "execution_mode": "not_run",
        "cuda_validated": False,
    }


def run(config_path: str | Path, output_root: str | Path = "runs", provider=None) -> Path:
    config = load_config(config_path)
    provider = provider or ManualProvider()
    output_base = Path(output_root).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / run_id
    if run_dir.exists():
        suffix = 1
        while (output_base / f"{run_id}_{suffix}").exists():
            suffix += 1
        run_dir = output_base / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = collect_environment()
    _write_json(run_dir / "environment.json", environment)
    baseline_workspace = prepare_workspace(config, run_dir / "baseline_workspace")
    baseline_build = build_workload(config, config.baseline_variant, baseline_workspace)
    baseline_correctness = run_correctness(
        config, config.baseline_variant, baseline_build, baseline_workspace
    )
    baseline_benchmark = (
        run_benchmark(config, config.baseline_variant, baseline_build, baseline_workspace)
        if baseline_correctness.get("correctness_pass") and baseline_correctness.get("return_code") == 0
        else _skip_benchmark("NOT_RUN_BASELINE_CORRECTNESS_FAILED", config.baseline_variant)
    )
    baseline_metrics = {
        "workload": config.name,
        "environment": environment,
        "workspace": "baseline_workspace",
        "build": baseline_build,
        "correctness": baseline_correctness,
        "benchmark": baseline_benchmark,
    }
    _write_json(run_dir / "baseline_metrics.json", baseline_metrics)
    _write_json(run_dir / "baseline_build.json", baseline_build)

    profiler_summary = profile_workload(
        config,
        baseline_build,
        baseline_workspace,
        run_dir / "profile",
        config.baseline_variant,
    )
    _write_json(run_dir / "profiler_summary.json", profiler_summary)
    prompt = render_optimization_prompt(config, baseline_benchmark, profiler_summary)
    (run_dir / "optimization_prompt.md").write_text(prompt, encoding="utf-8")

    candidate = provider.suggest(config)
    (run_dir / "candidate.patch").write_text(candidate.patch, encoding="utf-8")
    application = apply_candidate(candidate, run_dir, config)
    _write_json(run_dir / "candidate_application.json", application)

    if application.get("status") == "APPLIED":
        candidate_workspace = run_dir / "candidate_workspace"
        candidate_build = build_workload(config, candidate.variant, candidate_workspace)
        candidate_correctness = run_correctness(
            config, candidate.variant, candidate_build, candidate_workspace
        )
        candidate_benchmark = (
            run_benchmark(config, candidate.variant, candidate_build, candidate_workspace)
            if candidate_correctness.get("correctness_pass")
            and candidate_correctness.get("return_code") == 0
            else _skip_benchmark("NOT_RUN_CANDIDATE_CORRECTNESS_FAILED", candidate.variant)
        )
    else:
        candidate_build = {
            "status": "NOT_RUN_PATCH_REJECTED",
            "mode": "none",
            "command": [],
            "source": config.candidate_source_file.name,
            "executable": None,
            "cuda_validated": False,
            "hardware_mode": environment["hardware_mode"],
            "detail": application.get("detail", "candidate patch was rejected"),
            "return_code": 1,
            "duration_ms": 0.0,
            "compiler_version": environment["toolchain"].get("nvcc"),
            "source_sha256": None,
            "output_sha256": None,
            "stdout": "",
            "stderr": "",
            "timeout_seconds": 600,
        }
        candidate_correctness = {
            "correctness_pass": False,
            "status": "NOT_RUN_PATCH_REJECTED",
            "variant": candidate.variant,
            "return_code": 1,
        }
        candidate_benchmark = _skip_benchmark("NOT_RUN_PATCH_REJECTED", candidate.variant)

    _write_json(run_dir / "candidate_build.json", candidate_build)
    _write_json(
        run_dir / "source_hashes.json",
        _source_hashes(
            config,
            baseline_workspace,
            candidate_workspace if application.get("status") == "APPLIED" else None,
        ),
    )

    _write_json(
        run_dir / "correctness.json",
        {
            "baseline_build": baseline_build,
            "baseline": baseline_correctness,
            "candidate": candidate_correctness,
            "candidate_build": candidate_build,
            "candidate_application": application,
        },
    )
    _write_json(
        run_dir / "benchmark.json",
        {
            "baseline": baseline_benchmark,
            "candidate": candidate_benchmark,
            "hardware_mode": environment["hardware_mode"],
            "cuda_validated": bool(
                baseline_benchmark.get("cuda_validated", False)
                and candidate_benchmark.get("cuda_validated", False)
            ),
        },
    )
    decision = decide(config, candidate_correctness, baseline_benchmark, candidate_benchmark)
    decision.update(
        {
            "workload": config.name,
            "candidate": candidate.name,
            "candidate_variant": candidate.variant,
            "candidate_application": application["status"],
        }
    )
    _write_json(run_dir / "decision.json", decision)
    print(json.dumps({"run_dir": str(run_dir), "decision": decision["decision"]}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
