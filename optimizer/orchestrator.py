from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from optimizer.config import load_config
from optimizer.environment import collect_environment
from optimizer.evaluator.decision import decide
from optimizer.prompts import render_optimization_prompt
from optimizer.providers.manual import ManualProvider
from optimizer.runner.benchmark import run_benchmark
from optimizer.runner.build import build_workload
from optimizer.runner.correctness import run_correctness
from optimizer.runner.profiler import profile_workload


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(config_path: str | Path, output_root: str | Path = "runs", provider=None) -> Path:
    config = load_config(config_path)
    provider = provider or ManualProvider()
    run_id = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).resolve() / run_id
    if run_dir.exists():
        suffix = 1
        while (Path(output_root).resolve() / f"{run_id}_{suffix}").exists():
            suffix += 1
        run_dir = Path(output_root).resolve() / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    baseline_build = build_workload(config, config.baseline_variant)
    baseline_correctness = run_correctness(config, config.baseline_variant)
    baseline_benchmark = run_benchmark(config, config.baseline_variant)
    baseline_metrics = {
        "workload": config.name,
        "environment": collect_environment(),
        "build": baseline_build,
        "correctness": baseline_correctness,
        "benchmark": baseline_benchmark,
    }
    _write_json(run_dir / "baseline_metrics.json", baseline_metrics)

    profiler_summary = profile_workload(config)
    _write_json(run_dir / "profiler_summary.json", profiler_summary)
    prompt = render_optimization_prompt(config, baseline_benchmark, profiler_summary)
    (run_dir / "optimization_prompt.md").write_text(prompt, encoding="utf-8")

    candidate = provider.suggest(config)
    (run_dir / "candidate.patch").write_text(candidate.patch.rstrip() + "\n", encoding="utf-8")
    candidate_build = build_workload(config, candidate.variant)
    candidate_correctness = run_correctness(config, candidate.variant)
    candidate_benchmark = run_benchmark(config, candidate.variant)
    _write_json(
        run_dir / "correctness.json",
        {"baseline": baseline_correctness, "candidate": candidate_correctness, "candidate_build": candidate_build},
    )
    _write_json(
        run_dir / "benchmark.json",
        {"baseline": baseline_benchmark, "candidate": candidate_benchmark},
    )
    decision = decide(config, candidate_correctness, baseline_benchmark, candidate_benchmark)
    decision.update({"workload": config.name, "candidate": candidate.name, "candidate_variant": candidate.variant})
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

