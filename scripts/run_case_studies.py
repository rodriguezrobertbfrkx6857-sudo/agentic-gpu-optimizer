#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.orchestrator import run


ARTIFACTS = [
    "baseline_metrics.json",
    "profiler_summary.json",
    "optimization_prompt.md",
    "candidate.patch",
    "correctness.json",
    "benchmark.json",
    "decision.json",
]


def _summary(experiment: Path, run_dir: Path) -> None:
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    benchmark = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    correctness = json.loads((run_dir / "correctness.json").read_text(encoding="utf-8"))
    base = benchmark["baseline"]
    candidate = benchmark["candidate"]
    lines = [
        f"# {decision['workload']}",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Reason: {decision['reason']}",
        f"- Baseline median: `{base.get('median_ms')} ms`",
        f"- Candidate median: `{candidate.get('median_ms')} ms`",
        f"- Speedup: `{decision['performance_gate'].get('speedup')}x`",
        f"- Correctness: `{decision['correctness_gate']['status']}`",
        f"- Hardware mode: `{base.get('hardware_mode', 'cpu_only')}`",
        "",
        "This case study is generated from the run artifacts in this directory. Timing values are not edited by hand.",
    ]
    (experiment / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (experiment / "run_metadata.json").write_text(
        json.dumps({"run_id": run_dir.name, "decision": decision["decision"], "correctness": correctness}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = [
        root / "examples" / "transpose_case_study.yaml",
        root / "examples" / "reduction_case_study.yaml",
    ]
    for config in configs:
        run_dir = run(config, root / args.output_root)
        experiment = root / "experiments" / f"{config.stem}"
        experiment.mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACTS:
            shutil.copy2(run_dir / artifact, experiment / artifact)
        _summary(experiment, run_dir)
        print(json.dumps({"experiment": str(experiment), "run_id": run_dir.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
