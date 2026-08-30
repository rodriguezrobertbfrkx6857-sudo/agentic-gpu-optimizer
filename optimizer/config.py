from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GateConfig:
    min_improvement: float = 0.03
    max_coefficient_of_variation: float = 0.25
    low_improvement_policy: str = "inconclusive"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    variant: str
    rationale: str
    patch: str


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    workload_dir: Path
    fallback_script: Path
    source_file: Path
    input_size: int
    baseline_variant: str
    candidate: CandidateConfig
    warmup: int = 3
    iterations: int = 8
    gate: GateConfig = field(default_factory=GateConfig)


def load_config(path: str | Path) -> WorkloadConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    workload_dir = (config_path.parent / raw["workload_dir"]).resolve()
    candidate_raw = raw["candidate"]
    gate_raw = raw.get("gate", {})
    return WorkloadConfig(
        name=str(raw["name"]),
        workload_dir=workload_dir,
        fallback_script=(workload_dir / raw.get("fallback_script", "reference.py")).resolve(),
        source_file=(workload_dir / raw.get("source_file", "baseline.cu")).resolve(),
        input_size=int(raw.get("input_size", 256)),
        baseline_variant=str(raw.get("baseline_variant", "baseline")),
        candidate=CandidateConfig(
            name=str(candidate_raw["name"]),
            variant=str(candidate_raw["variant"]),
            rationale=str(candidate_raw["rationale"]),
            patch=str(candidate_raw["patch"]),
        ),
        warmup=int(raw.get("benchmark", {}).get("warmup", 3)),
        iterations=int(raw.get("benchmark", {}).get("iterations", 8)),
        gate=GateConfig(
            min_improvement=float(gate_raw.get("min_improvement", 0.03)),
            max_coefficient_of_variation=float(gate_raw.get("max_coefficient_of_variation", 0.25)),
            low_improvement_policy=str(gate_raw.get("low_improvement_policy", "inconclusive")),
        ),
    )

