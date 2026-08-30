from __future__ import annotations

from .config import WorkloadConfig


def render_optimization_prompt(config: WorkloadConfig, baseline_metrics: dict, profiler_summary: dict) -> str:
    return f"""# Optimization hypothesis: {config.name}

## Contract

The candidate is a hypothesis. It must compile, pass the workload correctness gate, and improve the baseline median by at least {config.gate.min_improvement:.1%} with coefficient of variation no larger than {config.gate.max_coefficient_of_variation:.1%}.

## Workload

- Source: `{config.source_file.name}`
- Candidate source: `{config.candidate_source_file.name}`
- Fallback runner: `{config.fallback_script.name}`
- Input size: `{config.input_size}`
- Baseline variant: `{config.baseline_variant}`
- Baseline metrics: `{baseline_metrics}`
- Profiler summary: `{profiler_summary}`

## Candidate request

{config.candidate.rationale}

Return a real unified diff that adds or updates the configured candidate source. The diff is path-validated and applied in an isolated workspace before the candidate build. The candidate must remain independently testable. Do not treat generated code as trusted until the gates pass; CPU fallback timings cannot establish CUDA acceptance.
"""
