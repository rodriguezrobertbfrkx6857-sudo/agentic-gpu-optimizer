# agentic-gpu-optimizer

A benchmark-driven closed-loop workflow for AI-assisted GPU kernel optimization.

The system treats an optimization proposal as a hypothesis and runs it through build, correctness, benchmark, optional profiler discovery, and explicit decision gates. A Manual Provider makes the loop reproducible without an API key; a Command Provider can invoke an installed coding-agent CLI through `GPU_OPTIMIZER_AGENT_COMMAND` when one is intentionally configured.

The checked-in case studies use CPU fallbacks because the current host has no NVIDIA driver or CUDA Toolkit. The CUDA workload sources remain in each workload directory, and every report distinguishes `cpu_only` from a CUDA run.

## Workflow

```text
workload + test/benchmark spec
          ↓
build → correctness → benchmark → profiler discovery
          ↓
provider hypothesis → candidate build → correctness → benchmark
          ↓
                  ACCEPT / REJECT / INCONCLUSIVE
```

The acceptance default is correctness pass, at least 3% median improvement, and acceptable coefficient of variation. A correct regression is rejected; a correct but unstable or sub-threshold candidate is inconclusive. Thresholds live in the YAML configuration.

## Quick start

```powershell
python -m pytest
python scripts/run_case_studies.py
```

To run one configuration:

```powershell
python -m optimizer.orchestrator examples/transpose_case_study.yaml
```

Each run writes the requested artifacts under `runs/YYYYMMDD_HHMMSS/`. The two public-facing summaries and artifacts are copied to [experiments/transpose_case_study](experiments/transpose_case_study) and [experiments/reduction_case_study](experiments/reduction_case_study).

## Case studies

- Transpose: scalar CPU reference versus a NumPy analogue of the coalesced 32×33 tiled transpose. The correctness gate runs before timing; the measured decision is generated from the current host.
- Reduction: a fast float32 baseline versus a precision-oriented float64 conversion candidate. It is intentionally useful as a rejection example when added precision does not pay for its conversion cost.

Inspect `decision.json`, `correctness.json`, `benchmark.json`, `optimization_prompt.md`, `candidate.patch`, and `candidate_application.json` in each experiment directory. The controlled runner applies a proposal by selecting an explicit registered workload variant; it does not execute arbitrary patch text. No decision is based on the candidate text alone.

## Providers

`ManualProvider` is the default and does not contact a model. `CommandProvider` is an adapter for an external command and never reads or writes credentials. Set `GPU_OPTIMIZER_AGENT_COMMAND` only when the command is installed and the user explicitly wants to use it. The candidate still must pass the same gates.

## CUDA path and limitations

The `.cu` files are small standalone workload sources designed to be integrated with a target-specific build. The current runner executes the Python reference path when CUDA is unavailable and records `NOT BENCHMARKED ON CURRENT HARDWARE` for GPU claims. Nsight data is reported as unavailable when `ncu` and `nsys` are absent; no profiler counters are inferred from wall-clock timing.

This repository is a workflow demonstration, not an autonomous code-trust mechanism. AI-generated code is never accepted without validation, and the decision gate is the source of truth.
