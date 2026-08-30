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
provider hypothesis → validate/apply unified diff → candidate build → correctness → benchmark
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

Each run writes the requested artifacts under `runs/YYYYMMDD_HHMMSS/`. The two public-facing summaries and artifacts are copied to [experiments/transpose_case_study](experiments/transpose_case_study) and [experiments/reduction_case_study](experiments/reduction_case_study). Candidate code is applied only inside a fresh run-local workspace after path validation and `git apply --check`.

## Case studies

- Transpose: scalar CPU reference versus a tiled NumPy analogue of the coalesced 32×33 CUDA transpose. The correctness gate runs before timing; the measured decision is generated from the current host, and CPU-only speedups are not CUDA claims.
- Reduction: a fast float32 baseline versus a precision-oriented float64 conversion candidate. It is intentionally useful as a rejection example when added precision does not pay for its conversion cost.

Inspect `environment.json`, `source_hashes.json`, `baseline_build.json`, `candidate_build.json`, `decision.json`, `correctness.json`, `benchmark.json`, `optimization_prompt.md`, `candidate.patch`, and `candidate_application.json` in each experiment directory. The runner records whether the real unified diff was applied, whether `nvcc` built the candidate, and which backend produced the timing. No decision is based on the candidate text alone.

## Providers

`ManualProvider` is the default and does not contact a model. `CommandProvider` is an adapter for an external command and never reads or writes credentials. Set `GPU_OPTIMIZER_AGENT_COMMAND` only when the command is installed and the user explicitly wants to use it. The candidate still must pass the same gates.

## CUDA path and limitations

The `.cu` files are standalone workload sources with a common CLI. On a CUDA host, the runner builds baseline and candidate executables with `nvcc`, runs correctness before benchmarking, times kernels with CUDA Events, and can invoke `ncu` or `nsys`. On the current CPU-only host it executes the Python reference path, records `NOT BENCHMARKED ON CURRENT HARDWARE`, and marks the decision `INCONCLUSIVE` when a CPU-only improvement cannot establish CUDA acceptance. No profiler counters are inferred from wall-clock timing.

This repository is a workflow demonstration, not an autonomous code-trust mechanism. AI-generated code is never accepted without validation, and the decision gate is the source of truth.
