# Architecture

The orchestrator coordinates five small subsystems:

1. `config.py` loads a workload, test/benchmark parameters, provider candidate, and gate thresholds.
2. `runner/` performs path-validated unified-diff application in an isolated repository, CUDA/CPU build selection, correctness execution, benchmark execution, and profiler discovery.
3. `providers/` supplies a candidate through a stable interface. Manual input is deterministic; command input is optional.
4. `evaluator/` converts raw evidence into correctness, performance, and final decision records.
5. `experiments/` keeps a reviewable copy of the generated artifacts for public inspection.

The runner passes only relative display commands and filenames into committed artifacts. Runtime paths are used internally but are not part of the public evidence files. CUDA acceptance is gated on explicit CUDA backend fields rather than inferred from a CPU fallback.
