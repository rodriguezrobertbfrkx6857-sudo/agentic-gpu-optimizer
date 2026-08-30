# Optimization hypothesis: transpose_case_study

## Contract

The candidate is a hypothesis. It must compile, pass the workload correctness gate, and improve the baseline median by at least 3.0% with coefficient of variation no larger than 25.0%.

## Workload

- Source: `baseline.cu`
- Fallback runner: `reference.py`
- Input size: `192`
- Baseline variant: `baseline`
- Baseline metrics: `{'correctness_pass': True, 'max_abs_error': 0.0, 'variant': 'baseline', 'size': 192, 'status': 'BENCHMARKED_CPU_ONLY', 'median_ms': 4.325, 'mean_ms': 4.5205375000000005, 'min_ms': 4.1801, 'std_ms': 0.45260050524027245, 'command': ['python', 'reference.py', '--variant', 'baseline', '--size', '192', '--warmup', '3', '--iterations', '8', '--benchmark'], 'return_code': 0}`
- Profiler summary: `{'available': False, 'status': 'NOT_AVAILABLE', 'note': 'Nsight profiling was not available in the current environment.', 'workload': 'transpose_case_study'}`

## Candidate request

Replace scalar strided transpose stores with a coalesced shared-memory tile and one-column padding to reduce bank conflicts.

The candidate must remain independently testable. Do not treat generated code as trusted until the gates pass.
