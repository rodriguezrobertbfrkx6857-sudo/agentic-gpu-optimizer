# Optimization hypothesis: reduction_case_study

## Contract

The candidate is a hypothesis. It must compile, pass the workload correctness gate, and improve the baseline median by at least 3.0% with coefficient of variation no larger than 25.0%.

## Workload

- Source: `baseline.cu`
- Fallback runner: `reference.py`
- Input size: `1048576`
- Baseline variant: `baseline`
- Baseline metrics: `{'correctness_pass': True, 'max_abs_error': 2.7328048417984974e-05, 'variant': 'baseline', 'size': 1048576, 'status': 'BENCHMARKED_CPU_ONLY', 'median_ms': 0.19345, 'mean_ms': 0.193075, 'min_ms': 0.1878, 'std_ms': 0.0027738736452837935, 'command': ['python', 'reference.py', '--variant', 'baseline', '--size', '1048576', '--warmup', '3', '--iterations', '8', '--benchmark'], 'return_code': 0}`
- Profiler summary: `{'available': False, 'status': 'NOT_AVAILABLE', 'note': 'Nsight profiling was not available in the current environment.', 'workload': 'reduction_case_study'}`

## Candidate request

Try a double-precision accumulation candidate to reduce summation error; the gate must prove that the added conversion and precision cost are justified.

The candidate must remain independently testable. Do not treat generated code as trusted until the gates pass.
