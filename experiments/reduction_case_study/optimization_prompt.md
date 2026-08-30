# Optimization hypothesis: reduction_case_study

## Contract

The candidate is a hypothesis. It must compile, pass the workload correctness gate, and improve the baseline median by at least 3.0% with coefficient of variation no larger than 25.0%.

## Workload

- Source: `baseline.cu`
- Candidate source: `candidate.cu`
- Fallback runner: `reference.py`
- Input size: `1048576`
- Baseline variant: `baseline`
- Baseline metrics: `{'correctness_pass': True, 'max_abs_error': 2.7328048417984974e-05, 'variant': 'baseline', 'size': 1048576, 'status': 'BENCHMARKED_CPU_ONLY', 'backend': 'numpy_cpu_reference', 'hardware_mode': 'cpu_only', 'device': 'Intel64 Family 6 Model 170 Stepping 4, GenuineIntel', 'dtype': 'float32', 'warmup': 3, 'iterations': 8, 'timer': 'perf_counter_ns', 'cuda_validated': False, 'median_ms': 0.33345, 'mean_ms': 0.3459375, 'min_ms': 0.3019, 'std_ms': 0.042905155211815746, 'command': ['python', 'reference.py', '--variant', 'baseline', '--size', '1048576', '--warmup', '3', '--iterations', '8', '--mode', 'benchmark'], 'return_code': 0, 'duration_ms': 360.29659998894203, 'execution_mode': 'cpu_reference', 'stdout': '{"correctness_pass": true, "max_abs_error": 2.7328048417984974e-05, "variant": "baseline", "size": 1048576, "status": "BENCHMARKED_CPU_ONLY", "backend": "numpy_cpu_reference", "hardware_mode": "cpu_only", "device": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel", "dtype": "float32", "warmup": 3, "iterations": 8, "timer": "perf_counter_ns", "cuda_validated": false, "median_ms": 0.33345, "mean_ms": 0.3459375, "min_ms": 0.3019, "std_ms": 0.042905155211815746}\n', 'stderr': '', 'timeout_seconds': 600}`
- Profiler summary: `{'available': False, 'status': 'NOT_AVAILABLE', 'hardware_mode': 'cpu_only', 'note': 'Nsight profiling was not run because a validated CUDA executable is unavailable.', 'workload': 'reduction_case_study'}`

## Candidate request

Try a double-precision accumulation candidate to reduce summation error; the gate must prove that the added conversion and precision cost are justified.

Return a real unified diff that adds or updates the configured candidate source. The diff is path-validated and applied in an isolated workspace before the candidate build. The candidate must remain independently testable. Do not treat generated code as trusted until the gates pass; CPU fallback timings cannot establish CUDA acceptance.
