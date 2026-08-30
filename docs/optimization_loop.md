# Optimization loop

Each run follows the same sequence:

1. Load the workload specification.
2. Create an isolated baseline workspace and either compile the CUDA source with `nvcc` or compile-check the CPU fallback, recording the actual mode, return code, duration, compiler, and hashes.
3. Execute the baseline correctness check.
4. Measure the baseline after warm-up and repeated samples.
5. Record profiler availability and render an optimization prompt.
6. Ask a provider for a candidate hypothesis and preserve its patch text.
7. Validate the patch paths, initialize the isolated candidate workspace, and apply the real unified diff with `git apply --check` followed by `git apply`.
8. Build and test the candidate in the isolated workspace; a CUDA host must use `nvcc` and a CPU-only host may use only the explicitly labeled fallback.
9. Measure the candidate with the same shape, timer, warm-up, and iteration count.
10. Apply the correctness and performance gates.
11. Write JSON evidence for every stage and a concise case-study summary. Each run also keeps `environment.json`, `source_hashes.json`, `baseline_build.json`, and `candidate_build.json` as separate audit artifacts.

The current case studies use CPU wall-clock timing because the host has no NVIDIA driver or CUDA Toolkit. A CUDA host uses CUDA Events and the same evidence schema. CPU-only timing can document the control flow and fallback, but it cannot produce an `ACCEPT` decision for a CUDA optimization.
