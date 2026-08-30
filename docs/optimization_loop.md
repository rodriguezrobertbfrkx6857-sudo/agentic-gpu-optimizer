# Optimization loop

Each run follows the same sequence:

1. Load the workload specification.
2. Compile-check the fallback runner and record the CUDA source presence.
3. Execute the baseline correctness check.
4. Measure the baseline after warm-up and repeated samples.
5. Record profiler availability and render an optimization prompt.
6. Ask a provider for a candidate hypothesis and preserve its patch text.
7. Apply the proposal through an isolated, pre-registered workload variant and record that application step.
8. Build and test the candidate in the same runner.
9. Measure the candidate with the same shape, timer, warm-up, and iteration count.
10. Apply the correctness and performance gates.
11. Write JSON evidence for every stage and a concise case-study summary.

The current case studies use CPU wall-clock timing because the host has no CUDA runtime. A CUDA runner should replace the timing backend with CUDA Events while retaining the same evidence schema and gate logic.
