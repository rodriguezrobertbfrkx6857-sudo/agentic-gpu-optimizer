# Safety and validation

An optimization proposal is untrusted input. The system does not merge arbitrary source text or infer correctness from compilation. It validates the unified-diff paths against the workload contract, applies the patch only in a fresh run-local repository, builds the selected source, checks its output against a reference tolerance, measures it repeatedly, and records the reason for the final decision.

The default policy is:

- correctness failure: `REJECT`;
- negative median improvement: `REJECT`;
- correct and stable improvement at or above the configured threshold: `ACCEPT`;
- a correct CPU-only improvement while CUDA is unavailable: `INCONCLUSIVE`;
- correct but noisy or below threshold: `INCONCLUSIVE`.

The CUDA acceptance path requires both baseline and candidate to report `hardware_mode=cuda`, `backend=cuda`, and `cuda_validated=true`. A CPU fallback is useful for smoke testing and reproducibility, but it is never substituted for a CUDA build or profiler result.

External command providers are opt-in. API credentials are not accepted in configuration files; a future provider may read them from environment variables owned by the caller. Current public artifacts contain no provider output requiring an account.
