# Safety and validation

An optimization proposal is untrusted input. The system does not merge arbitrary source text or infer correctness from compilation. It executes a known workload variant, checks its output against a reference tolerance, measures it repeatedly, and records the reason for the final decision.

The default policy is:

- correctness failure: `REJECT`;
- negative median improvement: `REJECT`;
- correct and stable improvement at or above the configured threshold: `ACCEPT`;
- correct but noisy or below threshold: `INCONCLUSIVE`.

External command providers are opt-in. API credentials are not accepted in configuration files; a future provider may read them from environment variables owned by the caller. Current public artifacts contain no provider output requiring an account.

