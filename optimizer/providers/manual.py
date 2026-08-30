from __future__ import annotations

from optimizer.config import WorkloadConfig

from .base import Candidate


class ManualProvider:
    """Provider used for deterministic runs without an external model or API key."""

    def suggest(self, config: WorkloadConfig) -> Candidate:
        return Candidate(
            name=config.candidate.name,
            variant=config.candidate.variant,
            rationale=config.candidate.rationale,
            patch=config.candidate.patch,
        )

