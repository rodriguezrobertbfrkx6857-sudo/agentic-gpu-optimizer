from __future__ import annotations

from optimizer.providers.base import Candidate


def apply_candidate(candidate: Candidate, run_dir) -> dict:
    """Select a pre-registered workload variant without executing arbitrary patch text."""

    return {
        "status": "APPLIED",
        "mode": "isolated_registered_variant",
        "candidate": candidate.name,
        "variant": candidate.variant,
        "patch_artifact": "candidate.patch",
        "note": "The controlled case-study runner applies a provider proposal by selecting an explicit workload variant; arbitrary source execution is not enabled.",
    }

