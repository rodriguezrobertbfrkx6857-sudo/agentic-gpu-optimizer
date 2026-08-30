from __future__ import annotations

import shutil

from optimizer.config import WorkloadConfig


def profile_workload(config: WorkloadConfig) -> dict:
    ncu = shutil.which("ncu")
    nsys = shutil.which("nsys")
    if not ncu and not nsys:
        return {
            "available": False,
            "status": "NOT_AVAILABLE",
            "note": "Nsight profiling was not available in the current environment.",
            "workload": config.name,
        }
    return {
        "available": True,
        "status": "NOT_RUN",
        "tools": {"ncu": bool(ncu), "nsys": bool(nsys)},
        "note": "Profiler availability was detected; this CPU fallback run does not claim GPU counters.",
        "workload": config.name,
    }

