from __future__ import annotations

import os
import shlex
import shutil
import subprocess

from optimizer.config import WorkloadConfig

from .base import Candidate


class CommandProvider:
    """Adapter for an installed coding-agent CLI; never stores credentials."""

    def __init__(self, command: str | None = None) -> None:
        configured = command or os.environ.get("GPU_OPTIMIZER_AGENT_COMMAND", "")
        self.command = shlex.split(configured) if configured else []

    @property
    def available(self) -> bool:
        return bool(self.command and shutil.which(self.command[0]))

    def suggest(self, config: WorkloadConfig) -> Candidate:
        if not self.available:
            raise RuntimeError("configured agent command is not available")
        prompt = f"{config.candidate.rationale}\nReturn a patch only."
        result = subprocess.run([*self.command, prompt], capture_output=True, text=True, timeout=120, check=True)
        return Candidate(
            name=f"{config.candidate.name}-command",
            variant=config.candidate.variant,
            rationale=config.candidate.rationale,
            patch=result.stdout,
        )

