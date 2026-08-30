from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from optimizer.config import WorkloadConfig


@dataclass(frozen=True)
class Candidate:
    name: str
    variant: str
    rationale: str
    patch: str


class Provider(Protocol):
    def suggest(self, config: WorkloadConfig) -> Candidate:
        ...

