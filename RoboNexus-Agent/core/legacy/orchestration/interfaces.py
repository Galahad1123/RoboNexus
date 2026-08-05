"""Execution boundary consumed by the legacy goal state machine."""

from __future__ import annotations

from typing import Protocol

from core.legacy.execution.schemas import AttemptResult

from .schemas import AtomicAttemptRequest


class AtomicAttemptRunner(Protocol):
    def run_attempt(self, request: AtomicAttemptRequest) -> AttemptResult: ...
