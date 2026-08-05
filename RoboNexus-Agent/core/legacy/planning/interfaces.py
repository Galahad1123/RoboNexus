"""Legacy planner protocol retained for deterministic regression tests."""

from __future__ import annotations

from typing import Protocol

from .schemas import GoalSpec, PlannerContext, PlannerDecision


class Planner(Protocol):
    def start(self, goal: GoalSpec) -> None: ...
    def next_step(self, context: PlannerContext) -> PlannerDecision: ...
    def replan(self, context: PlannerContext) -> PlannerDecision: ...
