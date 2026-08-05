"""Deterministic legacy planner used only by compatibility tests."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import GoalSpec, PlannerContext, PlannerDecision


class ScriptedPlanner:
    """Yield one atomic skill from an initial script or explicit replan scripts."""

    def __init__(
        self,
        steps: Iterable[str],
        replan_scripts: Iterable[Iterable[str]] = (),
        skill_registry=None,
    ) -> None:
        self._initial_script = _validated_script(steps, "steps", skill_registry)
        self._replan_scripts = tuple(
            _validated_script(script, "replan_scripts", skill_registry)
            for script in replan_scripts
        )
        self._goal = None
        self._active_script = self._initial_script
        self._cursor = 0
        self._replan_cursor = 0
        self.request_count = 0

    def start(self, goal: GoalSpec) -> None:
        if not isinstance(goal.goal_id, str) or not goal.goal_id.strip():
            raise ValueError("goal.goal_id must be a non-empty string")
        self._goal = goal
        self._active_script = self._initial_script
        self._cursor = 0
        self._replan_cursor = 0
        self.request_count = 0

    def next_step(self, context: PlannerContext) -> PlannerDecision:
        self._require_started(context)
        self.request_count += 1
        if self._cursor >= len(self._active_script):
            return PlannerDecision.complete()
        skill_id = self._active_script[self._cursor]
        self._cursor += 1
        return PlannerDecision.execute(skill_id)

    def replan(self, context: PlannerContext) -> PlannerDecision:
        self._require_started(context)
        if context.feedback is None:
            raise ValueError("replan requires failure feedback")
        if self._replan_cursor >= len(self._replan_scripts):
            self.request_count += 1
            return PlannerDecision.abort("scripted_replan_unavailable")
        self._active_script = self._replan_scripts[self._replan_cursor]
        self._replan_cursor += 1
        self._cursor = 0
        return self.next_step(context)

    def _require_started(self, context: PlannerContext) -> None:
        if self._goal is None:
            raise RuntimeError("Planner has not been started")
        if context.goal.goal_id != self._goal.goal_id:
            raise ValueError(
                "Planner context goal_id {!r} does not match {!r}".format(
                    context.goal.goal_id, self._goal.goal_id
                )
            )


def _validated_script(values, path, registry):
    if isinstance(values, (str, bytes)):
        raise TypeError("{} must be an iterable of skill IDs".format(path))
    script = tuple(values)
    for index, skill_id in enumerate(script):
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("{}[{}] must be a non-empty skill ID".format(path, index))
        if registry is not None and skill_id not in registry:
            raise ValueError("{}[{}] is unknown: {!r}".format(path, index, skill_id))
    return script
