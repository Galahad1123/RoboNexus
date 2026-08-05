"""Goal state machine with bounded attempts and bounded replanning."""

from __future__ import annotations

import hashlib

from core.legacy.inspection import InspectionStatus
from core.logging import NullEventLogger
from core.legacy.planning import (
    PlannerContext,
    PlannerDecisionType,
    PlannerFeedback,
)

from .schemas import (
    AtomicAttemptRequest,
    GoalExecutionConfig,
    GoalExecutionResult,
    GoalStatus,
    SkillExecutionRecord,
)


class GoalOrchestrator:
    def __init__(
        self,
        planner,
        attempt_runner,
        config=None,
        events=None,
        state_inspector=None,
    ) -> None:
        self.planner = planner
        self.attempt_runner = attempt_runner
        self.config = config or GoalExecutionConfig()
        self.events = events or NullEventLogger()
        self.state_inspector = state_inspector

    def run(self, goal, initial_observation=None) -> GoalExecutionResult:
        self.planner.start(goal)
        observation = initial_observation
        completed = []
        records = []
        attempts_executed = 0
        replans_used = 0
        plan_step = 0
        feedback = None
        self.events.emit(
            "goal_started",
            goal_id=goal.goal_id,
            instruction=goal.instruction,
            max_attempts_per_skill=self.config.max_attempts_per_skill,
            max_replans=self.config.max_replans,
        )

        decision = self.planner.next_step(
            self._context(goal, completed, observation, replans_used, feedback)
        )
        while True:
            self._log_decision(goal.goal_id, decision, replans_used)
            if decision.decision_type == PlannerDecisionType.COMPLETE:
                return self._finish(
                    goal,
                    GoalStatus.SUCCEEDED,
                    completed,
                    records,
                    attempts_executed,
                    replans_used,
                    decision.reason or "planner_complete",
                    observation,
                )
            if decision.decision_type == PlannerDecisionType.ABORT:
                return self._finish(
                    goal,
                    GoalStatus.ABORTED,
                    completed,
                    records,
                    attempts_executed,
                    replans_used,
                    decision.reason or "planner_abort",
                    observation,
                )

            skill_id = decision.skill_id
            plan_step += 1
            record = SkillExecutionRecord(skill_id, plan_step, replans_used)
            records.append(record)
            self.events.emit(
                "skill_started",
                goal_id=goal.goal_id,
                skill_id=skill_id,
                plan_step=plan_step,
                replans_used=replans_used,
            )
            inspection = self._inspect(skill_id, observation)
            if inspection is not None:
                self.events.emit(
                    "skill_inspection",
                    goal_id=goal.goal_id,
                    skill_id=skill_id,
                    plan_step=plan_step,
                    status=inspection.status.value,
                    reason=inspection.reason,
                    evidence=inspection.evidence,
                )
                if inspection.status == InspectionStatus.SATISFIED:
                    record.skipped = True
                    record.inspection_reason = inspection.reason
                    completed.append(skill_id)
                    feedback = None
                    self.events.emit(
                        "skill_skipped",
                        goal_id=goal.goal_id,
                        skill_id=skill_id,
                        plan_step=plan_step,
                        reason=inspection.reason,
                    )
                    decision = self.planner.next_step(
                        self._context(
                            goal, completed, observation, replans_used, feedback
                        )
                    )
                    continue
                if inspection.status == InspectionStatus.INVALID_PRECONDITION:
                    record.inspection_reason = inspection.reason
                    feedback = PlannerFeedback(
                        failed_skill_id=skill_id,
                        attempts=0,
                        termination_reason=inspection.reason,
                    )
                    self.events.emit(
                        "skill_precondition_invalid",
                        goal_id=goal.goal_id,
                        skill_id=skill_id,
                        plan_step=plan_step,
                        replans_used=replans_used,
                        reason=inspection.reason,
                        evidence=inspection.evidence,
                    )
                    if replans_used >= self.config.max_replans:
                        return self._finish(
                            goal,
                            GoalStatus.ABORTED,
                            completed,
                            records,
                            attempts_executed,
                            replans_used,
                            "replan_budget_exhausted",
                            observation,
                        )
                    replans_used += 1
                    self.events.emit(
                        "replan_started",
                        goal_id=goal.goal_id,
                        failed_skill_id=skill_id,
                        replans_used=replans_used,
                        max_replans=self.config.max_replans,
                        trigger="invalid_precondition",
                    )
                    decision = self.planner.replan(
                        self._context(
                            goal, completed, observation, replans_used, feedback
                        )
                    )
                    continue
            for attempt in range(1, self.config.max_attempts_per_skill + 1):
                request = AtomicAttemptRequest(
                    goal=goal,
                    skill_id=skill_id,
                    attempt=attempt,
                    plan_step=plan_step,
                    replans_used=replans_used,
                    episode_seed=_attempt_seed(
                        self.config.base_seed,
                        goal.goal_id,
                        plan_step,
                        replans_used,
                        attempt,
                    ),
                    observation=observation,
                )
                self.events.emit(
                    "goal_attempt_started",
                    goal_id=goal.goal_id,
                    skill_id=skill_id,
                    plan_step=plan_step,
                    replans_used=replans_used,
                    attempt=attempt,
                    episode_seed=request.episode_seed,
                    scene_continuation=True,
                )
                result = self.attempt_runner.run_attempt(request)
                record.attempts.append(result)
                attempts_executed += 1
                observation = result.observation
                self.events.emit(
                    "goal_attempt_succeeded"
                    if result.success
                    else "goal_attempt_failed",
                    goal_id=goal.goal_id,
                    skill_id=skill_id,
                    plan_step=plan_step,
                    replans_used=replans_used,
                    attempt=attempt,
                    steps=result.steps,
                    reason=result.termination_reason,
                )
                if result.success:
                    break

            if record.success:
                completed.append(skill_id)
                feedback = None
                self.events.emit(
                    "skill_succeeded",
                    goal_id=goal.goal_id,
                    skill_id=skill_id,
                    plan_step=plan_step,
                    attempts=len(record.attempts),
                )
                decision = self.planner.next_step(
                    self._context(
                        goal, completed, observation, replans_used, feedback
                    )
                )
                continue

            last_result = record.attempts[-1]
            feedback = PlannerFeedback(
                failed_skill_id=skill_id,
                attempts=len(record.attempts),
                termination_reason=last_result.termination_reason,
            )
            self.events.emit(
                "skill_attempts_exhausted",
                goal_id=goal.goal_id,
                skill_id=skill_id,
                plan_step=plan_step,
                attempts=len(record.attempts),
                replans_used=replans_used,
                reason=last_result.termination_reason,
            )
            if replans_used >= self.config.max_replans:
                return self._finish(
                    goal,
                    GoalStatus.ABORTED,
                    completed,
                    records,
                    attempts_executed,
                    replans_used,
                    "replan_budget_exhausted",
                    observation,
                )
            replans_used += 1
            self.events.emit(
                "replan_started",
                goal_id=goal.goal_id,
                failed_skill_id=skill_id,
                replans_used=replans_used,
                max_replans=self.config.max_replans,
            )
            decision = self.planner.replan(
                self._context(goal, completed, observation, replans_used, feedback)
            )

    def _inspect(self, skill_id, observation):
        if self.state_inspector is None:
            return None
        return self.state_inspector.inspect(skill_id, observation)

    @staticmethod
    def _context(goal, completed, observation, replans_used, feedback):
        return PlannerContext(
            goal=goal,
            completed_skill_ids=tuple(completed),
            observation=observation,
            replans_used=replans_used,
            feedback=feedback,
        )

    def _log_decision(self, goal_id, decision, replans_used):
        self.events.emit(
            "planner_decision",
            goal_id=goal_id,
            decision=decision.decision_type.value,
            skill_id=decision.skill_id,
            reason=decision.reason,
            replans_used=replans_used,
        )

    def _finish(
        self,
        goal,
        status,
        completed,
        records,
        attempts_executed,
        replans_used,
        reason,
        observation,
    ):
        self.events.emit(
            "goal_succeeded" if status == GoalStatus.SUCCEEDED else "goal_aborted",
            goal_id=goal.goal_id,
            completed_skill_ids=list(completed),
            attempts_executed=attempts_executed,
            replans_used=replans_used,
            reason=reason,
        )
        return GoalExecutionResult(
            goal=goal,
            status=status,
            completed_skill_ids=tuple(completed),
            skill_records=tuple(records),
            attempts_executed=attempts_executed,
            replans_used=replans_used,
            termination_reason=reason,
            observation=observation,
        )


def _attempt_seed(base_seed, goal_id, plan_step, replans_used, attempt):
    digest = hashlib.sha256(
        "{}:{}:{}:{}:{}".format(
            base_seed, goal_id, plan_step, replans_used, attempt
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)
