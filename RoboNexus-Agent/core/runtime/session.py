"""State owner and hard-budget guard for one Qwen-Agent robot goal."""

from threading import RLock

from core.domain.schemas import (
    AgentSessionStatus,
    BudgetSnapshot,
    ErrorCode,
    GroundedObject,
    ObservationRef,
    ToolError,
    ToolResultEnvelope,
    ToolStatus,
)


class SessionRejected(RuntimeError):
    def __init__(self, code, message, retryable=False, details=None):
        self.error = ToolError(code, message, retryable, details or {})
        super().__init__(message)


class AgentSession:
    """Own mutable state; tools may mutate it only through these methods."""

    def __init__(self, goal, session_id=None, events=None):
        goal.validate()
        self.goal = goal
        self.session_id = session_id or goal.goal_id
        self.events = events
        self.status = AgentSessionStatus.ACTIVE
        self.termination_reason = None
        self.termination_details = {}
        self.environment_revision = goal.initial_observation.environment_revision
        self.environment_steps_used = goal.initial_observation.captured_at_step
        self.tool_calls_used = 0
        self.replans_used = 0
        self._attempts = {}
        self._achieved_skills = set()
        self._achievement_sequence = []
        self._logical_subtasks = {}
        self._logical_subtask_counter = 0
        self._observations = {
            goal.initial_observation.observation_id: goal.initial_observation
        }
        self._latest_observation_id = goal.initial_observation.observation_id
        self._groundings = {}
        self._completed_calls = {}
        self._in_progress_calls = set()
        self._observation_counter = 1
        self._grounding_counter = 0
        self._tool_call_counter = 0
        self._lock = RLock()

    @property
    def latest_observation(self):
        return self._observations[self._latest_observation_id]

    def budget_snapshot(self):
        limits = self.goal.limits
        return BudgetSnapshot(
            tool_calls_used=self.tool_calls_used,
            tool_calls_remaining=max(0, limits.tool_call_limit - self.tool_calls_used),
            environment_steps_used=self.environment_steps_used,
            environment_steps_remaining=max(
                0, limits.environment_step_limit - self.environment_steps_used
            ),
            replans_used=self.replans_used,
            replans_remaining=max(0, limits.replan_limit - self.replans_used),
        )

    def next_tool_call_id(self):
        with self._lock:
            self._tool_call_counter += 1
            return "tc_{:06d}".format(self._tool_call_counter)

    def start_tool_call(self, tool_call_id, tool_name):
        """Return a stored result for an idempotent replay, otherwise None."""
        with self._lock:
            if tool_call_id in self._completed_calls:
                return self._completed_calls[tool_call_id]
            self._require_active()
            if tool_call_id in self._in_progress_calls:
                raise SessionRejected(
                    ErrorCode.TOOL_CALL_IN_PROGRESS,
                    "tool call {!r} is already running".format(tool_call_id),
                    retryable=True,
                )
            if self.tool_calls_used >= self.goal.limits.tool_call_limit:
                self._exhaust_budget("tool_call_limit")
            self.tool_calls_used += 1
            self._in_progress_calls.add(tool_call_id)
            self._emit("tool_call_started", tool_call_id=tool_call_id, tool=tool_name)
            return None

    def finish_tool_call(self, result):
        with self._lock:
            call_id = result.tool_call_id
            if call_id not in self._in_progress_calls:
                raise ValueError("tool call {!r} was not started".format(call_id))
            self._in_progress_calls.remove(call_id)
            self._completed_calls[call_id] = result
            self._emit(
                "tool_call_finished",
                tool_call_id=call_id,
                status=result.status.value,
            )

    def fail_tool_call(self, tool_call_id):
        """Release a call after an uncaught wrapper exception."""
        with self._lock:
            self._in_progress_calls.discard(tool_call_id)

    def result(self, tool_call_id, status, data=None, error=None, artifacts=()):
        return ToolResultEnvelope(
            tool_call_id=tool_call_id,
            session_id=self.session_id,
            status=status,
            data=data or {},
            budget=self.budget_snapshot(),
            error=error,
            artifacts=tuple(artifacts),
        )

    def add_observation(self, static_image, wrist_image, environment_changed=False):
        with self._lock:
            self._require_active()
            if environment_changed:
                self.environment_revision += 1
            self._observation_counter += 1
            observation_id = "obs_{:06d}".format(self._observation_counter)
            observation = ObservationRef(
                observation_id=observation_id,
                environment_revision=self.environment_revision,
                captured_at_step=self.environment_steps_used,
                static_image=static_image,
                wrist_image=wrist_image,
            )
            observation.validate()
            self._observations[observation_id] = observation
            self._latest_observation_id = observation_id
            self._emit("observation_created", observation_id=observation_id)
            return observation

    def get_observation(self, observation_id, require_current=False):
        try:
            observation = self._observations[observation_id]
        except KeyError:
            raise SessionRejected(
                ErrorCode.STALE_OBSERVATION,
                "unknown observation {!r}".format(observation_id),
                retryable=True,
            )
        if require_current and observation.environment_revision != self.environment_revision:
            raise SessionRejected(
                ErrorCode.STALE_OBSERVATION,
                "observation {!r} belongs to revision {}, current is {}".format(
                    observation_id,
                    observation.environment_revision,
                    self.environment_revision,
                ),
                retryable=True,
            )
        return observation

    def add_environment_steps(self, steps):
        with self._lock:
            self._require_active()
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("steps must be a non-negative integer")
            if self.environment_steps_used + steps > self.goal.limits.environment_step_limit:
                self._exhaust_budget("environment_step_limit")
            self.environment_steps_used += steps
            if steps:
                self.environment_revision += 1

    def consume_attempt(self, logical_subtask_id):
        with self._lock:
            self._require_active()
            used = self._attempts.get(logical_subtask_id, 0)
            if used >= self.goal.limits.skill_attempt_limit:
                raise SessionRejected(
                    ErrorCode.ATTEMPT_BUDGET_EXHAUSTED,
                    "attempt budget exhausted for {!r}".format(logical_subtask_id),
                    retryable=False,
                    details={"attempts_used": used},
                )
            used += 1
            self._attempts[logical_subtask_id] = used
            return used

    def attempts_used(self, logical_subtask_id):
        return self._attempts.get(logical_subtask_id, 0)

    def maximum_attempts_used(self):
        return max(self._attempts.values()) if self._attempts else 0

    @property
    def achieved_skill_ids(self):
        return tuple(sorted(self._achieved_skills))

    @property
    def achievement_sequence(self):
        return tuple(self._achievement_sequence)

    def record_skill_achievement(self, skill_id, logical_subtask_id):
        with self._lock:
            self._require_active()
            skill_id = str(skill_id).strip()
            if not skill_id:
                raise ValueError("skill_id must be non-empty")
            self._achieved_skills.add(skill_id)
            # The evaluator sequence may legitimately contain the same atomic
            # skill more than once. Preserve occurrences while keeping the
            # achieved_skill_ids compatibility set unique.
            self._achievement_sequence.append(skill_id)
            self._emit(
                "skill_achieved",
                skill_id=skill_id,
                logical_subtask_id=str(logical_subtask_id),
            )

    def classify_successful_skill(self, skill_id):
        """Classify success using evaluator-private sequence state."""

        if self.goal.success_spec.get("mode") != "skill_sequence":
            return "goal_progress"
        predicates = self.goal.success_spec.get("predicates", ())
        expected = [
            item.get("name") if isinstance(item, dict) else item
            for item in predicates
        ]
        position = len(self._achievement_sequence)
        if position < len(expected) and skill_id == expected[position]:
            return "goal_progress"
        return "recovery"

    def record_recovery_achievement(self, skill_id, logical_subtask_id):
        with self._lock:
            self._require_active()
            self._emit(
                "recovery_skill_achieved",
                skill_id=str(skill_id),
                logical_subtask_id=str(logical_subtask_id),
            )

    def resolve_logical_subtask(self, skill_id, groundings):
        """Return a stable ID across re-grounding of the same selected instances."""
        fingerprint = (
            str(skill_id),
            tuple(
                sorted(
                    (
                        grounding.role.value,
                        grounding.metadata.get("instance_id")
                        or grounding.semantic_id
                        or grounding.grounding_id,
                    )
                    for grounding in groundings
                )
            ),
        )
        with self._lock:
            logical_subtask_id = self._logical_subtasks.get(fingerprint)
            if logical_subtask_id is None:
                self._logical_subtask_counter += 1
                logical_subtask_id = "subtask_{:03d}".format(
                    self._logical_subtask_counter
                )
                self._logical_subtasks[fingerprint] = logical_subtask_id
            return logical_subtask_id

    def consume_replan(self):
        with self._lock:
            self._require_active()
            if self.replans_used >= self.goal.limits.replan_limit:
                raise SessionRejected(
                    ErrorCode.REPLAN_BUDGET_EXHAUSTED,
                    "replan budget exhausted",
                    retryable=False,
                )
            self.replans_used += 1
            return self.replans_used

    def add_grounding(self, grounding):
        with self._lock:
            self._require_active()
            if not isinstance(grounding, GroundedObject):
                raise TypeError("grounding must be GroundedObject")
            grounding.validate()
            observation = self.get_observation(
                grounding.observation_id, require_current=True
            )
            if grounding.environment_revision != observation.environment_revision:
                raise SessionRejected(
                    ErrorCode.STALE_GROUNDING,
                    "grounding revision does not match observation",
                    retryable=True,
                )
            if grounding.grounding_id in self._groundings:
                raise ValueError(
                    "duplicate grounding_id {!r}".format(grounding.grounding_id)
                )
            self._groundings[grounding.grounding_id] = grounding
            self._emit("grounding_created", grounding_id=grounding.grounding_id)

    def next_grounding_id(self):
        with self._lock:
            self._grounding_counter += 1
            return "ground_{:06d}".format(self._grounding_counter)

    def get_grounding(self, grounding_id, require_current=True):
        try:
            grounding = self._groundings[grounding_id]
        except KeyError:
            raise SessionRejected(
                ErrorCode.GROUNDING_NOT_FOUND,
                "unknown grounding {!r}".format(grounding_id),
                retryable=True,
            )
        if (
            require_current
            and grounding.tracker_handle is None
            and grounding.environment_revision != self.environment_revision
        ):
            raise SessionRejected(
                ErrorCode.STALE_GROUNDING,
                "grounding {!r} belongs to revision {}, current is {}".format(
                    grounding_id,
                    grounding.environment_revision,
                    self.environment_revision,
                ),
                retryable=True,
            )
        return grounding

    def record_verification(self, status, reason, evidence=None):
        with self._lock:
            self._require_active()
            self._emit(
                "verification_result",
                status=str(status),
                reason=str(reason),
                evidence=dict(evidence or {}),
            )

    def mark_succeeded(self, verification_status, reason, details=None):
        with self._lock:
            self._require_active()
            if verification_status != "success":
                raise SessionRejected(
                    ErrorCode.INVALID_PRECONDITION,
                    "session success requires verify_goal status=success",
                    retryable=True,
                )
            self.status = AgentSessionStatus.SUCCEEDED
            self.termination_reason = reason
            self.termination_details = dict(details or {})
            self._emit(
                "goal_succeeded", reason=reason, details=self.termination_details
            )

    def abort(self, status, reason, details=None):
        with self._lock:
            self._require_active()
            if status not in {
                AgentSessionStatus.ABORTED_BY_AGENT,
                AgentSessionStatus.ABORTED_BY_USER,
                AgentSessionStatus.DEPENDENCY_FAILURE,
                AgentSessionStatus.SAFETY_STOP,
            }:
                raise ValueError("invalid abort status {!r}".format(status))
            self.status = status
            self.termination_reason = reason
            self.termination_details = dict(details or {})
            self._emit(
                "goal_aborted",
                status=status.value,
                reason=reason,
                details=self.termination_details,
            )

    def _exhaust_budget(self, limit_name):
        self.status = AgentSessionStatus.BUDGET_EXHAUSTED
        self.termination_reason = limit_name
        self._emit("goal_aborted", status=self.status.value, reason=limit_name)
        raise SessionRejected(
            ErrorCode.GOAL_BUDGET_EXHAUSTED,
            "goal budget exhausted: {}".format(limit_name),
            retryable=False,
        )

    def _require_active(self):
        if self.status != AgentSessionStatus.ACTIVE:
            raise SessionRejected(
                ErrorCode.SESSION_TERMINAL,
                "session is not active: {}".format(self.status.value),
                retryable=False,
            )

    def _emit(self, event, **payload):
        if self.events is not None:
            self.events.emit(event, session_id=self.session_id, **payload)
