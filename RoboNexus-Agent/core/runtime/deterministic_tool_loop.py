"""Deterministic, backend-neutral harness for exercising the Agent tools."""

from dataclasses import asdict, dataclass

from core.domain import ToolStatus


@dataclass(frozen=True)
class DeterministicLoopConfig:
    skill_id: str
    proposals: tuple
    max_attempts: int = 3
    approach_role: str = "source"
    xy_threshold_m: float = 0.10

    def validate(self):
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be non-empty")
        if not isinstance(self.proposals, tuple) or not self.proposals:
            raise ValueError("proposals must be a non-empty tuple")
        if self.approach_role not in {"source", "target"}:
            raise ValueError("approach_role must be source or target")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be positive")


class DeterministicToolLoop:
    """Drive one atomic goal only through the frozen public tool contracts."""

    def __init__(
        self,
        session,
        inspect_scene,
        ground_targets,
        execute_skill,
        approach_target,
        verify_goal,
        abort_goal,
    ):
        self.session = session
        self.inspect_scene = inspect_scene
        self.ground_targets = ground_targets
        self.execute_skill = execute_skill
        self.approach_target = approach_target
        self.verify_goal = verify_goal
        self.abort_goal = abort_goal

    def run(self, config):
        config.validate()
        trace = []
        initial = self._call(
            trace,
            "inspect_scene",
            self.inspect_scene,
            {"reason": "start deterministic tool loop"},
        )
        if initial.status != ToolStatus.SUCCEEDED:
            return self._abort(trace, "insufficient_observation", "initial inspection failed")

        for attempt in range(1, config.max_attempts + 1):
            ground = self._ground(trace, config, "attempt_{}".format(attempt))
            if ground.status != ToolStatus.SUCCEEDED:
                return self._abort(trace, "insufficient_observation", "target grounding failed")
            grounding_ids = [item["grounding_id"] for item in ground.data["groundings"]]
            execution = self._call(
                trace,
                "execute_skill",
                self.execute_skill,
                {
                    "skill_id": config.skill_id,
                    "grounding_ids": grounding_ids,
                    "intent": "deterministic attempt {} for the current goal".format(attempt),
                },
            )
            if execution.status in {ToolStatus.REJECTED, ToolStatus.UNAVAILABLE}:
                return self._abort(
                    trace,
                    "required_skill_unavailable",
                    "execute_skill was {}".format(execution.status.value),
                )

            verification = self._call(
                trace,
                "verify_goal",
                self.verify_goal,
                {
                    "claim": "the complete requested goal is satisfied",
                    "observation_id": self.session.latest_observation.observation_id,
                },
            )
            if (
                verification.status == ToolStatus.SUCCEEDED
                and verification.data.get("status") == "success"
            ):
                return self._report(trace, "succeeded", attempt)
            if verification.status in {ToolStatus.REJECTED, ToolStatus.UNAVAILABLE}:
                return self._abort(
                    trace,
                    "required_skill_unavailable",
                    "verify_goal was {}".format(verification.status.value),
                )
            if attempt == config.max_attempts:
                break

            refreshed = self._call(
                trace,
                "inspect_scene",
                self.inspect_scene,
                {"reason": "inspect failed attempt before deterministic recovery"},
            )
            if refreshed.status != ToolStatus.SUCCEEDED:
                return self._abort(
                    trace, "insufficient_observation", "recovery inspection failed"
                )
            recovery_ground = self._ground(
                trace, config, "recovery_{}".format(attempt)
            )
            if recovery_ground.status != ToolStatus.SUCCEEDED:
                return self._abort(
                    trace, "insufficient_observation", "recovery grounding failed"
                )
            recovery_id = self._approach_grounding_id(
                recovery_ground, config.approach_role
            )
            approach = self._call(
                trace,
                "approach_target",
                self.approach_target,
                {
                    "grounding_id": recovery_id,
                    "xy_threshold_m": config.xy_threshold_m,
                    "intent": "approach the target before deterministic retry",
                },
            )
            if approach.status in {ToolStatus.REJECTED, ToolStatus.UNAVAILABLE}:
                return self._abort(
                    trace,
                    "required_skill_unavailable",
                    "approach_target was {}".format(approach.status.value),
                )
            after_approach = self._call(
                trace,
                "inspect_scene",
                self.inspect_scene,
                {"reason": "refresh observation after deterministic recovery"},
            )
            if after_approach.status != ToolStatus.SUCCEEDED:
                return self._abort(
                    trace, "insufficient_observation", "post-recovery inspection failed"
                )

        return self._abort(
            trace,
            "repeated_execution_failure",
            "goal remained unsatisfied after {} attempts".format(config.max_attempts),
        )

    def _ground(self, trace, config, phase):
        proposals = []
        for proposal in config.proposals:
            item = dict(proposal)
            item.setdefault("description", "deterministic {} selection".format(phase))
            proposals.append(item)
        return self._call(
            trace,
            "ground_targets",
            self.ground_targets,
            {
                "skill_id": config.skill_id,
                "observation_id": self.session.latest_observation.observation_id,
                "proposals": proposals,
            },
        )

    @staticmethod
    def _approach_grounding_id(result, role):
        for grounding in result.data["groundings"]:
            if grounding["role"] == role:
                return grounding["grounding_id"]
        raise ValueError("grounding result has no {!r} role".format(role))

    @staticmethod
    def _call(trace, name, tool, params):
        result = tool.call(params)
        trace.append({"tool": name, "result": result.to_dict()})
        return result

    def _abort(self, trace, reason_code, summary):
        if not self.session.status.terminal:
            self._call(
                trace,
                "abort_goal",
                self.abort_goal,
                {
                    "reason_code": reason_code,
                    "summary": summary,
                    "last_observation_id": self.session.latest_observation.observation_id,
                },
            )
        return self._report(trace, self.session.status.value, None)

    def _report(self, trace, status, successful_attempt):
        return {
            "status": status,
            "successful_attempt": successful_attempt,
            "session_status": self.session.status.value,
            "termination_reason": self.session.termination_reason,
            "budget": asdict(self.session.budget_snapshot()),
            "trace": trace,
        }
