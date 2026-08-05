"""Domain types for model deployment configuration."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    enabled: bool
    checkpoint: Optional[str]
    host: str
    port: int

    @property
    def endpoint(self):
        return "ws://{}:{}".format(self.host, self.port)


@dataclass(frozen=True)
class RouteDecision:
    """Immutable result of resolving one canonical skill to a deployment."""

    skill: Any
    policy: PolicySpec
