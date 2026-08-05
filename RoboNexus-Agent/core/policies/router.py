"""Pure skill-to-policy routing without transport or model lifecycle concerns."""

from .schemas import RouteDecision


class PolicyRoutingError(LookupError):
    pass


class PolicyRouter:
    def __init__(self, skill_registry, policy_catalog):
        self.skill_registry = skill_registry
        self.policy_catalog = policy_catalog
        self._validate_routes()

    def route(self, skill_id):
        try:
            skill = self.skill_registry.get(skill_id)
        except KeyError as exc:
            raise PolicyRoutingError("Unknown skill_id {!r}".format(skill_id)) from exc
        try:
            policy = self.policy_catalog.get(skill.policy_group)
        except KeyError as exc:
            raise PolicyRoutingError(
                "Skill {!r} references unknown policy {!r}".format(
                    skill_id, skill.policy_group
                )
            ) from exc
        if not policy.enabled:
            raise PolicyRoutingError(
                "Skill {!r} routes to disabled policy {!r}".format(
                    skill_id, policy.policy_id
                )
            )
        return RouteDecision(skill=skill, policy=policy)

    def policy_id_for(self, skill_id):
        return self.route(skill_id).policy.policy_id

    def _validate_routes(self):
        errors = []
        for skill_id in self.skill_registry.skill_ids():
            skill = self.skill_registry.get(skill_id)
            try:
                policy = self.policy_catalog.get(skill.policy_group)
            except KeyError:
                errors.append(
                    "skill {!r} references unknown policy {!r}".format(
                        skill_id, skill.policy_group
                    )
                )
                continue
            if not policy.enabled:
                errors.append(
                    "skill {!r} references disabled policy {!r}".format(
                        skill_id, skill.policy_group
                    )
                )
        if errors:
            raise PolicyRoutingError(
                "Invalid policy routes:\n- {}".format("\n- ".join(errors))
            )
