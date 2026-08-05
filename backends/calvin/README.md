# CALVIN backend

This package implements CALVIN-specific adapters behind the Agent's generic
ports and tools. New code should import implementations from their capability
package rather than from the compatibility modules at this directory's root.

```text
calvin/
  environment/   observation capture and environment adapters
  execution/     policy attempts, routed goal attempts, and retry loops
  recovery/      Oracle localization and conservative recovery motion
  verification/  CALVIN Oracle verification and facility-state inspection
  logging/       CALVIN evaluation event logging
```

Future target-selection implementations belong in `grounding/`. Environment-
independent contracts and orchestration must remain in `RoboNexus-Agent/core`.

The package root intentionally contains no implementation modules. Add new
CALVIN integrations to the matching capability package and expose only stable,
dependency-safe public names from `__init__.py` when a package-level import is
actually needed.
