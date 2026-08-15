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
  grounding/     CALVIN Oracle semantic grounding and masks
  logging/       CALVIN evaluation event logging
  patches/       narrowly scoped simulator anomaly guards
```

Environment-independent contracts and orchestration must remain in
`RoboNexus-Agent/core`.

The package root intentionally contains no implementation modules. Add new
CALVIN integrations to the matching capability package and expose only stable,
dependency-safe public names from `__init__.py` when a package-level import is
actually needed.

## Grounding vocabulary

The default Oracle semantic vocabulary is configured in
`grounding/semantic_vocabulary.json`. It is versioned and declares the CALVIN
splits it supports. Extend or replace this file for ABCD experiments instead of
adding semantic IDs to Python code. `CalvinOracleGrounder` also accepts a
`vocabulary_path` or an already loaded `vocabulary` object, which allows each
experiment to select its vocabulary explicitly.

Policy-specific outline rendering is configured in
`execution/outline_styles.json`. These styles mirror the specialist training
pipelines and must be updated alongside any checkpoint trained with a different
outline convention.

`recovery/approach_controller.py` implements the optional Agent recovery tool.
It resolves the current grounded instance, lifts to a conservative safe height,
and approaches only until the configured XY threshold is reached. It does not
perform grasping, placement, or task-specific recovery.

`verification/goal_verifier.py` evaluates every predicate in the immutable goal
success specification against the CALVIN goal-start state. Goal reference
images remain UI/Qwen evidence and do not affect Oracle conclusions.

## Simulator patches

`patches/table_cube_penetration.py` is a CALVIN-only guard for the known Bullet
failure where an off-center top-down gripper contact pushes a tabletop cube
through the table. It is enabled only for table-source lift, lateral push, and
rotate skills. Valid motion is read-only: no friction, timestep, solver, action,
or policy input is modified. Two consecutive samples more than 8 mm below the
attempt's valid table height stop the attempt; after the robot is restored to a
safe pose, only the affected cube is returned to its last valid pose with zero
velocity. Drawer/slider placement and push-into-drawer are explicitly outside
its scope. Detection is additionally restricted to the tabletop XY footprint
(with a conservative allowance for normal lateral pushes). Once the cube leaves
that footprint the detector disables itself for the attempt, so a genuine fall
past the table edge is never rewound as simulator penetration.
