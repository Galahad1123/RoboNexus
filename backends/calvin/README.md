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
