# Agent code migration map

This file records which paths are canonical, compatible, legacy, or scheduled
for removal while RoboNexus moves to the Qwen-Agent architecture.

## Canonical paths

| Concern | Canonical path | Notes |
|---|---|---|
| Public domain types | `RoboNexus-Agent/core/domain/` | The only contracts new tools may use |
| Session and budgets | `RoboNexus-Agent/core/runtime/` | Owns state, revisions, idempotency and terminal status |
| Tool implementations | `RoboNexus-Agent/core/tools/` | Backend-neutral; implemented in protocol order |
| Backend interfaces | `RoboNexus-Agent/core/ports/` | No CALVIN/Qwen/model imports |
| Skill registry | `RoboNexus-Agent/core/skills/` | Retained |
| Policy routing/clients | `RoboNexus-Agent/core/policies/` | Migrated from `core/routing` |
| Events | `RoboNexus-Agent/core/logging/` | Retained until artifact/event runtime is consolidated |
| Qwen integration | `RoboNexus-Agent/adapters/qwen_agent/` | The only package allowed to import `qwen_agent` |
| CALVIN implementation | `backends/calvin/` | Implements core ports; no Agent planning |

## Removed old core paths

The following ambiguous paths were removed rather than kept as directory-level
compatibility facades:

| Removed path | Replacement/archive |
|---|---|
| `core/agent` | `core/domain` + `core/runtime` |
| `core/routing` | `core/policies` |
| `core/planning` | `core/legacy/planning` |
| `core/orchestration` | `core/legacy/orchestration` |
| `core/execution` | `core/legacy/execution` or new `core/ports` |
| `core/inspection` | `core/legacy/inspection` or new verifier/environment ports |
| `core/common` | `core/legacy/perception/schemas.py` |
| `core/perception` | `core/legacy/perception` |
| `core/transport` | `core/legacy/transport` |
| `core/localization` | `core/ports/grounding.py` |
| `core/motion` | `core/ports/motion.py` |
| `core/verification` | `core/ports/verification.py` |

Repository-level `agent/` and flat `core.schemas/core.clients/...` imports remain
only for old external entry points. No new behavior may be added to them.

The former `core/common`, `core/perception`, and `core/transport` directories
were removed from canonical core. Their still-tested early perception pipeline
is archived under `core/legacy/perception` and `core/legacy/transport` and is
reachable only through legacy flat compatibility imports. The former
`core/localization`, `core/motion`, and `core/verification` forwarding packages
were removed entirely; their replacements are the interfaces in `core/ports`.

## Retained implementations to adapt behind tools

- `core/skills`: 34-skill config, loader, validator;
- `core/policies`: router, client pool, handshake validation;
- `backends/calvin/execution/attempt_runner.py`: low-level continuous-scene action loop;
- `backends/calvin/verification/task_verifier.py`: Oracle atomic verifier;
- `backends/calvin/verification/state_inspector.py`: facility state and preconditions;
- `backends/calvin/recovery/approach_recovery.py`: candidate implementation for the approach port;
- `calvin/evaluation/remote_policy/calvin_env_adapter.py`: simulator adapter;
- `calvin/evaluation/remote_policy/masking/`: GT masks and outline renderer;
- `core/logging`: event sinks;
- existing video writer behavior.

These modules may be wrapped first. They are only rewritten when their current
contract cannot implement a port cleanly.

## Legacy-only implementations

- `core/legacy/planning/ScriptedPlanner`;
- `core/legacy/orchestration/GoalOrchestrator`;
- `scripts/agent/run_calvin_goal.py`;
- `--replan-script` and the old scripted goal CLI;
- old goal acceptance tests that exercise the deterministic harness.
- `core/legacy/perception` and `core/legacy/transport` early model-service pipeline.

They remain for regression and comparison. Qwen tools must not import them.

## Types scheduled for consolidation

The following older types remain because existing tests/backends use them, but
new tools must use `core.domain`:

- `core/legacy/perception/schemas.py` perception pipeline records;
- `core/legacy/execution/schemas.py` retry-era task, verification and motion records;
- `core/legacy/inspection/schemas.py` fixed-state inspection result;
- `core/legacy/planning/schemas.py` scripted plan records;
- `core/legacy/orchestration/schemas.py` scripted goal records.

They will be removed only after every retained backend implements a new port
and all compatibility tests have a replacement. No second public schema set may
be added during the migration.

## Dependency rules

```text
adapters/qwen_agent
        ↓
core/tools → core/runtime → core/domain
        ↓             ↓
      core/ports   core/logging
        ↓
backends/calvin
```

Canonical core packages must not import:

- `qwen_agent`;
- `calvin` or `backends.calvin`;
- `torch`, `transformers`, SAM or Cutie;
- removed legacy aliases or flat compatibility modules;
- legacy planning/orchestration.

## Removal gate

An old implementation can be removed only when:

1. its replacement port/tool has unit tests;
2. the CALVIN implementation passes a real integration test;
3. no canonical module imports it;
4. repository search finds only explicit compatibility tests/imports;
5. the removal is recorded in this map.
