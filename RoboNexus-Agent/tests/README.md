# Agent test layout

This suite is grouped by ownership and retention policy:

- `runtime/`: active session runtime, event logging, and deterministic diagnostic harness;
- `tool_contracts/`: active backend-neutral Qwen-Agent tool contracts;
- `config/`: registry, routing, handshake, and architecture-boundary validation;
- `backend_adapters/`: active environment-specific adapter behavior;
- `legacy/`: compatibility coverage for code under `core/legacy` and its retained entrypoints.

`legacy/` is the only directory intended for eventual bulk removal. Delete a legacy test only
in the same change that removes the implementation or compatibility entrypoint it protects.
Do not place temporary smoke scripts or real-environment outputs in this tree; those belong in
`scripts/agent/` and ignored `outputs/` directories respectively.

Run the complete environment-neutral suite with:

```bash
PYTHONPATH=RoboNexus-Agent:. python -m unittest discover -s RoboNexus-Agent/tests
```

Qwen-dependent adapter tests remain in the sibling `tests_qwen/` directory and must run inside
the `robonexus-qwen` environment.
