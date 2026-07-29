# CALVIN remote-policy evaluation

This directory contains the environment side of closed-loop, single-task
evaluation. The policy runs in the independent `multitask_dit_policy` Python
environment and exposes one action per WebSocket request. CALVIN owns simulator
reset, observation adaptation, success checking, videos, and metrics.

Task-set differences live in suite YAML files. The generic rollout code does not
contain CALVIN-D-10 task names. Dataset-specific entry points live with their
dataset workspace, for example `datasets/calvin_d_10/scripts/evaluate.py`.

## Install client dependencies

```bash
conda activate robonexus-calvin
python -m pip install -r calvin/evaluation/remote_policy/requirements.txt
```

After adding the server dependencies to `pyproject.toml`, synchronize the policy
environment:

```bash
cd multitask_dit_policy
uv sync
```

## Start the policy server

```bash
cd multitask_dit_policy

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run \
  python -m multitask_dit_policy.serving.policy_server \
  --checkpoint outputs/calvin_d_10_dino/checkpoint_70000 \
  --host 127.0.0.1 \
  --port 10093
```

The server loads `dataset_stats.json`, keeps the observation and action queues,
and returns exactly one unnormalized action for every `step` request.

## Smoke-test one episode

Run this from the RoboNexus root in another shell:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/evaluate.py \
  --task lift_blue_block_drawer \
  --episodes-per-task 1 \
  --replan-interval 4 \
  --fail-fast \
  --overwrite \
  --output-dir outputs/calvin_d10_eval/checkpoint_70000_smoke
```

## Evaluate all ten tasks

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/evaluate.py \
  --episodes-per-task 10 \
  --replan-interval 4 \
  --save-video \
  --output-dir outputs/calvin_d10_eval/checkpoint_70000_r4
```

Use separate output directories for `--replan-interval 1`, `4`, and `16`.
Results are written incrementally to `episodes.jsonl`; aggregate metrics are
updated in `summary.json` after every episode.
