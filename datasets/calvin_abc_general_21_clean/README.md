# CALVIN ABC→D General-21 clean-RGB evaluation

This suite evaluates the ABC-trained clean-RGB General-21 policy on the
official D validation environment. The audited manifest currently contains 47
independent initial states across all 21 tasks. The default
`--episodes-per-task 10` evaluates every available state without replacement.

Start the policy server from the repository root:

```bash
uv run --project multitask_dit_policy python \
  -m multitask_dit_policy.serving.policy_server \
  --checkpoint /home/hello/lizk/vlas/RoboNexus/outputs/calvin_abc_general_21/final_model \
  --vision-pretrained-path /home/hello/.cache/huggingface/hub/vit_base_patch16_dinov3.lvd1689m \
  --text-pretrained-path /home/hello/.cache/huggingface/hub/clip-vit-base-patch16 \
  --host 127.0.0.1 \
  --port 10093
```

The path overrides replace stale `/root/vlas/...` cache paths stored by the
training machine. They do not modify the checkpoint or replace learned
weights.

Run a one-episode smoke test in another terminal:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_abc_general_21_clean/scripts/evaluate_d.py \
  --task open_drawer \
  --episodes-per-task 1 \
  --replan-interval 16 \
  --fail-fast \
  --save-video \
  --overwrite \
  --output-dir outputs/calvin_abc_general_21_d_eval/smoke
```

Evaluate all 47 audited D initial states:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_abc_general_21_clean/scripts/evaluate_d.py \
  --episodes-per-task 10 \
  --replan-interval 16 \
  --save-video \
  --output-dir outputs/calvin_abc_general_21_d_eval/final_model_r16
```

Do not pass `--mask-provider gt`: the policy was trained with clean RGB.
Results are appended to `episodes.jsonl`; `summary.json` reports overall,
macro, and per-task success rates, and `video_index.csv` maps saved videos to
their tasks. Use a distinct output directory for every checkpoint or replan
interval.
