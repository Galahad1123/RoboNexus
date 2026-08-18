# CALVIN ABC→D Manipulation-13 clean-RGB evaluation

This suite evaluates the ABC-trained clean-RGB policy on the official D
validation environment. It covers Push-6, Rotate-6, and Stack-1. The audited D
manifest currently contains 44 independent initial states; the default
`--episodes-per-task 10` therefore evaluates every available state without
replacement.

Start the policy server from the repository root:

```bash
uv run --project multitask_dit_policy python \
  -m multitask_dit_policy.serving.policy_server \
  --checkpoint /home/hello/lizk/vlas/RoboNexus/outputs/calvin_abc_manip_13/final_model \
  --vision-pretrained-path /home/hello/.cache/huggingface/hub/vit_base_patch16_dinov3.lvd1689m \
  --text-pretrained-path /home/hello/.cache/huggingface/hub/clip-vit-base-patch16 \
  --host 127.0.0.1 \
  --port 10093
```

The two path overrides are needed because this checkpoint records the training
machine's `/root/vlas/...` cache paths. They only redirect backbone
construction and do not modify the checkpoint or replace its learned weights.

In another terminal, run a one-episode smoke test:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_abc_manipulation_13_clean/scripts/evaluate_d.py \
  --task push_blue_block_left \
  --episodes-per-task 1 \
  --replan-interval 16 \
  --fail-fast \
  --save-video \
  --overwrite \
  --output-dir outputs/calvin_abc_manip_13_d_eval/smoke
```

Evaluate all 44 audited D initial states:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_abc_manipulation_13_clean/scripts/evaluate_d.py \
  --episodes-per-task 10 \
  --replan-interval 16 \
  --save-video \
  --output-dir outputs/calvin_abc_manip_13_d_eval/final_model_r16
```

This is a clean-RGB evaluation, so do not pass `--mask-provider gt`. Results
are written incrementally to `episodes.jsonl`; `summary.json` contains overall,
macro, and per-task success rates. Use a new output directory for each
checkpoint or replan interval.
