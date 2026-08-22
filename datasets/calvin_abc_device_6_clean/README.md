# CALVIN ABC Device-6 clean-RGB dataset

This dataset contains Slider-2, LED-2, and Lightbulb-2 as an audited subset of
`calvin_abc_general_21_clean`. It inherits the parent's physical Oracle audit
and overlap-cluster deduplication and does not apply image masks.

Build or refresh the manifest:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_abc_lift_device_manifests.py \
  --dataset calvin_abc_device_6_clean
```

Convert all three clean-RGB LeRobot splits:

```bash
for split in training abc_validation d_validation; do
  multitask_dit_policy/.venv/bin/python \
    datasets/calvin_specialists/scripts/convert_to_lerobot.py \
    --dataset calvin_abc_device_6_clean \
    --split "$split" \
    --storage image \
    --output "datasets/calvin_abc_device_6_clean/data/${split}_image"
done
```

Do not pass `--mask-cache`. Add `--overwrite` only when intentionally replacing
an existing conversion.

Closed-loop evaluation entry points:

```bash
# task_ABC_D/validation (D domain)
conda run -n robonexus-calvin python \
  datasets/calvin_abc_device_6_clean/scripts/evaluate_d.py \
  --episodes-per-task 100 --replan-interval 16

# Independent task_D_D/validation benchmark
conda run -n robonexus-calvin python \
  datasets/calvin_abc_device_6_clean/scripts/evaluate_task_dd.py \
  --episodes-per-task 100 --replan-interval 16
```
