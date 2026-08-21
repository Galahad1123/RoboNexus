# CALVIN ABC Lift-9 clean-RGB dataset

This dataset is an audited task subset of
`calvin_abc_general_21_clean`. It inherits the parent's physical CALVIN Oracle
audit and overlap-cluster deduplication, then revalidates task coverage,
retained Oracle flags, unique segment IDs, and source boundary frames.

Build or refresh the manifest:

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_abc_lift_device_manifests.py \
  --dataset calvin_abc_lift_9_clean
```

Convert all three clean-RGB LeRobot splits:

```bash
for split in training abc_validation d_validation; do
  multitask_dit_policy/.venv/bin/python \
    datasets/calvin_specialists/scripts/convert_to_lerobot.py \
    --dataset calvin_abc_lift_9_clean \
    --split "$split" \
    --storage image \
    --output "datasets/calvin_abc_lift_9_clean/data/${split}_image"
done
```

Do not pass `--mask-cache`. Add `--overwrite` only when intentionally replacing
an existing conversion.
