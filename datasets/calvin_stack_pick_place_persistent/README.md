# CALVIN Stack+Pick+Place Persistent

桌面 Pick、Full Pick-and-Stack、Held-start Stack、Place-in-Drawer 和
Place-in-Slider 混合数据。source 使用全程持续的 2 px 黑色 outline；destination 使用
2 px 青色 outline + 1 px 白色 halo。

| Split | Episodes | Frames |
|---|---:|---:|
| training | 452 | 22,367 |
| validation | 135 | 7,058 |

完整审计标准和生成命令见
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md)。

## 闭环评测

评测入口为 `scripts/evaluate.py`，默认读取 `configs/evaluation.yaml`。GT mask 会从
manifest 恢复：source 始终保留 2 px 黑色边框，destination 使用 2 px 青色边框和
1 px 白色 halo。Stack 结果同时记录 `full_pick_and_stack` 与 `held_start_stack`
episode variant，支持用 `--episode-variant` 单独评测。

评测时必须使用 `--mask-provider gt`。`--episodes-per-task 100` 会覆盖 validation 中
全部 135 条轨迹；Stack、Place 的成功还要求夹爪完成并稳定保持释放。

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_stack_pick_place_persistent/scripts/evaluate.py \
  --episodes-per-task 100 --replan-interval 16 --mask-provider gt \
  --save-video --overwrite \
  --output-dir outputs/calvin_stack_pick_place_persistent_eval/checkpoint_80000_r16
```

只评测完整 Pick-and-Stack 时，可追加
`--task stack_block --episode-variant full_pick_and_stack`；只评测手持起点则改为
`--episode-variant held_start_stack`。
