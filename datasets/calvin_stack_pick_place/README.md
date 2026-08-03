# CALVIN Stack+Pick+Place

包含桌面 Pick、Full Pick-and-Stack、Held-start Stack、Place-in-Drawer 和
Place-in-Slider 的混合专项数据集。

| Split | Episodes | Frames |
|---|---:|---:|
| training | 452 | 22,367 |
| validation | 135 | 7,058 |

完整审计标准、动态 source/destination outline 语义和构造命令见
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md)。

## 闭环评测

启动对应 checkpoint 的策略服务后运行：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_stack_pick_place/scripts/evaluate.py \
  --episodes-per-task 20 --replan-interval 16 \
  --mask-provider gt --save-video \
  --output-dir outputs/calvin_stack_pick_place_eval/checkpoint_STEP_r16
```

`stack_block` 会使用 manifest 中全部 validation 初始状态，其中包括
`full_pick_and_stack` 和 `held_start_stack`。source 第一次被抓住后黑边会永久隐藏，
destination 的青色边框和白色 halo 会持续保留。
`place_in_drawer`、`place_in_slider` 和 `stack_block` 同样要求 oracle 成立后夹爪连续
3 步真实打开；三个 Table Pick 任务则以抓持并抬起目标为成功，不要求释放。

如需分别评测两种 Stack 起点：

```bash
# Full Pick-and-Stack（validation 共 4 个）
conda run -n robonexus-calvin python \
  datasets/calvin_stack_pick_place/scripts/evaluate.py \
  --task stack_block --episode-variant full_pick_and_stack \
  --episodes-per-task 10 --replan-interval 16 \
  --mask-provider gt --save-video \
  --output-dir outputs/calvin_stack_pick_place_eval/full_stack_r16

# Held-start Stack（validation 共 17 个）
conda run -n robonexus-calvin python \
  datasets/calvin_stack_pick_place/scripts/evaluate.py \
  --task stack_block --episode-variant held_start_stack \
  --episodes-per-task 20 --replan-interval 16 \
  --mask-provider gt --save-video \
  --output-dir outputs/calvin_stack_pick_place_eval/held_stack_r16
```
