# CALVIN Rotate-2 Release

保留六类官方 CALVIN Rotate oracle 和逐颜色评测分组，但将训练语言合并为左旋、右旋
两个方向任务。目标方块全程使用 2 px 黑色 outline。

| Split | Episodes | Frames |
|---|---:|---:|
| training | 275 | 26,375 |
| validation | 45 | 4,308 |

完整审计标准和生成命令见
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md)。

## 闭环评测

评测入口为 `scripts/evaluate.py`，默认读取 `configs/evaluation.yaml`。训练指令虽合并为
顺时针/逆时针两类，评测仍按 CALVIN 六类颜色与方向 oracle 分组。成功要求 oracle
成立后夹爪宽度至少为 0.04 m，并连续保持 3 步，以排除没有完成释放的结果。

评测时必须使用 `--mask-provider gt`，以生成与训练阶段一致的持续 2 px 黑色目标边框。
`--episodes-per-task 20` 已足以覆盖 validation 中全部 45 条轨迹。

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_rotate_2_release/scripts/evaluate.py \
  --episodes-per-task 20 --replan-interval 16 --mask-provider gt \
  --save-video --overwrite \
  --output-dir outputs/calvin_rotate_2_release_eval/checkpoint_80000_r16
```

若 server 在另一台主机，再添加 `--host <SERVER_IP> --port 10093`。
