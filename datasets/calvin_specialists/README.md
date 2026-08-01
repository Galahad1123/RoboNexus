# CALVIN 专项数据集

本目录基于 CALVIN 官方 task-D 语言标注构造两套专项数据集，整个过程不会修改
`calvin_d_10` 或源 NPZ 文件：

- `calvin_push_rotate_12`：6 类方块推动任务和 6 类方块旋转任务；
- `calvin_stack_place_3`：堆叠、放入抽屉和放入滑动柜三类任务。

## 审计约束

`scripts/build_audited_manifests.py` 会对相同任务的标注片段进行聚类：当两个片段的
交集至少覆盖较短片段的 50% 时，将其视为同一重叠簇。在每个重叠簇内，脚本只保留
恢复官方起止状态后能够通过对应 CALVIN task oracle 的候选片段，并从中选择最短的
完整片段。

语言标注区间可能遗漏开始阶段的接近动作或最后的释放动作。因此，审计脚本允许将
任一边界保守地扩展最多 15 帧（0.5 秒）。所有边界调整都会记录在 manifest 和完整
审计报告中。调整后仍无法通过 oracle 的候选片段会被拒绝，审计过程不会放宽 oracle
本身的判定阈值。

对于堆叠任务，初始状态必须恰好抓持一个方块。演示中的 destination 定义为成功结束
状态下与 source 接触的唯一可移动方块，并且该接触关系必须在最后 5 帧中至少出现
3 帧。

重新生成全部 manifest 和审计报告：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_audited_manifests.py
```

生成结果分别保存在两套数据集各自的 `manifests/` 和 `reports/` 目录中。

## Mask 语义

- Push+Rotate：仅为指令指定的方块添加此前已验证有效的 2 px 黑色边框。
- Stack+Place：只标记 destination，绝不标记夹爪当前抓持的 source。destination 使用
  2 px 青色边框和 1 px 白色外描边，以减少 CALVIN 场景中黑色按钮和阴影造成的干扰。
- Stack 的训练目标来自演示结束时实际发生接触的目标方块。这属于带特权信息的模仿
  学习监督；部署时需要由后续 agent 侧的目标选择器从 `feasible_destinations` 中选择
  destination。

分别为两套数据集的 training 和 validation split 生成完整 mask cache：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_push_rotate_12 --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_push_rotate_12 --split validation

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_place_3 --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_place_3 --split validation
```

EGL cache 构造器会逐帧恢复仿真状态，因此完整生成可能需要较长时间。小规模测试时可以
使用 `--max-episodes`、`--max-episodes-per-task`，以及可重复传入的 `--task` 参数。
如果目标 cache 目录已经存在，必须显式添加 `--overwrite` 才能覆盖。

## 转换为 LeRobotDataset

使用 MultiTask-DiT 环境运行转换。所有输出路径均需显式指定，并且不会覆盖或影响
`calvin_d_10`：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_push_rotate_12 --split training --storage image \
  --mask-cache datasets/calvin_push_rotate_12/mask_cache/training \
  --output datasets/calvin_push_rotate_12/data/training_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_push_rotate_12 --split validation --storage image \
  --mask-cache datasets/calvin_push_rotate_12/mask_cache/validation \
  --output datasets/calvin_push_rotate_12/data/validation_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_place_3 --split training --storage image \
  --mask-cache datasets/calvin_stack_place_3/mask_cache/training \
  --output datasets/calvin_stack_place_3/data/training_destination_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_place_3 --split validation --storage image \
  --mask-cache datasets/calvin_stack_place_3/mask_cache/validation \
  --output datasets/calvin_stack_place_3/data/validation_destination_outline_image
```

如需生成便于人工检查的视频预览数据集，可将 `--storage image` 改为
`--storage video`，同时指定一个不同的输出目录。
