# CALVIN 专项数据集

本目录基于 CALVIN 官方 task-D 语言标注构造多套专项数据集，整个过程不会修改
`calvin_d_10` 或源 NPZ 文件。

当前推荐使用的第三版数据集：

- `calvin_rotate_2_release`：保留 6 类官方 oracle/评测分组，但训练语言合并为左旋和
  右旋两个任务；
- `calvin_stack_pick_place_persistent`：桌面 Pick、完整 Pick-and-Stack、Held-start
  Stack、Place-in-Drawer 和 Place-in-Slider，source 全程持续标记。

先前实验数据 `calvin_push_rotate_12`、`calvin_stack_place_3`、
`calvin_rotate_6_release` 和 `calvin_stack_pick_place` 会继续保留，便于复现已有
checkpoint，不会被新脚本覆盖。

## 0. CALVIN-D 34类四路模型方案

34类官方标签现已形成互斥的部署路由：

| 模型 | 官方任务数 | 数据集 |
|---|---:|---|
| Push | 6 | `calvin_push_6` |
| Rotate | 6（语言合并为左右2类） | `calvin_rotate_2_release` |
| Stack | 1 | `calvin_stack_pick_place_persistent` |
| General | 21 | `calvin_general_21` |

完整标签、原始标注数和唯一归属保存在
`datasets/calvin_d34/reports/task_inventory.json`。Stack 训练允许重复使用少量 Pick/Place
辅助任务，但这些任务的部署路由仍属于 General，不会造成34类统计重复。

重新生成标签清单以及 Push/General 审计 manifest：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_d34_manifests.py --inventory-only

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_d34_manifests.py \
  --dataset calvin_push_6

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_d34_manifests.py \
  --dataset calvin_general_21
```

审计结果：

| 数据集 | Training episodes | Validation episodes |
|---|---:|---:|
| `calvin_push_6` | 220 | 39 |
| `calvin_general_21` | 2,203 | 433 |

Push 数据集保留六个颜色+方向指令、oracle 和 task id，沿用已验证成功的原始任务语义。
General 不建议开启任务均衡采样：21类经去重后每类约80～131条，差异有限，完全
均衡反而会放大较小类别；先使用 `--task_sampling=proportional`。

生成新数据集的 mask cache：

```bash
for split in training validation; do
  conda run -n robonexus-calvin python \
    datasets/calvin_specialists/scripts/build_mask_cache.py \
    --dataset calvin_push_6 --split "$split"

  conda run -n robonexus-calvin python \
    datasets/calvin_specialists/scripts/build_mask_cache.py \
    --dataset calvin_general_21 --split "$split"
done
```

转换为 image-mode LeRobotDataset：

```bash
for split in training validation; do
  multitask_dit_policy/.venv/bin/python \
    datasets/calvin_specialists/scripts/convert_to_lerobot.py \
    --dataset calvin_push_6 --split "$split" --storage image \
    --mask-cache "datasets/calvin_push_6/mask_cache/$split" \
    --output "datasets/calvin_push_6/data/${split}_outline_image"

  multitask_dit_policy/.venv/bin/python \
    datasets/calvin_specialists/scripts/convert_to_lerobot.py \
    --dataset calvin_general_21 --split "$split" --storage image \
    --mask-cache "datasets/calvin_general_21/mask_cache/$split" \
    --output "datasets/calvin_general_21/data/${split}_role_outline_image"
done
```

### Full Stack 扩展评测

官方 validation 只有4条 Full Pick-and-Stack。独立扩展构造器会在 validation 的连续专家
轨迹内部向前回溯，要求 source 有物理支撑、夹爪未持物，且到原终点仍满足 Stack
oracle。目前可得到21个独立初始状态（4 official + 17 backtracked）。所有条目均带有
`training_eligible: false`，不可用于训练。

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_full_stack_extended_eval.py \
  --lookback-frames 90

conda run -n robonexus-calvin python \
  datasets/calvin_full_stack_extended_eval/scripts/evaluate.py \
  --episodes-per-task 100 --replan-interval 16 --mask-provider gt \
  --save-video --overwrite \
  --output-dir outputs/calvin_full_stack_extended_eval/checkpoint_80000_r16
```

这21个状态可以分别使用多个 policy seed 重复评测，但汇总时必须同时报告“按独立初始
状态”和“按 rollout”的成功率，不能把不同 seed 当作新的场景。

## 1. 通用去重规则

相同任务的标注片段会先按时间重叠聚类：当两个片段的交集至少覆盖较短片段的 50%
时，将其视为同一个重叠簇。每个簇内只选择通过当前 CALVIN task oracle 的最短完整
候选，避免同一次专家操作因为语言标注重叠而被重复训练。

全部候选、拒绝原因、重叠簇和最终选择均保存在各数据集的 `reports/` 中；最终转换只
读取 `manifests/segments.json`。

## 2. 第三版审计标准

### Rotate-2 Release

Rotate 候选必须满足：

1. 起点未抓持物体；语言区间若遗漏接近阶段，最多向前修复 15 帧；
2. 对应 Rotate oracle 成功；
3. 夹爪连续至少 3 帧发出 open action，gripper width 已打开，目标方块有物理支撑；
4. oracle 与 release 最多只能出现在原标注末尾后 15 帧内；
5. release 后至少保留 15 帧、最多保留 30 帧真实稳定后缀；
6. 稳定后缀中方块必须保持支撑，且相邻帧平移和旋转变化不超过配置阈值。

当前审计结果：

| Split | 候选 | 去重簇 | 保留 episodes | 帧数 |
|---|---:|---:|---:|---:|
| training | 901 | 366 | 275 | 26,375 |
| validation | 178 | 64 | 45 | 4,308 |

六个官方 Rotate 类别在 training 和 validation 中均有覆盖，但写入 LeRobotDataset 的
规范化语言只有两个：

```text
rotate the black-outlined block counterclockwise
rotate the black-outlined block clockwise
```

因此 outline 指定实例、语言指定方向；CALVIN oracle 与逐颜色评测仍保持六类。

### Stack+Pick+Place

Stack 不再要求语言标注起点已经抓住方块。审计器会从专家轨迹末尾唯一确定：

- `source`：最终位于上层的方块；
- `destination`：最终与 source 稳定接触、位于下层的方块。

随后在轨迹内查找 source 连续至少 3 帧被稳定抓持的第一个状态。每条合格专家轨迹
至少生成一个 `held_start_stack`；若原始起点没有抓持物体，还会额外生成一个
`full_pick_and_stack`。两者使用同一个 Stack 目标和 oracle。

转换后的 Parquet 还会保留 `calvin.episode_variant_id`，供后续按完整轨迹、Held-start、
Pick 和 Place 分层采样；当前策略模型会安全忽略这个额外审计字段。

混合数据还包含三色桌面 Pick、Place-in-Drawer 和 Place-in-Slider。Pick 必须以目标
方块被抓住并抬起结束；Stack 必须以夹爪打开、source 与唯一 destination 稳定接触
结束。

当前审计结果：

| Split | 保留 episodes | 帧数 |
|---|---:|---:|
| training | 452 | 22,367 |
| validation | 135 | 7,058 |

Training 的 episode variant：

| Variant | Episodes |
|---|---:|
| `table_pick` | 100（蓝/粉/红为 34/33/33） |
| `full_pick_and_stack` | 42 |
| `held_start_stack` | 110 |
| `place_in_drawer` | 100 |
| `place_in_slider` | 100 |

Stack 覆盖全部 6 种有序 source→destination 颜色组合。

以上上限只作用于 training。脚本从完整 oracle-consistent 候选中沿时间轴确定性均匀
取样，避免只保留数据集开头；validation 不截断，以保留全部独立评测初始状态。

## 3. Mask 语义

### Rotate

目标方块使用 2 px 黑色边框，无填充、无 halo。

### Stack+Pick+Place

- source：2 px 黑色边框；
- destination：2 px 青色边框 + 1 px 白色外 halo；
- source 在 Pick、Full Stack 和 Held-start Stack 中全程持续显示；
- destination 在 Stack 和 Place 过程中始终显示；
- Table Pick 只有 source，没有 destination。

持续标记与部署时 SAM/Cutie 对已指定目标的连续跟踪语义一致，不再根据仿真接触状态
动态撤销 source 提示。

Stack 的 source/destination 来自专家轨迹，这是训练阶段的特权监督。真实部署时仍需由
agent 侧目标选择器提供对应角色。

## 4. 重新生成审计 manifest

无 `--dataset` 参数时，默认只生成当前推荐的第三版两套数据：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py
```

也可以单独生成：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py \
  --dataset calvin_rotate_2_release

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py \
  --dataset calvin_stack_pick_place_persistent
```

第一版 manifest 仍由 `scripts/build_audited_manifests.py` 构造。

## 5. 生成 Mask Cache

必须先生成 mask cache，再转换 LeRobotDataset。

```bash
# Rotate training / validation
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_rotate_2_release --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_rotate_2_release --split validation

# Stack+Pick+Place training / validation
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place_persistent --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place_persistent --split validation
```

EGL cache 构造器会逐帧恢复仿真状态，完整生成需要较长时间。小规模检查可使用
`--max-episodes`、`--max-episodes-per-task` 和可重复传入的 `--task`。已有目标目录时
必须显式添加 `--overwrite`。

例如，每类生成 1 个视频预览所需的 cache：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place_persistent --split validation \
  --episode-variant full_pick_and_stack \
  --max-episodes-per-task 1 \
  --output datasets/calvin_stack_pick_place_persistent/mask_cache/validation_preview
```

## 6. 转换为 LeRobotDataset

使用 MultiTask-DiT 环境运行转换。Image 模式会把图像列写入 Parquet，不会生成逐帧
图片文件。

```bash
# Rotate
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_rotate_2_release --split training --storage image \
  --mask-cache datasets/calvin_rotate_2_release/mask_cache/training \
  --output datasets/calvin_rotate_2_release/data/training_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_rotate_2_release --split validation --storage image \
  --mask-cache datasets/calvin_rotate_2_release/mask_cache/validation \
  --output datasets/calvin_rotate_2_release/data/validation_outline_image

# Stack+Pick+Place
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place_persistent --split training --storage image \
  --mask-cache datasets/calvin_stack_pick_place_persistent/mask_cache/training \
  --output datasets/calvin_stack_pick_place_persistent/data/training_role_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place_persistent --split validation --storage image \
  --mask-cache datasets/calvin_stack_pick_place_persistent/mask_cache/validation \
  --output datasets/calvin_stack_pick_place_persistent/data/validation_role_outline_image
```

视频预览使用 `--storage video` 并指定不同输出目录，例如：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place_persistent --split validation --storage video \
  --episode-variant full_pick_and_stack \
  --max-episodes-per-task 1 \
  --mask-cache datasets/calvin_stack_pick_place_persistent/mask_cache/validation_preview \
  --output datasets/calvin_stack_pick_place_persistent/data/validation_role_outline_preview_video
```
