# CALVIN 专项数据集

本目录基于 CALVIN 官方 task-D 语言标注构造多套专项数据集，整个过程不会修改
`calvin_d_10` 或源 NPZ 文件。

当前推荐使用的第二版数据集：

- `calvin_rotate_6_release`：6 类完成真实释放并带稳定尾段的 Rotate；
- `calvin_stack_pick_place`：桌面 Pick、完整 Pick-and-Stack、Held-start Stack、
  Place-in-Drawer 和 Place-in-Slider。

第一版实验数据 `calvin_push_rotate_12` 与 `calvin_stack_place_3` 会继续保留，便于复现
已有 checkpoint，不会被第二版脚本覆盖。

## 1. 通用去重规则

相同任务的标注片段会先按时间重叠聚类：当两个片段的交集至少覆盖较短片段的 50%
时，将其视为同一个重叠簇。每个簇内只选择通过当前 CALVIN task oracle 的最短完整
候选，避免同一次专家操作因为语言标注重叠而被重复训练。

全部候选、拒绝原因、重叠簇和最终选择均保存在各数据集的 `reports/` 中；最终转换只
读取 `manifests/segments.json`。

## 2. 第二版审计标准

### Rotate-6 Release

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

六个 Rotate 类别在 training 和 validation 中均有覆盖。

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
- source 仅在尚未抓住时显示，抓住后黑色边框动态消失；
- destination 在 Stack 和 Place 过程中始终显示；
- Held-start Stack 从第一帧开始只显示 destination；
- Table Pick 只有 source，没有 destination。

Stack 的 source/destination 来自专家轨迹，这是训练阶段的特权监督。真实部署时仍需由
agent 侧目标选择器提供对应角色。

## 4. 重新生成审计 manifest

只生成第二版两套数据：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py
```

也可以单独生成：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py \
  --dataset calvin_rotate_6_release

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_refined_manifests.py \
  --dataset calvin_stack_pick_place
```

第一版 manifest 仍由 `scripts/build_audited_manifests.py` 构造。

## 5. 生成 Mask Cache

必须先生成 mask cache，再转换 LeRobotDataset。

```bash
# Rotate training / validation
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_rotate_6_release --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_rotate_6_release --split validation

# Stack+Pick+Place training / validation
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place --split training

conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place --split validation
```

EGL cache 构造器会逐帧恢复仿真状态，完整生成需要较长时间。小规模检查可使用
`--max-episodes`、`--max-episodes-per-task` 和可重复传入的 `--task`。已有目标目录时
必须显式添加 `--overwrite`。

例如，每类生成 1 个视频预览所需的 cache：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_specialists/scripts/build_mask_cache.py \
  --dataset calvin_stack_pick_place --split validation \
  --episode-variant full_pick_and_stack \
  --max-episodes-per-task 1 \
  --output datasets/calvin_stack_pick_place/mask_cache/validation_preview
```

## 6. 转换为 LeRobotDataset

使用 MultiTask-DiT 环境运行转换。Image 模式会把图像列写入 Parquet，不会生成逐帧
图片文件。

```bash
# Rotate
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_rotate_6_release --split training --storage image \
  --mask-cache datasets/calvin_rotate_6_release/mask_cache/training \
  --output datasets/calvin_rotate_6_release/data/training_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_rotate_6_release --split validation --storage image \
  --mask-cache datasets/calvin_rotate_6_release/mask_cache/validation \
  --output datasets/calvin_rotate_6_release/data/validation_outline_image

# Stack+Pick+Place
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place --split training --storage image \
  --mask-cache datasets/calvin_stack_pick_place/mask_cache/training \
  --output datasets/calvin_stack_pick_place/data/training_role_outline_image

multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place --split validation --storage image \
  --mask-cache datasets/calvin_stack_pick_place/mask_cache/validation \
  --output datasets/calvin_stack_pick_place/data/validation_role_outline_image
```

视频预览使用 `--storage video` 并指定不同输出目录，例如：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_specialists/scripts/convert_to_lerobot.py \
  --dataset calvin_stack_pick_place --split validation --storage video \
  --episode-variant full_pick_and_stack \
  --max-episodes-per-task 1 \
  --mask-cache datasets/calvin_stack_pick_place/mask_cache/validation_preview \
  --output datasets/calvin_stack_pick_place/data/validation_role_outline_preview_video
```
