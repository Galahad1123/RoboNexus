# CALVIN-D-10

CALVIN task-D 中 10 类代表性原子任务的 LeRobotDataset 实验工作区。

转换器中的相对 `--output` 路径始终相对于 RoboNexus 仓库根目录解析，
与执行命令时所在的工作目录无关；转换完成后也会打印最终绝对输出路径。

## 数据来源

- 原始数据：`calvin/dataset/task_D_D/`
- 训练和验证划分：沿用官方 `training` / `validation`
- 标签来源：`lang_annotations/auto_lang_ann.npy`
- 图像：`rgb_static` 保持 200×200；原始 84×84 的 `rgb_gripper` 双线性上采样到
  200×200，以满足 MultiTask DiT 在模型预处理前堆叠相机张量的要求
- 状态：原始 15 维 `robot_obs`
- 动作：原始 7 维 `rel_actions`

原始 task-D 不复制到本目录。转换产物应放入后续创建的 `data/`，并由 `.gitignore`
排除大型数据文件。

## 目录

```text
calvin_d_10/
├── configs/
│   ├── tasks.json
│   └── evaluation.yaml
├── manifests/
│   └── segments.json
├── reports/
│   ├── initial_state_audit.json
│   └── task_D_D_task_stats.json
├── scripts/
│   ├── build_manifest.py
│   ├── build_gt_mask_cache.py
│   ├── convert_to_lerobot.py
│   └── evaluate.py
└── README.md
```

后续稳定、可重复执行的转换和校验脚本放入 `scripts/`。临时探查脚本与生成的预览
文件使用 `/tmp`，不放进仓库。

`scripts/evaluate.py` 是本数据集的专用闭环评测入口；WebSocket client、CALVIN
环境适配、rollout engine 和结果记录等通用实现位于
`calvin/evaluation/remote_policy/`。

## 已确定原则

1. 一个筛选后的 CALVIN 语言片段转换成一个 LeRobot episode。
2. 使用规范化指令，同时保留原始 `calvin_task` 和自然语言标注供追踪。
3. 首帧状态用于闭环评测的 `reset(robot_obs, scene_obs)`。
4. 源区间为闭区间 `[start, end]`；最后一帧保留 observation，但
   `action_is_valid=false`，避免跨片段动作损失。
5. train/validation 沿用官方 split，禁止按帧随机拆分。
6. 正式闭环评测使用 EGL；Direct/TinyRenderer 与录制图像存在明显视觉域差异。

## 片段过滤规则

所有需要重新接近物体或机构的任务都要求首帧未持有物体，避免语言窗口包含上一任务
的持物尾段。两个放置类任务要求首帧恰好持有一个物体。

其他前置条件：

- `lift_blue_block_drawer`、`close_drawer`、`place_in_drawer`：抽屉已打开；
- `open_drawer`：抽屉已关闭；
- `move_slider_left`：slider 位于右侧；
- `turn_on_led`：LED 关闭。

过滤不是根据数组近似猜测接触关系；`build_manifest.py` 会在 CALVIN/Bullet 中恢复每个
候选首帧，并从机器人—物体 contact 信息判断是否持物。

当前 manifest 统计：

| Split | 候选 episodes | 保留 episodes | 保留帧数 |
|---|---:|---:|---:|
| training | 1,507 | 1,324 | 78,910 |
| validation | 297 | 257 | 15,027 |

重新生成 manifest：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/build_manifest.py
```

## LeRobotDataset feature schema

两种存储模式只有图像 feature 的 dtype 不同，episode 集合、顺序和其他字段一致。

| Feature | Shape | Dtype | 用途 |
|---|---:|---|---|
| `observation.images.static` | 200×200×3 | image/video | 静态相机 |
| `observation.images.gripper` | 200×200×3 | image/video | 84×84 原图双线性上采样 |
| `observation.state` | 15 | float32 | CALVIN `robot_obs` |
| `action` | 7 | float32 | CALVIN `rel_actions` |
| `action_is_valid` | 1 | bool | 片段最后一帧为 false |
| `calvin.scene_obs` | 24 | float32 | 闭环初始状态恢复 |
| `calvin.task_id` | 1 | int64 | 固定的 10 类 ID |
| `calvin.source_frame_index` | 1 | int64 | 回溯原始 NPZ |
| `task` | string | LeRobot 内建 | 规范化指令 |

转换器要求 LeRobot 0.6，使用 `multitask_dit_policy/.venv`。

正式训练用无视频/逐帧图像格式：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_d_10/scripts/convert_to_lerobot.py \
  --split training --storage image
```

用于可视化检查的 MP4 格式：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_d_10/scripts/convert_to_lerobot.py \
  --split validation --storage video
```

默认输出到 `datasets/calvin_d_10/data/`，该目录不会进入 Git。可通过 `--output`
指定其他位置，使用 `--max-episodes`、`--max-episodes-per-task` 和可重复的
`--task TASK` 做小规模转换。
已有输出不会被静默覆盖，必须显式传入 `--overwrite`。

## 初始状态审计结论

10 类任务各抽取 20 个 validation 片段，200/200 状态恢复和保持测试通过。重点结论：

- `place_in_drawer`：20/20 首帧恰好持有一个方块，适合直接作为原子放置任务。
- `stack_block`：15/20 首帧持物，5/20 尚未持物。第一版只保留首帧持物片段。
- `lift_blue_block_drawer`：16/20 首帧未持物，4/20 持有其他颜色方块。第一版只保留
  首帧未持物片段。
- EGL 恢复图像与录制 static RGB 的抽样 MAE 中位数为 1.41/255。

完整逐样本结果见 `reports/initial_state_audit.json`。

## 下一步

1. 生成完整 image 训练集与 video 验证集。
2. 实现基于相同初始状态与 CALVIN task oracle 的闭环评测。
3. 建立 multitask DiT 首版训练配置。

## multitask DiT 短片段配置

保留片段长度为 34–65 帧，中位数 65。策略默认 `horizon=100` 会使短 episode
无法采样。首版建议：

```text
n_obs_steps = 2
horizon = 32
n_action_steps = 16
do_mask_loss_for_padding = true
```

`action_is_valid` 用于数据审计；DiT 时序 padding 的实际 loss mask 由 LeRobot
产生的 `action_is_pad` 和 `do_mask_loss_for_padding=true` 负责。

## GT mask overlay 实验

mask 是可选的图像增强，task label 和 clean 数据均不改动。实现分为两个独立阶段：

1. CALVIN 环境按恢复状态渲染 object/link segmentation，并把 overlay 图像写入独立
   cache；
2. LeRobot 转换器仅在显式传入 `--mask-cache` 时读取该 cache，输出路径也必须使用
   新目录。

角色由 `configs/evaluation.yaml` 声明。`source` 默认红色、
`destination` 默认蓝色、单目标 `target` 默认黄色；`place_in_drawer` 和
`stack_block` 的 source 会在首帧根据夹爪接触动态确定。GT 扰动支持平移、膨胀/腐蚀
和像素 dropout，且由 seed 确定性复现。可选的 `--outline-px` 会保留角色色半透明
填充，并在 mask 边界绘制不透明黑色描边；默认值 0 保持原先的纯填充行为。

先做每类一个 episode 的验证 cache：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/build_gt_mask_cache.py \
  --split training --max-episodes-per-task 1 \
  --output datasets/calvin_d_10/mask_cache/training_preview
```

正式 cache 可加入扰动，例如：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/build_gt_mask_cache.py \
  --split training --alpha 0.45 \
  --outline-px 3 \
  --max-shift-px 3 --morphology-px 2 --dropout-probability 0.03 \
  --output datasets/calvin_d_10/mask_cache/training_gt_aug
```

随后在 DiT 环境生成全新的 LeRobotDataset：

```bash
multitask_dit_policy/.venv/bin/python \
  datasets/calvin_d_10/scripts/convert_to_lerobot.py \
  --split training --storage image \
  --mask-cache datasets/calvin_d_10/mask_cache/training_gt_aug \
  --output datasets/calvin_d_10/data/calvin_d_10_training_gt_mask_image
```

不传 `--mask-cache` 时转换器行为与原先完全相同。

对应 checkpoint 的闭环评测需要显式开启相同 overlay；默认仍是 clean：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/evaluate.py \
  --mask-provider gt --mask-alpha 0.45 --mask-outline-px 3 \
  --output-dir outputs/calvin_d10_eval/checkpoint_gt_mask
```

## Oracle 原地 retry

`--max-attempts 1`（默认）仍使用原始 clean rollout。大于 1 时才启用
`agent/calvin_oracle_retry.py`：一次尝试超时后不恢复初态，而是从当前仿真状态开启
新的 policy session；每次 session 使用独立且可复现的 seed。CALVIN oracle 只负责
判定任务是否已经成功，第一版不含学习式成功检测。

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_d_10/scripts/evaluate.py \
  --max-attempts 3 --replan-interval 16 --save-video \
  --output-dir outputs/calvin_d10_eval/checkpoint_80000_retry3
```

结果 JSONL 中会额外记录 `attempt_count` 和逐次 `attempts`。该 agent 也可与
`--mask-provider gt` 组合，但建议先分别做 clean retry 和 GT-mask 单变量实验。
