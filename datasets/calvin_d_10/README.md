# CALVIN-D-10

CALVIN task-D 中 10 类代表性原子任务的 LeRobotDataset 实验工作区。

## 数据来源

- 原始数据：`calvin/dataset/task_D_D/`
- 训练和验证划分：沿用官方 `training` / `validation`
- 标签来源：`lang_annotations/auto_lang_ann.npy`
- 图像：原始 `rgb_static`（200×200）和 `rgb_gripper`（84×84）
- 状态：原始 15 维 `robot_obs`
- 动作：原始 7 维 `rel_actions`

原始 task-D 不复制到本目录。转换产物应放入后续创建的 `data/`，并由 `.gitignore`
排除大型数据文件。

## 目录

```text
calvin_d_10/
├── configs/
│   └── tasks.json
├── reports/
│   ├── initial_state_audit.json
│   └── task_D_D_task_stats.json
└── README.md
```

后续稳定、可重复执行的转换和校验脚本放入 `scripts/`。临时探查脚本与生成的预览
文件使用 `/tmp`，不放进仓库。

## 已确定原则

1. 一个 CALVIN 语言片段转换成一个候选 LeRobot episode。
2. 使用规范化指令，同时保留原始 `calvin_task` 和自然语言标注供追踪。
3. 首帧状态用于闭环评测的 `reset(robot_obs, scene_obs)`。
4. 源区间为闭区间 `[start, end]`；最后一帧保留 observation，但
   `action_is_valid=false`，避免跨片段动作损失。
5. train/validation 沿用官方 split，禁止按帧随机拆分。
6. 正式闭环评测使用 EGL；Direct/TinyRenderer 与录制图像存在明显视觉域差异。

## 初始状态审计结论

10 类任务各抽取 20 个 validation 片段，200/200 状态恢复和保持测试通过。重点结论：

- `place_in_drawer`：20/20 首帧恰好持有一个方块，适合直接作为原子放置任务。
- `stack_block`：15/20 首帧持物，5/20 尚未持物。第一版只保留首帧持物片段。
- `lift_blue_block_drawer`：16/20 首帧未持物，4/20 持有其他颜色方块。第一版只保留
  首帧未持物片段。
- EGL 恢复图像与录制 static RGB 的抽样 MAE 中位数为 1.41/255。

完整逐样本结果见 `reports/initial_state_audit.json`。

## 下一步

1. 固化 LeRobotDataset feature schema。
2. 实现候选片段筛选和转换。
3. 校验 episode 数量、帧数、动作边界和视频编码。
4. 实现基于相同初始状态与 CALVIN task oracle 的闭环评测。
