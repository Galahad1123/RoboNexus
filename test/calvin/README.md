# CALVIN 独立验收

这里的验收只依赖 CALVIN 环境和官方 debug 数据，不加载策略模型或训练
dataloader。模拟器检查在独立子进程运行，因此 EGL 插件崩溃也会被记录为失败，
不会带走主验收进程。

默认数据位置：

```text
calvin/dataset/calvin_debug_dataset/
```

## 第一阶段

在仓库根目录执行：

```bash
conda run -n robonexus-calvin python test/calvin/acceptance.py \
  --report test/calvin/report.json
```

这会检查数据结构，并在无头 EGL 模式完成：

- `reset()` 和连续 30 次 `step()`；
- static/gripper RGB-D 的 shape、dtype、NaN/Inf 和非黑帧；
- 从首个 validation 数据帧的 `robot_obs`、`scene_obs` 恢复状态；
- 在恢复后的环境执行该帧记录的 `rel_actions`。

debug 采集配置还包含可选的 TACTO 触觉相机。它的独立渲染进程在无
`DISPLAY` 的 DIRECT 模式会阻塞，因此本阶段显式只实例化 CALVIN 标准的
static/gripper RGB-D 相机；TACTO 留作单独的硬件相关验收项。

附加 CPU/DIRECT 对照测试：

```bash
conda run -n robonexus-calvin python test/calvin/acceptance.py --direct
```

GUI 必须从具有 X11 `DISPLAY` 的桌面终端执行：

```bash
conda run -n robonexus-calvin python test/calvin/acceptance.py --gui --steps 30
```

GUI 窗口会在测试运行期间出现，完成后自动关闭。当前终端没有 `DISPLAY` 时，该项会
明确标记为 skipped，不能据此宣称 GUI 已通过。

常用参数：

- `--dataset PATH`：指定其他 CALVIN 数据集根目录；
- `--split training|validation`：选择 split，默认 `validation`；
- `--steps N`：连续 step 次数；
- `--timeout SECONDS`：每种模拟器模式的子进程超时；
- `--report PATH`：保存 JSON 报告。
