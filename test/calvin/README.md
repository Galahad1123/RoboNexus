# CALVIN 独立验收

这里的验收只依赖 CALVIN 环境和官方 debug 数据，不加载策略模型或训练
dataloader。模拟器检查在独立子进程运行，因此 EGL 插件崩溃也会被记录为失败，
不会带走主验收进程。

## 隔离的 Qwen 子任务终止判定实验

`subtask_verifier.py` 是一个不依赖 Runtime、不会修改现有 Agent 行为的实验入口，当前实现：

- 使用当前时刻的 static（第三视角）和 wrist（腕部）图像；
- 逐条判断显式成功谓词，输出 `success`、`in_progress`、`failure` 或 `uncertain`；
- 保留任务初始双相机状态，帮助模型识别当前状态相对任务起点的变化；
- 只要存在近期帧，就一次性输入最多两组稀疏双相机帧和当前帧进行时序判定；
- 不再先调用单帧判定，避免默认短时序模式产生两次 Qwen 推理延迟；
- 报告单帧判定、时序复核和总耗时。

先运行不加载模型的状态机测试：

```bash
conda run -n robonexus-qwen python -m unittest discover \
  -s test/calvin -p 'test_subtask_verifier.py' -v
```

用本地 Qwen3-VL 对保存的 CALVIN 图像做一次真实判定：

```bash
conda run -n robonexus-qwen python test/calvin/subtask_verifier.py \
  --spec test/calvin/verifier_spec.example.json \
  --static /path/to/current_static.png \
  --wrist /path/to/current_wrist.png \
  --output /tmp/calvin_verifier_result.json
```

需要测试按需短时序复核时，按从旧到新的顺序成对重复传入：

```bash
  --history-static /path/to/old_static.png \
  --history-wrist /path/to/old_wrist.png
```

历史图像只用于消除当前状态的遮挡或歧义，不作为参考轨迹。第一版暂不输入夹爪开度。

### CALVIN-D 与官方 Oracle 对比

以下脚本默认从 ABC→D 数据的 D validation split 选择 8 个单帧可判定技能，每个技能
取中位长度示范并生成起始负例和结束正例。样本标签通过恢复 CALVIN 状态并调用官方
Task Oracle 得到，而不是直接采用示范标签：

```bash
PYTHONPATH=calvin/calvin_env:calvin/calvin_models:. \
  conda run -n robonexus-calvin python \
  test/calvin/prepare_verifier_benchmark.py \
  --output-dir outputs/calvin_d_verifier_benchmark
```

在有 CUDA 的机器上运行 Qwen，并输出 JSON 与嵌入双相机图像的自包含 HTML：

```bash
conda run -n robonexus-qwen python test/calvin/run_verifier_benchmark.py \
  --manifest outputs/calvin_d_verifier_benchmark/manifest.json \
  --output outputs/calvin_d_verifier_benchmark/qwen_results.json \
  --html outputs/calvin_d_verifier_benchmark/qwen_results.html
```

当前默认探索批次为 `rotate_red_block_right` 和 `rotate_blue_block_left`，每项各有
一个负例和正例。Qwen 仅接收 Current Wrist，并通过两瓣夹爪是否接触彩色矩形物体判断
最终是否释放，不验证旋转角度或方向。具体任务分组记录在 `TASK_STATUS.md`。

重新运行准备脚本后，manifest 还会为每个任务保存一张来自**另一条示范**的完成末帧，
避免把待测正例本身作为参考。增加目标参考图的对照实验使用独立结果文件：

```bash
conda run -n robonexus-qwen python test/calvin/run_verifier_benchmark.py \
  --manifest outputs/calvin_d_verifier_benchmark/manifest.json \
  --use-goal-reference \
  --output outputs/calvin_d_verifier_benchmark/qwen_results_with_reference.json \
  --html outputs/calvin_d_verifier_benchmark/qwen_results_with_reference.html
```

原来的 `qwen_results.json/html` 保留为无目标图基线；两个实验使用完全相同的测试样本和
Oracle 标签。

默认数据位置：

```text
calvin/dataset/calvin_debug_dataset/
```

## 第一阶段

在仓库根目录执行：

```bash
conda run -n robonexus-calvin python test/calvin/acceptance.py \
  --report /tmp/calvin_acceptance_report.json
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

## 浏览 task-D 的任务示范

交互式依次播放全部 34 个任务，每类自动选择一个长度接近该类中位数的官方示范：

```bash
conda run -n robonexus-calvin python test/calvin/visualize_tasks.py
```

窗口快捷键：空格暂停，`R` 重播，`N` 下一个，`P` 上一个，`Q`/Esc 退出。
画面左侧是 static 相机，右侧是 gripper 相机。

只看指定任务：

```bash
conda run -n robonexus-calvin python test/calvin/visualize_tasks.py \
  --task lift_red_block_table --task place_in_drawer
```

无 GUI 时也可以批量导出全部任务的 MP4：

```bash
conda run -n robonexus-calvin python test/calvin/visualize_tasks.py \
  --export-dir /tmp/calvin_task_videos
```

可用 `--selection first|median|random` 控制代表片段的选择方式，`--fps` 调整播放
速度，`--scale` 调整显示/导出分辨率。

## 对比 ABC→D 的 34 个任务

`task_ABC_D` 的 training 同时包含 A、B、C，具体帧区间记录在官方
`scene_info.npy`；validation 是 D。以下查看器会按这些边界为每个任务分别选择
A/B/C/D 的一条官方示范，并以四宫格同步播放：

```bash
conda run -n robonexus-calvin python \
  test/calvin/visualize_abcd_tasks.py
```

窗口快捷键与单套件查看器相同：空格暂停，`R` 重播，`N`/右方向键进入下一项，
`P`/左方向键返回上一项，`Q`/Esc 退出。不同环境的示范长度可能不同，查看器按
归一化任务进度同步四个画面。

只比较指定任务：

```bash
conda run -n robonexus-calvin python \
  test/calvin/visualize_abcd_tasks.py \
  --task stack_block --task push_into_drawer
```

无 GUI 时可导出全部 34 个四宫格视频和所选片段的 manifest：

```bash
conda run -n robonexus-calvin python \
  test/calvin/visualize_abcd_tasks.py \
  --export-dir outputs/calvin_abcd_task_comparison
```

`--selection median|first|random` 控制每套环境中代表示范的选择方式；随机选择可配合
`--seed` 复现。`--fps` 和 `--scale` 分别控制播放帧率与输出缩放。
