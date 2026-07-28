# RoboNexus perception pipeline

第一阶段实现 Qwen3-VL 定位、SAM 首帧分割、Cutie 有状态逐帧跟踪，以及低质量
mask 的自动纠偏。`core` 不导入任何模型包，三个模型可运行在独立 Conda 环境中。

## 协议约定

- 图像为 RGB `uint8 H×W×3`。
- bbox 为原图坐标的半开区间 `[x1, y1, x2, y2)`。
- mask 为与原图同尺寸的二值 `uint8`。
- HTTP 传输使用 base64 PNG；请求和响应都带 `frame_id`。
- 一个 CALVIN episode 对应一个 Cutie session；对象 ID 在 session 内固定。
- `source=1`，后续对象按 Qwen 返回顺序分配。角色语义始终以响应中的 `role` 为准。

## 启动真实服务

从仓库根目录分别启动：

```bash
PYTHONPATH=RoboNexus-Agent conda run -n robonexus-qwen python \
  -m qwen_service.server --model-path weights/Qwen3-VL-4B-Instruct

PYTHONPATH=RoboNexus-Agent conda run -n robonexus-sam python \
  -m segmentation_service.server

PYTHONPATH=RoboNexus-Agent conda run -n robonexus-cutie python \
  -m tracking_service.server
```

默认端口依次为 8101、8102、8103。开发时可给每个命令传 `--backend mock`，无需
加载 GPU 模型即可验证进程间完整链路。

## CALVIN 离线回放

服务启动后，用 DiT venv（其中已有 LeRobot/pyarrow）回放验证集：

```bash
PYTHONPATH=RoboNexus-Agent multitask_dit_policy/.venv/bin/python \
  RoboNexus-Agent/core/calvin_replay.py \
  --episode 0 --max-frames 65 --output /tmp/robonexus-replay
```

输出包含逐帧角色 mask 叠加图和 `report.json`。报告记录每个对象的 tracker
confidence、时序 IoU、面积比例、motion score、总质量和是否发生纠偏。

正常帧只调用 Cutie。质量连续两帧低于阈值时，默认先用最近的 bbox 调 SAM；
若 SAM 返回空 mask，再调用 Qwen 重新定位并用 SAM/Cutie 修正。阈值集中定义在
`core.quality.QualityConfig`。

## 测试

```bash
PYTHONPATH=RoboNexus-Agent conda run -n robonexus-agent \
  python -m unittest discover -s RoboNexus-Agent/tests -v
```

测试覆盖编码、初始化、持续跟踪、错 episode 防护，以及连续低质量触发 SAM
纠偏，不需要 GPU。
