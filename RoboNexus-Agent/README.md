# RoboNexus perception pipeline

> 新的正式 Agent 主干采用 Qwen-Agent + Qwen3-VL。总体设计和工具协议分别见
> [Qwen-Agent 架构](../docs/agent/QWEN_AGENT_ARCHITECTURE.md) 与
> [Qwen-Agent 工具协议](../docs/agent/QWEN_AGENT_TOOL_PROTOCOL.md)。现有
> ScriptedPlanner/GoalOrchestrator 作为确定性 Runtime 测试和兼容入口保留。

早期 Qwen3-VL 定位、SAM 首帧分割和 Cutie 跟踪流水线已归档到
`core/legacy/perception` 与 `core/legacy/transport`，仅供回归和后续能力复用。

## Core 分层

新 Agent 代码只使用以下主路径：

- `core/domain`：唯一公共领域类型；
- `core/runtime`：Session、预算、revision、幂等与终态；
- `core/tools`：Qwen-Agent 可调用的 backend-neutral 工具；
- `core/ports`：环境、grounding、技能、运动、验证和 artifact 接口；
- `core/skills`：34-skill Registry；
- `core/policies`：Policy Router、Client Pool 和 handshake；
- `adapters/qwen_agent`：唯一允许导入 `qwen_agent` 的位置。

迁移状态与旧路径处置见
[Agent code migration map](../docs/agent/CODE_MIGRATION_MAP.md)。

旧 `core.agent`、`core.routing`、`core.planning`、`core.orchestration`、
`core.execution` 和 `core.inspection` 目录已移除；仍需回归的实现统一位于
`core/legacy`。根目录 `agent/` 及 `core.schemas`、`core.orchestrator`、
`core.clients` 等平面入口只为旧外部入口保留。CALVIN/PyBullet 具体实现位于仓库级
`backends/calvin/`，不属于通用 core。

## CALVIN-D34 Skill 与 Policy 配置

34 个 canonical skill 定义在 `configs/calvin/skills_d34.yaml`。该文件保存 DiT
实际使用的固定 instruction、CALVIN oracle task、policy group、每次执行步数、
attempt 上限和 success guard。Planner 后续只选择 `skill_id`，不能改写这里的
instruction。

`planner_instruction` 与 DiT `instruction` 明确分离：前者保留颜色、位置等 goal
语义并只提供给 Qwen；后者保持 checkpoint 训练时的 outline 指令，仍是唯一发送给
DiT policy server 的文本。未显式配置 `planner_instruction` 时默认沿用 `instruction`。

每个 skill 必须声明 `category`，并可声明结构化 `planner_guidance` 与条件
`recovery`。Runtime 将完整 34-skill catalog 按 category 发布给 Qwen；官方评测的
`goal_skill_ids`/Oracle predicates 仅保存在 Runtime 内部，不进入模型上下文。Qwen
先选类别、再选一个原子技能，并在每轮的 `ROBOT_GOAL_CONTEXT` 中收到 guidance。
guidance 仅作为高层
Planner 的建议，backend 工具不会把它当作语义 validator；新增针对性经验应写入对应
skill，而不是继续膨胀全局 system prompt。

`recovery` 为失败率较高的技能列出条件、候选技能/候选类别和返回技能。失败后 Qwen
必须先查看最新双相机图像，再决定直接重试或调用配置允许的恢复技能。恢复技能执行成功
只记录 `recovery_skill_achieved`，不会推进 evaluator-private 的官方序列。

四组模型部署配置位于 `configs/calvin/policies.yaml`。当前已为 `general`、`push`、
`rotate`、`stack` 选定 checkpoint；后续替换权重只需修改对应路径，例如：

```yaml
general:
  enabled: true
  checkpoint: outputs/calvin_general_21_dino/checkpoint_50000
  host: 127.0.0.1
  port: 10093
```

相对 checkpoint 路径以 `checkpoint_base: multitask_dit_policy` 为基准。运行静态
配置检查：

```bash
python scripts/agent/validate_calvin_config.py
```

完整可执行性检查还会验证每个目录包含
`config.json`、`dataset_stats.json` 和 `model.safetensors`：

```bash
python scripts/agent/validate_calvin_config.py --require-runnable
```

`core.policies.PolicyRouter` 只负责把 canonical `skill_id` 解析为对应的 skill 与
policy 配置，不加载模型，也不持有连接。`core.policies.PolicyClientPool` 根据 policy
配置懒连接四个 WebSocket endpoint，并为每个 policy 复用一个连接。启动 Agent 前可
调用 `connect_all()` 或 `health_check_all()` 预连接四个服务。

连接建立时会校验 DiT protocol v2 handshake，并确保四个 endpoint 的图像键与尺寸、
state/action 维度、观测步数、动作块大小和 horizon 一致。不兼容会在执行 episode 前
直接报错，并关闭本轮已经创建的连接。

`core.legacy.execution.AtomicSkillExecutor` 是旧单原子技能的策略会话边界。调用
`begin_attempt()` 后，执行端逐帧把当前 observation 传给 `step()`，每次只取得一个
动作。`begin_attempt()` 只重置远端 DiT 的时序缓存，不重置仿真或真实环境；因此下一次
attempt 可以直接使用失败后的当前场景。完成后必须调用 `end_attempt()`，再开始下一次
attempt 或切换 skill。

CALVIN 环境侧的逐步 action loop 位于仓库级
`backends/calvin/execution/attempt_runner.py`。旧的 `CalvinOracleRetryAgent` 和
`evaluate.py --max-attempts` 入口保持不变，但内部已委托给可复用的
`CalvinSkillAttemptRunner`。Oracle verifier 在整个连续 retry episode 中共用首次
attempt 前的 baseline，并与 clean rollout 使用相同的 success guard 规则。

## Goal Planner 与状态机

`core.legacy.planning.ScriptedPlanner` 是第一版确定性 Planner。初始脚本和每次 replan
脚本都由 canonical `skill_id` 组成，但 `next_step()`/`replan()` 每次只返回一个
原子技能或一个终止决定，不把整条计划泄漏给执行层。

`core.legacy.orchestration.GoalOrchestrator` 负责旧 goal 状态机。默认每个技能最多执行三次；
三次均失败后请求 Planner replan，最多允许两次。第二次 replan 后的新技能仍连续
失败三次时，以 `replan_budget_exhausted` 终止整个 goal。每个 attempt 都接收上一
attempt 的 observation，因此状态机不会隐式恢复环境。Planner 决策、attempt、
replan、skill 和 goal 终止均写入统一的结构化事件接口。

## CALVIN StateInspector

`backends/calvin/verification/state_inspector.py` 从 CALVIN `scene_obs` 或 `scene_info` 读取固定
设施的绝对状态。第一版覆盖 drawer、slider、lightbulb 和 LED。对于
open/close、left/right、on/off 任务，如果目标状态已经满足，Orchestrator 会记录
`skill_skipped` 并直接请求下一步，不调用 DiT。

依赖打开抽屉的 lift/place/push 任务在抽屉未打开时返回
`INVALID_PRECONDITION`，不浪费 attempt，直接触发 replan；这类 replan 同样受两次
预算约束。状态缺失或格式异常返回 `UNKNOWN`，状态机继续执行技能，不因 inspector
不可用而错误跳过任务。

## CALVIN scripted goal 入口

四个 policy server 启动后，可在 `robonexus-calvin` 环境运行完整 goal 闭环：

```bash
conda run -n robonexus-calvin python scripts/agent/run_calvin_goal.py \
  --goal-id drawer-cycle \
  --goal "open and then close the drawer" \
  --skill open_drawer \
  --skill close_drawer \
  --replan-script open_drawer,close_drawer \
  --replan-script open_drawer,close_drawer \
  --initial-task open_drawer \
  --mask-provider gt \
  --save-video \
  --output-dir outputs/calvin_agent/drawer-cycle
```

入口启动时连接 `policies.yaml` 中全部四个 endpoint 并验证 handshake，然后只在实际
执行某个 skill 时使用 Router 选中的 client。`--suite` 与 `--initial-task` 仅用于选择
goal 的初始 CALVIN 状态。输出目录包含 `result.json` 和逐事件刷新的 `events.jsonl`；
传入 `--save-video` 后，每次实际执行的 attempt 会单独写入 `videos/`，文件名包含
plan step、replan 次数、attempt 次数及成功/失败状态。

对使用 outline 数据训练的 checkpoint，应传入 `--mask-provider gt`。入口会从当前
suite 的 `mask_style` 读取颜色、透明度和轮廓宽度，并从所选初始 episode 的 manifest
读取角色目标；同一张加 outline 的图像会同时送入 DiT 和写入视频。若启用 mask 但
当前 skill 没有角色定义，执行会明确失败，而不会静默退化为原始 RGB。多技能 goal
后续需要由 Planner/grounding 模块为每个新 skill 提供对应的 `mask_roles`。

旧阶段 6 的兼容验收位于 `tests/legacy/orchestration/test_agent_acceptance.py`，覆盖单/双技能、同/跨 policy、
attempt retry、两级 replan、abort、已满足 skip 和终止后不再调用 Planner。

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

六个新工具的 CALVIN 真实环境 smoke 测试入口为：

```bash
# 先启动 general policy endpoint（checkpoint 路径相对仓库根）
uv run --project multitask_dit_policy python \
  -m multitask_dit_policy.serving.policy_server \
  --checkpoint multitask_dit_policy/outputs/calvin_general_21_dino/checkpoint_80000 \
  --port 10093

# 另一个终端运行真实 CALVIN episode；输出含 report、events、图像、mask 和 MP4
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-calvin python \
  scripts/agent/smoke_calvin_tools.py \
  --output-dir outputs/calvin_agent/tool_smoke_full
```

只验证不依赖 DiT 的 `inspect_scene`、`ground_targets`、`approach_target`、
`verify_goal` 和 `abort_goal` 时，可增加 `--skip-execute`，无需启动 policy endpoint。

不接 Qwen 的确定性工具闭环使用同一个 policy endpoint，运行：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-calvin python \
  scripts/agent/run_calvin_deterministic_tools.py \
  --task lift_red_block_table \
  --semantic-id block_red \
  --output-dir outputs/calvin_agent/deterministic_tool_loop
```

该 harness 严格通过公共工具推进单技能 goal：先检查和定位，再执行一次技能并验证完整
goal；失败时保持当前场景，重新检查和定位，调用 `approach_target` 后再次定位并重试。
最多三次 attempt，成功后立即停止，三次仍失败则显式调用 `abort_goal`。它只用于在接入
Qwen-Agent 前验证 Runtime 和工具状态转换，不承担正式 Agent 的语义规划职责。

## Qwen-Agent 工具闭环

正式 Qwen-Agent 接入跨两个 Conda 环境运行。先启动对应 DiT endpoint，再在 CALVIN
环境创建一个连续的 Tool Session Runtime：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-calvin python \
  scripts/agent/serve_calvin_tool_session.py \
  --task lift_red_block_table \
  --disable-approach \
  --episode-index 0 \
  --output-dir outputs/calvin_agent/qwen_single_skill
```

然后在另一个终端加载本地 Qwen3-VL-4B，并交给 Qwen-Agent 驱动多轮工具调用：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-qwen python \
  scripts/agent/run_qwen_single_skill.py \
  --runtime-url http://127.0.0.1:8110 \
  --expect-skill lift_red_block_table \
  --model-path weights/Qwen3-VL-4B-Instruct
```

Qwen 进程只持有 `BaseTool` RPC adapter；CALVIN 进程独占环境、Session、Oracle 和六个
具体工具。`--disable-approach` 会从该 Session 的 capability 和 Qwen 工具列表中彻底移除
`approach_target`，不是只在提示词中建议模型不要调用。省略该参数则重新启用工具。

双子任务可在同一个连续环境中显式发布两个 skill：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-calvin python \
  scripts/agent/serve_calvin_tool_session.py \
  --task lift_red_block_table \
  --skill open_drawer \
  --skill move_slider_left \
  --goal "open the drawer and move the sliding door to the left" \
  --disable-approach \
  --environment-step-limit 1200 \
  --output-dir outputs/calvin_agent/qwen_two_skills

PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-qwen python \
  scripts/agent/run_qwen_single_skill.py \
  --runtime-url http://127.0.0.1:8110 \
  --expect-skill open_drawer \
  --expect-skill move_slider_left \
  --model-path weights/Qwen3-VL-4B-Instruct
```

这里 `--task` 选择 CALVIN 初始 episode；重复的 `--skill` 定义本次 Agent 可路由的技能和
Oracle 成功条件。Qwen 客户端会检查 Runtime 发布的 skill 集合与 `--expect-skill` 完全
一致，避免误连到残留 Session。Runtime 根据 CALVIN
grounder capability 向 Qwen 发布 semantic-only Schema。模型生成的参数仍会在 Runtime
用原始严格 Schema 二次校验。每次 Qwen 更新都会原子写入
`qwen_agent_checkpoint.json`；正常结束或捕获到异常时，最终报告原子写入
`qwen_agent_result.json`。CALVIN Runtime 独立原子写入 `runtime_result.json`，工具事件
继续逐行刷新到 `events.jsonl`，每次 `execute_skill` 另存 outlined MP4。

如果外层 Qwen 进程被终止，可不加载模型，直接用 Runtime 结果、终态事件和最近一次
Agent 检查点重建最终报告：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-qwen python \
  scripts/agent/run_qwen_single_skill.py \
  --recover-output-dir outputs/calvin_agent/<run-directory>
```

恢复时优先采用 `runtime_result.json` 的终态；若它尚未来得及产生，则从
`events.jsonl` 的 `goal_succeeded`/`goal_aborted` 终态事件判定。这样批量评测不会把
“机器人已完成但 Qwen 启动器未正常退出”误报为结果缺失。

`inspect_scene` 的远程工具结果同时返回结构化 JSON 和带明确
`CURRENT_STATIC_VIEW`/`CURRENT_WRIST_VIEW` 标签的两个 `ContentItem(image=...)`。因此
Qwen 在技能执行、整体校验或失败之后重新检查场景时，看到的是连续环境中的最新图像，
而不是只有 artifact 路径或重复使用初始帧。其他工具保持紧凑 JSON，控制多轮上下文大小。

Qwen 环境中的多模态适配测试运行：

```bash
PYTHONPATH=RoboNexus-Agent:. conda run -n robonexus-qwen python \
  -m unittest discover -s RoboNexus-Agent/tests_qwen -v
```

```bash
PYTHONPATH=RoboNexus-Agent conda run -n robonexus-agent \
  python -m unittest discover -s RoboNexus-Agent/tests -v
```

## 零依赖离线评测 Dashboard

评测结束后可用 Python 标准库把日志、图片和 MP4 组织为静态 HTML。报告只通过相对路径
引用 artifacts，不复制或 base64 嵌入大文件，也不需要启动 Web 服务。

生成单个 sequence 报告：

```bash
python scripts/agent/generate_eval_report.py \
  outputs/calvin_agent/<eval-run>/<sequence>
```

生成整个批次的可筛选索引，并同时更新每个 sequence 的详细报告：

```bash
python scripts/agent/generate_eval_report.py \
  outputs/calvin_agent/<eval-run> --recursive
```

批次入口是 `<eval-run>/index.html`，详细入口是每个 sequence 下的 `report.html`。详细页
包含 goal 状态、技能进度、重试统计、DiT policy/步数/视频、grounding preview、前后相机
图像、Qwen 工具参数、Runtime 返回以及 Oracle 验证。页面内容均经过 HTML escaping。
视频状态优先取 `execute_skill` 的真实 `success/verification`，因此超时和 Oracle 未满足会
显示为 `failed`，而不是因为缺少独立成功事件显示为 `unknown`。视频和工具时间线同时
显示原计划 `stage N/M`、真实 `skill_id`、内部 logical subtask ID 与 attempt；重规划产生
新的 `subtask_00N` 时，不会再按该数字错误映射原计划技能。

测试覆盖编码、初始化、持续跟踪、错 episode 防护，以及连续低质量触发 SAM
纠偏，不需要 GPU。
