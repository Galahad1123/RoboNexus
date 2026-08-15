# RoboNexus Qwen-Agent 架构

状态：Draft v1，作为下一阶段实现的方向性约束。

本文档定义 RoboNexus Agent 的核心架构。若现有代码或旧文档与本文冲突，后续实现以本文为准；但在迁移完成前，不删除旧的确定性测试入口。

## 1. 核心决定

RoboNexus 不再自行实现一套完整的高层 Agent 框架。正式 Agent 主干采用 `qwen-agent`，Qwen3-VL-4B 负责多模态理解、逐步规划、工具选择、目标选择、失败分析和多轮任务记忆。

现有代码重新定位为 Qwen-Agent 可调用的机器人 Runtime 和工具 backend：

- Skill Registry 定义允许执行的原子技能；
- Policy Router 和 Client Pool 选择并连接四个 DiT；
- Skill Runtime 执行有界 action loop；
- Grounder 将语义目标转换成 mask、位置或 tracker；
- Verifier 判断原子技能和最终 goal 是否完成；
- Runtime 维护环境连续性、硬预算、安全约束、日志和产物。

Qwen-Agent 做软决策，确定性 Runtime 做硬约束。LLM 不能绕过技能白名单、动作限制、attempt 上限、goal 预算或最终成功验证。

## 2. Goal 输入

一次 Agent session 的输入定义为：

```text
GoalInput
├── goal_id
├── instruction                    用户自然语言任务
├── success_spec                   Runtime 可验证的最终条件
├── current_observation
│   ├── static_rgb                 固定第三人称相机
│   ├── wrist_rgb                  夹爪手腕相机
│   └── robot_state_ref            Runtime 内部状态引用
├── target_reference_image?        任务完成时的第三人称参考图
└── constraints
    ├── skill_attempt_limit
    ├── replan_limit
    ├── tool_call_limit
    └── environment_step_limit
```

`target_reference_image` 是可选但重要的一等输入。它表达最终目标状态，而不是某个中间步骤，也不是普通历史帧。Session 在每轮重要决策和最终验证时都应保证模型能区分：

- `CURRENT_STATIC_VIEW`：当前场景；
- `CURRENT_WRIST_VIEW`：当前腕部视角；
- `GOAL_REFERENCE_VIEW`：任务成功时应接近的最终状态。

参考图不能直接替代成功判定。它帮助 Qwen 避免多轮执行后偏离目标，最终完成仍必须由 `verify_goal` 工具确认。

`success_spec` 由任务入口、评测配置或经过校验的 goal compiler 提供，例如要求 `place_in_drawer(red_block)`。Qwen 可以提出补充条件，但不能自行删除或放宽用户原始成功条件。没有可执行的 `success_spec` 时，Runtime 可以允许探索和规划，但不得把 goal 标记为成功。

目标参考图可能来自另一条演示轨迹，因此主要表达物体关系和期望状态，不要求与当前场景做逐像素匹配。

## 3. 总体调用链

```text
用户 Goal + 当前双相机图像 + 可选完成参考图
                         │
                         ▼
              Qwen-Agent / Qwen3-VL-4B
       理解目标、每次选择一个原子技能、调用工具
                         │
                         ▼
                    Tool Registry
      ┌──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
 inspect_scene  ground_targets  execute_skill  verify_goal / abort_goal
                       │              │
                       ▼              ▼
                 Grounding Backend  Skill Runtime
                  ├─ CALVIN GT       ├─ Skill Registry
                  └─ SAM + Cutie     ├─ Policy Router
                                    ├─ four DiT clients
                                    └─ atomic action loop
                         │
                         ▼
                    CALVIN Environment
                         │
                         ▼
      新观测 + Oracle 结果 + 失败信息 + 视频/事件
                         │
                         └──────────→ Qwen-Agent 下一轮
```

每轮 Qwen-Agent 只选择一个原子技能或一个恢复/验证工具。它不输出低层机器人动作，不直接选择 checkpoint，不直接访问 CALVIN 环境对象。

## 4. 模块职责

### 4.1 Qwen-Agent

负责：

- 根据自然语言和多视角观测判断下一步；
- 从 34-skill Registry 中选择一个技能；
- 从多个候选物体中选择 source、destination 或 target；
- 根据工具结果决定继续、重试、重新定位、接近目标或改变计划；
- 维护任务级语义记忆；
- 在认为任务完成时调用 `verify_goal`；
- 在确认无法继续时调用 `abort_goal`，并提供结构化原因；
- 向用户给出结构化决策摘要和最终结果。

不负责：

- 生成或修改 7D action；
- 决定某技能使用哪个 DiT checkpoint；
- 修改 Runtime 预算；
- 仅凭语言声明 goal 成功；
- 直接使用 CALVIN 私有状态做高层选择，除非该状态由明确的工具结果公开。

### 4.2 Agent Runtime

负责：

- 创建和维护 session；
- 保存当前 observation，并以不可变 `observation_id` 暴露给工具；
- 注册 Qwen-Agent tools；
- 验证所有工具参数；
- 维护 attempt、replan、tool call、environment step 和 wall-clock 预算；
- 保证所有 attempt 在当前场景继续，不隐式 reset；
- 串行化会改变环境的工具调用；
- 记录结构化事件、图片、视频和最终报告；
- 在 Qwen 输出异常、工具异常或预算耗尽时安全终止。

Runtime 不替代 Qwen 规划。旧 `GoalOrchestrator` 和 `ScriptedPlanner` 保留为确定性测试 harness，不再作为正式智能主干继续扩展。

### 4.3 Skill Runtime

负责执行一次原子技能 attempt：

```text
validated skill request
→ route skill to policy
→ prepare outlined observations
→ reset policy session（不 reset 环境）
→ step DiT and CALVIN
→ verify after every environment step
→ stop on success, timeout or environment failure
→ return structured result
```

原子技能 attempt 内部继续使用快速 verifier，以便成功后立刻停止动作。Qwen-Agent 不参与逐个控制 step。

### 4.4 Grounding

Grounding 的职责是把 Qwen 选择的语义对象绑定到当前场景中的具体实例：

```text
semantic selection
→ GroundedObject
   ├── semantic_id
   ├── bbox / point / mask
   ├── optional world_position
   ├── confidence
   └── source backend
```

第一版：

```text
Qwen 选择 semantic_id
→ CalvinOracleGrounder 用 PyBullet 真值生成 mask/world position
```

这里 Oracle 只负责将 Qwen 已选择的对象身份精确落到像素和坐标，不能替 Qwen 决定应该选择哪个候选对象。

后续版本：

```text
Qwen bbox/point
→ SAM mask
→ Cutie tracker session
→ 每帧传播 mask
```

上层工具只依赖统一 `GroundedObject`，不依赖具体 backend。

### 4.5 Verifier

区分两级验证：

- Atomic verifier：`execute_skill` 内部每步运行，第一版使用 CALVIN Oracle；
- Goal verifier：Qwen 认为整体任务完成时显式调用 `verify_goal`。

第一版 goal verifier 可组合：

- CALVIN Oracle 谓词；
- 固定设施状态；
- 夹爪状态和持物启发式；
- 当前图像与目标参考图，先只作为 Qwen 判断辅助和日志证据。

后续再用 Qwen3-VL + 少量启发式规则替换 Oracle 在线判定。离线评测仍保留 Oracle 作为真值。

## 5. Outline 设计

Outline 是 Skill Runtime 的输入渲染步骤，不属于 Planner，也不再从 CLI 或初始 suite 临时拼装。

```text
Qwen selected roles
→ ground_targets
→ role-labelled masks
→ MaskOverlay
→ outlined static/wrist RGB
→ DiT
```

角色与训练数据保持一致：

- `source`：黑色 outline；
- `target`：黑色 outline；
- `destination`：青色 outline；
- 默认只画边界，不填充内部，即 `alpha=0.0`。

保存的视频默认展示 DiT 实际接收到的第三人称 outlined RGB。原始图、outlined 图和 grounding 元数据均应可选落盘，便于诊断模型失败与 grounding 失败。

## 6. 连续环境、重试和重规划

任何工具失败都不隐式恢复 CALVIN 环境。最新 observation 成为下一轮 Qwen-Agent 的输入。

Qwen 可以选择：

- 再次执行同一技能；
- 重新 ground；
- 调用 `approach_target`；
- 执行前置技能；
- 选择另一个候选对象；
- 改变剩余计划；
- 请求验证或终止。

### 分层 Skill 选择与评测隔离

Runtime 向 Qwen 发布按 `lift/place/push/rotate/articulation/switch/stack` 分组的完整
34-skill catalog。Planner 先依据自然语言 goal 和最新观测选类别，再选择一个原子
skill；它不接收官方 CALVIN 序列的 `goal_skill_ids` 或 Oracle predicate 列表。

官方序列只属于 evaluator-private success spec，用于 Oracle 判断 goal 是否按顺序推进。
为高失败技能配置的 recovery candidates 是建议集合，不是 backend validator。Qwen 在
失败后的 `inspect_scene` 图像上决定直接重试或恢复；`execute_skill.decision_type` 明确
标记 `goal_progress/direct_retry/recovery`。Runtime 独立判定实际进度，恢复技能成功不会
推进官方序列。Dashboard 只展示该结构化决策、简短依据与结果，不记录隐藏思维链。

Skill Registry 分别保存面向 Planner 的 `planner_instruction` 和与训练分布对齐的 DiT
`instruction`，Qwen 不再从缺失颜色信息的 outline policy prompt 推断用户目标。
`inspect_scene.object_locations` 是 backend-neutral 的当前物体位置契约；CALVIN 初版由
Oracle contact metadata 实现，后续可替换为 VLM/检测跟踪或真机感知。Planner 选择
table/drawer/slider lift 变体时必须采用最新 object_locations，而不是原始 goal 中的旧位置。

Runtime 维护硬上限：

- 同一逻辑子任务默认最多 3 次 `execute_skill`；
- 失败耗尽后默认最多允许 2 个新的 replan epoch；
- 另有 goal 级工具调用数、环境步数和时间限制。

Qwen-Agent round 上限耗尽而 Session 仍为 active 时，runner 使用最新 observation
自动执行一次终局 `verify_goal`。该调用不占用 Qwen round，但仍计入 Runtime tool-call
budget，并完整写入事件与 Dashboard。只有 verifier 返回 success 才结束 goal；验证失败
仍按 round exhaustion 处理，不能将兜底验证等同于兜底成功。

官方序列的 attempt 预算按 evaluator-private 的当前 goal stage 计数，而不是按 grounding
handle 或 Qwen 的自然语言 intent 计数。重新 grounding 不清零当前 stage 的计数；当前
stage 验证成功后进入下一 stage，并获得独立的三次机会。恢复技能使用单独的 recovery
scope，因此提前执行某个未来 goal skill 不会消耗它日后作为正式 stage 的额度。当前
stage 第三次执行仍失败时，Runtime 立即以 `BUDGET_EXHAUSTED / skill_attempt_limit`
终止整个 goal，不允许 Qwen 跳过该 stage 继续后续计划。

## 7. Memory 与上下文控制

Qwen-Agent 维护语义对话记忆，但不应无限累积全分辨率历史帧。Session 保存：

- 用户原始 goal；
- 不变的目标参考图；
- 最新 current static/wrist 图；
- 已完成原子技能；
- 最近若干次工具调用的结构化摘要；
- 当前 grounding handles；
- 预算状态；
- 关键失败前后快照引用。

视频、完整 action、全部中间帧只作为 artifact 保存，不直接塞入模型上下文。工具结果向 Qwen 返回紧凑 JSON 和少量关键图像。

## 8. Web UI 与可观测性

Web UI 是同一 Agent session 的视图和控制面，不实现另一套业务逻辑。UI 至少展示：

- 用户 goal 和目标参考图；
- 当前双相机观测；
- Qwen 的结构化决策摘要；
- 工具调用及经过脱敏/裁剪的参数；
- 当前 grounding 与 outlined policy 输入；
- 路由到的 policy ID；
- attempt、replan 和预算；
- verifier 结果和视频；
- goal 最终状态。

UI 展示决策摘要，不依赖或持久化模型隐藏思维链。后续中途修改任务时，作为带版本号的新 `GoalUpdate` 注入同一 session；正在运行的动作必须先通过 Runtime 的取消/安全停止机制结束。

## 9. 依赖方向和环境边界

```text
qwen-agent (robonexus-qwen)
        ↓ tool calls / protocol
Agent Runtime (environment-neutral)
        ↓ interfaces
CALVIN Backend (robonexus-calvin)
        ↓ websocket
DiT Policy Servers (multitask_dit_policy venv)
```

不要求把 Qwen、CALVIN 和 DiT 安装到同一 Conda 环境。跨环境边界使用明确协议；图像可先用路径/共享 artifact store，稳定后再决定是否改为共享内存或流式传输。

## 10. 迁移原则

1. 先冻结工具协议和领域 Schema；
2. 将现有能力封装为工具，但暂不接 Qwen；
3. 使用 fake session 和真实 CALVIN backend 逐个测试工具；
4. 接入 `qwen-agent==0.0.34` 的 `Assistant`、`BaseTool` 和 `register_tool`；
5. 验证 Qwen 多轮工具调用；
6. 先跑单技能，再跑多技能和失败恢复；
7. 最后接 Web UI、SAM 和 Cutie。

迁移期间不得继续把 grounding、recovery 或 VLM 决策堆入 `run_calvin_goal.py`。新能力必须先落到工具协议和对应 backend 接口中。

## 11. 第一版验收顺序

1. `inspect_scene` 能返回最新双相机观测及引用；
2. `ground_targets` 能把 Qwen 选择的 semantic ID 转为 CALVIN GT mask；
3. `execute_skill` 能正确路由 DiT，并使用 outlined 输入；
4. `approach_target` 能在连续场景中接近目标；
5. `verify_goal` 能用 Oracle 给出结构化结论，`abort_goal` 能安全终止；
6. 不接 Qwen，直接调用工具完成确定性闭环；
7. 接入 Qwen-Agent，完成单技能 goal；
8. 完成包含参考图的多技能 goal；
9. 失败后由 Qwen 选择重试、approach 或 replan；
10. 预算耗尽、非法工具参数和服务异常均能安全 abort。
