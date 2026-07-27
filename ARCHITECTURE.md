# RoboNexus 系统架构设计

> **RoboNexus** is a modular agent framework for orchestrating perception, planning, and interchangeable robot executors.

RoboNexus 将机器人任务拆分为“规划—目标定位—技能执行—完成判定”的闭环流程。Agent 负责高层决策与状态管理，底层策略作为可替换的 Executor 接入，仿真环境或真实机器人仅负责提供观测并执行动作。

## 1. 系统流程

```text
当前图像 + 语言指令 [+ 目标图像]
                  │
                  ▼
          Qwen3-VL Planner
          生成或修复结构化计划
                  │
                  ▼
      Grounding：目标物体与目标区域
                  │
                  ▼
       SAM2 初始化 mask，Cutie 持续跟踪
                  │
                  ▼
        Atomic DiT 执行原子技能
                  │
                  ▼
      Rule Predicate + VLM Verifier
          │                    │
        成功                 失败/不确定
          │                    │
      下一子任务        重新定位/重试/重规划
```

Planner 初期使用 Qwen3-VL-4B 或 8B，并根据规划、grounding 效果和资源占用选择最终版本。当前图像与目标图像均来自固定第三人称相机；目标图像用于帮助模型理解期望的物体关系和最终状态。

## 2. 核心模块

### Agent Orchestrator

Agent 是系统的控制中心，负责：

- 调用 Planner 生成或修复任务计划；
- 维护子任务状态机；
- 调用 grounding、segmentation 与 tracking 服务；
- 构造 mask-enhanced Policy 输入；
- 调用 Executor 获取 action chunk；
- 完成判定、错误恢复和日志记录。

Planner 只提供语义计划，具体状态切换、超时、重试和模块调用由确定性的 Orchestrator 管理。

### Executor / Policy Server

Executor 根据标准化输入生成 action chunk，不负责高层规划、技能切换或完成判定。第一版接入开源 `multitask_dit_policy`，后续可替换为其他 DiT、Diffusion Policy、ACT 或 VLA。

### Environment Client

Environment Client 负责：

- 提供第三人称图像、腕部图像和机器人状态；
- 执行 Agent 下发的 action chunk；
- 返回新的观测和执行状态；
- 支持动作取消与安全停止。

第一版仿真环境使用 CALVIN。

## 3. 结构化任务计划

Planner 只能从已注册的技能和谓词中生成计划。每个子任务至少包含：

- 原子技能；
- 操作对象；
- 目标对象或目标区域；
- 前置条件；
- 成功谓词；
- 超时时间；
- 重试与恢复策略。

示例 Skill Contract：

```json
{
  "step_id": "step_002",
  "primitive": "grasp",
  "source_ref": "red_cube",
  "target_ref": null,
  "preconditions": [
    {"predicate": "gripper_empty"}
  ],
  "success_conditions": [
    {"predicate": "holding", "subject": "red_cube"}
  ],
  "timeout_seconds": 8,
  "recovery": {
    "max_retries": 2,
    "first_action": "re_ground",
    "fallback": "replan"
  }
}
```

`red_cube` 属于语义对象引用。完成 grounding 和 tracking 后，Orchestrator 将其绑定到具体视觉实例、tracker 和 mask，再生成可执行的 Policy 请求。

## 4. 闭环执行与恢复

每个子任务采用以下状态机：

```text
PENDING
   ↓
GROUNDING
   ↓
EXECUTING
   ↓
VERIFYING
   ├── DONE
   ├── RETRY
   ├── RE_GROUND
   ├── REPLAN
   └── ABORT
```

恢复策略分为四级：

```text
Level 1：重新跟踪或重新生成 mask
Level 2：重新执行当前原子技能
Level 3：执行 retreat / reset / reposition
Level 4：根据当前观测重新规划剩余任务
```

例如 `grasp(red_cube)` 失败时：目标丢失则重新定位；物体仍可操作则重试；场景发生明显变化则重规划；超过最大重试次数后终止任务。

## 5. Mask-enhanced Atomic DiT

第一版不引入 affordance heatmap，也不显式指定抓取方式或精确落点，而是验证 Atomic DiT 能否根据原子技能与角色 mask 学习动作模式。

推荐使用固定角色颜色叠加到原始 RGB：

```text
source：红色半透明覆盖
destination：蓝色半透明覆盖
```

```text
enhanced_rgb = alpha × role_color + (1 - alpha) × original_rgb
```

Policy 条件保持结构化：

```json
{
  "primitive": "grasp",
  "source": "source_mask",
  "target": null,
  "target_region": null
}
```

训练时加入与真实分割和跟踪误差相近的扰动，包括 mask 腐蚀/膨胀、边缘误差、短暂缺失、遮挡、位置抖动和跟踪漂移，避免模型仅适应仿真 GT mask。

## 6. 完成判定

第一版采用简单规则谓词与 VLM 判定相结合：

1. 根据执行步数、夹爪状态、目标运动或其他部署可获得信号触发检查；
2. VLM 结合操作前后第三人称图像、腕部图像和成功谓词，输出 `SUCCESS`、`FAILURE` 或 `CONTINUE`；
3. 必要时要求结果连续成立多次，再切换到下一子任务。

仿真特权信息不参与 Agent 在线决策，只用于离线评测真值和 Oracle 上界实验。

## 7. Registry 与接口约束

Agent 启动时加载技能、谓词和 Executor Registry。Planner 不得生成未注册的技能或谓词，输出计划必须经过：

1. Schema 校验；
2. 技能与谓词白名单校验；
3. 参数完整性检查；
4. 子任务依赖检查。

Executor 需要声明自身能力：

```json
{
  "policy_name": "atomic_dit_v1",
  "supported_primitives": ["grasp", "transport", "place"],
  "required_inputs": [
    "third_person_rgb",
    "proprioception",
    "primitive",
    "source_mask"
  ],
  "optional_inputs": ["target_mask", "wrist_rgb"],
  "action_horizon": 16,
  "control_frequency_hz": 20,
  "requires_reset": true
}
```

共享消息结构由独立的 protocol package 维护，Agent、Policy Server 和 Environment Client 不得分别复制协议定义。

## 8. Server–Client 架构

系统由三个相对独立的服务组成：

```text
Environment Client
        │ observations / execution result
        ▼
Agent Orchestrator
        │ policy request
        ▼
Policy Server
        │ action chunk
        ▼
Agent Orchestrator
        │ validated action chunk
        ▼
Environment Client
```

约束如下：

- Environment 不直接调用 Policy；
- Policy 不直接控制 Environment；
- 所有动作必须经过 Agent 校验和转发；
- Agent 是 episode、step 和 action 状态的唯一可信来源。

Qwen、SAM2 和 Cutie 可作为独立推理服务部署，以隔离依赖和 GPU 环境。第一版使用 SAM2 初始化 mask、Cutie 持续跟踪；TAPNext 作为后续扩展。

低频控制消息，高频图像、机器人状态与动作通过 websockts 传输，与 starvla 等主流项目保持一致。所有信息在逻辑上由 Agent 统一管理，但应避免不必要的图像编码和内存复制。

## 9. 第一版技术选择

| 模块 | 初始方案 |
|---|---|
| Planner / Grounder | Qwen3-VL-4B 或 8B |
| Segmentation | SAM2 |
| Tracking | Cutie |
| Executor | `multitask_dit_policy` |
| Environment | CALVIN |
| Completion Verifier | 简单规则谓词 + VLM |
| Control Communication | WebSocket |
| High-frequency Data | WebSocket |

## 10. 开发顺序

1. 冻结代码架构、统一消息协议和核心 Schema；
2. 使用 Mock Agent、Policy 和 Environment 跑通完整消息流；
3. 并行搭建 Agent Orchestrator，并验证 mask-enhanced DiT；
4. 接入 CALVIN 与 `multitask_dit_policy`，打通 action chunk 闭环；
5. 接入 Qwen3-VL、SAM2 和 Cutie；
6. 加入简单规则谓词、VLM 完成判定和恢复机制；
7. 完成端到端评测，并逐步替换或扩展 Executor。

项目初期采用单一 monorepo，包含共享 protocol package、Agent、Policy Server 和 Environment Client 四个独立模块；三者以独立进程运行，待协议稳定后再按维护和发布需求拆分仓库。
