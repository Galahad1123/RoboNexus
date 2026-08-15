# Qwen-Agent Robot Tool Protocol

状态：Draft v1。本文档冻结第一版 Qwen-Agent 工具表面；实现变更必须先更新协议和测试。

目标运行环境已安装 `qwen-agent==0.0.34`。工具实现应继承 `qwen_agent.tools.base.BaseTool`，使用 OpenAI-compatible JSON Schema 声明 `parameters`，并通过 `register_tool` 或显式实例传给 `qwen_agent.agents.Assistant(function_list=...)`。

## 1. 设计原则

- Qwen 每次调用一个语义级、有界工具，不调用低层 7D action；
- 参数必须是 JSON object，并由 Runtime 二次校验；
- 改变环境的工具严格串行；
- 工具使用 handle 引用大对象，避免在 JSON 中传 mask、视频或完整 observation；
- 所有结果都有统一 envelope；
- 所有错误都是可枚举 code，不要求 Qwen 解析 Python traceback；
- 工具结果可以同时返回 JSON 文本与 `ContentItem(image=...)`；
- 当前图像和目标参考图必须带清晰角色标签；
- CALVIN 私有真值只能存在于 backend 内部或明确标记的 Oracle 结果中。

## 2. 公共类型

### 2.1 ToolResultEnvelope

所有工具的逻辑结果遵循：

```json
{
  "protocol_version": 1,
  "tool_call_id": "tc_000042",
  "session_id": "session_2026_001",
  "status": "succeeded",
  "error": null,
  "data": {},
  "artifacts": [],
  "budget": {
    "tool_calls_used": 4,
    "tool_calls_remaining": 28,
    "environment_steps_used": 117,
    "environment_steps_remaining": 1883,
    "replans_used": 0,
    "replans_remaining": 2
  }
}
```

`status` 取值：

```text
succeeded | failed | rejected | unavailable
```

- `failed`：工具合法执行，但任务结果未达成；
- `rejected`：参数、前置条件或预算不允许执行；
- `unavailable`：依赖服务或环境异常。

### 2.2 ToolError

```json
{
  "code": "INVALID_PRECONDITION",
  "message": "drawer must be open before place_in_drawer",
  "retryable": true,
  "details": {}
}
```

第一版公共错误码：

```text
INVALID_ARGUMENT
UNKNOWN_SKILL
UNSUPPORTED_GROUNDING
GROUNDING_NOT_FOUND
STALE_OBSERVATION
STALE_GROUNDING
INVALID_PRECONDITION
ATTEMPT_BUDGET_EXHAUSTED
REPLAN_BUDGET_EXHAUSTED
GOAL_BUDGET_EXHAUSTED
POLICY_UNAVAILABLE
ENVIRONMENT_UNAVAILABLE
VERIFIER_UNAVAILABLE
TOOL_INTERNAL_ERROR
```

### 2.3 ObservationRef

```json
{
  "observation_id": "obs_000117",
  "captured_at_step": 117,
  "static_image": "artifact://session_2026_001/obs_000117/static.png",
  "wrist_image": "artifact://session_2026_001/obs_000117/wrist.png"
}
```

Observation 是不可变快照。改变环境的工具完成后，之前的 observation 可用于历史比较，但不能作为新动作的默认当前状态。

### 2.4 GroundingProposal

Qwen 可以用不同粒度提出目标。第一版 CALVIN backend 主要支持 `semantic`：

```json
{
  "role": "source",
  "representation": "semantic",
  "semantic_id": "block_red",
  "description": "the red block on the table"
}
```

未来视觉 backend 支持：

```json
{
  "role": "destination",
  "representation": "bbox",
  "camera": "static",
  "bbox_xyxy": [82, 61, 139, 126],
  "description": "the open drawer interior"
}
```

统一允许的 role：

```text
source | destination | target
```

### 2.5 GroundedObject

```json
{
  "grounding_id": "ground_000031",
  "role": "source",
  "semantic_id": "block_red",
  "source": "calvin_gt",
  "observation_id": "obs_000117",
  "bbox_xyxy": [82, 61, 139, 126],
  "mask_handle": "mask://session_2026_001/ground_000031",
  "tracker_handle": null,
  "world_position": [0.12, -0.24, 0.46],
  "confidence": 1.0
}
```

Grounding 与产生它的 observation 绑定。Runtime 可以允许 tracker 更新后的 grounding 继续使用；没有 tracker 且场景已明显改变时，应返回 `STALE_GROUNDING`。

### 2.6 ArtifactRef

```json
{
  "kind": "outlined_static_image",
  "uri": "artifact://session_2026_001/attempt_003/outlined_static.png",
  "mime_type": "image/png"
}
```

## 3. Qwen 多模态消息约定

初始用户消息使用 Qwen-Agent `ContentItem` 列表，顺序和标签固定：

```python
[
    ContentItem(text="USER_GOAL: Put the red block in the drawer."),
    ContentItem(text="CURRENT_STATIC_VIEW"),
    ContentItem(image=".../current_static.png"),
    ContentItem(text="CURRENT_WRIST_VIEW"),
    ContentItem(image=".../current_wrist.png"),
    ContentItem(text="GOAL_REFERENCE_VIEW"),
    ContentItem(image=".../goal_reference.png"),
]
```

没有参考图时省略最后两项，并显式加入：

```text
GOAL_REFERENCE_VIEW: not provided
```

后续工具轮次通常只返回最新关键图像。目标参考图由 Session 作为不变 goal context 管理，并在上下文压缩或关键 replan 后重新附加，防止模型遗忘最终目标。

## 4. 工具一：inspect_scene

用途：获取当前连续环境的最新观测和可公开状态摘要，不改变环境。

### 参数 Schema

```json
{
  "type": "object",
  "properties": {
    "reason": {
      "type": "string",
      "description": "Why a fresh observation is needed."
    },
    "include_robot_summary": {
      "type": "boolean",
      "default": true
    },
    "include_facility_summary": {
      "type": "boolean",
      "default": true
    }
  },
  "additionalProperties": false
}
```

### 成功结果

```json
{
  "observation": {
    "observation_id": "obs_000117",
    "captured_at_step": 117,
    "static_image": "artifact://.../static.png",
    "wrist_image": "artifact://.../wrist.png"
  },
  "robot_summary": {
    "gripper": "open",
    "holding_object": "unknown"
  },
  "facility_summary": {
    "drawer": "closed",
    "slider": "left",
    "lightbulb": "off",
    "led": "on"
  }
}
```

第一版 facility summary 来自 CALVIN StateInspector。Qwen 必须知道这些是工具公开的 backend 状态，而不是视觉推断。

## 5. 工具二：ground_targets

用途：将 Qwen 选择的 source、destination 或 target 绑定到当前场景实例。工具不替 Qwen 决定选哪个候选目标。

### 参数 Schema

```json
{
  "type": "object",
  "required": ["skill_id", "observation_id", "proposals"],
  "properties": {
    "skill_id": {"type": "string"},
    "observation_id": {"type": "string"},
    "proposals": {
      "type": "array",
      "minItems": 1,
      "maxItems": 2,
      "items": {
        "type": "object",
        "required": ["role", "representation", "description"],
        "properties": {
          "role": {
            "type": "string",
            "enum": ["source", "destination", "target"]
          },
          "representation": {
            "type": "string",
            "enum": ["semantic", "bbox", "point"]
          },
          "semantic_id": {"type": "string"},
          "camera": {"type": "string", "enum": ["static", "wrist"]},
          "bbox_xyxy": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"}
          },
          "point_xy": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number"}
          },
          "description": {"type": "string"}
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

JSON Schema 的跨字段约束由 Runtime 补充验证：`semantic` 必须有 `semantic_id`，`bbox` 必须有 camera 和 bbox，`point` 必须有 camera 和 point。

### 第一版 CALVIN semantic vocabulary

至少包括：

```text
block_red
block_blue
block_pink
drawer
drawer_interior
slider_left_door
slider_right_door
slider_interior
light_switch
led_button
```

具体 vocabulary 由 backend capability/config 发布，不硬编码在 Qwen tool 实现里。

### 成功结果

```json
{
  "skill_id": "stack_block",
  "observation_id": "obs_000117",
  "groundings": [
    {
      "grounding_id": "ground_000031",
      "role": "source",
      "semantic_id": "block_red",
      "source": "calvin_gt",
      "mask_handle": "mask://.../ground_000031",
      "confidence": 1.0
    },
    {
      "grounding_id": "ground_000032",
      "role": "destination",
      "semantic_id": "block_blue",
      "source": "calvin_gt",
      "mask_handle": "mask://.../ground_000032",
      "confidence": 1.0
    }
  ],
  "preview": "artifact://.../grounding_preview.png"
}
```

## 6. 工具三：execute_skill

用途：在当前场景执行一次有界原子技能 attempt。它是唯一调用 DiT action loop 的工具。

### 参数 Schema

```json
{
  "type": "object",
  "required": ["skill_id", "grounding_ids", "intent"],
  "properties": {
    "skill_id": {"type": "string"},
    "grounding_ids": {
      "type": "array",
      "items": {"type": "string"},
      "uniqueItems": true
    },
    "intent": {
      "type": "string",
      "description": "Short summary of why this atomic skill advances the user goal."
    }
  },
  "additionalProperties": false
}
```

Qwen 不能传 checkpoint、endpoint、attempt number、seed 或任意 `max_steps`。这些由 Registry 和 Runtime 决定。

### Runtime 校验

- skill 属于 34-skill Registry；
- grounding role 满足该 skill 的 role contract；
- source 与 destination 不是同一实例；
- grounding 未过期且 mask 非空；
- StateInspector 前置条件成立；
- 当前逻辑子任务 attempt 未超过 3；
- policy endpoint 健康；
- goal 预算未耗尽。

### 成功或正常失败结果

```json
{
  "skill_id": "lift_red_block_table",
  "logical_subtask_id": "subtask_003",
  "attempt": 2,
  "policy_id": "general",
  "success": false,
  "termination_reason": "oracle_not_satisfied",
  "steps": 150,
  "verification": {
    "backend": "calvin_oracle",
    "status": "failure",
    "reason": "predicate_not_satisfied"
  },
  "scene_continuation": true,
  "before_observation_id": "obs_000117",
  "after_observation_id": "obs_000267",
  "state_delta": {
    "gripper": "open",
    "holding_object": "false",
    "target_moved": "unknown"
  },
  "recommended_actions": [
    "inspect_scene",
    "ground_targets",
    "approach_target",
    "replan"
  ]
}
```

工具返回 `failed` 不等于系统异常。Qwen 应基于 after observation 和 failure data 做下一轮决策。

## 7. 工具四：approach_target

用途：把末端执行器移动到某个 grounded target 附近，作为 Qwen 可选择的恢复动作。第一版只追求接近目标，不统计训练数据 approach pose。

### 参数 Schema

```json
{
  "type": "object",
  "required": ["grounding_id", "intent", "failed_skill_id"],
  "properties": {
    "grounding_id": {"type": "string"},
    "failed_skill_id": {
      "type": "string",
      "description": "Current rotate skill with at least one failed attempt."
    },
    "intent": {"type": "string"},
    "xy_threshold_m": {
      "type": "number",
      "minimum": 0.02,
      "maximum": 0.20,
      "default": 0.08
    }
  },
  "additionalProperties": false
}
```

Runtime 对阈值设上下限，并可使用 Registry 默认值覆盖不合理请求。

该实验工具默认关闭，只有 CALVIN Session 显式传入 `--enable-approach`（批测入口为
`--approach`）才会注册。启用后，当前 CALVIN Qwen Session 将其作为可插拔的 rotate recovery：第一次 rotate
attempt 前禁止调用；grounding 必须是当前 rotate 对应的同色 source cube，且 cube
仍由桌面支撑。控制器使用 episode 初始、与训练分布一致的 top-down 末端姿态，保持
安全悬停高度并打开夹爪。距离已经足够近时仍会校正姿态，但不会做无意义的 XY 移动。
approach 后旧 grounding 失效，必须重新 inspect 和 ground 才能重试 DiT。

### 结果

```json
{
  "grounding_id": "ground_000031",
  "failed_skill_id": "rotate_red_block_left",
  "failed_attempts": 1,
  "success": true,
  "steps": 23,
  "initial_xy_distance_m": 0.31,
  "final_xy_distance_m": 0.06,
  "threshold_m": 0.08,
  "motion_metadata": {
    "reference_orientation_euler": [3.06, 0.0, 1.46],
    "final_orientation_error_rad": 0.04
  },
  "after_observation_id": "obs_000290",
  "scene_continuation": true
}
```

`approach_target` 改变环境，因此会消耗 environment step 预算，并使无 tracker 的旧 grounding 可能失效。

## 8. 工具五：verify_goal

用途：验证整个用户 goal 是否完成。Qwen 在准备结束任务前必须调用；Runtime 只接受 verifier 的成功结果作为终态成功。

### 参数 Schema

```json
{
  "type": "object",
  "required": ["claim", "observation_id"],
  "properties": {
    "claim": {
      "type": "string",
      "description": "Concise description of the final state believed to be satisfied."
    },
    "observation_id": {"type": "string"},
    "use_goal_reference": {
      "type": "boolean",
      "default": true
    }
  },
  "additionalProperties": false
}
```

### 结果

```json
{
  "goal_id": "goal_001",
  "status": "success",
  "backend": "calvin_oracle",
  "reason": "all_required_predicates_satisfied",
  "confidence": 1.0,
  "observation_id": "obs_000412",
  "goal_reference_used": true,
  "evidence": {
    "predicates": [
      {"name": "place_in_drawer", "satisfied": true}
    ]
  }
}
```

Verifier 状态取值：

```text
success | failure | uncertain
```

第一版使用 CALVIN Oracle 时 `goal_reference_used` 表示参考图是否同时作为 Qwen/UI 证据展示，不表示 Oracle 通过图像进行判断。后续视觉 verifier 才真正比较 current view 与 goal reference。

## 9. 工具六：abort_goal

用途：当 Qwen 判断任务无法安全或合理继续时，显式请求终止 goal。它不把失败伪装成成功，也不重置环境。

### 参数 Schema

```json
{
  "type": "object",
  "required": ["reason_code", "summary"],
  "properties": {
    "reason_code": {
      "type": "string",
      "enum": [
        "task_infeasible",
        "required_skill_unavailable",
        "repeated_execution_failure",
        "unsafe_to_continue",
        "insufficient_observation"
      ]
    },
    "summary": {"type": "string"},
    "last_observation_id": {"type": "string"}
  },
  "additionalProperties": false
}
```

Runtime 可以接受该请求并标记 `ABORTED_BY_AGENT`；若安全停止本身失败，则升级为 `SAFETY_STOP` 或 `DEPENDENCY_FAILURE`。普通自然语言答复不能代替 `abort_goal` 或 `verify_goal` 改变 session 终态。

## 10. Tool lifecycle

一次典型 session：

```text
1. Runtime 创建 GoalInput 和初始 ObservationRef
2. Qwen-Agent 接收 goal、双相机图像和可选参考图
3. Qwen 调 inspect_scene（若需要刷新/读取状态摘要）
4. Qwen 选择一个原子 skill 和具体对象
5. Qwen 调 ground_targets
6. Qwen 调 execute_skill
7. Runtime 返回新 observation 和 verifier 结果
8. Qwen 决定下一 skill、重新 ground、approach 或 replan
9. Qwen 认为整体完成后调 verify_goal
10. verify_goal=success 后 Runtime 将 session 标记 SUCCEEDED；无法继续时调用 abort_goal
```

Qwen 可以跳过不必要的 `inspect_scene`，直接使用上一个工具返回的最新 observation。不能跳过 grounding 执行需要角色提示的 DiT 技能。

## 11. 幂等性、并发和状态版本

- `inspect_scene`、`ground_targets` 和 `verify_goal` 是只读工具；
- `execute_skill` 和 `approach_target` 会改变环境，必须串行；
- 每个调用携带 Runtime 生成的唯一 `tool_call_id`；
- 同一个改变环境的 `tool_call_id` 重复提交时返回第一次结果，不重复执行；
- Runtime 用单调递增 environment revision 检测 stale observation/grounding；
- Web UI 中途更新 goal 时递增 `goal_version`，旧版本未执行的工具调用必须被拒绝。

## 12. 预算与终止

第一版默认值：

```text
skill_attempt_limit = 3
replan_limit = 2
tool_call_limit = 32
environment_step_limit = 2000
```

预算由 Runtime 维护并随每次工具结果返回。官方序列中每个当前 goal stage 独立拥有
`skill_attempt_limit` 次机会：换 grounding 不刷新，成功进入下一 stage 后刷新。恢复
skill 的额度与未来正式 stage 隔离。当前 stage 的第 3 次 attempt 仍失败会立即进入
`BUDGET_EXHAUSTED`，而不是只拒绝第 4 次调用或允许跳到后续 stage。Qwen 不能通过新
grounding handle、改变自然语言 intent 或重建 policy session 绕过该限制。

终态：

```text
SUCCEEDED               verify_goal 返回 success
ABORTED_BY_AGENT        Qwen 明确放弃且 Runtime 接受
ABORTED_BY_USER         用户从 UI 取消
BUDGET_EXHAUSTED        任一硬预算耗尽
DEPENDENCY_FAILURE      环境/必要服务不可恢复
SAFETY_STOP             Runtime 安全检查触发
```

## 13. 日志事件

每个工具调用至少记录：

```text
agent_turn_started
agent_decision_summary
tool_call_requested
tool_call_validated | tool_call_rejected
tool_call_started
tool_call_succeeded | tool_call_failed
observation_created
grounding_created | grounding_invalidated
policy_routed
verification_result
budget_updated
goal_succeeded | goal_aborted
```

事件不得记录模型隐藏思维链。记录模型输出的工具调用、最终答复和单独生成的简短决策摘要即可。

## 14. 第一版工具测试矩阵

### inspect_scene

- 返回两张非空 RGB 图；
- observation ID 单调更新；
- 不消耗 environment steps；
- 状态缺失时返回 `unknown`，不伪造值。

### ground_targets

- 红/蓝/粉方块 semantic grounding；
- drawer、slider、按钮和开关 grounding；
- source/destination role 正确；
- 空 mask、未知 semantic ID 和 stale observation 被拒绝；
- preview 与 mask 可视化一致。

### execute_skill

- 单 skill 单 policy；
- 四条 policy route；
- outlined 图确实发送给 DiT；
- Oracle 成功立即停止；
- timeout 返回正常 failure；
- 连续场景最多三次；
- attempt 上限不可绕过。

### approach_target

- 初始距离大于阈值时接近；
- 已在阈值内时零步成功；
- 不可达、碰撞或 grounding 失效时返回结构化失败。

### verify_goal

- 已满足 goal；
- 未满足 goal；
- uncertain/依赖不可用；
- 存在目标参考图和不存在参考图；
- Qwen 不能绕过 verifier 直接把 session 标记成功。

### abort_goal

- 主动终止不会调用环境 reset；
- 原因和最后 observation 被记录；
- 普通文本输出不能改变 session 终态；
- 已成功或已终止的 session 拒绝重复终止。
