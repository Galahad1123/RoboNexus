# RoboNexus 展示图的数据链路依据

本文件依据本地代码静态阅读编写，没有运行模型、仿真或测试。

## 当前主链路

1. `RoboNexus-Agent/adapters/qwen_agent/single_skill_runner.py` 使用 Qwen-Agent Assistant，向模型提供任务、技能目录、当前静态与腕部图像。每轮选择工具，不输出底层机器人动作。
2. `core/tools/inspect_scene.py`、`ground_targets.py`、`execute_skill.py`、`verify_goal.py`、`abort_goal.py` 提供观察、语义目标绑定、执行、验证和终止接口。当前 CALVIN runner 提示词限定 semantic grounding，不应将早期 SAM/Cutie 设计画为必经链路。
3. `core/runtime/session.py` 与 `core/tools/execute_skill.py` 维护观测引用、技能调用和环境步数预算，检查技能与角色参数，再执行一次有界技能尝试。
4. `backends/calvin/execution/skill_executor.py` 通过 policy_router 选择策略，再由 client_pool 获取策略客户端，使用 CALVIN rollout 执行器与任务验证器驱动动作循环。CALVIN GT mask 与任务/goal verifier 属于仿真后端的一部分，不能描述为纯视觉无特权信息系统。
5. 执行结果含前后观测、步数、策略标识和失败信息；Agent 根据新图像继续选择技能、重试或终止。完整目标完成必须调用 verify_goal。
6. `core/logging/result_store.py` 持久化 JSON 结果；`scripts/agent/generate_eval_report.py` 提供报告入口。日志记录能力不等于本次恢复了历史结果。

## 图中抽象

任务与双视角观测 → Qwen-Agent / Qwen3-VL → Robot Tools & Runtime → Policy Router → DiT Policies → CALVIN。
CALVIN 的新观测与验证结果回传 Runtime，再回传 Agent，形成闭环。将模拟器辅助验证明确标出。
不展示未核实的成功率，不把早期架构草案中的可选感知服务画成当前必需模块。

## 素材检索

2026-11-17 查阅 p5000pro 的 ~/lizk/vlas/RoboNexus。服务器含大量未提交修改与已删除数据文件，未做修改或同步。排除虚拟环境、第三方源码和数据目录后的媒体检索只发现上游 multitask_dit_policy 模型架构图，未找到可用的本项目运行视频、GIF 或 JSONL。根目录 outputs/results/runs/artifacts 不存在。不据此断言整台服务器没有备份。

## 视觉参考

https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
仅参考其浅色分组、圆角容器、简洁图标和闭环箭头布局，不借用其方法或结果。
