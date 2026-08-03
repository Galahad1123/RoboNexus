# CALVIN Rotate-6 Release

6 类完成真实释放并带 0.5～1 秒稳定后缀的 Rotate 专项数据集。

| Split | Episodes | Frames |
|---|---:|---:|
| training | 275 | 26,375 |
| validation | 45 | 4,308 |

完整审计标准和构造命令见
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md)。

## 闭环评测

启动对应 checkpoint 的策略服务后运行：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_rotate_6_release/scripts/evaluate.py \
  --episodes-per-task 10 --replan-interval 16 \
  --mask-provider gt --save-video \
  --output-dir outputs/calvin_rotate_6_release_eval/checkpoint_STEP_r16
```

默认每个 episode 最多执行 150 步，用于观察新模型能否完成旋转后的释放阶段。
成功条件为 CALVIN Rotate oracle 成立，并且夹爪连续 3 步发出 open command、开口宽度
不小于 0.04 m；因此不会把“角度已达到但仍未释放”计为成功。
