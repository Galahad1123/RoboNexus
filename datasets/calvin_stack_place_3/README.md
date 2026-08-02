# CALVIN Stack+Place 3

Audited destination-conditioned specialist dataset containing `stack_block`,
`place_in_drawer`, and `place_in_slider`. See
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md) for
construction rules and commands.

Final audited counts:

| Split | Raw annotations | Overlap clusters | Retained episodes | Frames |
|---|---:|---:|---:|---:|
| training | 454 | 370 | 325 | 14,527 |
| validation | 88 | 65 | 60 | 2,626 |

Training contains 69 stable stack episodes covering all six ordered
source-to-destination color pairs. Validation contains 13 stable stack episodes.
Only the unique destination is outlined; the held source is left unchanged.

## 闭环评测

启动对应 checkpoint 的策略服务后，在 RoboNexus 根目录运行：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_stack_place_3/scripts/evaluate.py \
  --episodes-per-task 10 \
  --replan-interval 16 \
  --mask-provider gt \
  --save-video \
  --output-dir outputs/calvin_stack_place_3_eval/checkpoint_STEP_r16
```

suite 会逐 episode 读取审计 manifest 中的唯一 destination，并使用与训练数据一致的
2 px 青色边框和 1 px 白色外描边；夹爪当前抓持的 source 不会被标记。
