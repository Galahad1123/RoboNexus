# CALVIN Push+Rotate 12

Audited specialist dataset containing six push and six rotate tasks. See
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md) for
construction rules and commands.

Final audited counts:

| Split | Raw annotations | Overlap clusters | Retained episodes | Frames |
|---|---:|---:|---:|---:|
| training | 1,801 | 728 | 552 | 35,022 |
| validation | 353 | 123 | 94 | 5,933 |

All retained episodes pass the corresponding CALVIN oracle after at most a
0.5-second recorded boundary repair. Every push episode starts and ends on the
table support under the oracle's same-contact constraint.

## 闭环评测

启动对应 checkpoint 的策略服务后，在 RoboNexus 根目录运行：

```bash
conda run -n robonexus-calvin python \
  datasets/calvin_push_rotate_12/scripts/evaluate.py \
  --episodes-per-task 10 \
  --replan-interval 16 \
  --mask-provider gt \
  --save-video \
  --output-dir outputs/calvin_push_rotate_12_eval/checkpoint_STEP_r16
```

suite 默认使用与训练数据一致的 2 px 黑色目标边框。
