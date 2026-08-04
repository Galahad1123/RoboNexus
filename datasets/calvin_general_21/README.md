# CALVIN General-21

覆盖 CALVIN-D 中除 Push、Rotate、Stack 之外的21类任务。操作对象使用黑色 outline，
放置目标使用青色 outline + 白色 halo。训练路由与 Stack 辅助样本保持分离：Lift 和
Place 属于 General，但可在 Stack specialist 中作为辅助训练数据重复出现。

当前审计保留 training 2,203 条、validation 433 条 oracle-consistent 去重轨迹。构造、
mask cache 和转换命令见 [`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md)。
