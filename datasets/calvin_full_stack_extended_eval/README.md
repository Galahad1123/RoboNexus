# CALVIN Full Stack Extended Eval

仅用于评测，不可用于训练。它将 Persistent Stack validation 中17条 Held-start 轨迹在
同一连续专家轨迹内回溯到 source 有支撑且夹爪未持物的状态，加上4条官方 Full Stack，
形成21个 Full Pick-and-Stack 初始状态。报告时应将 official 与 extended_backtracked
分层呈现，不与官方 validation 指标混算。
