# 给 Claude Code 的说明

这个 repo 让 agent 只靠它就能把一份数据在单机 GPU 节点上训练成 OneJev 决策模型。先读 `skills/` 下的两个 skill（`.claude/skills/` 里是指向它们的符号链接）：`training-with-onejev`（训练全流程、判断点、审批规则、故障表）和 `adapting-data-for-onejev`（数据不是 OneJev 行时写适配器）。所有参数来自机器配置、实验配置和 lock，不要写死路径或机器取值；删 checkpoint、读 report/test、推送 repo、全参与 LoRA 切换都要先问用户。设计见 `docs/design.md`，同样的说明也在 `AGENTS.md`。
