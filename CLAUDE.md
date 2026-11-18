# 给 Claude Code 的说明

这个 repo 是 skill + 固定版本的 OneJev + `ojh` 脚本：用户给**数据位置、GPU、基座**，agent 把数据训练成 OneJev 决策模型。

- 入口 skill：`training-with-onejev`；认数据和写 `exp.yaml` 用 `adapting-data-for-onejev`（`.claude/skills/` 里是指向 `skills/` 的符号链接）。主路径：`ojh inspect` → 问用户/写适配器 → `exp.yaml` → `ojh run` → 交回、不轮询 → 读 `RESULT.md`。
- 基座、卡、训练方式、标签含义由用户定；不可行或校验失败时报原因、问用户，不自动切换。
- 实验工作区放在 repo 之外；`eval-report`、`--allow-test`、`cleanup --confirm`、任何 override、`git push` 都先问用户。同样的说明在 `AGENTS.md`，设计见 `docs/design.md`。
