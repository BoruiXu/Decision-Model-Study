# 给 agent 的说明

这个 repo 是 skill + 固定版本的 OneJev + `ojh` 脚本：用户给**数据位置、GPU、基座**，agent 把数据训练成 OneJev 决策模型。

- 入口 skill：`skills/training-with-onejev/SKILL.md`。主路径：确认三样输入 → 认数据、写 `exp.yaml`（`skills/adapting-data-for-onejev/SKILL.md`，从 `ojh inspect` 开始）→ `ojh run exp.yaml --base … --gpus …` → 交回、不轮询 → 读 `RESULT.md` 汇报。
- 基座、卡、训练方式、标签含义由用户定；不可行或校验失败时报原因、问用户，不自动切换。
- 实验工作区（`exp.yaml`、适配器）放在 repo 之外；不要在 repo 代码里写路径或机器取值。
- 先问用户：`eval-report`、`--allow-test`、`cleanup --confirm`、任何 override、`git push`。设计见 `docs/design.md`。
