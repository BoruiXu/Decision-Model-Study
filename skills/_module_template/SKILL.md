---
name: rsi-<模块名>
description: 由 rsi-orchestrator 派发<模块名>模块时使用：<一句话说明本模块做什么>。Use when the orchestrator dispatches the RSI <模块名> module; not for general <领域> requests.
---

# rsi-<模块名>：<模块中文名>

这是模块 skill 的模板，复制到 `skills/rsi-<模块名>/SKILL.md` 后替换尖括号里的内容。前四节由框架写好接口，其余六节由模块负责人填写。模块内部怎么做由负责人定，接口（输入、输出、门、单独运行）以顶层 `rsi-orchestrator` 的主流程表为准。

## 输入

列出产物类别（用反引号）和每个的用途；只读这里列出的产物，不读工作区和登记表。

## 输出

列出产物类别和每个的字段文档链接；每个产物写成输出目录下的一个子目录，manifest 的 `id` 写 null，`round` 按简报写。每个改进实验另交一份 `record`。

## 门

本模块交回前必须满足的检查，写明数值、阈值和谁来核对。每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验；报告里的指标、对照组和差值用 `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <标签行文件> --predictions <预测文件>` 算。模块只能运行这两个 `rsi` 命令。

## 单独运行

负责人在自己的沙盒里，用已登记产物的副本（路径手工指定）运行本模块，输出用 validate 检查，不需要顶层。

## 步骤和策略子 skill

由模块负责人填写：本模块的步骤和策略子 skill 清单，哪些步骤是脚本、哪些需要判断。

## 模块指标

由模块负责人填写：本模块自己的指标及测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)）。

## 长任务

由模块负责人填写：哪些步骤是长任务；在后台启动，写完成标记（第一行 ok 或 failed），交回标记路径后不轮询。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。
