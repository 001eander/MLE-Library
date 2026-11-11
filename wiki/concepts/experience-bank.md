---
title: 经验库（Experience Bank, EB）
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - memory
sources:
  - "[[hyra-1-0]]"
status: draft
---

# 经验库（Experience Bank, EB）

在自动化研究智能体里，EB 是一个显式的、结构化的「失败与成功的记忆」：记录每个被提出过的 solution、它的运行日志、评估器反馈与源码，并在需要时被合成为新的探索起点。本页主要依据 Hyra-1.0 的描述（[[hyra-1-0]]），属于**来源论述 + 综合**。

## 来源论述

- Context Agent 的职责之一是维护 EB 数据结构，把探索中提出的 solution 及其评估结果写入 EB，再从 EB 「整合出」一份份灵感（inspirations）送进任务队列（[[hyra-1-0]], section "Hyra Harness"，原文 `raw/articles/Hyra.md:27`）。
- 「灵感」被定义为一组上下文，可能包含 EB 中各种形式的代码、文件、产物、日志，作为启发出更好 solution 的经验信息（同上）。
- Context Agent 的目标是让灵感「尽可能多样化」，使多个 Proposal Agent 能朝不同方向探索（[[hyra-1-0]], `raw/articles/Hyra.md:28`）。
- 运行完毕的 solution 被提交回 EB，形成闭环；流水线用信号量控制资源满载（[[hyra-1-0]], `raw/articles/Hyra.md:30`）。

## 综合：EB 与「长上下文」的区别

| 方面 | 直接塞历史上下文 | 显式 EB + 灵感合成 |
| --- | --- | --- |
| 容量 | 受上下文窗口限制 | 可超出窗口，按需采样 |
| 多样性 | 由最近历史主导，易同质化 | 可以刻意采样不同分支/失败案例 |
| 可审计 | 难追溯某结论来自哪次运行 | 每条记录可关联日志与评分 |
| 成本 | 每轮重复付 token 费用 | 先摘要再注入，但多一层整理开销 |

（综合自 [[hyra-1-0]], section "Hyra Harness"；为本库整理，来源未做此对比。）

## 实践含义

- EB 的**写入口径**决定后续可复用性：只存分数不够，需同时存源码、日志与评估器版本，否则历史结果无法重放。
- 灵感合成是「检索 + 重写」而非简单拼接；如果只看高分 solution，搜索必然收敛到同一盆地，因此失败案例与低分分支同样需要入库。
- EB 也承担反 reward hacking 的取证作用：异常高分的 solution 应可被回溯审计（参见 [[evaluator-design-and-reward-hacking]]）。

## 假设（待验证）

- 假设：灵感多样性比灵感「质量」更影响长期搜索上限。
- 假设：EB 需要按评估器版本分区，否则评估器升级后旧记录会误导新轮次。

## 待解问题

- EB 的规模上限、淘汰策略与去重方式？来源未披露。
- 灵感由什么模型/规则合成？是否会被自身的 reward hacking 记录污染？
- 跨任务共享 EB 是否有效，还是任务间隔离更好？

## 关联页面

- [[hyra-1-0]]、[[hyra]]、[[research-agent-harness]]、[[recursive-self-improvement]]、[[evaluator-solution-co-evolution]]
