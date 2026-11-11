---
title: 科研智能体 Harness 设计（轻量框架 + 宽广动作空间）
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - agent-harness
sources:
  - "[[hyra-1-0]]"
status: draft
verified: 2026-11-11
applies_to:
  - Hyra-1.0（来源描述的设计）
---

# 科研智能体 Harness 设计（轻量框架 + 宽广动作空间）

面向「可执行、可评估、可迭代」任务搭建研究型 agent 时的工程指导。以下先给来源论述，再给本库的落地建议；两者明确分开。

## 来源论述：Hyra 的做法

1. **原则对齐 The Bitter Lesson**：框架尽量轻量简单，智能体的动作空间尽量广阔（[[hyra-1-0]], section "Hyra Harness"，原文 `raw/articles/Hyra.md:25`）。
2. **统一循环**：给定任务描述后持续「基于历史经验提出更好的 solution」，经验包含过往运行日志、评估器反馈与源码；终止条件为 Agent 主动结束或预算耗尽，最终返回历史最优方案（同上）。
3. **角色分离**：
   - Context Agent 维护 [[experience-bank]]，把 solution 与评估结果入库，并合成灵感投入任务队列（`raw/articles/Hyra.md:27`）。
   - 多个 Proposal Agent 各自领取灵感、写出 `solution/`（入口脚本 `solve.sh`）、在**全新沙盒**中运行与评分（`raw/articles/Hyra.md:28`）。
4. **异步生产者-消费者**：Context Agent 填队列、Proposal Agent 抽队列，结果回写 EB；用信号量控制沙盒、模型推理与队列资源「健康地满载」（`raw/articles/Hyra.md:30`）。
5. **无评估器时双层循环**：内层优化 solution，外层优化评估器（`raw/articles/Hyra.md:32`，详见 [[evaluator-solution-co-evolution]]）。

## 可迁移的工程要点（综合）

- **动作空间优先于流程复杂度**：如果 agent 能直接写代码、跑实验、读日志，就不必为每类任务设计专用 DSL；轻量 harness 的代价是必须把「评估与沙盒」做扎实。
- **解耦思考与执行**：proposal 生成（模型调用，贵）与沙盒执行（CPU/GPU，贵但可并行）分别限流，避免一方空转。信号量应至少覆盖三类资源：并发沙盒数、模型并发数、队列长度。
- **每个 solution 都在干净沙盒里跑**：共享工作区会让「成功」混入未记录的副作用，也让失败无法归因。
- **入口约定固定**（如 `solve.sh` + `solution/` 目录）：评估器只需面向一个稳定契约，proposal 侧可自由换实现语言与结构。
- **终点判定显式化**：Agent 主动结束 + 预算耗尽两条路径都要有，否则容易出现无限微调或过早停止。同时要定义「历史最优」的选取口径（同一评估器版本内比较）。
- **保留可重放证据**：日志、源码、评估器版本三者一起入库，否则最优结果无法复算。

## 适用与不适用

- 适用：有明确执行入口与可自动打分（或可构建打分）的任务——训练 recipe 搜索、kernel 优化、组合优化、结构化产物生成。
- 谨慎：评估成本高于生成成本的任务（每轮评估都很贵时，异步流水线的收益被稀释）。
- 不适用/需改造：目标只能由人判断、且反馈稀疏的场景；此时外层循环要引入用户反馈，并接受偏好漂移风险。

## 版本与时效

- `verified: 2026-11-11`；`applies_to: Hyra-1.0`。本页描述的是来源博客表述的 Hyra-1.0 设计，且是单厂商的 demo 阶段经验。
- `raw/articles/` 中存在「Hyra 2.0」剪藏但尚未收录；若其设计有变化，需复核本页。

## 待解问题

- 灵感合成的具体策略（如何去重、如何保证多样性）未披露。
- 资源限流的实际参数与瓶颈位置未披露，因此「高效扩展」缺少可复现依据。
- 轻量 harness 在长周期任务（数天级）上的稳定性未讨论。

## 关联页面

- [[hyra-1-0]]、[[hyra]]、[[experience-bank]]、[[evaluator-design-and-reward-hacking]]、[[recursive-self-improvement]]
