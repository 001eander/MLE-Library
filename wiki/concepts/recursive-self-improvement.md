---
title: 递归自我改进（Recursive Self-Improvement, RSI）
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - rsi
sources:
  - "[[hyra-1-0]]"
status: draft
---

# 递归自我改进（Recursive Self-Improvement, RSI）

指系统把自身的改进过程（scaffold、数据、模型、评估）纳入同一个可迭代循环，使每一轮产出成为下一轮的输入，从而形成自我加速的研究闭环。在本库中，这个概念目前主要依据腾讯混元 Hyra-1.0 博客的表述（[[hyra-1-0]]），属于**来源论述 + 综合**，尚无同行评审论文支撑。

## 来源论述

- 博客把递归自我改进（RSI）与自动化研究称为过去一年 AI 领域最活跃的前沿之一，并引 AlphaEvolve（4×4 复数矩阵乘法只需 48 次标量乘法）、Together AI 的多智能体数学探索（11 维 kissing number 下界 593→604）、Karpathy 的 autoresearch 作为同类进展（[[hyra-1-0]], section "概要"，原文 `raw/articles/Hyra.md:15-19`）。这些数字在本库中**仅作为博客的转述**记录，未核对原始出版物。
- Hyra 把「基模研发」视为 RSI 的现实起点，理由是训练 recipe、数据策略、训练/推理系统、底层算子与评估体系都具备「可执行、可评估、可迭代」的性质（[[hyra-1-0]], section "AI for AI"）。
- 博客提出的闭环表述是 **scaffold–data–model 进化循环**：更好的 scaffold 产出更好的数据与经验，锻造更强的模型，更强模型再驱动更强的 scaffold 与更多发现；作者称新一代 Hy 模型乃至腾讯部分产品会与 Hyra 持续共进化（[[hyra-1-0]], section "The path forward"）。

## 综合

从 Hyra 的工程描述可以提炼出三类循环，它们在时间尺度上不同：

| 循环 | 一轮的内容 | 反馈来源 | 失败模式 |
| --- | --- | --- | --- |
| solution 循环（内层） | 由灵感生成一个新 solution 并运行 | 固定评估器 | 局部最优、探索方向同质化 |
| 评估器循环（外层） | 修改评估器再开新一轮 | EB 中的历史经验 | reward hacking、评估器与目标脱节 |
| scaffold–data–model 循环（最外层） | 用 agent 产出的经验训练更强模型 | 下游任务收益 | 收益归因困难、算力成本上升 |

（综合 [[hyra-1-0]], section "Hyra Harness" 与 section "The path forward"；这是本库的整理，不是来源原话。）

## 假设（待验证）

- 假设 1：在**可自动评估**的任务上，RSI 的收益主要来自搜索广度与经验复用，而非单次 proposal 质量。
- 假设 2：最外层 scaffold–data–model 循环存在收益递减，因为评估器质量与验证成本会成为瓶颈。
- 假设 3：同一套轻量 harness 能在训练算法、训练系统与底层算子之间迁移，这一点依赖「反馈信号结构相似」而非任务领域相似（Hyra 的三项 AI for AI 结果可视为弱证据，但缺少消融）。

## 与其他概念的关系

- 落地形态：[[research-agent-harness]]（工程侧）、[[experience-bank]]（经验复用）、[[evaluator-solution-co-evolution]]（评估侧）。
- 反面约束：更快的搜索会更快撞上评估器边界，参见 [[evaluator-design-and-reward-hacking]]。

## 待解问题

- RSI 的收益如何与「底层模型本身的进步」分离？现有材料未给出消融。
- 循环终止条件（预算 vs 饱和 vs 评估器升级）如何选择？
- 当任务不可自动评估时，RSI 的闭环是否仍然成立？
