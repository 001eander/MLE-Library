---
title: Hyra（Hunyuan Research Agent）
type: tool
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
  - Hyra-1.0
---

# Hyra（Hunyuan Research Agent）

腾讯混元提出的通用科研智能体（Research Agent）框架，首个版本 Hyra-1.0 于官方博客发布（[[hyra-1-0]], section "概要"）。定位不是单一基准上的求解器，而是「把智能和算力转化为真实且实用的收益」的通用 harness：同一套循环用于模型研发（AI for AI）、科学发现（AI for Science）与开放创作（AI for Fun）。

## 架构要点

| 组件 | 职责 |
| --- | --- |
| Context Agent | 维护经验库（Experience Bank, EB）；把 solution 与评估结果写入 EB；从 EB 整合「灵感」（inspirations）投入任务队列 |
| Proposal Agent（多个） | 领取一份灵感，反思后写出新 solution（含 `solve.sh` 的 `solution/` 目录），在全新沙盒中执行并评分 |
| 任务队列 + 信号量 | 异步生产者-消费者流水线，控制沙盒、模型推理、队列等资源保持满载 |
| 评估器（可选双层） | 任务未提供评估器时，先由 Hyra 设计评估器；内层优化 solution，外层再优化评估器 |

（来源论述，[[hyra-1-0]], section "Hyra Harness"；原文 `raw/articles/Hyra.md:25-32`。）

设计原则显式对齐 The Bitter Lesson：框架尽量轻量简单、动作空间尽量广阔（[[hyra-1-0]], section "Hyra Harness"）。

## 文中报告的适用场景与结果

| 场景 | 任务 | 报告结果 |
| --- | --- | --- |
| AI for AI | NanoChat Autoresearch | Validation BPB 0.9015（对照 Recursive 0.9109） |
| AI for AI | NanoGPT Speedrun | Time to 3.28 loss 76.4s（对照 77.5s） |
| AI for AI | SOL-ExecBench | Mean SOL 0.771（对照 0.754，†本地运行、Evaluation Stack v1.0） |
| AI for Science | 55 个数学开放问题 | 29 个刷新历史最好结果 |
| AI for Science | 太阳黑子递推公式 | 1932–2026 样本外 $R^2 = 0.77$ |
| AI for Science | 10 位数加法 Transformer | 15 个可训练参数（Adderboard 纪录 36 个，减少 58.3%） |
| AI for Science | 量子比特路由（IBM Q20） | 相比 SABRE 路由效率 +44.4% |
| AI for Fun | 黑白棋 bot（Botzone） | 730 个参赛程序中排名第 3 |
| AI for Fun | 2D 参考图 → 3D 模型 | rubrics VLM judge；文中称优于 Claude Code goal 模式 |
| AI for Fun | 主旋律配器 | 7 轮进化，规则内层 + 用户反馈外层 |

（来源论述，[[hyra-1-0]], section "关键主张"；原文 `raw/articles/Hyra.md:38-97`。）

## 使用时需要注意的边界

- **版本敏感**：以上能力描述仅对应 Hyra-1.0 的官方博客表述（verified 2026-11-11）。框架细节、可用接口与结果集都可能随版本变化；`raw/articles/` 中已有「Hyra 2.0」剪藏待收录，届时需重新核验本页。
- **开放程度未说明**：文中称 demo 产物在 GitHub 仓库 `Tencent-Hunyuan/hyra-results` 开源，但未说明 harness 本身是否可获取、许可方式与运行成本。
- **自评结果**：所有对比数字来自作者自述，缺少第三方复现与算力披露（详见 [[hyra-1-0]], section "局限与阅读提示"）。
- **非标量任务的评估依赖外部信号**：AI for Fun 类任务依靠对手池、VLM judge 或用户反馈，属于「评估器也是被优化对象」的场景，见 [[evaluator-design-and-reward-hacking]]。

## 与其设计最相关的可复用经验

1. 把「经验」显式结构化：EB + 灵感合成，而不是把所有历史塞进上下文（见 [[experience-bank]]）。
2. 用异步生产者-消费者 + 信号量把「思考」和「执行」解耦，提高沙盒与推理资源利用率（见 [[research-agent-harness]]）。
3. 无评估器时先造评估器，并让评估器随搜索一起升级（见 [[evaluator-solution-co-evolution]]）。
4. 预期 reward hacking，并把 eval/validation 当作研究循环的一等公民。

## 关联页面

- 来源：[[hyra-1-0]]
- 概念：[[recursive-self-improvement]]、[[experience-bank]]、[[evaluator-solution-co-evolution]]
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 待验证问题：[[automated-research-open-questions]]
