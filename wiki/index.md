---
title: 知识索引
type: index
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - index
---

# 知识索引

每当新增、重命名或实质修改页面时，图书馆员都会更新此目录。每个条目包含一个链接和一句话摘要。

## 开始

- [[overview|MLE Library 概览]] — 用途、分层结构与阅读路径。
- [[catalog|动态目录]] — 使用 Dataview 展示 wiki 页面及其时效性元数据的表格。

## 来源

- [[aide|AIDE：AI-Driven Exploration in the Space of Code（arXiv 2502.13138v1）]] — 把 MLE 改写为代码空间的解树搜索；自建 Weco-Kaggle 与 MLE-Bench / RE-Bench 汇总，含「4 倍」口径与 63 vs 80 竞赛列表的内部不一致。
- [[hyra-1-0|Hyra-1.0：腾讯混元的递归自我改进科研智能体（官方博客）]] — 轻量 harness + 经验库 + 双层评估循环，覆盖 AI for AI / Science / Fun 十余场景的自评结果。
- [[hyra-2-0|Hyra 2.0：从解题到造工具（腾讯混元官方博客）]] — 多项目长周期 harness（Overall/Mission/Worker）、分层记忆与知识分级，以及 HyInv、HyPLM、HyFF、HyProver、HyQEC、旋梯手、香山七组成果。

## 概念

- [[code-space-search|代码空间搜索与解树（Code-Space Search）]] — 解即程序、边即改进尝试；原子改动、`Σ(T)` 摘要、树 vs 种群 vs 经验库的取舍与失败模式。
- [[tool-centric-research|工具中心的研究（Tool-Centric Research）]] — 把产出单位从「答案」换成「工具」：符号化/参数化载体、通用性阶梯与摊销式经济性。
- [[recursive-self-improvement|递归自我改进（RSI）]] — scaffold–data–model 闭环、四层循环及其失败模式、以瓶颈分类代替能力上限叙事。
- [[research-memory-tiers|研究记忆分层与知识成熟度分级]] — 项目状态／agent 记忆／跨任务 Books 三层，观察/已验证/约定三级，以及 skill 沉淀。
- [[experience-bank|经验库（Experience Bank）]] — 结构化记录 solution/日志/评分，并合成为多样化「灵感」的机制；含与 2.0 Books 的对照。
- [[evaluator-solution-co-evolution|评估器与解的共进化]] — 无现成评估器时的双层循环，以及 2.0 把外层折成独立复核与人工审批后的新瓶颈。

## 实践

- [[research-agent-harness|科研智能体 Harness 设计]] — 三种形态：树搜索脚手架（AIDE）、单任务搜索循环（Hyra 1.0）与多项目长周期组织（2.0，含队列、唤醒、看板、审批门与证据要求）。
- [[evaluator-design-and-reward-hacking|评估器设计与 reward hacking 防护]] — 1.0 的两个作弊案例、2.0 的证据提交与独立复核，以及 AIDE 暴露的「在固定验证划分上搜索」这类非恶意失效。

## 工具

- [[aide-scaffold|AIDE 脚手架（aideml）]] — 开源脚手架：解树、三算子、摘要算子的实现形态，作为基线的用法与版本风险。
- [[hyra|Hyra（Hunyuan Research Agent）]] — 腾讯混元科研智能体的双版本页：1.0 单循环架构、2.0 多项目架构、两代结果表与使用边界。

## 比较

- [[hyra-1-0-vs-2-0|Hyra 1.0 与 2.0 对照：解题范式 vs 工具范式]] — 逐维度对照两版的目标、组织结构、记忆、验证方式、结果形态与证据强度，附阅读顺序建议。

## 研究

- [[automated-research-open-questions|自动化研究待验证问题]] — AIDE 与 1.0/2.0 三代来源中结果可信度、机制假设、工程风险与待补充资料清单。
