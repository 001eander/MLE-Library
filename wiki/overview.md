---
title: MLE Library 概览
type: overview
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
---

# MLE Library

由 Pi 图书馆员在人工监督下维护的个人机器学习工程知识库。

## 用途

- 学习新的机器学习工程概念，并复习实用指导。
- 为 MLE AutoResearch Agent 提供对精选知识和来源证据的只读访问。

## 分层结构

- `raw/` 保存不可变的来源材料。PDF、剪藏文章和原始资产绝不在原处编辑。
- `wiki/` 保存由图书馆员维护的 Markdown 页面。重要论述会链接到来源页面和精确的来源位置。
- `AGENTS.md` 和 `.agents/skills/` 定义图书馆员如何收录、查询、维护和提交知识。

## 阅读路径

从 [[index|知识索引]] 开始。主题页面位于 `concepts/`，工程指导位于 `practices/`，特定工具的笔记位于 `tools/`，来源摘要位于 `sources/`。研究草稿保留在 `research/`，直到有人决定将其提升为稳定知识。

目前收录六个来源：[[aide|AIDE（arXiv 2502.13138v1）]]、[[alphaevolve|AlphaEvolve 官方博客]]、[[autoresearch|autoresearch 仓库 README]]、[[hyra-1-0|Hyra-1.0 官方博客]]、[[hyra-2-0|Hyra 2.0 官方博客]]与 [[kaggle-getting-started|Kaggle 入门指南（知乎专栏）]]。AIDE 把机器学习工程写成代码空间的解树搜索，是本库唯一同时提供公开实现与第三方基准（MLE-Bench、RE-Bench）的来源；AlphaEvolve 用程序数据库与进化选择搜索整个代码库，在数据中心、芯片、训练内核与数学问题上给出生产或发现结果（全部为厂商自述）；autoresearch 把循环缩到最小——人写 `program.md`、智能体只改 `train.py`、每次实验固定 5 分钟——是另一份有公开实现的来源（但只有仓库进度图，无第三方基准）；Hyra-1.0 展开了「单循环 + 经验库 + 双层评估器」的解题范式，2.0 把它推进到多项目长周期的工具研发，并新增分层记忆与知识分级。Kaggle 入门指南是唯一以**人类选手**为对象的来源，描述七段竞赛流程与人类如何判断自己的改进是否可信，为前五个智能体来源提供了评估尺度的另一侧。五种 harness 形态的对照见 [[research-agent-harness]]，搜索与记忆机制的概念页为 [[code-space-search]]，Hyra 两代差异见 [[hyra-1-0-vs-2-0]]。随着所有者添加资料和问题，知识库的范围将逐步扩大。
