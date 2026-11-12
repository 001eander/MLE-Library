---
title: MLE Library 概览
type: overview
created: 2026-11-11
updated: 2026-11-12
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

建议先读：

1. [[cross-validation]] 与 [[cross-validation-design-and-tuning]]：先理解分数如何估计；
2. [[evaluation-metrics]] 与 [[metric-selection-and-thresholding]]：再确定分数本身该是谁；
3. [[hyperparameter-search]] 与 [[hyperparameter-search-and-tuning]]：最后分配搜索预算；
4. 按任务进入模型概念页及对应的选型实践页。

## 当前来源

目前收录 **25 个来源**。

### 自动化科研与竞赛

- [[aide|AIDE]]：把 MLE 写成代码空间的解树搜索，提供论文与公开实现，含基准和数字口径分歧。
- [[alphaevolve|AlphaEvolve]]：程序数据库 + 自动评估器，结果为厂商自述，无公开实现或版本。
- [[autoresearch|autoresearch]]：三文件契约、固定 5 分钟训练预算与单链贪心；进度图提示种子噪声风险。
- [[hyra-1-0|Hyra-1.0]]：单任务搜索循环、经验库与双层评估。
- [[hyra-2-0|Hyra 2.0]]：多项目长周期组织、分层记忆与知识分级。
- [[kaggle-getting-started|Kaggle 入门指南]]：人类选手的七段竞赛流程、Stacking 与本地 CV / 排行榜张力；代码属于旧 API 时代。

相关概念与实践：[[code-space-search]]、[[recursive-self-improvement]]、[[research-memory-tiers]]、[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]、[[leaderboard-vs-cv]]、[[competition-ml-workflow]]。

### scikit-learn 模型族与工具行为

- [[sklearn-linear-models|1.1 Linear Models]]：目标函数、正则与求解器。
- [[sklearn-lda-qda|1.2 LDA/QDA]]：生成式判别与协方差收缩。
- [[sklearn-svm|1.4 SVM]]：最大间隔、核与两种实现。
- [[sklearn-nearest-neighbors|1.6 Nearest Neighbors]]：实例模型、邻域、搜索结构与 NCA。
- [[sklearn-naive-bayes|1.9 Naive Bayes]]：条件独立、变体与平滑。
- [[sklearn-decision-trees|1.10 Decision Trees]]：递归划分、缺失值与结构剪枝。
- [[sklearn-ensembles|1.11 Ensembles]]：平均与顺序两类组合策略。
- [[sklearn-feature-selection|1.13 Feature selection]]：过滤、嵌入与包裹式选择。
- [[sklearn-neural-networks|1.17 Neural networks]]：MLP、非凸优化与求解器契约。

对应概念：[[regularization-and-sparsity]]、[[gaussian-discriminant-analysis]]、[[support-vector-machines]]、[[nearest-neighbors]]、[[naive-bayes]]、[[decision-trees]]、[[ensembles]]、[[feature-selection]]、[[multi-layer-perceptron]]。

### 无监督产物

- [[sklearn-manifold-learning|2.2 Manifold learning]]：输出新坐标，按被保量选方法。
- [[sklearn-clustering|2.3 Clustering]]：输出离散簇标签，按几何与密度选族。
- [[sklearn-decomposition|2.5 Decomposition]]：同时输出编码与成分，约束就是模型。
- [[sklearn-covariance|2.6 Covariance estimation]]：估计 $\Sigma$ 与精度矩阵 $K$，不是预测模型。
- [[sklearn-outlier-detection|2.7 Novelty and Outlier Detection]]：输出是否属于训练分布及阈值化标签。

对应概念：[[manifold-learning]]、[[clustering]]、[[matrix-factorization]]、[[covariance-estimation]]、[[anomaly-detection]]。

### 方法学与预处理

- [[sklearn-cross-validation|3.1 Cross-validation]]：分数如何估计，splitter 如何声明依赖结构。
- [[sklearn-hyperparameter-tuning|3.2 Hyperparameter tuning]]：网格、随机与逐次减半的预算曲线。
- [[sklearn-metrics|3.4 Metrics and scoring]]：目标泛函与严格一致评分；剪藏只有 4/54 小节，不能当指标定义大全。
- [[sklearn-preprocessing|8.3 Preprocessing data]]：有状态变换、类别编码、展开与目标编码交叉拟合。

对应概念与实践：[[cross-validation]]、[[hyperparameter-search]]、[[evaluation-metrics]]、[[feature-preprocessing]]、[[cross-validation-design-and-tuning]]、[[hyperparameter-search-and-tuning]]、[[metric-selection-and-thresholding]]、[[preprocessing-pipeline-design]]。

### 表格数据与模型比较

- [[ft-transformer|FT-Transformer]]：同协议比较 9 个表格 DL 模型与 2 个 GBDT；两个可复用基线（ResNet、FT-Transformer）、注意力特征重要性与合成 α 实验；含官方实现核查。

对应概念与实践：[[tabular-deep-learning]]、[[tabular-model-selection]]、[[gbdt-vs-tabular-dl]]。

工具总表见 [[scikit-learn]]，动态目录见 [[catalog]]。

## 证据与验证纪律

- **来源论述、综合与假设分开记录**；厂商自评、文档转述和本库推断不能混为事实。
- **来源缺陷保留**：多份 sklearn 剪藏丢失折叠块、参考文献或链接文字，各来源页均有转换损失说明。
- **版本敏感结论带范围**：API 默认值、弃用和运行时行为必须写版本。
- **不包含历史本机实验**：旧默认环境不完整且过旧，相关实验段、性能读数和派生数字已移除。
- **今后只在 `uv` 创建的正确版本环境中验证**：需记录 Python、scikit-learn、NumPy/SciPy 版本、数据、种子、脚本和实际环境；未经此流程的行为只标为待验证。
- 原始 `raw/` 来源不因清理而修改；历史日志保持追加-only，不重写当时的操作记录。

随着所有者添加资料和问题，知识库的范围将逐步扩大。
