---
title: 集成方法（Ensembles）：平均降方差、boosting 降偏差、元学习器与随机化预算
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - ensemble
  - gradient-boosting
  - random-forest
  - bagging
  - stacking
  - adaboost
sources:
  - "[[sklearn-ensembles]]"
  - "[[sklearn-outlier-detection]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1 的 ensemble 模块（文档与 API 签名，见 [[sklearn-ensembles]]）
---

# 集成方法（Ensembles）

## 两条轴：平均与顺序

来源给出的选型判据是：bagging 适合强而复杂的模型，boosting 适合弱模型（[[sklearn-ensembles]], L574）。

| | 平均式 | 顺序式 |
| --- | --- | --- |
| 代表 | Bagging、RF、Extra-Trees、Voting、Stacking | GBDT、HGBT、AdaBoost |
| 主要收益 | 降低方差 | 降低偏差 |
| 基模型 | 强、复杂、可独立训练 | 弱、偏差大、逐轮纠错 |
| 训练 | 可并行 | 每轮依赖上一轮，难并行 |
| 预算 | `n_estimators`、特征/样本随机性 | 轮数 × 学习率、树大小、早停 |

方差不是会降到零：若基模型误差相关，平均能消除的部分有限。Boosting 也不是轮数越多越好，后期会过拟合，需要早停或收缩。

## HGBT 与旧 GBDT：先选实现

`HistGradientBoosting*` 使用直方图分箱，`GradientBoosting*` 使用精确切分（[[sklearn-ensembles]], L22–L28、L249–L261）。

| | HGBT | `GradientBoosting*` |
| --- | --- | --- |
| 单节点切分 | $O(n_f n)$ | $O(n_f n\log n)$ |
| 缺失值 | 原生 | 不支持 |
| 类别特征 | 原生 | 不支持 |
| 单调/交互约束 | 支持 | 不支持 |
| 早停 | 大样本默认开启 | 默认关闭 |
| 小样本 | 分箱可能过粗 | 官方丢失的主题框建议优先考虑 |

`max_bins` 默认 255；源码注释中的 `max_bins + 1` 对应数值箱 + 缺失箱，因此「256 bins」与「255」不是同一计数口径。

## HGBT 的结构能力

- **缺失值**：每个阈值同时评估 NaN 左右方向；训练未见 NaN 时预测走样本更多分支（L76–L105）。
- **类别特征**：支持布尔掩码、索引、列名与 `from_dtype`；基数检查是 `>`，所以允许等于 `max_bins`（L140–L166，源码外部核查）。
- **样本权重**：乘梯度与 Hessian，但分箱阶段的分位数不看权重（L113–L136）。
- **单调约束**：只约束其他特征不变时的偏效应，不支持多类与类别特征（L174–L209）。
- **交互约束**：`interaction_cst` 指定允许交互的特征组，未列出的特征自成一组（L217–L228）。

这些约束不是系数惩罚，而是直接删除部分候选函数。

## RF 与 Extra-Trees

RF 的随机性来自 bootstrap 与节点特征子集，目的在让树间误差解耦并降低方差（L432–L454）。scikit-learn 平均的是 `predict_proba`，不是硬投票（L442），所以概率是树间平均，仍不自动等于校准概率。

Extra-Trees 进一步随机化阈值（L456–L489）。`max_features` 默认随任务变化：RF 分类 `'sqrt'`、回归 `1.0`；ExtraTree 的默认 bootstrap 与 RF 不同（L493，API 外部核查）。

## Bagging、Voting、Stacking、AdaBoost

- **Bagging**：四种抽样变体由样本/特征是否有放回抽取组合而成（L572–L596）。
- **Voting**：硬投票多数类，平票按类标签升序；软投票平均概率，要求基模型提供 `predict_proba`（L598–L693）。
- **Stacking**：用折外预测训练第二层，CV 从评估手段变成训练机制（L727–L829）。二分类时每个基模型的两列概率共线，只保留一列（L795–L796）。
- **AdaBoost**：逐轮调整样本权重；默认分类器是 stump，回归器是 `max_depth=3` 的回归树（L831–L866，API 外部核查）。

Stacking 的关键不是「第二层更复杂」，而是折外预测是否近似未来数据。若训练集有组或时间结构，随机 K 折可能让折外预测泄漏，见 [[cross-validation]]。

## 正则化的三种机制

1. **罚**：叶子级 `l2_regularization`、`ccp_alpha`；
2. **删**：树深、叶子数、单调/交互约束；
3. **平均/随机化**：`subsample`、`max_features`、bootstrap、不同阈值。

学习率 $\nu$ 与轮数必须联合看：$F_m=F_{m-1}+\nu h_m$（L353–L363）。HGBT 的 `max_iter` 是上限，早停开启时不保证跑满。

## 常见失效模式

- 把 HGBT 的速度优势当成所有规模上都更合适；
- 忽略小样本分箱过粗；
- 把 RF 当作硬投票；
- 只用 `max_iter`/`n_estimators` 报预算，忽略早停；
- 把 `subsample` 当作 `learning_rate` 的替代；
- Stacking 使用非折外预测；
- 软投票平均未校准概率；
- 忽略类别基数、缺失值与版本约束；
- 用 MDI 解释高基数或 ID 特征，忘记其训练集统计与高基数偏好（L511–L526）。

## 待解问题

- 小样本 HGBT 与旧 GBDT 的分界；
- 学习率与轮数的联合预算；
- `interaction_cst` 与 `monotonic_cst` 的组合；
- Stacking 折外预测与最终基模型的分布差异；
- AdaBoost 对标签噪声的敏感性；
- IsolationForest 作为 bagging 骨架 + 路径聚合的成本模型（[[sklearn-outlier-detection]]）。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-ensembles]]、[[sklearn-outlier-detection]]
- 概念：[[decision-trees]]、[[cross-validation]]、[[regularization-and-sparsity]]、[[anomaly-detection]]
- 实践：[[ensembles-selection-and-tuning]]、[[hyperparameter-search-and-tuning]]
