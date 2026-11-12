---
title: 集成模型的选型与调参：HGBT / 随机森林 / Extra-Trees / Bagging / Voting / Stacking / AdaBoost
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - ensemble
  - gradient-boosting
  - random-forest
  - bagging
  - stacking
  - workflow
sources:
  - "[[sklearn-ensembles]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1（文档与 API 签名，见 [[sklearn-ensembles]]）
---

# 集成模型的选型与调参

## 一、先选实现，再调预算

| 场景 | 候选 | 理由 |
| --- | --- | --- |
| 大样本、缺失/类别、需要约束 | `HistGradientBoosting*` | 分箱、原生缺失/类别、早停、单调与交互约束 |
| 小样本、分箱可能过粗 | `GradientBoosting*` | 官方丢失主题框建议优先考虑精确切分 |
| 强而复杂的基模型、要降方差 | RandomForest / Extra-Trees | 深树 + 平均与随机化 |
| 基模型不是树 | Bagging | 可换任意估计器，检查成本 |
| 异质且同强模型 | Voting | 硬/软投票接口不同 |
| 有可靠折外预测与第二层收益 | Stacking | CV 变成训练机制，成本与泄漏风险高 |
| 弱学习器 + 顺序纠错 | AdaBoost | 逐轮重加权 |

不要先比较学习率或树数，再发现实现不支持数据接口。

## 二、HGBT 的五步调参

1. **早停状态**：确认 `early_stopping`、验证比例、`n_iter_no_change` 与 `tol`；`max_iter` 是上限。
2. **学习率 × 轮数**：联合调整，不把小学习率当成独立正则。
3. **树大小**：`max_leaf_nodes`、`max_depth`、`min_samples_leaf`，对应交互阶数。
4. **正则与约束**：`l2_regularization`、`subsample`、`max_features`、单调/交互约束。
5. **数据接口**：`max_bins`、类别基数、缺失值、`categorical_features='from_dtype'`。

`max_bins` 合法区间是 [2,255]；源码的 `max_bins + 1` 是数值箱 + 缺失箱。类别基数检查允许等于 `max_bins`（[[sklearn-ensembles]] 外部核查）。

## 三、RF 与 Extra-Trees

- 先用深树和默认 `max_features` 做基线，再扫描；
- RF 分类常见默认 `'sqrt'`，回归 `1.0`；
- RF 平均 `predict_proba`，不是硬投票；
- Extra-Trees 随机化阈值，bootstrap 默认不同；
- `n_estimators` 增加到收益平台后，应比较成本而不是继续堆树。

MDI 是训练期统计且偏好高基数，不适合当无偏特征解释；需要替代方案时另找 permutation importance 来源。

## 四、Bagging / Voting / Stacking

- Bagging 适合基模型不是树或需要自定义抽样策略时；默认只有 10 个基模型，需主动设预算。
- 硬投票看多数类，平票按类标签升序；软投票看平均概率，要求 `predict_proba` 且最好校准。
- Stacking 使用折外预测，成本约为基模型 ×(K+1)，还要加上第二层。分组/时序数据不能随意随机 K 折。
- 多层 stacking 不会自动提高收益，先核对折外分布与最终基模型的差异。

## 五、AdaBoost

默认分类器是 stump，回归器是 `max_depth=3` 的回归树。它逐轮重加权，对标签噪声与异常样本要单独检查，不能只用「抗偏差」概括。

## 六、报告与常见坑

记录实现、早停状态、实际迭代数、基模型、折数、概率响应、数据接口与版本。常见坑：

- 把 HGBT 的速度当成所有规模的优势；
- 忽略小样本分箱；
- 把 RF 当硬投票；
- 忽略早停；
- 把 `subsample` 当学习率；
- 把 `max_bins` 硬上限当建议；
- 忽略类别基数与未见类别；
- 用非折外预测做 stacking；
- 用未校准概率做软投票；
- 把 MDI 当因果重要性；
- 忽略版本弃用，如 `GradientBoosting*.criterion`。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 实现与接口筛选 | 高 |
| 预算与学习率/树大小扫描 | 高 |
| 早停与实际轮数报告 | 高 |
| 折外预测与泄漏审计 | 中 |
| 约束、校准与最终解释 | 中–低 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-ensembles]]、[[sklearn-outlier-detection]]
- 概念：[[ensembles]]、[[decision-trees]]、[[cross-validation]]
- 实践：[[decision-trees-selection-and-tuning]]、[[hyperparameter-search-and-tuning]]、[[feature-selection-workflow]]
