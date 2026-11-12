---
title: 特征选择（Feature Selection）：过滤式/嵌入式/包裹式，代理指标的假设与「选了哪些」的模型依赖
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - feature-selection
  - dimensionality-reduction
  - scikit-learn
sources:
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-decision-trees]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1 的 `sklearn.feature_selection`（文档论述与 API 签名，见 [[sklearn-feature-selection]]）
---

# 特征选择（Feature Selection）

## 删列不是换坐标

特征选择保留原始列的子集；矩阵分解则构造新成分（[[matrix-factorization]]）。它适合需要保留原特征语义、降低方差或减少接口成本的场景，不会自动创造信息或提升准确率（[[sklearn-feature-selection]], L10）。

「选中集合」是 **(数据, 模型, 超参)** 的函数，不是数据的固有属性。不同评分函数、正则强度和评估协议可以留下不同列。

## 三家族的判据

| 家族 | 目标函数从哪里来 | 是否迭代拟合 | 主要盲点 |
| --- | --- | --- | --- |
| 过滤式 | 单变量统计量 | 否 | 不见冗余与交互；继承统计假设 |
| 嵌入式 | 全模型的系数或重要性 | 通常一次 | 继承模型偏好与正则尺度 |
| 包裹式 | 子集的 CV 分数 | 多次 | 成本高，继承 CV 方差与选择偏差 |

过滤式包括 `VarianceThreshold`、`SelectKBest`、`SelectPercentile`、`SelectFpr`/`Fdr`/`Fwe`；嵌入式包括 L1 与 MDI + `SelectFromModel`；包裹式包括 `RFE`/`RFECV`/`SequentialFeatureSelector`（[[sklearn-feature-selection]] 文档结构）。

## 代理指标的假设

- F 检验：测线性依赖，不能代表所有非线性信号；
- $\chi^2$：要求非负特征，带有频次语义；
- 互信息：形式更通用，但估计需要更多样本且有方差；
- L1：零系数可用来删列，但相关组中选择受模型与正则影响；
- MDI：训练期统计，偏好高基数特征，见 [[ensembles]]；
- CV 分数：贴近目标，却会继承切分结构与选择偏差。

`SelectFpr` 只做逐特征检验，不做多重校正；`SelectFdr` 对应 Benjamini–Hochberg，`SelectFwe` 对应 Bonferroni（源码外部核查）。`alpha` 同名但控制目标不同。

## 阈值与个数都是超参

- `SelectKBest` 默认 `k=10`，`SelectPercentile` 默认 10%；
- `GenericUnivariateSelect` 默认 `mode='percentile'`、`param=1e-5`；
- `SelectFromModel(threshold=None)`：L1 模型用 `1e-5`，其他模型用 `'mean'`（1.9.1 API 外部核查）；
- `RFE`/`SequentialFeatureSelector` 默认保留一半左右；
- `RFECV` 用 CV 的 argmax 选个数，也可能选满。

`prefit=True` 时，`transform` 可通过 `estimator` 回退工作，但未 `fit` 就读 `threshold_` 会因缺少 `estimator_` 失败（源码外部核查）。仅按 `max_features` 选择时可使用官方建议的 `threshold=-np.inf`。

## 成本按估计器拟合次数计算

- 过滤式：不拟合预测模型；
- `SelectFromModel`：通常一次；
- RFE：随递归删除次数增加；
- SFS：随候选子集与 CV 折数增加；
- RFECV：RFE 路径 × CV 折数。

因此「包裹式更准」必须与预算、切分与方差一起判断，不能只看最终分数。

## 必须放进 Pipeline

 supervised 选择器使用 $y$，先在全量数据上选列再做 CV 会让测试折参与选择。正确做法是在每个训练折内重新拟合选择器（[[sklearn-feature-selection]], L165–L173；[[cross-validation-design-and-tuning]]）。

这条规则也适用于目标编码与其他使用 $y$ 的预处理。报告时应记录选择器构造、评分函数、阈值、折数与最终列名，而不只报「选了多少列」。

## 常见失效模式

- 把选择结果当作稳定真值；
- 忽略过滤式看不见冗余；
- 把 MDI 当无偏重要性；
- 忽略 L1 与树默认阈值的差异；
- 以为 RFECV 一定会减维；
- 先选后 CV 造成泄漏；
- 用同一 CV 选特征数并报最优分，忽略选择偏差；
- 稀疏输入与输出类型未检查；
- 忽略多重检验与随机性；
- 把 `prefit=True` 的推理可用误当成所有诊断属性都可用。

## 待解问题

- RFECV 的嵌套评估与乐观偏差；
- 多输出/多类的 `importance_getter` 聚合；
- 各选择器的稀疏输入/输出类型；
- SFS 的 `tol` 与目标个数优先级；
- 重采样下的选择频率与稳定性选择；
- Lasso 恢复支撑集的条件与 `alpha` 选择；
- 同源特征组必须整组进出的实现方式。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-feature-selection]]、[[sklearn-linear-models]]、[[sklearn-ensembles]]、[[sklearn-decision-trees]]
- 概念：[[feature-preprocessing]]、[[regularization-and-sparsity]]、[[matrix-factorization]]
- 实践：[[feature-selection-workflow]]、[[cross-validation-design-and-tuning]]
