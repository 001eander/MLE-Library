---
title: 异常检测（Anomaly / Novelty / Outlier Detection）
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - one-class-svm
  - isolation-forest
  - local-outlier-factor
  - robust-statistics
  - unsupervised
sources:
  - "[[sklearn-outlier-detection]]"
  - "[[sklearn-covariance]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-svm]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档论述与 API 签名，见 [[sklearn-outlier-detection]]）
---

# 异常检测（Anomaly / Novelty / Outlier Detection）

## 定义：先区分两种任务

**异常检测学习「正常」的判据，给样本一个异常分数，再用阈值变成 ±1。** 在 scikit-learn 用户指南中，outlier detection 与 novelty detection 不是同一任务（[[sklearn-outlier-detection]], L12–L20）：

| 任务 | 训练集 | 要回答的问题 |
| --- | --- | --- |
| Outlier detection | 已被污染 | 哪些训练样本偏离最密集区域？ |
| Novelty detection | 假定干净 | 新样本是否来自同一总体？ |

来源还给出「outlier detection 的异常不能成簇」这一前提：估计器假设异常位于低密度区。它应被理解为模型假设，而不是所有算法在所有数据上的性能排序。

## 四种形状假设

| 方法 | 「正常」的形状 | 分数来源 | 主要边界 |
| --- | --- | --- | --- |
| `EllipticEnvelope` | 高斯椭圆 | 稳健 Mahalanobis 距离 | 分布形状不符时，距离大不等于业务异常；继承 MCD 的适用条件 |
| `OneClassSVM` | 支持边界 | 核空间中的边界与 `nu` | 依赖核与尺度；`nu` 的口径需与普通 SVM 分开 |
| `SGDOneClassSVM` | 同一线性目标的近似求解 | 线性模型 + Nyström 近似 | 近似程度与核参数需要评估 |
| `IsolationForest` | 随机轴对齐隔离 | 平均路径长度 | 子样本与随机阈值影响分数；默认 `max_samples='auto'` 对应 `min(256,n)`（API 外部核查） |
| `LocalOutlierFactor` | 局部密度比 | 邻域平均密度 / 自身密度 | 依赖 $k$、度量与密度变化；局部排名不等于全局异常程度 |

这些方法的输出没有统一量纲。跨算法比较「默认谁报得多」没有意义，因为形状假设、分数尺度和阈值规则都不同。

## 分数、阈值与标签

来源给出的共同契约是（[[sklearn-outlier-detection]], L34–L36、L172）：

```text
score_samples → offset_ → decision_function → predict ∈ {+1, −1}
```

- `score_samples`：原始分数；
- `offset_`：由训练集和 `contamination` 等规则确定的阈值；
- `decision_function = score_samples − offset_`：负值为异常，非负为正常；
- `contamination`：训练集分数的分位数规则，不是跨模型统一的异常比例。

API 外部核查显示，IF 与 LOF 的 `'auto'` 分别使用固定偏移 −0.5 与 −1.5，`EllipticEnvelope` 默认 0.1，`OneClassSVM` 没有 `contamination`，而使用 `nu`。因此默认值不能直接横向比较。

## LOF 的两个模式

LOF 是接口上的特例（[[sklearn-outlier-detection]], L42–L59、L187–L201）：

- `novelty=False`：主要用于训练集，`fit_predict` 与 `negative_outlier_factor_`；
- `novelty=True`：用于新样本的 `predict`、`decision_function`、`score_samples`，此时不能再把训练集 `fit_predict` 当作同一口径使用。

文档明确警告：`novelty=True` 时训练集上的 `predict` 与 `fit_predict` 可能不同。工程上应分别保存训练集分数与推理分数，不混用阈值、缓存或评估结果。

## 选型顺序

1. **训练集是否被污染？** 决定 outlier 还是 novelty。
2. **是否要给新样本打分？** 决定能否使用传导式训练集接口，尤其检查 LOF 的模式。
3. **正常数据有什么形状？** 高斯椭圆、支持边界、轴对齐隔离或局部密度。
4. **异常能否成簇、密度是否变化？** 决定局部密度假设是否合理。
5. **阈值先验从哪里来？** 没有污染率先验时，不应把 `'auto'` 当作通用答案。
6. **如何评估？** 有真值时可用分类指标；无真值时需要稳定性、业务规则或人工复核，不把无标签分数当事实。

## 常见失效模式

- 把异常分数当作跨数据集可比较的概率；
- 把 `'auto'` 当作「自动 10%」；
- 用 outlier 模型处理干净训练集上的新样本，或反过来；
- 忽略尺度、度量和 $k$ 对局部密度的影响；
- 用训练集 `negative_outlier_factor_` 与新样本 `score_samples` 混算阈值；
- 用一次实验的排名替代方法假设与适用范围；
- 忽略 `MinCovDet` 的 `raw_support_` 与重加权 `support_` 的区别（[[sklearn-covariance]] 外部核查）。

## 待解问题

- `OneClassSVM` 的 `nu` 与「新正常样本落在边界外概率」的精确关系；
- LOF 的自身排除与推理分数差异；
- `contamination` 并列分数时的阈值规则；
- Nyström 成分数与 `nu` 的联合影响；
- 无标签场景下的稳定性与业务回测协议。

以上运行时问题只在 `uv` 创建的目标版本环境中复核，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-outlier-detection]]、[[sklearn-covariance]]、[[sklearn-nearest-neighbors]]、[[sklearn-ensembles]]、[[sklearn-svm]]
- 实践：[[outlier-detection-selection-and-tuning]]、[[evaluator-design-and-reward-hacking]]
- 概念：[[nearest-neighbors]]、[[covariance-estimation]]、[[support-vector-machines]]、[[ensembles]]
