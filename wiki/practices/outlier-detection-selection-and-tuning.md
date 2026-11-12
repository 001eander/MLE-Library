---
title: 异常检测的选型与调参：先判污染与是否要上线，再选形状假设，最后定阈值
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - workflow
sources:
  - "[[sklearn-outlier-detection]]"
  - "[[sklearn-covariance]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-ensembles]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档版本与 API 签名，见 [[sklearn-outlier-detection]]）
---

# 异常检测的选型与调参

## 一、五个准入问题

1. 训练集是否被污染？
2. 要给新样本打分吗？
3. 异常能成簇吗？
4. 正常数据是单峰高斯、支持边界还是局部密度？
5. 规模与维度能否承受核方法或邻居计算？

先决定 outlier 还是 novelty，再选形状，不要先比算法名（[[sklearn-outlier-detection]], L12–L20）。

## 二、按形状选族

| 假设 | 方法 | 必查项 |
| --- | --- | --- |
| 高斯椭圆 | `EllipticEnvelope` | MCD、$n>p$、`raw_support_` vs `support_`、污染率 |
| 支持边界 | `OneClassSVM` | 核、`gamma`、`nu`、尺度与成本 |
| 同目标的近似路径 | `SGDOneClassSVM` + Nyström | 近似成分、核参数、`nu` |
| 轴对齐隔离 | `IsolationForest` | `max_samples`、树数、随机种子、`max_features` |
| 局部密度 | `LocalOutlierFactor` | $k$、度量、污染率、`novelty` 模式 |

不同方法的分数与阈值没有统一量纲，不能用默认输出数量排名。

## 三、定阈值五步

1. **先看排序**：在有真值或可靠样本上检查异常分数排序；
2. **再定先验**：明确污染率与误报/漏报成本；
3. **检查报出数**：同时看分数、阈值、异常数与业务覆盖；
4. **不在训练集上验证 novelty**：文档警告训练集 `predict` 与 `fit_predict` 可不同；
5. **上线后重算**：部署分布或先验变化后重新确定 `offset_`。

`contamination` 是训练集分位数规则，不是统一异常比例。IF/LOF 的 `'auto'` 是固定偏移，EE 默认 0.1，OCSVM 用 `nu`。

## 四、四个容量旋钮

- `n_neighbors`：LOF 的局部密度尺度；需同时考虑簇规模与 $n$；
- `nu`：OneClassSVM 的边界/支持向量口径，与新样本概率的关系需验证；
- `max_samples`：IF 的子样本上限，默认 `'auto'` 对应 `min(256,n)`（API 外部核查）；
- `n_estimators`：IF 的树数与随机平均预算。

缩放与度量属于模型定义，特别是核方法和局部密度；轴对齐隔离与协方差方法也不能用一次实验推断完全尺度不变。

## 五、评估路径

- **有真值**：ROC/PR、召回与误报，按最终阈值计算；
- **无真值**：跨种子/重采样稳定性、业务规则与人工复核；
- **上线**：报出量、覆盖率、人工成本与分布漂移；
- **LOF**：分别评估训练集 `negative_outlier_factor_` 与推理 `score_samples`，不混用。

## 常见坑

- 把 `'auto'` 当自动 10%；
- 把分数当跨模型概率；
- 混用 outlier 与 novelty；
- 忽略 LOF 两种模式的接口差异；
- 用训练分数与新样本分数混算阈值；
- 忽略密集异常簇与局部/全局差异；
- 只用单次 AUC 或耗时比较方法；
- 把 `support_` 当 MCD 的 $h$；
- 用历史本机结果代替目标环境验证。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 任务与接口检查 | 高 |
| 形状/参数扫描 | 中–高 |
| 阈值与报出量报告 | 高 |
| 稳定性与业务评估 | 中 |
| 最终异常语义与上线 | 低 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-outlier-detection]]、[[sklearn-covariance]]、[[sklearn-nearest-neighbors]]、[[sklearn-ensembles]]
- 概念：[[anomaly-detection]]、[[covariance-estimation]]、[[nearest-neighbors]]
- 实践：[[covariance-estimation-selection-and-tuning]]、[[nearest-neighbors-selection-and-tuning]]、[[metric-selection-and-thresholding]]
