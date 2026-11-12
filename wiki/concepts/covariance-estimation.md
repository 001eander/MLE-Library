---
title: "协方差估计（Covariance Estimation）：Σ 与 K 的四种估计器、三种损失口径与 Mahalanobis 作为共同货币"
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - covariance
  - shrinkage
  - precision-matrix
  - robust-statistics
  - estimation
  - anomaly-detection
sources:
  - "[[sklearn-covariance]]"
  - "[[sklearn-outlier-detection]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.0 – 1.9.1（文档论述与 API 签名，见 [[sklearn-covariance]]）
---

# 协方差估计（Covariance Estimation）

本页研究估计量本身：不预测 $y$、不分组、不降维，只估计二阶结构 $\Sigma$ 与精度矩阵 $K=\Sigma^{-1}$。下游 LDA/QDA、Mahalanobis、异常检测与因子模型都是消费者（[[sklearn-covariance]], L10–L18）。

## 一张速查表

| 估计器 | 结构假设 | 强度来源 | 能恢复非对角零结构？ | 主要风险 |
| --- | --- | --- | --- | --- |
| `EmpiricalCovariance` | 无 | 无 | 否 | 小样本特征值不准、可能奇异；离群点敏感 |
| `ShrunkCovariance` | 向 $\frac{\mathrm{Tr}}pI$ 收缩 | 手设 `shrinkage` | 否 | 默认 0.1 只是任意起点 |
| `LedoitWolf` | 同上 | 解析 α，目标是 $\Sigma$ 的 MSE | 否 | MSE 最优不等于下游最优 |
| `OAS` | 同上 + 高斯假设 | 解析 α | 否 | 假设不成立时优势无保证 |
| `GraphicalLasso` | $K$ 稀疏 | `alpha` 的 L1 强度 | **是，$K$ 的非对角元** | 数值稳定、收敛与模型误设 |
| `GraphicalLassoCV` | 同上 | CV 选择 α | 是 | 似然目标可能多于真值边 |
| `MinCovDet` | 修剪 + 重加权 | `support_fraction` | 否 | $n>p$ 前提、支持集解释与随机搜索 |

来源位置：[[sklearn-covariance]], L16、L28–L32、L79–L83、L118–L128；默认值以 1.9.1 API 签名外部核查为准。

## 收缩：一个算子，三种 α 来源

基本形式（L32）：

$$\Sigma_{\text{shrunk}}=(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$$

- 经验协方差对应 α=0；
- 手设 α 是直接选择偏差–方差折中；
- LW 与 OAS 用不同规则从数据估计 α。

因此可将它们理解为**同一个收缩算子 + 不同 α 来源**（本库综合）。α=1 时对角为 $\mathrm{Tr}\hat\Sigma/p$、非对角为 0，但它不是「无结构的最优答案」。

LW 的目标是协方差 MSE（L42），OAS 在高斯假设下追求更小 MSE（L61）。这不是 held-out 似然、支持集 F1 或下游分类准确率。

## 各向同性 note 的正确读法

L46–L51 指出：当总体协方差是单位阵倍数时，样本量增大不会让 LW 的 α 趋于 0，反而趋于 1，因为最优估计本来就接近单位阵倍数。该 note 说的是**目标形态**，不是「α 随样本量单调变化」的路径保证。

## 稀疏精度矩阵：学条件独立图

$K_{ij}=0$ 表示给定其余变量时 $x_i$ 与 $x_j$ 条件独立（L79）。因此：

- $\Sigma$ 稀疏更接近边际独立；
- $K$ 稀疏对应条件独立图；
- 收缩可以压小非对角，但不能把非对角置零；
- 只有 $K$ 上的 L1 能直接恢复非对角零结构。

GraphicalLasso 的目标是（L101–L103）：

$$\hat K=\arg\min_K \operatorname{tr}SK-\log\det K+\alpha\lVert K\rVert_1$$

其中 L1 惩罚作用于非对角元素。CV 可能因 held-out 似然而选择过多边，不能把「边数正确」与「协方差误差小」当成同一目标（L97，本库综合）。

## 稳健估计与 MCD

经验和收缩估计都对离群点敏感。MCD 的流程是（L124–L128）：

1. 选 $h$ 个「好」观测并估计；
2. 一致性校正；
3. 按 Mahalanobis 距离重加权。

`raw_location_`/`raw_covariance_` 是校正前的量，`location_`/`covariance_` 是校正后的量。源码外部核查还显示：`raw_support_` 才对应最初的 $h$ 个支持样本，`support_` 会被重加权掩码覆盖，二者不能混读。

## 三种损失口径

| 口径 | 回答的问题 | 不能推出什么 |
| --- | --- | --- |
| $\Sigma$ 的 MSE/Frobenius 误差 | 矩阵估得准不准 | 下游预测或图结构更好 |
| held-out NLL | 概率模型对新数据是否合适 | 支持集边数正确 |
| 支持集 precision/recall/F1 | 条件独立边是否恢复 | 协方差矩阵误差最小或 NLL 最优 |

Mahalanobis 分离度是另一种诊断视角，仍受所选 $\Sigma$ 影响。工程上应先声明目标，再选 α 与估计器。

## 常见失效模式

- 把 `assume_centered=True` 当成「自动居中」；它表示不再减均值，训练与测试数据都应满足零均值契约（L18）。
- 把 MLE 的 $1/n$ 与无偏样本协方差的 $1/(n-1)$ 混用。
- 把 LW/OAS 的 MSE 最优当成下游最优。
- 把 $\Sigma$ 稀疏与 $K$ 稀疏混为一谈。
- 忽略 CV 的目标函数，把 GLassoCV 当结构恢复保证。
- 把 `support_` 当作 $h$。
- 奇异时只检查 `precision_` 是否存在，不检查伪逆与分数语义。

## 待解问题

- LW/OAS 的解析 α 公式与假设；
- MCD breakdown point 与 $h$ 的关系；
- 收缩与稀疏能否组合；
- 相关阵与协方差阵上的收缩选择；
- GLassoCV 的 refit 语义；
- 高维、共线与不同 α 下的稳定性；
- `EmpiricalCovariance.score` 的常数项与跨实现可比性。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-covariance]]、[[sklearn-outlier-detection]]
- 概念：[[gaussian-discriminant-analysis]]、[[regularization-and-sparsity]]、[[matrix-factorization]]、[[anomaly-detection]]
- 实践：[[covariance-estimation-selection-and-tuning]]、[[linear-models-as-baselines]]
