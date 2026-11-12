---
title: 流形学习（Manifold Learning）：被保量决定方法、kNN 图是共同底座
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - dimensionality-reduction
  - manifold-learning
  - unsupervised
sources:
  - "[[sklearn-manifold-learning]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.0 – 1.9.1（文档论述与 API 签名，见 [[sklearn-manifold-learning]]）
---

# 流形学习（Manifold Learning）

## 输出是一套坐标

流形学习是非线性降维：假设数据的表观高维只是人为高维，目标是学习一套新坐标（[[sklearn-manifold-learning]], L28）。

它没有 $y$，优化的是几何量而不是预测误差。「人为高」没有通用判定准则，需要本征维数、重构误差或下游任务等额外证据。

`Isomap` 与 `LocallyLinearEmbedding` 提供 `transform`；`SpectralEmbedding`、`TSNE`、`MDS`、`ClassicalMDS` 没有统一的训练后映射接口（API 外部核查）。后者不能把测试集当作普通新样本 `transform`，放进训练/推理分离管线前必须特别设计。

## 被保量决定方法

| 方法 | 被保量 | 主要目标 |
| --- | --- | --- |
| PCA / classical MDS | 成对欧氏内积 | 全局线性投影、strain |
| Isomap | kNN 图上的测地距离 | 全局距离结构 |
| LLE 四变体 | 局部重构权重 | 局部线性关系与全局对齐 |
| Spectral Embedding | 图上邻近 | Laplacian Eigenmaps |
| MDS metric | 不相似度 | 距离逼近 |
| MDS non-metric | 不相似度的序 | 单调关系 |
| t-SNE | 邻域概率 | 局部邻域与多尺度结构 |

**本库综合**：方法名不如被保量重要。要求展开单个连续流形与要求突出局部簇不是同一目标；t-SNE 官方判据更偏后者（[[sklearn-manifold-learning]], L155）。

## kNN 图是共同底座

Isomap、LLE、Spectral 与 t-SNE 都依赖邻域结构。$k$ 太小可能图不连通，太大可能抹平局部性。尺度决定谁是谁的邻居，因此「换缩放 = 换模型」（L176）。

成本可分三阶段：邻居搜索、权重/图构造、特征分解。LLE 的 `reg` 是邻域权重矩阵的数值稳定器，不应直接当作一般模型正则强度。

## MDS 与 t-SNE

MDS 分 metric、non-metric、classical 三型（L118–L138）：分别保距离、距离的序与内积。classical MDS 在欧氏距离下等价于 PCA；raw stress、Stress-1 与 strain 的量纲不同，不能跨配置比较。

t-SNE 使用原空间高斯联合概率与嵌入空间 Student-t 概率，KL 非凸，官方允许不同种子后选最低 KL（L157–L163）。它不显式保留全局结构，`init='pca'` 可缓解；Barnes-Hut 限制输出维数，其他输入与近似限制见官方折叠块。

## 重构误差不是通用选维判据

L177 建议用重构误差选输出维数，但各方法没有统一的重构误差接口：Isomap 与 LLE 的形式不同，其他方法也不统一提供。不能把「每种方法都有同一种误差曲线」当成事实，方向与拐点需在目标版本验证。

`trustworthiness` 是 API 中可用的补充指标，但本节正文没有讨论。它惩罚假邻居，不全面衡量全局结构；不同评估口径应并列报告，而不是混为一个「保真度」。

## 常见失效模式

- 未统一尺度就建 kNN 图；
- 忽略图不连通与噪声短路（L178）；
- 把没有 `transform` 的方法放进普通推理管线；
- 用重构误差通用选维；
- 跨配置比较 `stress_`；
- 用坐标差判断两次 t-SNE 谁更好，而不是按 KL 与任务评估；
- 把 ARPACK 失败都归因于重复点，忽略不连通与求解器选择（L179）；
- 只看局部指标就宣称全局结构保留。

## 待解问题

- 可操作的本征维数估计；
- 各方法重构误差的方向与拐点；
- MLLE/HLLE/LTSA 的细丝塌缩与 $k$/`reg`；
- 大样本下 `SpectralEmbedding.n_neighbors=None` 的图成本；
- 噪声短路的定量阈值；
- 非欧氏距离下 ClassicalMDS 与 PCA 的差异；
- 小样本 t-SNE 的学习率下限；
- 降维后接下游模型的泄漏与外推误差。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-manifold-learning]]
- 概念：[[matrix-factorization]]、[[nearest-neighbors]]、[[clustering]]
- 实践：[[manifold-learning-selection-and-tuning]]、[[preprocessing-pipeline-design]]
