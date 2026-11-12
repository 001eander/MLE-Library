---
title: 矩阵分解与成分分解（Matrix Factorization / Component Decomposition）
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - matrix-factorization
  - decomposition
  - dimensionality-reduction
  - pca
  - nmf
  - ica
  - topic-model
sources:
  - "[[sklearn-decomposition]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档与 API 签名，见 [[sklearn-decomposition]]）
---

# 矩阵分解与成分分解（Matrix Factorization / Component Decomposition）

## 同一个骨架，约束就是模型

矩阵分解同时给出编码与成分：$X\approx UV$。它不同于删列（[[feature-selection]]）、分组（[[clustering]]）和只输出坐标的流形学习（[[manifold-learning]]），因为 `components_` 也可作为一等解释对象（[[sklearn-decomposition]], L16）。

| 方法 | 核心约束 | 主要输出 |
| --- | --- | --- |
| PCA | 正交、最大方差 | scores 与正交成分 |
| Incremental/Randomized PCA | 同上 + 小批量/近似 | 近似成分与编码 |
| SparsePCA / DictionaryLearning | $ℓ_1$ 稀疏 | 稀疏字典与编码 |
| KernelPCA | 核空间方差 | 非线性编码与近似反投影 |
| TruncatedSVD / LSA | 不中心化的截断 SVD | 稠密或稀疏输入的低秩表示 |
| FactorAnalysis | 低秩 + 逐特征对角噪声 | 因子载荷与噪声方差 |
| ICA | 成分最大独立 | 独立源与混合矩阵 |
| NMF | 非负加法分解 | $W,H\ge0$ |
| LDA | 文档/主题 Dirichlet 先验 | 主题-词与文档-主题矩阵 |

## 预处理契约

- PCA 只中心化，不缩放；是否缩放由调用者决定（L16）。
- `whiten=True` 把成分缩到单位方差，不是「更好的 PCA」。
- TruncatedSVD 不中心化；先减每列均值时才与 PCA 等价（L134–L136）。
- NMF 要求非负数据与成分（L286）。
- ICA 模型没有噪声项，必须白化（L268）。
- `IncrementalPCA` 的内存取决于 batch，不是总样本数（L34–L40）。

因此预处理不是统一步骤，而是模型定义的一部分。

## 成分方向与返回值

- PCA：`fit_transform` 返回 scores，`components_` 是 $k\times p$；
- NMF：`fit_transform` 返回 $W$（$n\times k$），`components_` 是 $H$（$k\times p$）；
- LDA：`components_` 是主题-词，`transform` 是文档-主题（L398）；
- `SparseCoder` 使用固定字典，不学习字典；实现中的空操作 `fit` 只为 Pipeline API 保留；
- `MiniBatchSparsePCA` 的「online」沿特征方向，不提供样本方向 `partial_fit`（L102）。

`transform` 不是 `fit_transform` 的缓存：NMF 在固定 H 后重新求编码，不能保证逐位相同。需要训练集编码时应保存 `fit_transform` 的返回值。

## 非唯一性与初始化

PCA 的成分有符号/旋转不唯一；FA 成分不保证正交（L254）；NMF 非凸，初始化会改变局部极小；LDA 在线 VB 只保证局部最优（L396）。

NMF 的 `init` 可选 NNDSVD 家族或 random。基本 NNDSVD 的零无法被 `solver='mu'` 更新，稠密数据更适合 NNDSVDa/NNDSVDar（L300–L306）。`beta_loss` 的目标与尺度不同，重建误差不能跨 β 直接比较。

`split_sign` 把编码的正部与负部绝对值拼接，列数翻倍且非负；正文写作 `split_code`，API 参数名是 `split_sign`。`transform_method` 在 API 中对应 `transform_algorithm`（L152、L160，外部核查）。

## 评估与选择

不同模型的分数不可直接比较：

- PCA 的方差比与 `score`；
- NMF 的 `reconstruction_err_`；
- LDA 的 perplexity；
- 字典学习/KernelPCA 的重构误差。

选择 $k$ 时应结合任务：压缩、解释、下游预测或主题稳定性，而不是寻找统一最优维数。 randomized SVD 的收益主要在 $k\ll\min(n,p)$，反投影仍不是精确逆（L63–L67）。

## 常见失效模式

- 把 `components_` 的方向读反；
- 把 `transform` 当 `fit_transform` 缓存；
- 以为 MiniBatch 一定能 `partial_fit`；
- 用不同量纲或不同 β 的重建误差比较模型；
- 以为 NMF 自动产生人类可解释零件；
- 忽略初始化与局部极小；
- 把正文参数名直接复制到 API；
- 把 FA 的异方差噪声当作 PCA 的同构变体；
- 用一次反投影误差推断所有核/任务都适用。

## 待解问题

- PCA 的 `n_components='mle'` 与重构/下游选维；
- 白化对下游模型的影响；
- randomized SVD 的 oversamples 与误差曲线；
- KernelPCA 求解器选择；
- 字典大小与 α；
- NMF 的 `n_components='auto'`；
- LDA online/batch 的质量差；
- ICA 的 `n_components` 与白化方式；
- 分解器的 Pipeline 泄漏与增量成本。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-decomposition]]
- 概念：[[manifold-learning]]、[[feature-selection]]、[[covariance-estimation]]
- 实践：[[decomposition-selection-and-tuning]]、[[preprocessing-pipeline-design]]
