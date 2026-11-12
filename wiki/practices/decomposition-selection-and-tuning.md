---
title: 成分分解的选型与调参：先问「要方差、要零件、还是要主题」
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - matrix-factorization
  - decomposition
  - dimensionality-reduction
  - workflow
sources:
  - "[[sklearn-decomposition]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档与 API 签名，见 [[sklearn-decomposition]]）
---

# 成分分解的选型与调参

## 一、四问准入

1. 要压缩/解释、去噪，还是主题？
2. 数据有负值吗？有则 NMF 不适用。
3. 需要给新样本 `transform` 吗？检查估计器接口与输出方向。
4. 评估看方差、重构、下游任务还是主题稳定性？

若只需要删列并保留原特征语义，先用 [[feature-selection]]；若只需要局部邻域布局，考虑 [[manifold-learning]]。

## 二、按目标选族

| 目标 | 首选 | 关键约束 |
| --- | --- | --- |
| 全局方差/正交压缩 | PCA / IncrementalPCA / randomized PCA | 只中心化不缩放；近似反投影不精确 |
| 稀疏成分与字典 | SparsePCA / DictionaryLearning | `alpha`、字典大小、非凸与成本 |
| 非线性核表示 | KernelPCA | 核与 γ、样本数上界、近似反投影 |
| 文本 LSA | TruncatedSVD | 不中心化；中心化后才与 PCA 等价 |
| 异方差噪声 | FactorAnalysis | 逐特征对角噪声；成分不保证正交 |
| 盲源分离 | FastICA | 必须白化；通常不是普通降维 |
| 非负零件 | NMF / MiniBatchNMF | 非负输入、初始化、β 与局部极小 |
| 主题模型 | LDA | 文档-词输入、Dirichlet 先验、online/batch 选择 |

## 三、预处理契约

- PCA 不缩放；是否缩放决定所得主成分；
- `whiten=True` 将成分缩到单位方差；
- NMF 输入必须非负；
- ICA 必须白化；
- IncrementalPCA 的内存由 batch 决定；
- 所有拟合都放在训练折内，避免测试分布进入成分学习，见 [[preprocessing-pipeline-design]]。

## 四、定 $k$ 与正则

- 压缩：看方差与重构，并结合下游任务；
- 稀疏字典：同时报告 `alpha`、零元素、重构与解释；
- NMF：比较初始化、`solver`、`beta_loss`，不要跨 β 比重建误差；
- LDA：检查主题稳定性、文档-主题分配与 perplexity，不能只按主题数排序；
- `n_components='auto'` 的行为按版本与 API 文档核对，不当作通用自动选维。

## 五、保存与推理

- PCA scores 与 `components_` 方向不同；
- NMF 的 `fit_transform` 返回 W，`components_` 是 H；
- `transform` 在固定 H 后重新求编码，不保证等于 `fit_transform`；
- `split_sign` 把编码正负部拼接并翻倍；
- API 参数名是 `split_sign` 与 `transform_algorithm`，不是正文的 `split_code` 与 `transform_method`；
- `MiniBatchSparsePCA` 不提供样本方向 `partial_fit`，不要按名字推断。

## 六、报告与失效模式

记录估计器构造、输入预处理、$k$、初始化、随机种子、输出矩阵方向、成本与目标。常见坑：

- 把 `components_` 方向读反；
- 用不同 β 或不同模型分数直接比较；
- 以为 MiniBatch 都能增量；
- 忽略 NMF 初始化；
- 把 NMF 当自动语义零件；
- 把 KernelPCA 反投影当精确逆；
- 用正文参数名调用 API；
- 忽略测试折参与无监督分解的协议问题。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 输入约束与矩阵形状检查 | 高 |
| $k$/α/初始化扫描 | 高 |
| 重构与下游评估 | 中 |
| 成分/主题解释 | 低–中 |
| 部署与增量策略 | 中–低 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-decomposition]]
- 概念：[[matrix-factorization]]、[[feature-selection]]、[[manifold-learning]]
- 实践：[[preprocessing-pipeline-design]]、[[feature-selection-workflow]]
