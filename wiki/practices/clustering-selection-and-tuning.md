---
title: "聚类的选型与调参：先缩放与度量、再选族、最后定 $k$/$eps$"
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - clustering
  - scikit-learn
  - unsupervised
  - workflow
sources:
  - "[[sklearn-clustering]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档论述与 API 签名，见 [[sklearn-clustering]]）
---

# 聚类的选型与调参

本页把 [[clustering]] 的机制折成选型流程。先判断聚类是否适合，再处理尺度与度量，最后确定 $k$、`eps` 或密度参数。

## 0. 先判该不该用

| 问题 | 决策 |
| --- | --- |
| 要给新样本打标签吗？ | 只选有归纳路径的方法，或显式包一层；`SpectralClustering`、`AgglomerativeClustering`、`DBSCAN`、`HDBSCAN`、`OPTICS` 不为新样本设计预测接口（[[sklearn-clustering]], L39） |
| 有独立判据吗？ | 有真值用外部指标；没有则用内部指标 + 稳定性 + 业务复核 |
| 簇是凸且各向同性吗？ | 是才优先考虑 KMeans 系；任意形状考虑密度或图方法 |
| 密度一致吗？ | 不一致时考虑 HDBSCAN/OPTICS，而不是单纯调 DBSCAN 的 `eps` |
| 输出要当事实吗？ | 没有独立判据时只能当探索结果，不应直接写成真实类别 |

## 1. 缩放与度量是硬前提

- KMeans 的 inertia、DBSCAN 的 `eps`、RBF 亲和矩阵与层次聚类都依赖尺度。
- 先检查单位与下游距离定义；需要时在训练过程内做缩放，见 [[preprocessing-pipeline-design]]。
- Ward 只能用欧氏距离；其他 linkage 可按数据选 l1、l2、cosine 或预计算矩阵（[[sklearn-clustering]], L198–L204）。
- 不要把「无监督」理解为预处理可以跳过，评估分布本身也是协议的一部分。

## 2. 按旋钮而不是名字选族

| 旋钮/目标 | 候选 | 主要检查 |
| --- | --- | --- |
| 显式 `n_clusters`、凸簇 | KMeans / MiniBatch / Bisecting | 初始化、`n_init`、尺度、空簇与局部极小 |
| exemplar | AffinityPropagation | `preference`、`damping`、$O(N^2)$ 相似度成本 |
| 密度众数 | MeanShift | `bandwidth` 与最近邻成本；没有统一的 `bandwidth_` 可读 |
| 图割 | SpectralClustering | 图/亲和矩阵、`n_clusters`、`assign_labels` 与随机性 |
| 层次树 | AgglomerativeClustering | linkage、连通性、距离阈值、Ward 的欧氏限制 |
| 单密度 | DBSCAN | `eps`、`min_samples`、边界点与样本顺序 |
| 多密度 | HDBSCAN / OPTICS | `min_cluster_size`、层次树、reachability 与噪声 |
| 数据缩减 | BIRCH | `threshold`、`branching_factor`、默认 `n_clusters=3` vs `None` |

`n_clusters=None` 与 BIRCH 默认全局聚类是不同路径；`n_clusters=8`、`eps=0.5` 等只是 API 默认值，不是推荐值。

## 3. 定 $k$、`eps` 与密度参数

### $k$

1. 优先使用业务判据或外部真值；
2. 再比较 silhouette 等内部曲线，但注意其凸簇假设与随 $k$ 变化的趋势；
3. 用不同种子、重采样与 ARI/AMI 检查稳定性；
4. 报告成本与解释性，不只报内部指标最大值。

### `eps`

1. 计算并绘制 k-距离排序曲线，寻找 knee；
2. 扫描多档 `eps`，同时记录簇数、噪声比与边界点；
3. 可用 OPTICS 的 reachability 辅助观察多密度结构；
4. 密度差异明显时优先 HDBSCAN，而不是把单一 `eps` 当全局真理。

### 其他参数

- `min_samples`/`min_cluster_size`：同时影响噪声与最小簇规模；
- `bandwidth`：决定 MeanShift 核大小与簇数；
- `preference`：影响 AP exemplar 数量；
- `linkage`：决定层次簇间几何，不能只看名称。

## 4. 评估与报告

- **有真值**：ARI/AMI 等机会校正指标；需要共识时选对称指标。
- **无真值**：内部指标 + 稳定性 + 业务判据，不能把 silhouette 当真实准确率。
- 记录 splitter、随机种子、样本顺序、预处理与完整估计器构造。
- 保存 `labels_`、簇数、噪声数、参数与必要的中心/图/树。
- 明确 `predict` 是否存在，以及新样本的处理方式。

## 常见坑

- 把标签数值当类别语义；
- 忽略默认簇数与密度参数；
- 把 `eps` 留成默认；
- 只看内部指标；
- 把 Ward 配非欧氏距离；
- 忽略 `assign_labels` 与随机性；
- 把 `cluster_optics_dbscan` 与 DBSCAN 当无条件等价；
- 用旧 AMI doctest 作为当前版本预期；
- 忽略边界点顺序依赖；
- 以单次运行替代稳定性验证。

## 可自动化程度

| 工作 | 自动化 | 原因 |
| --- | --- | --- |
| 尺度与接口检查 | 高 | 可由 schema、稀疏性与缺失值检查驱动 |
| 参数扫描与指标汇总 | 高 | 可枚举，但应限制预算 |
| 稳定性评估 | 中 | 需固定重采样协议与阈值 |
| 形状/密度判读 | 低–中 | 依赖图与领域知识 |
| 最终簇语义与上线 | 低 | 需要业务判据与人工复核 |

所有运行时扫描只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-clustering]]
- 概念：[[clustering]]、[[evaluation-metrics]]、[[manifold-learning]]
- 实践：[[preprocessing-pipeline-design]]、[[manifold-learning-selection-and-tuning]]、[[nearest-neighbors-selection-and-tuning]]
