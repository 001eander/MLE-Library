---
title: 聚类（Clustering）：四条几何假设、$k$ 的三个来源、归纳/传导断层与指标的三个可判定性质
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - clustering
  - unsupervised
  - scikit-learn
sources:
  - "[[sklearn-clustering]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档论述与 API 签名，见 [[sklearn-clustering]]）
---

# 聚类（Clustering）

## 定义与三个推论

聚类对无标签数据分组；scikit-learn 中的算法既有 `fit` 后读取 `labels_` 的类，也有直接返回整数标签数组的函数（[[sklearn-clustering]], L10–L12）。

1. **没有 $y$，也没有唯一正确答案**。标签数值本身无意义，评估要看划分与外部真值或内部几何，而不是分类准确率。
2. **输出是每个样本的离散分配**。它可作为最终结果或中间压缩，但不能未加处理就当作连续特征。
3. **能否给新样本打标签是硬断层**。`SpectralClustering`、`AgglomerativeClustering`、`DBSCAN`、`HDBSCAN`、`OPTICS` 不为新样本设计预测接口；KMeans 系、MeanShift、BIRCH 等提供归纳路径（[[sklearn-clustering]], L39）。

## 判别方法的第一性判据：假设了什么簇

| 方法族 | 几何/密度假设 | 簇数来源 | 主要代价 |
| --- | --- | --- | --- |
| KMeans / MiniBatch / Bisecting | 凸、各向同性、等方差，优化 inertia | 显式 `n_clusters` | 初始化与局部极小；高维距离失真 |
| AffinityPropagation |  exemplar 与消息传递 | 由 `preference` 等间接决定 | 相似度矩阵 $O(N^2)$ 内存、$O(N^2T)$ 时间 |
| MeanShift | 平滑密度中的 blob | 由 `bandwidth` 间接决定 | 反复最近邻搜索 |
| Spectral | 图上 normalized cut | 显式 `n_clusters` | 图构造、亲和矩阵与标签分配策略 |
| Agglomerative | linkage 定义的簇间距离 | `n_clusters` 或距离阈值 | 无连通性时合并成本高；Ward 只支持欧氏距离 |
| DBSCAN | 全局同质的密度连通 | `eps`/`min_samples` 间接决定 | `eps` 强依赖尺度与密度 |
| HDBSCAN / OPTICS | 多密度层次结构 | `min_cluster_size` 等间接决定 | 参数解释与边界点处理更复杂 |
| BIRCH | CF 树的数据缩减 | `n_clusters` 全局聚类或 `None` 读叶子 | 有损压缩；API 默认 `n_clusters=3` 易被忽略 |

**本库综合**：选算法应先问「簇的形状与密度是什么」，再问方法名。inertia、密度连通、图割和 Ward 方差可以对同一数据给出不同但都自洽的划分。

## $k$ 从哪里来

- **显式给定**：`n_clusters`，本质是模型结构选择；
- **间接决定**：`eps`、`min_cluster_size`、`bandwidth`、`preference` 控制密度或消息过程；
- **算法自定**：AP、MeanShift、HDBSCAN 等由内部判据产生，但仍有影响簇数的参数。

没有任何默认簇数或密度参数是数据中立的。`Birch()` 的 `n_clusters=3`、`KMeans` 的 `n_clusters=8`、`DBSCAN.eps=0.5` 都来自 1.9.1 API 签名，而不是推荐值。

## 缩放与度量

KMeans 的 inertia、DBSCAN 的 `eps`、RBF 亲和矩阵与层次聚类的度量都依赖尺度。应先判断各列是否可比较，必要时在训练过程内做缩放；不要用「聚类无监督」推出「预处理可以跳过」。Ward 只能使用欧氏距离；其他 linkage 可按数据结构选择 l1、l2、cosine 或预计算矩阵（[[sklearn-clustering]], L198–L204、L244–L251）。

## 指标的三个可判定性质

| 性质 | 含义 | 代表 |
| --- | --- | --- |
| 置换不变 | 改标签数值不改变结果 | 正规聚类指标都应具备，否则不适合评估划分 |
| 机会校正 | 随机划分附近是否被校准到零附近 | ARI、AMI；RI、普通 MI 不具备 |
| 对称 | 能否当共识度量，而非单向真值评分 | Rand/ARI、MI/NMI/AMI、V-measure；homogeneity/completeness 单独使用不对称 |

- **外部指标**：有真值时用 ARI/AMI 等回答「与参考划分多一致」。
- **内部指标**：silhouette、Calinski-Harabasz、Davies-Bouldin 只评价某种几何，不证明簇有业务意义，也不做机会校正。
- **两个矩阵**：contingency matrix 汇总真值簇 × 预测簇；pair confusion matrix 汇总样本对，不能直接混用。

V-measure 是 homogeneity 与 completeness 的调和平均，`beta` 控制二者权重（[[sklearn-clustering]], L542–L552）。当前 `average_method='arithmetic'` 的版本变更会影响 AMI/NMI 数值，旧 doctest 不能跨版本引用。

## 常见失效模式

- 把簇标签数值当作有语义类别；
- 没有 `predict` 却要求线上新样本打分；
- 忽略尺度与密度差异；
- 只报内部指标就宣称「分对了」；
- 把文档中的示例图或一次划分当成稳定性证据；
- 忽略 `assign_labels`、初始化和样本顺序；
- 把 `n_clusters=None` 与 BIRCH 默认全局聚类混为一谈；
- 使用旧 AMI doctest 作为当前版本预期值。

## 待解问题

- k-距离 knee 的可操作流程；
- 内部指标随 $k$ 的单调性与选择偏差；
- `discretize` 的随机性边界；
- OPTICS 与 DBSCAN 边界点差异；
- 大 $k$、大样本下 BisectingKMeans 的成本优势；
- 默认 RBF 亲和矩阵在未缩放数据上的奇异性。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-clustering]]
- 实践：[[clustering-selection-and-tuning]]、[[feature-selection-workflow]]、[[manifold-learning-selection-and-tuning]]
- 概念：[[manifold-learning]]、[[matrix-factorization]]、[[evaluation-metrics]]
