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
  - scikit-learn 1.9.1（文档论述与接口，见 [[sklearn-clustering]]）
  - scikit-learn 1.6.1（本页标「实测」的运行时行为）
---

# 聚类

## 一句话定义与三个推论

**定义（来源论述）**：聚类是对无标签数据做的分组；`sklearn.cluster` 里每个算法有两种形态——类（`fit` 后读 `labels_`）与函数（直接返回整数标签数组）。[[sklearn-clustering]], L10–L12

这句话里有三个可以直接用于工程判断的推论（本库综合）：

1. **没有 $y$，也没有"正确答案"**。"Evaluating the performance of a clustering algorithm is not as trivial as counting the number of errors or the precision and recall of a supervised classification algorithm"[[sklearn-clustering]], L381——因为标签的**数值本身无意义**（第 0 类和第 1 类互换后结果相同）。所以聚类**先要回答"我用什么标准判断好坏"**，才轮到选算法。这与 [[manifold-learning]]（输出坐标，好坏靠"保住了什么"）是同一类问题，但判据不同。
2. **输出的粒度是"每个样本一个整数"，不是特征**。因此聚类可以出现在管线**末端**（当预测用）或**中段**（当离散化/压缩用，如 BIRCH 的 data reduction，L377），但不能像 [[feature-selection]] 那样直接喂给下游模型当特征——除非再做 one-hot。
3. **「能不能给新样本打标签」是硬断层**。`KMeans`/`MiniBatchKMeans`/`AffinityPropagation`/`MeanShift`/`Birch`/GMM 有 `predict`；`SpectralClustering`/`AgglomerativeClustering`/`DBSCAN`/`HDBSCAN`/`OPTICS` **没有**，文档明说传导式方法"not designed to be applied to new, unseen data"[[sklearn-clustering]], L39。这是本库第二个"同一类任务被 API 一刀切开"的例子（第一个是 [[manifold-learning]] 里 `transform` 的有无）。

## 判别方法的第一性判据：假设了什么样的簇

这是本页的核心表。**方法名几乎不携带信息，几何假设才是判据**（来源把假设分散在各小节，从不并列；本库合并）：

| 方法 | 必须指定的旋钮 | 假设的簇 | 归纳/传导 | 成本证据（来源/实测） |
| --- | --- | --- | --- | --- |
| `KMeans` | `n_clusters` | **凸 + 各向同性 + 等方差**（inertia 的隐含假设，L59） | 归纳 | 快；本身 $O(nkdi)$ 量级，官方图 1 面板 0.00–0.02s（本库目视） |
| `MiniBatchKMeans` | `n_clusters` | 同上（同一目标函数，L95） | 归纳 | 小数据上比 KMeans 快 3–24×，inertia 差 3.8–5.4%（本机实测） |
| `BisectingKMeans` | `n_clusters` | 同上，但**分裂式**（L261） | 归纳 | digits 上比 KMeans 差约 6% inertia、0.09–0.10s（本机实测） |
| `AffinityPropagation` | **无**（`preference`/`damping`） | exemplar（真实样本点）为中心 | 归纳 | $O(N^{2}T)$ 时间、$O(N^{2})$ 内存（L117）——只能中小数据 |
| `MeanShift` | **无**（`bandwidth` 可估） | 平滑密度的众数（blob，L126） | 归纳 | 反复最近邻搜索，不可扩展（L130） |
| `SpectralClustering` | `n_clusters` | **图上割**（normalized cut，L146） | **传导** | 稀疏亲和 + `amg` 时高效（L142）；图 1 面板 0.01–0.02s |
| `AgglomerativeClustering` | `n_clusters` 或 `distance_threshold` | 由 `linkage` 决定：Ward＝方差、complete＝最远、average＝平均、single＝最近（L191–L194） | **传导** | 无连通性约束时"computationally expensive"（L196）；加约束变快（实测 8.7→5.5ms） |
| `DBSCAN` | `eps` + `min_samples` | **任意形状的高密度连通区**（L284） | **传导** | 有噪声标签 `-1`；`eps` 敏感（实测 4 档） |
| `HDBSCAN` | `min_cluster_size`（+`min_samples`） | 同上，但**允许密度不同**（L302） | **传导** | 无 `eps`；MST 剪枝（L332） |
| `OPTICS` | `max_eps`（默认 ∞）+ `xi` | 同上，`eps` 从单值放松成区间（L353） | **传导** | 可对任意 `eps` 事后抽取（线性时间） |
| `Birch` | `threshold` + `branching_factor`（+`n_clusters`） | CF 子簇（球状、阈值内） | 归纳 | **数据缩减**：有损压缩后可直接用或再聚（L377） |
| `GaussianMixture`（2.1 节，未入库） | `n_components` | 高斯混合；KMeans 是等协方差特例（L37） | 归纳 | 有软分配与密度估计能力 |

由此得到四条互不相同的**目标函数**（本库综合，来源分散在 L51–L55、L284–L325、L146、L191）：

1. **簇内平方和**（inertia）：越小越好，0 最优、**没有上界也没归一化**（L60）。
2. **密度连通**：不是优化一个全局目标，而是"在 `eps` 半径内数够 `min_samples` 个点"的**判定 + 传播**（L286）。HDBSCAN 把 $\epsilon$ 的所有取值一次性做成层次（L341），于是"目标"变成剪枝规则。
3. **图割**：最小化"割边权重 / 簇内权重"（L146）——这是唯一一个**显式带图上正则化**的目标。
4. **方差最小化**（Ward，L191）：与 inertia 同族，但用层次合并实现，因此可以用树状图读出多尺度结构（L187、L213）。

## 「$k$ 从哪来」是本节第一个工程问题

三种来源，代价完全不同（本库综合）：

| 来源 | 方法 | 代价 |
| --- | --- | --- |
| **用户给定** | KMeans/MiniBatch/Bisecting/Spectral/Agglomerative | 需要外部判据（业务约束、silhouette 曲线、稳定性），文档**不给**判据 |
| **由密度阈值间接决定** | DBSCAN（`eps`+`min_samples`）、HDBSCAN（`min_cluster_size`）、OPTICS（`xi`）、BIRCH（`threshold`） | 旋钮从"个数"换成"尺度"；`eps` 被文档点名"usually cannot be left at the default value"（L290） |
| **算法自己决定** | AffinityPropagation（`preference`）、MeanShift（`bandwidth`） | 看似自由，实际把自由度转移到了 `preference`/`bandwidth` 上（见 [[sklearn-clustering]] 分歧 8） |

一个必须记住的算术事实：**`eps` 的敏感性是可量化的、`k` 的敏感性不是**。iris 上 `min_samples=5`，`eps` 从 0.05 → 10 会经历"0 个簇 / 2 个簇 / 2 个簇 / 1 个簇"四段（本机实测），而 `n_clusters` 从 2 → 10 永远返回 10 个非空簇——**KMeans 不会告你 $k$ 错了，DBSCAN 会**（用 `-1` 标签告你）。这是密度方法相对 KMeans 的唯一"自诊断"能力，也是它难调的原因。

## 归纳 vs 传导：断层在哪

| 有 `predict` | 没有（只能 `fit_predict`） |
| --- | --- |
| `KMeans`、`MiniBatchKMeans`、`BisectingKMeans`、`AffinityPropagation`、`MeanShift`、`Birch`、`GaussianMixture` | `SpectralClustering`、`AgglomerativeClustering`、`DBSCAN`、`HDBSCAN`、`OPTICS` |

这条断层的工程后果与 [[manifold-learning]] 里"有没有 `transform`"完全同构，但更硬：

- **传导式方法不能进"训练一次、推理多次"的线上路径**。测试集来了要重跑全量数据，簇编号还会变（同一结构的不同标号）。
- 官方给的桥接方式是**把聚类器包一层**：`plot_inductive_clustering` 示例（L43）——先用无 `predict` 的算法在训练集上聚类，再训一个分类器把标签"蒸"成归纳式的。本节只给链接，不给方法论。
- **`MeanShift` 的"归纳"是名义上的**：L132 说"Labelling a new sample is performed by finding the nearest centroid"，即**新样本只找最近质心，不参与密度漂移**——这与 `KMeans.predict` 的行为同构，但代价模型完全不同（训练时做过多次最近邻搜索）。

## 聚类与"相邻类目"的定位

| 类目 | 输出 | 评估方式 | 本库页面 |
| --- | --- | --- | --- |
| 特征选择 | **列的子集** | 下游任务指标 + 稳定性 | [[feature-selection]] |
| 流形学习 | **新坐标**（每维有数值语义） | 保真度（trustworthiness、全局相关） | [[manifold-learning]] |
| **聚类** | **整数标签**（无数值语义） | 有真值→外部指标；无真值→内部指标/稳定性 | 本页 |
| 分类/回归 | 预测值 | 准确率/损失 | [[regularization-and-sparsity]] 等各模型族 |

**关键区别**：前三者都是无监督，但只有聚类的输出**连排序都没有**。因此"类别 0 比类别 1 大"这句话在聚类里永远无意义，而流形学习的第 1 维可以比较大小。这条差别决定了两者的评估方式不能互换：流形学习要看"距离是否被保住"，聚类要看"划分是否一致"。

## 指标：三个可判定的性质

§2.3.11 给了 9 个指标 + 2 个矩阵。它们能被三个二值性质完全刻画（本库整理，来源把这些性质散落在各处）：

| 性质 | 含义 | 满足 | 不满足 |
| --- | --- | --- | --- |
| **置换不变** | 同时重命名两边的标签，分数不变 | **全部 9 个**（L385、L412、L470） | — |
| **机会校正** | 随机划分的期望分数 ≈ 0 | **ARI**（L395）、**AMI**（L448） | RI（L395："does not ensure … close to 0.0"）、MI/NMI、homogeneity/completeness/V-measure、FMI |
| **对称** | 交换 `labels_true` 与 `labels_pred` 分数不变 | ARI、RI、MI/NMI/AMI、**V-measure**、FMI（L412、L470、L574） | **homogeneity / completeness**（`homogeneity_score(a,b) == completeness_score(b,a)`，L574–L589） |

三个直接推论（本库综合）：

1. **只有 ARI 与 AMI 适合"基线校正"**。想回答"这比随机好多少"，只能用这两个；RI/NMI/V-measure/FMI 都可能给一个非零的"随机分数"，且**随 $n$ 与簇数变化**（官方示例 `plot_adjusted_for_chance_measures` 专测这件事，剪藏里这张图**缺失**）。
2. **只有对称指标能当"共识度量"**（consensus measure，L412）：比较两个独立聚类结果时用 ARI/AMI/NMI/V-measure/FMI；比较"聚类 vs 真值"时用 homogeneity/completeness 才有方向含义（谁纯、谁全）。
3. **homogeneity 与 completeness 是"一对相反的错误"**：homogeneity 高＝每个簇很纯（但一个类可能被切碎），completeness 高＝每个类完整（但一个簇可能混类）。V-measure 是两者的调和平均（L538），$\beta$ 控制偏向（L542）。

**内部指标（无真值）的三个已知盲点**（本库综合，来源只在描述里暗示）：

| 指标 | 方向 | 盲点 |
| --- | --- | --- |
| `silhouette_score` | 越大越好（−1..1） | 基于"到次近簇的平均距离"，**偏好凸且分离的簇**；对密度不均与流形形状无感；$O(n^2)$ 距离计算（`sample_size=None` 时不采样） |
| `calinski_harabasz_score` | 越大越好 | 是**簇间/簇内离散度之比**，随 $k$ 增大通常单调上升，**不能直接拿来选 $k$** |
| `davies_bouldin_score` | **越小越好**（0 最优） | 用"簇内散度 / 簇间距离"的最大比值，同样偏好球状簇 |

这三个指标**都不做机会校正**，且**都只在"簇是凸的、用欧氏距离"的前提下有意义**——恰恰是 DBSCAN/HDBSCAN 想避开的假设。因此密度聚类的结果用 silhouette 评估，等于用 KMeans 的世界观给它打分（本库判断）。

## 失效模式清单

按"踩到才发现"排序：

1. **`eps` 留在默认 0.5**：文档自己说不能留（L290）。实测 iris 上 0.05 会让全部 150 个点变噪声。
2. **DBSCAN 的边界点归属依赖样本顺序**：同一份数据、同一组参数，换个行序标签就变（本机复现；这条只在官方丢失的 `Implementation` 折叠块里）。
3. **`n_init='auto'` = 只跑一次**（1.4 起）：KMeans 配 `k-means++` 时为 1 次，实测 inertia 标准差 2.06 vs `n_init=10` 的 0.19。
4. **`Birch()` 默认 `n_clusters=3`**：不显式设 `None` 就会多做一次全局聚类，簇数被静默定成 3。
5. **`SpectralClustering(affinity='rbf')` 是默认**（`gamma=1.0`）：未缩放数据上容易撞上"相似度分布不好 → 问题奇异"的警告（L153）。
6. **`assign_labels='discretize'` 不是"100% 可复现"**（L168）：它吃 `random_state`，`None` 时可能 run-to-run 不同（本机复现）。
7. **`assign_labels='kmeans'` 的簇编号不稳定**：即使划分相同，标签值也会变（本机实测）。
8. **Ward 只能用欧氏距离**：配曼哈顿直接 `ValueError`（L204 的"cannot be varied"是硬约束）。
9. **连通性约束 + single/average/complete 会放大"富者愈富"**（L234）：小 $k$ 时出现"一个巨大簇 + 几个近乎空的簇"。
10. **single linkage 的簇大小极不均**：iris 上 `[98, 50, 2]`（本机实测）。
11. **`MeanShift` 没有 `bandwidth_` 属性**：估出来的带宽事后读不到，必须自己调 `estimate_bandwidth` 存下。
12. **`AffinityPropagation(preference=None)` = 相似度中位数**：这个"自动"决定了簇数，且 $O(N^2)$ 内存让它只能中小数据。
13. **AMI/NMI 的 `average_method` 默认 0.22 起是 `'arithmetic'`**：旧代码/旧文档里的 AMI 数字不可比（剪藏的两个 doctest 就是旧口径）。
14. **homogeneity 与 completeness 不能混用**：一个高一个低是常态（L566–L571 的例子），报单个数字会误导。
15. **V-measure 与 NMI（算术平均）是同一个数**：本机实测两函数返回逐位相同值，同时报两个等于重复。
16. **silhouette 不能跨 $k$ 比较**：它随 $k$ 单调性无保证，且不同 $k$ 下的"次近簇"定义都不同。
17. **内部指标对密度聚类不公平**：凸簇假设与 DBSCAN 的任意形状假设冲突。
18. **传导式方法不能直接上线**：`SpectralClustering`/`AgglomerativeClustering`/`DBSCAN`/`HDBSCAN`/`OPTICS` 都没有 `predict`。

## 与其他概念页的关系

- [[manifold-learning]]：同是无监督，但那边**换坐标**、这边**分组**；共同底座是 kNN 图（`SpectralClustering` 的 `nearest_neighbors` 亲和、`DBSCAN`/`OPTICS` 的邻居搜索后端、`connectivity` 矩阵）。因此**缩放**在这里同样重要，但本节**一个字都没提**（文档缺口，见 [[sklearn-clustering]] 局限）。
- [[nearest-neighbors]]：`eps` 半径查询与 `kneighbors_graph` 的 `connectivity` 矩阵都来自这个模块；DBSCAN 的 `algorithm`/`leaf_size` 与 KNN 完全同源。
- [[feature-selection]]：另一条"无监督降维"路线——删列不改行，聚类不改列只改行。
- [[ensembles]]：`BisectingKMeans` 是"用层次结构组织同一目标"的案例，与 GBDT 的"顺序纠错"结构同构（都是把一个大问题拆成小问题），但优化目标单一。
- [[gaussian-discriminant-analysis]]：KMeans = 等协方差高斯混合的硬分配（L37），因此"LDA 是生成式、KMeans 是它的无监督退化"这条链可以从 [[gaussian-discriminant-analysis]] 接过来。

## 待解问题

1. **"该用哪个算法"在本库还没有判据**：来源的总览表给的是描述而非规则；唯一可操作的是"簇是凸的还是任意形状的"与"要不要给新样本打标签"这两问（本页第二节与第三节）。有没有更细的判据（如按噪声比例、按簇密度差异）？
2. **内部指标与 $k$ 的单调关系未量化**：CH 是否随 $k$ 单调上升？若成立，则"用 CH 选 $k$"是错的，而文档不说。
3. **密度聚类怎么评估**：silhouette 的凸性假设与 DBSCAN 冲突，那应该用什么？（DBCV 一类指标本库无来源覆盖。）
4. **传导式方法包成归纳式**（L43 示例）的正确写法与代价未考察。
5. **聚类结果的不稳定性**在本库只有两个数字（DBSCAN 顺序依赖、`n_init='auto'`），缺少种子/重采样扫描的系统方法——这与 [[leaderboard-vs-cv]] 里"代理指标"的讨论同源但没人接。
