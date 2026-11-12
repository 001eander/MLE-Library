---
title: "聚类的选型与调参：先缩放与度量、再选族、最后定 $k$/$eps$；14 条写法坑与可自动化程度"
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
  - scikit-learn 1.9.1（文档论述与接口；默认值取自 1.9.1 API 页签名）
  - scikit-learn 1.6.1（本页标「实测」的行为；`HDBSCAN.copy`、`MDS` 一类默认值在 1.9/1.10 变过，引用时须带版本）
---

# 聚类的选型与调参

本页回答四个问题：**该不该用聚类、用哪个族、$k$ 或 `eps` 怎么定、结果怎么验**。机制与"几何假设"对照表在 [[clustering]]，来源、图读数与 20 条实测在 [[sklearn-clustering]]。引用写作 `[[sklearn-clustering]], L290`。

## 0. 先判该不该用

聚类最常被误用的场合是"没有标签但想要一个分类器"。先过这四问（本库判断，来源不提供这份清单）：

| 问题 | 若答案是「是」 | 后果 |
| --- | --- | --- |
| 下游需要**给新样本**打分吗？ | 只能选归纳式（`KMeans`/`MiniBatchKMeans`/`BisectingKMeans`/`AffinityPropagation`/`MeanShift`/`Birch`），或把传导式包一层（L43 示例） | `SpectralClustering`/`AgglomerativeClustering`/`DBSCAN`/`HDBSCAN`/`OPTICS` **没有 `predict`**（L39） |
| 你有**独立判据**说明"分对了"吗？ | 有 → 外部指标（ARI/AMI）；没有 → 只能内部指标 + 稳定性 | 无真值时的"好"是本库最容易被自欺的一步 |
| 簇的形状是**凸的**吗？ | 凸 → KMeans 系；任意形状/带噪声 → 密度系 | silhouette/CH/DB 三个内部指标都隐含凸性假设 |
| 每个簇的**密度相同**吗？ | 不同 → HDBSCAN（L302）或 OPTICS（L353） | DBSCAN 假设密度"globally homogeneous" |

如果四问都答不上来，**聚类输出不能当事实用**——它只能当探索工具。

## 1. 缩放与度量（硬前提，第一步）

文档在这一节**完全没有提缩放**（这是与 [[sklearn-clustering]] 局限里记录一致的缺口），但两条约束是硬的：

1. **基于距离与基于密度的算法对特征尺度敏感**。KMeans 的 inertia、DBSCAN 的 `eps`、`SpectralClustering` 的 `affinity='rbf'`（`gamma=1.0`）、`AgglomerativeClustering` 的 `metric` 全部吃原始尺度。**先 `StandardScaler`，除非有单位一致的物理意义**。这一条与 [[manifold-learning-selection-and-tuning]] 的实测同源（缩放使 Isomap 局部保真度 0.5029 → 0.8952），本页不重复测。
2. **Ward 只能用欧氏距离**：`AgglomerativeClustering(linkage='ward', metric='manhattan')` 直接 `ValueError`（实测 14）。非欧氏只能换 `average`/`complete`/`single`（L204）。

度量选择的官方准则只有一句（L251）："use one that maximizes the distance between samples in different classes, and minimizes that within each class"——**这句话预设了有类别标签**，无监督时只能当经验规则。两条具体经验：`l1` 适合稀疏特征（文本里稀有词的 occurrence，L248），`cosine` 对整体缩放不变（L250）。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

model = make_pipeline(StandardScaler(), KMeans(n_clusters=8, n_init=10, random_state=0))
labels = model.fit_predict(X)          # 归纳式：之后可 model.predict(X_new)
```

## 2. 选族：按"旋钮"而不是按"名字"

来源的总览表本来是为这件事设计的，但**剪藏丢了第一列（方法名）**（[[sklearn-clustering]] 分歧 6），所以实际可操作的判据只有两条（本库整理）：

| 你的约束 | 选 |
| --- | --- |
| 必须给 $k$，且 $n$ 很大 | `MiniBatchKMeans`（`batch_size=1024`、`n_init='auto'`）或 `BisectingKMeans`（大 $k$ 时） |
| 必须给 $k$，但要**层级** | `AgglomerativeClustering(linkage='ward')` 或 `BisectingKMeans`（L280） |
| 必须给 $k$，且簇是**图上切的** | `SpectralClustering`（少量簇，L144） |
| 不想给 $k$，数据中小 | `AffinityPropagation`（$O(N^2)$ 内存，L117）或 `MeanShift` |
| 任意形状 + 有噪声 + **密度一致** | `DBSCAN`（`eps` 是硬伤，L290） |
| 任意形状 + 有噪声 + **密度不一致** | `HDBSCAN`（只需 `min_cluster_size`，L341） |
| 要**事后**用不同 `eps` 看结果 | `OPTICS` + `cluster_optics_dbscan`（L353） |
| 数据大到放不进内存 | `Birch(n_clusters=None)` 拿子簇（L377），或 `MiniBatchKMeans` |

**不要按名字选**：`Birch` 与 `MiniBatchKMeans` 都是"大数据"路线，但 BIRCH 在高维上不占优——官方丢失的折叠块写着"if `n_features` is greater than twenty, it is generally better to use MiniBatchKMeans"（外部核查，剪藏不含）。

## 3. 定 `k` 与定 `eps`

### 3.1 定 `k`（有 `n_clusters` 的算法）

三条路，按可靠性排序：

1. **业务/下游判据**（最可靠）：下游模型在 $k\in\{3..12\}$ 上的 CV 分数曲线；聚类只是中间步骤时，用最终指标选 $k$，而不是用 silhouette。
2. **silhouette 曲线**（官方示例 `plot_kmeans_silhouette_analysis`，L303 附近引用）：对每个 $k$ 画各样本的 silhouette 并算均值。**注意它随 $k$ 无单调性保证，且偏好凸簇**——只在你已认定凸簇时用它。
3. **稳定性/重采样**：多个种子或多次子采样后，用 ARI/AMI 比较两次聚类结果的一致性（这两个指标**机会校正且对称**，L395、L448）。这是本库推荐但本节文档**完全没有**的方法。

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

for k in range(2, 11):
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
    ref = KMeans(k, n_init=10, random_state=1).fit_predict(Xs)
    print(k, silhouette_score(Xs, labels), adjusted_rand_score(labels, ref))
```

**不要用 `calinski_harabasz_score` 选 $k$**：它是簇间/簇内离散度之比，随 $k$ 增大通常单调上升（[[clustering]] 第二节）。它只适合**固定 $k$ 时比较两个划分**。

### 3.2 定 `eps`（DBSCAN / OPTICS）

`eps` 是本节唯一被文档点名"不能留默认值"的参数（L290，默认 0.5）。四条落地做法：

1. **k-距离曲线**：对每个点算到第 `min_samples` 近邻的距离，排序后画曲线，取**拐点**（knee）。这是 L290 唯一给的启发式。
2. **扫一遍看噪声比例**：`eps` 太小 → 大量 `-1`；太大 → 全并成一个簇。实测 iris（`min_samples=5`）：0.05 → 0/2/2/1 个簇，噪声 150/17/0/0（[[sklearn-clustering]] 实测 8）。
3. **用 OPTICS 一次性看完**：`OPTICS(max_eps=np.inf)` 然后用 `cluster_optics_dbscan(eps=...)` 对多个 `eps` 抽取（L353）。实测在 iris 上 `eps=0.5` 的结果与 `DBSCAN(eps=0.5, min_samples=5)` 完全一致。
4. **HDBSCAN 代替手工扫**：`min_samples=None` 等价于 `min_cluster_size`（外部核查），只需调 `min_cluster_size`；实测两团数据上 5/20/50 → 噪声 0/11/24。

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

k = 5
d, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
kdist = np.sort(d[:, -1])
# 取曲率最大点（或用分位数）作为 eps 起点，再在 ±30% 上扫
eps0 = kdist[int(0.9 * len(kdist))]
for eps in eps0 * np.array([0.7, 0.85, 1.0, 1.2, 1.5]):
    lab = DBSCAN(eps=eps, min_samples=k).fit_predict(Xs)
    print(round(eps, 4), len(set(lab) - {-1}), (lab == -1).sum())
```

## 4. 评估：先分"有没有真值"

| 情形 | 用什么 | 必须注意 |
| --- | --- | --- |
| **有真值**（有标签的探索、或模拟数据） | **ARI**、**AMI**（唯一两个做了机会校正的）；需要"纯/全"分开时用 `homogeneity_completeness_v_measure` | homogeneity 与 completeness **不对称**（L574）；AMI 的 `average_method` 默认 0.22 起是 `'arithmetic'`，**旧结果不可比** |
| **无真值** | `silhouette_score`、`davies_bouldin_score`（越小越好）、`calinski_harabasz_score`（只用于比较同一 $k$ 的两个划分） | 三者都**隐含凸簇假设**，对密度聚类不公平；都**不做机会校正** |
| **诊断"哪个类被切碎/混类"** | `contingency_matrix`（L729）、`pair_confusion_matrix`（L758） | 后者**不对称**（L791）；前者给的是充分统计量 |
| **报告显著性** | 无现成工具 | 本节文档不给，本库也未覆盖 |

三条硬规则：

1. **不要只报一个数字**。至少报「外部（若有）+ 内部 + 噪声比例 + 簇大小分布」。
2. **V-measure 与 NMI（算术平均）是同一个数**（实测逐位相同 0.5158037429793889），同时报两个等于重复。
3. **不要跨 $k$、跨算法比较内部指标**：silhouette 的"次近簇"定义随 $k$ 变，DB 的散度随簇形状变。

```python
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             silhouette_score, davies_bouldin_score)

def report(Xs, labels, y_true=None):
    n_clusters = len(set(labels) - {-1})
    out = {"k": n_clusters, "noise": int((labels == -1).sum())}
    if n_clusters > 1:
        out["silhouette"] = round(silhouette_score(Xs, labels), 4)
        out["davies_bouldin"] = round(davies_bouldin_score(Xs, labels), 4)
    if y_true is not None:
        out["ARI"] = round(adjusted_rand_score(y_true, labels), 4)
        out["AMI"] = round(adjusted_mutual_info_score(y_true, labels), 4)
    return out
```

## 5. 14 条写法坑

按"写代码时最容易踩"排序（机制层面的 18 条清单在 [[clustering]]）：

1. **`n_init` 不显式给**：1.4 起默认 `'auto'`，KMeans 配 `k-means++` 时**只跑 1 次**。实测 10 个种子上 inertia 32.77±2.06（最差 35.76），而 `n_init=10` 是 30.21±0.19。**要可复现就写死 `n_init=10` 与 `random_state`**。
2. **`MiniBatchKMeans.tol=0.0` 是默认**：早停只由 `max_no_improvement=10` 控制，`tol` 形同关闭。
3. **`Birch()` 默认 `n_clusters=3`**：不写 `n_clusters=None` 就会多做一次全局聚类，簇数被静默定成 3（实测：iris 上 `None` → 13 个子簇，默认 → 3）。
4. **`DBSCAN(eps=0.5)` 是默认**，而文档说不能留（L290）。
5. **DBSCAN 的标签受样本顺序影响**（边界点归属）：换 `train_test_split` 的顺序就可能换标签；要可复现就固定行序（实测复现：`A,M,B` 与 `B,M,A` 给不同标签）。
6. **`SpectralClustering(affinity='rbf', gamma=1.0)` 是默认**：文档警告相似度分布不好会让问题奇异（L153）；未缩放数据上尤其容易触发。要么显式 `affinity='nearest_neighbors'`，要么先缩放。
7. **`assign_labels='discretize'` 会吃 `random_state`**：写 `random_state=0` 才能复现（实测：同一嵌入、`random_state` 0 与 1 给不同标签）。
8. **`assign_labels='kmeans'` 的簇编号不稳定**：划分可复现、标签值不可复现。跨运行比标签必须用 ARI/AMI，不能比 `labels_`。
9. **`SpectralClustering.n_init=10` 只在 `assign_labels='kmeans'` 时生效**：换成 `discretize`/`cluster_qr` 后这个参数无用。
10. **`AgglomerativeClustering` 想用 `compute_distances=True` 就必须 `n_clusters=None`**（否则报错），要画树状图时注意。
11. **`MeanShift` 没有 `bandwidth_` 属性**：估出来的带宽事后读不到；要报告就自己调 `estimate_bandwidth(X, quantile=0.3)` 并保存。
12. **`AffinityPropagation(preference=None)` = 相似度中位数**：换数据就换簇数；想控制簇数就显式给 `preference`（iris 上 `None`→7 簇、`-10`→5 簇）。
13. **`HDBSCAN(min_samples=None)` 等于 `min_cluster_size`**：以为"没设 min_samples 就是自动"会误判参数耦合。
14. **`HDBSCAN.copy` 在 1.10 会从 `False` 变 `True`**（1.9.1 显示 `'warn'`）：跨版本行为会变，涉及 `metric='precomputed'` 的原地修改。

## 6. 可自动化程度

| 步骤 | 可自动化 | 条件 |
| --- | --- | --- |
| 缩放 | **完全** | `StandardScaler` 放进 `Pipeline`；不允许关闭（除非单位一致） |
| 选族 | **部分** | 两问可判定（凸/任意形状、是否需要 `predict`）；其余靠扫描 |
| `k` 扫描 | **完全** | 需要同时报 silhouette 与 ARI 稳定性，单一指标会误导 |
| `eps` 扫描 | **完全** | k-距离 knee + 多档扫描 + 噪声比例；需固定行序 |
| 评估（有真值） | **完全** | ARI/AMI + 混淆矩阵；注意 `average_method` 版本 |
| 评估（无真值） | **部分** | 内部指标可算，但"多少算好"依赖数据；建议加稳定性 ARI |
| 结果解释 | **不可** | 簇编号无排序语义；簇的"含义"需要人工命名 |
| 上线（传导式） | **不可** | 无 `predict`，必须重跑或包一层归纳化（L43） |

## 待解问题

1. **`eps` 的 knee 到底怎么取**？本页给的是分位数 + 多档扫描的工程近似，来源只写"基于最近邻距离图的 knee"（L290）。
2. **CH 是否随 $k$ 单调上升**未系统验证，本页只按"不可用于选 $k$"处理。
3. **密度聚类的内部评估**：silhouette 的凸性假设与 DBSCAN 冲突，本库没有替代指标的来源。
4. **跨版本的 AMI 数字**：`average_method` 0.22 变更后，任何 0.22 之前的 AMI 结论都需要重跑。
5. **1.9.1 上未复跑**：本页所有「实测」都是 1.6.1；`HDBSCAN.copy`、`set_output`、`KMeans.algorithm='elkan'` 的行为未核。

## 关联页面

- 概念：[[clustering]]（几何假设、$k$ 的三个来源、指标三性质、18 条失效模式）
- 来源：[[sklearn-clustering]]（23 张图、20 条实测、8 条分歧、版本判定到 1.9.1）
- 工具：[[scikit-learn]]（`cluster` 模块的估计器矩阵与默认值清单）
- 相邻实践：[[manifold-learning-selection-and-tuning]]（同样先缩放；换坐标而非分组）、[[nearest-neighbors-selection-and-tuning]]（`eps` 半径查询与 `connectivity` 矩阵的来源）、[[feature-selection-workflow]]（另一条无监督路线：删列）
