---
title: "scikit-learn 用户指南 2.3「Clustering」：k-means 家族、密度聚类、层次聚类、BIRCH 与 9 个评估指标（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - clustering
  - unsupervised
  - reference-doc
  - evaluation
source_path: raw/articles/sklearn/2.3. Clustering.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 用户指南 2.3「Clustering」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 2.3. Clustering（scikit-learn 用户指南第 2.3 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**35 个代码块／92 行 doctest、7 个行间公式、5 个 admonition（2 个 `[!warning]` + 3 个 `[!note]`）、23 张图（官方 26 张）、11 个二级小节 + 20 个子小节** |
| 原文 | <https://scikit-learn.org/stable/modules/clustering.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/2.3. Clustering.md`（804 行、71,859 字节、`wc -w` 7,867 词；**未做任何回填**） |
| SHA-256 | `12e437b55a1b1e3d6ee1d39106a5a9942f74a56707f742be01448f949cac2aa8` |
| 完整性 | 正文（散文 + doctest + 7 个行间公式）逐句保留，但：**官方 30 个内容 `<details>` 折叠块全部丢失（3,073 词）**、**官方 16 个 References 折叠块内容全部丢失（剪藏只剩 2 处空标题 L349、L587）**、**总览表第一列（12 个方法名）整列丢失**、26 张图少 3 张。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x，工作版本 1.9.1**（见「版本判定」） |
| 插图 | 23 张已于 2026-11-12 下载入库 `raw/assets/sklearn-clustering/`（官方 26 张），其中 8 张逐张目视核查（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-clustering]], L128`。凡超出剪藏内容的部分（API 默认值、版本史、丢失折叠块的原文）都显式标注「外部核查」或「官方 1.9.1 页」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `12e437b5…`）。丢失的 30 个折叠块、16 个参考文献块与总览表首列只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库**第一个「无监督、输出标签」的类目**，也是**第一个自带评估指标的 sklearn 章节**。它对知识库的意义有六层。

1. **第一次出现「没有 $y$、也没有坐标」的输出形态。** 上一节 [[manifold-learning]] 输出的是**一套新坐标**，本节的估计器输出的是**每个样本的整数标签**（`labels_`，L12）。因此本库现在有三类无监督产物：「删列」（[[feature-selection]]）、「换坐标」（[[manifold-learning]]）、「分组」（本页）。三者的评估方式完全不同：分组结果**对标签置换不变**，所以不能数准确率（L385、L412）。
2. **第一次把「必须给 $k$」与「自己定 $k$」写成选型轴。** KMeans/MiniBatch/BisectingKMeans/Spectral/Agglomerative 要 `n_clusters`；DBSCAN/OPTICS/HDBSCAN/BIRCH 用 `eps`/`max_eps`/`min_cluster_size`/`threshold` 间接定簇数；Affinity Propagation 与 MeanShift 完全不需要（L115、L128）。**「$k$ 从哪来」是本节的第一个工程问题**，而文档只在 DBSCAN 一节给了唯一一条具体建议（最近邻距离的 knee，L290）。
3. **第一次把「归纳 / 传导」写成一等公民**（L39）：`KMeans`、`MiniBatchKMeans`、`AffinityPropagation`、`MeanShift`、BIRCH、GMM 有 `predict`，能给新样本打标签；`SpectralClustering`、`AgglomerativeClustering`、`DBSCAN`、`HDBSCAN`、`OPTICS` **没有 `predict`**，文档明说它们「not designed to be applied to new, unseen data」（L39）。这与 [[manifold-learning]] 里「有没有 `transform`」是同一类断层，但这次断层直接决定**能不能上线**。
4. **第一次出现「四种互不相同的几何假设」并列**：凸＋各向同性（inertia，L59）、密度连通（core sample / mutual reachability，L284–L325）、图割（normalized cut，L146）、方差最小化（Ward，L191）。同一份数据在这四种假设下会得到四个不同的正确答案——这正是总览表（L18–L29）想表达但**首列丢失后读不出来**的东西。
5. **第一次在 sklearn 来源里出现「评估」这件事本身**：9 个指标分成「有真值」（Rand/ARI、MI/NMI/AMI、homogeneity/completeness/V-measure、FMI）与「无真值」（silhouette、Calinski-Harabasz、Davies-Bouldin）两类，另有 2 个诊断工具（contingency matrix、pair confusion matrix）。**机会校正**（ARI、AMI）与**对称性**（ARI/AMI/NMI/FMI/V-measure 对称，homogeneity/completeness 不对称，L574）是本库第一次把「指标能不能当共识度量」写成可判定性质。
6. **这是本库目前「正文完整、数学全丢」最极端的一份剪藏**：所有 `Mathematical formulation`（5 个）、`Algorithm description`（AP、BIRCH）、`Implementation`（DBSCAN）、`Comparison with DBSCAN`（OPTICS）、`Computational Complexity`（OPTICS）与 16 个 References 都在折叠块里（3,073 词），剪藏一个都没有。因此**本节的可引用部分几乎只有「散文级主张 + doctest 数字」**，任何公式级结论都必须引官方页。（上一节 [[sklearn-manifold-learning]] 是「散文全在、数学全丢」，本节连散文的一部分也丢了：总览表首列。）

## 文档结构

| 章节 | 解决什么问题 | 关键参数 | 行号 |
| --- | --- | --- | --- |
| 引子 | 模块名与「类 + 函数」双 API；`labels_` | — | L10–L12 |
| 2.3.1 Overview | 12 个方法的方法名／参数／可扩展性／用途／几何对照表 | — | L14–L43 |
| 2.3.2 K-means | inertia 目标、Lloyd 三步、Voronoi 读法、k-means++、`sample_weight` | `n_clusters`、`init`、`n_init`、`tol` | L45–L82 |
| 2.3.2.1 低层并行 | OpenMP、256 样本分块 | — | L84–L91 |
| 2.3.2.2 Mini Batch | 小批量 + 流式均值更新；质量略降 | `batch_size`、`max_no_improvement` | L93–L107 |
| 2.3.3 Affinity Propagation | 样本间消息传递选 exemplar；自行定簇数 | `preference`、`damping` | L109–L122 |
| 2.3.4 Mean Shift | 密度众数漂移；`estimate_bandwidth` | `bandwidth` | L124–L138 |
| 2.3.5 Spectral | 亲和矩阵低维嵌入 + 二次聚类；normalized cut | `n_clusters`、`affinity`、`assign_labels` | L140–L164 |
| 2.3.5.1 label 分配 | `kmeans`（不稳）/ `discretize`（称 100% 可复现）/ `cluster_qr`（确定性） | `assign_labels` | L166–L172 |
| 2.3.5.2 图分割 | `affinity='precomputed'` 吃邻接矩阵 | — | L174–L183 |
| 2.3.6 层次聚类 | 自底向上合并；4 种 linkage；连通性约束；度量；Bisecting K-Means | `linkage`、`connectivity`、`metric`、`distance_threshold` | L185–L280 |
| 2.3.7 DBSCAN | 密度连通 + 核心样本；`eps` 是关键 | `eps`、`min_samples` | L282–L298 |
| 2.3.8 HDBSCAN | 互达距离图 + MST + 全 $\epsilon$ 层次的 DBSCAN\* | `min_cluster_size`、`min_samples` | L300–L349 |
| 2.3.9 OPTICS | reachability 图，把单一 `eps` 放松成区间 | `max_eps`、`xi`、`cluster_method` | L351–L361 |
| 2.3.10 BIRCH | CF 树、有损压缩、可选全局聚类器 | `threshold`、`branching_factor`、`n_clusters` | L363–L377 |
| 2.3.11 评估 | 为什么不能数准确率；9 个指标 + 2 个矩阵 | `metric`、`sample_size`、`average_method` | L379–L804 |
| （丢失 ×14） | 5 个数学表述、AP/BIRCH 算法描述、DBSCAN 实现与内存、OPTICS 对比与复杂度、BIRCH 三问 | — | 官方页；剪藏无 |

## 关键主张（按剪藏顺序）

### A. 双 API 与总览（L10–L43）

- **「每个算法有两种形态」**：类（`fit` + `labels_`）与函数（给训练数据直接返回整数标签数组）（L12）。这条解释了为什么同一算法在文档里出现两次名字（`KMeans` / `k_means`、`AgglomerativeClustering` / `ward_tree`……）。
- 总览表 12 行（L18–L29）按「参数／可扩展性／用途／几何」四列组织。**剪藏丢了第一列，所以现在只能靠图与后文倒推哪一行是谁**（见「分歧」6）。官方表的方法名依次是 K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、HDBSCAN、OPTICS、Gaussian mixtures、BIRCH、Bisecting K-Means（外部核查：官方 1.9.1 页同一张表的 12 行逐行核对）。
- **非平坦几何**（L35）："Non-flat geometry clustering is useful when the clusters have a specific shape, i.e. a non-flat manifold, and the standard euclidean distance is not the right metric. This case arises in the two top rows of the figure above." —— 图 1 的前两行正是同心圆与双月。
- **KMeans 是 GMM 的特例**（L37）："KMeans can be seen as a special case of Gaussian mixture model with equal covariance per component." 这条把本节与 2.1 节（GMM，未入库）接上，也是"硬分配 vs 软分配"的最短解释。
- **归纳 vs 传导**（L39）：传导式方法「not designed to be applied to new, unseen data」。官方另给一个 `plot_inductive_clustering` 示例（L43），演示如何把没有 `predict` 的聚类器包成归纳式。

### B. K-means 家族（L45–L107）

- **目标就是 inertia**（within-cluster sum-of-squares，L51–L55 的公式；"Inertia can be recognized as a measure of how internally coherent clusters are"，L57）。
- **inertia 的两个已知缺陷**（L59–L60）：① 隐含「凸且各向同性」，对细长簇与不规则流形响应差；② 不是归一化指标（只有"越小越好、0 最优"），高维下欧氏距离膨胀（维度灾难），官方给的缓解办法是**先降维**（PCA）再聚类（L60）——这是本库第一条「降维 → 聚类」的官方建议，而 2.2 节（[[manifold-learning]]）从未回头提聚类。
- **三处实现细节**：Lloyd 三步（选初始中心 → 分配到最近中心 → 取均值更新）（L65）；"K-means is equivalent to the expectation-maximization algorithm with a small, all-equal, diagonal covariance matrix"（L69）；**停止条件是"质心移动小于容差"，而不是"目标函数相对下降小于容差"**——文档明确对比了这两种惯例（L71）。
- **局部极小与 k-means++**（L73）："Given enough time, K-means will always converge, however this may be to a local minimum. This is highly dependent on the initialization of the centroids." → 多次初始化 + `init='k-means++'`；`kmeans_plusplus` 也可单独调用来给别的算法选种子（L75）。
- **`sample_weight` 的语义**（L77）："assigning a weight of 2 to a sample is equivalent to adding a duplicate of that sample"。
- **MiniBatch 的两步与代价**（L95–L99）：小批量随机抽样 → 逐样本用流式均值更新被分配的质心（L97），因此"the rate of change for a centroid over time"下降；"converges faster than `KMeans`, but the quality of the results is reduced"（L99）。
- **并行**（L86）：OpenMP + 256 样本分块，顺带降低内存占用。文档**不给任何数字**（分块大小之外）。

### C. Affinity Propagation 与 Mean Shift（L109–L138）

- AP **不需要 $k$**（L115）："it chooses the number of clusters based on the data provided"；两个关键参数是 `preference`（控制 exemplar 数量）与 `damping`（抑制消息振荡）。
- AP 的代价被明确写出（L117）：时间 $O(N^{2}T)$、稠密相似度矩阵内存 $O(N^{2})$ → "most appropriate for small to medium sized datasets"。
- MeanShift 找的是**平滑密度里的 blob**（L126），质心候选沿密度梯度更新，后处理去掉近重复候选；**簇数自动确定**，`bandwidth` 决定搜索区域大小，**不给就用 `estimate_bandwidth` 估**（L128）。
- MeanShift 的可扩展性差，原因是**算法执行期间要反复做最近邻搜索**（L130）；给新样本打标签就是找最近质心（L132）——这一句是它被列为「归纳式」的全部依据。

### D. Spectral clustering（L140–L183）

- 机制一句话（L142）：先对样本间的亲和矩阵做**低维嵌入**，再对特征向量的分量做聚类（如 KMeans）。稀疏亲和矩阵 + `amg` 求解器时尤其高效（需要 `pyamg`）。
- **当前版本必须指定簇数**，且"works well for a small number of clusters, but is not advised for many clusters"（L144）。
- 两簇时它是 **normalized cuts 的凸松弛**（L146）：切开图使割边权重相对簇内权重很小；文档特别指出这在**图像分割**上最有意义（顶点＝像素，边权＝图像梯度的函数）。
- **`[!warning]` 相似度矩阵的分布问题**（L150–L160）：若相似度里有负值、或给的是距离矩阵而不是相似度，"the spectral problem will be singular and the problem not solvable"，官方建议做变换，例如有符号距离用热核 `similarity = np.exp(-beta * distance / distance.std())`（L158 附近代码块）。
- **三种 `assign_labels`**（L168）：`"kmeans"` 能匹配更细的结构但**不稳定**（不固定 `random_state` 可能 run-to-run 不可复现）；`"discretize"` **被文档称为 100% 可复现**但倾向于产生大小接近、几何规整的分块；`"cluster_qr"` 是确定性替代，且在官方示例上"tends to create the visually best partitioning"（图 21/23 的目视读数支持 cluster_qr 的观感优势）。`discretize` 的可复现性边界需在目标环境中重新验证。
- **图分割用法**（L176–L183）：把邻接矩阵当亲和矩阵，`affinity='precomputed'`；官方 doctest 用 `n_init=100` 与 `assign_labels='discretize'`（L180）。

### E. 层次聚类（L185–L280）

- 层次聚类是"build nested clusters by merging or splitting them successively"的**一族**方法，结果是树／树状图：根是全体、叶是单样本（L187）。
- `AgglomerativeClustering` 是**自底向上**（L189），四种 linkage（L191–L194）：Ward（最小化簇内平方差之和，"similar to the k-means objective function but tackled with an agglomerative hierarchical approach"）、complete（最小化簇对之间观测的**最大**距离）、average（**平均**距离）、single（**最近**观测距离）。
- **无连通性约束时计算昂贵**——每一步都要考虑所有可能合并；加了 `connectivity` 矩阵后可扩展到大规模样本（L196）。
- **「富者愈富」**（L204）：层次聚类会导致簇大小不均，single linkage 最严重、Ward 最规整；但**Ward 不能用非欧氏度量**，"thus for non Euclidean metrics, average linkage is a good alternative"；single linkage 对噪声不鲁棒，但**计算非常高效**、且"can also perform well on non-globular data"。
- 树状图可视化在小样本上更有用（L213）。
- **连通性约束 = 只允许相邻簇合并**（L223），用途是防止跨越流形折叠处合并（Swiss-roll 例子），文档称它还能在大样本上加速（L227）；矩阵可以是 scipy 稀疏矩阵，来自先验（网页链接）或数据（`kneighbors_graph`、`grid_to_graph`）（L229）。
- **`[!warning]` 连通性 + single/complete/average 会放大「富者愈富」**（L234）："In the limit of a small number of clusters, they tend to give a few macroscopically occupied clusters and almost empty ones"，single linkage 最脆。
- **度量选择**（L246–L251）：single/average/complete 可配 l2、l1（cityblock）、cosine、预计算矩阵；l1 适合稀疏特征（文本里稀有词的 occurrence），cosine 对整体缩放不变；**选度量的准则**是"one that maximizes the distance between samples in different classes, and minimizes that within each class"——注意这句话预设了**有类别标签**，用于无监督场景时只能当经验。
- **BisectingKMeans**（L261–L280）：KMeans 的**分裂式**变体，反复把一个簇切成两个直到达到目标数；簇数大时比 KMeans 高效（每次只处理数据的子集，L263）；**按设计无法享受 k-means++**（L265），但 inertia 与 `KMeans(init="k-means++")` 相当而更便宜（L267）；**不产生空簇**（L269）；两种切分策略 `largest_cluster`（点最多）与 `biggest_inertia`（SSE 最大，L271–L274）；文档称按点数切"in most cases produces result as accurate as picking by inertia and is faster"（L276）并"will also likely produce clusters of similar sizes"（L278）；结果"well ordered and create quite a visible hierarchy"（L280）。

### F. 密度聚类：DBSCAN / HDBSCAN / OPTICS（L282–L361）

- **DBSCAN 把簇看成被低密度区分开的高密度区域**，因此"clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped"（L284）。核心概念是**核心样本**；簇 = 核心样本集合（彼此足够近）+ 附着的非核心样本（L284–L286）。
- 两个参数定义"dense"：`min_samples` 与 `eps`；"Higher `min_samples` or lower `eps` indicate higher density necessary to form a cluster"（L286）。
- **非核心且距任何核心样本都 ≥ `eps` 的样本被标为离群**（L288）。
- **`eps` 是本节唯一被点名"不能留默认值"的参数**（L290）："crucial to choose appropriately for the data set and distance function and usually cannot be left at the default value"。太小 → 大部分数据不聚类（标 `-1`）；太大 → 邻近簇合并，最终全变成一个簇；文献里的启发式是**最近邻距离图的 knee**。`min_samples` 则主要控制对噪声的容忍度（噪声大、数据大就调大）。
- **HDBSCAN = DBSCAN + OPTICS 的推广**（L302）：DBSCAN 假设密度判据**全局同质**，所以难以同时抓住不同密度的簇；HDBSCAN"explores all possible density scales"，代价是不再需要 `eps`、只需 `min_samples`——文档称它"tends to be a more robust hyperparameter"（L341）。
- **互达距离的构造**（L313–L325）：核心距离 $d_c(x_p)$ = 到第 `min_samples` 近邻的距离（**含自身**）；互达距离 $d_m(x_p,x_q)=\max\{d_c(x_p),d_c(x_q),d(x_p,x_q)\}$；据此建图 $G_{ms}$，按 $\epsilon$ 剪边得 $G_{ms,\epsilon}$，其连通分量"equivalent to running DBSCAN\* with `min_samples` and $\epsilon$"（L328）。
- **层次化 = 对所有 $\epsilon$ 做 DBSCAN\***（L332–L341）：取互达图的最小生成树（MST）→ 给每个顶点加一条权重等于核心距离的自环 → 反复删最重边（平票同时删）→ 连通分量即为各 $\epsilon$ 下的簇；"This allows HDBSCAN to perform clustering across multiple densities"（L341）。
- **`min_cluster_size` 把层次树剪成可用结果**（L347）：小于该规模的连通分量归为噪声；官方建议"one can set `minimum_cluster_size = min_samples` to couple the parameters and simplify the hyperparameter space"（L347）。
- **OPTICS 把 `eps` 从单值放松成区间**（L353）：它构建 **reachability 图**，给每个样本 `reachability_` 距离与 `ordering_` 位置；`max_eps=inf`（默认）时可用 `cluster_optics_dbscan` 以**线性时间**对任意 `eps` 反复做 DBSCAN 式抽取；调小 `max_eps` 可缩短运行时间。
- **reachability plot 的读法**（L357）：Y 轴是密度、X 轴按 `ordering_` 排列使近邻相邻；在某一高度"切一刀"= DBSCAN 结果（切线上方为噪声，从左到右每断一次=一个新簇）；默认抽取看**陡坡**，陡峭程度由 `xi` 定义；层次结构通过 `cluster_hierarchy_` 访问（L357 说"parameter"，实为属性）。

### G. BIRCH（L363–L377）

- **CF 树 = 数据的有损压缩**（L365）：整份数据被压成一组 Clustering Feature 节点／子簇，子簇只需保存"样本数、线性和、平方和、质心、质心的平方范数"，因此**不必把全部输入留在内存**（L367–L373）。
- 两个参数：`branching_factor`（每个节点最多多少子簇）与 `threshold`（新样本与现有子簇的距离上限）（L375）。
- **BIRCH 是数据缩减（data reduction）**（L377）：叶子上的子簇可直接使用，也可喂给**全局聚类器**（由 `n_clusters` 指定）；`n_clusters=None` 时直接读叶子子簇，否则先把子簇聚成全局簇、再把样本映射到最近子簇的全局标签（L377）。

### H. 评估（L379–L804）

- **为什么不能数准确率**（L381、L385）：标签的**绝对值**无意义，只应看「划分是否与某个真值一致」或「是否满足同类更相似」；所有指标都"ignoring permutations"。
- **Rand index 与 ARI**（L383–L431）：RI 度量两个分配的**相似性**，但"does not ensure to obtain a value close to 0.0 for a random labelling"；ARI **做了机会校正**（L395）。两者都**对称**，因此可当**共识度量**（L412）。doctest 数字：RI 0.66、ARI 0.24；完美 1.0；独立划分 RI 0.39、ARI **-0.072**（L431 附近）。
- **MI / NMI / AMI**（L446–L503）：三者都是**对称**的共识度量；AMI "was proposed more recently and is **normalized against chance**"（L448）；"This is not true for `mutual_info_score`, which is therefore harder to judge"（L485）。
- **Homogeneity / Completeness / V-measure**（L505–L589）：两个目标来自 Rosenberg & Hirschberg (2007)——homogeneity（每个簇只含一个类）、completeness（一个类的成员都在同一簇）（L511–L512）；V-measure 是两者的**调和平均**，`beta<1` 更重 homogeneity、`beta>1` 更重 completeness（L542–L552）；**V-measure 等价于 NMI（算术平均聚合）**（L556 附近）；`[!note]` 指出 V-measure **对称**，而 homogeneity 与 completeness **不对称**，满足 `homogeneity_score(a, b) == completeness_score(b, a)`（L574–L589）。
- **Fowlkes-Mallows**（L591–L642）：FMI 是**成对 precision 与 recall 的几何平均**（L593）：$\text{FMI}=\text{TP}/\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})}$，其中 TP/FP/FN 都是**样本对**的计数（L601–L603）。
- **内部指标（无真值）**（L644–L725）：silhouette 由每个样本的 $a$（到同簇其他点的平均距离）与 $b$（到**次近簇**的平均距离）构成，$s=(b-a)/\max(a,b)$；整个集合取均值（L649–L656）。Calinski-Harabasz（又名 Variance Ratio Criterion）是**簇间离散度／簇内离散度**之比（L685）。Davies-Bouldin 是"average 'similarity' between clusters"，**越小越好、0 最优**（L709–L711）。
- **两个矩阵**（L727–L804）：contingency matrix 给出每个「真值簇 × 预测簇」的交叉计数，"provides sufficient statistics for all clustering metrics where the samples are i.i.d. and one doesn't need to account for some instances not being clustered"（L729）；pair confusion matrix 是 2×2 的**成对**相似度矩阵，$C_{11}$=TP、$C_{01}$=FP、$C_{10}$=FN、$C_{00}$=TN（L758–L767），**且不对称**（L791）。

## 版本判定

**结论：1.9.x，工作版本 1.9.1，证据强。** 判定链如下（全部为本库 2026-11-12 的文本比对）。

1. **句子探针**：把剪藏散文切成 126 条 ≥14 词的句子（取前 12 词），在官方 1.6/1.7/1.8/1.9 四版页正文中查找。命中数：1.6 = **107**、1.7 = **108**、1.8 = **110**、1.9 = **111**。15 条未命中项逐条核为公式渲染差异（`$…$`、`\left(`、`\*`），**无一条是内容缺失**。
2. **1.6 → 1.7 的措辞判据（剪藏全部含有 → ≥1.7）**："The first row of the output array"（1.6 作 "of output array"）、"Furthermore, both `rand_score` and `adjusted_rand_score` are"（1.6 缺"Furthermore"）、"Davies-**Bouldin**"（1.6 拼作 Boulding）、"density-based clusters"（1.6 作 "density based"）、"of an **inductive** clustering model for handling new data"（1.6 只有 "handling new data"）。
3. **1.7 → 1.8 的措辞判据（剪藏全部含有 → ≥1.8）**："**Swiss**-roll"（1.7 作小写 "swiss-roll"，L225、L237）、"These constraints are **not only** useful to impose…"（1.7 无 "not only"，L227）、"via **a** connectivity matrix"（1.7 作 "an connectivity"，L229）、"instance **of a** data reduction method"（1.7 作 "instance or data reduction"，L377）、"an AMI score"（1.7 作 "a AMI"）、示例标题 "Hierarchical clustering with and without structure"（1.7 作 "Agglomerative clustering with and without structure"，L236）。
4. **1.8 → 1.9 的措辞判据（剪藏含有 → ≥1.9）**："This allows **assigning** more weight to some samples"（1.8 作 "Allows to assign"，L77）；另一处 1.9 修正是 "Egineering"→"Engineering"（在 K-means 参考文献块里，剪藏中该块为空，无法用于判定）。
5. **1.9 → 1.9.1 无法区分**：1.8→1.9 之后本节正文再无改动；抓取日 `/stable/modules/clustering.html` 与 `/1.9/modules/clustering.html` **字节级相同**（245,702 B，sha256 `3c8d5481…`），页脚写着 "scikit-learn 1.9.1"，故取 1.9.1。

本节的功能性结论（除默认值外）**不依赖** 1.9.0 与 1.9.1 的区分。

## 转换损失

与 [[sklearn-manifold-learning]]（丢数学、留散文与 14 条文献）相比，本次是**结构损失最大**的一份：散文在，但**所有数学、所有实现细节、所有文献、以及总览表的第一列**都不在。

| 丢失项 | 词数 | 内容 | 后果 |
| --- | --- | --- | --- |
| 总览表第一列（12 行） | — | 官方表首列是方法名：K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、HDBSCAN、OPTICS、Gaussian mixtures、BIRCH、Bisecting K-means | **表在剪藏里不可读**：每行只剩参数/可扩展性/用途/几何，方法名要靠上下文倒推（见「分歧」6） |
| `Affinity Propagation` → Algorithm description | 250 | responsibility $r(i,k)$ 与 availability $a(i,k)$ 的迭代式、`damping` 如何混合新旧值、`convergence_iter` 的判据 | 剪藏只能给"消息传递"这个比喻，**无法解释 `preference` 与簇数的定量关系** |
| `Mean Shift` → Mathematical details | 190 | 核密度估计、hill climbing、`estimate_bandwidth` 的推导 | "密度众数漂移"只剩一句话 |
| `DBSCAN` → Implementation | 202 | 算法**确定性**但有例外：边界点归属依赖**数据顺序**；`sample_weight` 的处理；不同 `algorithm` 后端的差异 | 剪藏只字未提"顺序依赖"，需要目标环境验证其具体影响 |
| `DBSCAN` → Memory consumption for large sample sizes | 165 | 稀疏矩阵 + `metric='precomputed'` 的内存行为 | 大数据内存规划无依据 |
| `OPTICS` → Comparison with DBSCAN | 146 | `cluster_optics_dbscan` 与 DBSCAN 结果"very similar, but not always identical"，差异在边界点 | 不能把两者写成无条件等价 |
| `OPTICS` → Computational Complexity | 116 | 空间索引树避免全距离矩阵 | 复杂度无从引用 |
| `BIRCH` → Algorithm description | 200 | 新样本如何插入 CF 树、如何分裂节点、`partial_fit` 的插入路径 | 剪藏只描述"CF 子簇保存什么"，不描述**怎么用** |
| `BIRCH` → BIRCH or MiniBatchKMeans? | 67 | "if `n_features` is greater than twenty, it is generally better to use MiniBatchKMeans"（外部核查原文） | **本节唯一的 BIRCH 选型判据丢失** |
| `BIRCH` → How to use partial_fit? | 57 | 先 `n_clusters=None` 训练、最后再设全局聚类器 | 增量用法丢失 |
| `Rand index` → Mathematical formulation | 181 | 列联表下的 $a$、$b$ 定义与 ARI 的机会校正公式 | 只能引 doctest 数字，无法推导 |
| `Mutual Information` → Mathematical formulation | 499 | 熵、条件熵、MI 的定义与 NMI/AMI 的三种归一化、`average_method` | **本节最痛的丢失**：`average_method` 默认值变更（0.22）导致剪藏里的 doctest 数字**全部过期**（见「分歧」1） |
| `Homogeneity/completeness/V` → Mathematical formulation | 149 | $h=1-H(C\mid K)/H(C)$、$c=1-H(K\mid C)/H(K)$ | 只能引调和平均公式（L538） |
| `Calinski-Harabasz` → Mathematical formulation | 125 | $s=\dfrac{\mathrm{tr}(B_k)}{\mathrm{tr}(W_k)}\times\dfrac{n_E-k}{k-1}$ 的完整定义 | 剪藏只给"簇间/簇内离散度之比"的文字 |
| `Davies-Bouldin` → Mathematical formulation | 112 | $R_{ij}=(s_i+s_j)/d_{ij}$ 与 $D_i=\max_j R_{ij}$ | 同上 |
| References（16 块，剪藏只剩 2 处空标题） | ≈600 | K-means++（Arthur & Vassilvitskii）、Web-Scale K-Means（Sculley）、Mean shift（Comaniciu & Meer）、Multiclass spectral clustering（Yu & Shi）、A Tutorial on Spectral Clustering（von Luxburg）、Normalized cuts（Shi & Malik）、Random Walks（Meila）、Document clustering comparison（Steinbach et al.）、DBSCAN（Ester et al.）、OPTICS（Ankerst et al.）、BIRCH（Zhang et al.）、Comparing Partitions（Hubert & Arabie）、ARI 性质（Steinley）、Fowlkes & Mallows 1983、Silhouettes（Rousseeuw 1987）、Calinski-Harabasz 1974、Davies-Bouldin 1979、V-Measure（Rosenberg & Hirschberg 2007） | **剪藏在 L349、L587 之外没有任何 References 标题，这两处下面也是空的**；所有文献只能从官方页补 |
| 图片 3 张 | — | `sphx_glr_plot_adjusted_for_chance_measures_001.png`（机会校正示例图）、`sphx_glr_plot_birch_vs_minibatchkmeans_001.png`、页脚 `probabl.png` | 官方 26 张、剪藏 23 张；前两张的读数因此无法获得 |

**未丢失的部分**：全部 92 行 doctest、**7 个行间公式**（inertia、HDBSCAN 的核心距离与互达距离、V-measure、FMI、silhouette、pair confusion 矩阵）、5 个 admonition 都在。因此本节能对「散文主张」与「doctest 数字」逐条复核，但对**机制**只能引官方页。

## 图表核查

23 张于 2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-clustering/`。**本次共对 8 张做了目视核查**：下表 6 张（标「✓」）+ 表后单独记录的 2 张硬币分割图。其余 **15 张仅入库、未查看**，故本页不对其内容作任何描述。

| # | 文件 | 尺寸 | SHA-256 前 16 | 目视读数 |
| --- | --- | --- | --- | --- |
| 1 | `plot_cluster_comparison_001` ✓ | 2100×1300 | `f9722ff1d6f26abf` | 11 列 × 6 行面板。列序：MiniBatch KMeans、Affinity Propagation、MeanShift、Spectral、Ward、Agglomerative、DBSCAN、HDBSCAN、OPTICS、BIRCH、Gaussian Mixture。**第 1 行（同心圆）：只有 Spectral/Ward/Agglomerative/DBSCAN/HDBSCAN/OPTICS/BIRCH 把内圈与外圈分开**，MiniBatchKMeans、AP、MeanShift、GMM 都把两圈各切一半。**第 3 行起 DBSCAN/HDBSCAN/OPTICS 出现黑色噪声点**，其余方法没有。**最后一行（无结构数据）DBSCAN/HDBSCAN/OPTICS 全蓝（整体成一块或全噪声）**。面板右下角时间为：AP 0.10–0.14s、OPTICS 0.45–0.51s（最慢）、DBSCAN/HDBSCAN 0.00–0.01s、GMM 0.00–0.01s |
| 2 | `plot_kmeans_assumptions_002` ✓ | 1200×1200 | `1f1e609bb3ffca93` | 标题 "Unexpected KMeans clusters"，2×2 面板："Non-optimal Number of Clusters"（三个真实团被切成 2 黄 + 大紫）、"Anisotropically Distributed Blobs"（斜向团被按方向而非按团切开）、"Unequal Variance"（大方差黄团吞并邻近小团）、"Unevenly Sized Blobs"（小团被并进近邻）。这是 L59–L60 两条缺陷的图示 |
| 3 | `plot_dbscan_002` ✓ | 640×480 | `6fbc56c13ca9ce3e` | 标题 "Estimated number of clusters: 3"。三个团（深红、浅绿、橙）；**大圆点＝core sample、小圆点＝border/non-core、黑色小点＝噪声**；噪声点散布在团外与团间。与 L284–L288 的文字一致 |
| 4 | `plot_linkage_comparison_001` ✓ | 1370×1450 | `af2e081f5d1f7586` | 4 列（Single／Average／Complete／Ward）× 6 个 toy 数据集。**Single 在「varied」那行几乎全并成一个大蓝簇**，是「富者愈富」最直观的一格；Ward 在同心圆与双月上**切错**（把两圈各分一半），在 blobs 上最规整。时间标签：Single 0.01s，其余 0.03–0.06s |
| 5 | `plot_optics_001` ✓ | 1000×700 | `29d7b49b78395dbd` | 上半 "Reachability Plot"（6 段彩色曲线，两条虚线为 0.5 与 2.0 的 $\epsilon$ 切）；下半三面板："Automatic Clustering OPTICS"、"Clustering at 0.5 epsilon cut DBSCAN"（大量灰点＝噪声、黄团整块变成噪声）、"Clustering at 2.0 epsilon cut DBSCAN"（右侧红/蓝/紫**合并成一个品红簇**）。这正是 L357 所说「切一刀＝DBSCAN、断口＝新簇」的可视化 |
| 6 | `plot_hdbscan_007` ✓ | 1000×400 | `5e310ccccda975ab` | 标题 "Estimated number of clusters: 4"，黑色 × 为噪声。左侧有两个贴得很近的簇（绿、米白），中部与下方各一个大簇（深红、橙）；噪声主要落在簇间与边缘。对应 L347 的 `min_cluster_size` 剪枝效果 |
| 7–21 | `plot_affinity_propagation_001`、`plot_agglomerative_clustering_metrics_00{5,6,7}`、`plot_agglomerative_dendrogram_001`、`plot_coin_segmentation_002`、`plot_hdbscan_005`、`plot_kmeans_digits_001`、`plot_mean_shift_001`、`plot_mini_batch_kmeans_001`、`plot_segmentation_toy_00{1,2}`、`plot_ward_structured_vs_unstructured_00{1,2,3}` | 640×480 – 1000×700 | 见目录 | **未查看**，本页不作描述 |

补充读数（同批下载、已在核查中打开）：`plot_coin_segmentation_001`（"Spectral clustering: **kmeans, 0.19s**"）与 `plot_coin_segmentation_003`（"Spectral clustering: **cluster_qr, 0.07s**"）。两者对照下，`kmeans` 面板里有一条**细长的黄色连通块横穿硬币之间**（把背景/缝隙连成一片），`cluster_qr` 面板里每枚硬币的边界**干净闭合**、没有跨硬币区域。这支持 L168 的"visually best partitioning"与"确定性"，并给出 **2.7×** 的时间差（0.19s vs 0.07s）。

## 外部核查（官方 1.9.1 API 页与 `whats_new`，不在剪藏内）

本节**几乎不给默认值**：正文只出现 `n_clusters`、`init='k-means++'`、`eps`、`min_samples`、`bandwidth`、`n_init`（示例里）、`assign_labels`、`affinity='precomputed'`、`bisecting_strategy`、`n_clusters=None`、`max_eps`、`xi`、`average_method`（未点名）。以下取自官方 1.9.1 生成页签名：

| 估计器 | 1.9.1 签名（默认值） | 剪藏是否提及 |
| --- | --- | --- |
| `KMeans` | `n_clusters=8`、`init='k-means++'`、`n_init='auto'`、`max_iter=300`、`tol=1e-4`、`algorithm='lloyd'`、`copy_x=True`、`random_state=None` | 只提 `init='k-means++'`（L73）与 `sample_weight`（L77），**无一个默认值** |
| `MiniBatchKMeans` | `n_clusters=8`、`init='k-means++'`、`max_iter=100`、`batch_size=1024`、`tol=0.0`、`max_no_improvement=10`、`init_size=None`、`n_init='auto'`、`reassignment_ratio=0.01` | 一个都没提 |
| `AffinityPropagation` | `damping=0.5`、`max_iter=200`、`convergence_iter=15`、`copy=True`、`preference=None`、`affinity='euclidean'`（负平方欧氏距离）、`random_state=None` | 只提 `preference`/`damping` 的**名字**（L115） |
| `MeanShift` | `bandwidth=None`、`seeds=None`、`bin_seeding=False`、`min_bin_freq=1`、`cluster_all=True`、`n_jobs=None`、`max_iter=300` | 只说"不给就用 `estimate_bandwidth`"（L128）；`estimate_bandwidth(quantile=0.3, n_samples=None, random_state=0)` |
| `SpectralClustering` | `n_clusters=8`、`eigen_solver=None`、`n_components=None`、`random_state=None`、`n_init=10`、`gamma=1.0`、`affinity='rbf'`、`n_neighbors=10`、`eigen_tol='auto'`、`assign_labels='kmeans'`、`degree=3`、`coef0=1` | **`affinity` 的默认是 `'rbf'` 而正文示例全用 `'precomputed'`/`nearest_neighbors`**；`assign_labels` 三种取值在 L168 |
| `AgglomerativeClustering` | `n_clusters=2`、`metric='euclidean'`、`connectivity=None`、`compute_full_tree='auto'`、`linkage='ward'`、`distance_threshold=None`、`compute_distances=False` | 只提 `linkage`/`connectivity`/`metric` 的语义（L189–L246） |
| `BisectingKMeans` | `n_clusters=8`、`init='random'`、`n_init=1`、`max_iter=300`、`tol=1e-4`、`algorithm='lloyd'`、`bisecting_strategy='biggest_inertia'` | 提 `bisecting_strategy` 的两个取值（L273–L274），**不提默认是 `biggest_inertia`**；也不提 `init` 可设 `'k-means++'`（L265 反着说） |
| `DBSCAN` | `eps=0.5`、`min_samples=5`、`metric='euclidean'`、`algorithm='auto'`、`leaf_size=30`、`p=None`、`n_jobs=None` | 只说 `eps` "usually cannot be left at the default value"（L290），**不说默认是 0.5** |
| `HDBSCAN` | `min_cluster_size=5`、`min_samples=None`（**= `min_cluster_size`**）、`cluster_selection_epsilon=0.0`、`max_cluster_size=None`、`metric='euclidean'`、`alpha=1.0`、`algorithm='auto'`、`leaf_size=40`、`cluster_selection_method='eom'`、`allow_single_cluster=False`、`store_centers=None`、`copy='warn'` | 只提 `min_samples` 与 `min_cluster_size`（L341–L347），**不提 `min_samples=None` 的耦合** |
| `OPTICS` | `min_samples=5`、`max_eps=inf`、`metric='minkowski'`、`p=2`、`cluster_method='xi'`、`eps=None`、`xi=0.05`、`predecessor_correction=True`、`min_cluster_size=None`、`algorithm='auto'`、`leaf_size=30` | 提 `max_eps=inf`（L353）与 `xi`（L357），不提 `cluster_method` 的默认 `'xi'` |
| `Birch` | `threshold=0.5`、`branching_factor=50`、`n_clusters=3`、`compute_labels=True` | 提 `threshold`/`branching_factor` 的语义（L375）与 `n_clusters=None`（L377），**不提默认 `n_clusters=3`** |
| `silhouette_score` | `(X, labels, *, metric='euclidean', sample_size=None, random_state=None)` | 只在 doctest 里用 `metric='euclidean'`（L674） |
| `calinski_harabasz_score` / `davies_bouldin_score` | `(X, labels)` | 只给 doctest |
| `adjusted_mutual_info_score` / `normalized_mutual_info_score` | `(labels_true, labels_pred, *, average_method='arithmetic')` | **完全没提 `average_method`** |
| `v_measure_score` | `(labels_true, labels_pred, *, beta=1.0)` | 提 `beta` 的语义与默认 1.0（L542） |
| `contingency_matrix` | `(labels_true, labels_pred, *, eps=None, sparse=False, dtype=int64)` | 只给示例（L735–L742） |
| `pair_confusion_matrix` | `(labels_true, labels_pred)` | 只给示例（L768–L804） |

版本史（`whats_new` + API 页 "Changed in version" 注记，均属外部核查）：

- **`KMeans.n_init` / `MiniBatchKMeans.n_init`**：`'auto'` 选项 1.2 新增，**1.4 起成为默认**。`'auto'` 的运行次数**两个类取反**——`KMeans`：`init='random'` → 10、`k-means++` → 1；`MiniBatchKMeans`：`init='random'` → **3**、`k-means++` → 1。
- **`MiniBatchKMeans.batch_size`** 在 1.0 从 100 改为 **1024**。
- **`KMeans.algorithm`**：1.1 把 `"full"` 改名 `"lloyd"`，并让 `"auto"` 用 `lloyd` 而非 `elkan`。
- **`HDBSCAN` 1.3 新增**；`copy` 默认值将在 **1.10** 从 `False`（1.9.1 显示为 `'warn'`）变为 `True`——即当前版本已有弃用警告。
- **`BisectingKMeans` 1.1 新增**，1.3 增补参数。
- **`SpectralClustering.assign_labels='cluster_qr'` 1.1 新增**；`n_init` 至今默认 10（只在使用 `assign_labels='kmeans'` 时生效）。
- **`adjusted_mutual_info_score` / `normalized_mutual_info_score` 的 `average_method`**：0.22 把默认从 `'max'`（AMI）与 `'geometric'`（NMI）统一改为 **`'arithmetic'`**——这是剪藏 doctest 数字过期的原因（「分歧」1）。
- **`AgglomerativeClustering.metric`** 1.2 起取代 `affinity`；`linkage='single'` 0.20 新增；`distance_threshold` 0.21 新增；`compute_distances` 0.24 新增。
- **`Birch`** 0.16 新增，**1.0 起 `n_features_out` 语义固定**；1.4 增补 `set_output`。
- **`OPTICS`** 0.24 新增；1.0 增补参数。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 分歧

1. **§2.3.11.2 的 AMI doctest 是旧默认口径。** 文档给 `adjusted_mutual_info_score([0,0,0,1,1,1],[0,0,1,1,2,2]) = 0.22504`（L456）与独立划分 `-0.10526`（L498）；外部核查显示 0.22 已把默认 `average_method` 从 `'max'` 改为 `'arithmetic'`。这些数字不应直接用于当前版本。
2. **§2.3.11.3 的 doctest 在页内自相矛盾。** 同一组标签，`homogeneity_score` 显示 **0.66**（L528）、`completeness_score` 显示 **0.42**（L530），而 `homogeneity_completeness_v_measure` 的三元组显示 **(0.67, 0.42, 0.52)**（L563）；`v_measure_score` 的 0.516（L532）也与三元组里的 0.52 采用了不同舍入。
3. **`assign_labels="discretize"` 的可复现性边界待验证。** L168 称其为 "100% reproducible"，但历史运行时反例已按所有者决定移除；在 `uv` 目标版本环境中复核前，不把该表述扩展为无条件保证。
4. **"`BisectingKMeans` can't benefit from k-means++ by design"（L265）与 API 不符。** 1.9.1 签名有 `init={'k-means++','random'}`（默认 `'random'`），`n_init=1`。文档既没提这个参数，也没提默认 `bisecting_strategy='biggest_inertia'`；L278 的"largest_cluster 会产生更均匀的簇"仍需独立验证。
5. **KMeans 停止条件的描述不完整。** L71 说"iteration stops when centroids move less than the tolerance"，API 页说"Relative tolerance with regards to Frobenius norm of the difference in the cluster centers"，源码比较的是**各质心位移的平方和**与 `tol × mean(var(X))`。三者不是同一句话：**平方** vs 范数、**绝对** vs 相对。
6. **总览表的第一列在剪藏里整列丢失。** 官方表的每一行以方法名开头（K-Means、Affinity propagation…），剪藏 12 行的第一格**全空**（L18–L29）。后果：这张"选型主表"在本地无法独立阅读，必须配合图 1 或后文小节倒推；引用该表时应改成引用小节（L45、L109、L124…）而非引用表行。
7. **BIRCH 的默认 `n_clusters=3` 从未出现在正文里。** L377 只说"`n_clusters` 设为 None 时直接读叶子子簇"，读起来像"默认不做全局聚类"，但 1.9.1 API 签名明确默认 `n_clusters=3`，即默认会做全局聚类。
8. **MeanShift 那句话的因果是反的。** L128："The algorithm automatically sets the number of clusters, instead of relying on a parameter `bandwidth`, which dictates the size of the region to search through." 实际机制依赖 `bandwidth` 决定核大小，只是可以在 `bandwidth=None` 时用 `estimate_bandwidth` 估一个（默认 `quantile=0.3`）：**"自动"只是"自动估带宽"**。

## 局限

- **本节没有任何一条"该用哪个算法"的判据**。总览表用"usecase"列给的是描述（"General-purpose, even cluster size, flat geometry…"），不是决策规则；唯一的定量建议是 DBSCAN 的 knee 启发式（L290）。
- **评估部分与选型脱节**：9 个指标都在 §2.3.11，而前面 10 个小节一次都没提"怎么知道这次聚类好不好"。文档也没说内部指标（silhouette/CH/DB）**随 $k$ 单调**的倾向，因此"用 silhouette 选 $k$"在文档里只以示例形式出现（`plot_kmeans_silhouette_analysis`，L303 附近引用）。
- **所有数学都不在剪藏里**（5 个 `Mathematical formulation`、AP 的消息公式、MeanShift 的 hill climbing、DBSCAN 的实现细节、OPTICS 的复杂度）。因此本页对"为什么"的解释只能停在散文层面。
- **不给复杂度**：只有 AP 的 $O(N^{2}T)$ / $O(N^{2})$ 是正文里的（L117），其余方法的成本只能从示例图标题的时间读数（图 1 面板）间接估。
- **官方 16 个 References 折叠块的内容全丢**（剪藏只剩 2 处空标题），所以"k-means++ 更好"（L73）、"Normalized cuts"（L146）、"Rosenberg & Hirschberg 2007"（L509）都只能引正文的转述，不能引原文。
- **没有 `sklearn.mixture`**：GMM 只以一行表项 + 一句"KMeans 是特例"出现（L37），完整内容在 2.1 节（未入库）。
- **没有 `sklearn.metrics.pairwise`**：`affinity='precomputed'` 的矩阵该用什么度量构造，本节不讨论。
- **示例脚本一个都没运行**（`plot_cluster_comparison`、`plot_kmeans_assumptions`、`plot_dbscan`、`plot_hdbscan`、`plot_optics`、`plot_linkage_comparison` 等 20 余个）。图 1 面板的时间只能当"同机同版本的相对量级"。
- **运行时行为尚未在 `uv` 目标版本环境中复核**。`HDBSCAN.copy` 的 1.10 变更、`KMeans.algorithm` 的 `elkan` 语义与 1.9 的 `set_output` 行为都只保留文档或版本史记录。

## 待解问题

1. **`eps` 的 knee 启发式**（L290）在剪藏里只有一句"文献里讨论过"。具体怎么做（k-距离排序曲线的最大曲率？分位数？）本节不给，本库也没有来源覆盖——这是本页最想补的一条。
2. **`silhouette_score` / `calinski_harabasz_score` / `davies_bouldin_score` 随 $k$ 的行为**：文档不说单调性，本库也未系统测。若 CH 随 $k$ 单调上升，那它选 $k$ 就是错的。
3. **`HDBSCAN.copy='warn'` 在 1.9.1 上的实际警告文本与 1.10 的破坏面**待目标环境复核。
4. **`discretize` 的随机性来源与触发条件**需在目标版本源码和运行环境中共同核对。
5. **`OPTICS` 与 `DBSCAN` 的"边界点差异"**需要专门的边界点数据才能验证，不能用一般示例替代。
6. **BisectingKMeans 在 $k$ 大时的优势**（L263 说"只处理子集"）需要更大 $k$ 与 $n$ 的目标环境对照。
7. **`SpectralClustering(affinity='rbf')` 是默认**，而正文的警告说相似度分布不好会让问题奇异（L153）。默认 `gamma=1.0` 在未缩放数据上的触发条件待验证。
8. **`AgglomerativeClustering` 的 `distance_threshold`**（外部核查：0.21 新增）在剪藏里**完全没出现**——"用距离阈值代替簇数"这条路线在本节缺席，需要时只能查 API。
9. **BIRCH 的 `partial_fit`** 与"数据缩减后再全局聚类"的组合（丢失折叠块里）在 1.9.1 上的行为未核。
10. **聚类结果的不稳定性**（种子、样本顺序、`eps` 微小变化）在本节没有量化讨论；后续应在固定版本的 `uv` 环境中统一验证。

## 行号索引

| 内容 | 行号 |
| --- | --- |
| 引子（模块、类/函数双形态、`labels_`） | L10–L12 |
| 2.3.1 总览（图、12 行表、非平坦几何、GMM 特例、归纳/传导） | L14–L43 |
| 2.3.2 K-means（inertia 定义与公式、两缺陷、Lloyd、EM 等价、Voronoi、停止条件、局部极小、k-means++、`sample_weight`） | L45–L82 |
| 2.3.2.1 并行（OpenMP、256 样本分块） | L84–L91 |
| 2.3.2.2 Mini Batch（小批量、流式均值、质量略降） | L93–L107 |
| 2.3.3 Affinity Propagation（exemplar、`preference`/`damping`、$O(N^2T)$） | L109–L122 |
| 2.3.4 Mean Shift（blob、`bandwidth`/`estimate_bandwidth`、不可扩展、新样本） | L124–L138 |
| 2.3.5 Spectral（嵌入 + 聚类、`amg`、normalized cut、相似度警告、热核） | L140–L164 |
| 2.3.5.1 三种 `assign_labels` | L166–L172 |
| 2.3.5.2 图分割（`affinity='precomputed'`） | L174–L183 |
| 2.3.6 层次聚类（树/树状图、4 种 linkage、连通性成本） | L185–L196 |
| 2.3.6.1 linkage 对照（富者愈富、Ward 只能用欧氏） | L198–L209 |
| 2.3.6.2 树状图可视化 | L211–L219 |
| 2.3.6.3 连通性约束（矩阵构造、更快、`[!warning]` 富者愈富） | L221–L242 |
| 2.3.6.4 度量选择（l2/l1/cosine/precomputed、选度量准则） | L244–L257 |
| 2.3.6.5 Bisecting K-Means（分裂式、成本、两策略、均匀性主张） | L259–L280 |
| 2.3.7 DBSCAN（密度视角、core/border/noise、`eps` 关键） | L282–L298 |
| 2.3.8 HDBSCAN（全局同质假设、核心距离、互达距离、MST、`min_cluster_size`） | L300–L349 |
| 2.3.9 OPTICS（reachability/ordering、`max_eps`、`xi`、`cluster_hierarchy_`） | L351–L361 |
| 2.3.10 BIRCH（CF 树、5 项统计量、`threshold`/`branching_factor`、数据缩减） | L363–L377 |
| 2.3.11 评估（为何不能数准确率） | L379–L381 |
| 2.3.11.1 Rand / ARI | L383–L443 |
| 2.3.11.2 MI / NMI / AMI | L446–L503 |
| 2.3.11.3 Homogeneity / Completeness / V-measure | L505–L589 |
| 2.3.11.4 Fowlkes-Mallows | L591–L642 |
| 2.3.11.5 Silhouette | L644–L679 |
| 2.3.11.6 Calinski-Harabasz | L681–L703 |
| 2.3.11.7 Davies-Bouldin | L705–L725 |
| 2.3.11.8 Contingency matrix | L727–L744 |
| 2.3.11.9 Pair confusion matrix | L746–L804 |

## 关联页面

- 概念：[[clustering]]（本页的机制整理：四条几何假设、$k$ 的三个来源、归纳/传导断层、指标的可判定性质）、[[manifold-learning]]（换坐标 vs 分组）、[[nearest-neighbors]]（kNN 图与 `eps` 的邻居搜索后端）、[[feature-selection]]（删列 vs 分组）、[[gaussian-discriminant-analysis]]（KMeans = 等协方差 GMM 的生成式读法）
- 实践：[[clustering-selection-and-tuning]]（选型、`k`/`eps` 怎么定、评估与 18 条坑）
- 工具：[[scikit-learn]]（`cluster` 模块的估计器矩阵与第 116–135 条默认值）
- 其他来源：[[sklearn-manifold-learning]]（同为「无监督但输出不同」的类目，也是本库另一份「数学全丢」的剪藏）、[[sklearn-ensembles]]（另一个「同一族多实现、代价不同」的对照）
