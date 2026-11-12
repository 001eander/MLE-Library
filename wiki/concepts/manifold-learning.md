---
title: 流形学习（Manifold Learning）：被保量决定方法、kNN 图是共同底座
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - dimensionality-reduction
  - manifold-learning
  - unsupervised
sources:
  - "[[sklearn-manifold-learning]]"
status: draft
---

# 流形学习

## 一句话定义与三个推论

**定义（来源论述）**：「Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this task are based on the idea that the dimensionality of many data sets is only artificially high.」[[sklearn-manifold-learning]], L28

这句话里有三个可以直接用于工程判断的推论（本库综合）：

1. **没有 $y$。** 全节唯一的优化对象是几何量（距离、权重、概率、内积），因此它是一个**不用标签的特征变换器**——评估标准不是准确率，而是「保住了什么」（下一节）。
2. **输出是一套坐标，不是预测。** `Isomap` 与 `LocallyLinearEmbedding` 有 `transform`；`SpectralEmbedding`、`TSNE`、`MDS`、`ClassicalMDS` **只有 `fit_transform`**（1.6.1 实测 `hasattr(..., 'transform') == False`）。这意味着**新样本无法定位**，想给测试集降维只能 `fit_transform` 重跑，而且两次结果不在同一坐标系里。
3. **"artificially high" 没有判定准则。** 来源不给「怎么知道维数是被噪声撑起来的」的方法（[[sklearn-manifold-learning]] 待解问题 1）。工程上只能绕道：看本征维数估计、看重构误差是否出现平台、看下游任务是否在低维上不掉点。

## 判别方法的第一性判据：保什么

这是本页的核心表。**方法名几乎不携带信息，被保量才是判据**（来源逐条给了目标，但从不并列比较）：

| 方法 | 被保的量 | 优化对象 | 损失/目标 | 依据 |
| --- | --- | --- | --- | --- |
| PCA / classical MDS | **成对欧氏内积** | 全局线性投影 | strain（$B$ 与 $ZZ^\top$ 的 Frobenius 相对误差） | [[sklearn-manifold-learning]], L128、L138；PCA 机制见 [[matrix-factorization]] |
| Isomap | **测地距离**（kNN 图上的最短路） | 全局 | MDS 的 stress（距离用图距离替换） | L56 |
| LLE / MLLE / HLLE / LTSA | **局部重构权重**（或局部切空间） | 局部 → 全局对齐 | 权重残差，取 $M=(I-W)^\top(I-W)$ 的零空间 | L66、L80、L90、L108 |
| Spectral Embedding | **图上的局部邻近**（单向） | 图 → 谱 | 图 Laplacian 的广义特征分解 | L100 |
| metric MDS | **不相似度本身** | 全局 | raw stress $\sum(\hat d_{ij}-d_{ij}(Z))^2$ | L122、L126 |
| non-metric MDS | **不相似度的序** | 全局 | Stress-1（保序回归的残差） | L122（机制在官方折叠块，见来源页转换损失） |
| t-SNE | **邻域概率**（高斯 ↔ Student-t） | 局部 | KL divergence | L149、L157 |

由此得到四条不对称结论：

- **全局 vs 局部是两种不同的失败模式**，不能互相折算。t-SNE 的局部保真度最高（实测 trustworthiness 0.9957），全局保真度却低于 PCA（Spearman 0.6201 vs 0.6456）；Isomap 反过来（0.9143 / 0.5518）。数字见 [[sklearn-manifold-learning]] 的「本机实测」补充段。
- **「保距离」与「保序」在 metric/non-metric MDS 之间不是同一个目标**：前者关心数值，后者只关心 $d_{ij}<d_{kl}$ 的比较（L122）。同一份数据上两者的 `stress_` 分别是 780.03 与 0.29784，**量纲不同，无法互比**（实测 13）。
- **只有 classical MDS / PCA 有解析解**（特征分解），其余全部是迭代或图算法，因此「随机种子」「初始化」「收敛判据」只对后面几类有意义。
- **只有 LLE 家族保的是「重构权重」而不是「距离」**，这也是它最容易被误读的一点：L66 说「preserves distances within local neighborhoods」，但实现解的是局部重构问题；标准 LLE 在 digits 上把类 0 拉成一条长尾（图 6）正是「权重保住了、几何没保住」的证据。

## kNN 图是共同底座（Isomap / LLE / Spectral / t-SNE）

除 MDS 三型与 PCA 外，其余方法的第一阶段都是**同一个操作**：在原始空间建 $k$ 近邻图。

- Isomap 用 BallTree/KD 树（`neighbors_algorithm='auto'`）找 $k$ 邻居，然后在图上跑 Dijkstra 或 Floyd–Warshall 求全源最短路（官方折叠块，见来源页转换损失）。
- LLE 用 $k$ 邻居解 $k\times k$ 的局部线性方程（$O[DNk^3]$）。
- Spectral Embedding 用 $k$ 邻居建**亲和度矩阵**，再算 Laplacian（$L=D-A$，或用归一化 $D^{-1/2}(D-A)D^{-1/2}$）。
- t-SNE 用 `perplexity` 定义的有效邻居数做高斯带宽（perplexity $=2^{S}$，$S$ 是条件分布的 Shannon 熵）。

由此有三条工程后果（本库综合）：

1. **距离度量与尺度就是模型的定义**，不是优化项。来源自己承认：「manifold learning methods are based on a nearest-neighbor search, the algorithm may perform poorly otherwise」[[sklearn-manifold-learning]], L176。实测：S-curve 混入一个放大 1,000 倍的特征后，Isomap 的局部保真度从 0.8952 掉到 **0.5029**（来源页本机实测补充段）。
2. **$k$ 的敏感度高于 KNN 分类器**。KNN 里 $k$ 只是平滑程度；这里 $k$ 决定图的连通性（Isomap 会 warning 并**自动补边**，实测 11）、决定 LLE 的秩亏程度（$k>D$ 即秩亏，L78）、决定 MLLE/HLLE 是否直接报错（硬下界，L80/L90）。同一族的四种 `method` 共享 `n_neighbors` 参数名，**下界却不同**。
3. **图是可缓存的中间产物**：`SpectralEmbedding(affinity='precomputed')`、`Isomap(metric='precomputed')` 都能吃预计算矩阵，这与 [[nearest-neighbors]] 里 `KNeighborsTransformer` 的位置完全一样——把最贵的一步（邻居搜索）从「每次拟合重做」变成「算一次复用」。t-SNE 也能吃预计算距离（`metric='precomputed'`），但没有暴露图本身。

## 三阶段成本模型

官方把 Isomap/LLE 家族的复杂度拆成三个阶段（折叠块，剪藏全丢，见来源页转换损失）：

| 阶段 | Isomap | 标准 LLE | MLLE | HLLE | Spectral | LTSA |
| --- | --- | --- | --- | --- | --- | --- |
| 邻居搜索 | $O[D\log(k)N\log N]$ | 同 | 同 | 同 | 同（建图） | 同 |
| 权重/图构造 | 最短路 $O[N^2(k+\log N)]$（Dijkstra）或 $O[N^3]$（Floyd–Warshall） | $O[DNk^3]$ | $+\,O[N(k-D)k^2]$ | $+\,O[Nd^6]$ | Laplacian $O[N^2]$（稠密图） | $+\,O[k^2d]$ |
| 特征分解 | $O[dN^2]$（ARPACK 更快） | $O[dN^2]$ | 同 | 同 | $O[dN^2]$ | 同 |

三条读法：

- **主导项几乎总是最后一项 $O[dN^2]$**（除非 $k$ 或 $d$ 很大）——所以这些方法对 $N$ 的伸缩性都是二次的，属于「几千到几万样本」的工具，不是百万级的工具。官方对 t-SNE 也给了同一量级：精确法 $O[dN^2]$、Barnes-Hut $O[dN\log N]$（折叠块）。
- **HLLE 的 $d^6$ 是它唯一真正贵的地方**：$d=2$ 时 $d^6=64$ 可忽略，$d=10$ 时 $10^6$ 直接失控。这解释了来源页的分歧 3——L90 说「cost comparable … for small output dimension」，而同页 $d=2$ 的实测差距是 9.7 倍（0.187s vs 1.811s），说明常数因子本身就很大。
- **MLLE 的额外项 $O[N(k-D)k^2]$ 随 $k-D$ 增长**，所以在 $k\gg D$ 时它取代标准 LLE 成为第二贵项。图 7 的 2.434s（最慢）与这条一致。

## 正则化：LLE 的 `reg` 与「消除秩亏」的两条路

这一节在本库的正则化谱系里补了一个**新的作用对象**（详见 [[regularization-and-sparsity]] 的对应小节）：

- **标准 LLE 的 `reg`（默认 `1e-3`）不是惩罚项，而是数值稳定器**。来源写得比大多数文档清楚：「the matrix defining each local neighborhood is rank-deficient … standard LLE applies an arbitrary regularization parameter $r$, which is chosen relative to the trace of the local weight matrix」，且「as $r \to 0$, the solution converges to the desired embedding」但没有 $r>0$ 的最优性保证，表现为「embeddings which distort the underlying geometry」[[sklearn-manifold-learning]], L78。
- 因此 **MLLE 与 HLLE 不是"更好的 LLE"，而是"不用 $r$ 的两条替代路线"**：MLLE 在每个邻域里用多个权重向量（把秩补回来），HLLE 用 Hessian 二次型（换一个二次型）。它们各自引入新的超参（`modified_tol`、`hessian_tol`）与新的下界（$k>n_c$、$k>n_c(n_c+3)/2$）。
- **实测显示默认 `reg` 恰好在甜点上**：$10^{-6}$ → trust 0.9104、$10^{-3}$ → **0.9861**、$10^{-1}$ → 0.9521、$1.0$ → 0.9519。两个方向的失效机制不同（过小＝秩亏，过大＝几何被抹平），所以「调小 `reg` 让结果更准」是错的（来源页本机实测 7）。
- **它不是第八个正则化面**：`reg` 罚的是**局部重构权重矩阵**，与模型系数、协方差、树结构、集成预算都不同层级；而且它不产生稀疏、不改变假设空间。

## 评估：两个互补的指标

来源**没有给任何评估指标**（[[sklearn-manifold-learning]] 局限 1）。可用的量化手段有三类：

1. **局部保真度 `trustworthiness(X, X_embedded, n_neighbors=5)`**（`sklearn.manifold` 里的函数，外部核查）。读法：嵌入空间里的近邻有多少在原始空间里也近。它**只惩罚"不该近的变近了"**（把假邻居引入的失真），对「真邻居被推远」不敏感（那是 continuity 指标管的事），也不衡量簇间距离是否合理——所以它给 t-SNE 打高分是**设计使然**。
2. **全局保真度（本库补充，非官方 API）**：原始与嵌入的成对距离的 Spearman 相关。它捕捉 t-SNE 类方法丢掉的全局结构。
3. **任务代理**：把降维结果接下游分类器做 CV。它最贴近用途，但会把「降维质量」与「下游容量」混在一起。

实测的三者并不一致（**局部列：digits 前 500 样本、2 维；全局列：另取 200 样本子集单独拟合**——两列样本不同，只能看相对次序）：

| 方法 | 局部 trustworthiness | 全局 Spearman | 结构读数 |
| --- | --- | --- | --- |
| 随机嵌入（基线） | 0.5091 | — | 无结构 |
| PCA | 0.8538 | **+0.6456** | 全局最好、局部最差 |
| Isomap | 0.9143 | +0.5518 | 两端均衡 |
| LLE | 0.9197 | +0.4616 | 局部尚可、全局弱 |
| Spectral Embedding | 0.9665 | +0.4199 | 局部强、全局最弱（$k$ 大） |
| t-SNE(`init='pca'`) | **0.9957** | +0.6201 | 局部最好、全局逼近 PCA |
| t-SNE(`init='random'`) | 0.9918 | +0.5141 | 同上，但全局掉一档 |

**这张表最重要的读法**：`init='pca'` 把 t-SNE 的全局保真度从 0.5141 拉到 0.6201（+0.106），而局部保真度几乎不动（0.9918 → 0.9916）。这正面验证了来源 L164 的说法——「global structure is not explicitly preserved. This problem is mitigated by initializing points with PCA」——并且给出它的量级：**缓解，不是解决**（仍低于 PCA 的 0.6456）。

## 与其他类目的定位

| 类目 | 输出 | 是否用 $y$ | 能否给新样本定位 | 本库页面 |
| --- | --- | --- | --- | --- |
| 特征选择（删列） | 原列的子集 | 可以（有监督） | **能**（同一套列） | [[feature-selection]] |
| 流形学习（换坐标） | 新坐标 | 本节的典型情形不用 | **不能**（除 Spectral/Isomap 的近似外） | 本页 |
| PCA（线性换坐标） | 新坐标 | 不用 | 能（`transform`） | [[sklearn-manifold-learning]] 的对照 |
| LDA（有监督线性换坐标） | 新坐标（$\le K-1$ 维） | 用 | 能 | [[gaussian-discriminant-analysis]] |
| KNN（非泛化） | 预测 | 用 | 能（但成本 ∝ $N$） | [[nearest-neighbors]] |
| MLP（学表示） | 预测 + 隐层表示 | 用 | 能 | [[multi-layer-perceptron]] |

两条边界：

- **「删列」与「换坐标」不可互换**：删列保住原列的可解释性与下游稀疏性，换坐标保住几何；`feature selection` 那节的引言也写了 "feature selection/dimensionality reduction"（见 [[feature-selection]] 的分歧 1），但页内并没有降维方法。
- **本节的输出不是特征工程的安全替身**：`SpectralEmbedding`/`TSNE`/`MDS` **没有 `transform`**，根本不能放进「训练一次、推理多次」的管线；`Isomap`/`LLE` 虽然有 `transform`，但它是对新样本的外推近似（实测训练集上 `transform` 与 `embedding_` 差 3.4e-13 / 2.3e-4），与训练集的坐标含义不完全相同，官方本节也不说明它做了什么。

## 失效模式

1. **未缩放**。kNN 决定图，图决定一切；一个量纲大的特征可以独占全部邻居（实测 trust 0.8952 → 0.5029）。
2. **$k$ 太小**：图不连通 → Isomap 静默补边（保的是人造距离）、Spectral 只 warning。
3. **$k$ 太大**：局部性被抹平 → 邻域跨越流形边界，LLE/Spectral 都会把两个流形粘在一起。
4. **噪声短路**：桥接噪声让本应分离的区域在图上相连（L178）。实测方向一致但温和（桥点 400 时分离比 12.54 → 10.71）。
5. **秩亏/奇异权重矩阵**：等价重复点、不连通组 → `eigen_solver='arpack'` 抛 `Factor is exactly singular`（实测复现，触发条件比 L179 描述的窄）。
6. **把 t-SNE 当降维预处理**：没有 `transform`、簇间距离无意义、不同种子的坐标不可比较（实测坐标最大差 3.54，KL 只差 0.0035）。`SpectralEmbedding` 同样没有 `transform`（这不是 t-SNE 特例）。
7. **用 `stress_` 比较不同 MDS 配置**：量纲随 `metric_mds`/`normalized_stress` 变（780 → 0.298），跨版本也不可比（`n_init` 1.9 从 4 变 1）。
8. **以为 MLLE/HLLE 是"更准的 LLE"**：它们在 digits 的官方示例里塌成细丝（图 7–9），而标准 LLE 至少保住了一个可读的中央团。
9. **在稀疏矩阵上直接跑**：LLE 拒绝稀疏；t-SNE 的默认 `init='pca'` 拒绝稀疏（要 `init='random'`）；Spectral 的 `nearest_neighbors` 亲和在稀疏输入上**静默回退 `rbf`**。
10. **忽略 LLE 家族的上界**：`n_components > n_features` 直接 `ValueError`（Isomap 却允许，甚至 $n_c >$ 特征数也能跑）。

## 待解问题

1. 本征维数（"artificially high" 的实际维数）怎么估？本库没有来源覆盖 MLE/correlation dimension。
2. `SpectralEmbedding.n_neighbors=None` → $n/10$ 的默认在图规模上的可行性（$n=10^6$ → $10^5$ 邻居）未测、未见讨论。
3. 噪声短路的**定量阈值**（桥密度 vs 分离比）只测了一个点。
4. `transform` 的语义：`Isomap` 与 `LLE` 对**新样本**到底做了什么外推？官方本节不提；`SpectralEmbedding` 为什么没有 `transform`（算法上没有自然的外推？）也没有来源解释。
5. 降维放进 `Pipeline` 后的泄漏与成本：每一折都要重建图，成本 $\times$ 折数；且 `fit_transform` 在测试集上的语义与训练集不同。本库只做了推理，未实测。
6. MLLE/HLLE 的细丝塌缩是否可用 `k` 与 `reg` 的二维扫描消除。
7. 与 UMAP（本库无来源）的关系：t-SNE 的全局结构缺陷是否被 UMAP 实质上解决，本库**不主张**。

## 关联页面

- 来源：[[sklearn-manifold-learning]]（含 18 张图的目视读数、15 条本机实测、8 条分歧）
- 实践：[[manifold-learning-selection-and-tuning]]
- 概念：[[nearest-neighbors]]（kNN 图与距离尺度）、[[clustering]]（同一份无监督数据的另一种产物：分组而非换坐标；两者共用 kNN 图底座）、[[matrix-factorization]]（第三条无监督路线：矩阵分解，输出可解释成分）、[[feature-selection]]（删列 vs 换坐标）、[[regularization-and-sparsity]]（LLE 的 `reg` 与其他正则面的区别）、[[gaussian-discriminant-analysis]]（LDA 作为有监督线性降维）、[[multi-layer-perceptron]]（另一种"学表示"，但有监督且可 `transform` 到输出）
- 工具：[[scikit-learn]]
