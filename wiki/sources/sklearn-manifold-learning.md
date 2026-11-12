---
title: "scikit-learn 用户指南 2.2「Manifold learning」：Isomap、LLE 四个变体、三种 MDS 与 t-SNE（stable 文档剪藏，未回填）"
type: source
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - dimensionality-reduction
  - manifold-learning
  - reference-doc
source_path: raw/articles/sklearn/2.2. Manifold learning.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 2.2「Manifold learning」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 2.2. Manifold learning（scikit-learn 用户指南第 2.2 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**0 段 doctest、0 段代码块、1 个行间公式（L134–L136 的 strain，跨 3 行）、大量行内公式（如 `$\delta_{i j}$`）、1 个 admonition（`[!note]`）、18 张图、0 张表** |
| 原文 | <https://scikit-learn.org/stable/modules/manifold.html>（剪藏 frontmatter `source`，未钉版本；2026-11-11 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/2.2. Manifold learning.md`（**182 逻辑行**、23,039 字节、`wc -w` 2,334 词，含 frontmatter 与页首《Jungle Book》题词；`wc -l` 报 181 是因为**末行无换行**；**未做任何回填**） |
| SHA-256 | `2c89c6f15c8d9ecc734e0378cc6ef23668e153273ed8a175eddca7d172ed1d44` |
| 完整性 | **本库目前最完整的一份剪藏**：正文（含小节的文献列表）逐句保留，**14 条参考文献一条不少**；缺的只有官方页 **10 个 `<details>` 折叠块（1,669 词）**、**1 张图**（`sphx_glr_plot_mds_001.png`）与公式渲染噪声。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.8.0 – 1.9.1**，工作版本取 1.9.1（见「版本判定」） |
| 插图 | 18 张（官方页 19 张），已于 2026-11-11 全部下载入库 `raw/assets/sklearn-manifold/` 并逐张目视核查（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-manifold-learning]], L128`。凡超出剪藏内容的部分（API 默认值、版本史、运行时行为、官方折叠块原文）都显式标注「外部核查」「官方 1.9.1 页」或「本机实测」。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `2c89c6f1…`）。丢失的 10 个折叠块只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库**第一个「纯表征学习（representation-only）」类目**：前面十五个来源虽然在题材上横跨智能体与模型族，但凡是给出估计器的，都是「给定 $X$（可以有 $y$），输出一个预测器或一个列子集」；本节是第一个「只给 $X$，输出一套新坐标」的类目。这对本库的意义有五层。

1. **第一次出现「没有 $y$」的几何目标**。全节没有损失与标签，只有「保住什么量」：Isomap 保测地距离（L56），LLE 保局部重构权重（L66），Spectral Embedding 保图上的邻近（L100），MDS 三型分别保距离/序/内积（L122、L128），t-SNE 保邻域概率（L149）。**方法的差别就是被保量的差别**（本库综合，来源从不这样组织）。
2. **第一类「输出形状固定、但没有任何预测语义」的估计器**。上一节 [[multi-layer-perceptron]] 至少还能 `predict`；这里只有坐标：`Isomap` 与 `LocallyLinearEmbedding` 有 `transform`，`SpectralEmbedding` **只有 `fit_transform`**（无 `transform`），`TSNE`、`MDS`、`ClassicalMDS` 也没有 `transform`（实测 1.6.1 的 `hasattr`）。**无法给新样本定位不是缺陷，是定义**：`TSNE` 学到的是这一批点的布局，不是一个可外推的映射。**把这类降维器当特征提取器塞进 `Pipeline` 时，测试集只能用 `fit_transform` 重跑**——这直接决定了它们不能出现在需要「训练/推理分离」的线上路径里（本库判断）。
3. **第一次把「同一个算法名字后面挂着四个变体」写实**：`LocallyLinearEmbedding(method=...)` 的 `standard`/`modified`/`hessian`/`ltsa` 用的是**同一个类、同一批参数名**，但约束与代价完全不同（`n_neighbors` 的下界从 $\varnothing$ 变成 $> n_\text{components}$ 再变成 $> n_c(n_c+3)/2$，L80、L90；成本见「图表核查」的时间读数）。这是本库第二个「同一入口、不同模型」的案例（第一个是 `SVC(kernel=...)`）。
4. **又一次「官方资产与官方正文互相矛盾」，而这次矛盾在同一页内部**：L90 说 sklearn 对 HLLE 的算法改进让它「cost comparable to that of other LLE variants」，而同一页图 8 的标题写着 `Hessian LLE embedding (time 1.811s)`，标准 LLE 那张（图 6）是 `0.187s`——**同一份示例、同一个输出维数、差 9.7 倍**。（本库前一次同类发现是 [[sklearn-decision-trees]] 里 2014 年的陈旧 `iris.svg`，那次是**跨版本不变**的资产，这次是**同版同页**的正文与图。）
5. **它是本库目前唯一「正文完整、只有数学被折叠掉」的剪藏**：14 条文献全在，散文逐句对齐，丢的 1,669 词全是复杂度三阶段、MDS 的 metric/non-metric 细节、t-SNE 五个优化参数与 Barnes-Hut 的四条限制。因此本页可以**逐条比对正文与实现**，而不像前几份那样只能讨论「丢了什么」。

## 文档结构

| 章节 | 解决什么问题 | 保什么 | 行号 |
| --- | --- | --- | --- |
| 引子 | 一句话定义：非线性降维；"dimensionality is only artificially high" | — | L28 |
| 2.2.1 Introduction | 为什么要降维（可视化）；随机投影与线性方法（PCA/ICA/LDA）不够；5 个官方示例 | — | L30–L52 |
| 2.2.2 Isomap | MDS/KernelPCA 的推广，保**测地距离** | 全局测地距离 | L54–L62 |
| 2.2.3 LLE | 保**局部邻域内的距离**（来源措辞；实现保的是重构权重，见「分歧」7），等价于「一串局部 PCA 再全局对齐」 | 局部重构权重 | L64–L72 |
| 2.2.4 MLLE | 用多个权重向量解决 LLE 的**正则化问题**（秩亏）；要求 `n_neighbors > n_components` | 同上 + 多个权重 | L76–L86 |
| 2.2.5 HLLE | 用每个邻域的 **Hessian 二次型**解决同一个秩亏；要求 `n_neighbors > n_components*(n_components+3)/2` | 局部 Hessian | L88–L96 |
| 2.2.6 Spectral Embedding | 图 Laplacian 的谱分解（**Laplacian Eigenmaps**）；图是低维流形的离散近似 | 图上邻近 | L98–L104 |
| 2.2.7 LTSA | 用局部**切空间**刻画几何再全局对齐；不是 LLE 的变体但算法相近 | 局部切空间 | L106–L114 |
| 2.2.8 MDS | 三大类（metric / non-metric / classical）；raw stress、Stress-1、strain 三个目标 | 距离 / 距离的序 / 内积 | L116–L145 |
| 2.2.9 t-SNE | 原空间高斯联合概率 ↔ 嵌入空间 Student-t；最小化 KL | 邻域概率 | L147–L172 |
| 2.2.10 Tips | 缩放、用重构误差选维数、噪声短路、奇异权重矩阵与 `solver='dense'` | — | L174–L179 |
| `[!note]` See also | `RandomTreesEmbedding` 也能给非线性表示，但不降维 | — | L181–L182 |
| （丢失的复杂性折叠块 ×6） | Isomap/LLE/MLLE/HLLE/Spectral/LTSA 的三阶段复杂度 | — | 官方页；剪藏无 |
| （丢失的 MDS ×2、t-SNE ×2 折叠块） | metric/non-metric 的差异公式、Stress-1、五个优化参数、Barnes-Hut 四条限制 | — | 官方页；剪藏无 |

## 关键主张（按剪藏顺序）

### A. 定位（L28–L42）

- **"Manifold learning is an approach to non-linear dimensionality reduction"**（L28）。这是全节唯一的定义句，且紧接着给出一句无操作意义的动机："the dimensionality of many data sets is only artificially high"——**没有给判定准则**（怎么知道是"人为高"？见「待解问题」）。
- **随机投影是最简单的降维，但"the randomness of the choice leaves much to be desired"**（L34）——官方明确不推荐，却仍然把它作为 `plot_lle_digits` 的基准面板（图 2）。
- **线性框架（PCA/ICA/LDA）"can be powerful, but often miss important non-linear structure"**（L38），而流形学习是"an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure"（L42）。这条把 Isomap 放在 MDS/KernelPCA 的延长线上（L56），是理解全节的钥匙：**这些方法不是 PCA 的替代品，而是「换掉被保量」的 PCA**。
- 官方示例 5 个（L46–L50）：手写数字（`plot_lle_digits`）、S-curve 对比（`plot_compare_methods`）、股市结构、被切开的球面、Swiss Roll。剪藏本身只保留了前两个示例的图。

### B. Isomap（L56）

- **"seeks a lower-dimensional embedding which maintains geodesic distances between all points"**（L56）——注意是**所有点**。因此它依赖 kNN 图上的最短路作为测地距离的估计：$k$ 太小 → 图不连通（实测会 warning 并**自动补边**，见「本机实测」11）；$k$ 太大 → 最短路穿过流形外部的"捷径"。
- L56 说它是"MDS 或 KernelPCA 的扩展"。本库读法：Isomap = **MDS + 用图距离替换欧氏距离**，这一步替换是它唯一的创新，其余（双重中心化、特征分解）与 classical MDS 共用（L128、L138 给的 strain 公式就是这条桥）。

### C. LLE 家族（L66–L96）

- LLE："seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods"（L66），并给出一个非常好用的实现直觉——**"a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding"**（L66）。
- **LLE 的已知病**（L78）：邻域数大于输入维数时，每个局部邻域的权重矩阵**秩亏**；标准 LLE 加一个相对局部权重矩阵迹的任意正则参数 $r$；"as $r \to 0$ the solution converges to the desired embedding"，但 $r>0$ 时**没有最优性保证**，表现为"embeddings which distort the underlying geometry"。
- 两个补丁，两条不同路线（L80、L90）：MLLE 用**多个权重向量**（每邻域多组解），HLLE 用**Hessian 二次型**（直接用二阶结构）。两者都只是"解决同一个正则化问题的另一种办法"，**不是更高的精度档位**（本库判断：本页 2.2.4/2.2.5 两节没有给任何"哪个更好"的判据）。
- HLLE 的成本声明（L90）："other implementations note its poor scaling with data size, `sklearn` implements some algorithmic improvements which make its cost comparable to that of other LLE variants for small output dimension"——这句话与同页图 8 的标题时间冲突（见「分歧」3）。
- 约束（L80、L90）：MLLE 要求 `n_neighbors > n_components`；HLLE 要求 `n_neighbors > n_components*(n_components+3)/2`。**这两个下界是硬报错，不是建议**（实测见「本机实测」4–5）。

### D. Spectral Embedding 与 LTSA（L100、L108）

- Spectral Embedding 在本库实现的是**Laplacian Eigenmaps**（L100），不是一般意义上的"谱方法"。它的输入是图：`affinity='nearest_neighbors'`（默认）先建 kNN 图，`'rbf'` 建全连接核图，`'precomputed'` 直接吃邻接矩阵。
- 关键句："The graph generated can be considered as a discrete approximation of the low dimensional manifold in the high dimensional space"（L100）——因此**图的构造方式就是模型的定义**，与 KNN 的距离尺度问题同源（见 [[nearest-neighbors]]）。
- 它优化的是"图上相近 → 嵌入中相近"的单向约束（L100），所以**不保证远点被推开**：实测在 S-curve 上它把二维流形压成一条弧（图 5），这是这条单向约束的直接后果（本库综合）。
- LTSA（L108）："not technically a variant of LLE"但算法相近；**保的不是邻域距离而是局部切空间**，再做"global optimization to align these local tangent spaces"。

### E. MDS 三型（L118–L138）

- 三种算法（L122）：**metric**（嵌入距离尽量逼近不相似度）、**non-metric**（只保序，求单调关系）、**classical**（近似**成对的内积**而非距离，有特征分解的解析解）。
- 目标函数（L126）：disparities $\hat d_{ij}=f(\delta_{ij})$，raw stress $=\sum_{i<j}(\hat d_{ij}-d_{ij}(Z))^2$。
- classical MDS（L128）：又称 **PCoA** 或 **Torgerson's scaling**，独立类 `ClassicalMDS`；损失叫 *strain* 而非 stress，$B=-\frac12 C\Delta C$（$C=I-J/n$ 是双重中心化矩阵），"can be minimized exactly using the eigendecomposition of $B$"（L138）。
- 最重要的等价关系（L128）：**"If the dissimilarity matrix consists of the pairwise Euclidean distances between some vectors, then classical MDS is equivalent to PCA applied to this set of vectors."** 这是本库「PCA / MDS / Isomap / spectral 是同一个特征分解的四张脸」的第一条官方依据。
- 尺度陷阱（本库综合 + 实测）：raw stress、Stress-1 与 strain **三者量纲不同**，`stress_` 的数值不能跨 `metric_mds` 与 `normalized_stress` 组合比较（实测同一数据 780.03 vs 0.29784，见「本机实测」13）。

### F. t-SNE（L149–L164）

- 机制一句话（L149）：原空间的亲和度用**高斯联合概率**表示，嵌入空间的用**Student's t 分布**表示，于是对局部结构特别敏感。
- 三条优点（L151–L153）：单张图上看多个尺度、能分开多个流形/簇、减少"点都挤在中心"的倾向。
- **与 Isomap/LLE 的分工**（L155）："While Isomap, LLE and variants are best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure"——这句话是全节唯一的方法选择判据，本库把它写进实践页的第一屏。
- KL 非凸（L157）："multiple restarts with different initializations will end up in local minima"，所以官方**明确允许**"try different seeds and select the embedding with the lowest KL divergence"，且 L163 再说一次"it is perfectly legitimate to pick the embedding with the least error"。
- 四条缺点（L161–L164）：慢（百万样本数小时 vs PCA 秒级）、Barnes-Hut 限 2–3 维、随机性、全局结构不显式保留（用 `init='pca'` 缓解）。

### G. 实践建议（L176–L179）

- **缩放**（L176）："Make sure the same scale is used over all features. Because manifold learning methods are based on a nearest-neighbor search, the algorithm may perform poorly otherwise."——全节唯一一条硬前提，且给了理由（kNN）。
- **用重构误差定维数**（L177）："The reconstruction error computed by each routine can be used to choose the optimal output dimension. For a $d$-dimensional manifold embedded in a $D$-dimensional parameter space, the reconstruction error will decrease as `n_components` is increased until `n_components == d`."
- **噪声短路**（L178）："noisy data can 'short-circuit' the manifold, in essence acting as a bridge between parts of the manifold that would otherwise be well-separated"——并承认"active area of research"。
- **奇异权重矩阵**（L179）：两点以上完全相同、或数据被切成不相连的组 → `solver='arpack'` 找不到零空间；"The easiest way to address this is to use `solver='dense'`"，或者增大 `n_neighbors`（若因不连通）、删掉重复点（若因重复）。

## 版本判定

**结论：1.8.0 – 1.9.1，工作版本 1.9.1，证据强。** 判定链如下（全部为本库 2026-11-11 的实测比对）。

1. **句子探针**：把剪藏散文切成 82 条 ≥14 词的句子（取前 12 词），在官方 1.6/1.7/1.8/1.9 四版页正文中查找。命中数：1.6 = **61**、1.7 = **68**、1.8 = **75**、1.9 = **75**、`/stable/` = **75**。7 条未命中项逐条核为公式/markdown 渲染差异（`\left(\right.`、`$$…$$`、`[!note]` 记号、`**` 残片），**无一条是内容缺失**。
2. **1.7 → 1.8 的 11 处逐词差异**中，有 4 处是剪藏**含有**的新内容，因此剪藏 ≥ 1.8：
   - `MDS` 的类名从 "MDS" 变为 "**MDS and ClassicalMDS**"（L118）；
   - "There exist **three** types of MDS algorithm: metric, **non-metric, and classical**"（L122，1.7 只有两种）；
   - "**Finally, classical MDS is close to PCA and, instead of approximating distances, approximates pairwise scalar products**"（L122）；
   - 整段 "Classical MDS, also known as *principal coordinates analysis (PCoA)* or *Torgerson's scaling*…"（L128）与 strain 公式（L132–L138）。
   外部旁证：`whats_new/v1.8` 的 `sklearn.manifold` 段落写着 "Major Feature `manifold.ClassicalMDS` was implemented…"（#31322）与 "`manifold.MDS` now supports arbitrary distance metrics … initialization via classical MDS"（#32229）。
3. **1.8 → 1.9 的正文 diff 为空**（逐词比较，无任何 opcode），因此无法再细分；抓取日 `/stable/modules/manifold.html` 的页标题是 **1.9.1**，与 `/1.9/` 页字节级相同（95,499 B），故取 1.9.1 作为工作版本。
4. **反证**：1.6 版把折叠块标题写成 "Nonmetric MDS"，1.8 起改为 "Non-metric MDS"；1.7 页有 `sphx_glr_plot_lle_digits_013.png` 而 1.8 页换成了 `_011`/`_012`/`_015` 三张——剪藏的图号集合与 1.8+ 一致。

本节的功能性结论（除默认值外）**不依赖** 1.8 vs 1.9 的区分。

## 转换损失

与 [[sklearn-ensembles]]（丢 21%）、[[sklearn-neural-networks]]（丢 36%）相比，本次是**结构损失最小、数学损失最大**的一份：散文与文献都在，丢的全是「公式与参数说明」。

| 丢失项 | 词数 | 内容 | 后果 |
| --- | --- | --- | --- |
| `Complexity`（Isomap） | 206 | 三阶段：BallTree 邻居搜索 $O[D\log(k)N\log N]$；最短路（Dijkstra $O[N^2(k+\log N)]$ 或 Floyd–Warshall $O[N^3]$，由 `path_method` 选）；部分特征分解（稠密 $O[dN^2]$，ARPACK 更快） | 剪藏**完全没有复杂度**，选型只能靠本节图标题的时间读数 |
| `Complexity`（LLE） | 97 | 权重矩阵构造 $O[DNk^3]$；总复杂度 $O[D\log(k)N\log N]+O[DNk^3]+O[dN^2]$ | 同上 |
| `Complexity`（MLLE） | 126 | 多权重构造**额外** $O[N(k-D)k^2]$，并注明"added cost … relatively small compared to stages 1 and 3" | 解释了为什么 MLLE 的实测时间主要不是来自权重构造 |
| `Complexity`（HLLE） | 102 | 额外 $O[Nd^6]$，来自"QR decomposition of the local hessian estimator" | 这是 L90「cost comparable」的**量化对手**：$d$ 大时 $d^6$ 立刻失控 |
| `Complexity`（Spectral） | 105 | 三步：加权图、Laplacian（$L=D-A$ 与归一化 $D^{-1/2}(D-A)D^{-1/2}$）、特征分解 | 剪藏只给了 L100 的定性描述，**归一化 Laplacian 的公式只在这里** |
| `Complexity`（LTSA） | 89 | 额外 $O[k^2 d]$ | — |
| `Metric MDS` | 23 | "In the metric MDS model (sometimes also called **absolute MDS**), disparities are simply equal to the input dissimilarities $\hat d_{ij}=\delta_{ij}$" | 这是 L122「metric = 距离尽量贴近」的**公式落点**；剪藏只有描述，没有 $\hat d=\delta$ |
| `Non-metric MDS` | 148 | **isotonic regression** 求 disparities、单调变换、每步优化后重做；disparities 归一化避免全点重合；Stress-1 公式；`normalized_stress=True` | 剪藏 L122 只有一句"preserve the order of the distances"，**没有任何实现机制**；Stress-1 的公式完全不在剪藏里 |
| `Optimizing t-SNE` | 417 | **五个参数**（perplexity、early exaggeration、learning rate、max iterations、angle）；perplexity $=2^{S}$（$S$ 为条件分布的 Shannon 熵）的读法；early exaggeration 的两阶段；学习率过低/过高的两种坏结果；Belkina et al. (2019) 的 $n/\text{ee}$ 启发式与 `learning_rate='auto'`；angle 的精度/速度权衡 | **全节最痛的丢失**：剪藏只列了 t-SNE 的优点与缺点，没有一句调参指导。实践页的 t-SNE 部分因此只能引官方 1.9.1 页 |
| `Barnes-Hut t-SNE` | 332 | 梯度 $O[dN\log N]$ vs 精确 $O[dN^2]$；**四条限制**（目标维 ≤3、仅稠密输入、angle 仅在近似法有意义、可扩展到数十万点 vs 精确法几千点）；"for visualization purpose … strongly recommended"；以及一段重要的反例警告——"failing to visualize well separated … does not necessarily imply that the data cannot be correctly classified" | 剪藏 L162 只保留了"限 2–3 维"一条限制，另外三条与那句反例警告全丢 |
| 图片 `sphx_glr_plot_mds_001.png` | — | 官方 2.2.8 的第三张图（在 `_011` 与 `_012` 之间） | 剪藏只有 18/19 张图；**未见该图，故本页不对它作任何描述** |
| 公式渲染噪声 | — | `\left(\right.`、`\underset{i}{\sum}` 被拆成 `$\underset{i , j}{\sum} \left(\right. …`（L126、L135）、`$r 0$`（L78，原文 $r>0$ 的 `>` 被吞） | L78 的 "$r>0$ 时无保证" 在剪藏里读作 `$r 0$`，会读成"$r$ 到 0" |

**未丢失的部分**：14 条参考文献全在（L62、L72、L86、L96、L104、L114、L142–L145、L169–L172），页内锚点与交叉引用也基本完整。这是本库第一份「文献清单完整」的 sklearn 剪藏。

## 图表核查

18 张图于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-manifold/`，**逐张目视核查**并记录尺寸与 SHA-256 前 16 位。图的标题自带运行时间，是本页唯一的成本证据。

| # | 文件 | 尺寸 | SHA-256 前 16 | 目视读数 |
| --- | --- | --- | --- | --- |
| 1 | `plot_compare_methods_001` | 600×600 | `8476461c82fb113b` | 标题 "Original S-curve samples"；3D 散点，坐标轴约 $x\in[-1.5,1.5]$、$z\in[-2,2]$、色标 $-4..4$；点云是一个完整的 S 形带 |
| 2 | `plot_lle_digits_002` | 640×480 | `6d7892f9b970afea` | 标题 "Random projection embedding (**time 0.001s**)"；全部类别的数字缩略图叠成一团，**没有可辨的簇**；无坐标刻度 |
| 3 | `plot_lle_digits_003` | 640×480 | `9e9ff627fa2dc4b5` | 标题 "**Truncated SVD** embedding (**time 0.003s**)"；轻微分离（0/1 在左、4 在上），大部分仍重叠——**注意标题不是 PCA**（见「分歧」2） |
| 4 | `plot_lle_digits_004` | 640×480 | `afa5614ebfab106c` | 标题 "Linear Discriminant Analysis embedding (**time 0.006s**)"；类间分离明显（0 在下方、4 在右、3 在左），因为用了标签 |
| 5 | `plot_lle_digits_005` | 640×480 | `f02e8684cbb074ad` | 标题 "Isomap embedding (**time 0.346s**)"；0 聚成一团在左、4 在上、1 在右、3 在下；类间有重叠但拓扑保持 |
| 6 | `plot_lle_digits_006` | 640×480 | `66c17a3da57c1302` | 标题 "Standard LLE embedding (**time 0.187s**)"；中央一团 + **一条绿色长尾（类 0）被拉成细丝**，是标准 LLE 在 $k=5$ 下的典型畸变 |
| 7 | `plot_lle_digits_007` | 640×480 | `cb568199345eb09b` | 标题 "Modified LLE embedding (**time 2.434s**)"；**整团塌成一条竖直细丝**，类 0 单独成一小簇在右侧；类别沿细丝排列 |
| 8 | `plot_lle_digits_008` | 640×480 | `3572971e5da5bbf9` | 标题 "Hessian LLE embedding (**time 1.811s**)"；同样是细丝（比 7 稍宽），类 0 分成 3 个小簇；**与 L90「cost comparable」冲突** |
| 9 | `plot_lle_digits_009` | 640×480 | `4c66cacb006ce902` | 标题 "LTSA LLE embedding (**time 2.090s**)"；两条细丝，类 0 沿左侧分离；与 7/8 同族畸变 |
| 10 | `plot_lle_digits_010` | 640×480 | `79436c824288a082` | 标题 "Metric MDS embedding (**time 2.393s**)"；圆盘状布局，类簇按角度排开但边界互相穿插 |
| 11 | `plot_lle_digits_011` | 640×480 | `9c213b1147432480` | 标题 "Non-metric MDS embedding (**time 16.625s**)"；布局与 10 相似（3 在左下、0 在中下），**慢 7 倍** |
| 12 | `plot_lle_digits_012` | 640×480 | `f46b77c0ad587105` | 标题 "Classical MDS embedding (**time 0.105s**)"；同样是圆盘，**比 metric MDS 快 23 倍**，且类 0 在右下更紧 |
| 13 | `plot_lle_digits_015` | 640×480 | `aacf9820f43f484d` | 标题 "t-SNE embedding (**time 2.413s**)"；**6 个清晰分离的簇**（2 左上、1 上、4 右、0 右下、5 下、3 左），簇间大片空白 |
| 14 | `plot_compare_methods_003` | 300×300 | `282cb0396763d74b` | 标题 "Isomap Embedding"；S-curve 被摊成一块**矩形带**（不是直线），带内有可见褶皱；无刻度数字 |
| 15 | `plot_compare_methods_004` | 700×350 | `e1aa17aef2137312` | 三面板 "Multidimensional scaling"：Metric MDS / Non-metric MDS / Classical MDS；三者都是 **")(" 形双弧**，classical 面板最光滑、metric 面板褶皱最多 |
| 16 | `plot_compare_methods_005` | 300×300 | `e3e094b72bb6d1aa` | 标题 "Spectral Embedding"；**S-curve 塌成一条单弧**（二维流形压成一维曲线），颜色沿弧单调过渡 |
| 17 | `plot_compare_methods_006` | 300×300 | `ac7aa5821553e992` | 标题 "T-distributed Stochastic Neighbor Embedding"；一条弯曲的带，**带上有多个白色空洞**（局部排斥造成的撕裂），颜色顺序保持 |
| 18 | `plot_lle_digits_001` | 600×600 | `6f9b79826d98cdf4` | 标题 "A selection from the 64-dimensional digits dataset"；10×10 个数字缩略图，行内标签循环 0–5，可见像素化与模糊 |

**读数要点**：

- 时间口径全部来自**图的标题**（官方示例脚本自己打印的 `time`），不是本节正文；同一份 digits 数据、同样 2 维输出的横向对比是：随机投影 0.001s < TruncatedSVD 0.003s < LDA 0.006s < classical MDS 0.105s < 标准 LLE 0.187s < Isomap 0.346s < HLLE 1.811s < LTSA 2.090s < metric MDS 2.393s < t-SNE 2.413s < MLLE 2.434s < non-metric MDS 16.625s。
- 同一张图上「哪个方法更好看」与「哪个方法保住了什么」经常**互相矛盾**：t-SNE（图 13）视觉上最好，但它在 S-curve 上把流形撕出空洞（图 17）；Isomap（图 14）没有分离簇，却保住了二维拓扑。

## 外部核查（官方 1.9.1 API 页与 `whats_new`，不在剪藏内）

本节**不给任何一个默认值**——11 个小节里出现的参数只有 `method=`、`n_neighbors`、`n_components`、`eigen_solver`（"`solver`" 的口语写法）、`init='pca'`、`learning_rate`（隐含）。以下取自官方 1.9.1 生成页签名与版本注记：

| 估计器 | 1.9.1 签名（默认值） | 剪藏是否提及 |
| --- | --- | --- |
| `Isomap` | `n_neighbors=5`、`radius=None`、`n_components=2`、`eigen_solver='auto'`、`tol=0`、`max_iter=None`、`path_method='auto'`、`neighbors_algorithm='auto'`、`n_jobs=None`、`metric='minkowski'`、`p=2` | 只提 `path_method`/`eigen_solver` 的**名字**（在丢失折叠块里），正文无默认值 |
| `LocallyLinearEmbedding` | `n_neighbors=5`、`n_components=2`、`reg=0.001`、`eigen_solver='auto'`、`tol=1e-06`、`max_iter=100`、`method='standard'`、`hessian_tol=0.0001`、`modified_tol=1e-12`、`neighbors_algorithm='auto'`、`random_state=None` | 正文只说 "$r$ chosen relative to the trace"（L78），**不给 `reg` 的名字与默认值** |
| `SpectralEmbedding` | `n_components=2`、`affinity='nearest_neighbors'`、`gamma=None`、`random_state=None`、`eigen_solver=None`、`eigen_tol='auto'`、`n_neighbors=None` | **一个都没提** |
| `MDS` | `n_components=2`、`metric_mds=True`、`n_init=1`、`init='warn'`、`max_iter=300`、`verbose=0`、`eps=1e-06`、`dissimilarity='deprecated'`、`metric='euclidean'`、`normalized_stress='auto'` | 正文只有 `MDS`/`ClassicalMDS` 两个类名 |
| `ClassicalMDS` | `n_components=2`、`metric='euclidean'`、`metric_params=None`（**无 `random_state`、无 `max_iter`**） | 只有名字与 strain 目标（L128、L138） |
| `TSNE` | `n_components=2`、`perplexity=30.0`、`early_exaggeration=12.0`、`learning_rate='auto'`、`max_iter=1000`、`n_iter_without_progress=300`、`min_grad_norm=1e-07`、`metric='euclidean'`、`init='pca'`、`method='barnes_hut'`、`angle=0.5`、`n_jobs=None` | 只提 `init='pca'`（L164）；五个优化参数全在丢失折叠块里 |
| `trustworthiness` | `trustworthiness(X, X_embedded, *, n_neighbors=5, metric='euclidean')` | **完全没提**：本节没有给出任何量化评估方法 |

版本史（`whats_new` + API 页 "Changed in version" 注记，均属外部核查）：

- `MDS.metric` → **`metric_mds`**（1.8 改名）；`dissimilarity` 1.8 弃用、**1.10 移除**；`init` 1.8 新增（默认值将在 1.10 变为 `classical_mds`）；`n_init` **1.9 从 4 改为 1**；`eps` **1.7 从 `1e-3` 改为 `1e-6`**（伴随收敛判据的 bugfix）；`normalized_stress` 1.2 引入、1.4 默认变 `'auto'`、**1.7 起 metric MDS 也支持归一化 stress**。
- `ClassicalMDS` **1.8 新增**（#31322）；`MDS` 支持任意度量与 classical-MDS 初始化也是 1.8（#32229）。
- `TSNE.learning_rate` 默认值 **1.2 起为 `'auto'`**；1.9 页签名已无 `n_iter`（1.6.1 里它还是 deprecated 别名）。
- 1.9 的 `sklearn.manifold` 变更只有两项：SpectralEmbedding/SpectralClustering 的 ARPACK 调用方式提速（#33262），与 `MDS.fit_transform` 在 `init='classical_mds'` 下返回正确维数（#33318，bugfix）。
- `Isomap` 的邻居计算用 `NearestNeighbors(algorithm=self.neighbors_algorithm)`（1.6.1 源码），**默认 `'auto'`**；丢失折叠块里写的「Isomap uses BallTree」只是 `'auto'` 可能选中的一种后端（$D<15$ 时实际是 KD 树），见「分歧」4。

## 本机实测（scikit-learn 1.6.1，15 条）

**版本警告**：本机是 1.6.1（PyPI 当前可安装的最高版本；1.8/1.9 未安装，`ClassicalMDS` 在 1.6.1 里根本 `import` 不到）。以下只证明**机制存在**，不能替代对 1.9.1 的数值核对；涉及默认值（`n_init`、`eps`、`metric_mds`）的条目已经在 1.9 变了，引用时必须带版本。

1. **「重构误差选维数」（L177）只在 Isomap 上成立，而且在 LLE 上方向相反**。S-curve（1,000 点、$d_\text{intrinsic}=2$、`n_neighbors=10`）：`Isomap.reconstruction_error()` = 0.39864（1 维）、0.09177（2 维）、0.08016（3 维）——**下降并趋于平台**；digits（500×64）：3141.41 → 2360.35 → 1327.00 → 955.55 → 898.16 → 884.27（$d=1,2,5,10,20,30$）。而 `LocallyLinearEmbedding.reconstruction_error_` 在 digits 上**单调上升**：$d=1$ 时 0.000000、$d=2$ 时 0.000007、$d=12$ 时 **0.007899**；在 S-curve 上也是 1.08e-09 → 1.84e-07 → 4.32e-07（$d=1,2,3$）。
2. **API 不对称**：`Isomap` 提供的是**方法** `reconstruction_error()`（不是属性），`LocallyLinearEmbedding` 提供**属性** `reconstruction_error_`；`SpectralEmbedding`、`TSNE` 两者都没有（t-SNE 有 `kl_divergence_`），`MDS` 有 `stress_`。因此 L177 的「each routine」在 API 层不成立。
3. **LLE 的 `n_components` 有硬上界**：$d > D_\text{input}$ 时 `ValueError: output dimension must be less than or equal to input dimension`（S-curve 是 3 维，故 $d\ge4$ 直接报错）。Isomap 没有这个限制：`Isomap(n_components=70)` 在 64 维 digits 上成功返回 `(500, 70)` 的嵌入。
4. **MLLE 的约束是 `>=`，HLLE 的是严格 `>`**：$n_c=3$ 时 modified 在 `n_neighbors=2` 报 `ValueError: modified LLE requires n_neighbors >= n_components`，`k=3` 通过；hessian 在 $k=3$ 报 `ValueError: for method='hessian', n_neighbors must be greater than [n_components * (n_components + 3) / 2]`，$k=10$（$>6$）通过。**与 L80/L90 的措辞（">"）一致**。
5. **LTSA 的下界没有校验**：$k=2$、$n_c=3$ 时不报参数错误，而是在内部抛 `ValueError: could not broadcast input array from shape (2,2) into shape (2,3)`——错误信息对用户无用。
6. **复现了 L179 的 ARPACK 失败**：`LocallyLinearEmbedding(method='modified', n_neighbors=3, n_components=3)`（S-curve，`eigen_solver='arpack'`）→ `ValueError: Error in determining null-space with ARPACK … 'Factor is exactly singular'`，同参数改 `eigen_solver='dense'` 立即成功。**但**我构造的另外三种退化输入（60 个完全相同点；30+30 个两簇重复点；两个相距 50 的高斯团）在 `arpack`/`dense`、$k=5/10$ 下**全部成功**——所以 L179 的触发条件比文字描述的窄得多。
7. **LLE 的 `reg` 真的会改结构**（S-curve，$k=10$，$d=2$）：`reg=1e-6` → trustworthiness 0.9104；`reg=1e-3`（默认）→ **0.9861**；`reg=0.1` → 0.9521；`reg=1.0` → 0.9519。默认值恰好是这一段里的最好点，但**过小与过大都会掉**，且掉的方向不同（过小＝秩亏，过大＝几何被抹平）。
8. **LLE 的坐标不可复现，结构可复现**：同一数据连续两次 `fit_transform`（`eigen_solver='arpack'`）坐标最大差 **3.39e-06**，`trustworthiness` 0.9196 vs 0.9197；`eigen_solver='dense'` 两次完全相同；arpack 与 dense 的坐标最大差 **0.2755**（同一结构的不同基）。`Isomap` 与 `SpectralEmbedding(random_state=0)` 则逐位相同。
9. **`locally_linear_embedding` 家族不吃稀疏输入**：`csr_matrix` 喂给 LLE 直接 `TypeError: Sparse data was passed for X, but dense data is required`；而 `Isomap`、`SpectralEmbedding` 接受稀疏（Spectral 在 `affinity='nearest_neighbors'` + 稀疏时会 **warning 并静默回退到 `rbf`**）。另外 `SpectralEmbedding` **没有 `transform`**（`hasattr` 为 False），只能 `fit_transform`。
10. **t-SNE 的稀疏失败点是 `init`，不是 `method`**：`TSNE` 默认 `init='pca'`，喂稀疏矩阵时无论 `method='exact'` 还是 `'barnes_hut'` 都报 `TypeError: PCA initialization is currently not supported with the sparse input matrix. Use init="random" instead.`——而丢失折叠块只说「Barnes-Hut 仅支持稠密」，会让人以为 `exact` 可以吃稀疏。
11. **不连通图的两种沉默处理**：`SpectralEmbedding` 在明显分离的两团上报 `UserWarning: Graph is not fully connected, spectral embedding may not work as expected.`（仍返回结果）；`Isomap` 报 `UserWarning: The number of connected components of the neighbors graph is 2 > 1. Completing the graph to fit Isomap might be slow.` 并**自动补边**（源码注释同）。后者意味着「保测地距离」在补出来的边上保的是**人造距离**。
12. **t-SNE 的随机性与「按 KL 选种子」（L157、L163）可操作**：同一份 500×64 digits，5 个种子的 `kl_divergence_` = 0.39931 / 0.39915 / 0.39934 / 0.39932 / **0.39580**（跨度 0.00354），而坐标最大差 **3.54**。KL 差异很小、坐标差异很大——**用 KL 排序挑种子是合法的，用坐标比较是无意义的**。
13. **`stress_` 的量纲随配置变（MDS，1.6.1）**：同一组 3 维点、`dissimilarity='precomputed'`，`metric=True, normalized_stress='auto'` → `stress_ = 780.0346`；`metric=False, normalized_stress='auto'` → **0.29784**（Stress-1）；`metric=False, normalized_stress=False` → 140.1556。且 1.6.1 **没有** `normalized_stress_` 属性，`metric=True` + `normalized_stress=True` 直接 `ValueError`（1.7 起才支持）。**结论：`stress_` 不能跨配置比较，也不能跨版本比较。**
14. **`n_init` 是一个真旋钮（1.6.1 默认 4）**：3 维 60 点、`metric=True`、`max_iter=300`：`n_init=1` → stress 431.06（100 轮）、`n_init=4` → **419.09**（66 轮，取四次里最好的一次）。1.9 把默认改成 1，**同一份代码在 1.9 上会变差**（除非显式给 `n_init`）。
15. **`n_components == n_samples` 的报错不可读**：`Isomap(n_components=400)`（400 个样本）→ `ValueError: There are significant negative eigenvalues (0.0836843 of the maximum positive)…`，提示的是「矩阵不是 PSD」，不是「你要求的维数等于样本数」。

另附两条本库补充度量（非官方 API，仅用于本页对照，方法写在 [[manifold-learning-selection-and-tuning]]）：

- **局部保真度 `trustworthiness(n_neighbors=5)`**（digits 前 500 个样本，2 维）：随机嵌入 0.5091、PCA 0.8538、Isomap 0.9143、LLE 0.9197、SpectralEmbedding 0.9665、t-SNE **0.9957**。
- **全局保真度（本库自定义）**：200 个样本上原始欧氏距离与嵌入欧氏距离的 Spearman 相关——PCA **+0.6456**、t-SNE(`init='pca'`) +0.6201、Isomap +0.5518、t-SNE(`init='random'`) +0.5141、LLE +0.4616、SpectralEmbedding +0.4199。**这条同时验证了 L164 的说法**：t-SNE 的全局结构确实差，但 `init='pca'` 把它从 0.5141 提到 0.6201（局部保真度几乎不动：0.9918 → 0.9916）。
- **缩放的影响（L176）**：S-curve（400 点）+ 4 个噪声特征（其中一个放大 1,000 倍），`n_neighbors=10`：不缩放时 Isomap 对真实流形坐标的 trustworthiness 只有 **0.5029**，`StandardScaler` 之后 **0.8952**（LLE 0.5028 → 0.5671）。这是本库目前对「缩放是硬前提」最直接的一组数字。
- **噪声短路（L178）**：两个相距 12 的高斯团（各 150 点）+ $[0,10]$ 区间的均匀桥噪声，`n_neighbors=10`：桥点数 0 → 400 时，kNN 图里跨簇边比例 0.000 → **0.242**，Isomap 嵌入的两簇分离比 12.54 → **10.71**（PCA 的分离比在有无桥噪声时都是 9.76）。**方向与 L178 一致，但在这组参数下效应很温和**——需要更强的桥接或更大的 $k$ 才能复现「短路」的戏剧性版本。

## 分歧

1. **L177「用重构误差选最优输出维数」与实现不匹配。** ① 「each routine」在 API 层不成立（只有 Isomap 与 LLE 暴露重构误差，且一个是方法一个是属性，见实测 2）；② 在 LLE 上误差随 `n_components` **单调上升**（实测 1），与「decrease until `n_components == d`」相反；③ 在 Isomap 上它确实下降，但 digits 上到 30 维仍在缓慢下降（884 vs 898），**没有 L177 预测的那个拐点**。本库结论：这句话只能当「Isomap 的经验做法」，不能当通用判据。
2. **图的标题与正文的称呼不一致。** L40 的图注把 `sphx_glr_plot_lle_digits_003.png` 标成 `PCA_img`，而图片本身的标题是 "**Truncated SVD** embedding"（图 3）。两者在这份数据上结果接近但不是同一个变换（TruncatedSVD 不做中心化）。引用时应写「图里的 TruncatedSVD」，不要写成 PCA。
3. **L90 的「cost comparable」被同页的图否证。** 同一示例、同样 2 维输出：标准 LLE `0.187s`、HLLE `1.811s`（图 8）、MLLE `2.434s`（图 7）、LTSA `2.090s`（图 9）。丢失折叠块里的 $O[Nd^6]$（HLLE）与 $O[N(k-D)k^2]$（MLLE）正好给出解释。**L90 那句话的限定词是 "for small output dimension"，本示例 $d=2$ 正是小输出维——所以这不是措辞问题，而是与图冲突。**
4. **「Isomap uses BallTree」（丢失折叠块）与实现不符。** 1.6.1 源码用 `NearestNeighbors(algorithm=self.neighbors_algorithm)`，默认 `'auto'`；$D<15$ 时后端是 **KD 树**（与 [[sklearn-nearest-neighbors]] 里实测的 `auto` 阈值同源）。BallTree 只是可能被选中的一种。
5. **「learning rate = n / early exaggeration」（丢失折叠块）与实现不符。** 1.6.1 源码与 docstring：`'auto'` = **`max(N / early_exaggeration / 4, 50)`**。实测 $N=500$ → 50（被下限卡住）、$N=8000$ → 166.67（正好 $N/48$）。官方那句话漏了 **÷4 与下限 50**。
6. **`stress_` 的「同一个名字、三种量纲」。** L126 定义 raw stress、L128 引入 strain、丢失折叠块引入 Stress-1。实测同一数据：metric MDS 780.03、non-metric Stress-1 0.29784、non-metric 未归一化 140.16。正文没有一句提醒「这些数不能互比」。
7. **「LLE preserves distances within local neighborhoods」（L66）是简化说法。** LLE 保的是**局部重构权重**（$\sum_j w_{ij}x_j \approx x_i$），距离只是它的近似目标；丢失折叠块里的 $O[DNk^3]$「解 $k\times k$ 线性方程」才是它实际做的事。标准 LLE 在 digits 上的绿色长尾（图 6）正是「权重保住了、几何没保住」的可见证据。
8. **L162 只保留了 Barnes-Hut 四限之一。** 另外三条（仅稠密输入、angle 仅在近似法有意义、可扩展规模差两个数量级）与那句重要反例（"failing to visualize well separated … does not necessarily imply that the data cannot be correctly classified"）都在丢失折叠块里，剪藏读者会得到过度乐观的印象。

## 局限

- **本节没有任何量化评估指标**。`trustworthiness` 这个函数在 API 里存在（外部核查），本节一个字没提；「用重构误差选维数」是唯一的定量建议，而它只对 Isomap 成立（实测 1）。
- **本节不给默认值**（只有 `init='pca'` 与 `method=` 的取值），因此“用默认参数跑一下”在文档里查不到具体数字——本页的默认值全部来自外部核查。
- **本节不给 `n_neighbors` 的选取方法**，但它是全节最敏感的参数（LLE 家族的硬下界、Isomap 的连通性、Spectral 的图构造都依赖它）。
- **本节没有下游用法**：降维结果如何进模型、如何防泄漏（在 `Pipeline` 里 `fit_transform` 训练集 + `fit_transform` 测试集是两件事）、`TSNE`/`MDS`/`SpectralEmbedding` 无 `transform` 的后果，全节未提。
- **5 个示例脚本一个都没运行**（`plot_lle_digits`、`plot_compare_methods`、`plot_manifold_sphere`、`plot_swissroll`、`plot_stock_market`），图中的时间读数只能当「同机同版本的相对量级」，不能当绝对基准。
- **数字上的对比都在小样本上**：实测用的 digits 是前 400–500 个样本，不能外推到全量 1,797 或更大。

## 待解问题

1. 「dimensionality is only artificially high」（L28）**没有判定准则**。有没有可操作的本征维数估计（如 MLE、correlation dimension）可以接入？本库目前没有来源覆盖它。
2. L177 的重构误差在 **1.9.1** 上是否仍然「LLE 单调上升」？本机只有 1.6.1。若 1.8 对 `LocallyLinearEmbedding` 的 `reconstruction_error_` 语义做过修改，这条分歧就要改写。
3. MLLE/HLLE/LTSA 在 digits 上的**细丝塌缩**（图 7–9）是 $k=5$、$d=2$、未缩放造成的，还是方法固有？需要一次 $k$ 与 `reg` 的二维扫描才能定位——本次只扫了 `reg` 的一维。
4. `SpectralEmbedding.n_neighbors=None` → $n/10$（外部核查）这个默认在 $n=10^6$ 时会建 $10^5$ 邻居的图，内存与时间如何？本库**未测**，也未在任何来源里找到讨论。
5. 噪声短路的**定量阈值**：桥噪声密度到多少时 Isomap 的嵌入会真正崩塌？本次只测到桥点 400（跨簇边 24%）时分离比仍为 10.71。
6. `ClassicalMDS`（1.8+）与 `PCA` 在**非欧氏**预计算距离下的差异有多大？剪藏只给"若距离是欧氏的则等价"这一条。
7. t-SNE 的 `perplexity` 与 `early_exaggeration`（丢失折叠块有详细指导）在**小样本**（$n<100$）上的行为：实测 $n=60$ 时 `learning_rate='auto'` 被下限 50 卡住，是否意味着小样本上 `auto` 失效？
8. 降维后接下游模型的**泄漏风险**：`Pipeline` 里 `Isomap` 会在 CV 的每一折重跑最短路（成本 $\times$ 折数）。另外 `SpectralEmbedding` **没有 `transform`**，`TSNE`/`MDS` 也没有，因此它们在 `Pipeline` 里只能 `fit_transform`（测试折也当训练数据重新布局）；`Isomap`/`LLE` 的 `transform` 在训练集上与 `embedding_` 几乎一致（实测最大差 3.4e-13 / 2.3e-4），但对新样本是外推近似。本库只把这条写进实践页，未实测泄漏幅度。
9. 官方页 2.2.1 提到的 **ICA/LDA 作为线性基线**在剪藏里只有一句（L38）；LDA 的降维用法已在 [[gaussian-discriminant-analysis]] 覆盖，ICA 未入库。
10. `sphx_glr_plot_mds_001.png`（官方第 17 张图）**未入库、未查看**，内容是 MDS 的什么示例未知。

## 行号索引

| 内容 | 行号 |
| --- | --- |
| 题词（Jungle Book）、首页图 | L10–L26 |
| 定义句 | L28 |
| 2.2.1 Introduction（随机投影、PCA/ICA/LDA、示例列表） | L30–L52 |
| 2.2.2 Isomap（测地距离） | L54–L62 |
| 2.2.3 LLE（局部 PCA 比喻） | L64–L72 |
| 2.2.4 MLLE（正则化问题与多权重） | L76–L86 |
| 2.2.5 HLLE（Hessian、$O[Nd^6]$ 的替代说法、`n_neighbors` 下界） | L88–L96 |
| 2.2.6 Spectral Embedding（Laplacian Eigenmaps） | L98–L104 |
| 2.2.7 LTSA（切空间对齐） | L106–L114 |
| 2.2.8 MDS（三类、raw stress、classical/PCoA/Torgerson、strain） | L116–L145 |
| 2.2.9 t-SNE（机制、三优点、分工句、KL 非凸、四缺点） | L147–L172 |
| 2.2.10 Tips（缩放、重构误差、噪声短路、奇异矩阵） | L174–L179 |
| `[!note]` RandomTreesEmbedding | L181–L182 |
| 参考文献（14 条） | L62、L72、L86、L96、L104、L114、L142–L145、L169–L172 |

## 关联页面

- 概念：[[manifold-learning]]（本页的机制整理）、[[nearest-neighbors]]（kNN 图与距离尺度）、[[feature-selection]]（删列 vs 换坐标）、[[regularization-and-sparsity]]（LLE 的 `reg` 作为「作用在局部权重上」的正则）、[[gaussian-discriminant-analysis]]（LDA 作为线性降维的对照）
- 实践：[[manifold-learning-selection-and-tuning]]（选型、缩放、维数与种子策略、坑清单）
- 工具：[[scikit-learn]]（`manifold` 模块的估计器矩阵与默认值）
- 其他来源：[[sklearn-neural-networks]]（同为接口完整、内部机制折叠的剪藏）、[[sklearn-ensembles]]（另一个「同族多实现」的代价对比）
