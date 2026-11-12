---
title: "scikit-learn 用户指南 1.6「Nearest Neighbors」：非泛化学习、三种搜索结构与学到的度量（stable 文档剪藏）"
type: source
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - nearest-neighbors
  - instance-based
  - metric-learning
  - reference-doc
source_path: raw/articles/sklearn/1.6. Nearest Neighbors.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.6「Nearest Neighbors」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 1.6. Nearest Neighbors（scikit-learn 用户指南第 1.6 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；8 段 doctest、9 张外链插图、4 个行间公式块 |
| 原文 | <https://scikit-learn.org/stable/modules/neighbors.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/1.6. Nearest Neighbors.md`（324 行、40,126 字节、约 3,770 词；**未做任何回填**） |
| SHA-256 | `f2e24dbd4e002e3cbdf42eb163aee35ce40190525c5474b789fccf37a0172084` |
| 完整性 | **6 个折叠块缺 5 个（正文 959 词 + References 下拉块 34 词 ≈ 官方正文 20%）、[1] Goldberger 2005 一条文献、5 处页内锚点文字与 1 个引用标记**；公式有转换噪声。全文未回填（未获授权），详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.7.x – 1.9.x（无法再细分）**，依据是 1 处拼写差异，见「版本判定」 |
| 插图 | 9 张，已入库 `raw/assets/sklearn-nearest-neighbors/`（全部目视核查，读数见表） |
| 代码 | 8 段 doctest：`NearestNeighbors`（L31–L54）、`kneighbors_graph`（L56–L67）、`KDTree`（L73–L85）、两个 `valid_metrics` 列表（L89–L95）、`sorted(KDTree.valid_metrics)`（L160–L164）、`NearestCentroid`（L170–L180）、`KNeighborsTransformer`+`Isomap` 缓存（L201–L214）、NCA 流水线（L254–L270） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-nearest-neighbors]], L184`。凡超出剪藏内容的部分（API 默认值、`auto` 的选择规则、丢失折叠块里的原文）都显式标注「外部核查」或「本库」。

> [!warning] 本文件未被修改
> 与 [[sklearn-svm]]、[[sklearn-naive-bayes]] 不同，本次**没有取得所有者对 `raw/` 回填的授权**，因此剪藏保持抓取时的原始字节（sha256 `f2e24dbd…`）。丢失内容只在「转换损失」一节里按官方 1.9 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节引入了本库此前完全没有的一个模型族：**非泛化（non-generalizing）/ 基于实例（instance-based）**的方法。它不拟合参数，只保存训练集；「模型」就是训练数据本身（L12、L99）。

对本库的意义有五层：

1. **第一个没有参数的模型族**。前面入库的线性模型（[[sklearn-linear-models]]）、判别分析（[[sklearn-lda-qda]]）、朴素贝叶斯（[[sklearn-naive-bayes]]）、SVM（[[sklearn-svm]]）都有需要拟合的参数或需要选的正则强度；KNN 的「容量旋钮」是 $k$、权重与度量，**参数个数为零**。这使它在自动化搜索里成为一个特殊对象：没有 `alpha`、没有 `C`、没有 `shrinkage`，但仍然会过拟合（$k=1$）。
2. **第一个把「距离」暴露成超参的章节**。度量从固定假设（欧氏）变成可替换项（L12、L16），尺度随之从一个数值细节变成**模型定义的一部分**。这与 [[svm-selection-and-tuning]] 里「缩放是硬前提」是同一类约束，但更彻底：SVM 的核仍隐含一个距离，KNN 的距离就是模型。
3. **第一个「实现只影响速度、不影响结果」的章节**。brute / KD tree / ball tree 三种搜索结构优化的是同一个数学对象（L22、L138–L156），`algorithm` 与 `leaf_size` 原则上不改预测——这是本库目前唯一一处可以**把「实现选择」与「模型选择」彻底分开**的地方（[[svm-selection-and-tuning]] 里 `SVC` 与 `LinearSVC` 恰恰相反）。
4. **第一个零参数基线**。`NearestCentroid`（L168）只需每类一个质心，官方直接称其为「good baseline classifier」；`shrink_threshold`（L184）又让它变成可去除噪声特征的最近收缩质心（Nearest Shrunken Centroid）。
5. **第一个「学度量」的模型**。NCA（L240–L320）不学判别函数，而是学一个线性变换 $L$，把「谁是谁的邻居」这件事本身优化掉——等价于学一个 Mahalanobis 度量 $M=L^{T}L$（该说法在丢失的折叠块里，见「转换损失」）。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 监督/无监督两种用法；k 与 radius 两条路线；任意度量；非泛化 | L10–L18 |
| 1.6.1 Unsupervised | `NearestNeighbors` 的三种后端；`algorithm` 取值；**距离并列时依赖训练数据顺序**的警告 | L20–L25 |
| 1.6.1.1 Finding the Nearest Neighbors | `kneighbors` 的返回值；查询集＝训练集时自身距离为 0；`kneighbors_graph` 的块对角结构 | L27–L67 |
| 1.6.1.2 KDTree / BallTree | 直接使用两个树类；`valid_metrics` 两个列表（KD 9 个、Ball 24 个） | L69–L95 |
| 1.6.2 Classification | 多数投票；k 与 r 两种定义；$k$ 的偏置-方差取舍；`weights` 三档 | L97–L113 |
| 1.6.3 Regression | 邻域标签均值；同样的 `weights`；多输出（人脸补全） | L115–L132 |
| 1.6.4.1 Brute Force | $O[D N^{2}]$；小样本上仍然有竞争力 | L136–L138 |
| 1.6.4.2 K-D Tree | 轴向递归划分；构造快；查询 $O[\log N]$ 次距离计算；$D<20$ 才高效 | L140–L144 |
| 1.6.4.3 Ball Tree | 超球嵌套；三角不等式给上下界；高维/结构化数据可胜过 KD | L146–L156 |
| Valid Metrics（残缺） | 只剩标题与代码块，解释段落丢失 | L158–L164 |
| 1.6.5 Nearest Centroid | 每类一个质心；「无参数可选」；等方差假设的失效场景 | L166–L180 |
| 1.6.5.1 Nearest Shrunken Centroid | 除以类内方差 → 减去阈值 → 过零置零 ＝ 去掉特征；0.81→0.82 | L182–L192 |
| 1.6.6 Nearest Neighbors Transformer | 预计算邻居图（connectivity/distance 两种模式）；缓存复用；格式契约；`n_neighbors` 含自身的歧义 | L194–L236 |
| 1.6.7 NCA | 学线性变换以最大化随机 LOO-KNN 概率；分类、降维、公式、实现、复杂度 | L238–L320 |
| References | 标题 + 1 条 Wikipedia 链接（[1] Goldberger 2005 丢失） | L322–L324 |

## 关键主张

### 1. 「非泛化」不是修辞：模型就是训练集（来源论述）

来源把这一族定义为「不构造一般化的内部模型，只是保存训练实例」（L99），并在引言里用同一句话给出机制："they simply 'remember' all of its training data (possibly transformed into a fast indexing structure such as a or )"（L12）。

**本库综合**：这句话里有一个被剪藏打断的并列——原文是「Ball Tree 或 KD Tree」，两个词都丢失了（见「转换损失」B）。丢掉这一句的直接后果是：引言读起来像「KNN 没有数据结构」，而实际上**整节剩下的三分之一都在讲数据结构**（L136–L156、L194–L236）。

这条定义还带出三个可检验的推论：

- **没有训练阶段**（拟合只是建索引与存标签），所以「训练时间」不是模型复杂度的函数，而是索引结构的函数；
- **推理成本随训练集成正比**（每次查询要算 $k$ 个近邻），所以它在**小样本、低维、非规则边界**上最有优势（L14）；
- **没有参数就没有参数共享**，所以多输出回归可以按列分别做——人脸补全那个例子（L125）就是 768 维输出各自取邻域均值。

### 2. 两种邻域定义：固定数量 vs 固定体积（来源论述）

| | `KNeighbors*`（k 近邻） | `RadiusNeighbors*`（半径邻域） |
| --- | --- | --- |
| 邻域大小 | 固定 $k$（用户给的整数，L101） | 固定半径 $r$（用户给的浮点数，L101） |
| 邻域内的样本数 | 恒为 $k$ | 随局部密度变化（L105） |
| 适合的数据 | 均匀采样 | **非均匀采样**：稀疏处自动少用邻居（L105） |
| 失效场景 | 局部密度差异大时，稀疏区的 $k$ 个邻居其实很远 | 高维时半径内可能一个样本都没有；「curse of dimensionality」（L105） |
| 空邻域 | 不可能（$k\le n$ 时） | 可能；默认抛错（外部核查，见下） |

**本库综合**：这两条路线对应两种**正则化方向**。$k$ 控制「平均多少个样本」，越大越平滑（L103 的原话是 "a larger $k$ suppresses the effects of noise, but makes the classification boundaries less distinct"）；$r$ 控制「平均多大范围」，它在密集区等价于大 $k$、在稀疏区等价于小 $k$。所以 $r$ 是**局部自适应**的 $k$，代价是失去「每点计算量恒定」这个工程性质。

### 3. $k$ 是这一族唯一的容量旋钮（来源论述 + 本库判断）

L103 是本节对 $k$ 的唯一论述：最优 $k$ "highly data-dependent"；大 $k$ 抑制噪声但边界变模糊。**没有给默认值，也没有给选取方法**。

**本库判断**：这使 $k$ 与 `alpha`、`C` 处于同一个位置（容量旋钮），但不能像 `C=1/alpha` 那样跨模型换算（对比 [[sklearn-svm]], L344）。它的特殊之处是**离散且与实现耦合**：$k$ 同时决定（a）模型的平滑度、（b）每次查询的耗时、（c）`algorithm='auto'` 是否退回 brute（外部核查：`k >= n_samples // 2` 直接走 brute）。见 [[nearest-neighbors-selection-and-tuning]]。

### 4. 权重：从「投票」到「加权投票」，以及零距离的语义（来源论述 + 外部核查）

L107（分类）与 L121（回归）给出同一套三档 `weights`：

- `'uniform'`（默认）：每个邻居等权；分类＝多数投票，回归＝邻域均值（L99、L117）；
- `'distance'`：权重 $\propto 1/d$，"nearer neighbors contribute more to the fit"；
- 可调用对象：接收距离数组、返回权重数组。

**外部核查（不在剪藏内）**：`weights='distance'` 遇到**零距离**时不是 $\infty$ 而是「命中点权重 1、其余权重 0」——`sklearn/neighbors/_base.py` @ 1.9.1 的 `_get_weights` 先算 $1/d$，再把含 $\infty$ 的行整体替换成「是否等于 0」的布尔掩码（分类与回归共用该函数）。**含义**：如果查询点与某个训练点完全重合，`weights='distance'` 会退化成 1-NN，且**静默发生**。

**本库综合（假设，待验证）**：这条规则正是图 2 右下角那几个尖峰的来源——当查询点靠近某个训练点时 $1/d$ 很大，预测曲线被拉到该点标签上。剪藏没有任何一句提示这件事。

### 5. 三种搜索结构：同一数学对象，三种代价（来源论述）

| 结构 | 构造代价 | 查询代价（来源写法） | 维度的作用 | 位置 |
| --- | --- | --- | --- | --- |
| brute force | 无（只存数据） | $O[D N^{2}]$（全对距离；单次查询 $O[DN]$） | 无 | L138 |
| KD tree | 很快（只沿坐标轴切分，不算 $D$ 维距离） | 约 $O[D\log N]$；构造后每次查询 $O[\log N]$ 次距离计算 | **$D<20$ 才高效**，之后退化到近 $O[DN]$ | L142–L144 |
| ball tree | 比 KD 贵（要算质心与半径） | 用三角不等式剪枝，单次距离计算即可给出节点内所有点的上下界 | 高维、结构化的数据上可胜过 KD | L146–L156 |

三角不等式在 L150–L154，公式被转换器压成 `\left|x + y \left|\right. \leq …`——**源码已坏，语义是 $\lVert x+y\rVert \le \lVert x\rVert+\lVert y\rVert$**。

**本库综合（重要边界）**：来源给了两个**互不一致**的复杂度口径。正文 L142 说搜索总代价可降到 $O[D N\log N]$；而丢失的折叠块（官方原文）把查询代价写成 $O[D\log N]$、并明确「$D$ 大时 KD 树接近 $O[DN]$，树的开销甚至可能让它比暴力更慢」。两者不是矛盾（一个说总共、一个说每次），但**剪藏只剩前者，读起来像「树永远更快」**。工程含义：`algorithm` 不是性能 knob，而是**性能上界取决于数据结构的 knob**——它不改变结果，只改变你在什么规模上能跑完。

另有一个阈值不一致值得记录：正文说 KD 树在 $D<20$ 高效（L144），而 `algorithm='auto'` 的实现规则是 $n\_features>15$ 就走 brute（外部核查）。两者都是经验阈值，**不是同一个数**。

### 6. 距离并列：结果是训练数据顺序的函数（来源论述）

L25 的警告是本节唯一与「可复现性」直接相关的论述：

> Regarding the Nearest Neighbors algorithms, if two neighbors $k + 1$ and $k$ have identical distances but different labels, the result will depend on the ordering of the training data.

**外部核查**：并列**投票**（而非并列距离）时也有确定但同样隐蔽的规则——`weighted_mode` 按升序遍历唯一标签、只在计数**严格大于**时替换（`sklearn/utils/extmath.py` @ 1.9.1），因此平票时**最小标签获胜**；`weights='uniform'` 走 `scipy.stats.mode`，同样是「平票取最小」。

**本库综合（修正）**：两条合起来意味着 KNN 的预测对**训练集行顺序**（L25）与**标签编码**（平票规则）都敏感，但两种敏感性的性质不同：前者是距离并列、后者是计数并列。测试「结果是否稳定」的成本很低（打乱行顺序、随机重映射标签各跑一次），但文档不会提醒你去做。

### 7. `NearestCentroid`：一个质心，一条线性边界（来源论述 + 本库推导）

L168：每类用成员质心表示，效果类似 `KMeans` 的标签更新阶段；「**没有参数可选**，是个好基线」；但在**非凸类**与**类方差差异极大**时失效，因为「equal variance in all dimensions is assumed」，并指向 LDA/QDA 作为「不做此假设的更复杂方法」。

**本库综合（推导）**：到两类质心等距的集合是一个超平面，所以 `NearestCentroid` 的判定边界**是线性的**——它是「LDA 假设所有方向方差相同」的极限情形（此时类内协方差 $\Sigma=\sigma^2 I$，$\sigma^2$ 在比较中约掉）。这也解释了 L168 为什么把它与 LDA/QDA 并置：三者是同一条谱上的不同假设强度（详见 [[gaussian-discriminant-analysis]]）。

### 8. 最近收缩质心：软阈值作用在质心上（来源论述）

L184 是本节唯一出现「按特征置零」的地方，机制写得很具体：

1. 每个类、每个特征的值 **除以该特征的类内方差**；
2. 减去 `shrink_threshold`；
3. **一旦越过 0 就置 0**，于是该特征不再影响分类；
4. 用途：去掉噪声特征。

L186 给了唯一的量化结果：小阈值把准确率从 **0.81 提到 0.82**。

**外部核查**：`sklearn/neighbors/_nearest_centroid.py` @ 1.9.1 的实现是标准的软阈值算子——`|deviation| - shrink_threshold`，`clip(0, None)`，再乘回符号（L265–L272）；`deviation` 的分母含 `within_class_std_dev_ + median(within_class_std_dev_)`，即**先标准化再用中位数稳健化**。这与 Lasso 的 $S(z,\alpha)=\operatorname{sign}(z)\max(0,|z|-\alpha)$ 是**同一个算子**（对比 [[regularization-and-sparsity]] 的软阈值一节）。

**本库综合（重要差异）**：作用对象不同。Lasso 阈值作用在**回归系数**上，`shrink_threshold` 作用在**标准化后的类质心偏移**上；前者让「特征不进模型」，后者让「特征不参与最近质心比较」。但两者都产生**特征维的精确 0**，也都不能读成「该特征不重要」。

### 9. 预计算邻居图：把「邻居」从估计器里拿出来（来源论述）

L196–L236 是本节最有工程价值的一段，也是本库第一次遇到「把中间结果做成可缓存的稀疏图」的 API 设计：

- **两种图**：`mode='connectivity'` 给二值邻接（`SpectralClustering` 需要），`mode='distance'` 给距离图（`DBSCAN` 需要）（L198）。
- **三个好处**（L200–L218）：同一张图可在改其他超参时复用（配 `Pipeline(memory=…)` 缓存，doctest L201–L214）；可用 `n_jobs` 细分并行（不是所有估计器都暴露）；可替换成**近似最近邻**或特殊数据类型的自定义实现。
- **格式契约**（L220–L230，六条）：CSR（COO/CSC/LIL 也接受）；只存相对训练集的近邻，**必须包含距离 0 的点与对角线**；每行 `data` 升序（可选，乱序会被稳定排序，多一次开销）；值非负；行内 `indices` 不得重复（附 scipy/scipy#5807）；若下游用 k 近邻，则每行至少存 $k$ 个（或 $k+1$ 个）。
- **`n_neighbors` 含自身的歧义**（L232 note）：含自身 → 训练与测试时非自身邻居数不同；不含自身 → `fit(X).transform(X)` 与 `fit_transform(X)` 不一致（违反 sklearn API）。`KNeighborsTransformer` 选择**含自身**，但在 `mode='distance'` 时**多算 1 个**以兼容其他估计器；官方建议自定义实现一律多给 1 个。

**本库综合**：这段把「KNN 是一个模型」和「KNN 是一种数据变换」分开。后果是**邻居计算可以被缓存、被并行、被替换**，而模型本身不变——这是本库目前唯一一条「把最贵的一步变成可复用中间产物」的具体 API 路径。

### 10. NCA：把度量当参数来学（来源论述）

L240：NCA 是**距离度量学习**算法，目标是让最近邻分类比标准欧氏距离更准；它直接最大化「训练集上随机化 LOO-KNN 分数」的变体；副产品是一个低维线性投影（可做可视化与快速分类）。

数学形式（L288–L304）：

$$\underset{L}{\arg\max}\sum_{i=0}^{N-1}p_i,\qquad p_i=\sum_{j\in C_i}p_{ij},\qquad p_{ij}=\frac{\exp\left(-\lVert Lx_i-Lx_j\rVert^{2}\right)}{\sum_{k\ne i}\exp\left(-\lVert Lx_i-Lx_k\rVert^{2}\right)},\quad p_{ii}=0$$

其中 $C_i$ 是与 $i$ 同类的样本集合（L300）。实现用 scipy 的 L-BFGS-B，每步全梯度，"to avoid to tune the learning rate and provide stable learning"（L308）。

**外部核查（丢失的折叠块）**：官方原文紧接着写「NCA 可以看作在学习一个（平方）Mahalanobis 度量」：$\lVert L(x_i-x_j)\rVert^{2}=(x_i-x_j)^{T}M(x_i-x_j)$，$M=L^{T}L$ 是对称半正定矩阵，尺寸 $(n\_features, n\_features)$。**这句话在剪藏里完全不存在**（见「转换损失」A）。

**本库综合**：这条等式是理解 NCA 的钥匙——它把「学一个 $n_{components}\times n_{features}$ 的 $L$」翻译成「学一个 $d\times d$ 的度量矩阵 $M$」，于是 NCA 与 [[gaussian-discriminant-analysis]] 的 Mahalanobis 几何、与 [[support-vector-machines]] 的核（隐式内积）三者可以放在同一张「改几何 vs 改权重」的表里比较（见 [[nearest-neighbors]]）。

代价（L316、L320）：训练要存 $n_{samples}^{2}$ 的成对距离矩阵，每次迭代 $O(n_{components}\cdot n_{samples}\cdot\min(n_{samples},n_{features}))$；`transform` 是 $L X^{T}$，$O(n_{components}\cdot n_{features}\cdot n_{samples}^{test})$，无额外空间。

### 11. 「学到的度量」不保证更好（图读数，本库）

L240 的措辞是 "aims to improve the accuracy"，L250 是 "has been shown to work well in practice"——都是**目标**，不是保证。官方示例图给出的图内数字反而是一次反例：

- 图 8（`plot_nca_classification_001`）：`KNN (k = 1)`，准确率 **0.90**；
- 图 9（`plot_nca_classification_002`）：`NCA, KNN (k = 1)`，准确率 **0.89**。

同一份示例脚本、同一个切分（`test_size=0.7, random_state=42`）、两个特征（`X[:, [0, 2]]`，外部核查示例脚本），加 NCA 之后**掉了 1 个点**。

**本库综合**：这与剪藏 doctest 的 0.9619（L269，四特征、$k=3$、同一 random_state）并列看才有意义——doctest 没有给同设置的 KNN 对照，因此**无法从剪藏本身判断 NCA 的增益量级**。实践含义见 [[nearest-neighbors-selection-and-tuning]]：把 NCA 当作「需要验证的增强」，不是默认项。

## 版本判定

方法同前几次 ingest：把官方 1.6 / 1.7 / 1.8 / 1.9 页的 article 正文清洗成词序列后逐词比对。

| 版本 | 该节的差异 |
| --- | --- |
| 1.6 | 「the precomputed graph can be **re-used** multiple times」（1.6.6 节，对应剪藏 L203） |
| **1.7** | 改为「can be **reused** multiple times」——**唯一一处实质差异** |
| 1.8 | 与 1.7 逐词相同（差异只在页脚年份 2025→2026） |
| **1.9（当前 stable = 1.9.1）** | 与 1.8 逐词相同（差异只在导航栏与赞助声明） |

剪藏 L203 写的是 `reused`（无连字符），因此**排除 1.6**；1.7/1.8/1.9 在正文层面完全一致，**无法再细分**。

结论：`applies_to: scikit-learn 1.7.x – 1.9.x（用户指南 1.6 节无实质改动）`。证据强度**弱于** [[sklearn-svm]]（后者有整节改写作为界标），本页所有功能性结论（默认值、`auto` 规则）都不依赖这条判定——它们来自源码核查，见下节。

## 转换损失

### A. 丢失的 4 个内容折叠块（约 966 词）

根因同前几次 ingest：官方页把正文放在 `<details>` 折叠块里，转换器整块丢弃。官方 1.9 页在本节共有 6 个内容型折叠块，**剪藏只剩 1 个（Valid Metrics，且残缺）**。

| 折叠块 | 词数 | 丢失的是什么 | 剪藏残留 |
| --- | --- | --- | --- |
| **Choice of Nearest Neighbors Algorithm** | 632 | 本节真正讲「何时用哪种结构」的地方：brute 查询 $O[DN]$、ball $O[D\log N]$、KD 随 $D$ 从 $O[D\log N]$ 退化到近 $O[DN]$；小样本（$N\lesssim30$）时 $\log N$ 与 $N$ 同级、暴力更快；`leaf_size` 作为「切回暴力」的阈值；**内在维度**与**稀疏性**（数据填充参数空间的程度）两个因素 | 只剩 L22 结尾的断句「For a discussion of the strengths and weaknesses of each option, see.」与 L158 的孤立标题 |
| **Effect of leaf_size** | 200 | `leaf_size` 的三个后果：构造时间（越大越快）、查询时间（两端都差，30 是折中且为默认值）、内存（BallTree 约 $1/\text{leaf\_size}$ 倍训练集）；末尾一句「leaf_size 对暴力查询没有意义」 | 无（L77 只在代码里出现 `leaf_size=30`） |
| **Mahalanobis distance** | 40 | NCA = 学平方 Mahalanobis 度量：$\lVert L(x_i-x_j)\rVert^2=(x_i-x_j)^TM(x_i-x_j)$，$M=L^TL$ 对称半正定 | 无 |
| **Valid Metrics**（半残留） | 87 | 「可用度量见 `DistanceMetric` 与 `PAIRWISE_DISTANCE_FUNCTIONS`；`cosine` 走 `cosine_distances`；任何算法的 `valid_metric` 属性可列出」 | 标题（含残留的 `[#]` 锚点）与代码块在（L158–L164），**两段说明全丢** |
| References ×2（KD 树、ball tree 两节各一个） | 18 + 16 | Bentley 1975（*Multidimensional binary search trees used for associative searching*, CACM）；Omohundro 1989（ball tree） | 无残留——两个下拉块整块消失。L322 的 References 是**另一处**（NCA 节末尾），见下 |

合计 **959 词**（4 个正文折叠块）＋ 34 词（2 个 References 下拉块）；官方同页按同样清洗约 4,780 tokens，剪藏约 3,770 词 → **缺失约 20%**。

**另有一处独立的文献缺失**（不在折叠块里）：官方 1.9 页的末尾 References 段共两条，剪藏只剩第二条。见下方 B 节与「修正记录」。

**后果**：剪藏读起来「重数学、轻选型」。L138–L156 只讲结构原理，不讲**什么时候选哪个**；`leaf_size` 只在代码块里出现过一次且无解释。本库因此在 [[nearest-neighbors]] 与 [[nearest-neighbors-selection-and-tuning]] 里把这几段按官方原文补成「外部核查」，并逐条标注。

### B. 丢失的 5 处页内锚点文字与 1 个引用标记

同一根因（`href` 以 `#` 开头的页内锚点会丢文字）：

| 位置 | 丢失文字 | 现状 |
| --- | --- | --- |
| L10 | `classification` / `regression` | 「comes in two flavors: for data with discrete labels, and for data with continuous labels」——两个并列项都空了 |
| L12 | `Ball Tree` / `KD Tree` | 「such as a or」 |
| L22 | `Nearest Neighbor Algorithms` | 「For a discussion of the strengths and weaknesses of each option, see.」 |
| L244 | `mathematical formulation` | 「See the for more details.」 |
| L308 | `[1]` | 「follows what is explained in the original paper.」——引用标记丢失，对应的 [1]（Goldberger, Roweis, Hinton, Salakhutdinov，*Neighbourhood Components Analysis*, NIPS 17, May 2005, pp. 513–520）在 References 段也丢，只剩空标题 L322 与其后的 Wikipedia 条目 |

### C. 未修的公式与文本噪声

- 三角不等式被压成 `\left|x + y \left|\right. \leq \left|\right. x \left|\right. + \left|\right. y\right|`（L153）；NCA 的 softmax 与目标函数同样带 `\left(\right.` 噪声（L291–L303）。
- 复杂度一节把 $n_{samples}$、$n_{features}$ 拆成 `n_{s a m p l e s}` / `n_{f e a t u r e s}`（L278、L316）。
- `> [!warning]` / `> [!note]` 是 Markdown 转换产物（L24、L232），官方页对应普通 admonition。

**未修**：本次未获 `raw/` 修改授权，以上全部保留原样。

### D. 外部核查（不在剪藏内，逐条标注）

核查对象：`sklearn/neighbors/{_base,_classification,_regression,_nearest_centroid,_graph,_nca}.py` @ tag `1.9.1`、`sklearn/utils/extmath.py` @ 1.9.1、官方示例脚本 `examples/neighbors/plot_*.py` @ 1.9.1。

| 参数 | 1.9.1 默认值 | 剪藏是否提及 |
| --- | --- | --- |
| `n_neighbors`（KNN 四个估计器 + `NearestNeighbors` + `KNeighborsTransformer`） | `5` | **完全未提**（只讲 $k$ 的语义与取舍，L103） |
| `weights` | `'uniform'` | 提了（L107、L121） |
| `algorithm` | `'auto'` | 提了取值（L22），**未提选择规则** |
| `leaf_size` | `30` | 只在 doctest 里出现（L77），**未提语义与默认值** |
| `p` / `metric` | `2` / `'minkowski'` | `metric` 只泛泛提（L12、L16），`p` 未提 |
| `metric_params` / `n_jobs` | `None` / `None` | `n_jobs` 提了用途（L216） |
| `radius` | `1.0` | 未提 |
| `outlier_label` | `None`（有离群点即 `ValueError`） | **完全未提** |
| `NearestCentroid.metric` | `'euclidean'`（1.5 起只接受 `'euclidean'`/`'manhattan'`，其余直接报错） | 未提（正文只说等方差假设） |
| `NearestCentroid.shrink_threshold` | `None` | 提了语义（L184），未提默认值 |
| `NearestCentroid.priors` | `'uniform'`（**1.6 新增**） | **完全未提** |
| `KNeighborsTransformer.mode` / `n_neighbors` | `'distance'` / `5` | 提了两种 mode（L198），未提默认 mode |
| `NeighborhoodComponentsAnalysis.n_components` | `None` → 取 `n_features`（即**默认不降维**） | 只提「可用 `n_components` 设维数」（L278） |
| `.init` / `.max_iter` / `.tol` | `'auto'` / `50` / `1e-5` | 未提（正文只说用 L-BFGS-B，L308） |
| `.callback` / `.warm_start` / `.verbose` | `None` / `False` / `0` | 未提 |

三条实现层的规则（剪藏完全没有，且都「沉默改变结果或成本」）：

1. **`algorithm='auto'` 的判定链**（`_base.py`）：稀疏输入或 `metric='precomputed'` → brute（稀疏走树会发 warning）；`n_features > 15` → brute；`n_neighbors >= n_samples // 2` → brute；`metric='minkowski'` 且 `p < 1` → brute；`minkowski` 带权重 $w$ → ball_tree；否则 `metric` 在 `VALID_METRICS['kd_tree']` 里就用 kd_tree，否则 ball_tree。
2. **`p` 只在 `metric='minkowski'` 时生效**，且会被归一化：$p=1$ → `effective_metric_='manhattan'`、$p=2$ → `'euclidean'`、$p=\infty$ → `'chebyshev'`；带 $w$ 时保留通用 Minkowski。给 `metric='cosine'` 再传 `p=3` 不会有任何效果，也不报错。
3. **`weights='distance'` 的零距离规则**：见「关键主张 4」；平票取最小标签；`predict_proba` 的归一化是「加权投票占比」（`_classification.py` 中直接写着 `normalize 'votes' into real [0,1] probabilities`），不是校准概率。

图表参数（外部核查，用于解释读数）：图 1 用 `Pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))`，两格分别为 `weights='uniform' / 'distance'`；图 2 用 `KNeighborsRegressor(5)`；图 3–4 用 `NearestCentroid(shrink_threshold=None / 0.2)`；图 5–6 用 `NeighborhoodComponentsAnalysis(max_iter=30, random_state=0)`；图 7–9 用 iris 的 `X[:, [0, 2]]`（花萼长、花瓣长）、`test_size=0.7`、`k=1`、`StandardScaler`。

### E. 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 分歧

### 1. 「没有参数可选」与 16 行后的参数

- **来源论述**（L168）：「It also has no parameters to choose, making it a good baseline classifier.」
- **同一页的来源论述**（L184）：`NearestCentroid` 「has a `shrink_threshold` parameter」，并给出 0.81→0.82 的调参收益。
- **外部核查**：1.9.1 的签名是 `NearestCentroid(metric='euclidean', *, shrink_threshold=None, priors='uniform')`；`metric` 自 1.5 起被限制为两个取值，`priors` 是 1.6 新增。
- **性质**：与 [[sklearn-lda-qda]] 的「have no hyperparameters to tune」是同一类内部张力。**本库处理**：照录原话，同时把「无参数」限定为**默认构造**（不收缩、欧氏距离、均匀先验）——此时它确实是零超参基线；一旦为了去噪而设 `shrink_threshold`，就需要 CV，且（本库判断）必须先标准化，否则「除以类内方差」的尺度依赖于特征单位。

### 2. `NearestCentroid.priors` 的文档自相矛盾

- **API docstring**（1.9.1）：「`priors` … default="uniform"」紧接着写「**By default, the class proportions are inferred from the training data.**」
- **源码**（`_nearest_centroid.py`，1.9.1）：`priors == 'empirical'` 才用训练集比例；`'uniform'` 分支直接赋 $1/K$；预测时把 $2\log(\text{prior})$ 加进距离比较。
- **性质**：docstring 的两句话不可能同时成立。用户指南（剪藏）根本没提这个参数，所以剪藏不受影响。
- **处理**：在 [[scikit-learn]] 的默认值清单里标为「外部核查 + 文档内部矛盾」，不断言哪个是本意；**未做运行时验证**（不跑本机环境），只断言源码分支。

### 3. 图上的 NCA 没有赢过不打 NCA 的 KNN

- **来源论述**（L240、L250）：NCA「aims to improve the accuracy」，并「has been shown to work well in practice」。
- **同一页的官方图**（图 8 vs 图 9，本库目视核查）：KNN 0.90，NCA+KNN 0.89。
- **性质**：不是矛盾（目标是提升，不是保证），但**剪藏没有给出任何 NCA 的对照数字**，读者容易把 doctest 的 0.9619 读成「NCA 有增益」。这一页因此把「0.89 vs 0.90」写成必须并置的读数。
- **处理**：[[nearest-neighbors-selection-and-tuning]] 里把 NCA 列为「需要 A/B 验证的增强」，不列为默认步骤。

### 4. 复杂度口径：正文 vs 丢失的折叠块

见「关键主张 5」：L142 的 $O[D N\log N]$ 是**总共**，丢失折叠块里的 $O[D\log N]$ 是**每次查询**；且折叠块明确说 $D$ 大时 KD 树接近 $O[DN]$、可能比暴力更慢。**剪藏只留前者**，因此本页把「树不一定更快」标为外部核查而非来源论述。

### 5. `leaf_size` 只在代码里出现一次

L77 的 doctest 写着 `KDTree(X, leaf_size=30, metric='euclidean')`，但**整个剪藏没有一句解释它**（解释在丢失的折叠块里）。这不是矛盾，是**结构性缺口**：读者会把它当成一个普通参数抄下来，而它其实是「树在多大规模上切回暴力查询」的阈值。

## 图表核查

9 张官方示例图于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-nearest-neighbors/`，**逐张目视核查**。

| 图 | 文件 | 尺寸 | SHA-256（前 16） | 对应正文 |
| --- | --- | --- | --- | --- |
| 图 1 | `sphx_glr_plot_classification_001.png` | 1200×500 | `f37f3e9f146fefca` | L109 |
| 图 2 | `sphx_glr_plot_regression_001.png` | 640×480 | `7c8a248b0715cd8e` | L123 |
| 图 3 | `sphx_glr_plot_multioutput_face_completion_001.png` | 1000×1130 | `33839c9d4aecf7e8` | L127 |
| 图 4 | `sphx_glr_plot_nearest_centroid_001.png` | 640×480 | `408a4915d3d1c3e4` | L188（左） |
| 图 5 | `sphx_glr_plot_nearest_centroid_002.png` | 640×480 | `94d153b2316e8439` | L188（右） |
| 图 6 | `sphx_glr_plot_nca_illustration_001.png` | 640×480 | `3d00566d6c7d4a47` | L242（左） |
| 图 7 | `sphx_glr_plot_nca_illustration_002.png` | 640×480 | `8b1c0aba04a434c4` | L242（右） |
| 图 8 | `sphx_glr_plot_nca_classification_001.png` | 640×480 | `7f567d13e30e2730` | L272（左） |
| 图 9 | `sphx_glr_plot_nca_classification_002.png` | 640×480 | `f8e03f6d504b612a` | L272（右） |

**图 1（iris 两特征，$k=11$，uniform vs distance）**：两格标题给出「k=11」；横轴 sepal length、纵轴 sepal width，setosa 蓝色在左下、versicolor 黄色成楔形居中、virginica 红色在右上。`uniform` 格的三个区域边界是**折线**（少量长直段）；`distance` 格的黄区边界出现明显的**锯齿与小块凹陷**（尤其 versicolor 点密集的 $x\approx5.5$–$6.5$、$y\approx2.5$–$3.0$ 一带），红黄交界也变得更碎。读数印证 L107「近邻贡献更大」＝边界更贴合局部点，代价是不平滑。

**图 2（一维回归，$k=5$）**：上格 `uniform` 的预测是**分段常数阶梯**（每个邻域一个水平段），下格 `distance` 的曲线穿过数据点附近、并在 $x\approx0.75$（峰值 $\approx1.2$）与 $x\approx4.6$（谷值 $\approx-0.5$）出现**尖峰**。这直接可视化了 $1/d$ 权重的两个后果：更贴合局部，也会被单点拽走——正是「关键主张 4」里那条零距离规则的近邻版本。

**图 3（人脸补全，多输出回归）**：5 行 × 5 列（真值、Extra trees、K-nn、Linear regression、Ridge）。K-nn 列的下半脸明显更平滑、更「平均脸」，第 1、3、5 行还能看出**不属于该个体的轮廓残影**；Linear regression 列出现明显条带/振铃噪声；Extra trees 与 Ridge 更干净。读数支持「多输出 KNN 是逐像素邻域平均」（L125），但没有量化。

**图 4 / 图 5（NearestCentroid，iris 两特征）**：两图标题的准确率分别是 **0.81**（`shrink_threshold=None`）与 **0.82**（`0.2`），与 L186 一致。无收缩时三个区域是多边形（Voronoi 型）划分，黄区成窄楔形；加阈值后边界只**轻微移动**（三区交汇点与红蓝竖直边界略有位移），看不出「去掉某个特征」的剧烈变化——与本库预期一致：阈值 0.2 相对于标准化偏移很小，主要效果是把接近 0 的偏移压成 0。

**图 6 / 图 7（NCA 示意，9 个点）**：图 6「Original points」中样本 3（红）位于中心，连线粗细 ∝ 概率：**最粗的是两条绿色连线（通往 1 与 7，异类）**，通往蓝点 4/5/6 的连线较细，通往同类 2/8 的红线也细——即原空间里「最近的邻居是错的」。图 7「NCA embedding」中三类明显分开（蓝点 4/5/6 聚在左、绿点 0/1/7 散在右、红点 2/3/8 聚在中），样本 3 只剩一条**很粗的红线**通往 2，其余连线基本消失。这与 L244 的叙述逐字吻合，是本节唯一「图上能直接看见目标函数」的图。

**图 8 / 图 9（NCA 分类，iris 两特征，$k=1$）**：见「分歧 3」——0.90 vs 0.89。两图边界形状相近（红区在上、黄区斜带、蓝区在下），NCA 版的黄带在 $x\approx6.5$、$y\approx5.5$ 附近多出一小块红色侵入。**两图没有坐标轴标签**（只有刻度），特征由示例脚本决定（外部核查：`X[:, [0, 2]]`）。

## 局限

- **剪藏缺失约 20% 正文**，缺的正好是「选型」部分（4 个折叠块）。本页对「何时用 KD / ball / brute」「`leaf_size` 怎么设」的所有说法都来自**外部核查**（官方 1.9 页原文 + 1.9.1 源码），不是剪藏论述。
- **没有缩放警告**。整节没有一句提到特征缩放（对比 [[sklearn-svm]], L203 的 "highly recommended to scale"），而**官方示例自己给 KNN 套了 `StandardScaler`**（图 1、图 8/9 的 pipeline）。这是本节最值得记录的「文档缺口」，但缺口不等于反例：不能说「文档认为不需要缩放」。
- 本节**不覆盖**缺失值（`nan_euclidean`）、`sklearn.metrics.pairwise` 的距离函数细节、`sklearn.cluster.DBSCAN` / `SpectralClustering` / `sklearn.manifold` 的实际用法——它们只是被点名为下游消费者（L10、L18、L198）。
- 本页**未读** [1]（Goldberger et al. 2005，NCA 原文）、Bentley 1975、Omohundro 1989。因此「NCA 最大化 LOO-KNN 的随机变体」在本库中仍是**来源转述**。
- 运行时行为只保留 1.9.1 源码与 docstring 层面的外部核查，不包含历史本机实验。`outlier_label` 等分支需在 `uv` 目标版本环境中另行验证。
- 图 3 的「哪个更干净」是**目视定性**，没有 PSNR 之类的量化读数。

## 待解问题

1. **NCA 的增益到底有多大？** 图 8/9 是 $k=1$、两个特征、单次切分的反例（0.89 < 0.90）；doctest（L269）是四特征、$k=3$ 的 0.9619 但**没有对照**。需要同一脚本里跑 KNN 与 NCA+KNN 的多折对比才能给量级。
2. **`weights='distance'` 在回归路径上如何处理零距离？** 1.9.1 源码显示分类与回归共用 `_get_weights` 的零距离掩码；具体运行时表现仍需在 `uv` 目标版本环境中复核。
3. **KD 树的维度拐点**：正文说 $D<20$（L144），`auto` 规则用 $D>15$。**修正**：`auto` 的 15 不是「外部核查」——它写在官方 1.9 页的 *Choice of Nearest Neighbors Algorithm* 折叠块里（剪藏丢失），原文为「$D > 15$」并注明假设「when $D>15$, the intrinsic dimensionality of the data is generally too high for tree-based methods」。两个阈值的来源仍不清楚。
4. **`leaf_size` 的最优值**：丢失的折叠块只说 30 是「good compromise」、并指出两端都差。在 $N=10^5$ 量级上这个默认值是否仍然合理，无数据。
5. **ball tree 相对 brute 的收益条件**：折叠块称「highly structured data, even in very high dimensions」，但没有给量化的维度/稀疏度门槛。
6. **`NearestCentroid.priors` 的实际用途**：1.6 新增的参数，用户指南完全不提；$2\log(\text{prior})$ 的修正与 LDA 的 `priors` 是否同一口径、默认 `'uniform'` 是不是有意的（与 docstring 矛盾），都需要 changelog 或 issue 佐证。
7. **预计算图的 `k+1` 兼容规则**：L232 只说「for compatibility reasons with other estimators」，没有说明是哪几个估计器需要多一个邻居。若要写自定义 ANN 变换器，这条必须知道。
8. **NCA 的 `max_iter=50` 是否足够**：示例图用 `max_iter=30`，doctest 用默认值且收敛到 0.9619。L-BFGS-B 的收敛判据（`tol=1e-5`）在小样本上是否总是达到，未验证。
9. **平票取最小标签是否在 `weights` 为 callable 时仍成立**：`weighted_mode` 的相等判断与自定义权重路径需在目标版本源码和运行环境中共同核对。

## 行号索引

- L10 两种监督用法；L12 原理（k / radius / 任意度量 / 非泛化）；L14 非参数与不规则边界；L16 dense/sparse 与稀疏只支持 Minkowski；L18 KDE
- L20–L25 `NearestNeighbors` 三后端、`auto` 取值、**并列警告**
- L27–L67 `kneighbors` doctest（自身距离 0）；`kneighbors_graph` 块对角
- L69–L95 `KDTree` 直接使用；`KDTree.valid_metrics`（9 个）/ `BallTree.valid_metrics`（24 个，含 `pyfunc`）
- L97–L113 分类：多数投票；k vs r；$k$ 的取舍；`weights` 三档
- L115–L132 回归：邻域均值；同样的 `weights`；多输出与人脸补全
- L136–L138 brute $O[D N^{2}]$；L140–L144 KD 树（构造、$O[\log N]$、$D<20$）
- L146–L156 ball 树；三角不等式（L153，噪声）；质心-半径上下界
- L158–L164 残缺的 Valid Metrics 折叠块
- L166–L180 `NearestCentroid`：质心、KMeans 类比、「无参数」、等方差失效；doctest
- L182–L192 最近收缩质心：除方差 → 减阈值 → 过零置零；0.81→0.82
- L194–L236 预计算图：两种 mode、三个好处、缓存 doctest、格式六条、`n_neighbors` 歧义 note
- L238–L244 NCA 定义与示意图说明；**L244 断句「See the for more details.」**
- L246–L274 分类用法、doctest 0.9619、图 8/9
- L276–L284 降维与 `n_components`
- L286–L304 目标函数、$p_i$、softmax、$p_{ii}=0$
- L306–L310 L-BFGS-B、全梯度；L312–L320 复杂度（$N^2$ 内存）
- L322–L324 References：标题 + Wikipedia 条目（L324）；[1] Goldberger et al. 2005 的条目与 L308 的引用标记均丢失

## 关联页面

- 概念：[[nearest-neighbors]]（本来源的主概念页）、[[regularization-and-sparsity]]（原型收缩＝软阈值的第五个面）、[[gaussian-discriminant-analysis]]（`NearestCentroid` 与 LDA 的等方差极限）
- 实践：[[nearest-neighbors-selection-and-tuning]]（本来源的实践页）、[[linear-models-as-baselines]]（零参数基线）、[[svm-selection-and-tuning]]（同一类「缩放是硬前提」）
- 工具：[[scikit-learn]]（`neighbors` 估计器矩阵与默认值）
- 同批来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]
