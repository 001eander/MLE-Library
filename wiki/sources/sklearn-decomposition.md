---
title: "scikit-learn 用户指南 2.5「Decomposing signals in components」：PCA 家族、字典学习、FA、ICA、NMF 与 LDA（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - decomposition
  - dimensionality-reduction
  - pca
  - nmf
  - ica
  - topic-model
  - reference-doc
source_path: raw/articles/sklearn/2.5. Decomposing signals in components (matrix factorization problems).md
source_kind: article
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档正文；行为在本机 1.6.1 复核）
---

# scikit-learn 用户指南 2.5「Decomposing signals in components」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 2.5. Decomposing signals in components (matrix factorization problems)（scikit-learn 用户指南第 2.5 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南 HTML→Markdown 剪藏；**1 段代码块（8 行 doctest）、17 个行间公式块（34 行 `$$`）、3 个 `[!note]` 提示、23 张唯一图（31 次引用）、0 张表** |
| 原文 | <https://scikit-learn.org/stable/modules/decomposition.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/2.5. Decomposing signals in components (matrix factorization problems).md`（**463 逻辑行**、56,104 字节、`wc -w` 5,581 词；`wc -l` 报 462 是因为**末行无换行**；**未做任何回填**） |
| SHA-256 | `7a95373ad08211aa2b8f03bced7a0847f44fdf5a772f94a204a588cd4bed338f` |
| 完整性 | **正文（散文、公式、示例列表、doctest）完整**；缺 **4 个 `<details>` 折叠块（833 词）**、**约 14 条参考文献条目**、**4 个空 `References` 标题**；另有 **6 个脚注把 NMF 小节重复了 886 词**（Web Clipper 伪影）。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.1**（1.8→1.9 的关键词改动已出现在剪藏里；见「版本判定」） |
| 插图 | **23 张唯一图，全部下载入库** `raw/assets/sklearn-decomposition/`，其中 **15 张已目视核查**（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-decomposition]], L16`。凡超出剪藏内容的部分（API 默认值、官方折叠块原文、运行时行为）都显式标注「外部核查」「官方 1.9.1 页」或「本机实测」。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `7a95373a…`）。丢失的 4 个折叠块与参考文献只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库**第三个「无监督产物」类目**：1.13 节是删列（保留原列），2.2 节是换坐标（只有坐标），2.3 节是分组（输出标签）；本节是**矩阵分解**——它既换坐标（`transform` 输出编码），又同时输出**一套可解释的成分**（`components_`）。这五个层面的意义：

1. **第一次把「成分」当成一等输出。** 前三个无监督类目里，模型参数要么是列掩码、要么是一个嵌入、要么是簇中心；这里的 `components_` 是一组**与原特征同维的向量**，可以单独拿出来看、可以反投影、可以解释为「脸的一部分」「一个主题」「一个独立源」。这使「降维」第一次同时成为「解释」。
2. **第一次出现「同一个 $X\approx UV$ 骨架、靠约束区分模型」的家族。** PCA 约束 $U$ 正交、SparsePCA 加 $ℓ_1$、NMF 要求 $U,V\ge0$、ICA 要求成分最大独立、FA 约束噪声协方差、LDA 给主题加 Dirichlet 先验（L16、L81、L170、L214、L268、L286、L379）。**约束就是模型**；方法名只是约束的缩写（本库综合）。
3. **第一份「接口完整、机制全在折叠块里」的 sklearn 剪藏。** 所有类名、参数名、示例、doctest 都在；丢的 4 块恰好是「KernelPCA 选哪个特征求解器」「TruncatedSVD/LSA 的数学」「NMF 两个求解器的差异」「LDA 的生成过程」。因此本页可以逐条比对正文与实现，但**不能**从剪藏引用这些机制。
4. **第一次把「非唯一性」写成工程风险。** PCA 的成分只到符号/旋转；FA 的成分不保证正交（L254）；NMF 是非凸的、不同初始化会落到不同局部极小（官方折叠块）；LDA 的在线 VB 也只保证局部最优（L396）。这直接决定了「换一个 `random_state` 结果会变」不是 bug。
5. **它把前面几页接了起来**：PCA vs 有监督 LDA（名称冲突，[[gaussian-discriminant-analysis]]）、NMF vs 聚类（软编码 vs 硬标签，[[clustering]]）、TruncatedSVD vs 特征选择（投影 vs 删列，[[feature-selection]]）、KernelPCA vs 流形学习（核方法 vs 图方法，[[manifold-learning]]）。

## 文档结构

| 章节 | 解决什么问题 | 约束/目标 | 行号 |
| --- | --- | --- | --- |
| 2.5.1 PCA | 正交、最大方差、可白化、可增量、可随机化 | 方差 / 重构 | L10–L108 |
| 2.5.1.1 Exact PCA | 中心化 + SVD；概率解释与 `score` | 方差 | L12–L30 |
| 2.5.1.2 Incremental PCA | 外存/小批量，内存随 batch 不随 $N$ | 方差 | L32–L49 |
| 2.5.1.3 Randomized SVD | 只要前 $k$ 个成分时的近似解法与复杂度 | 方差 | L51–L77 |
| 2.5.1.4 Sparse PCA | 用 $ℓ_1$ 把成分变稀疏；MiniBatch 沿特征方向在线 | 重构 + 稀疏 | L79–L108 |
| 2.5.2 Kernel PCA | 用核做非线性降维；成分数受样本数限制 | 核空间方差 | L110–L130 |
| 2.5.2.1 Exact kPCA | `transform` 与 `inverse_transform`；反投影是近似 | 核空间方差 | L112–L126 |
| 2.5.2.2 Choice of solver | 大样本时用近似特征求解器 | 核空间方差 | L128–L130 |
| 2.5.3 Truncated SVD / LSA | 不中心化的 SVD；中心化后等价 PCA | 重构 | L132–L144 |
| 2.5.4 Dictionary Learning | 固定字典的稀疏编码；学习过完备字典 | 重构 + 稀疏 | L146–L210 |
| 2.5.4.1 Sparse coding | `SparseCoder`、五种 `transform` 方法、`split_code` | 稀疏编码 | L148–L166 |
| 2.5.4.2 Generic dictionary | 交替更新码与字典；正性约束 | 重构 + 稀疏 | L168–L192 |
| 2.5.4.3 Mini-batch dictionary | `partial_fit`、无停止条件 | 重构 + 稀疏 | L194–L210 |
| 2.5.5 Factor Analysis | 潜变量高斯模型；PPCA 与 FA 的噪声协方差差异 | 似然 | L212–L264 |
| 2.5.6 ICA | 最大独立成分；必须白化；盲源分离 | 非高斯性/独立 | L266–L282 |
| 2.5.7 NMF | 非负加法分解；Frobenius 与 beta-divergence；初始化 | 重构 + 非负 | L284–L375 |
| 2.5.7.1 Frobenius NMF | 目标函数、初始化、L1/L2 先验 | 重构 + 非负 | L286–L316 |
| 2.5.7.2 beta-divergence | KL/IS 与 $\beta$ 家族；`fit_transform`/`components_` 语义 | 重构 + 非负 | L318–L363 |
| 2.5.7.3 Mini-batch NMF | `partial_fit`、`forget_factor` | 重构 + 非负 | L365–L375 |
| 2.5.8 LDA | 三层生成模型、在线 VB、主题-词/文档-主题矩阵 | 主题似然 | L377–L407 |
| See also | NCA 作为另一种降维 | — | L414 |
| （脚注 1–6） | **重复** NMF 小节正文 | — | L416–L463 |

## 关键主张（按剪藏顺序）

### A. PCA 家族（L16–L67）

- **PCA 只中心化、不缩放**（L16）。这是全节最重要的一条预处理契约：`PCA` 的 `fit` 会减去每列均值，但不会除以标准差。**是否缩放由调用者决定**，而缩放会改变主成分（本机实测 iris 上 2 维 PCA + logistic 的 5 折 CV：原始 0.96、标准化后 0.9133——不是「一定更好」，而是「换了一个模型」）。
- **`whiten=True` 把各成分缩到单位方差**（L16），用于下游模型假设信号各向同性的场合（RBF SVM、KMeans）。本机实测 iris：白化后两轴方差 0.9933 / 0.9933；不白化是 4.2001 / 0.2411。**白化不是「更好的 PCA」，是把协方差改成单位阵**。
- **PCA 有概率解释，`score` 可用于交叉验证**（L20）。剪藏只给这一句，没有给似然公式；因此「PCA 的 `score` 一定比重建误差好」不能从本页推出。
- **`IncrementalPCA` 用小批量近似 PCA，并支持外存**（L34–L40）：`partial_fit` 逐块喂数据，或用 `numpy.memmap`。它只存成分与噪声方差的估计，因此内存取决于**每批样本数**，不是总样本数（L40）。来源说结果「almost exactly match」PCA；本机实测 batch=150（全量）时与 PCA 逐位相同，batch=10/30 时坐标最大差 0.01271 / 0.00970。
- **randomized SVD 的代价**（L63、L65）：时间复杂度 $O(n_{max}^2\cdot n_{components})$，精确法是 $O(n_{max}^2\cdot n_{min})$；内存是 $2\cdot n_{max}\cdot n_{components}$，精确法是 $n_{max}\cdot n_{min}$。**这是在「成分数远小于 $\min(n,p)$」时才有意义**。
- **randomized 的 `inverse_transform` 不是精确逆**（L67），即使 `whiten=False`。本机 digits 上 10 维：randomized 与 full 的重建误差同为 751.787，但两者的反投影矩阵差 0.567——**差异小不等于可忽略，做重构任务时要显式比较**。
- **SparsePCA 的目标**（L81–L99）：在重构误差上加 $ℓ_{1,1}$ 惩罚，`alpha` 控制稀疏度；成分被约束为单位球内的向量。来源说「larger values shrink many coefficients to zero」。本机 iris（3 成分）：`alpha` 0.01/0.1/1/10 → 零元素 3/3/8/10（共 12 个），重建误差 1.903/1.911/2.4/10.721——**稀疏的代价是重构**，且 `alpha=10` 已经明显过罚。
- **`MiniBatchSparsePCA` 不实现 `partial_fit`**（L102）：它「在精神上」是在线算法，但方向是**特征**而不是样本，因此不能按样本增量喂。本机 `hasattr(...,'partial_fit')` 为 False。这是本库第一处「MiniBatch 名字不等于可在线学习」的反例。

### B. Kernel PCA（L114–L130）

- **KernelPCA = 核 + PCA**（L114）：用核把样本映到高维（隐式）空间再做线性 PCA，因此能做非线性降维、去噪、压缩与结构化预测；支持 `transform` 与 `inverse_transform`。
- **反投影是近似**（L118）：`KernelPCA.inverse_transform` 用 kernel ridge 学「PCA 基 → 原空间」的映射，不是精确逆。本机 iris（RBF, 4 成分, `fit_inverse_transform=True`）：KernelPCA 重建误差 6.9775，线性 PCA 3.8993——**核 PCA 的「重构」不能与线性 PCA 的重构误差直接比**。
- **成分数上界从特征数换成样本数**（L130）：$n_{components}\le n_{samples}$。当 $n$ 很大而只要前几十个成分时，完整 kPCA 是浪费；中心化 Gram 矩阵的有效秩远小于其尺寸，因此近似特征求解器能以很小精度损失换速度。
- 剪藏**没有**说 `eigen_solver` 的三个取值与 arpack 的自动启用条件；这些在丢失的 *Eigensolvers* 折叠块里（官方 1.9.1 页）。因此「`eigen_solver='auto'` 在 $n_{components}<10$ 且 $n_{samples}>200$ 时用 arpack」**不能引剪藏**，只能引官方页。

### C. TruncatedSVD 与 LSA（L134–L136）

- **`TruncatedSVD` 只算前 $k$ 个奇异值**（L134），不要求 $X$ 中心化；**若先减去每列均值，则截断 SVD 等价于 PCA**（L136）。本机实测：对已中心化的 iris，`TruncatedSVD(2)` 与 `PCA(2)` 的投影最大差 0.0；未中心化时前两个「解释方差比」是 0.5288 / 0.4485，与 PCA 的 0.9246 / 0.0531 完全不同。
- 这条等价关系是**文本 LSA 与 PCA 的桥**：词-文档矩阵通常稀疏、非负、不中心化，直接做 TruncatedSVD 就是 LSA。
- 剪藏**丢了** LSA 的数学（$X\approx U_k\Sigma_k V_k^\top$、训练集变换 $U_k\Sigma_k$、测试集乘 $V_k$）与「LSA/LSI 名称」的折叠块；这些只在官方 1.9.1 页。

### D. 字典学习与稀疏编码（L150–L206）

- **`SparseCoder` 用固定字典做稀疏编码，不学习字典**（L150）。来源说它「does not implement a `fit` method」；1.6.1 实际有一个**空操作 `fit`**（为 Pipeline API 存在，返回 self、不改变状态）——见「分歧」1。读文档时应把「没有 `fit`」理解为「没有学习」，不是「对象没有这个方法」。
- **五种 `transform` 方法**（L152–L160）：OMP、LARS、Lasso-LARS、坐标下降 Lasso、阈值法。来源明确：**阈值法最快但重构不准**；图像重建用 OMP 最准、无偏（L158）。本机验证：`SparseCoder(transform_algorithm='omp', transform_n_nonzero_coefs=3)` 每行恰好 3 个非零；`threshold` 的零比例取决于 `transform_alpha`，默认 `None` 时不会自动稀疏。
- **`split_code`（API 里叫 `split_sign`）把编码长度翻倍**（L160–L162）：正部在前、负部的绝对值在后，因此结果非负，长度 `2 * n_components`。本机 `DictionaryLearning(split_sign=True)`：普通编码 `(3, 8)` → split 编码 `(3, 16)`，且全部非负。**这是给下游线性模型「分别给正负载荷不同权重」用的**，不是更稀疏的编码。
- **`DictionaryLearning` 学一个（通常过完备的）字典**（L170），优化目标是重构 + 对码的 $ℓ_{1,1}$ 惩罚，字典原子约束在单位球内；求解方式是**交替**：固定字典解多个 Lasso 得码，再固定码更新字典（L174）。这解释了它为什么慢、为什么非凸。
- **字典/码都可以加正性约束**（L184）；源图用红/蓝/白表示负/正/零。
- **`MiniBatchDictionaryLearning` 有 `partial_fit`**（L196–L198），但**默认没有停止条件**（L198）：轮数由 `max_iter` 决定，早停只看 `max_no_improvement`。这是它与 `MiniBatchNMF` 的共同点，也是「在线」不等于「会自动停」的例子。

### E. Factor Analysis（L214–L260）

- **潜变量模型**（L214–L220）：$x_i = W h_i + \mu + \epsilon$，$h_i\sim\mathcal N(0,I)$，$\epsilon\sim\mathcal N(0,\Psi)$；边缘分布 $x\sim\mathcal N(\mu, WW^\top+\Psi)$。这就是「把 $X$ 分解成低秩 + 噪声」的概率版本。
- **$\Psi=\sigma^2 I$ → 概率 PCA；$\Psi=\operatorname{diag}(\psi_1,\dots,\psi_n)$ → Factor Analysis**（L246–L250）。两者的差别只有一句：**噪声是否各向同性**。FA 的 $W$ 叫因子载荷矩阵。
- **FA 的主要优势是异方差噪声**（L258）：它能为每个方向单独建模噪声方差，因此在异方差数据上模型选择优于 PPCA。本机 iris：FA 的 `noise_variance_` 是 `[0.016, 0.098, 0.004, 0.041]`（4 个不同值），PCA 只有一个各向同性噪声尺度。
- **FA 的成分不保证正交**（L254）：来源明确说「one can not make any general statements about these components (e.g. whether they are orthogonal)」。本机 FA（3 成分）的载荷相关系数最大 0.555；PCA 的成分则两两正交。
- **旋转用于解释**（L260）：`rotation='varimax'` 最大化载荷平方的方差，倾向于「简单结构」（每个因子只受少数特征影响）。但**旋转不改变拟合优度，也不保证在你的数据上更稀疏**；本机 iris 上 varimax 的「$|loading|<0.1$」个数 4，不旋转是 3，最大非对角相关反而从 0.7399 升到 0.8904（见「分歧」6）。

### F. ICA（L268–L282）

- **ICA 把多变量信号分成最大独立的加性子成分**（L268），用 FastICA 实现。它**通常不是降维**，而是分离叠加信号（盲源分离）。
- **因为 ICA 模型没有噪声项，必须白化**（L268）：可以内部用 `whiten` 参数，或外部先用 PCA 白化。**白化不是可选预处理，是模型正确性的条件**。
- 本机 1.6.1 有两个容易踩的行为：`whiten='unit-variance'` 输出 2 维、各轴方差 1；`whiten='arbitrary-variance'` 输出 2 维但方差 0.007；`whiten=False` 时**输出维数变成 4**、`components_` 是 `(4,4)`，尽管 `n_components` 属性仍是 2——**`whiten=False` 会忽略 `n_components`**。另外 `whiten=True`（布尔）在 1.6.1 直接 `InvalidParameterError`，只接受两个字符串或 `False`（1.3 起默认 `'unit-variance'`；外部核查）。

### G. NMF（L286–L373）

- **NMF 假设数据与成分都非负**（L286），把 $X$ 分解成 $W,H\ge0$，最小化 $\frac12\lVert X-WH\rVert_F^2$；与 PCA 的差别是**表示是加法的**（叠加成分，不做减法），因此适合图像与文本的「零件」表示（L290）。
- **初始化决定结果**（L300–L306）：`init` 可为 `'nndsvd'`/`'nndsvda'`/`'nndsvdar'`/`'random'`。来源明确：基本 NNDSVD 更适合稀疏分解，稠密数据推荐 NNDSVDa/NNDSVDar；**`solver='mu'` 无法更新初始化中的零**，与基本 NNDSVD 的大量零一起用会更差（L304）。本机验证（4 成分、`solver='mu'`）：`nndsvd` 留 5 个零、重构误差 4.4395；`nndsvda` 0 个零、2.3878；`nndsvdar` 0 个零、1.4714；`random` 0 个零、2.0252。**同一模型、同一数据，初始化能把误差差近 3 倍。**
- **L1/L2 先验**（L306–L316）：`alpha_W`/`alpha_H` 控制强度，`l1_ratio`（$\rho$）控制 L1/L2 混合；先验按 `n_features`（对 W）与 `n_samples`（对 H）缩放，以保持与数据项的相对量级。剪藏给了完整目标函数，但公式被渲染成 `n _ f e a t u r e s` 这类拆字。
- **beta-divergence**（L320–L347）：Frobenius（$\beta=2$）、KL（$\beta=1$）、IS（$\beta=0$）是同一族；$\beta\in(0,1)$ 的定义要连续延拓。本机同一份非负 iris（加 0.01）上：$\beta=2$ 误差 3.9579、$\beta=1$ 2.475、$\beta=1.5$ 3.091、$\beta=0$ 1.6254——**不同 $\beta$ 的 `reconstruction_err_` 不可直接比较**，因为量纲/尺度不同。
- **`fit_transform` 返回 W，`components_` 是 H，`transform` 用 H 分解新 X**（L349–L358）。本机实测 `transform(X)` 与 `fit_transform(X)` 并不逐位相同：`init='random'` 最大差 3.72e-4，`init='nndsvda'` 7.68e-4。**不要把两者当成同一个矩阵**；需要训练集编码时用 `fit_transform` 的返回值，或固定 `H` 后另行保存。
- **`MiniBatchNMF` 有 `partial_fit`**（L367–L373），`batch_size` 控制批大小，`forget_factor` 让旧批次的权重衰减；本机连续 5 次 `partial_fit` 后 `components_` 为 `(3,4)`、`transform` 可用。

### H. LDA 主题模型（L379–L407）

- **LDA 是三层生成模型**（L379–L388）：语料有 $D$ 篇文档、每篇 $N_d$ 个词、共 $K$ 个主题；$\alpha$ 是文档-主题 Dirichlet 先验，$\eta$ 是主题-词 Dirichlet 先验。图里的阴影节点是观测到的词，未阴影是潜变量。
- **`LatentDirichletAllocation` 用在线变分贝叶斯**（L392–L396），支持 batch 与 online 两种更新；online 保证收敛到局部最优，但**最优点的质量与收敛速度依赖 mini-batch 大小与学习率设置**（L396）。
- **矩阵方向**（L398）：输入是「文档-词」矩阵；分解得到「主题-词」矩阵（存 `components_`）与「文档-主题」矩阵（由 `transform` 计算）。本机：2 主题、4 词表 → `components_` `(2,4)`、`transform` `(6,2)`、每行和为 1、perplexity 5.3101。
- **`partial_fit` 可用于顺序到达的数据**（L400）。
- LDA 的生成过程细节（每篇文档的主题比例、每个词的主题分配、ELBO）在丢失的 *Details on modeling text corpora* 折叠块里；剪藏只有图与记号说明。

## 版本判定

**结论：1.9.1，证据强。** 判定链：

1. **1.8→1.9 的唯一正文差异**是 L69 的示例名：`Faces recognition example using eigenfaces and SVMs` → `... and kernel approximation`。剪藏 L69 是 **kernel approximation**，因此 ≥1.9。
2. **1.7→1.8 的差异**包括 L204 的逗号（`a dictionary, learned from ... face, looks like`）与 LDA 记号（`d \in D`、`N_d`、`z_{dn}`、`w_{dn}`）。剪藏 L204、L388 均已是 1.8 形式。
3. **1.6→1.7 的差异**包括 `preserve`→`preserving`（L55）、`yield`→`yields`（L158）、`are`→`is`、`dataset`→`datasets`（L34、L196、L379）。剪藏均已是 1.7 形式。
4. 抓取日 `/stable/modules/decomposition.html` 与 `/1.9/modules/decomposition.html` **字节级相同**（133,829 B），页标题 `scikit-learn 1.9.1 documentation`。
5. 1.9.1 的 4 个 `<details>` 块与剪藏的缺失块一一对应；官方页共 23 张唯一图，剪藏也是 23 张，**图集一致**。

本节的功能性结论（除默认值外）不依赖 1.9.1 与 1.8 的区分。

## 转换损失

这是本库**结构损失最小、机制损失最大**的一份剪藏之一：散文、公式、示例、doctest 都在，丢的恰好是「为什么这样选」的四块。

| 丢失项 | 词数 | 内容 | 后果 |
| --- | --- | --- | --- |
| `Eigensolvers`（KernelPCA） | 224 | `eigen_solver='randomized'` 的复杂度 $O(n^2 n_c)$ vs 精确 $O(n^3)$、内存 $2nn_c$ vs $n^2$；`arpack` 的适用条件（$n_c<10$ 且 $n>200$ 时默认启用）与三个求解器的参考文献 | 剪藏 L130 只说「近似求解器可提速」，**没有任何复杂度或自动选择规则**；本页只能引官方页 |
| `About truncated SVD and LSA` | 279 | LSA 的定义、同义词/多义词、$X\approx U_k\Sigma_k V_k^\top$、训练集 $U_k\Sigma_k$、测试集 $XV_k$、轴方向的约定 | 剪藏只有 L134–L136 两句；**LSA 的数学与「为什么文本要用 TruncatedSVD」全丢** |
| `NMF implemented solvers` | 125 | `cd` 与 `mu` 两个求解器；`mu` 支持所有 beta-divergence、$\beta\in(1,2)$ 更快、$\beta\le0$ 时输入不能含零；`cd` 只支持 Frobenius；非凸导致不同局部极小 | 剪藏 L300–L304 只讲初始化与零锁定，**不讲求解器能力边界**；`beta_loss` 与 `solver` 的合法组合只能查官方页 |
| `Details on modeling text corpora` | 205 | LDA 的逐词生成过程、后验不可解、变分分布 $q(z,\theta,\beta)$、ELBO | 剪藏 L379–L396 只有图与「online VB」一句；**LDA 的目标函数与变分参数全丢** |
| 参考文献（SparsePCA） | 2 条 | `[Mrl09]` Mairal et al. 2009；`[Jen09]` Jenatton et al. 2009 | L108 的 `References` 为空；正文 L88 的「based on.」与「see for a review」两处引用标记被吞 |
| 参考文献（KernelPCA） | 2 条 | `[Scholkopf1997]`、`[Bakir2003]` | L126 的 `References` 为空 |
| 参考文献（KernelPCA solver） | 4 条 | `scipy.linalg.eigh`、Halko et al. 2009、Szlam et al. 2014、Lehoucq et al. 1998 | 2.5.2.2 整段 References 丢失，连标题都不在剪藏里 |
| 参考文献（NMF） | 6 条 | Lee & Seung 1999、Hoyer 2004、Boutsidis & Gallopoulos 2008、Cichocki & Phan 2009、Févotte & Idier 2011、Lefèvre et al. 2011 | L375 的 `References` 为空；正文的 `[1]`/`[2]` 等标记被吞 |
| 脚注重复 | 886 词 | `[^1]`–`[^6]` 把 NMF 小节正文整段重复一遍 | 词数虚高（5,581 vs 官方正文约 4,626 词）；搜索时会命中两份相同段落 |
| 公式渲染噪声 | — | `n_{c o m p o n e n t s}`、`n_{s a m p l e s}`、`n_{f e a t u r e s}`、`\left|\right.` 残片、`$ℓ_{1}$`、`n _ f e a t u r e s` | 目标函数可读性差；复制到代码/笔记时需手工还原 |

**未丢失的部分**：所有 23 张唯一图、8 行 NMF doctest、示例列表、`See also`、PCA/Incremental/randomized/SparsePCA/TruncatedSVD/DictionaryLearning/FA/ICA/NMF/LDA 的散文与公式。与 [[sklearn-clustering]]（30 块全丢）相比，这份剪藏在**接口层**几乎完整；与 [[sklearn-manifold-learning]]（14 条文献全在）相比，它的**文献层**损失更大。

## 图表核查

23 张唯一图于 2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-decomposition/`。以下 15 张逐张目视核查并记录尺寸与 SHA-256 前 16 位；其余 8 张未打开，**不作描述**。

| # | 文件 | 尺寸 | SHA-256 前 16 | 目视读数 |
| --- | --- | --- | --- | --- |
| 1 | `sphx_glr_plot_faces_decomposition_002` | 600×460 | `e1f2c7a8ca10f3f0` | 标题「Eigenfaces - PCA using randomized SVD」；6 个**全局**人脸基，明暗渐变覆盖整张脸；色标 −0.04…0.04 |
| 2 | `sphx_glr_plot_faces_decomposition_003` | 600×460 | `9343ad781644e7de` | 标题「Non-negative components - NMF」；6 个灰度人脸成分，暗部与亮部并存；色标 −0.8…0.8（图例范围，不代表成分有负值） |
| 3 | `sphx_glr_plot_faces_decomposition_004` | 600×460 | `a0cd2dae536ff229` | 标题「Independent components - FastICA」；6 个局部化滤波器（眼/嘴/轮廓），对比度强；色标 −0.0015…0.0015 |
| 4 | `sphx_glr_plot_faces_decomposition_005` | 600×460 | `46de7f79d8fc38eb` | 标题「Sparse components - MiniBatchSparsePCA」；**局部斑块**（眼、鼻、嘴）在黑底上，明显稀疏；色标 −0.04…0.04 |
| 5 | `sphx_glr_plot_faces_decomposition_007` | 600×460 | `dc4a93194f23d7ea` | 标题是「**Cluster centers - MiniBatchKMeans**」，不是字典；6 个簇中心人脸——与正文把它当 `dict_img2` 引用不符（见「分歧」5） |
| 6 | `sphx_glr_plot_faces_decomposition_008` | 600×460 | `787d3fc081e66a92` | 标题「Factor Analysis (FA)」；6 个成分，明暗分布与 PCA 类似但**不保证正交**；色标 −0.04…0.04 |
| 7 | `sphx_glr_plot_faces_decomposition_009` | 320×360 | `6bb38fce4188c12f` | 标题「Pixelwise variance from Factor Analysis (FA)」；一张人脸的**逐像素方差热图**，不同区域亮度不同；色标 −0.02…0.02 |
| 8 | `sphx_glr_plot_faces_decomposition_010` | 600×460 | `46789e09fab27150` | 标题「Faces from dataset」；红/蓝双色人脸，表示正负载荷；色标 −0.4…0.4 |
| 9 | `sphx_glr_plot_faces_decomposition_013` | 600×460 | `c2ef263cd77512ae` | 标题「Dictionary learning - positive dictionary & code」；蓝/白局部斑块，无红色（全正）；色标 −0.04…0.04 |
| 10 | `beta_divergence` | 640×480 | `4aa0ea81355a0a58` | 横轴 $x\in[0,4]$、纵轴 $d_\beta(1,x)\in[0,3]$；5 条曲线 $\beta=0,0.5,1,1.5,2$，都在 $x=1$ 取 0；$\beta=2$ 增长最快 |
| 11 | `lda_model_graph` | 1081×262 | `464aa5a24ca333db` | 三层盘式图：$\alpha\to\theta_d$；$z_{dn}\to w_{dn}$（$w$ 灰色观测）；$\eta\to\beta_k$；外层盘 $D$、内层 $N_d$、右盘 $K$ |
| 12 | `sphx_glr_plot_pca_vs_lda_001` | 640×480 | `b6af9d52b73acce5` | 标题「PCA of IRIS dataset」；二维散点，setosa 集中在左、versicolor 居中、virginica 在右；坐标约 −3…4 / −1.5…1.5 |
| 13 | `sphx_glr_plot_kernel_pca_002` | 1400×400 | `6c4647e2b3d4b3c2` | 三面板：「Testing data」同心圆、「Projection using PCA」仍是同心圆（线性投影失败）、「Projection using KernelPCA」内外圈被拉开；坐标轴各自不同 |
| 14 | `sphx_glr_plot_incremental_pca_002` | 800×800 | `2b5dfbf835a0385d` | 标题「PCA of iris dataset」；与 #12 同一布局，说明 IncrementalPCA 的结果在视觉上与 PCA 不可区分 |
| 15 | `sphx_glr_plot_image_denoising_001` | 500×330 | `47a091daed4e0fca` | 标题「Distorted image」；左侧浣熊照片、右侧噪声差分，标题内 `Difference (norm: 11.71)` |

**读数要点**：

- **PCA/NMF/FA/ICA 的面板差别是「约束的可见化」**：#1 的 PCA 基是全局明暗模式；#2 的 NMF 仍接近整脸（非负但未强制稀疏）；#4 的 SparsePCA 才是明显的局部斑块；#3 的 ICA 是局部化滤波器但对比度方向不同；#6 的 FA 与 PCA 外观相近，差异在噪声模型而不是图像。
- **NMF 的「parts-based」不是自动的**：#2 的 NMF 面板并没有把脸切成眼睛/鼻子；真正的局部斑块出现在 #4（SparsePCA）与 #9（正字典）。这提醒「NMF = 零件」是**在合适约束与数据下**的倾向，不是定义（本库综合）。
- **KernelPCA 的图是本页最直观的「线性失败」证据**：#13 中 PCA 对同心圆完全无能为力，KernelPCA 把内外圈分开；但核参数（RBF `gamma`）没有标在图上，因此不能从图推断「KernelPCA 总更好」。
- **#5 是明确的资产错配**：正文 L186 把 `_007` 当 `dict_img2`，图标题却是 MiniBatchKMeans 的簇中心。引用该图时必须写「图里实际是 MiniBatchKMeans」，不要写成「字典」。
- **#7 是 FA 异方差噪声的图证据**：逐像素方差不是常数，眼/嘴/鼻周围与脸颊的亮度不同；正文 L258 的「heteroscedastic noise」由此可视化。注意它是**方差图**，不是 4 个特征的条形图。

## 外部核查（官方 1.9.1 API 页，不在剪藏内）

本节**不给任何一个默认值**（正文只出现 `whiten`、`svd_solver='randomized'`、`alpha`、`init`、`beta_loss`、`n_components` 等名字）。以下取自官方 1.9.1 生成页签名：

| 估计器/函数 | 1.9.1 签名（关键默认值） | 剪藏是否提及 |
| --- | --- | --- |
| `PCA` | `n_components=None`、`copy=True`、`whiten=False`、`svd_solver='auto'`、`tol=0.0`、`iterated_power='auto'`、`n_oversamples=10`、`power_iteration_normalizer='auto'`、`random_state=None` | 只提 `whiten` 与 `svd_solver='randomized'` |
| `IncrementalPCA` | `n_components=None`、`whiten=False`、`copy=True`、`batch_size=None` | 只提 `partial_fit`/`memmap` |
| `SparsePCA` | `n_components=None`、`alpha=1`、`ridge_alpha=0.01`、`max_iter=1000`、`tol=1e-08`、`method='lars'`、`random_state=None` | 只提 `alpha` |
| `MiniBatchSparsePCA` | `n_components=None`、`alpha=1`、`ridge_alpha=0.01`、`max_iter=1000`、`batch_size=3`、`shuffle=True`、`method='lars'`、`tol=0.001`、`max_no_improvement=10`、`random_state=None` | 只提「沿特征方向在线」 |
| `KernelPCA` | `n_components=None`、`kernel='linear'`、`gamma=None`、`degree=3`、`coef0=1`、`alpha=1.0`、`fit_inverse_transform=False`、`eigen_solver='auto'`、`tol=0`、`max_iter=None`、`iterated_power='auto'`、`remove_zero_eig=False`、`random_state=None` | 只提 `transform`/`inverse_transform` |
| `TruncatedSVD` | `n_components=2`、`algorithm='randomized'`、`n_iter=5`、`n_oversamples=10`、`random_state=None`、`tol=0.0` | 只提 $k$ |
| `SparseCoder` | `dictionary`（必填）、`transform_algorithm='omp'`、`transform_n_nonzero_coefs=None`、`transform_alpha=None`、`split_sign=False`、`positive_code=False`、`transform_max_iter=1000` | 提五种 `transform`，但不给默认 |
| `DictionaryLearning` | `n_components=None`、`alpha=1`、`max_iter=1000`、`tol=1e-08`、`fit_algorithm='lars'`、`transform_algorithm='omp'`、`split_sign=False`、`positive_code=False`、`positive_dict=False`、`random_state=None` | 提 `split_code`（正文名）与正性约束 |
| `MiniBatchDictionaryLearning` | `n_components=None`、`alpha=1`、`batch_size=256`、`shuffle=True`、`transform_algorithm='omp'`、`tol=0.001`、`max_no_improvement=10`、`random_state=None` | 提 `partial_fit`，不提默认 |
| `FactorAnalysis` | `n_components=None`、`tol=0.01`、`max_iter=1000`、`noise_variance_init=None`、`svd_method='randomized'`、`iterated_power=3`、`rotation=None`、`random_state=0` | 提 `rotation` 的 varimax，不提名/默认 |
| `FastICA` | `n_components=None`、`algorithm='parallel'`、`whiten='unit-variance'`、`fun='logcosh'`、`max_iter=200`、`tol=0.0001`、`whiten_solver='svd'`、`random_state=None` | 只提 `whiten` 与 FastICA |
| `NMF` | `n_components='auto'`、`init=None`、`solver='cd'`、`beta_loss='frobenius'`、`tol=0.0001`、`max_iter=200`、`alpha_W=0.0`、`alpha_H='same'`、`l1_ratio=0.0`、`shuffle=False` | 提 `init`、`alpha_W/alpha_H`、`l1_ratio`，不给默认 |
| `MiniBatchNMF` | `n_components='auto'`、`init=None`、`batch_size=1024`、`beta_loss='frobenius'`、`tol=0.0001`、`max_no_improvement=10`、`max_iter=200`、`forget_factor=0.7`、`fresh_restarts=False`、`random_state=None` | 提 `forget_factor`/`partial_fit`，不给默认 |
| `LatentDirichletAllocation` | `n_components=10`、`doc_topic_prior=None`、`topic_word_prior=None`、`learning_method='batch'`、`learning_decay=0.7`、`learning_offset=10.0`、`max_iter=10`、`batch_size=128`、`evaluate_every=-1`、`perp_tol=0.1`、`mean_change_tol=0.001`、`max_doc_update_iter=100`、`random_state=None` | 只提 `partial_fit`/online，不给默认 |

版本注记（官方 API 页，外部核查）：

- `NMF.n_components` 与 `MiniBatchNMF.n_components` 自 **1.6** 起默认 `'auto'`（1.4 加入 `'auto'` 取值）。
- `FastICA.whiten` 自 **1.3** 起默认 `'unit-variance'`，参数类型从布尔改为字符串（`True` 不再合法）。
- `MiniBatchDictionaryLearning.batch_size` 自 **1.3** 起默认从 3 改为 256。
- `DictionaryLearning.alpha` / `MiniBatchDictionaryLearning.alpha` 在 1.2 起「为 `None` 时默认值从 1.0 改为 `alpha`」。
- `KernelPCA` 的 `'randomized'` 求解器自 **1.0** 加入。
- `LatentDirichletAllocation` 的默认 `learning_method` 自 **0.20** 起是 `'batch'`；`n_topics` 在 0.19 改名为 `n_components`。

## 本机实测（scikit-learn 1.6.1，20 条）

**版本警告**：本机是 1.6.1（PyPI 当前可安装的最高版本；1.9.1 未安装）。以下只证明**机制存在**，不能替代对 1.9.1 的数值核对；涉及默认值（`n_components='auto'`、`FastICA.whiten`、`MiniBatchDictionaryLearning.batch_size`）的条目已在 1.9 变过，引用时必须带版本。

1. **PCA 不缩放，缩放会换模型**：iris 上 2 维 PCA + logistic 的 5 折 CV，原始 0.96、`StandardScaler` 后 0.9133；`PCA.mean_` 是原均值 `[5.843, 3.057, 3.758, 1.199]`，不是 0——**中心化在内部，缩放不在**。
2. **白化确实把方差变成 1**：iris、2 成分，`whiten=True` 的两轴方差 0.9933 / 0.9933；不白化 4.2001 / 0.2411。
3. **randomized 的反投影不精确**：digits、10 成分，randomized 与 full 的重建误差都是 751.787，但两个反投影矩阵的 Frobenius 差 0.567；`inverse_transform(transform(X))` 不等于 `X` 的低秩投影。
4. **IncrementalPCA 的 batch 效应**：iris、2 成分，`batch_size=150`（全量）与 PCA 逐位相同；`batch_size=30` 坐标最大差 0.00970；`batch_size=10` 0.01271。解释方差比 0.9237/0.0529（batch=10）→ 0.9246/0.0531（batch=150）。
5. **TruncatedSVD 的中心化等价**：对已中心化的 iris，`TruncatedSVD(2)` 与 `PCA(2)` 投影最大差 0.0；未中心化时前两个解释方差比 0.5288/0.4485。
6. **SparsePCA 的 alpha 同时改稀疏度与重构**：iris、3 成分，`alpha=0.01/0.1/1.0/10.0` → 零元素 3/3/8/10（共 12 个），重建误差 1.903/1.911/2.4/10.721。**`alpha=10` 已经过罚**。
7. **KernelPCA 的反投影是近似**：iris、RBF、4 成分、`fit_inverse_transform=True`，重建误差 6.9775；线性 PCA 3.8993。两者不可直接比。
8. **FastICA 的 `whiten=False` 会忽略 `n_components`**：1.6.1 上 `FastICA(n_components=2, whiten=False)` 的 `transform(X)` 形状 `(150, 4)`、`components_` `(4,4)`，但 `n_components` 属性仍是 2；`whiten='unit-variance'` 输出 `(150,2)`、方差 1；`whiten='arbitrary-variance'` 输出 `(150,2)`、方差 0.007。`whiten=True` 直接 `InvalidParameterError`。
9. **NMF 的 `transform` 与 `fit_transform` 不逐位相同**：iris 非负、2 成分，`init='random'` 最大差 3.72e-4；`init='nndsvda'` 7.68e-4。训练集编码应保存 `fit_transform` 的返回值。
10. **NMF 初始化决定零与误差**：4 成分、`solver='mu'`，`nndsvd` 零 5 个、误差 4.4395；`nndsvda` 零 0、2.3878；`nndsvdar` 零 0、1.4714；`random` 零 0、2.0252。
11. **NMF 的 beta 尺度不可比**：同一份非负 iris（+0.01），`beta_loss=2/1/1.5/0` → 误差 3.9579/2.475/3.091/1.6254。
12. **FactorAnalysis 的噪声方差是逐特征**：iris、2 成分，`noise_variance_` = `[0.016, 0.098, 0.004, 0.041]`（4 个不同值）；载荷最大非对角相关 0.555。
13. **varimax 不保证更稀疏**：iris、3 成分，`rotation='varimax'` 的 `|loading|<0.1` 个数 4（不旋转 3），最大非对角载荷相关 0.8904（不旋转 0.7399）。**旋转改善的是「简单结构」的解释，不是这个阈值下的稀疏度**。
14. **LDA 的矩阵方向**：6 篇文档、4 词、2 主题，`components_` `(2,4)`、`transform` `(6,2)`、每行和 1、perplexity 5.3101。
15. **DictionaryLearning 的 `split_sign` 把编码翻倍**：8 原子，普通 `(3,8)`，`split_sign=True` `(3,16)`，且全部非负（45 个零）。
16. **SparseCoder 的 `fit` 是空操作**：1.6.1 `hasattr(sc, 'fit')` 为 True，源码是 `Do nothing and return the estimator unchanged`；`transform_algorithm='omp'` + `n_nonzero_coefs=3` 每行 3 个非零。
17. **三个 MiniBatch 的 `partial_fit` 不一致**：`MiniBatchNMF` 有（连续 5 次后 `components_` `(3,4)`、`transform` `(5,3)`）；`MiniBatchDictionaryLearning` 有；`MiniBatchSparsePCA` **没有**。
18. **PCA 的 `n_components` 默认是 `None`**（1.9.1 签名；1.6.1 同）：它会取 `min(n_samples, n_features)`；`PCA(2)` 的简写只指定 2，不等于默认。
19. **`IncrementalPCA` 的 `batch_size=None`** 在 1.6.1 等价于 `5 * n_features`（外部核查源码；本机未逐值验证），因此“默认 batch”不是 1，也不是全量。
20. **`FactorAnalysis(random_state=0)` 是默认**：同一数据重复 `fit` 的载荷逐位相同；换 `random_state` 会改结果（本机未量化差异，只确认默认固定）。

## 分歧

1. **「`SparseCoder` does not implement a `fit` method」（L150）与 1.6.1 实现不符。** 它有一个空操作 `fit`（为 Pipeline API 存在），`hasattr(...,'fit')` 为 True。正确读法：**没有学习**，不是没有方法。引用时应写「不学习字典」，不要写「不能放进 Pipeline」。
2. **「`MiniBatchSparsePCA` 是 online algorithm」（L102）与「不实现 `partial_fit`」并存，容易误读。** 它的「online」是沿**特征**方向，不是样本方向；因此不能 `partial_fit` 新样本。本机 `hasattr` 为 False。
3. **FastICA 的 `whiten` 在 1.6.1 不接受布尔 `True`，且 `whiten=False` 会忽略 `n_components`。** 正文 L268 只说「whitening must be applied」「using the `whiten` argument」，没有说参数类型与维数后果。实测：`whiten=False` 输出 4 维（原始特征数），`components_` `(4,4)`；`whiten='unit-variance'` 才是 2 维。
4. **NMF 的 `transform` 不是 `fit_transform` 的精确复用。** L349–L358 的示例把 `W = fit_transform(X)` 与 `W_new = transform(X_new)` 并列，读者容易以为 `transform(X)` 会返回同一个 W；实测差 3.72e-4（random）/7.68e-4（nndsvda）。差异小，但在重构/对齐任务里不能假设逐位相等。
5. **正文把 `_007` 当 `dict_img2`（L186），但图标题是「Cluster centers - MiniBatchKMeans」。** 这是官方资产错配：同一张图在 `plot_faces_decomposition` 示例里是聚类中心，不是字典。引用该图必须按图标题写。
6. **「Varimax tends to produce sparser factors」（L260）在 iris 的 3 成分上没有兑现。** 实测 `|loading|<0.1` 的个数 4（不旋转 3），最大非对角相关反而从 0.7399 升到 0.8904。这不是说来源错，而是说「稀疏」是**渐近/数据依赖**的倾向，不能当成保证；在小样本、少因子上尤其如此。
7. **「PCA centers but does not scale」（L16）与「PCA 是标准预处理」的常见说法冲突。** 来源说得对：缩放不在 `PCA` 里。但这也意味着 `PCA` 之后接距离型模型时，**未缩放的主成分仍受原始量纲支配**；本机 iris 上缩放前后 CV 0.96 → 0.9133，方向与「缩放总是更好」相反。
8. **`KernelPCA.inverse_transform` 是近似（L118），但剪藏没给「近似到什么程度」。** 本机 6.9775 vs 线性 PCA 3.8993，说明核 PCA 的「重构」在 RBF 核下可能明显差于线性 PCA；不能用它做精确去噪评估。
9. **正文的 `split_code` 与 API 的 `split_sign` 不同名（L160）。** `SparseCoder`/`DictionaryLearning`/`MiniBatchDictionaryLearning` 的签名都只有 `split_sign`；按正文写 `split_code=` 会直接 `TypeError`。这是本页最容易被复制错的一个参数名。
10. **正文的 `transform_method` 与 API 的 `transform_algorithm` 不同名（L152）。** 正文说「五个 transform 方法由 `transform_method` 初始化参数控制」，而 1.9.1 签名是 `transform_algorithm`；同段还提到 `transform_n_nonzero_coefs`/`transform_alpha`，这些名字倒是对的。

## 局限

- **本节没有评估指标**：PCA 有 `score`/`explained_variance_ratio_`，NMF 有 `reconstruction_err_`，LDA 有 `perplexity`，但**不同模型的这些数不可比**（PCA 的方差比、NMF 的重构误差、LDA 的 perplexity 是三种量纲）。本节没有给「如何选 $k$」的统一判据。
- **本节不给默认值**（只有 `whiten=True`、`svd_solver='randomized'`、`alpha`、`init` 等名字），因此「用默认参数跑一下」在文档里查不到具体数字；本页默认值全部来自外部核查。
- **本节几乎没有预处理指导**：只在 PCA 处说「center but not scale」（L16），没有说「文本用 tf-idf」「图像归一化」「NMF 输入必须非负」「ICA 必须先白化」的完整流程。
- **本节没有下游用法**：分解结果如何进 `Pipeline`、如何防泄漏、`transform` 与 `fit_transform` 在测试集上的语义、NMF 的 W/H 如何保存，全节未提。
- **本节没有复杂度表**：只有 randomized PCA 与 KernelPCA 的少量复杂度；字典学习、NMF、LDA 的成本只能从实现/经验推断。
- **5 个官方示例脚本一个都没运行**（`plot_pca_vs_lda`、`plot_faces_decomposition`、`plot_incremental_pca`、`plot_kernel_pca`、`plot_image_denoising`），图中的时间/数值只能当量级。
- **本机实测只到 1.6.1**：`FastICA.whiten` 的类型、`NMF.n_components='auto'`、`MiniBatchDictionaryLearning.batch_size` 都已在 1.9 不同；不能把 1.6.1 的默认值当成 1.9.1。

## 待解问题

1. **PCA 的 `n_components` 选择**：本节只给「保留大部分方差」的定性说法，没有平行分析、交叉验证重构误差、`n_components='mle'` 的用法。是否应把 `n_components='mle'` 纳入实践页？（本库未验证 1.9.1 行为。）
2. **`whiten=True` 的下游影响**：本节说对 RBF SVM/KMeans 有用，但没有给「白化前后 CV」的对照。本库只测了方差，没有测下游。
3. **randomized SVD 的精度**：`n_oversamples=10`、`iterated_power='auto'` 的默认是否足以让反投影误差可忽略？本机 digits 上差 0.567，但缺少「误差 vs `n_oversamples`」的曲线。
4. **KernelPCA 的 `eigen_solver='auto'` 规则**：官方折叠块说 $n_c<10$ 且 $n>200$ 时用 arpack；本机 1.6.1 没有 `eigen_solver_` 属性可读，无法确认。需要在 1.9.1 上用源码/日志验证。
5. **TruncatedSVD 的 LSA 等价**：本节只说「中心化后等价 PCA」；文本上 tf-idf 后再 TruncatedSVD 与 LSA 的差别（是否中心化、是否用 `sublinear_tf`）没有覆盖。
6. **字典学习的 `alpha` 与 `n_components` 选择**：本节给目标函数与五种 transform，但没有给「字典多大」「alpha 多大」的判据；本库只测了 SparsePCA 的 alpha，没有扫 DictionaryLearning。
7. **NMF 的 `beta_loss` 选择**：KL 适合计数、IS 适合频谱，但本节只说三者是同一族；没有给「什么时候用哪个」的工程判据。
8. **NMF 的 `n_components='auto'`**：1.6 起默认 `'auto'`，但本节完全没解释 `'auto'` 如何选（外部核查只看到签名）。这是一个高影响的默认值。
9. **LDA 的 `learning_method='online'` 与 `batch` 的质量差**：本节只说 online 依赖 mini-batch 与学习率，没有给对照；本机只跑了 batch。
10. **FA 的 `rotation` 选择**：本节只举 varimax，没有说 oblimin/promax 等其它旋转，也没有说「旋转后载荷的置信区间」。
11. **ICA 的 `n_components` 与 `whiten=False` 的语义**：本机发现输出维数变成原特征数，但官方文档没有警告；需要在 1.9.1 上复核是否是 1.6.1 的 bug。
12. **所有分解器的 `Pipeline` 泄漏/成本**：PCA/TruncatedSVD 是无监督的，理论上可在训练集 `fit`、测试集 `transform`；但 `IncrementalPCA` 的 `partial_fit` 若在全量数据上跑，仍然会用测试集信息。本库未实测。

## 行号索引

| 内容 | 行号 |
| --- | --- |
| frontmatter | L1–L9 |
| 2.5.1 PCA | L10–L108 |
| 2.5.1.1 Exact PCA（中心化/白化/score） | L12–L30 |
| 2.5.1.2 Incremental PCA | L32–L49 |
| 2.5.1.3 Randomized SVD（复杂度/反投影） | L51–L77 |
| 2.5.1.4 Sparse PCA（目标/alpha/MiniBatch 注） | L79–L108 |
| 2.5.2 Kernel PCA | L110–L130 |
| 2.5.2.1 Exact kPCA（反投影近似） | L112–L126 |
| 2.5.2.2 Choice of solver | L128–L130 |
| 2.5.3 TruncatedSVD / LSA | L132–L144 |
| 2.5.4 Dictionary Learning | L146–L210 |
| 2.5.4.1 Sparse coding（五种 transform/split_code） | L148–L166 |
| 2.5.4.2 Generic dictionary（目标/正性） | L168–L192 |
| 2.5.4.3 Mini-batch dictionary（partial_fit/无停止） | L194–L210 |
| 2.5.5 Factor Analysis（潜变量/PPCA/FA/旋转） | L212–L264 |
| 2.5.6 ICA（盲源分离/白化） | L266–L282 |
| 2.5.7 NMF | L284–L375 |
| 2.5.7.1 Frobenius NMF（初始化/L1/L2） | L286–L316 |
| 2.5.7.2 beta-divergence（KL/IS/doctest） | L318–L363 |
| 2.5.7.3 Mini-batch NMF（forget_factor） | L365–L375 |
| 2.5.8 LDA（生成模型/矩阵方向） | L377–L407 |
| See also（NCA） | L414 |
| 脚注 1–6（重复 NMF） | L416–L463 |

## 关联页面

- 概念：[[matrix-factorization]]（本页的机制整理：$X\approx UV$ 骨架与六族约束）、[[manifold-learning]]（另一种换坐标：图方法 vs 核方法）、[[feature-selection]]（删列 vs 投影）、[[clustering]]（硬标签 vs 软编码）、[[gaussian-discriminant-analysis]]（有监督 LDA 与主题模型 LDA 的名称冲突）、[[regularization-and-sparsity]]（SparsePCA/NMF 的 $ℓ_1$）
- 实践：[[decomposition-selection-and-tuning]]（选型、$k$/`alpha`、初始化、`transform` 语义与坑清单）
- 工具：[[scikit-learn]]（`decomposition` 模块的估计器矩阵与默认值）
- 其他来源：[[sklearn-manifold-learning]]（同为无监督换坐标，但只有坐标没有成分）、[[sklearn-clustering]]（同为无监督，但输出标签）、[[sklearn-feature-selection]]（输出列子集）
