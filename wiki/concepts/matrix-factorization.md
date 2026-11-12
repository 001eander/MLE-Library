---
title: 矩阵分解与成分分解（Matrix Factorization / Component Decomposition）
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - decomposition
  - dimensionality-reduction
  - pca
  - nmf
  - ica
  - topic-model
  - unsupervised
sources:
  - "[[sklearn-decomposition]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档）；行为在 1.6.1 复核
---

# 矩阵分解与成分分解

## 一句话

矩阵分解把数据矩阵 $X$ 写成两个（或更多）低秩矩阵的乘积，并要求**其中一组矩阵具有某种结构**（正交、稀疏、非负、独立、主题分布）。结构不是事后解释，而是模型定义的一部分：换约束就是换模型。

## 与相邻类目的定位

本库的无监督产物现在有四种，区别不在「有没有 $y$」，而在**输出什么**：

| 类目 | 输出 | 是否保留原列 | 代表 | 页面 |
| --- | --- | --- | --- | --- |
| 特征选择 | 原列的子集 | 是 | `SelectKBest`、`RFE` | [[feature-selection]] |
| 流形学习 | 每个样本的一套新坐标 | 否（坐标不可反解释） | Isomap、LLE、t-SNE | [[manifold-learning]] |
| 聚类 | 每个样本一个离散标签 | 否 | KMeans、DBSCAN | [[clustering]] |
| 矩阵分解 | **编码矩阵 + 成分矩阵** | 否，但成分与原特征同维、可解释 | PCA、NMF、FA、ICA、LDA | 本页 |

**关键差别**：流形学习只给你坐标，坐标轴没有名字；矩阵分解给你 `components_`，每一行/列都是原特征空间里的一个向量，因此可以回答「这个成分主要由哪些特征构成」。这也是 PCA 能画 eigenfaces、NMF 能画「人脸零件」、LDA 能列「主题词」的原因。

> [!warning] 名称冲突
> `LDA` 在本库有两个完全不同的意思：[[gaussian-discriminant-analysis]] 里的 **Linear Discriminant Analysis** 是**有监督**分类/降维；本节 sklearn `decomposition.LatentDirichletAllocation` 的 **Latent Dirichlet Allocation** 是**无监督**主题模型。两者只共享缩写。引用时必须写全名。

## 共同骨架：$X\approx UV$

所有方法都可以写成「找低秩因子」：

$$
X \approx U V
$$

- $X$：$n_{\text{samples}}\times n_{\text{features}}$（sklearn API 的行方向）。
- $U$：$n_{\text{samples}}\times k$，样本的**编码**（PCA 的 score、NMF 的 W、LDA 的文档-主题）。
- $V$：$k\times n_{\text{features}}$，**成分**（PCA 的 `components_`、NMF 的 H、LDA 的主题-词）。

**不同的约束落在不同的矩阵上**：

| 方法 | 约束 | 目标 | 来源 |
| --- | --- | --- | --- |
| PCA | $V$ 的行正交；$X$ 先中心化 | 最大化解释方差 / 最小化重构误差 | [[sklearn-decomposition]], L16 |
| TruncatedSVD | 不做中心化；只取前 $k$ 个奇异值 | 低秩重构 | [[sklearn-decomposition]], L134–L136 |
| SparsePCA | $V$ 加 $ℓ_{1,1}$；$U$ 的列在单位球内 | 重构 + 稀疏 | [[sklearn-decomposition]], L91–L99 |
| DictionaryLearning | $U$（码）加 $ℓ_{1,1}$；$V$（字典）的原子在单位球内 | 重构 + 稀疏 | [[sklearn-decomposition]], L174–L180 |
| NMF | $U\ge0$、$V\ge0$；可选 L1/L2 | 重构（Frobenius/KL/IS）+ 非负 | [[sklearn-decomposition]], L286–L316 |
| FactorAnalysis | $V$ 低秩；噪声协方差 $\Psi$ 为对角 | 高斯似然 | [[sklearn-decomposition]], L214–L250 |
| ICA | 成分最大独立；需白化 | 非高斯性/独立性 | [[sklearn-decomposition]], L268 |
| LDA | 文档-主题、主题-词都来自 Dirichlet | 主题似然/ELBO | [[sklearn-decomposition]], L379–L396 |

> [!note] 记号陷阱
> 来源在 Factor Analysis 一节把 $x_i$ 当成**列**（L216–L218），而 sklearn 的 `fit(X)` 把样本当成**行**。同一公式在 API 里转置了一次；读 `components_` 的形状时要按 API 的行方向理解，不要按文档的列方向。

**综合（本库）**：这四类约束不是互斥的「技巧」，而是回答四个不同问题：

1. **PCA/TruncatedSVD**：怎样用尽量少的正交方向保住方差？
2. **SparsePCA/DictionaryLearning**：怎样用尽量少的原特征/原子重建样本？
3. **NMF/FA/ICA**：数据由哪些「正向零件」「潜变量」「独立源」叠加而成？
4. **LDA**：文档由哪些主题混合而成？

## PCA 家族：正交方差

### 精确 PCA

- `PCA` 在 `fit` 里学 $k$ 个成分，`transform` 把新样本投影到这些成分上（L14）。
- **只中心化、不缩放**（L16）。`PCA` 的 `mean_` 是训练集均值；`components_` 是中心化后协方差的主轴。**要不要先 `StandardScaler` 是模型选择**，不是 `PCA` 的参数。
- **`whiten=True`** 把每个成分除以它的标准差，使输出各轴方差为 1（L16）。这对 RBF SVM、KMeans 这类假设各向同性的下游有用；但它也**放大了噪声方向**，因为小方差成分被拉到与方法差成分同尺度。
- **概率解释**：`PCA` 有 `score`，可用于交叉验证（L20）；剪藏没有给似然公式，因此不能从本页推出「`score` 一定优于重构误差」。

### 增量与随机化

- `IncrementalPCA` 用 `partial_fit` 分块拟合，内存随**每批样本数**而不是总样本数增长（L34–L40）；支持 `numpy.memmap` 外存。来源说结果「almost exactly match」PCA；本机 batch=10/30 与 PCA 的坐标最大差 0.01271/0.00970，batch=150（全量）为 0.0。
- `PCA(svd_solver='randomized')` 只近似前 $k$ 个奇异向量；时间 $O(n_{max}^2 n_{components})$ vs 精确 $O(n_{max}^2 n_{min})$，内存 $2n_{max}n_{components}$ vs $n_{max}n_{min}$（L63–L65）。
- **randomized 的 `inverse_transform` 不是精确逆**（L67）。本机 digits 10 维：重建误差与 full 相同（751.787），但反投影矩阵差 0.567。**做重构/去噪时不能假设 `inverse_transform(transform(X))` 是 $X$ 的最优低秩投影。**

### Kernel PCA

- 用核函数把样本映到高维空间再做 PCA；支持 `transform` 与 `inverse_transform`（L114）。
- **反投影是近似**：`inverse_transform` 用 kernel ridge 学「PCA 基 → 原空间」的映射（L118）。本机 iris（RBF）重建误差 6.9775 vs 线性 PCA 3.8993。
- **成分数上界是样本数**（L130），不是特征数；大样本时完整 kPCA 浪费，近似特征求解器可提速。剪藏没有给 `eigen_solver` 的三个取值与自动规则（在丢失折叠块里）。

## 稀疏与字典：用少量原子重建

### SparsePCA

- 在 PCA 的重构目标上加 $ℓ_{1,1}$ 惩罚，使成分稀疏（L81–L99）。`alpha` 越大，零越多。
- 本机 iris（3 成分）：`alpha=0.01/0.1/1/10` → 零 3/3/8/10（共 12），重构误差 1.903/1.911/2.4/10.721。**稀疏与重构是对立的**；`alpha=10` 已经过罚。
- `MiniBatchSparsePCA` 更快但更粗；它的「online」是沿**特征**方向，因此**没有 `partial_fit`**（L102；本机 `hasattr` 为 False）。

### DictionaryLearning / SparseCoder

- `SparseCoder` 用**固定字典**做稀疏编码，不学习字典（L150）。五种 `transform` 方法：OMP、LARS、Lasso-LARS、坐标下降 Lasso、阈值法（L152–L160）。
- 来源的取舍：**阈值法最快但重构不准**；图像重建用 OMP 最准、无偏（L158）。本机 `SparseCoder(transform_algorithm='omp', n_nonzero_coefs=3)` 每行 3 个非零。
- `DictionaryLearning` 学一个过完备字典，交替更新「码」与「字典」（L170–L174）；码/字典都可加正性约束（L184）。
- `split_sign`（正文写作 `split_code`）把编码长度翻倍：正部 + 负部的绝对值，结果非负（L160–L162）。本机 `(3,8)` → `(3,16)`。**这是给下游线性模型区分正负载荷的，不是更稀疏的编码。**
- `MiniBatchDictionaryLearning` 有 `partial_fit`，但**默认没有停止条件**（L198）。

## 概率与潜变量：PPCA / FA / ICA

### 从 PCA 到 FA

来源给了一个非常干净的推导（L214–L250）：

$$
x_i = W h_i + \mu + \epsilon,\qquad h_i\sim\mathcal N(0,I),\qquad \epsilon\sim\mathcal N(0,\Psi)
$$

- 若 $\Psi=\sigma^2 I$：噪声各向同性 → **概率 PCA（PPCA）**。
- 若 $\Psi=\operatorname{diag}(\psi_1,\dots,\psi_n)$：噪声逐特征 → **Factor Analysis（FA）**。
- 两者都估计「低秩协方差 + 噪声」；FA 多出来的自由度就是**异方差噪声**（L258）。
- **FA 的成分不保证正交**（L254）。这是它与 PCA 最容易被忽略的几何差别：PCA 的 `components_` 是正交基；FA 的载荷矩阵只是一个低秩因子，旋转后仍等价。
- `rotation='varimax'` 让载荷趋向「简单结构」（L260），但**旋转不改变似然，也不保证在你的数据上更稀疏**；本机 iris 3 成分上 varimax 的 $|loading|<0.1$ 个数 4（不旋转 3），最大非对角相关 0.8904（不旋转 0.7399）。

### ICA

- ICA 分离**最大独立的加性子成分**（L268），常用于盲源分离；它通常不是降维。
- **模型没有噪声项，所以必须白化**（L268）：可以内部 `whiten`，也可以先用 PCA 白化。白化是模型正确性的条件，不是可选优化。
- 本机 1.6.1 的坑：`whiten=False` 时 `n_components` 被忽略，输出维数回到原特征数（4），`components_` 是 `(4,4)`；`whiten='unit-variance'` 才是 2 维、方差 1；`whiten=True` 直接报错（1.3 起参数类型改为字符串）。
- 直觉：PCA 找**方差最大**的方向，ICA 找**最不高斯/最独立**的方向；两者可以给出完全不同的轴。

## NMF：加法零件

- NMF 要求 $X\ge0$、$W\ge0$、$H\ge0$，因此表示是**叠加**而不是加减（L286–L290）。这使它在图像、文本、音频等非负数据上产生「零件」。
- **目标函数**：$\frac12\lVert X-WH\rVert_F^2$ 加可选 L1/L2（L306–L316）；`alpha_W`/`alpha_H` 控制强度，`l1_ratio` 控制混合，先验按特征数/样本数缩放。
- **beta-divergence**（L320–L347）：Frobenius（$\beta=2$）、KL（$\beta=1$）、IS（$\beta=0$）是同一族；不同 $\beta$ 的 `reconstruction_err_` 量纲不同，**不可直接比较**。本机同一数据：3.9579/2.475/3.091/1.6254（$\beta=2/1/1.5/0$）。
- **初始化决定局部极小**（L300–L306）：`nndsvd` 适合稀疏，`nndsvda`/`nndsvdar` 适合稠密；`solver='mu'` 不能更新初始化中的零。本机 4 成分：`nndsvd` 误差 4.4395、`nndsvda` 2.3878、`nndsvdar` 1.4714、`random` 2.0252——**同一模型差近 3 倍**。
- **矩阵语义**：`fit_transform(X)` 返回 $W$；`components_` 是 $H$；`transform(X_new)` 用 $H$ 求新样本的 $W$（L349–L358）。本机 `transform(X)` 与 `fit_transform(X)` 不逐位相同（差 3.72e-4 / 7.68e-4），训练集编码要保存前者。
- `MiniBatchNMF` 有 `partial_fit` 与 `forget_factor`（L367–L373）。

## 主题模型：NMF 与 LDA

- LDA 是三层生成模型：文档-主题 Dirichlet（$\alpha$）、主题-词 Dirichlet（$\eta$）、每个词的主题分配（L379–L388）。
- `LatentDirichletAllocation` 用在线变分贝叶斯；batch 与 online 两种更新；online 只保证局部最优，质量/速度依赖 mini-batch 与学习率（L392–L396）。
- **矩阵方向**：输入「文档-词」，`components_` 是「主题-词」，`transform` 是「文档-主题」（L398）。本机 2 主题/4 词：`components_ (2,4)`、`transform (6,2)`、每行和 1。
- NMF 与 LDA 都是主题模型的常用实现，但目标不同：NMF 最小化重构（Frobenius/KL），LDA 最大化主题模型的 ELBO；NMF 的 `components_` 也是主题-词，但**没有 Dirichlet 先验**，因此不自动产生「稀疏主题」。选择时应先问「要不要生成式概率模型」，而不是「哪个主题更漂亮」（本库综合）。

## 可辨识性、旋转与随机性

| 模型 | 唯一性 | 工程后果 |
| --- | --- | --- |
| PCA | 成分在符号与退化子空间旋转下不唯一 | 换 `random_state`/`svd_solver` 可能翻符号；比较成分前先对齐 |
| TruncatedSVD | 同 PCA，但不中心化 | 与 PCA 的成分不可直接比 |
| SparsePCA / DictionaryLearning | 非凸；初始化/`alpha` 改变解 | 必须固定 `random_state`，报告 `alpha` |
| NMF | 非凸；初始化改变局部极小；尺度可互相吸收 | 必须固定 `init`/`random_state`；`W` 与 `H` 的尺度不唯一 |
| FA | 因子旋转不唯一 | `rotation` 改变解释，不改变似然 |
| ICA | 独立成分在符号/排列下不唯一 | 结果可复现但轴的顺序/符号可能变 |
| LDA | 在线 VB 只保证局部最优 | mini-batch/学习率/种子都会改主题 |

**综合（本库）**：这张表解释了为什么「分解结果不可复现」经常不是实现 bug，而是模型本身的对称性。工程上要固定的是**种子 + 初始化 + 预处理**，并在报告里写清「成分只到符号/旋转/排列」。

## 成本与规模

| 方法 | 主要成本 | 规模建议 |
| --- | --- | --- |
| `PCA(svd_solver='full')` | $O(n_{max}^2 n_{min})$ | 中小规模、要精确 |
| `PCA(svd_solver='randomized')` | $O(n_{max}^2 k)$ | $k\ll\min(n,p)$ |
| `IncrementalPCA` | 每批 $O(\text{batch}\cdot p^2)$ 量级 | 数据不能进内存 |
| `KernelPCA` | 核矩阵 $O(n^2)$ 内存；精确特征分解 $O(n^3)$ | $n$ 大时用近似求解器；成分数受 $n$ 限制 |
| `SparsePCA` / `DictionaryLearning` | 多次 Lasso/交替优化 | 中规模；大数据用 MiniBatch |
| `NMF` / `MiniBatchNMF` | 矩阵乘法 + 迭代 | 大数据用 MiniBatch + `partial_fit` |
| `LatentDirichletAllocation` | 每轮遍历文档-词 | 大数据用 `learning_method='online'` + `partial_fit` |

> [!note] 复杂度不是文档给的
> 除 randomized PCA 与 KernelPCA 的少量公式外，本节没有给复杂度表；上表的「主要成本」是**本库综合**，不是 [[sklearn-decomposition]] 的原文。引用具体复杂度时必须区分「来源论述」与「本库估计」。

## 失效模式

1. **不缩放就 PCA**：PCA 只中心化（L16）；量纲大的特征会独占前几个成分。本机 iris 上缩放前后 CV 0.96 → 0.9133，方向与「缩放总是更好」相反——**缩放是模型选择，不是必然改进**。
2. **把 `whiten=True` 当默认**：白化放大低方差方向；下游若对噪声敏感，白化会让结果更差。
3. **把 `TruncatedSVD` 当 PCA**：它不中心化；在非文本数据上，未中心化的 SVD 会保留均值方向。
4. **把 `SparsePCA` 的 `alpha` 调大当「更可解释」**：零越多，重构越差；`alpha=10` 在 iris 上误差 10.721。
5. **把 `split_sign` 当稀疏化**：它只是把正负分开，长度翻倍，不增加稀疏度。
6. **NMF 喂负值**：NMF 要求非负；负值会直接报错或需要平移，平移会改变「零」的含义。
7. **NMF 用 `mu` + `nndsvd`**：初始化的大量零无法被 `mu` 更新，结果更差（L304；本机误差 4.4395）。
8. **比较不同 `beta_loss` 的 `reconstruction_err_`**：量纲不同；本机 3.9579 vs 1.6254 不能说明 IS 更好。
9. **把 `transform(X)` 当 `fit_transform(X)`**：两者不逐位相同；训练集编码要保存 `fit_transform` 的返回值。
10. **FastICA 用 `whiten=False` 还想降维**：它忽略 `n_components`，输出回到原特征数。
11. **FA 把成分当正交基**：FA 的载荷不保证正交；旋转后解释会变。
12. **LDA 主题数当 `n_clusters`**：LDA 的主题不是硬聚类，文档-主题是分布；用聚类指标评估会误导。
13. **在测试集上重新 `fit` 分解器**：`PCA`/`TruncatedSVD` 是无监督的，仍应只在训练集 `fit`，测试集只 `transform`；否则测试集信息进入成分。
14. **忽略符号/排列不唯一**：比较两次 PCA/NMF 的成分前先做对齐，否则会把「翻符号」当成「模型变了」。
15. **把 `components_` 当特征重要性**：PCA 的成分系数不是因果重要性，也不是「删掉这个特征」的依据；需要重要性时用 [[feature-selection]] 或 permutation importance。

## 待解问题

1. **如何选 $k$**：本节没有统一判据；PCA 可用 `explained_variance_ratio_`/`score`，NMF 用 `reconstruction_err_`，LDA 用 `perplexity`，但三者不可比。是否需要在本库建立「按任务选 $k$」的实践页？（本页已写进 [[decomposition-selection-and-tuning]]，但仍是经验规则。）
2. **`whiten=True` 的下游影响**：来源说对 RBF SVM/KMeans 有用（L16），但没有给对照；本库只测了方差。
3. **NMF 的 `n_components='auto'`**：1.6 起默认，但本节完全不解释它如何选；这是一个高影响的默认值。
4. **`KernelPCA` 的 `eigen_solver='auto'`**：官方折叠块说 $n_c<10$ 且 $n>200$ 时用 arpack；本机 1.6.1 无 `eigen_solver_` 可读，未验证。
5. **LDA 的 online vs batch**：质量/速度差没有对照；本机只跑 batch。
6. **FA 的旋转选择**：本节只举 varimax；其它旋转与载荷不确定性未覆盖。
7. **分解器的 `Pipeline` 泄漏/成本**：PCA/TruncatedSVD 理论上可训练/测试分离；但 `IncrementalPCA.partial_fit` 若在全量数据上跑仍会用测试信息。本库未实测。
8. **NMF 的 `beta_loss` 选择**：KL 适合计数、IS 适合频谱，但本节没有工程判据；需要更多来源。

## 关联页面

- 来源：[[sklearn-decomposition]]（本页的逐节摘要、版本判定、转换损失与本机实测）
- 实践：[[decomposition-selection-and-tuning]]（选型、$k$/`alpha`、初始化、`transform` 语义与坑清单）
- 相邻概念：[[manifold-learning]]（图方法换坐标）、[[feature-selection]]（删列）、[[clustering]]（硬标签）、[[gaussian-discriminant-analysis]]（有监督 LDA）、[[regularization-and-sparsity]]（$ℓ_1$/$ℓ_2$ 的其他作用面）
- 工具：[[scikit-learn]]（`decomposition` 模块的估计器矩阵与默认值）
