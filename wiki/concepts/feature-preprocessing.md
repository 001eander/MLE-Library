---
title: "特征预处理（Feature Preprocessing）：有状态变换、四个作用轴，以及「保秩/保相关/保稀疏」的三选"
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - preprocessing
  - feature-scaling
  - feature-engineering
  - categorical-encoding
  - target-encoding
  - discretization
  - scikit-learn
sources:
  - "[[sklearn-preprocessing]]"
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-nearest-neighbors]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1 的 `sklearn.preprocessing`（接口与文档论述，见 [[sklearn-preprocessing]]）
  - scikit-learn 1.6.1（本页标「实测」的运行时行为；与 1.9.1 并列时标两版）
---

# 特征预处理（Feature Preprocessing）

## 一句话定义与三个推论

**定义**：在拟合模型之前，把原始特征矩阵改写成另一种表示——同一个观测、不同的数（[[sklearn-preprocessing]], L10：「change raw feature vectors into a representation that is more suitable for the downstream estimators」）。

三个推论决定了它和其他章节的关系：

1. **它不创造信息，只改变「模型看到什么」**。所有变换都是**逐列/逐样本的确定性函数**（分位数变换也是：它只是把秩重排到目标分布），因此对 $y$ 的条件分布没有新信息；能改变的只有**模型假设与实际尺度/形状之间的匹配程度**。所以「加了预处理分数就涨」等价于「模型原来的假设被违背了」——实测最干净的一个例子：把第一个特征乘 1000（数据可分性不变），`SVC()` 的 5 折准确率从 **0.9833 掉到 0.7767**，加一层 `StandardScaler` 再回到 0.9833（[[sklearn-preprocessing]] 实测 36）。
2. **它是「有状态的」，所以必须放进管线**。16 个变换器中有 12 个在 `fit` 时学习数据相关的量（`scale_`/`quantiles_`/`lambdas_`/`categories_`/`bin_edges_`/`knots`/`encodings_`），`transform` 只把这个量套到新数据。因此「预处理」不是数据清洗，而是**模型的一部分**（[[sklearn-preprocessing]], L57）。不学习数据相关状态的另外 4 个是 `Normalizer`/`Binarizer`/`FunctionTransformer` 与 `PolynomialFeatures`：前三者的 `fit` 不学任何量，后者只记录由 `degree` 决定、与数据无关的 `powers_`。
3. **它的代价可以被量化，而且三种代价互斥**。同一份数据、同一列：
   - **保秩**（分位数变换）：Pearson 0.891 → **0.754**，Spearman 0.754 → 0.754（保序、毁线性相关）；
   - **保相关**（仿射缩放）：两者都不变，但对离群点不稳健（`StandardScaler` 的 `scale_` 从 1.0 被抬到 8.53/11.03，`RobustScaler` 保持在 1.29/1.35）；
   - **保稀疏**（`MaxAbsScaler`）：零保持零，但中心化被禁止（`StandardScaler(with_mean=True)` 在稀疏输入上直接 `ValueError`）。

这三条是本节唯一「可以同时给出读数」的取舍轴，也是本页的主结构。

## 四个作用轴：谁在看谁

「预处理」这个词把四种完全不同的几何操作混在一起。按**在哪个轴上作用**重排，才能看出它们为什么不能互换：

| 轴 | 代表 | 输出列数 | 改变什么 | 保留什么 |
| --- | --- | --- | --- | --- |
| **逐列**（每列一个变换） | 缩放器、分位数/幂变换、分箱 | 固定（分箱/低频聚合会变） | 每列的**单位、位置、形状** | 行与行的次序、列的意义 |
| **逐行**（每个样本一个变换） | `Normalizer`、`Binarizer` | 固定 | 每个样本的**长度/符号** | 列的意义、样本之间的**夹角**（$ℓ_2$） |
| **全矩阵** | `KernelCenterer` | 固定（$n\times n$ → $n\times n$） | Gram 矩阵的均值 | 特征空间里的内积结构（中心化后） |
| **列之间** | `PolynomialFeatures`、`SplineTransformer`（否）、`OneHotEncoder`（列数扩张） | **爆炸** | 特征集合本身 | 原始列（作为子集或基） |

**关键区别**：前两个轴的「单位」完全不同——`scale`/`StandardScaler` 的 `axis=0`（按列）与 `normalize` 的 `axis=1`（按行）是源码级的相反默认值（外部核查：`scale(X, *, axis=0, …)`、`normalize(X, norm='l2', *, axis=1, …)`）。这不是命名问题：**按行归一化后，同一个特征在不同样本上的尺度可以完全不同**，而按列缩放保证同一特征在所有样本上同一尺度。

**本库综合**：这条区分解释了为什么「文本分类要 `Normalizer`、表格回归要 `StandardScaler`」不是惯例而是几何：文本的「长度」是文档长度（噪声），表格的「列尺度」是量纲（噪声）。剪藏把两者分别写在 8.3.1 与 8.3.3，从不说它们是两种不同的轴。

## 七类变换与它们的假设

| 类 | 对象 | 它假设下游是什么 | 代价 | 剪藏位置 |
| --- | --- | --- | --- | --- |
| 中心化 + 缩放 | `StandardScaler` | 目标函数对尺度敏感（RBF 核、$ℓ_1$/$ℓ_2$、梯度下降） | 对离群点不稳健；破坏稀疏 | L14–L77 |
| 区间缩放 | `MinMaxScaler`、`MaxAbsScaler` | 下游要求有界输入（神经网络、图像） | `MinMaxScaler` 默认不裁剪，测试集可越界（实测 `-1.5 / 1.6667`） | L79–L147 |
| 稳健缩放 | `RobustScaler` | 同上，但数据有离群点 | 不能 `fit` 稀疏；`unit_variance=False` 默认，尺度不是方差 | L160–L162 |
| 秩/形状变换 | `QuantileTransformer`、`PowerTransformer` | 下游假设特征近似高斯或需要消除偏度 | 分位数变换毁相关（0.891→0.754）、区间外退化；幂变换 `standardize=True` 默认会再标准化 | L168–L252 |
| 逐样本归一化 | `Normalizer`、`Binarizer` | 相似度/核/概率模型；二值特征 | 抹掉长度；全零行保持全零 | L254–L296、L690–L725 |
| 类别编码 | `OrdinalEncoder`、`OneHotEncoder`、`TargetEncoder` | 估计器只吃数值；不同编码对应不同假设 | 序数强加顺序（$R^2$ 0.2236 vs 0.9845）；one-hot 列数随基数；目标编码必须折外（+0.055） | L298–L626 |
| 展开/离散化 | `PolynomialFeatures`、`SplineTransformer`、`KBinsDiscretizer` | 下游是线性模型，需要非线性 | 列数爆炸（20 维 3 阶 = 1771 列） | L628–L818 |

## 缩放：它改变的是模型，不是数值

剪藏给的理由只有一句（L20）：目标函数里的项「may assume that all features are centered around zero or have variance in the same order … If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function」。这句话可以拆成四个具体机制，每个都有独立证据：

1. **核方法**：RBF 核 $k(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$ 的距离被最大尺度的列支配。实测：`SVC()` 0.7767 → 0.9833（[[sklearn-preprocessing]] 实测 36）。
2. **正则化**：$ℓ_1$/$ℓ_2$ 惩罚对系数施加同一尺度的先验，因此**缩放等价于改每个特征的有效惩罚强度**。这条与 [[regularization-and-sparsity]] 的「惩罚在多参数化下的可比性」是同一件事；也解释了为什么 `RidgeCV`/`LassoCV` 的 `alpha` 只在给定的缩放下有可比较的含义。
3. **距离/最近邻**：kNN 与层次聚类完全由距离定义，尺度决定「谁是谁的邻居」（[[nearest-neighbors]]）；官方 1.6 节本身**完全不提缩放**，而官方示例自己套了 `StandardScaler`（[[sklearn-nearest-neighbors]] 的使用边界）。
4. **优化器**：`liblinear`/`sag`/`saga` 对未缩放数据不鲁棒（[[sklearn-linear-models]] L499）；本节不涉及求解器，但 L12 与 L20 是同一件事的另一半。

**本库综合**：因此「标准化是否必要」的正确判据不是「数据是否正态」，而是**「下游的假设里有没有一个「同尺度」的前提」**：

| 下游 | 需要缩放？ | 原因 |
| --- | --- | --- |
| 树模型（RF/GBDT/HGBT） | **不需要**（轴对齐切分对单调变换免疫） | 见 [[decision-trees]] 的四个推论 |
| 线性模型 + 正则 | **需要** | 惩罚等价于先验尺度 |
| SVM（RBF/多项式核） | **需要** | 距离被大尺度列支配 |
| kNN / 层次聚类 / KMeans | **需要** | 距离定义模型 |
| 神经网络（MLP） | **需要** | 输入尺度影响初始化与梯度 |
| 朴素贝叶斯（GaussianNB） | 按列独立估计，**不需要**（但也不受伤） | 每列自己的 $\sigma_j$ |
| `QuantileTransformer`/`PowerTransformer` 之后 | **通常不需要再标准化** | 前者输出已定区间；后者 `standardize=True` 默认已标准化 |

### 三种缩放的分工与两个「静默替换」

- `StandardScaler`：常数列的 `scale_` 被**替换为 1**（实测：`[[1,5],[1,3],[1,7]]` → `[1.0, 1.633]`）。好处是没有 nan，代价是「方差为 0」的信息消失 → `VarianceThreshold` 在缩放后失效（[[sklearn-feature-selection]] 实测 16）。
- `MinMaxScaler`：`clip=False` 是默认，测试集**可以越界**（L105–L106 的 doctest 自己印出 `-1.5 / 1.6667`）；`clip=True` 才截断。它也是唯一**输出范围由训练集决定**的缩放器。
- `RobustScaler`：`quantile_range=(25, 75)`、`unit_variance=False` 默认。所以它的 `scale_` 是 IQR 而不是标准差，**不能与 `StandardScaler.scale_` 互换解释**；`unit_variance=True` 才把 IQR 折成「正态等价标准差」。

实测对比（95 个正态点 + 5 个偏移点，两列）：`StandardScaler` 的 `mean_ = [1.46, 2.56]`、`scale_ = [8.53, 11.03]`；`RobustScaler` 的 `center_ = [-0.03, 0.10]`、`scale_ = [1.29, 1.35]`。也就是说 5 个离群点把标准差的估计放大了 **6–8 倍**，而中位数/IQR 几乎不动。

## 分位数变换与幂变换：保秩的两条路

两者都**单调**（L170），因此都保秩；差别在「用什么函数」与「学多少参数」：

| | `QuantileTransformer` | `PowerTransformer` |
| --- | --- | --- |
| 形式 | $G^{-1}(F(X))$（经验 CDF + 目标分位函数） | 参数化幂函数（Box-Cox / Yeo-Johnson） |
| 每列参数 | $n$ 个分位点（默认 `n_quantiles=1000`，会被样本量夹住 → 实测 112 行时 `n_quantiles_=112`） | **1 个 λ**（极大似然） |
| 输出 | 均匀 $[0,1]$ 或正态（截断 1e-7） | 近似正态；`standardize=True` 默认再标准化 |
| 训练范围外 | **退化**成 `0`/`1`（uniform）或 `±5.1993`（normal） | 可外推（幂函数有定义） |
| 保秩 | 是（严格保序） | 是（严格保序） |
| 毁什么 | 相关与距离（Pearson 0.891→0.754） | 相关也会变，但 λ 有可解释方向 |

**本库综合**：三个必须分开的直觉：

1. **「变成高斯」不等于「变成有用」**。官方配图（`plot_map_data_to_normal_001.png`，已入库）显示：Gaussian 输入经 Box-Cox 得 λ=0.54、Uniform 输入得 λ=0.63 且**变换后仍是平的**、Bimodal 输入得 λ=1.69 且**仍是双峰**。幂变换是**单调**的，所以**不可能把双峰变成单峰**——它只能改变偏度与尾部。这条是「为什么必须画图」的硬理由（L232）。
2. **λ 不可跨方法比较**。同一批 lognormal 数据上 Box-Cox 与 Yeo-Johnson 的 λ 差 2 倍以上（图里 lognormal：0 vs −0.83；Weibull：12.59 vs 24.53）。所以「λ 接近 0 说明该取对数」这句话只在同一种方法内部成立。
3. **`standardize=True` 是默认**（外部核查），它会**再套一层零均值单位方差**。所以「Box-Cox 之后数据是纯幂变换的结果」是错的；要保住 λ 的解释必须显式写 `standardize=False`（实测：两者输出的均值/标准差分别是 `0/1` 与 `[-0.155, -0.0216, 0.0451]`/`[0.866, 0.929, 1.004]`）。

## 归一化：逐样本，方向与长度

`Normalizer` 缩放的是**每一行**（L256：「scaling individual samples to have unit norm」），因此：

- 它**没有状态**（`fit` 是 no-op，L272），因为不需要训练集统计量。
- 它**只改变长度，不改变夹角**（$ℓ_2$）：这是「余弦相似度 = 归一化后的点积」的来源，也是 L296「L2 normalization is also known as spatial sign preprocessing」的含义。
- **全零行保持全零**（实测），所以 `Normalizer` 不解决零向量问题。
- `norm` 可选 `'l1'`/`'l2'`/`'max'`，默认 `'l2'`；`axis` 默认 **1**（与 `scale` 的 0 相反）。

**本库综合**：`Normalizer` 与 `StandardScaler` 的差别不是「行 vs 列」这条几何事实本身，而是**它们隐含的「什么不该影响模型」不同**——`Normalizer` 说「样本的总量不该影响判断」（文本长度、曝光量），`StandardScaler` 说「特征的单位不该影响判断」。选错时两者的症状也不同：选错 `Normalizer` 会让**短样本被放大**（噪声放大），选错 `StandardScaler` 会让**大尺度特征支配**（信号被淹）。

## 类别编码：三条代价曲线

| | 列数 | 强加顺序？ | 未见类别 | 吃 $y$？ | 实测代价 |
| --- | --- | --- | --- | --- | --- |
| `OrdinalEncoder` | 1 | **是**（字典序） | `handle_unknown='use_encoded_value'` | 否 | $R^2$ **0.2236**（vs one-hot 0.9845）；改名后 **0.0236** |
| `OneHotEncoder` | $n_{\text{cat}}$（`drop` 后 −1） | 否 | `'ignore'`→全零；`'infrequent_if_exist'`→低频列 | 否 | 400 个 zip 值 → 400 列 |
| `TargetEncoder` | 1（多类时 $n_{\text{classes}}$） | 否 | `target_mean_` | **是** | 折外 vs 非折外差 **+0.055** |

四条判据（本库综合）：

1. **序数编码的合法性 = 「目标效应在字典序上是否单调」**。这不是风格偏好：实测效应向量 `[3,-2,0.5,2,-3.5]` 关于字典序非单调时，序数编码把 $R^2$ 从 0.9845 压到 0.2236；而**把一个水平改名**（数据完全不变）就再掉到 0.0236。因此「能不能用 `OrdinalEncoder`」取决于**类别本身的序是否有意义**（教育程度、星级），而不是取决于「模型是树还是线性」。
2. **`drop` 与 `handle_unknown` 会撞车**：`drop='first'` + `handle_unknown='ignore'` 时，未见类别与「被 drop 的类别」共用同一个全零向量，`inverse_transform` 会把未知还原成被 drop 的那一类（L450 文档自认，实测确认）。上线后「未知类别被预测成某一已知类」的机制就在这里。
3. **`np.nan` 与 `None` 是两个类别**（L463–L489，实测 `categories_ = ['Firefox','Safari',None,nan]`）。所以「用 one-hot 处理缺失」会让缺失变成**两个**额外列，而不是一个。
4. **目标编码的泄漏是分层的**：`fit_transform`（折外）**好于** `fit().transform()`（+0.055），但**把 `fit_transform` 放在外层 CV 之外仍然泄漏**（实测 0.5869 vs 正确写法 0.5795）——因为外层 CV 的测试折也被算进了编码的统计量。这与 [[cross-validation-design-and-tuning]] 第五节「任何吃 $y$ 的步骤都必须在折内」是同一条规则的两个实例。

**低频聚合的两个补充**：`min_frequency`/`max_categories` 让 one-hot 的列数可控（实测 `max_categories=mc` 的 one-hot 输出恰好 `mc` 列），序数侧则是 `max_categories + 2`（未知 + 缺失各一码，L513，实测码 `[2,0,1,2,3,4]`）。并列时的取舍**与文档说明相反**（实测保留字典序更大的类别），见 [[sklearn-preprocessing]] 分歧 3。

## 离散化与展开：用列数买非线性

四个对象的成本与形状完全不同：

| 对象 | 输出列（每特征） | 成本 | 保序？ | 外推 |
| --- | --- | --- | --- | --- |
| `Binarizer` | 1 | $O(n)$ | 是（二值） | 恒为边界值 |
| `KBinsDiscretizer` | `n_bins`（one-hot）或 1（ordinal） | $O(n\log n)$（quantile 需排序） | **否**（箱内不可分） | 落进边界箱（实测静默） |
| `PolynomialFeatures` | $\binom{d+\text{deg}}{\text{deg}}$（20 维 2 阶 = 231；3 阶 = 1771） | $O(nd^{\text{deg}})$ | 是（单调单变量） | **发散** |
| `SplineTransformer` | `n_knots + degree - 1`（默认 5+3−1 = 7） | $O(nd\cdot(\text{knots}))$ | 是（每段单调） | 默认钉在边界（`'constant'`） |

实测（$y=\sin x$ + 噪声、800 样本、5 折 CV $R^2$）：线性 **0.6787** → 10 箱分箱 **0.9552** → 3 阶多项式 **0.9754** → 6 结点三次样条 **0.9798**。四条读数说明三件事：

1. **分箱是最便宜的「买到大部分非线性」的方式**（+0.28），代价是阶梯状预测（不可微、外推退化）。
2. **样条几乎免费地拿到多项式的一切**（0.9798 vs 0.9754），列数少一个数量级（20 维：140 vs 1771），且条件数好（L789 的带状结构）。
3. **多项式的正当用途**不是「加非线性」（样条更好），而是「显式要交互项」（`interaction_only=True`）与「要核方法的显式对应物」（L773：多项式核隐式地做同一件事）。

两条等价关系值得单独记住，因为它们把四个对象串成一个谱：`SplineTransformer(degree=0, n_knots=k)` ≡ `KBinsDiscretizer(n_bins=k-1, encode='onehot-dense')`（L814，实测逐位相同）；`Binarizer(threshold=t)` ≡ `KBinsDiscretizer(n_bins=2)` 且边界在 $t$（L725，实测成立）。

## 稀疏：一条硬约束，影响「中心化」的全部选项

稀疏矩阵存的是**非零项的位置**，中心化（减均值）会把每个零变成非零，从而**破坏稀疏结构**并把内存炸掉（L152：「would destroy the sparseness structure … rarely is a sensible thing to do」，L154 说实现会直接 `ValueError` 以避免静默分配过多内存）。

合法选项只有三个（L154–L158）：

1. `MaxAbsScaler`——「specifically designed for scaling sparse data, and is the recommended way」（除以每列最大绝对值，零保持零）。
2. `StandardScaler(with_mean=False)`——只缩放不中心化。
3. 显式 `toarray()`——只有在「中心化后仍然足够小」时才合理。

`RobustScaler` 是半合法状态：**不能 `fit` 稀疏**（`ValueError: Cannot center sparse matrices: use 'with_centering=False' instead.`），但**可以 `transform` 稀疏**（实测通过）。因此它不能进管线。

旁注：任何非 CSR/CSC 的稀疏输入会被**静默转成 CSR**（L156），所以「避免内存拷贝」需要上游就用 CSR/CSC。

## 泄漏的结构：吃不吃 $y$

| 变换 | 状态里含 $y$？ | 在 CV 外先做，代价 | 证据 |
| --- | --- | --- | --- |
| 缩放器 / 分位数 / 幂变换 / 分箱 / 样条 | 否 | **≤0.02，方向不定** | C=1000: 0.8833 vs 0.8667；C=1: 0.8917 vs 0.8917；C=0.01: 0.925 vs 0.9333 |
| 类别编码（Ordinal/One-hot） | 否 | 同上 | — |
| `TargetEncoder` | **是** | **+0.055**（CV 外先编码 0.6414 vs 管线内 0.5795） | [[sklearn-preprocessing]] 实测 30 |
| 特征选择（`SelectKBest`） | 是 | **+0.39**（纯噪声上 0.900 vs 0.508） | [[sklearn-feature-selection]] 实测 14 |

**本库综合**：这张表把「预处理必须放进管线」这条教条拆成三档。**「无监督变换的泄漏是二阶效应，监督变换的泄漏是一阶效应」**——前者只改变 $X$ 的分布估计（$n$ 越大越接近全量，代价越小），后者直接把 $y$ 的信息搬进特征。所以工程上的优先级是：先保证**所有吃 $y$ 的步骤**在折内，再考虑无监督步骤（后者只是「为了正确」，收益通常小于 1 个标准差）。

## 失效模式

| # | 失效模式 | 症状 | 机制 | 出处 |
| --- | --- | --- | --- | --- |
| 1 | 在 CV 外 `fit` 缩放器 | 分数略变（±0.02，方向不定） | 列统计量含测试折 | 实测 40 |
| 2 | 在 CV 外 `fit` 目标编码器 | 分数虚高 +0.05 以上 | 状态含 $y$ | 实测 30 |
| 3 | `TargetEncoder` 用 `fit().transform()` 代替 `fit_transform` | 训练集编码与推理编码不一致 | `fit` 不走交叉拟合 | L605–L614 |
| 4 | `StandardScaler` 喂稀疏输入 | `ValueError: Cannot center sparse matrices` | 中心化破坏稀疏 | L154 |
| 5 | `RobustScaler` 放进管线做 `fit` | `ValueError`（同上） | 它要中心化 | 实测 4 |
| 6 | 缩放后再 `VarianceThreshold` | 一个特征都不删 | 常数列 `scale_` 被替换为 1 | 实测 2 |
| 7 | `MinMaxScaler` 上线后越界 | 输出 `-1.5`/`1.6667` 之类 | `clip=False` 默认 | L105–L106 |
| 8 | `QuantileTransformer` 上线后异常值全同 | 输出全为 `0` 或 `1` | 训练范围外退化 | 实测 9 |
| 9 | `QuantileTransformer` 在小数据上 | `n_quantiles_` = 样本数（少一个数量级） | `n_quantiles` 被夹住 | 实测 7 |
| 10 | `PowerTransformer(method='box-cox')` 遇零/负 | `ValueError: strictly positive data` | Box-Cox 的定义域 | 实测 11 |
| 11 | 用了 Box-Cox 却按 λ 解释 | λ 与输出尺度都对不上 | `standardize=True` 默认 | 实测 12 |
| 12 | `OrdinalEncoder` 用在无序类别上 | 分数暴跌（0.9845 → 0.2236） | 强加字典序 | 实测 35 |
| 13 | 类别改名/重排 | 分数变化（0.2236 → 0.0236） | 同上 | 实测 35 |
| 14 | `OneHotEncoder` + `drop` + 未知类别 | 未知与「被 drop 的类」不可区分 | 都映射为全零 | L450、实测 16 |
| 15 | `OneHotEncoder` 训练集没见过的类别 | `ValueError: Found unknown categories` | `handle_unknown='error'` 默认 | 实测 17 |
| 16 | 高基数 one-hot | 列数爆炸、训练变慢 | 每类一列 | 实测 42（120 值 → 120 列） |
| 17 | `TargetEncoder` 遇未见类别 | 全部落到 `target_mean_` | 定义如此 | L619 |
| 18 | `KBinsDiscretizer` 默认参数 | 输出是 `csr_matrix`，下游报错 | `encode='onehot'` 默认 | 实测 21 |
| 19 | `KBinsDiscretizer` 的边界跨版本 | 同一代码得到不同 `bin_edges_` | `quantile_method` 1.9 改默认 | 实测 20 |
| 20 | `PolynomialFeatures(3)` 在中等维数上 | 列数 1771（20 维） | 组合爆炸 | 实测 25 |
| 21 | 样条外推 | 预测被钉在边界（不是线性） | `extrapolation='constant'` 默认 | 实测 26 |
| 22 | 样条遇 NaN | `ValueError`（1.8+ 有 `handle_missing`） | 默认 `'error'` | 实测 28 |
| 23 | `FunctionTransformer` 默认参数 | NaN 静默穿过 | `validate=False` 默认 | 实测 29 |
| 24 | 用 `normalize` 代替 `scale` | 变成逐行归一化 | `axis=1` vs `axis=0` | 实测 13 |
| 25 | `MaxAbsScaler` 在旧版本传 `clip` | `TypeError` | 1.8 才有该参数 | 外部核查 |
| 26 | `TargetEncoder(shuffle=...)` 在 1.9 上 | `FutureWarning`；1.11 移除 | 弃用 | 实测 34 |

## 与其他概念页的关系

- **与 [[feature-selection]]**：两者都在拟合前改矩阵，但选择**删列**、预处理**改数**。唯一的交叉是 `VarianceThreshold`——它必须排在缩放**之前**（否则常数列被判为方差 1）。另有共同结论：**两者都吃不出新信息**，收益必须来自「模型假设被修正」。
- **与 [[regularization-and-sparsity]]**：缩放 = 改每个特征的有效惩罚强度。这是「同一个 `alpha` 在不同缩放下的含义不同」的根因，也是「标准化后再比较系数」的前提。
- **与 [[cross-validation]] / [[leaderboard-vs-cv]]**：预处理是「泄漏预算」的主要来源。本页那张「吃不吃 $y$」的表给出三档代价（≤0.02 / +0.055 / +0.39），可直接用于判断一个泄漏可疑的分数差。
- **与 [[nearest-neighbors]] / [[support-vector-machines]]**：两者都以距离/内积为核心，因此缩放对它们是一阶的（实测 0.7767 → 0.9833）。
- **与 [[decision-trees]] / [[ensembles]]**：树对**单调**变换免疫（轴对齐切分只依赖序），因此缩放、分位数变换、幂变换对它们无效（但 one-hot 与目标编码有效，因为改变的是划分的依据）。
- **与 [[matrix-factorization]]**：分位数/幂变换之后再 PCA 与直接 PCA 不是同一件事——前者改变了协方差结构（Pearson 0.891→0.754）。所以「先中心化再 PCA」与「先分位数变换再 PCA」是两种不同的模型。
- **与 [[evaluation-metrics]]**：`FunctionTransformer(validate=False)` 与 `error_score=np.nan` 是两个独立的「静默 nan」来源。

## 待解问题

1. **「无监督变换的泄漏」在什么条件下会变大**？实测只覆盖了 120×20 的小数据与逻辑回归三种 `C`。$n$ 很小、列很多、模型非线性时，`StandardScaler` 的泄漏代价是否会超过 0.05？未做系统实验。
2. **`TargetEncoder` 的收缩强度**（`smooth='auto'`）与 `n_categories`/`n_samples` 的定量关系未知：剪藏的公式在丢失的折叠块里，本页只能定性说「小类向全局均值收缩」。
3. **序数编码的「合法序」如何自动判定**？本页给的是「效应是否单调」这个定义，但没有可执行的检验（例如用一维单调回归的残差判断）。
4. **分位数变换之后还能用线性模型解释系数吗**？变换后的系数是「对秩的效应」，不是「对原值的效应」（[[linear-models-as-baselines]] 的可解释性论述在这里需要一条限制）。
5. **样条与多项式的「外推」孰优**未做实验（本页只记录了默认行为）。
6. **`KBinsDiscretizer(strategy='kmeans')` 的稳定性**：1.6.1 上不警告、`random_state=None`，重复运行是否稳定？未测。
7. **`Normalizer` 与 `StandardScaler` 的顺序**：先按行归一化再按列标准化，与反过来，在文本 + 数值混合数据上差多少？未测。
8. **`PolynomialFeatures` 之后做特征选择**（[[feature-selection]]）是否比直接用样条更划算？两条路都能加非线性，缺一个并列的成本/效果实验。

## 关联页面

- 来源：[[sklearn-preprocessing]]（本页所有引用的出处）、[[sklearn-feature-selection]]、[[sklearn-linear-models]]、[[sklearn-nearest-neighbors]]
- 实践：[[preprocessing-pipeline-design]]（本页的落地流程：顺序、默认值、诊断信号、18 条坑）
- 概念：[[regularization-and-sparsity]]、[[feature-selection]]、[[cross-validation]]、[[decision-trees]]、[[ensembles]]、[[nearest-neighbors]]、[[support-vector-machines]]、[[matrix-factorization]]、[[evaluation-metrics]]、[[leaderboard-vs-cv]]
- 工具：[[scikit-learn|scikit-learn 工具页]]（变换器矩阵与默认值清单）
