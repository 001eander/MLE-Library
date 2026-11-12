---
title: "scikit-learn 用户指南 8.3「Preprocessing data」：缩放、分位数/幂变换、逐样本归一化、类别编码、离散化、多项式/样条展开与自定义变换器（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - preprocessing
  - feature-scaling
  - feature-engineering
  - categorical-encoding
  - target-encoding
  - discretization
  - reference-doc
source_path: raw/articles/sklearn/8.3. Preprocessing data.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 用户指南 8.3「Preprocessing data」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 8.3. Preprocessing data（scikit-learn 用户指南第 8.3 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**48 个代码块（47 段 doctest + 1 段公式）**、3 个 `[!note]` 提示框、2 处 `[#]` 折叠标题残留（`Box-Cox transform`、`Support of categorical features with missing values`）、**1 张位图 + 1 张 SVG 图**、0 张表 |
| 原文 | <https://scikit-learn.org/stable/modules/preprocessing.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/8.3. Preprocessing data.md`（843 行、56,686 字节、`wc -w` 5,918 词，其中代码 1,695 词 → 散文 **4,086 词**；**末行无换行**） |
| SHA-256 | `8f4d49a12644cb8dc65b28d986c2bafe6820997d24e058128469e1daad29483d` |
| 完整性 | **11 个 `<details>` 折叠块全丢（1,150 词，其中散文 ≈933）、3 处 `References` 段全丢（含 5 条文献）、1 处页内链接文字被吞成 `(See )`**；官方散文 5,048 词 → 剪藏 4,086 词（**保留 ≈81%**）。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`author`/`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x，可进一步定为 1.9.1**（七条独立证据，见「版本判定」） |
| 插图 | 官方 1.9.1 页 `<article>` 内 2 张：`sphx_glr_plot_map_data_to_normal_001.png`（400×800）、`target_encoder_cross_validation.svg`（472×237）。**两张均已入库** `raw/assets/sklearn-preprocessing/` 并逐张目视核查 |
| 代码 | 覆盖 16 个变换器 + 9 个函数级入口；`StandardScaler` 与 `Pipeline` 两段 doctest 有数值输出，其余多为「形状/编码值」级输出 |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-preprocessing]], L30`。凡超出剪藏内容的部分（API 默认值、版本史、运行时行为、丢失折叠块的文字）都显式标注「外部核查」「官方 1.9.1 页」或「本机实测」。**本次是本库第一次在目标版本 1.9.1 上实测**（此前十七篇都只能停在 1.6.1），因此「实测」条目分两组并标版本。

> [!warning] 本文件未被修改
> 未取得所有者对 `raw/` 回填的授权，剪藏保持抓取时的原始字节（sha256 `8f4d49a1…`）。丢失的 11 个折叠块只在「转换损失」一节里按官方 1.9.1 页逐块记录，**未写回原文件**。所有行号指向原始剪藏。另：`raw/assets/sklearn-preprocessing/` 下两张图是本次从 `https://scikit-learn.org/1.9/_images/` 新抓的（HTTP 200），属于新增资产而非对原文件的修改。

## 概要

这一节是本库**第一个「对列本身动手」的来源**：前面十七个 sklearn 章节都在回答「用哪个模型、怎么拟合、怎么评估」，8.3 回答的是**模型看到的那张矩阵长什么样**。对本库的意义有六层。

1. **它把「预处理」从一组技巧变成一个有状态的契约**。除了 `Normalizer`/`Binarizer`/`FunctionTransformer` 三个无状态变换器（外加只记录 `powers_` 的 `PolynomialFeatures`），其余 12 个都在 `fit` 时**学一个量**（`mean_`/`scale_`/`min_`/`center_`/`quantiles_`/`lambdas_`/`categories_`/`bin_edges_`/`knots`/`encodings_`），`transform` 只是把这个量套到新数据上。整节因此可以压缩成一句话：**预处理的状态只能从训练集学**——L57 明写 `StandardScaler`「compute the mean and standard deviation on a training set so as to be able to later re-apply the same transformation on the testing set」，L614 对 `TargetEncoder.fit` 说得更重（"discouraged … can introduce data leakage"）。这是本库第一次有来源**自己做**这件事，而不是让读者自己想到（[[cross-validation-design-and-tuning]] 第五节的「必须放进管线」在这里拿到了官方台词）。
2. **它显式声明了「变换不是无害的」**。L20 给出机制：目标函数里的项（RBF 核、$ℓ_1$/$ℓ_2$ 正则项）假设各特征同尺度，否则「可能支配目标函数」。本机实测把这句话变成 ≈21 个百分点：同一份二维数据把第一个特征乘 1000，`SVC()` 的 5 折准确率 **0.7767**，加一层 `StandardScaler` 后 **0.9833**（[[sklearn-preprocessing]], L20）。
3. **它第一次把「保什么、毁什么」写成可选性质**。分位数变换与幂变换是**单调**的（L170："both … are based on monotonic transformations … thus preserve the rank"），所以保秩；L172 立刻补上代价——"does, however, distort correlations and distances"。本机实测（Pearson 0.891 → **0.754**，Spearman 0.754 → **0.754**）第一次让「保秩 / 毁相关」变成一对可同时读出的数（[[feature-preprocessing]] §4）。
4. **它把「类别」这一整类输入接进估计器接口**，并暴露三个互不相同的代价：序数编码零列但**强加顺序**（本机实测：非单调效应下 `Ridge` 的 CV $R^2$ **0.2236** vs one-hot **0.9845**；把一个水平改名就掉到 **0.0236**）、one-hot 不强制顺序但**列数随基数线性增长**（400 个 zip 值 → 400 列）、目标编码列数恒定但**必须折外**（`fit(X,y).transform(X)` 的 in-sample 相关 0.8045 vs `fit_transform` 0.7708，下游 `Ridge` 的 CV $R^2$ 被抬高 **0.6414 vs 0.5869**）。
5. **它提供了本库第一批「用列数买非线性」的正规入口**。二值化、分箱、样条、多项式四项在本库此前只以「等价形式」出现过（分箱 = 直方图、多项式核 = 隐式展开，L773）：实测同一条 $\sin$ 数据上 `Ridge` 的 CV $R^2$ 是 线性 **0.6787** → 10 箱分箱 **0.9552** → 3 阶多项式 **0.9754** → 6 结点三次样条 **0.9798**。
6. **它是本库第一份「接口 100% 保留、论证 0% 保留」的样本**。剪藏里 16 个类名、9 个函数名、47 段 doctest 全在，但**11 个折叠块（含 `KernelCenterer` 的完整数学推导、`TargetEncoder` 的三套收缩公式、Box-Cox/Yeo-Johnson 的公式、稀疏输入说明、两条 FAQ）全部丢失**，`References` 三处全空（5 条文献：Schölkopf 1998、Micci-Barreca 2001、Pargent 2022、Eilers & Marx 1996、Perperoglou 2019）。因此本页的机制层论述**只能引官方 1.9.1 页或本机实测**，不能引剪藏行号——这与 [[sklearn-clustering]]、[[sklearn-metrics]] 的记录方式一致。

## 文档结构

| 章节 | 解决什么问题 | 状态来源（`fit` 学到什么） | 行号 |
| --- | --- | --- | --- |
| 引言 | 把原始特征向量改写成下游估计器更好用的表示；缩放重要性；离群点用稳健缩放器 | — | L10–L12 |
| 8.3.1 Standardization | 减均值除标准差；为什么目标函数需要它 | `mean_`、`scale_`（`with_mean`/`with_std` 可关） | L14–L77 |
| 8.3.1.1 Scaling to a range | `MinMaxScaler`（`feature_range`）、`MaxAbsScaler`（$[-1,1]$，为稀疏与已中心化数据设计） | `min_`/`scale_` ｜ `scale_` | L79–L147 |
| 8.3.1.2 Scaling sparse data | 中心化会破坏稀疏性；`StandardScaler(with_mean=False)` 才行；CSR/CSC 的转换 | —（只有约束） | L150–L158 |
| 8.3.1.3 Scaling with outliers | 稳健替代：`RobustScaler` | `center_`、`scale_`（四分位距） | L160–L162 |
| 8.3.1.4 Centering kernel matrices | `KernelCenterer`：把 Gram 矩阵在特征空间里中心化 | `K_fit_rows_`、`K_fit_all_` | L164–L166 |
| 8.3.2 Non-linear transformation | 分位数 vs 幂变换，两者都单调（保秩） | — | L168–L174 |
| 8.3.2.1 Mapping to a Uniform | `QuantileTransformer`：$G^{-1}(F(X))$，把任何分布压成 $[0,1]$ 均匀 | `quantiles_`、`references_` | L176–L209 |
| 8.3.2.2 Mapping to a Gaussian | `PowerTransformer`（Yeo-Johnson / Box-Cox，λ 由极大似然定）与 `QuantileTransformer(output_distribution='normal')`（截断在 1e-7 分位） | `lambdas_` ｜ `quantiles_` | L211–L252 |
| 8.3.3 Normalization | **逐样本**缩放为单位范数（$ℓ_1$/$ℓ_2$/max）；向量空间模型的前提；L2 = spatial sign | 无（`fit` 是 no-op） | L254–L296 |
| 8.3.4 Encoding categorical features | `OrdinalEncoder`（整数码，强加顺序）、缺失值直通、`encoded_missing_value`；`OneHotEncoder`（`categories_`、`drop`、`handle_unknown`、`np.nan` 与 `None` 是两个类别） | `categories_`、`infrequent_categories_` | L298–L490 |
| 8.3.4.1 Infrequent categories | `min_frequency` / `max_categories`：把低频类别聚成一列；`max_categories + 2` 个码的算术 | `infrequent_categories_` | L491–L599 |
| 8.3.4.2 Target Encoder | 用类内目标均值编码高基数类别；`fit_transform` 走交叉拟合，`fit` 不走（因此 docs 明确劝阻） | `encodings_`、`target_mean_`、`target_type_` | L601–L626 |
| 8.3.5 Discretization | 把连续量切成离散值（= 直方图但不计数） | — | L628–L632 |
| 8.3.5.1 K-bins | `KBinsDiscretizer`：`strategy` 三选（uniform/quantile/kmeans）、`encode` 三选；分箱边界 | `bin_edges_`、`n_bins_` | L634–L688 |
| 8.3.5.2 Feature binarization | `Binarizer`（`threshold`）、`binarize`；与 $k=2$ 的 `KBinsDiscretizer` 等价 | 无（`fit` no-op） | L690–L725 |
| 8.3.6 Imputation | **只有一句指针**，正文在 `sklearn.impute`（未入库） | — | L727–L729 |
| 8.3.7 Generating polynomial features | 加入非线性特征的两条路：纯多项式与样条 | — | L731–L733 |
| 8.3.7.1 Polynomial features | `PolynomialFeatures`（`degree`、`interaction_only`、`include_bias`） | `powers_`、`n_output_features_` | L735–L775 |
| 8.3.7.2 Spline transformer | `SplineTransformer`：B 样条基，四项相对多项式的优势（稳定性、Runge、外推、带状矩阵） | `knots`、`n_features_out_` | L777–L818 |
| 8.3.8 Custom transformers | `FunctionTransformer`（`validate`、`check_inverse`）；把任意函数塞进 `Pipeline` | 无（`fit` no-op） | L821–L843 |

## 关键主张

### 1. 这一节的对象是「矩阵的表示」，不是「模型」

L10 一句话给出边界：`sklearn.preprocessing` 提供的是「change raw feature vectors into a representation that is more suitable for the downstream estimators」。因此本节**不含任何估计器**、不含任何 `score`、不含任何指标——它是本库第一个**没有 $y$ 也没有分数**的 sklearn 章节（`TargetEncoder` 是唯一吃 $y$ 的对象，且它吃的 $y$ 只用于编码，不用于拟合模型）。

**本库综合**：这条边界解释了为什么本节的「好坏」无法自我评价。整节 0 个指标、0 个 CV 示例，唯一一次出现模型是 L68 的 `make_pipeline(StandardScaler(), LogisticRegression())` 与 `0.96` 这个孤立读数。所以「该不该标准化」的答案不可能来自本节，只能来自「下游是谁」——这是 [[feature-preprocessing]] 全篇的主轴。

### 2. 有状态 vs 无状态：12 : 4

真实的分界线不是「缩放 / 编码 / 离散化」这些名称，而是 **`fit` 有没有学到东西**：

| | `fit` 学到的量 | 变换的作用轴 |
| --- | --- | --- |
| `StandardScaler` | `mean_`、`var_`/`scale_` | 逐列 |
| `MinMaxScaler` | `data_min_`/`data_max_`/`scale_`/`min_` | 逐列 |
| `MaxAbsScaler` | `scale_`（每列最大绝对值） | 逐列 |
| `RobustScaler` | `center_`、`scale_`（分位数） | 逐列 |
| `QuantileTransformer` | `quantiles_`、`references_` | 逐列 |
| `PowerTransformer` | `lambdas_` | 逐列 |
| `KernelCenterer` | `K_fit_rows_`、`K_fit_all_` | 全矩阵（$n\times n$） |
| `OrdinalEncoder` / `OneHotEncoder` | `categories_`（+`infrequent_categories_`） | 逐列（列数可变） |
| `TargetEncoder` | `encodings_`、`target_mean_` | 逐列（吃 $y$） |
| `KBinsDiscretizer` | `bin_edges_` | 逐列（列数可变） |
| `SplineTransformer` | `knots` 的实际位置 | 逐列（列数可变） |
| `PolynomialFeatures` | 无状态但**改变列数**（`powers_` 只是记录） | 列之间（交互项） |
| `Normalizer` / `Binarizer` / `FunctionTransformer` | **无** | 逐**样本**（前两个） |

**本库综合**：这张表把工程后果分成三类，都已在实测中体现（`PolynomialFeatures` 严格说属于「无状态但改变列数」——它的 `fit` 只从 `degree` 推出 `powers_`，与数据无关）：

- **前六行**的状态是「列统计量」，泄漏只发生在 $X$ 的分布上（不影响 $y$ 与 $X$ 的关系），实测代价是 **±0.02 以内**（同一份数据，`StandardScaler` 先在全量上 `fit` 再 CV vs 放进管线：C=1000 时 0.8833 vs 0.8667、C=1 时 0.8917 vs 0.8917、C=0.01 时 0.925 vs 0.9333 —— **方向都不固定**）。
- **`TargetEncoder`** 的状态里含 $y$，泄漏直接抬高分：`fit().transform()` 编码的训练集上 `Ridge` 的 CV $R^2$ = **0.6414**，`fit_transform` 编码后 = **0.5869**（+0.055，且这还只是 CV 而不是 in-sample）。
- **`Normalizer`/`Binarizer`** 无状态，因此「放进管线」纯粹是为了形状与顺序一致，不是为了防止泄漏（L272 与 L723 都说 `fit` "does nothing"）。

### 3. 缩放：`StandardScaler` 是默认，但它的两个默认假设都是可关的

L77 一句给足接口：「可以直接传 `with_mean=False` 或 `with_std=False` 关掉中心化或缩放」。**外部核查**：`StandardScaler(*, copy=True, with_mean=True, with_std=True)`。这两个开关不是细节：

- `with_mean=False` 是**稀疏输入的唯一活路**。L154 明写 `StandardScaler` 能吃 `scipy.sparse` "as long as `with_mean=False` is explicitly passed … Otherwise a `ValueError` will be raised"。实测（1.6.1 与 1.9.1 相同）：`ValueError: Cannot center sparse matrices: pass 'with_mean=False' instead.`；同一段代码加上 `with_mean=False` 直接通过。
- `with_std=False` 是「只中心化」——这在「所有列本来就是同一量纲、只想消掉截距」时正确，而在**核方法/正则化**里几乎总是错的。

L154 还给了稀疏场景的完整排序：`MaxAbsScaler`「was specifically designed for scaling sparse data, and is the recommended way」；`RobustScaler` **不能 `fit` 稀疏**（实测：`ValueError: Cannot center sparse matrices: use 'with_centering=False' instead.`）但**可以 `transform` 稀疏**（实测通过，输出与稠密一致）。L156 补一条常被忽略的规则：任何非 CSR/CSC 的稀疏输入会被**静默转成 CSR**。

**本库综合**：这三条让「稀疏 + 缩放」只有三种合法组合——`MaxAbsScaler`、`StandardScaler(with_mean=False)`、手工 `toarray()`（L158）。`RobustScaler` 处在半合法状态：可以用于推理，不能用于训练，因此**不能放进 `Pipeline`**（`fit` 会炸）。

### 4. 常数特征：`scale_` 是 1，不是 0

L18 只说「dividing non-constant features by their standard deviation」——**没有说常数列怎么处理**。实测（1.6.1 与 1.9.1 相同）：`[[1,5],[1,3],[1,7]]` 上 `scale_ = [1.0, 1.633]`，即常数列的标准差被**替换为 1**，变换后该列恒为 0。

**本库综合**：这条静默替换让「标准化后没有 nan」成立，代价是**「方差为 0」这个信息在标准化后消失了**——`VarianceThreshold()` 放在 `StandardScaler` 之后将**一个特征都不删**（[[sklearn-feature-selection]] 的本机实测 16 已记录同一现象）。所以 `VarianceThreshold` 必须在缩放**之前**，或者改成看 `nunique`。

### 5. `MinMaxScaler` 默认不裁剪：`clip=False`，测试集可以越界

L105–L106 的 doctest 自己给了证据：训练集在 `[0,1]` 内，测试点 `[-3,-1,4]` 变换后是 `[[-1.5, 0, 1.6667]]` —— **越界**。文档没有评论这件事，只写「the same scaling and shifting operations will be applied to be consistent」。

**外部核查**：`MinMaxScaler(feature_range=(0,1), *, copy=True, clip=False)`。实测 `clip=True` 时同一输入给 `[[0, 0, 1]]`。**`MaxAbsScaler` 的 `clip` 是 1.8 才加的**（1.6.1/1.7 签名只有 `copy`，1.8/1.9 是 `(*, copy=True, clip=False)`）。

**本库综合**：`MinMaxScaler` 是本节唯一「输出区间由训练集极值直接定界」的缩放器（分位数变换也定界，但它不是线性的「缩放器」），因此它也是**最容易随部署数据漂移而静默越界**的一个：训练时 `[0,1]`，线上出现更大的值时输出会跑到 1 之外，下游若假设 `[0,1]`（如某些概率/图像流程）就会静默错。

### 6. 分位数变换：保秩、毁相关、截断两端

L172 把数学说全了：$G^{-1}(F(X))$ 依赖两个事实（$F(X)\sim U[0,1]$、$G^{-1}(U)\sim G$），因此它「smooths out unusual distributions and is less influenced by outliers than scaling methods. It does, however, distort correlations and distances within and across features.」

实测（1.9.1，400 个指数分布样本，$z=x^2+\varepsilon$）：**Pearson 0.891 → 0.754**，**Spearman 0.754 → 0.754**（`StandardScaler` 两者都不变：0.891/0.754）。这是本库第一次把「保秩 / 毁相关」拆成两个可同时读出的数。

三个静默默认值：

1. **`n_quantiles=1000` 会被样本量夹住**。`train_test_split` 后训练集只有 112 行，实测 `UserWarning: n_quantiles (1000) is greater than the total number of samples (112). n_quantiles is set to n_samples.`，`n_quantiles_ = 112`。也就是说「分位数」在小数据上退化成「每个样本一个分位」。
2. **`subsample=10000`（类）vs `subsample=100000`（函数）**。`QuantileTransformer` 与 `quantile_transform` 在 1.6.1 与 1.9.1 上都是这两个不同的默认值——同一算法的两个入口对「用多少样本估计分位」的默认承诺差 10 倍。
3. **两端被截断**。L252 说 normal 输出被裁剪，使输入的最小/最大值（对应 1e-7 与 $1-10^{-7}$ 分位）不变为无穷。实测：均匀输出在训练范围外的输入给 **0 / 1**（不再有区分度），normal 输出给 **±5.1993**。所以「分位数变换把异常值压掉」的准确说法是：**它把训练范围外的一切压成同一个值**。

### 7. 幂变换：λ 是学出来的，`standardize=True` 会再套一层标准化

L215 说 `PowerTransformer` 提供 Yeo-Johnson 与 Box-Cox 两种。剪藏保留了两段 doctest（Box-Cox 的 lognormal 例子，L220–L231），**但公式与三条限制全在丢失的折叠块里**（见「转换损失」）：Box-Cox 只能用于**严格正**数据、λ 由**极大似然**估计、`standardize` 默认 `True` 会再做零均值单位方差。

实测（1.9.1）：

- Box-Cox doctest 逐字复现（`[[0.49, 0.179, -0.156], [-0.051, 0.589, -0.576], [0.694, -0.849, 0.101]]`）。
- 喂含负值的矩阵 → `ValueError: The Box-Cox transformation can only be applied to strictly positive data`（Yeo-Johnson 无此限制）。
- `standardize=True`（默认）输出的均值/标准差是 **0/1**；`standardize=False` 时是 `[-0.155, -0.0216, 0.0451]` / `[0.866, 0.929, 1.004]`。**所以「用 Box-Cox 把数据变高斯」和「用 Box-Cox + 默认参数」不是同一件事**：默认还会顺带标准化，λ 的意义因此不再能单独解释。

**本库综合**：`PowerTransformer` 与 `QuantileTransformer` 都能「变高斯」，但代价完全不同——前者是**参数化的单调变换**（1 个参数/列、可逆、可外推），后者是**非参数的秩映射**（$n$ 个分位点、区间外退化）。剪藏把它们并列在 8.3.2 里（L168），但没给「何时用哪个」的判据，只在 L232 留了一句「某些分布上很有效，另一些上无效，所以要画图看」——这句话的证据就是本页入库的那张图（见「图表核查」）。

### 8. 归一化是逐**样本**的，方向与缩放相反

L256 的定义与第 2 条那张表的关键区别：`Normalizer` 缩放的是**每一行**，不是每一列。因此：

- 它**不携带任何训练集状态**（实测 `fit` 后只有 `n_features_in_`），L272 明写 "`fit` method is useless in this case: the class is stateless as this operation treats samples independently"。
- 它**不改变样本之间的夹角**（对 $ℓ_2$ 而言，这正是「余弦相似度」的来源），但**抹掉样本的长度**——全零行保持全零（实测）。
- `normalize(X, norm='l2', *, axis=1, ...)` 的 `axis` 默认是 **1**（按行），与 `scale(...)` 的 `axis=0`（按列）相反。**"`normalize` 与 `scale` 不是同一个轴上的操作"是本页最容易踩的一条**（实测 `normalize(X, axis=0)` 会按列归一，结果与默认完全不同）。

L296 一句小注被剪藏保留了下来：「L2 normalization is also known as spatial sign preprocessing」——这是本库第一次把「方向 × 长度」的分解写成术语对应。

### 9. 类别编码：三个对象的代价不同，且 `drop` 会与 `handle_unknown` 撞车

剪藏按「先整数码、再 one-hot、再低频聚合、再目标编码」的顺序展开（L298–L626），但**没有把它们并列比较**。本库按代价重排：

| | 列数 | 强加顺序？ | 能吃未见类别？ | 吃 $y$？ |
| --- | --- | --- | --- | --- |
| `OrdinalEncoder` | 1 | **是**（按 `categories_` 的字典序，L315 + 实测 `['female','male']`） | `handle_unknown='use_encoded_value'` + `unknown_value` | 否 |
| `OneHotEncoder` | $n_{\text{categories}}$（`drop` 后 −1） | 否 | `'ignore'` → 全零；`'infrequent_if_exist'` → 归入低频列 | 否 |
| `TargetEncoder` | 1（多类时 $n_{\text{classes}}$） | 否 | 用 `target_mean_`（实测 0.3867） | **是** |

序数编码的代价是可量化的：构造 5 个水平的类别、效应向量 `[3.0, -2.0, 0.5, 2.0, -3.5]`（关于字典序**非单调**），600 个样本，5 折 CV——`OrdinalEncoder` + `Ridge` 的 $R^2$ = **0.2236**，`OneHotEncoder` + `Ridge` = **0.9845**；把其中一个水平从 `'c1'` 改名成 `'aa'`（**数据完全不变**，只变字典序），序数编码的 $R^2$ 掉到 **0.0236**。这条读数第一次证明 L311 那句「would interpret the categories as being ordered」不是风格问题而是**模型假设**。

`drop` 与 `handle_unknown` 的碰撞是本节唯一「文档自己指出陷阱」的地方：L450 明写「unknown categories will have the same mapping as the dropped category」，并给出 `inverse_transform` 的证据（L453–L460：全零行逆变换成 `[['female', None, None]]`，即**未见类别被还原成被丢弃的那个类别**）。实测（1.6.1 与 1.9.1 相同）确认：`drop='first'` + `handle_unknown='ignore'` 时未知类别 → 全零，与「被 drop 的那一类」**不可区分**。

另一个被保留下来的细节是 L463–L489 的折叠块代码：`np.nan` 与 `None` 被当成**两个不同类别**（`categories_ = ['Firefox','Safari',None,nan]`，实测一致）。这意味着「缺失」在 one-hot 里不是缺失，而是两个额外列。

### 10. 目标编码：`fit` 与 `fit_transform` 不是同一个变换

L606 的 note 是本节最锋利的一句：「`fit(X, y).transform(X)` does not equal `fit_transform(X, y)`」。L608 解释机制：`fit_transform` 走**交叉拟合**（$k$ 折，每折用它之外的 $k-1$ 折学到的编码），L614 明写 `fit` "does not use any cross fitting … It is discouraged to use this method because it can introduce data leakage … Use `fit_transform` instead"，L616 补一条实现事实：`fit_transform` 里存的 `encodings_` 是**全量训练集**上学到的编码，折内的中间编码「are temporary and not saved」。

实测（1.9.1，300 个样本 / 20 个类别 / 信号 + 噪声）：

- 同一对象上 `fit_transform` 与 `fit().transform()` 的**最大绝对差 0.0519**（1.6.1 上 0.0837——两版都不同，说明折内洗牌/划分细节变过）。
- 与 $y$ 的 in-sample 相关：`fit_transform` **0.7708**、`fit().transform()` **0.8045**。
- 把这个编码当作特征再做 5 折 CV：`fit_transform` **0.5869**、`fit().transform()` **0.6414**、正确写法（`TargetEncoder` 放进管线）**0.5795**。

**本库综合**：这条把「目标编码 = 泄漏」这句口号变成了三层可区分的读数。注意第三个数（0.5795）比第二个（0.6414）低 0.06——**在 CV 之外先做目标编码，即使编码器内部是交叉拟合的，仍然是泄漏**，因为外层 CV 的测试折也被算进了编码的统计量里（[[cross-validation-design-and-tuning]] 第五节的同一句话，这里换成了吃 $y$ 的变换器版本）。

三个版本敏感项（外部核查）：

- `TargetEncoder` 的 `smooth='auto'`（经验贝叶斯，$m=\sigma_i^2/\tau^2$）与三套收缩公式**全在丢失的折叠块里**（见「转换损失」），剪藏只说「uses the target mean conditioned on the categorical feature」（L603）。
- **`shuffle` 与 `random_state` 在 1.9 被弃用**：1.6/1.7/1.8 签名是 `(..., cv=5, shuffle=True, random_state=None)`，1.9.1 是 `(..., cv=5, shuffle='deprecated', random_state='deprecated')`；传非 `'deprecated'` 值时 `fit_transform` 抛 `FutureWarning`（实测原文：「deprecated in version 1.9 and will be removed in version 1.11. Pass a cross-validation generator as `cv` argument」）。
- 代价是**新写法在旧版本上不存在**：1.9.1 接受 `cv=KFold(5, shuffle=True, random_state=1)`（实测通过），1.6.1 直接 `InvalidParameterError: The 'cv' parameter of TargetEncoder must be an int in the range [2, inf)`。所以「用 splitter 代替 shuffle」这条迁移建议**自带版本下界 1.9**。

### 11. 低频聚合：算术 `max_categories + 2`，但并列时的取舍与文档说法相反

L495–L496 给了两个参数的语义（`min_frequency` 是绝对数或比例、`max_categories` 含低频列本身）。L513–L531 的 doctest 是本节**唯一一条「上限算术」**：`max_categories=3` + `unknown_value=3` + `encoded_missing_value=4` 时，未知类别与缺失类别各占一个码，所以「最多 `max_categories + 2` 个整数码」，实测输出 `[2, 0, 1, 2, 3, 4]`（1.6.1 与 1.9.1 相同）。

L592 处有一段**自相矛盾**的说明（1.9.1 官方页同样如此，非剪藏伪影）：「with `max_categories=2`, 'b' and 'c' are infrequent because they have a higher lexicon order」——但紧随其后的代码是 `OneHotEncoder(max_categories=3)`。实测（四个水平，计数 20/10/10/10）：

| `max_categories` | `infrequent_categories_` | `get_feature_names_out()` |
| --- | --- | --- |
| 2 | `['b','c','d']` | `x0_a`, `x0_infrequent_sklearn` |
| 3 | `['b','c']` | `x0_a`, `x0_d`, `x0_infrequent_sklearn` |
| 4 | `['b']` | `x0_a`, `x0_c`, `x0_d`, `x0_infrequent_sklearn` |

即：**示例只在 `max_categories=3` 时成立**（正文写成 2 是错的），且并列时被聚成低频的是**字典序更小的那些**（`b`、`c`），保留的是**字典序更大的 `d`**——与正文「because they have a higher lexicon order」的因果完全相反。这条属于**来源论述与实现冲突**，见「分歧」第 4 条。

### 12. 分箱：默认 `encode='onehot'` 返回稀疏矩阵，且 1.9 的边界与文档印出的边界不一致

L645 说默认输出是 one-hot 稀疏矩阵，实测 `type(...) = csr_matrix`（1.6.1 与 1.9.1 相同）。**外部核查**：`KBinsDiscretizer(n_bins=5, *, encode='onehot', strategy='quantile', quantile_method=<随版本>, dtype=None, subsample=200000, random_state=None)`——正文从未说 `strategy` 的默认值，而它是 **`'quantile'`**（不是 `'uniform'`）。

两处与版本有关的读数：

1. **剪藏印出的分箱区间已经是旧口径**。L647–L649 写 `feature 1: [-∞,-1), [-1,2), [2,∞)`。**在 1.6.1 上实测 `bin_edges_ = [-3, -1, 2, 6]`，与印出的区间一致**；**在 1.9.1 上实测 `bin_edges_ = [-3, -1.5, 3, 6]`**，即区间应为 `[-∞,-1.5), [-1.5,3), [3,∞)`。原因是 1.9 把 `quantile_method` 的默认值从 `'warn'`（1.7/1.8）改成 `'averaged_inverted_cdf'`（外部核查三版签名）。**但 doctest 的 `transform` 输出仍然一致**（`[[0,1,1],[1,1,1],[2,0,0]]`）——因为这份玩具数据的三行在两种边界下落到同样的箱。所以「doctest 通过」不等于「解释文字正确」，这是本库第一个 **1.9.1 官方页与 1.9.1 运行时不一致**的实例。
2. **越界输入静默落进边界箱**。实测 `[[-100, 100, 100]]` → `[[0, 1, 1]]`（1.6.1）/ `[[0, 2, 2]]`（1.9.1，受上面边界变化影响），**不报错、不警告**。分箱是本节唯一「训练范围外的值仍能得到有意义输出」的变换（因为最外两箱是半开的）。

`strategy='kmeans'` 在 1.6.1 上按列独立跑 k-means，实测**没有**发出任何「结果不保证可复现」的警告；`random_state` 默认 `None`，`subsample=200000` 是它的上限。

### 13. 多项式与样条：两条「用列数买非线性」的路，成本差两个数量级

`PolynomialFeatures(2)` 在 20 个特征上给 **231** 列（含偏置），`degree=3` 给 **1771** 列；`interaction_only=True` 时是 **211** / **1351**。这是本节唯一**指数爆炸**的对象，而正文只说「can get features' high-order and interaction terms」（L737），完全没提列数。

`SplineTransformer` 的成本是**线性**的：每个特征输出 `n_knots + degree - 1` 列（默认 `n_knots=5, degree=3` → **7** 列/特征，实测 `n_features_out_ = 7`）。L786–L789 给了四条相对多项式的优势（低阶 + 加结点更稳、无 Runge 振荡、可控外推、带状矩阵条件数小），其中**只有最后一条是本节能验证的**：实测 `degree=2, n_knots=3` 的输出矩阵（L803–L812）确实带状（每行只有 3 个连续非零），而 `PolynomialFeatures` 的 `powers_` 在 20 维 3 阶时是 1771 行的稠密基。

两条等价关系被剪藏保留下来且实测成立：

- L814：`SplineTransformer(degree=0)` ≡ `KBinsDiscretizer(encode='onehot-dense', n_bins=n_knots-1)`。实测 `X=[0..4]`、`n_knots=4` 与 `n_bins=3` 的输出**逐位相同**。
- L725：`Binarizer` ≡ $k=2$ 且分箱边界在 `threshold` 的 `KBinsDiscretizer`。实测 `Binarizer(threshold=1.1)` 与手写的「$>1.1$」一致，且**恰好等于阈值时为 0**（`binarize([[0,1,1.1]], threshold=1)` → `[[0,0,1]]`）。

**本库综合**：`extrapolation` 的默认值是 `'constant'`，但**不是「外推为 0」**——实测训练范围外 `X=10` 给 `[0,0,0,0,0.167,0.667,0.167]`（边界处的基函数值被复制），`X=-10` 给对称的另一端。这说明「样条不会像多项式那样飞出边界」的正确理解是**输出被钉在边界值上**，因此外推区间的预测是常数而不是线性——`extrapolation='linear'` 才给线性外推（实测 `X=10` → `[0,0,0,-2.083,0.667,2.417]`）。

### 14. 自定义变换器：`validate=False` 是默认，NaN 会静默穿过

L823 说 `FunctionTransformer` 可以把任意函数变成变换器，L827–L834 的 doctest 用 `np.log1p`（`validate=True`）。L836 说可以用 `check_inverse=True` 校验 `func`/`inverse_func` 互逆。

实测（1.6.1 与 1.9.1 相同）：

- **`validate` 默认 `False`**：`FunctionTransformer(np.log1p).transform([[0, nan]])` → `[[0, nan]]`，**NaN 直接穿过**，不报错。
- `validate=True` 时 → `ValueError: Input X contains NaN. FunctionTransformer does not accept missing values…`。注意这条错误是「拒绝缺失值」，**不是**「拒绝非正数」——负值只有在函数内部才炸（`np.log1p(-5)` 给 `nan` + `RuntimeWarning`）。
- **`check_inverse=True` 是默认值**，但只有同时给了 `inverse_func` 且调用过 `fit` 才会检查；实测「无 `inverse_func`」与「精确互逆」都不发警告。

**本库综合**：`validate=False` 让 `FunctionTransformer` 成为整节**唯一能静默传播 NaN/Inf 的入口**——它是 `Pipeline` 里的「逃生舱」，也是「我的指标怎么会是 nan」的一条常见来源（与 [[evaluation-metrics]] 的 `error_score=np.nan` 是两个独立的 nan 来源）。

## 版本判定

结论：**1.9.x，可进一步定为 1.9.1**。七条独立证据（把剪藏逐句与官方 1.6/1.7/1.8/1.9 四版页比对）：

| # | 剪藏 | 官方页差异 | 排除的版本 |
| --- | --- | --- | --- |
| 1 | 标题与全部小节编号都是 **8.3.x**（L14、L79、L150…） | 1.6/1.7 是 **6.3.x**，1.8 是 **7.3.x** | 1.6、1.7、1.8 |
| 2 | L610「**This** diagram shows the cross fitting scheme」 | 1.7 是「**The following** diagram shows」 | 1.7 |
| 3 | L608「**For this reason, training data should always be trained and transformed with** `fit_transform(X_train, y_train)`」 | 1.7 无此句（1.7 的对应位置是「The following diagram shows…」） | 1.7 |
| 4 | L614「**It is discouraged to use this method** because it can introduce data leakage」 | 1.7 是「learns one encoding on the entire training set, **which is used to encode categories in `transform`**」 | 1.7 |
| 5 | L616「The intermediate encodings learned for each fold … **are temporary and not saved**」 | 1.7 是「Note that the encodings learned for each fold … are not saved to an attribute」 | 1.7 |
| 6 | L681 `Categories (5, **str**)` | 1.8 与更早是 `Categories (5, **object**)`（pandas 2→3 的 repr 变化） | 1.8 |
| 7 | L12「**are** highlighted」、L83「The motivation … **includes**」（1.6 分别是「is highlighted」「include」） | 1.6 的旧措辞 | 1.6 |

补充：**抓取日 `/stable/modules/preprocessing.html` 与 `/1.9/modules/preprocessing.html` 字节级相同**（205,574 B，sha256 `efe09a2fc1ff719716078e77cebc0620abe3a1d899f51ee0c90eb60660915c89`），页标题 `scikit-learn 1.9.1 documentation`，故上限取 1.9.1。

非判别性证据：133 条 ≥14 词的散文句在 1.6/1.7/1.8/1.9 四版页的命中数是 **118/120/124/124**。1.9 上未命中的 9 条逐条核为 Markdown 伪影（2 条以 `- ` 开头的列表项、4 条 `\(…\)` vs `$\…$` 的 LaTeX 差异、1 条被引成 `>` 的 note、1 条 `(See )` 的链接文字丢失、1 条 `φ\left(\cdot\right)` 转义），**无正文缺失**。

**注意**：本节有 4 条功能性结论**依赖版本**，不能跨版本引用——`KBinsDiscretizer` 的分箱边界（1.9 起 `quantile_method` 默认值变了，实测边界 $-1,2$ → $-1.5,3$）、`TargetEncoder` 的 `shuffle`/`random_state`（1.9 弃用）、`MaxAbsScaler.clip`（1.8 新增）、`SplineTransformer.handle_missing`（1.8 新增）。

## 转换损失

官方 1.9.1 页本节**散文 5,048 词**（去掉 `<pre>` 后）→ 剪藏散文 **4,086 词**，保留 ≈**81%**。丢的 962 词几乎全部集中在 **11 个 `<details>` 折叠块（1,150 词，其中散文 ≈933）与 3 处 `References`（5 条文献）**：

| 丢失项 | 形态 | 词数（其中代码） | 内容要点（据官方 1.9.1 页） |
| --- | --- | --- | --- |
| 8.3.1 *References* | `<details>` | 22 | 只一句：中心化/缩放的重要性见 FAQ「Should I normalize/standardize/rescale the data?」 |
| 8.3.1 *Scaling vs Whitening* | `<details>` | 50 | 逐列中心化+缩放不足以保证线性无关；要再用 `PCA(whiten=True)` 去掉线性相关 |
| 8.3.1.4 *Mathematical formulation* | `<details>` | 266 | `KernelCenterer` 的全部数学：$K(X,X)=\phi(X)\phi(X)^T$、$\tilde K=K-1K-K1+1K1$、测试集 $\tilde K_{test}$ 的对应式、$1_{n_{samples}}$ 的含义；引 Schölkopf 1998 附录 B |
| 8.3.2.2 *Yeo-Johnson transform* | `<details>` | 73 | 分段公式（$\lambda\neq0/2$ × $x_i\ge0$）——**剪藏里连公式都没有** |
| 8.3.2.2 *Box-Cox transform* | `<details>` | 163（71） | Box-Cox 公式、「**只能用于严格正数据**」、λ 由**极大似然**定、以及「`standardize=True` 默认会再做零均值单位方差」这句关键说明 |
| 8.3.3 *Sparse input* | `<details>` | 55 | `normalize`/`Normalizer` 接受 `scipy.sparse`，内部转 CSR、走 Cython；建议上游就用 CSR |
| 8.3.4 *Support of categorical features with missing values* | `<details>` | 202（146） | 「缺失值被当作**额外类别**」、「`np.nan` 与 `None` 是**两个**类别」、「见 `Loading features from dicts`」——**三段散文全丢，只留下两段代码**（L465–L489） |
| 8.3.4.2 *Binary classification targets* | `<details>` | 150 | $S_i=\lambda_i\frac{n_{iY}}{n_i}+(1-\lambda_i)\frac{n_Y}{n}$、收缩因子 $\lambda_i=\frac{n_i}{m+n_i}$、`smooth='auto'` 的经验贝叶斯 $m=\sigma_i^2/\tau^2$ |
| 8.3.4.2 *Multiclass classification targets* | `<details>` | 81 | 每类一个编码 $S_{ij}$（因此 $n_{\text{features}}\times n_{\text{classes}}$ 列） |
| 8.3.4.2 *Continuous targets* | `<details>` | 42 | 回归目标的同一形式 |
| 8.3.4.2 *References* | 段 | 55 | [MIC] Micci-Barreca 2001、[PAR] Pargent et al. 2022 ——**剪藏只剩一个空的 `References` 标题（L626）** |
| 8.3.5.2 *Sparse input* | `<details>` | 45 | `binarize`/`Binarizer` 同样接受稀疏、内部转 CSR |
| 8.3.7.2 *References* | 段 | 44 | Eilers & Marx 1996（B 样条 + 惩罚）、Perperoglou et al. 2019（R 的样条综述）——**剪藏里连标题都没有** |

**两处纯文本损失**：

1. **L645 的页内链接文字被吞**：官方是「By default the output is one-hot encoded into a sparse matrix (**See Encoding categorical features**)」，剪藏变成「(See )」。这使「默认 one-hot」这句在剪藏里**指向空处**，也让「稀疏输出去哪了」这个问题失去入口。
2. **`max_categories` 并列取舍的那段说明与实现相反**（见「关键主张 11」）——这条**不是剪藏伪影**，官方 1.9.1 页同样如此，属于来源自身的错误。

结构性后果（本库判断）：

- 丢掉的 11 块恰好覆盖**四类「为什么」**：为什么稀疏不能中心化（8.3.3）、为什么 λ 这样定（8.3.2.2）、为什么目标编码要收缩（8.3.4.2）、为什么核矩阵要中心化（8.3.1.4）。因此剪藏在**接口层 100% 完整**（16 个类、9 个函数、47 段 doctest），而**机制层几乎为零**：本页所有「为什么」都必须引官方 1.9.1 页或本机实测。
- 三处 `References` 全空意味着本节**没有任何文献支撑**——「目标编码优于 one-hot 的高基数场景」（[PAR]）、「B 样条优于多项式」（Eilers & Marx）这两条都是**有文献的论断**，剪藏把它们降级成了断言。
- 对 AutoResearch 的影响：读剪藏能拿到全部 API 用法与 doctest 期望值，但**拿不到任何公式**。若下游要生成「自定义目标编码收缩」或「核中心化」的代码，剪藏不足以支撑。

公式与文本的转换噪声（不影响语义，但影响搜索）：L166 的 `\(\phi \left(\cdot\right)\)` 被转成 `$\phi \left(\cdot\right)$`；L172 的 `\(F(X)\)` 被转成 `$F \left(X\right)$`；L647–L649 的分箱区间用 `$\left[- \infty , - 1\right)$` 形式（可读）；L784 附近 pandas repr 保留了 `Categories (5, str)`。

## 外部核查（不在剪藏内）

### A. 变换器与函数签名（官方 1.9.1 API 页）

| 对象 | 签名（1.9.1） |
| --- | --- |
| `StandardScaler` | `(*, copy=True, with_mean=True, with_std=True)` |
| `MinMaxScaler` | `(feature_range=(0, 1), *, copy=True, clip=False)` |
| `MaxAbsScaler` | `(*, copy=True, clip=False)`（**1.6.1/1.7 只有 `copy`**） |
| `RobustScaler` | `(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)` |
| `KernelCenterer` | `()`（无参数） |
| `QuantileTransformer` | `(*, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, subsample=10000, random_state=None, copy=True)` |
| `PowerTransformer` | `(method='yeo-johnson', *, standardize=True, copy=True)` |
| `Normalizer` | `(norm='l2', *, copy=True)` |
| `OrdinalEncoder` | `(*, categories='auto', dtype=<float64>, handle_unknown='error', unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)` |
| `OneHotEncoder` | `(*, categories='auto', drop=None, sparse_output=True, dtype=<float64>, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')` |
| `TargetEncoder` | `(categories='auto', target_type='auto', smooth='auto', cv=5, shuffle='deprecated', random_state='deprecated')` |
| `KBinsDiscretizer` | `(n_bins=5, *, encode='onehot', strategy='quantile', quantile_method='averaged_inverted_cdf', dtype=None, subsample=200000, random_state=None)` |
| `Binarizer` | `(*, threshold=0.0, copy=True)` |
| `PolynomialFeatures` | `(degree=2, *, interaction_only=False, include_bias=True, order='C')` |
| `SplineTransformer` | `(n_knots=5, degree=3, *, knots='uniform', extrapolation='constant', include_bias=True, order='C', handle_missing='error', sparse_output=False)` |
| `FunctionTransformer` | `(func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None)` |
| `normalize` | `(X, norm='l2', *, axis=1, copy=True, return_norm=False)` |
| `scale` | `(X, *, axis=0, with_mean=True, with_std=True, copy=True)` |
| `robust_scale` | `(X, *, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, unit_variance=False)` |
| `minmax_scale` / `maxabs_scale` | `(X, feature_range=(0, 1), *, axis=0, copy=True)` / `(X, *, axis=0, copy=True)` |
| `quantile_transform` | `(X, *, axis=0, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, **subsample=100000**, random_state=None, copy=True)` |
| `power_transform` | `(X, method='yeo-johnson', *, standardize=True, copy=True)` |
| `binarize` | `(X, *, threshold=0.0, copy=True)` |
| `add_dummy_feature` | `(X, value=1.0)` |

五条与剪藏直接冲突或补齐的项：

1. **`KBinsDiscretizer.strategy` 默认 `'quantile'`**——正文完全不说默认值（只在 L664 描述三种策略）；默认不是 `'uniform'`。
2. **`QuantileTransformer.subsample=10000` 而 `quantile_transform.subsample=100000`**——同一算法的类入口与函数入口差 10 倍（1.6.1 与 1.9.1 都是如此）。
3. **`SplineTransformer` 的 `knots='uniform'`、`extrapolation='constant'`、`include_bias=True`、`handle_missing='error'`、`sparse_output=False` 五个默认值正文一个都没写**；L788 只说「look at the option `extrapolation`」，没说默认是什么。
4. **`OneHotEncoder.sparse_output=True`**——正文只在 `drop`/`infrequent` 的 doctest 里出现 `sparse_output=False`，从不说明默认返回稀疏矩阵。
5. **`TargetEncoder` 的 `shuffle`/`random_state` 已是 `'deprecated'`**（剪藏是 1.9.1 的正文，但完全没提这件事，也没提 `cv` 现在可以吃 splitter）。

### B. 版本边界（四版 API 页比对）

| 参数 | 1.6 | 1.7 | 1.8 | 1.9 |
| --- | --- | --- | --- | --- |
| `KBinsDiscretizer.quantile_method` | **不存在** | `'warn'` | `'warn'` | `'averaged_inverted_cdf'` |
| `SplineTransformer.handle_missing` | **不存在** | **不存在** | `'error'` | `'error'` |
| `MaxAbsScaler.clip` | **不存在** | **不存在** | `False` | `False` |
| `TargetEncoder.shuffle`/`random_state` | `True`/`None` | `True`/`None` | `True`/`None` | `'deprecated'`/`'deprecated'` |
| `TargetEncoder.cv` 是否吃 splitter | **否**（`InvalidParameterError`） | 未测 | 未测 | **是**（实测通过） |

`quantile_method` 的默认值变更**会改变分箱边界**（实测 1.6.1 `[-3,-1,2,6]` vs 1.9.1 `[-3,-1.5,3,6]`），这是本节最实际的一条版本陷阱：同一段 `KBinsDiscretizer(n_bins=3)` 代码在两个版本上得到**不同的 bin 边界**（本次玩具数据上分箱编号恰好相同，但边界不同；若数据分布不同则编号也可能不同）。

### C. `TargetEncoder` 在 1.9 的弃用细节（源码 `preprocessing/_target_encoder.py` @ 1.9.1）

- 参数约束是 `"shuffle": ["boolean", StrOptions({"deprecated"})]`、`"random_state": ["random_state", StrOptions({"deprecated"})]`。
- 警告**只在 `fit_transform` 里发出**（`fit` 不发），原文：「`TargetEncoder.shuffle` and `TargetEncoder.random_state` are deprecated in version 1.9 and will be removed in version 1.11. Pass a cross-validation generator as `cv` argument to specify the shuffling behaviour instead.」
- 未显式给出时内部等价于 `shuffle=True`、`random_state=None`，并且通过 `check_cv(self.cv, y, classifier=self.target_type_ != "continuous", shuffle=..., random_state=...)` 构造；若 `cv` 已是 splitter，这些 kwargs 会被忽略。

## 本机实测

**第一组：scikit-learn 1.9.1（Python 3.14.7、NumPy 2.5.4）——本次首次在目标版本上复核**；**第二组：1.6.1（Python 3.9、NumPy 2.0.2、SciPy 1.13.1）**。两版并列时写作「1.9.1 / 1.6.1」。

1. **`StandardScaler` doctest 逐字复现**（L30–L52）：`mean_ = [1, 0, 0.333]`、`scale_ = [0.816, 0.816, 1.247]`，变换后 `[[0,-1.225,1.336],[1.225,0,-0.267],[-1.225,1.225,-1.069]]`（文档写作 `0.81`/`1.24` 与 `-0.267`，是同一组数的不同打印精度）。
2. **常数特征的 `scale_` 是 1**：`[[1,5],[1,3],[1,7]]` → `scale_ = [1.0, 1.633]`，该列被变换成恒 0。**文档未提**。
3. **`Pipeline` doctest 复现**（L66–L74）：`pipe.score = 0.96`（1.9.1 与 1.6.1 都是 0.96）。
4. **稀疏中心化的硬失败**：`StandardScaler().fit(csr_matrix(X))` → `ValueError: Cannot center sparse matrices: pass 'with_mean=False' instead.`；`with_mean=False` 通过。`RobustScaler().fit(sparse)` → `ValueError: … use 'with_centering=False' instead.`；但 `RobustScaler().fit(dense).transform(sparse)` **通过**且与稠密结果一致。
5. **`MinMaxScaler` 默认越界、`clip=True` 才截断**：`[-3,-1,4]` → `[[-1.5, 0, 1.6667]]`（默认）vs `[[0, 0, 1]]`（`clip=True`）。与 L105–L106 的 doctest 完全一致。
6. **`MaxAbsScaler`**：`scale_ = [2, 1, 2]`，与 L136–L147 的 doctest 一致；`clip` 参数在 1.9.1 存在、1.6.1 不存在。
7. **`QuantileTransformer` 的 `n_quantiles` 被夹住**：112 行训练集 + 默认 `n_quantiles=1000` → `UserWarning: n_quantiles (1000) is greater than the total number of samples (112). n_quantiles is set to n_samples.`，`n_quantiles_ = 112`（1.9.1 与 1.6.1 相同）。
8. **剪藏的 iris 分位数 doctest 两版都不复现**（L190–L209）：文档印出训练集 `[0.00, 0.24, 0.49, 0.73, 0.99]`、测试集 `[0.01, 0.25, 0.46, 0.60, 0.94]`；实测 1.9.1 是 `[0.0, 0.239, 0.509, 0.743, 1.0]` / `[0.014, 0.25, 0.477, 0.605, 0.941]`，1.6.1 是 `[0.0, 0.24, 0.51, 0.74, 1.0]` / `[0.01, 0.25, 0.48, 0.60, 0.94]`。**差 ≤0.02，机制相同、数值是旧口径**——所以「变换后接近原分位数」这句话成立，打印值不成立。
9. **分位数变换在训练范围外退化**：均匀输出给 `0` / `1`，normal 输出给 `±5.1993`（对应 L252 的 1e-7 截断）。
10. **`PowerTransformer` 的 Box-Cox doctest 逐字复现**（L220–L231）：`[[0.49, 0.179, -0.156], [-0.051, 0.589, -0.576], [0.694, -0.849, 0.101]]`（1.9.1 与 1.6.1 相同）。
11. **Box-Cox 对非正数据硬失败**：`ValueError: The Box-Cox transformation can only be applied to strictly positive data`（1.9.1/1.6.1 相同）；Yeo-Johnson 接受负值。
12. **`standardize=True` 是默认，且它会改变 λ 的读法**：`standardize=True` 输出均值/标准差 `0/1`；`standardize=False` 时是 `[-0.155, -0.0216, 0.0451]` / `[0.866, 0.929, 1.004]`；`lambdas_ = [-0.116, -0.058, -0.022]`（200 个 lognormal 样本）。剪藏**没有**这句话（它在丢失的 Box-Cox 折叠块里）。
13. **`Normalizer` 无状态**：`fit` 后属性只有 `norm`/`copy`/`n_features_in_`；全零行保持全零；`norm='l1'` 与 `'max'` 与文档口径一致；`normalize(axis=0)` 与默认（`axis=1`）**结果不同**（第 12 条主张）。
14. **`OrdinalEncoder` 按字典序编号且默认直通 NaN**：`categories_ = [['female','male'], ['from Europe','from US'], ['uses Firefox','uses Safari']]`，`[['female','from US','uses Safari']]` → `[[0, 1, 1]]`（与 L305–L313 一致）；含 `np.nan` 的列 → `nan` 原样输出；`encoded_missing_value=-1` → `-1`。
15. **`OrdinalEncoder` 对未见类别硬失败**：`ValueError: Found unknown categories ['other'] in column 0 during transform`。`handle_unknown='use_encoded_value'` + `unknown_value=3` 才静默。
16. **`OneHotEncoder` 的 `drop` 与未知类别撞车**（L443–L460 的文档论断被证实）：`drop='first'` + `handle_unknown='ignore'` 时未知类别 → **全零行**，`inverse_transform` 把它还原成**被 drop 的类别**（`[['female', None, None]]`）。
17. **`np.nan` 与 `None` 是两个类别**（L463–L489 的 doctest 复现）：`categories_ = ['Firefox','Safari',None,nan]`，四行分别落在四列。
18. **低频聚合的算术复现**（L513–L531）：`max_categories=3`+`unknown_value=3`+`encoded_missing_value=4` → 码 `[2, 0, 1, 2, 3, 4]`（`a`/`d` 归低频、`b`/`c` 各自一类、未知=3、缺失=4）。
19. **`max_categories` 并列时的取舍与文档相反**（L592，见「关键主张 11」）：计数 20/10/10/10 时 `max_categories=2/3/4` 分别给 `['b','c','d']`/`['b','c']`/`['b']`；**保留的是字典序更大的那个**。
20. **`KBinsDiscretizer` 的边界随版本变、分箱号不变**：`n_bins=[3,2,2]` 时 1.6.1 `bin_edges_ = [-3,-1,2,6]`（与剪藏 L647 印出的区间一致）、1.9.1 `[-3,-1.5,3,6]`；`transform` 输出两版都是 `[[0,1,1],[1,1,1],[2,0,0]]`（与 L651–L657 一致）。`n_bins=3` 时 1.6.1 `[-3,-1,2,6]` / 1.9.1 `[-3,-1.5,3,6]`，第 2、3 列也从 `[3,4.33,5.33,6]`/`[11,13,14.33,15]` 变成 `[3,4,5.5,6]`/`[11,12.5,14.5,15]`。
21. **默认输出是稀疏矩阵**：`KBinsDiscretizer()` 的 `transform` 返回 `csr_matrix`（1.9.1/1.6.1 相同）。
22. **越界输入静默进边界箱**：`[[-100,100,100]]` → `[[0,1,1]]`（1.6.1）/ `[[0,2,2]]`（1.9.1），无警告。
23. **`strategy='kmeans'` 在 1.6.1 上无任何复现性警告**（`random_state=None`、`subsample=200000` 默认）。
24. **`Binarizer` 在阈值处取 0**：`binarize([[0,1,1.1]], threshold=1)` → `[[0,0,1]]`；`Binarizer(threshold=1.1)` 的输出与 L716–L721 一致。
25. **`PolynomialFeatures` 的列数爆炸**：20 个特征 → `degree=2` **231** 列、`degree=2, interaction_only=True` **211**、`degree=3` **1771**、`degree=3, interaction_only=True` **1351**；两段 doctest（L747–L754、L764–L769）逐字复现。
26. **`SplineTransformer` 的默认列数与边界行为**：`degree=2,n_knots=3` → 4 列且 doctest 复现（L803–L812）；默认 `n_knots=5,degree=3` → **7** 列/特征；`X=10` → `[0,0,0,0,0.167,0.667,0.167]`、`X=-10` → `[0.167,0.667,0.167,0,0,0,0]`；`extrapolation='linear'` 时 `X=10` → `[0,0,0,-2.083,0.667,2.417]`。
27. **`degree=0` 的样条 ≡ 分箱**（L814 的论断成立）：`SplineTransformer(degree=0, n_knots=4)` 与 `KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')` 在 `X=[0..4]` 上输出**逐位相同**。
28. **样条对 NaN 默认硬失败（1.8+）**：`SplineTransformer()`（`handle_missing='error'`）→ `ValueError: Input X contains NaN.`；`handle_missing='zeros'` → 全零行（1.9.1 实测；1.6.1 无此参数，同样抛 `ValueError`）。
29. **`FunctionTransformer` 的 `validate=False` 静默放过 NaN**：`[[0, nan]]` → `[[0, nan]]`；`validate=True` → `ValueError: Input X contains NaN…`；两版相同。
30. **`TargetEncoder` 的三种写法差一个量级之外的量**：`fit_transform` vs `fit().transform()` 的最大绝对差 **0.0519（1.9.1）/ 0.0837（1.6.1）**；与 $y$ 的 in-sample 相关 **0.7708 vs 0.8045**；编码后 `Ridge` 的 5 折 CV $R^2$ **0.5869 vs 0.6414**；正确写法（编码器在管线内）**0.5795**。
31. **未见类别 → `target_mean_`**：`transform([['ZZZ']])` → `0.3867`（= `target_mean_`），两版一致。
32. **多类目标编码会成倍加列**：`target_type_` 自动判定（实测 `binary`/`multiclass`/`continuous` 三种）后，`'multiclass'` 时 `encodings_` 的**长度 = 特征数 × 类数**（1 个特征 3 类 → 3 项，2 个特征 3 类 → 6 项），每项形状 `(n_categories,)`，输出列数同样成倍（`(200,3)` / `(200,6)`）；`binary`/`continuous` 时长度 = 特征数、输出 1 列。
33. **`TargetEncoder` 的 `cv` 是否吃 splitter 是 1.9 的新能力**：1.9.1 `TargetEncoder(cv=KFold(5, shuffle=True, random_state=1))` 通过；1.6.1 `InvalidParameterError: The 'cv' parameter of TargetEncoder must be an int in the range [2, inf)`。
34. **`shuffle`/`random_state` 的 `FutureWarning` 只在 `fit_transform` 里发**：1.9.1 传 `shuffle=False` 时 `fit_transform` 抛 FutureWarning、`fit` 不抛（源码与实测一致）。
35. **序数编码「强加顺序」的代价**（本库自建实验，1.9.1）：5 个水平、效应 `[3,-2,0.5,2,-3.5]`、600 样本、5 折 CV——`OrdinalEncoder`+`Ridge` $R^2$ = **0.2236**、`OneHotEncoder`+`Ridge` = **0.9845**；把一个水平 `c1` 改名成 `aa`（**同数据**）后序数编码掉到 **0.0236**。
36. **缩放的重要性**（1.9.1）：二维可分数据、第一列放大 1000 倍——`SVC()` 5 折 **0.7767**、`Pipeline(StandardScaler, SVC)` **0.9833**。
37. **稳健缩放 vs 标准缩放在离群点上**（1.9.1）：95 个正态点 + 5 个偏移点，`StandardScaler` 的 `mean_ = [1.46, 2.56]`、`scale_ = [8.53, 11.03]`；`RobustScaler` 的 `center_ = [-0.03, 0.10]`、`scale_ = [1.29, 1.35]`。
38. **分位数变换保秩不保相关**（1.9.1）：Pearson 0.891 → **0.754**、Spearman 0.754 → **0.754**；`StandardScaler` 两者都是 0.891/0.754。
39. **「用列数买非线性」的四档读数**（1.9.1，$y=\sin x$ + 噪声，800 样本，5 折 CV $R^2$）：线性 `Ridge` **0.6787**、10 箱分箱 **0.9552**、3 阶多项式 **0.9754**、6 结点三次样条 **0.9798**。
40. **缩放器「先在全量上 fit 再 CV」的代价是方向不定的 ±0.02**（1.9.1）：`StandardScaler` 先 fit 全量 vs 放进管线，C=1000 时 0.8833 vs 0.8667、C=1 时 0.8917 vs 0.8917、C=0.01 时 0.925 vs 0.9333。**对比 `TargetEncoder` 的 +0.055（第 30 条）——「泄漏」的严重性取决于变换是否吃 $y$。**
41. **`KernelCenterer` 要求方阵**：3×2 的矩阵 `fit` → `ValueError: Kernel matrix must be a square matrix.`；3×3 通过，中心化后每行和为 0（1e-10 量级），且 `transform` 单行测试数据可用。
42. **`ColumnTransformer` + `TargetEncoder` 能直接跑**（1.9.1/1.6.1）：三路（`StandardScaler` 两个数值列 + `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10)` 一个类别列 + `TargetEncoder(cv=5)` 一个 120 值的 zip 列）+ `Ridge` 的 5 折 CV $R^2$ = **0.9915（1.9.1）/ 0.9916（1.6.1）**；`get_feature_names_out()` 给 7 个名字，`TargetEncoder` 那一路只有 1 列。
43. **`FunctionTransformer(pd.cut)` 的 doctest 复现**（L675–L684）：输出 `['infant','kid','teen','adult','senior citizen']` 与 `Categories (5, str): [...]`（1.9.1 + pandas；这也是版本判定第 6 条的证据）。

**版本警告**：第 20、33、34、2、6、28 条**只对特定版本成立**（分箱边界、`cv` 吃 splitter、`shuffle` 弃用、`MaxAbsScaler.clip`、`handle_missing`）。其余条目的机制在 1.6.1 与 1.9.1 上一致，但数值（尤其 `TargetEncoder` 的折内划分）两版有差异。

## 图表核查

官方 1.9.1 页本节 2 张图，**两张均已入库并逐张目视核查**（2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取，HTTP 200）：

| 文件 | 尺寸 | 字节 | SHA-256 前 16 位 | 内容 |
| --- | --- | --- | --- | --- |
| `sphx_glr_plot_map_data_to_normal_001.png` | 400×800 | 43,080 | `6def530fa4ef7154` | **8 行 × 3 列**的直方图矩阵：第 1 行三种原始分布（Lognormal / Chi-squared / Weibull），第 2–4 行分别是 Box-Cox、Yeo-Johnson、Quantile transform 之后；第 5 行换三种目标分布（Gaussian / Uniform / Bimodal），第 6–8 行重复三种变换 |
| `target_encoder_cross_validation.svg` | 472×237 | 32,077 | `7dcd241c9c2cbb4e` | **5 个面板 + 1 个结果列**：每个面板是一列 5 格，其中 1 格是该折（着色）、4 格写 `Train`；右侧箭头标注 **Combine Folds**，指向一个 5 行的 `Fold1…Fold5` 列 |

逐图读数（只记录真正看到的）：

- **`plot_map_data_to_normal_001.png`**：每格的横轴都是变换后的取值（约 −2.5…2.5），纵轴是计数。第 2 行印出的 λ 是 **0 / 0.27 / 12.59**（Box-Cox 对 lognormal、chi²、Weibull），第 3 行是 **−0.83 / −0.12 / 24.53**（Yeo-Johnson）——**同一种分布上两种变换给出的 λ 差 2 倍以上**，说明 λ 不是可比量。第 4 行「After Quantile transform」三格几乎完全重合（都近似均匀→正态的钟形），即**分位数变换对三种原始分布给出同一种结果**。第 6–8 行：Gaussian 输入经 Box-Cox 得 **λ=0.54**、Uniform 输入得 **λ=0.63** 且**变换后仍是平的（不变成钟形）**、Bimodal 输入得 **λ=1.69** 且**变换后仍是双峰**。这三格正是 L232 那句「某些分布上很有效，另一些上无效」的图证：**幂变换无法把均匀或双峰变成单峰高斯**。
- **`target_encoder_cross_validation.svg`**：右侧结果列的 5 格颜色与 5 个面板中被留出的那一格一一对应（浅蓝/浅绿/浅橙/粉/浅紫），即「每折用自己的 $k-1$ 折学编码、把 $k$ 折的编码拼回完整训练集」这件事。需要两点如实说明：① 该 SVG 用 **31 个 `<foreignObject>`** 承载文字，纯矢量渲染器（本次用 cairosvg）会显示占位符「Text is not SVG - cannot display」，因此箭头标签在图里只读到 **`Combine...`**，完整文字（`Combine Folds`）是从 SVG 源码里读出的，不是从渲染结果读出的；② 图内**没有任何数字**（不写 `cv=5`、不写折大小），所以「默认 `cv=5`」这句话只能引 L610 的文字，不能引这张图。

**未入库的图**：无（官方本节 2 张全部入库）。但官方本节另有 **10 条示例链接**（`plot_scaling_importance`、`plot_all_scaling`、`plot_map_data_to_normal`、`plot_target_encoder`、`plot_target_encoder_cross_val`、`plot_discretization`、`plot_discretization_classification`、`plot_discretization_strategies`、`plot_polynomial_interpolation`、`plot_column_transformer`、`plot_cyclical_feature_engineering`），**一个都没有跑**，因此本页不对这些示例的读数下任何结论。

## 关键判断

1. **本节的统一对象是「拟合出的表示」，而 12 : 4 的有状态/无状态之分比「缩放/编码/离散化」的分类更实用**。前者决定「必须放进管线」，后者只决定「输入是什么类型」。实测把这条区分量化了：列统计量类变换的泄漏代价 ≤0.02（方向不定），而吃 $y$ 的 `TargetEncoder` 是 +0.055。
2. **`StandardScaler` 的两个默认假设都可关，而稀疏数据只有三条合法路**（`MaxAbsScaler`、`with_mean=False`、手工 `toarray`）。`RobustScaler` 能 `transform` 稀疏但不能 `fit`——这个半合法状态使它**不能进管线**，是本节最容易在部署时才发现的接口陷阱。
3. **常数列的 `scale_` 被静默替换为 1**。这让「标准化后没有 nan」成立，但**抹掉了「方差为 0」这个信息**，因此 `VarianceThreshold` 必须放在缩放之前（[[sklearn-feature-selection]] 的同一现象在此得到根因）。
4. **`MinMaxScaler` 是唯一「输出范围依赖训练集、且默认不裁剪」的缩放器**（`clip=False`），因此它是唯一会随部署数据漂移而静默越界的缩放器；`MaxAbsScaler.clip` 直到 1.8 才有。
5. **分位数变换与幂变换的差别是「保秩的两个不同实现」**：分位数变换是纯秩映射（保秩序、毁相关：Pearson 0.891→0.754、Spearman 不变），幂变换是参数化单调映射（1 个 λ/列、可逆、可外推）。前者在训练范围外**退化成常数**（0/1 或 ±5.1993），后者可以外推但**默认还会标准化**（`standardize=True`），使 λ 不可单独解释。
6. **类别编码的三个对象不是「三种写法」而是三条不同的代价曲线**：序数=1 列但强加顺序（实测 0.2236 vs 0.9845，改名后 0.0236）、one-hot=$n$ 列且不强制顺序、目标编码=1 列但必须折外（+0.055 的静默高估）。**「序数编码能不能用」不是风格问题，而是「目标效应在字典序上是否单调」的模型假设**——这一点剪藏只用一句「would interpret the categories as being ordered」（L311）提示，没有任何量化。
7. **`handle_unknown` 与 `drop` 的组合会让「未知」和「被丢弃的类别」共用同一个向量**（L450 文档自认，实测逆变换给出被 drop 的类别）。这是本节唯一「两个不同语义映射到同一表示」的实例，也是上线后「未知类别被预测成某一已知类」的机制来源。
8. **「用列数买非线性」的价格差两个数量级**：多项式是 $O(d^{\text{degree}})$（20 维 3 阶 = 1771 列），样条是 $O(d\cdot(\text{n\_knots}+\text{degree}-1))$（20 维默认 = 140 列）。实测两者在 $\sin$ 数据上几乎等价（0.9754 vs 0.9798），**所以默认应该先试样条**；多项式的正当用途是「显式要交互项」（`interaction_only`）与「要核方法的显式对应物」（L773）。
9. **本节是本库第一份「接口全在、机制全无」的剪藏**：11 个折叠块（≈933 词散文）恰好覆盖稀疏、λ 的估计、目标编码收缩、核中心化的全部推导，5 条文献全丢。因此本页对「为什么」的所有陈述都带「外部核查」或「本机实测」标签，**没有一条能引剪藏行号**——这是对 AutoResearch 最重要的一条使用限制。
10. **文档自身有三处与实现/自身不一致**，且**都在 1.9.1 上可复现**：① `KBinsDiscretizer` 印出的分箱区间是 1.6 口径（1.9.1 实际不同，但 doctest 的编号输出恰好不变）；② `max_categories` 并列取舍的说明把 `max_categories=3` 写成 2、把「字典序更小」说成「更大」；③ `QuantileTransformer` 的 iris 分位数 doctest 在 1.6.1 与 1.9.1 上都不复现（差 ≤0.02）。而第 1 条尤其危险：**doctest 通过 ≠ 解释文字正确**。

## 分歧

1. **`KBinsDiscretizer` 印出的区间与 1.9.1 运行时不符**（见「关键主张 12」「关键判断 10」）。1.6.1 与剪藏一致、1.9.1 不一致，而 `transform` 的输出仍与剪藏一致。归因：1.9 把 `quantile_method` 默认值从 `'warn'` 改成 `'averaged_inverted_cdf'`（外部核查四版签名）。**引用 `[-1, 2)` 这组边界时必须写版本。**
2. **`QuantileTransformer` 的 doctest 数值在三版上都不复现**（剪藏 `[0.00,0.24,0.49,0.73,0.99]`，1.6.1 `[0,0.24,0.51,0.74,1.0]`，1.9.1 `[0,0.239,0.509,0.743,1.0]`）。这是「文档打印值滞后于实现」而非「机制变了」——L205 的论断（变换后接近原分位数）依然成立。
3. **`max_categories` 并列取舍：正文说 `max_categories=2`，代码是 3；正文说低频是「字典序更大」，实测是「字典序更小」**（L592 与 L596，1.9.1 官方页同样如此）。实测 `max_categories=3` 才得到 `['b','c']`，且保留的是 `d`。
4. **`max_categories + 2` 的算术只在 `OrdinalEncoder` 上被验证**：L513 明说「`OrdinalEncoder` 的 `max_categories` 不考虑缺失/未知」，但同一节把 `OneHotEncoder` 的 `min_frequency`/`max_categories` 与它并列介绍（L533 起），**没有说 one-hot 这边加了几列**。实测 `OneHotEncoder(max_categories=mc)` 的输出列数恰好等于 `mc`（低频列占用一个名额），与序数版的「+2」不同。
5. **`get_feature_names_out()` 把低频列命名为 `x0_infrequent_sklearn`**（L556–L560），而正文的说明里用 `'infrequent'`/`'infrequent_sklearn'` 两种写法混称（L563–L566 的编号列表），读者无法判断列名到底是哪一个。实测列名是 `x0_infrequent_sklearn`。
6. **`OneHotEncoder` 的 `sparse_output=True` 默认值正文从不说明**，而两个 doctest 里出现 `.toarray()`（L368、L391、L417…）暗示返回的是稀疏矩阵。同一件事在 L645 又被表述为「one-hot encoded into a sparse matrix (See )」——链接文字丢失使这句无法自证。
7. **`TargetEncoder` 的 note 与 L608 的机制描述互相依赖却分处两段**：L606 只说「两者不等」，L608 才给交叉拟合，L614 才说 `fit` 被劝阻。剪藏里 `fit` 与 `fit_transform` 的差别因此需要读三段才能拼出，而**收缩公式（`smooth='auto'` 的经验贝叶斯）整段丢失**，使「不等」的幅度无法从文档自行估计。
8. **`SplineTransformer` 的四条优势只有第四条可验证**：L786（低阶 + 加结点更稳）、L787（无 Runge）、L788（外推可控）、L789（带状矩阵）中，只有 L789 能用输出矩阵直接核对（实测带状结构成立）；L788 的「可控」在默认值下其实是「外推被钉在边界」（实测 `X=10` 给边界基函数值），与「线性外推」不是一回事，而默认值 `extrapolation='constant'` 正文从不写。
9. **`FunctionTransformer` 的 `validate` 默认 `False` 被 doctest 掩盖**：L828 的示例显式写了 `validate=True`，读者会以为默认开启；实测默认 `False` 且 NaN 静默穿过。这是本节唯一「示例的写法和默认值指向相反方向」的地方。
10. **L57 与 L614 对「何时必须用 `fit_transform`」的强度不同**：L57 对缩放器只说「在训练集上学、在测试集上套」（正面陈述），L614 对目标编码说「discouraged … can introduce data leakage」（负面警告）。两者其实是同一条规则，但剪藏只在吃 $y$ 的对象上给出警告，容易让人以为缩放器无所谓——而实测缩放的泄漏代价确实小（±0.02），**所以文档的措辞强度与实测代价是一致的**，只是从不说破为什么。

## 局限

- 本节**不覆盖**缺失值填补（L727 只有一句指针，正文在 `sklearn.impute`，未入库）、文本向量化（8.2，未入库）、`sklearn.compose`（`ColumnTransformer` 与 `Pipeline` 只有零散示例）、`sklearn.kernel_approximation`、`sklearn.feature_extraction`。因此「先补缺失还是先编码」「列级异构数据怎么拼」在本页**无法回答**——只能靠本页第 42 条实测的 `ColumnTransformer` 骨架。
- 本节**没有任何评估**：0 个指标、0 个 CV 示例、0 个「变换前后模型分数」的对比。唯一的分数是 L74 的 `0.96`（且没有对照）。因此「该不该做某个预处理」在本页无法用本节的证据回答，本页所有量化结论都来自本机实测。
- 丢失的 11 个折叠块含**全部公式**：核中心化的 $\tilde K$、Box-Cox/Yeo-Johnson 的分段式、目标编码的收缩因子与经验贝叶斯 $m$。本页对这些内容的引用**全部指向官方 1.9.1 页**，不能引剪藏行号。
- **`TargetEncoder` 的 `smooth` 语义在本页只能定性描述**（"shrunk estimate"，L603）：`smooth='auto'` 的具体公式、`smooth` 取 float 时的行为、多类目标的收缩是否共享 $m$，都需要读丢失的折叠块或源码。
- 运行时复核覆盖 **1.6.1 与 1.9.1** 两个版本，但**没有覆盖 1.7/1.8**；因此 `quantile_method='warn'` 的过渡行为（1.7/1.8 是否发 FutureWarning）**未经实测**，只按签名推断。

## 待解问题

1. `KBinsDiscretizer` 的 `quantile_method` 默认变更（`'linear'` → `'averaged_inverted_cdf'`）在**真实分布**上会改变多少分箱编号？本次玩具数据上编号恰好不变，这只是巧合还是结构性的（因为分箱只依赖秩）？未做系统实验。
2. `QuantileTransformer` 的 `subsample`（类 10000 / 函数 100000）在 $n>10^4$ 时是否导致「同一份数据、类与函数给出不同编码」？未实测。
3. `TargetEncoder` 的 `smooth='auto'` 与手设 `smooth` 在**高基数 + 小样本**下差多少？丢失的折叠块给了经验贝叶斯公式，但没有任何读数；未做实验。
4. `TargetEncoder` 嵌套在 `ColumnTransformer` 里时，**外层 CV 与内层交叉拟合的折是否应当对齐**？实测能跑通（第 42 条），但两套折独立抽取时的泄漏量未测。
5. `OneHotEncoder(handle_unknown='infrequent_if_exist')` 在**训练集没有低频类别**时的行为：L565 的编号列表说「没有低频类别 → 全零」，但「低频支持已配置、只是本次数据没有低频类别」这一分支的输出列数未实测（是会多出一个恒零列，还是不加列？）。
6. `PolynomialFeatures` 与 `SplineTransformer` 在**外推**上的差别只有 L788 一句；实测显示样条默认把外推钉在边界上，多项式会飞出——在真实的「测试集分布偏移」场景里哪一种更差？未做实验。
7. `KernelCenterer` 的丢失推导（$\tilde K = K - 1K - K1 + 1K1$）与本页实测的行和为 0 是否等价？未逐项核对公式（只核了性质）。
8. `FunctionTransformer` 的 `check_inverse=True` 在**近似互逆**（如 `log1p`/`expm1` 的浮点误差）时是否会误报？本次两个用例（精确互逆、无 `inv_func`）都不报警，边界未探。
9. `PowerTransformer` 的 `method='yeo-johnson'` 在**含零/负数**数据上的 λ 与 Box-Cox 的 λ 不可比（第 12 条实测的 λ 差异可能只是参数化不同）——本页不据此断言「哪个更稳」。
10. **节内 10 条示例脚本全部未跑**（`plot_scaling_importance`、`plot_all_scaling`、`plot_discretization*`、`plot_target_encoder*`、`plot_polynomial_interpolation`、`plot_cyclical_feature_engineering` 等）。其中 `plot_all_scaling` 是唯一能给出「七种缩放器在含离群点数据上的形状对比」的定量证据，`plot_target_encoder_cross_val` 是唯一能给出「交叉拟合与不交叉拟合的分数差」的官方读数——两者都值得优先录入。
11. 剪藏 L727–L729 的 8.3.6 只有一句指针，`sklearn.impute` 的 `SimpleImputer`/`IterativeImputer`/`KNNImputer` **完全不在本库**，而「缺失值」是预处理顺序里不可跳过的一步（本页实践页的骨架用到了 `SimpleImputer` 的等价写法，但没有实测 `IterativeImputer`）。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 引言：包的目的与「表示」的定位 | L10 |
| 缩放重要性、离群点用稳健缩放器 | L12 |
| 标准化是「常见要求」；零均值单位方差 | L14–L16 |
| 忽略分布形状、只做中心化 + 缩放 | L18 |
| 目标函数对尺度的假设（RBF / $ℓ_1$/$ℓ_2$） | L20 |
| `StandardScaler` doctest（`mean_`/`scale_`/变换结果） | L24–L52 |
| 零均值单位方差的核对 | L54 |
| Transformer API 与「训练集学、测试集套」 | L57 |
| `Pipeline` + `LogisticRegression` doctest（0.96） | L59–L75 |
| `with_mean=False` / `with_std=False` | L77 |
| 缩放到区间：`MinMaxScaler` / `MaxAbsScaler` | L81 |
| 动机：极小标准差鲁棒、保留稀疏零 | L83 |
| `MinMaxScaler` doctest | L87–L107 |
| `MinMaxScaler` 的 `scale_` / `min_` 与 `feature_range` 公式 | L109–L117 |
| `MaxAbsScaler` 的动机与 doctest | L119–L147 |
| 稀疏缩放：中心化破坏结构、`with_mean=False`、`RobustScaler` 不能 fit | L150–L158 |
| 离群点：`RobustScaler` 作为 drop-in 替代 | L160–L162 |
| `KernelCenterer` 与中心化 Gram 矩阵 | L164–L166 |
| 分位数 vs 幂变换：都单调、都保秩 | L168–L174 |
| 分位数变换 $G^{-1}(F(X))$ 与两个事实；毁相关/距离 | L172 |
| `QuantileTransformer` iris doctest | L176–L209 |
| 幂变换：Yeo-Johnson 与 Box-Cox | L211–L215 |
| Box-Cox 折叠块（公式与三条限制**已丢**） | L217 |
| Box-Cox doctest（lognormal → 正态） | L219–L231 |
| 「某些分布有效、某些无效，要画图」+ 图 | L232–L234 |
| `output_distribution='normal'` 与 1e-7 截断 | L236–L252 |
| 归一化：逐样本单位范数；向量空间模型 | L254–L258 |
| `normalize` 与 `Normalizer` doctest（`fit` 无用） | L260–L294 |
| L2 = spatial sign preprocessing | L296 |
| 类别特征与整数码的例子 | L300 |
| `OrdinalEncoder` doctest | L302–L313 |
| 序数码会被当成有序（"which is often not desired"） | L311 |
| NaN 直通与 `encoded_missing_value` | L315–L341 |
| 「等价于 `OrdinalEncoder` + `SimpleImputer` 管线」doctest | L343–L353 |
| one-hot（one-of-K）编码 | L355–L357 |
| `OneHotEncoder` doctest 与 `categories_` | L359–L375 |
| 显式 `categories` 的 doctest（10 列） | L377–L395 |
| `handle_unknown='infrequent_if_exist'` | L397–L406 |
| `drop` 参数与共线性（非正则回归） | L408–L424 |
| `drop='if_binary'` | L426–L441 |
| `handle_unknown='ignore'` + `drop` 撞车（未知 = 被丢弃类别） | L443–L461 |
| 缺失类别支持折叠块（**散文已丢，只留代码**） | L463–L489 |
| 低频类别：`min_frequency` / `max_categories` 语义 | L491–L496 |
| `OrdinalEncoder` 低频 doctest（dog/snake） | L498–L511 |
| `max_categories + 2` 的算术（unknown/missing 各占一码） | L513–L531 |
| `OneHotEncoder` 低频 doctest 与 `get_feature_names_out` | L533–L561 |
| `handle_unknown='infrequent_if_exist'` 的两条分支 | L563–L566 |
| `max_categories` 单独使用 | L568–L578 |
| `min_frequency` 与 `max_categories` 同时给出的先后 | L580–L590 |
| 并列时的取舍说明（**与实现相反**）与 doctest | L592–L599 |
| `TargetEncoder` 的用途（高基数类别） | L603 |
| note：`fit().transform()` ≠ `fit_transform()` | L605–L606 |
| 交叉拟合机制与「训练集必须用 `fit_transform`」 | L608 |
| 交叉拟合示意图（SVG） | L610 |
| `fit` 不走交叉拟合、被劝阻（泄漏） | L614 |
| `encodings_` 是全量编码、折内编码不保存 | L616 |
| note：缺失值算一类、未见类别用 `target_mean_` | L618–L619 |
| 示例链接 2 条 | L621–L624 |
| `References`（**已空**） | L626 |
| 离散化引言 | L628–L632 |
| `KBinsDiscretizer` doctest | L636–L643 |
| 默认 one-hot 稀疏矩阵（**链接文字丢成 `(See )`**） | L645 |
| 三个特征的分箱区间（**1.6 口径**） | L647–L649 |
| 变换结果 | L651–L657 |
| 与直方图的区别 | L658 |
| `strategy` 三选（uniform/quantile/kmeans） | L664 |
| 自定义分箱（`FunctionTransformer` + `pd.cut`） | L666–L684 |
| 示例链接 3 条 | L686–L688 |
| 二值化：`Binarizer`（`fit` 无用）doctest | L692–L711 |
| `threshold` 调整 doctest | L713–L721 |
| `binarize` 函数 | L723 |
| `Binarizer` ≡ $k=2$ 的 `KBinsDiscretizer` | L725 |
| 8.3.6 缺失值填补（**只有一句指针**） | L727–L729 |
| 8.3.7 引言：纯多项式 vs 样条 | L731–L733 |
| `PolynomialFeatures` 说明与 doctest | L735–L754 |
| `interaction_only=True` doctest | L756–L769 |
| 多项式特征隐式出现在核方法里 | L773 |
| `SplineTransformer` 与 B 样条基 | L779 |
| note：样条逐特征、不给交互项 | L781–L783 |
| 样条相对多项式的四条优势 | L786–L789 |
| 样条 doctest（带状矩阵） | L793–L812 |
| `degree=0` 的样条 ≡ `KBinsDiscretizer` | L814 |
| 示例链接 2 条 | L816–L818 |
| `FunctionTransformer` 与 `np.log1p` doctest | L823–L834 |
| `check_inverse` 与 `filterwarnings` | L836–L842 |
| 文本特征相关示例链接 | L844 |
