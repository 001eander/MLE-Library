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
applies_to:
  - scikit-learn 1.9.1（文档正文与 API 签名）
---

# scikit-learn 用户指南 8.3「Preprocessing data」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 8.3. Preprocessing data（scikit-learn 用户指南第 8.3 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**48 个代码块（47 段 doctest + 1 段公式）**、3 个 `[!note]` 提示框、2 处 `[#]` 折叠标题残留、**1 张位图 + 1 张 SVG 图** |
| 原文 | <https://scikit-learn.org/stable/modules/preprocessing.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/8.3. Preprocessing data.md`（843 行、56,686 字节、`wc -w` 5,918 词，其中代码 1,695 词 → 散文 **4,086 词**；末行无换行） |
| SHA-256 | `8f4d49a12644cb8dc65b28d986c2bafe6820997d24e058128469e1daad29483d` |
| 完整性 | **11 个 `<details>` 折叠块全丢（1,150 词，其中散文 ≈933）、3 处 `References` 段全丢（含 5 条文献）、1 处页内链接文字被吞成 `(See )`**；官方散文 5,048 词 → 剪藏 4,086 词（保留 ≈81%）。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`author`/`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.1**（七条独立证据，见「版本判定」） |
| 插图 | 官方 1.9.1 页 `<article>` 内 2 张：`sphx_glr_plot_map_data_to_normal_001.png` 与 `target_encoder_cross_validation.svg`，均已入库 `raw/assets/sklearn-preprocessing/` 并逐张目视核查 |
| 代码 | 覆盖 16 个变换器 + 9 个函数级入口；`StandardScaler` 与 `Pipeline` 两段 doctest 有数值输出，其余多为形状与编码值输出 |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-preprocessing]], L30`。凡超出剪藏内容的部分（API 默认值、版本史、丢失折叠块的文字）都显式标注「外部核查」或「官方 1.9.1 页」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 未取得所有者对 `raw/` 回填的授权，剪藏保持抓取时的原始字节（sha256 `8f4d49a1…`）。丢失的 11 个折叠块只在「转换损失」一节里按官方 1.9.1 页逐块记录，**未写回原文件**。所有行号指向原始剪藏。`raw/assets/sklearn-preprocessing/` 下两张图是新增资产，不是对原文件的修改。

## 概要

这一节回答的是**模型看到的那张矩阵长什么样**，而不是选哪个模型。其核心价值有六层：

1. **预处理是有状态的契约**。除了 `Normalizer`/`Binarizer`/`FunctionTransformer` 三个无状态变换器（外加只记录 `powers_` 的 `PolynomialFeatures`），其余 12 个都在 `fit` 时学习数据相关的量，`transform` 再复用这些量。状态只能从训练集学习：L57 明写 `StandardScaler` 应在训练集上计算均值和标准差，再应用到测试集；L614 对 `TargetEncoder.fit` 明确警告泄漏风险。
2. **变换会改变模型假设**。L20 指出 RBF 核与 $ℓ_1$/$ℓ_2$ 正则项都可能隐含「各特征同尺度」的前提，量纲过大的特征可能支配目标函数。因此缩放不是无害清洗，而是模型定义的一部分。
3. **不同变换保留和破坏的性质不同**。分位数变换与幂变换都是单调的，因此保秩（L170）；但分位数变换会扭曲相关性和距离（L172）。序数编码会强加顺序（L311），one-hot 扩张列数，目标编码则必须处理 $y$ 带来的泄漏。
4. **预处理提供多种获得非线性的入口**。二值化、分箱、样条和多项式展开都能改变线性模型看到的表示；多项式核则可看作隐式展开（L773）。它们的列数成本与外推行为不同。
5. **剪藏保留了接口，却丢掉了机制**。16 个类名、9 个函数名和 47 段 doctest 都在，但 11 个折叠块与 5 条文献全部丢失，涵盖核中心化推导、Box-Cox/Yeo-Johnson 公式、目标编码收缩公式与稀疏输入说明。机制层只能引官方 1.9.1 页，不能假装剪藏包含这些内容。
6. **工程上必须与 CV 一起理解**。L57 要求训练集学习、测试集复用；L606–L616 进一步区分 `TargetEncoder.fit` 与 `fit_transform`。这与 [[cross-validation-design-and-tuning]] 的「所有会 `fit` 的步骤都放进管线」是同一条规则。

## 文档结构

| 章节 | 解决什么问题 | 状态来源（`fit` 学到什么） | 行号 |
| --- | --- | --- | --- |
| 引言 | 把原始特征向量改写成下游估计器更好用的表示 | — | L10–L12 |
| 8.3.1 Standardization | 减均值除标准差；目标函数为何需要同尺度 | `mean_`、`scale_` | L14–L77 |
| 8.3.1.1 Scaling to a range | `MinMaxScaler` 与 `MaxAbsScaler` | `min_`/`scale_` | L79–L147 |
| 8.3.1.2 Scaling sparse data | 中心化破坏稀疏；`with_mean=False` 与格式转换 | — | L150–L158 |
| 8.3.1.3 Scaling with outliers | 稳健替代 `RobustScaler` | `center_`、`scale_` | L160–L162 |
| 8.3.1.4 Centering kernel matrices | 在特征空间里中心化 Gram 矩阵 | `K_fit_rows_`、`K_fit_all_` | L164–L166 |
| 8.3.2 Non-linear transformation | 分位数 vs 幂变换；两者都单调保秩 | — | L168–L174 |
| 8.3.2.1 Mapping to a Uniform | `QuantileTransformer` 的 $G^{-1}(F(X))$ | `quantiles_`、`references_` | L176–L209 |
| 8.3.2.2 Mapping to a Gaussian | `PowerTransformer` 与正态输出的分位数变换 | `lambdas_` / `quantiles_` | L211–L252 |
| 8.3.3 Normalization | 逐样本缩放为单位范数 | 无 | L254–L296 |
| 8.3.4 Encoding categorical features | 序数、one-hot、缺失与未知类别 | `categories_`、`infrequent_categories_` | L298–L490 |
| 8.3.4.1 Infrequent categories | 用 `min_frequency`/`max_categories` 聚合低频类别 | `infrequent_categories_` | L491–L599 |
| 8.3.4.2 Target Encoder | 目标均值编码与交叉拟合 | `encodings_`、`target_mean_`、`target_type_` | L601–L626 |
| 8.3.5 Discretization | 连续量离散化 | — | L628–L632 |
| 8.3.5.1 K-bins | `KBinsDiscretizer` 的三种策略与编码 | `bin_edges_`、`n_bins_` | L634–L688 |
| 8.3.5.2 Feature binarization | `Binarizer`/`binarize` | 无 | L690–L725 |
| 8.3.6 Imputation | 只有指向 `sklearn.impute` 的一句指针 | — | L727–L729 |
| 8.3.7 Generating polynomial features | 纯多项式与样条两条路 | — | L731–L733 |
| 8.3.7.1 Polynomial features | `PolynomialFeatures` 的次数、交互与偏置 | `powers_`、`n_output_features_` | L735–L775 |
| 8.3.7.2 Spline transformer | B 样条基、外推与带状矩阵 | `knots`、`n_features_out_` | L777–L818 |
| 8.3.8 Custom transformers | `FunctionTransformer` 与自定义函数 | 无 | L821–L843 |

## 关键主张

### 1. 预处理的对象是表示，不是模型

L10 把目的写成「change raw feature vectors into a representation that is more suitable for the downstream estimators」。因此本节不含估计器与统一评分标准；`TargetEncoder` 虽使用 $y$，也只是为了编码。是否该做某个变换，必须结合下游模型与评估协议判断。

### 2. 有状态与无状态是 12 : 4

| | `fit` 学到的量 | 作用轴 |
| --- | --- | --- |
| `StandardScaler` | `mean_`、`var_`/`scale_` | 逐列 |
| `MinMaxScaler` | `data_min_`/`data_max_`/`scale_`/`min_` | 逐列 |
| `MaxAbsScaler` | `scale_` | 逐列 |
| `RobustScaler` | `center_`、`scale_` | 逐列 |
| `QuantileTransformer` | `quantiles_`、`references_` | 逐列 |
| `PowerTransformer` | `lambdas_` | 逐列 |
| `KernelCenterer` | `K_fit_rows_`、`K_fit_all_` | 全矩阵 |
| `OrdinalEncoder` / `OneHotEncoder` | `categories_`（+`infrequent_categories_`） | 逐列，列数可变 |
| `TargetEncoder` | `encodings_`、`target_mean_` | 逐列，使用 $y$ |
| `KBinsDiscretizer` | `bin_edges_` | 逐列，列数可变 |
| `SplineTransformer` | `knots` 的实际位置 | 逐列，列数可变 |
| `PolynomialFeatures` | 只记录由 `degree` 决定的 `powers_`，与数据无关 | 列之间 |
| `Normalizer` / `Binarizer` / `FunctionTransformer` | 无数据相关状态 | 逐样本（前两个） |

**本库综合**：12 个变换器学习数据相关状态，4 个不学习。`PolynomialFeatures` 虽改变列数，但不从数据估计参数。无状态不等于可以任意调整顺序：为了复用一致的管线，仍应把变换与模型一起保存。

### 3. 缩放要匹配下游假设

L20 说明目标函数可能假定各特征 centered around zero 或方差同量级。由此可拆成四类约束（本库综合）：

- RBF 核与最近邻：距离会被大量纲列支配；
- $ℓ_1$/$ℓ_2$ 正则：同一惩罚施加在不同量纲的系数上；
- 迭代优化：尺度影响条件数与收敛；
- 轴对齐树：只比较顺序，通常不依赖仿射缩放（见 [[decision-trees]]）。

因此「要不要缩放」不能脱离下游模型回答。L57 同时要求：状态只在训练集上学，再复用到测试集。

### 4. 稀疏输入禁止随意中心化

L150–L158 说明稀疏结构会被中心化破坏，`StandardScaler(with_mean=False)` 与 `MaxAbsScaler` 是主要路径；其他非 CSR/CSC 输入会转换成 CSR。任何稀疏缩放都应先检查是否允许中心化，而不是先 `toarray()` 付出全部内存代价。

### 5. `MinMaxScaler` 默认不裁剪

L105–L106 的 doctest 显示，训练范围外的测试点可以映射到区间外。API 签名是 `MinMaxScaler(feature_range=(0,1), *, copy=True, clip=False)`；需要截断时才显式 `clip=True`。`MaxAbsScaler.clip` 是 1.8 新增的参数，默认同样为 `False`（外部核查）。

### 6. 分位数变换保秩，但会扭曲相关与距离

L172 给出 $G^{-1}(F(X))$ 的数学依据，并明确说它「distort correlations and distances」。因此「保秩」不等于「保留几何」。正常输出还会在 1e-7 分位附近截断（L252），训练范围外的信息不应被理解为完整保留。

三个容易忽略的默认项（外部核查）：

- `n_quantiles=1000`，样本不足时会按样本量收缩并发出警告；
- 类的 `subsample=10000` 与函数的 `subsample=100000` 不同；
- 均匀与正态输出的边界行为不同，不能混用同一外推直觉。

### 7. 幂变换是参数化单调映射，默认还会标准化

L215 提供 Yeo-Johnson 与 Box-Cox；丢失折叠块中的官方说明补充了限制：Box-Cox 只能用于严格正数据，λ 由极大似然估计，`standardize=True` 默认会再做零均值单位方差。它与分位数变换的差别是：前者参数化、可逆且可外推；后者依赖经验分位数，区间外会退化。L232 建议直接画图检查变换是否有效。

### 8. 归一化与缩放作用在不同轴

L256 定义 `Normalizer` 为逐样本单位范数；L272 明说 `fit` 不做任何事。`normalize(..., axis=1)` 默认按行，而 `scale(..., axis=0)` 默认按列。按行归一化会抹掉样本长度但保留方向（$ℓ_2$ 时对应余弦相似度）；L296 称其为 spatial sign preprocessing。

### 9. 类别编码是三种不同的代价

| 对象 | 列数 | 顺序假设 | 未见类别 | 是否使用 $y$ |
| --- | --- | --- | --- | --- |
| `OrdinalEncoder` | 每列 1 个码 | **有**，按 `categories_` 顺序（L311） | `handle_unknown='use_encoded_value'` + `unknown_value` | 否 |
| `OneHotEncoder` | 随类别数扩张 | 无 | `'ignore'` 全零；`'infrequent_if_exist'` 归低频列 | 否 |
| `TargetEncoder` | 通常每列 1 个，多类时按类扩张 | 无 | 用 `target_mean_` | **是** |

L450 还给出 `drop` 与 `handle_unknown='ignore'` 的碰撞：未知类别和被 drop 的类别会共用同一个向量，逆变换无法区分。L463–L489 的代码则显示 `np.nan` 与 `None` 可被视为两个不同类别。

### 10. 目标编码的 `fit` 与 `fit_transform` 不同

L606 明说二者不等；L608 解释 `fit_transform` 使用交叉拟合；L614 警告 `fit` 不做交叉拟合并可能泄漏；L616 说明最终 `encodings_` 由全量训练集学习，折内中间编码不保存。

**本库综合**：即使编码器内部做了交叉拟合，只要在外层 CV 之前先对全量数据编码，外层测试折仍参与了统计量计算。正确做法是让目标编码在每个外层训练折内重新拟合。

版本边界（外部核查）：

- `smooth='auto'` 的经验贝叶斯公式在丢失折叠块中；
- `shuffle`/`random_state` 在 1.9 弃用，1.11 将移除；
- 1.9 起应通过 `cv` 传 splitter 控制洗牌，旧写法不能当成跨版本通用接口。

### 11. 低频类别需要区分编码与列数算术

L495–L496 给出 `min_frequency` 与 `max_categories` 的语义。L513–L531 的 doctest 说明 `OrdinalEncoder` 在同时配置未知值和缺失值时可出现 `max_categories + 2` 个码；one-hot 的列数应按自己的低频列规则计算，不能直接套这条算术。

L592 另有文档内部问题：正文写 `max_categories=2`，紧随的代码却是 `max_categories=3`。并列类别如何处理，正文措辞与实现行为需在目标版本重新核对，本页不保留历史运行时读数。

### 12. 分箱的默认值与版本边界必须写明

L645 说明默认 one-hot 稀疏输出；API 外部核查给出 `encode='onehot'` 与 `strategy='quantile'`。`quantile_method` 的默认值在 1.7/1.8/1.9 间变化，因此剪藏印出的区间不能直接当成当前版本输出；实际边界需在 `uv` 目标版本重新计算。

越界输入会落到最外侧箱，这是分箱区间定义的一部分，不应默认它会报错。`strategy='kmeans'` 还引入随机性，需显式记录 `random_state`。

### 13. 多项式与样条的列数成本不同

`PolynomialFeatures` 的列数随特征数与次数组合增长。20 个特征时，二次为 231 列、三次为 1771 列；`interaction_only=True` 时为 211/1351 列。`SplineTransformer` 每个特征输出 `n_knots + degree - 1` 列，默认 5 结点、3 阶即 7 列（API 外部核查）。

L786–L789 给出样条的四条优势：低阶加结点更稳、无 Runge 振荡、可控外推、带状矩阵。`extrapolation='constant'` 表示使用边界基函数值，不等于外推为零；需要线性外推时应显式选择 `'linear'`。L814 还给出 `degree=0` 样条与分箱的等价关系。

### 14. `FunctionTransformer` 默认不做输入校验

L823 允许把任意函数变成变换器，L827–L834 的示例显式使用 `validate=True`。API 外部核查的默认却是 `validate=False`：缺失值与非法值是否传播取决于自定义函数。`check_inverse=True` 也只有提供 `inverse_func` 并执行检查时才有意义，不能替代输入校验。

## 版本判定

**1.9.1，证据强。** 主要判据：

1. 小节编号 **8.3**（1.6/1.7 为 6.3，1.8 为 7.3）。
2. L610 使用 `This diagram shows`，区别于 1.7 的 `The following`。
3. L608、L614、L616 的 `TargetEncoder` 措辞包含 1.7 之后的版本。
4. pandas 输出为 `Categories (5, str)`，与 1.8 的 `(5, object)` 不同。
5. L12 `are highlighted` 与 L83 `includes` 排除 1.6 的旧措辞。
6. 133 条长句探针在官方 1.6/1.7/1.8/1.9 页命中数为 118/120/124/124；未命中项为 Markdown 伪影。
7. 抓取日 `/stable/modules/preprocessing.html` 与 `/1.9/` 字节相同，页标题为 `scikit-learn 1.9.1 documentation`。

## 转换损失

官方散文 5,048 词 → 剪藏 4,086 词（保留约 81%）。丢失集中在 11 个 `<details>` 与 3 处 References：

1. 8.3.1 References；
2. *Scaling vs Whitening*；
3. 8.3.1.4 *Mathematical formulation*（`KernelCenterer` 推导与 Schölkopf 1998）；
4. *Yeo-Johnson transform*；
5. *Box-Cox transform*；
6. 8.3.3 *Sparse input*；
7. *Support of categorical features with missing values*（散文丢失、代码保留）；
8–10. 目标编码的 Binary / Multiclass / Continuous 三块（收缩因子与经验贝叶斯）；
11. 8.3.5.2 *Sparse input*；
12. `TargetEncoder` References（Micci-Barreca 2001、Pargent 2022）；
13. 样条 References（Eilers & Marx 1996、Perperoglou 2019）。

另有 L645 的页内链接文字被吞成 `(See )`。因此本页对核中心化、幂变换公式、目标编码收缩和稀疏输入的机制说明必须引官方 1.9.1 页，不能引剪藏行号。

## 外部核查（不在剪藏内）

### A. 主要对象签名（官方 1.9.1 API 页）

| 对象 | 关键默认值 |
| --- | --- |
| `StandardScaler` | `copy=True, with_mean=True, with_std=True` |
| `MinMaxScaler` | `feature_range=(0,1), copy=True, clip=False` |
| `MaxAbsScaler` | `copy=True, clip=False`（`clip` 为 1.8 新增） |
| `RobustScaler` | `with_centering=True, with_scaling=True, quantile_range=(25.0,75.0), unit_variance=False` |
| `QuantileTransformer` | `n_quantiles=1000, output_distribution='uniform', subsample=10000` |
| `quantile_transform` | `output_distribution='uniform', subsample=100000` |
| `PowerTransformer` | `method='yeo-johnson', standardize=True` |
| `Normalizer` / `normalize` | `norm='l2'`；函数 `axis=1` |
| `OrdinalEncoder` | `categories='auto', dtype=float64, handle_unknown='error', encoded_missing_value=nan` |
| `OneHotEncoder` | `categories='auto', drop=None, sparse_output=True, handle_unknown='error', min_frequency=None, max_categories=None` |
| `TargetEncoder` | `categories='auto', target_type='auto', smooth='auto', cv=5, shuffle='deprecated', random_state='deprecated'` |
| `KBinsDiscretizer` | `n_bins=5, encode='onehot', strategy='quantile', subsample=200000` |
| `Binarizer` / `binarize` | `threshold=0.0` |
| `PolynomialFeatures` | `degree=2, interaction_only=False, include_bias=True, order='C'` |
| `SplineTransformer` | `n_knots=5, degree=3, knots='uniform', extrapolation='constant', include_bias=True, handle_missing='error'` |
| `KernelCenterer` | 无额外构造参数 |
| `FunctionTransformer` | `func=None, inverse_func=None, validate=False, check_inverse=True` |

### B. 版本边界

| 参数 | 版本记录 |
| --- | --- |
| `KBinsDiscretizer.quantile_method` | 1.7 新增（默认 `'warn'`）；1.9 默认改为 `'averaged_inverted_cdf'` |
| `SplineTransformer.handle_missing` | 1.8 新增，默认 `'error'` |
| `MaxAbsScaler.clip` | 1.8 新增，默认 `False` |
| `TargetEncoder.shuffle`/`random_state` | 1.9 弃用，1.11 将移除；改用 `cv` splitter |
| `TargetEncoder.cv` | 1.9 起可传 splitter；旧版本兼容范围需另行核对 |

`quantile_method` 的变化会影响分箱边界，实际输出必须在目标版本重新计算。

### C. `TargetEncoder` 弃用细节（源码 @ 1.9.1）

- `shuffle` 与 `random_state` 的约束只接受布尔值或 `'deprecated'` 等旧值迁移形式；
- 警告在 `fit_transform` 路径发出，提示在 1.11 移除并改用 `cv` generator；
- 若 `cv` 已是 splitter，旧 shuffle kwargs 不再承担原来的控制作用。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 图表核查

官方本节 2 张图均已入库并逐张目视核查：

| 文件 | 内容 |
| --- | --- |
| `sphx_glr_plot_map_data_to_normal_001.png` | 8 行 × 3 列直方图矩阵，比较 Box-Cox、Yeo-Johnson 与 Quantile transform 对多种分布的效果。Uniform 变换后仍平、Bimodal 仍双峰，说明单调幂变换不能把任意分布变成单峰高斯。图内 λ 也显示两种方法的参数不可直接比较 |
| `target_encoder_cross_validation.svg` | 5 个交叉拟合面板与结果列，表达每折用其余折学习编码再合并。箭头完整文字 `Combine Folds` 从 SVG 源码确认；图内没有数字，默认 `cv=5` 只能引 L610 文本 |

官方另有 10 条示例链接未运行，本页不对这些示例的额外读数下结论。

## 关键判断

1. **12 : 4 的状态划分比「缩放/编码/离散化」更实用**：它决定哪些步骤必须从训练折学习，但不意味着无状态步骤可以随意调换顺序。
2. **缩放合法性来自下游假设**：核、距离、正则与优化需要同尺度；轴对齐树通常不需要。
3. **稀疏输入先检查中心化许可**：`MaxAbsScaler` 与 `with_mean=False` 是主要路径，不能默认稠密化。
4. **分位数变换与幂变换是两种保秩路径**：前者依赖经验秩映射并扭曲相关/距离，后者参数化、可逆、可外推，但默认还会标准化。
5. **类别编码的合法性依赖目标结构**：序数编码要求目标效应与编码顺序匹配；one-hot 改变列数；目标编码必须折内交叉拟合。
6. **`drop` 与 `handle_unknown='ignore'` 会合并未知与 drop 类别的表示**，上线后无法从向量区分二者。
7. **多项式与样条的选择应先算列数成本**：样条通常列数更少；需要显式交互项时才选择多项式展开。
8. **剪藏接口完整不代表机制完整**：所有「为什么」都应引官方折叠块或新的目标环境验证。
9. **doctest 通过不等于解释文字永远正确**：默认值变更可能使打印区间过时，引用具体输出必须带版本。

## 分歧

1. **`KBinsDiscretizer` 的打印区间与版本默认值变化有关**。剪藏给出旧区间，而 1.9 的 `quantile_method` 默认已变。实际 1.9.1 边界需在 `uv` 目标版本重新计算，不能沿用历史运行时读数。
2. **`QuantileTransformer` 的 doctest 打印值需复核**。本页保留 L205 的「输出接近原分位数」作为来源论述，不再用历史复现值证明或反驳它。
3. **`max_categories` 的示例编号不一致**：L592 写 2，代码却是 3。并列类别取舍方向也需在目标版本复核，当前只保留文档内部冲突。
4. **`max_categories + 2` 不是所有编码器的通用算术**：该式来自 `OrdinalEncoder` 的特定配置，one-hot 应按低频列自己的规则计算。
5. **`OneHotEncoder.sparse_output=True` 正文不展开**：只能从 API 与 `.toarray()` 示例确认默认输出形态。
6. **`TargetEncoder` 的关键说明分散且公式丢失**：L606、L608、L614、L616 必须连读；`smooth='auto'` 的具体机制不能只靠剪藏解释。
7. **`SplineTransformer` 的外推默认值正文未强调**：`extrapolation='constant'` 是钉在边界，不是线性外推，也不是输出零。
8. **`FunctionTransformer` 示例与默认值方向不同**：示例显式 `validate=True`，API 默认 `False`，不能由示例推断默认会拦截 NaN。
9. **L57 与 L614 的措辞强度不同**：前者要求训练集学习，后者明确警告泄漏。两者属于同一条管线纪律，但目标编码使用 $y$，风险更直接。

## 局限

- 本节不覆盖缺失值填补正文（`sklearn.impute`）、文本向量化、`sklearn.compose` 的完整用法、`sklearn.kernel_approximation` 与 `sklearn.feature_extraction`。
- 本节没有统一评估指标或「变换前后模型表现」对照；唯一的 `Pipeline` 示例分数不能替代模型选择证据。
- 11 个折叠块包含几乎全部数学推导与 5 条文献，机制层必须另引官方页。
- `TargetEncoder.smooth` 的公式、手设 float 的行为与多类目标收缩细节需读官方折叠块或源码。
- 不包含历史本机实验；所有运行时行为、性能与实际数据收益都需在 `uv` 目标版本环境中复核。

## 待解问题

1. `KBinsDiscretizer.quantile_method` 的默认变更在真实分布上会改变多少分箱边界与编号？
2. `QuantileTransformer` 的类与函数在 `subsample` 默认不同，大样本下是否会产生不同编码？
3. `TargetEncoder.smooth='auto'` 与手设 `smooth` 在高基数、小样本下差多少？
4. `TargetEncoder` 嵌套在 `ColumnTransformer` 与外层 CV 中时，内外层折不对齐会造成多大泄漏？
5. `OneHotEncoder(handle_unknown='infrequent_if_exist')` 在训练集没有低频类别时，输出列数与零列如何处理？
6. `PolynomialFeatures` 与 `SplineTransformer` 在测试分布偏移下的外推风险分别是什么？
7. `KernelCenterer` 的丢失推导与其输出性质如何逐项对应？
8. `FunctionTransformer.check_inverse=True` 对近似互逆与浮点误差的容忍度如何？
9. Yeo-Johnson 与 Box-Cox 的 λ 在不同分布上如何变化，何时值得使用？
10. 官方 10 条示例脚本中，`plot_all_scaling` 与 `plot_target_encoder_cross_val` 可作为后续目标环境验证的优先资料。
11. `sklearn.impute` 未入库，缺失值处理的完整顺序与版本行为需要独立来源。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 引言与缩放重要性 | L10–L20 |
| `StandardScaler`、训练集学习与 `Pipeline` | L24–L77 |
| `MinMaxScaler` / `MaxAbsScaler` | L79–L147 |
| 稀疏缩放、离群点与 `KernelCenterer` | L150–L166 |
| 分位数与幂变换 | L168–L252 |
| 逐样本归一化 | L254–L296 |
| 序数与 one-hot 编码、未知与缺失类别 | L298–L490 |
| 低频类别 | L491–L599 |
| `TargetEncoder` 与交叉拟合 | L601–L626 |
| 分箱与二值化 | L628–L725 |
| 缺失值指针 | L727–L729 |
| 多项式与样条 | L731–L818 |
| `FunctionTransformer` | L821–L843 |

## 关联页面

- 概念：[[feature-preprocessing]]（机制与失效模式）、[[feature-selection]]（选择与预处理的区别）、[[cross-validation]]（折内拟合）
- 实践：[[preprocessing-pipeline-design]]（管线设计）、[[feature-selection-workflow]]、[[cross-validation-design-and-tuning]]
- 工具：[[scikit-learn]]（变换器矩阵、默认值与版本边界）
- 来源：[[sklearn-feature-selection]]、[[sklearn-cross-validation]]、[[sklearn-linear-models]]、[[sklearn-nearest-neighbors]]
