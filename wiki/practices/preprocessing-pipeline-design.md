---
title: "预处理管线的落地设计：顺序、默认值、诊断信号与 18 条坑"
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - preprocessing
  - pipeline
  - feature-scaling
  - categorical-encoding
  - target-encoding
  - scikit-learn
  - workflow
sources:
  - "[[sklearn-preprocessing]]"
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-cross-validation]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（接口、默认值与行为；本次已在 1.9.1 上实测）
  - scikit-learn 1.6.1（并列给出差异的条目）
---

# 预处理管线的落地设计

本页回答五个工程问题：**要不要做、按什么顺序做、用哪个对象、哪些默认值必须写出来、怎么知道做错了**。机制与几何在 [[feature-preprocessing]]，来源与版本在 [[sklearn-preprocessing]]。引用写作 `[[sklearn-preprocessing]], L154`。

## 一、五个准入问题（任何一个答不上就先别动）

1. **下游的假设里有没有「同尺度」这一条？** 树模型没有（对单调变换免疫），核方法/距离/正则化/神经网络有。答「没有」→ 跳过缩放（见 [[feature-preprocessing]] 的判据表）。
2. **数据是稀疏的吗？** 是 → 只有三条路：`MaxAbsScaler`、`StandardScaler(with_mean=False)`、显式 `toarray()`（L154–L158）。稀疏 + 中心化 = `ValueError`。
3. **类别列是无序的吗？** 无序 → 不能用 `OrdinalEncoder`（实测 $R^2$ 0.9845 → 0.2236），只能 one-hot 或目标编码。
4. **基数多大？** 这是选编码器的唯一量化判据（见第三节的表）：低基数 one-hot、中基数 one-hot + 低频聚合、高基数目标编码。
5. **这一步吃 $y$ 吗？** 吃（`TargetEncoder`、任何编码后与目标相关的统计）→ **必须在折内**；不吃 → 仍应放进管线，但代价通常 ≤0.02（见第六节）。

## 二、顺序：七个阶段

顺序不是风格问题，每一步的顺序都有可失败的机制：

| # | 阶段 | 对象 | 为什么在这个位置 |
| --- | --- | --- | --- |
| 1 | **缺失值** | `SimpleImputer`（`sklearn.impute`，**本库未入库**） | 之后的每个变换都会把 NaN 变成错误或静默传播（实测 `SplineTransformer`、`FunctionTransformer(validate=True)` 都抛 `ValueError`） |
| 2 | **列类型拆分** | `ColumnTransformer` | 数值列与类别列必须走不同的路；`remainder='drop'` 才不会被静默放过 |
| 3 | **类别编码** | `OneHotEncoder` / `TargetEncoder` | 编码会**扩张列数**，若先缩放再编码，缩放器看到的列数与最终不一致 |
| 4 | **缩放** | `StandardScaler` / `RobustScaler` / `MaxAbsScaler` | 必须在**展开之前**：多项式/样条会造出新列，那些列的量纲由原列决定；缩放放在后面等于漏掉新列 |
| 5 | **非线性展开** | `SplineTransformer` / `PolynomialFeatures` / `KBinsDiscretizer` | 展开后的列数爆炸，之后只应做选择或建模 |
| 6 | **特征选择** | `SelectKBest` / `SelectFromModel` | 必须在缩放**之后**（否则尺度影响统计量）且在展开**之后**（否则选不到交互项）；见 [[feature-selection-workflow]] |
| 7 | **模型** | — | — |

三条容易搞反的顺序：

- **`VarianceThreshold` 在缩放之前**。`StandardScaler` 把常数列的 `scale_` 替换为 1（实测），所以缩放后「方差为 0」的列方差变成 1，一个都删不掉（[[sklearn-feature-selection]] 实测 16）。
- **`KBinsDiscretizer` 必须在缩放之前或不做缩放**：分箱的边界由原尺度决定，先标准化再分箱等价于「在标准化尺度上分位数分箱」——数学上常常相同（分位数分箱对单调变换不变），但 `strategy='uniform'` 与 `'kmeans'` 会不同。
- **`Normalizer` 放在最后（缩放之后）**：它按行归一化，会抹掉缩放刚建立的列间比例；如果下游是线性模型，`Normalizer` 之后 `StandardScaler` 的列统计量会被行归一化搅乱。

## 三、缩放器判定表

| 场景 | 用哪个 | 关键参数 | 不要用的理由 |
| --- | --- | --- | --- |
| 默认（稠密、无重离群点） | `StandardScaler` | `with_mean=True, with_std=True` | — |
| 稀疏 | `MaxAbsScaler` 或 `StandardScaler(with_mean=False)` | — | 中心化破坏稀疏（`ValueError`） |
| 有重离群点 | `RobustScaler` | `quantile_range=(25,75)`、`unit_variance=False`（默认） | **不能 `fit` 稀疏**（只能 `transform`） |
| 下游要求有界（神经网络、图像） | `MinMaxScaler(feature_range=(0,1), clip=True)` | **必须显式 `clip=True`** | 默认 `clip=False` 时测试集能越界（实测 `-1.5 / 1.6667`） |
| 已经中心化在 0（如 TF-IDF） | `MaxAbsScaler` | — | `StandardScaler` 会再中心化，破坏稀疏 |
| 核矩阵/Gram 矩阵 | `KernelCenterer` | 要求**方阵** | 3×2 输入直接 `ValueError` |
| 只想去掉偏度 | `QuantileTransformer` 或 `PowerTransformer` | 见第四节 | 两者都毁可解释的线性关系 |

一条容易被忽略的：`RobustScaler` 的 `scale_` 是 **IQR**，不是标准差；它与 `StandardScaler.scale_` **不可互换解释**。要「正态等价」必须 `unit_variance=True`。

## 四、分位数变换 vs 幂变换（都保秩，但代价不同）

| 需求 | 选 | 必须显式写的参数 | 已知陷阱 |
| --- | --- | --- | --- |
| 只要「压掉异常值、变成钟形/均匀」 | `QuantileTransformer` | `n_quantiles=min(1000, n_train)`、`subsample` | 训练范围外**退化成常数**（0/1 或 ±5.1993）；`n_quantiles` 会被样本量静默夹住 |
| 要一个**可解释、可外推**的单调变换 | `PowerTransformer(method='yeo-johnson')` | `standardize=False`（若要按 λ 解释） | Box-Cox 只能用于**严格正**数据；`standardize=True` 是默认，会再套一层标准化 |
| 类别分布严重偏斜（长尾计数） | 先 `log1p`（`FunctionTransformer`）再缩放 | `validate=True` | `validate=False` 是默认，NaN 会静默穿过 |

**同一个数据上两种方法给出的 λ 不能互比**（官方配图：同一 lognormal 分布 Box-Cox λ=0 vs Yeo-Johnson λ=−0.83），所以「λ̂ 接近 0 所以该取对数」只能在同一方法内部说。

## 五、类别编码判定表（按基数）

| 基数 | 编码器 | 参数 | 列数（实测） |
| --- | --- | --- | --- |
| ≤10 且**有序**（星级、教育程度） | `OrdinalEncoder` | 显式给 `categories=[...]`（按真实序，不是字典序） | 1 |
| ≤10 且无序 | `OneHotEncoder` | `handle_unknown='infrequent_if_exist'`（部署安全） | $n_{\text{cat}}$ |
| 10–50 | `OneHotEncoder` + 低频聚合 | `min_frequency=10..20`、`max_categories=...` | 实测 120 类 + `min_frequency=20` → **29** 列 |
| >50（zip、用户 ID、SKU） | `TargetEncoder` | `cv=5`（**只能 `fit_transform` 训练集**） | 1（多类时 $n_{\text{classes}}$） |
| 任意基数但必须避免共线性 | `OneHotEncoder(drop='first')` | **不要与 `handle_unknown='ignore'` 同时用** | $n_{\text{cat}}-1$ |

四条硬规则（每条都有实测支撑）：

1. **`OrdinalEncoder` 的合法性 = 「目标效应在字典序上单调」**。不是「树模型能不能吃」。实测：非单调效应下 `OrdinalEncoder`+`Ridge` 的 $R^2$ = 0.2236，`OneHotEncoder`+`Ridge` = 0.9845；**把一个水平改名**（数据不变）序数编码掉到 0.0236。
2. **`drop` + `handle_unknown='ignore'` 会让未知类别变成「被丢弃的那一类」**（L450 文档自认，实测 `inverse_transform` 还原成 `[['female', None, None]]`）。要么用 `drop=None`，要么用 `'infrequent_if_exist'`。
3. **不要手动给 `categories=`**（除非有业务上必须固定的全集）：L397 说训练期缺类别时「it can often be better to specify `handle_unknown='infrequent_if_exist'` instead」；实测手动给全集会产生恒零列。
4. **`np.nan` 与 `None` 是两个类别**（实测 `categories_ = ['Firefox','Safari',None,nan]`）。要先决定「缺失是一种取值还是缺失」，否则 one-hot 会多出两列。

## 六、`TargetEncoder` 的四条规则

1. **训练集只用 `fit_transform(X_train, y_train)`**。`fit().transform()` 不走交叉拟合，in-sample 相关 0.8045 vs 0.7708，下游 `Ridge` 的 CV $R^2$ 被抬高 +0.055（实测）。L614 明写 `fit` "is discouraged … can introduce data leakage"。
2. **测试集只用 `transform(X_test)`**，用的是全量训练集的 `encodings_`（L616）。
3. **它必须在 `Pipeline`/`ColumnTransformer` 内**——即使内部交叉拟合，在 CV 之外先做仍然泄漏（实测 0.6414 vs 管线内 0.5795）。
4. **`cv` 的写法有版本下界**：1.9 起可以传 splitter（`TargetEncoder(cv=KFold(5, shuffle=True, random_state=1))`），1.6.1 只接受整数（`InvalidParameterError`）。1.9 同时弃用了 `shuffle`/`random_state`（1.11 移除），所以**面向新版本写的代码要传 splitter，而不是 `shuffle=False`**。

## 七、非线性展开判定表

| 需求 | 选 | 每特征列数 | 已知行为 |
| --- | --- | --- | --- |
| 要平滑、可微、条件数好 | `SplineTransformer(n_knots=k, degree=3)` | `k + 2`（默认 5+3−1 = 7） | 外推被**钉在边界**（默认 `extrapolation='constant'`）；不给交互项 |
| 要**交互项** | `PolynomialFeatures(degree=2, interaction_only=True)` | $\binom{d}{2}+d+1$ | 20 维 2 阶 → 211 列；3 阶 → 1351 列 |
| 要「便宜的非线性 + 可解释的分段」 | `KBinsDiscretizer(n_bins=k, encode='onehot-dense')` | $k$ | 默认 `encode='onehot'` 返回**稀疏矩阵**；必须显式 `onehot-dense` 才能喂给需要稠密的模型 |
| 只要二值化 | `Binarizer(threshold=t)` | 1 | `> t` 严格大于（等于阈值为 0） |
| 每个特征都要非线性但列数敏感 | 样条 > 分箱 > 多项式 | 7 / 10 / 231（20 维） | 实测 $R^2$：0.9798 / 0.9552 / 0.9754（线性 0.6787） |

## 八、必须显式写出的默认值

这 14 条是本节里「不写就会静默改变结果」的默认值（外部核查 1.9.1 签名 + 实测）：

| # | 对象 | 默认值 | 不写会怎样 |
| --- | --- | --- | --- |
| 1 | `StandardScaler` | `with_mean=True, with_std=True` | 稀疏输入直接 `ValueError` |
| 2 | `MinMaxScaler` | `clip=False` | 测试集越界（`-1.5 / 1.6667`） |
| 3 | `MaxAbsScaler` | `clip=False`（**1.8+**） | 越界不截断；1.7 及更早传 `clip` 直接 `TypeError` |
| 4 | `RobustScaler` | `unit_variance=False` | `scale_` 是 IQR，不是标准差 |
| 5 | `QuantileTransformer` | `n_quantiles=1000`、`subsample=10000` | 小数据上 `n_quantiles_` 被夹到样本数；`quantile_transform` 函数的 `subsample` 是 **100000** |
| 6 | `PowerTransformer` | `method='yeo-johnson'`、`standardize=True` | Box-Cox 要显式 `method='box-cox'`；λ 会被标准化掩盖 |
| 7 | `Normalizer` | `norm='l2'`、`axis=1` | 想逐列归一却用了逐行 |
| 8 | `OneHotEncoder` | `sparse_output=True`、`handle_unknown='error'` | 下游拿到稀疏矩阵；线上遇未知类别直接崩 |
| 9 | `OrdinalEncoder` | `encoded_missing_value=nan`、`handle_unknown='error'` | NaN 直通到模型；未知类别崩 |
| 10 | `TargetEncoder` | `smooth='auto'`、`cv=5`、`shuffle='deprecated'`（1.9） | 1.6.1 上 `shuffle=True` 默认；跨版本行为不同 |
| 11 | `KBinsDiscretizer` | `strategy='quantile'`、`encode='onehot'`、`subsample=200000` | 默认不是 `'uniform'`；输出是稀疏矩阵 |
| 12 | `SplineTransformer` | `n_knots=5`、`degree=3`、`extrapolation='constant'`、`handle_missing='error'`（1.8+） | 外推被钉住；NaN 直接抛错 |
| 13 | `PolynomialFeatures` | `degree=2`、`include_bias=True` | 与下游 `fit_intercept` 重复一个常数列 |
| 14 | `FunctionTransformer` | `validate=False`、`check_inverse=True` | NaN 静默穿过；只有给了 `inverse_func` 才真的检查 |

## 九、可复制骨架（已在本机 1.9.1 与 1.6.1 上跑通）

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

prep = ColumnTransformer(
    [
        ("num", StandardScaler(), ["num1", "num2"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=10, sparse_output=False), ["cat"]),
        ("zip", TargetEncoder(cv=5), ["zip"]),      # 高基数：只编码，不扩张
    ],
    remainder="drop",                                # 默认是 'drop'，但写出来
    verbose_feature_names_out=True,
)
pipe = Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))])
scores = cross_val_score(pipe, df, y, cv=KFold(5, shuffle=True, random_state=0))
```

实测结果（1.9.1 / 1.6.1）：5 折 $R^2$ = **0.9915 / 0.9916**；`pipe.named_steps['prep'].get_feature_names_out()` 给出 7 个名字（`num__num1`、`num__num2`、`cat__cat_a…d`、`zip__zip`）——**`TargetEncoder` 那一路只有 1 列**，这是它与 one-hot 的核心差别。

四件必须写在骨架里的事：

1. **每一路显式给列名列表**（而不是让 `ColumnTransformer` 猜 dtype）：类别列被当成数值是本节最常见的静默错误。
2. **`remainder='drop'`** 写出来（虽然它是默认）：忘记的列会静默消失，而 `get_feature_names_out()` 不会提示。
3. **`TargetEncoder` 只在 `ColumnTransformer` 里用**（不要在 `cross_val_score` 之先把编码结果算好再传进去）。
4. **`OneHotEncoder(sparse_output=False)`** 当后面接 `StandardScaler`/`Ridge` 时；只有确认下游吃稀疏才留默认。

## 十、七条诊断信号

1. **`feature_names_in_` 与最终列数对不上预期** → 有列被 `remainder='drop'` 丢掉，或类别列被当数值。
2. **`categories_` 里出现 `nan`/`None`** → 缺失值被当成类别；要么补缺失，要么接受它是两个额外列。
3. **`n_quantiles_` ≠ 你设的 `n_quantiles`** → 样本量小于 1000，分位数已退化成「每样本一个分位」。
4. **训练时 `X_transformed.mean(0)` 为 0 而测试集不是** → 正常（缩放器用训练集统计量）；但如果测试集**均值漂移超过 1 个标准差**，说明训练/测试分布不同，缩放器把这个问题掩盖了。
5. **`bin_edges_` 与你读到的文档区间不一致** → 1.9 的 `quantile_method` 默认值变了（实测 `[-3,-1,2,6]` → `[-3,-1.5,3,6]`）；不要照抄文档里的区间。
6. **`TargetEncoder` 的训练集编码与推理编码差别很大** → 你用了 `fit().transform()` 而不是 `fit_transform`（实测最大差 0.05）。
7. **CV 分数比 LB / 测试集高一大截，且预处理里有吃 $y$ 的步骤** → 先查「编码是否在折内」，再看 `StandardScaler`（后者通常只值 ±0.02）。

## 十一、18 条坑

1. 稀疏矩阵上 `StandardScaler()` → `ValueError: Cannot center sparse matrices: pass 'with_mean=False' instead.`
2. `RobustScaler` 放进 `Pipeline` 做 `fit` → 同样的 `ValueError`（它要中心化）；但它**能** `transform` 稀疏，所以「离线训练稠密、线上推理稀疏」会突然失败。
3. 缩放之后 `VarianceThreshold()` → 一列都不删（常数列 `scale_` 被替换为 1）。
4. `MinMaxScaler` 默认 `clip=False` → 测试集输出越界，下游假设 `[0,1]` 的代码静默错。
5. `QuantileTransformer` 在训练范围外的输入 → 全部压成 `0`/`1`（均匀）或 `±5.1993`（正态），**异常值不再有区分度**。
6. `QuantileTransformer(n_quantiles=1000)` 在小数据上 → `UserWarning` + `n_quantiles_ = n_samples`。
7. `QuantileTransformer` 想让「相关不变」→ 做不到：Pearson 0.891→0.754（Spearman 不变）。
8. `PowerTransformer(method='box-cox')` 喂含 0/负值的数据 → `ValueError: strictly positive data`；默认 `method` 是 `'yeo-johnson'`，所以「没写 method」时行为完全不同。
9. Box-Cox 之后按 λ 解释 → 默认 `standardize=True` 已经又做了一次零均值单位方差。
10. `OrdinalEncoder` 用在无序类别上 → $R^2$ 从 0.9845 掉到 0.2236；**改一个类别的名字**就能再掉到 0.0236。
11. `OneHotEncoder(drop='first', handle_unknown='ignore')` → 未知类别与被 drop 的类别共用全零向量（`inverse_transform` 会把它还原成被 drop 的那类）。
12. `OneHotEncoder()` 默认 `handle_unknown='error'` → 线上第一个新类别就抛 `ValueError: Found unknown categories`。
13. 高基数类别直接 one-hot → 列数 = 基数（实测 120 值 → 120 列），训练变慢且系数不可信。
14. `TargetEncoder` 用 `fit().transform()` → 训练集编码与推理不一致，CV 分数虚高 +0.05。
15. `TargetEncoder` 在 `cross_val_score` 之外先算好编码 → 仍然泄漏（0.6414 vs 0.5795）。
16. `KBinsDiscretizer` 默认参数 → 输出 `csr_matrix`；接 `StandardScaler` 或 `Ridge` 时行为与预期不同（要 `encode='onehot-dense'`）。
17. `PolynomialFeatures(degree=3)` 在 20 维上 → 1771 列，训练时间与内存被下游模型放大（变换本身只要 7 ms，模型不是）。
18. `SplineTransformer` 遇 NaN → `ValueError`（1.8+ 有 `handle_missing`，默认 `'error'`）；外推被钉在边界，不是线性。

## 十二、成本表（1.9.1，2000 行 × 20 特征，单次 `fit_transform`）

| 对象 | 壁钟 | 输出列数 |
| --- | --- | --- |
| `StandardScaler` | **1.3 ms** | 20 |
| `KBinsDiscretizer(n_bins=10, encode='ordinal')` | 2.3 ms | 20 |
| `KBinsDiscretizer(n_bins=10)`（默认 one-hot） | 5.6 ms | 200 |
| `SplineTransformer()`（默认 5 结点三次） | 2.5 ms | 140 |
| `PolynomialFeatures(2)` | 1.1 ms | 231 |
| `PolynomialFeatures(3)` | 7.1 ms | **1771** |
| `QuantileTransformer(n_quantiles=1000)` | **22.5 ms** | 20 |
| `PowerTransformer()`（Yeo-Johnson，λ 由 MLE 估） | **41.1 ms** | 20 |
| `OrdinalEncoder`（120 类） | 0.5 ms | 1 |
| `OneHotEncoder`（120 类） | 0.5 ms | 120 |
| `OneHotEncoder(min_frequency=20)`（120 类） | 1.1 ms | 29 |
| `TargetEncoder(cv=5)`（120 类） | **44.7 ms** | 1 |

**读法**：预处理本身的成本几乎总是可以忽略（毫秒级），真正的成本是**它给下游增加的列数**——`PolynomialFeatures(3)` 的 1771 列会让线性模型的拟合时间涨两个数量级，而变换本身只要 7 ms。唯二例外是 `PowerTransformer`（λ 的 MLE 要迭代）与 `TargetEncoder`（内部 5 次折内拟合）。

## 十三、可自动化程度（面向智能体）

| 步骤 | 可自动化 | 判据/接口 |
| --- | --- | --- |
| 列类型识别 | **高** | dtype + `nunique`；但要人确认「ID 列不算数值」 |
| 缩放器选择 | **高** | 三条判据：稀疏→`MaxAbs`；有离群点（$\lvert z\rvert>3$ 的比例 >1%）→`Robust`；否则 `StandardScaler` |
| 编码器选择 | **中** | 基数阈值可自动化；「类别是否有序」**不能**（必须是业务知识） |
| 分位数/幂变换 | 中 | 偏度/峰度可测；但「要不要变成高斯」取决于下游假设 |
| 展开选择 | 中 | 列数预算（如 <500）可自动化；样条 vs 多项式需一次验证 |
| 缺失值策略 | **低** | 需要知道「缺失是否有信息」 |
| 顺序 | **高** | 本文第二节的表即规则 |
| 泄漏检查 | **高** | 「所有吃 $y$ 的步骤都在 `Pipeline` 内」可用静态检查（搜索 `fit_transform` 调用点） |

## 关联页面

- 机制：[[feature-preprocessing]]（四个作用轴、保秩/保相关/保稀疏、失效模式表）
- 来源：[[sklearn-preprocessing]]（行号、版本判定、转换损失、实测明细）
- 相邻实践：[[feature-selection-workflow]]（选择必须在缩放之后）、[[cross-validation-design-and-tuning]]（任何吃 $y$ 的步骤要在折内）、[[linear-models-as-baselines]]（正则化前的缩放）、[[nearest-neighbors-selection-and-tuning]]（距离类模型必须缩放）
- 工具：[[scikit-learn|scikit-learn 工具页]]（变换器矩阵与默认值清单）
