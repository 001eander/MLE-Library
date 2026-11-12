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
  - scikit-learn 1.9.1 的 `sklearn.preprocessing`（文档论述与 API 签名，见 [[sklearn-preprocessing]]）
---

# 特征预处理（Feature Preprocessing）

## 定义与三个推论

预处理把原始特征矩阵改成另一种表示（[[sklearn-preprocessing]], L10）。它不创造新信息，而是改变模型假设与数据尺度的匹配程度。

1. **有状态步骤是模型的一部分**：均值、分位点、类别、分箱边界与目标编码统计量只能从训练折学习。
2. **变换会改变模型**：距离、正则、邻居结构和优化条件数都受尺度影响；轴对齐树通常只依赖顺序。
3. **每种变换都有取舍**：保秩不保相关、保稀疏禁中心化、保原始列不等于换坐标。

## 有状态与无状态：12 : 4

16 个变换器中，12 个学习数据相关状态，4 个不学习：

- `Normalizer`、`Binarizer`、`FunctionTransformer`：`fit` 不学数据相关量；
- `PolynomialFeatures`：只记录由 `degree` 决定的 `powers_`，与数据无关；
- 其余 12 个学习均值、尺度、分位数、类别、分箱边界、样条结点或目标编码等状态。

无状态不等于可省略管线：为了复用顺序和接口，仍应将它们与模型一起保存。

## 四个作用轴

| 轴 | 代表 | 改变什么 |
| --- | --- | --- |
| 逐列 | 缩放器、分位数/幂变换、分箱 | 每列单位、位置、形状 |
| 逐行 | `Normalizer`、`Binarizer` | 样本长度或符号 |
| 全矩阵 | `KernelCenterer` | Gram 矩阵的中心化 |
| 列之间 | `PolynomialFeatures`、`SplineTransformer`、one-hot | 特征集合与列数 |

`scale`/`StandardScaler` 默认 `axis=0` 按列，`normalize` 默认 `axis=1` 按行。按行归一化抹掉样本长度但保留方向；按列缩放统一同一特征在样本间的尺度。

## 缩放：是否需要取决于下游

| 下游 | 缩放的原因 |
| --- | --- |
| RBF 核、KNN、局部密度 | 大量纲列支配距离 |
| 线性模型 + L1/L2 | 同一惩罚隐含同一系数尺度先验 |
| 迭代优化 | 尺度影响条件数与收敛 |
| 轴对齐树 | 只比较顺序，通常不需要 |

稀疏输入不能随意中心化：`MaxAbsScaler` 与 `StandardScaler(with_mean=False)` 是主要路径（[[sklearn-preprocessing]], L150–L158）。`MinMaxScaler` 默认 `clip=False`，测试集可越界；需要截断时才显式开启。

## 分位数与幂变换

- **分位数变换**：$G^{-1}(F(X))$ 保秩，但扭曲相关与距离（L170–L172）；正常输出有分位截断，训练范围外不能视为完整信息。
- **幂变换**：Yeo-Johnson/Box-Cox 是参数化单调映射；Box-Cox 要求严格正数据，λ 由极大似然估计，`standardize=True` 默认再标准化。

分位数是经验秩映射，幂变换是参数化、可逆、可外推的映射。不能把「保秩」理解为「保住几何」。

## 类别编码

| 对象 | 收益 | 主要代价 |
| --- | --- | --- |
| `OrdinalEncoder` | 列数少 | 强加类别顺序，只适合目标效应与编码顺序匹配的情形 |
| `OneHotEncoder` | 不强制顺序 | 列数随基数增长；`drop` 与未知类别可能共用向量 |
| `TargetEncoder` | 列数紧凑、使用目标信息 | 必须交叉拟合，否则泄漏 |

`drop` + `handle_unknown='ignore'` 会让未知类别与被 drop 类别共用表示（L450）。`np.nan` 与 `None` 也可能成为两个不同类别（L463–L489）。

`fit_transform` 的目标编码使用交叉拟合，`fit` 不做交叉拟合（L606–L616）。即使内部做了交叉拟合，只要在外层 CV 之前先对全量编码，外层测试折仍参与了统计量计算。

## 展开与离散化

- `PolynomialFeatures` 用组合数扩张：20 维二次 231 列、三次 1771 列；
- `SplineTransformer` 每列输出 `n_knots + degree - 1` 个基，默认 7 列，且有可控外推与带状结构；
- `KBinsDiscretizer` 用分箱获得分段表示，边界依赖 `strategy` 与版本；
- `Binarizer` 是阈值化的特例；
- `FunctionTransformer` 可包自定义函数，但 `validate=False` 默认不拦截非法输入。

样条的 `extrapolation='constant'` 是钉在边界，不是线性外推，也不是输出零。选择展开方式时先算列数，再评估目标与外推风险。

## 管线纪律

1. 先处理缺失与类型，再编码、缩放、展开、选择和建模；
2. 所有会 `fit` 的步骤放在训练折内；
3. supervised 选择和目标编码比普通无监督缩放更直接地泄漏 $y$；
4. 保存完整管线而不是分别保存变换与模型；
5. 记录 transformer 构造参数、输入列、类别顺序和版本；
6. 不用「无监督」推断预处理绝对不会泄漏，测试分布本身也可能是部署协议的一部分。

## 常见失效模式

- 稀疏输入直接中心化；
- 先对全量做目标编码再 CV；
- 序数编码用于非单调目标效应；
- 忽略未知类别与 drop 类别撞车；
- 把分位数变换当作保留相关结构；
- 把样条边界输出当线性外推；
- 以为示例中的 `validate=True` 就是默认；
- 把无状态理解为顺序无关紧要。

## 待解问题

- 内外层 CV 折不对齐时目标编码的泄漏量；
- 不同 `quantile_method` 的分箱边界差异；
- 类与函数的 `subsample` 默认值差异；
- `smooth='auto'` 与手设目标编码收缩；
- 低频类别缺失时的列数行为；
- 分布偏移下多项式与样条的外推风险。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-preprocessing]]、[[sklearn-feature-selection]]、[[sklearn-linear-models]]、[[sklearn-nearest-neighbors]]
- 概念：[[feature-selection]]、[[regularization-and-sparsity]]、[[cross-validation]]
- 实践：[[preprocessing-pipeline-design]]、[[feature-selection-workflow]]
