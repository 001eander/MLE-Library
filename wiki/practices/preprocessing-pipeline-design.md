---
title: 预处理管线的落地设计：顺序、默认值、诊断信号与常见坑
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - preprocessing
  - pipeline
  - feature-engineering
  - workflow
sources:
  - "[[sklearn-preprocessing]]"
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-cross-validation]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（接口与 API 签名，见 [[sklearn-preprocessing]]）
---

# 预处理管线的落地设计

## 一、五个准入问题

1. 有缺失值、类别列和稀疏列吗？
2. 下游模型是否依赖距离、正则或优化条件数？
3. 哪些步骤学习数据相关状态，哪些使用 $y$？
4. 输出列数与外推行为能否接受？
5. 训练、评估、部署是否共用同一条管线？

目标是保证**顺序正确、状态只从训练折学习、接口可复现**，而不是堆叠所有变换。

## 二、七阶段顺序

| 阶段 | 典型工作 | 注意 |
| --- | --- | --- |
| 1. 缺失值 | `SimpleImputer`（本库未入库） | 先明确缺失语义与版本；`np.nan` 与 `None` 可能不同 |
| 2. 类型拆分 | 数值/类别/稀疏/文本 | 不同列走不同分支 |
| 3. 编码 | `OrdinalEncoder` / `OneHotEncoder` / `TargetEncoder` | 顺序假设、未知类别、目标泄漏 |
| 4. 缩放 | `StandardScaler` / `MaxAbsScaler` / `RobustScaler` | 稀疏禁随意中心化；下游是否需要 |
| 5. 展开 | 分箱 / 多项式 / 样条 / 自定义 | 列数成本与外推 |
| 6. 选择 | `VarianceThreshold` / `SelectKBest` / `SelectFromModel` | supervised 选择必须折内 |
| 7. 模型 | 估计器 | 指标与阈值单独评估 |

容易搞反：目标编码在外层 CV 内、选择器在模型前、稀疏中心化在检查许可后、缩放不在树前盲目套用。

## 三、按下游选变换

| 下游 | 预处理重点 |
| --- | --- |
| 线性 + 正则 | 缩放、编码、列数与 α 尺度 |
| RBF / KNN / LOF | 尺度与度量是模型定义 |
| 树/集成 | 通常不需缩放，重点在缺失/类别接口 |
| MLP | 缩放是硬前提，检查早停内部划分 |
| 分解器 | 各自契约：PCA 不缩放、NMF 非负、ICA 白化 |
| 聚类/流形 | 尺度与邻居图，无监督也要折内协议 |

## 四、默认值与失效点

- `MinMaxScaler.clip=False`：测试集可越界；
- `QuantileTransformer` 类/函数的 `subsample` 默认不同；
- `PowerTransformer.standardize=True`：默认还会标准化；
- `OneHotEncoder.sparse_output=True`；
- `TargetEncoder` 的 `fit` 不做交叉拟合，`fit_transform` 才做；
- `SplineTransformer.extrapolation='constant'`：钉在边界；
- `FunctionTransformer.validate=False`：默认不拦截非法输入；
- `SelectFromModel` 默认阈值在 L1 与其他模型上不同；
- `cv=None` 默认 5 折，`error_score=np.nan` 可能隐藏失败。

这些默认值来自 1.9.1 API 签名，不来自本机运行时实验。

## 五、管线与 CV

所有会 `fit` 的步骤放在训练折内（[[sklearn-cross-validation]], L117）。尤其：

- supervised 特征选择；
- 目标编码；
- 分解器；
- 缩放与类别统计；
- 估计器与最终阈值。

即使目标编码内部做了交叉拟合，只要在外层 CV 之前先对全量编码，外层测试折仍参与了统计量计算。

`ColumnTransformer` 应记录每路输入列、变换器、输出列名和类别顺序。评估时把完整管线作为一个对象搜索与保存。

## 六、诊断信号

| 信号 | 优先检查 |
| --- | --- |
| 稀疏报错 | 是否中心化、格式是否 CSR/CSC |
| CV 异常好 | 目标编码/选择/预处理是否在全量拟合 |
| 分数 `nan` | 自定义函数、失败折、`error_score` |
| 类别被错认 | `drop` 与未知类别是否共用向量 |
| 输出越界 | `MinMaxScaler.clip`、分箱边界、样条外推 |
| 列数爆炸 | 多项式/one-hot/样条/多类目标编码 |
| 结果随改名变化 | 序数编码的字典序是否匹配目标效应 |

## 常见坑

- 先对全量做预处理再 CV；
- 稀疏输入直接中心化；
- 把序数编码当无假设；
- 把 `drop` 与未知类别混用；
- 把分位数保秩当保相关；
- 把样条常数外推当线性外推；
- 用 `validate=True` 示例推断默认开启；
- 只看变换器成本，不看下游列数与模型成本；
- 保存模型却不保存完整管线。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| schema 与接口检查 | 高 |
| 管线生成与列名报告 | 高 |
| 默认值与泄漏检查 | 中 |
| 变换/列数扫描 | 高 |
| 业务适配与上线 | 低 |

运行时验证只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-preprocessing]]、[[sklearn-feature-selection]]、[[sklearn-cross-validation]]
- 概念：[[feature-preprocessing]]、[[feature-selection]]、[[cross-validation]]
- 实践：[[feature-selection-workflow]]、[[cross-validation-design-and-tuning]]、[[hyperparameter-search-and-tuning]]
