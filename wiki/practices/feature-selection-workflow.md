---
title: 特征选择的落地流程：三类方法怎么选、阈值与个数怎么定、常见坑与可自动化程度
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - feature-selection
  - dimensionality-reduction
  - workflow
sources:
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1（文档论述与 API 签名，见 [[sklearn-feature-selection]]）
---

# 特征选择的落地流程

## 一、先判该不该选

选择只在它能降低方差、成本或接口复杂度时才有意义；它不创造信息，也不保证提升准确率。先跑无选择基线，再决定是否需要这条路径。

若需要新坐标而不是原始列，使用 [[matrix-factorization]]；若目标是处理尺度与编码，先用 [[feature-preprocessing]]。

## 二、按模型与约束选类

| 条件 | 路径 |
| --- | --- |
| 要便宜初筛、能接受单变量盲点 | 过滤式 `VarianceThreshold` / `SelectKBest` / `SelectPercentile` |
| 有 L1 或树等内生重要性 | `SelectFromModel` |
| 必须控制输出个数、能接受高成本 | RFE / SFS |
| 要让 CV 决定个数 | RFECV |
| 有多重检验目标 | `SelectFdr` / `SelectFwe`，不要把 `SelectFpr` 当校正方法 |

过滤式不见冗余，嵌入式继承模型偏好，包裹式继承 CV 方差与选择偏差。

## 三、阈值与个数的三条路

1. **领域/接口硬约束**：直接给 $k$ 或 `max_features`；
2. **模型启发式**：`SelectFromModel` 默认阈值、L1 的 `1e-5` 与其他模型的 `'mean'`；
3. **CV 搜索**：RFECV 或显式 $k$ 网格，但必须将选择过程纳入评估。

`RFE` 与 SFS 默认保留一半左右，RFECV 也可能选满。`SelectKBest` 默认 `k=10`，`SelectPercentile` 默认 10%，`GenericUnivariateSelect` 默认 `param=1e-5`，都不是中立推荐。

## 四、管线写法

 supervised 选择器必须在训练折内拟合（[[sklearn-feature-selection]], L165–L173）：

```python
Pipeline([
    ("preprocess", preprocess),
    ("select", SelectKBest(score_func=f_classif, k=k)),
    ("model", estimator),
])
```

先对全量选择再 CV 会让测试折参与列选择。目标编码等使用 $y$ 的步骤同理，见 [[preprocessing-pipeline-design]]。

报告最终列名与选择器构造，不只报列数。`prefit=True` 可用于推理，但未 `fit` 时 `threshold_` 不能按已拟合属性读取。

## 五、成本与稳定性

- 过滤式：不拟合预测模型；
- `SelectFromModel`：通常一次；
- RFE：递归拟合；
- SFS：子集 × CV；
- RFECV：RFE 路径 × 折数。

若需要稳定性，自行做重采样并统计选择频率；文档没有 stability selection 入口。不要把一次选择的集合当真值。

## 常见坑

- 先选后 CV 泄漏；
- 把 MDI 当无偏重要性；
- 忽略 L1 与树的默认阈值差异；
- 把 `SelectFpr` 当多重校正；
- 忽略互信息与 ExtraTrees 的随机性；
- 把 RFECV 当必然减维；
- 忽略 `prefit` 的诊断属性；
- 稀疏输入/输出类型未检查；
- 同源 one-hot 列未整组处理；
- 用同一 CV 选个数并报最优分。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 打分与阈值扫描 | 高 |
| 选择器与列名报告 | 高 |
| 管线与泄漏检查 | 中 |
| 稳定性与多重检验设计 | 中 |
| 最终特征语义与上线 | 低 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-feature-selection]]、[[sklearn-linear-models]]、[[sklearn-ensembles]]、[[kaggle-getting-started]]
- 概念：[[feature-selection]]、[[regularization-and-sparsity]]、[[cross-validation]]
- 实践：[[preprocessing-pipeline-design]]、[[cross-validation-design-and-tuning]]、[[competition-ml-workflow]]
