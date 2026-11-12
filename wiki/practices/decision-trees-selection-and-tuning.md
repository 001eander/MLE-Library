---
title: 决策树的选型与调参：前剪枝、代价复杂度剪枝、缺失值与 float32
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - decision-tree
  - cart
  - pruning
  - workflow
sources:
  - "[[sklearn-decision-trees]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.x 的 tree 模块（参数语义与默认值按 1.9.1 API 页核对）
---

# 决策树的选型与调参

## 一、先判适配

适合：需要轴对齐规则、非线性交互、可视化、数值与多输出、某些版本下的原生缺失值。

不适合：需要平滑外推、稳定全局函数、稀疏高维下的强线性基线，或希望数据小扰动后结构不变。

先跑 [[linear-models-as-baselines]] 与 `NearestCentroid` 等轻基线，再决定树是否提供额外收益。

## 二、先可视化，再定容量

来源建议 `max_depth=3` 起步、`min_samples_leaf=5` 起步（[[sklearn-decision-trees]], L218–L220）。流程：

1. 用小深度拟合并检查实际切分；
2. 逐步增加深度/叶子数；
3. 同时观察训练与验证表现及树形稳定性；
4. 不要只看单个最终分数。

`max_leaf_nodes` 与深度共同限制交互阶数：深度 $h$ 最多表达 $h$ 阶交互（[[sklearn-ensembles]], L337–L343）。

## 三、前剪枝旋钮

| 旋钮 | 作用 | 注意 |
| --- | --- | --- |
| `max_depth` | 限制深度与交互阶数 | 先小后大 |
| `max_leaf_nodes` | 限制叶子数 | 与深度不是同一形状约束 |
| `min_samples_split` | 最小可再分样本数 | **直接数样本，不看 `sample_weight`**（L221） |
| `min_samples_leaf` | 叶子最小样本数 | 影响叶内估计方差 |
| `min_impurity_decrease` | 最小切分收益 | 太小会保留弱切分 |
| `max_features` | 候选特征子集 | DecisionTree 与 ExtraTree 默认不同 |
| `splitter` | best/random | ExtraTree 默认 random |

权重场景需要权重感知停止条件，不要用 `min_samples_split` 代替它。

## 四、后剪枝：`ccp_alpha` 路径 + CV

$$R_\alpha(T)=R(T)+\alpha|\tilde T|$$

- 默认 `ccp_alpha=0.0`，不剪枝；
- 用 `cost_complexity_pruning_path` 生成候选 α；
- 在训练协议内用 CV 选择；
- 报告最终树深、叶子数与 α，不只报一个分数。

前剪枝控制「能长多大」，后剪枝控制「长完后删哪些子树」，两者不要混为一步。

## 五、缺失值与权重

训练时每个阈值分别评估 NaN 左右方向；预测沿用（L320–L324）。特别注意：

- 左右准则评估相同 → 走右；
- 训练未见 NaN → 走样本更多子节点；
- 这不是基于特征值的业务规则，部署漂移时需要审查；
- 具体准则支持随版本变化，1.9 的版本史扩展了 `absolute_error` 与单调约束组合。

`predict_proba` 在加权时是加权比例；平票按 `classes_` 的最小索引。多输出的 `predict_proba` 返回 list of arrays。

## 六、float32 与稀疏输入

内部会转换为 `np.float32` 并可能复制数据（L224–L225）：相邻 float64 阈值可能合并，审计精确阈值时要保留这一事实。极稀疏输入按来源建议在 fit 前转 `csc_matrix`、predict 前转 `csr_matrix`。

复杂度要区分每节点、整树与单次推理：推理是 $O(\text{depth})$，只有平衡树才近似对数，退化单链不是。

## 常见坑

- 用 `min_samples_split` 期待读取权重；
- 把 `predict_proba` 当校准概率；
- 忽略类别不平衡与加权；
- 把多数分支的 NaN 规则当业务判据；
- 把某版本的缺失值支持外推；
- 忽略 float32 阈值合并；
- 只看 `ccp_alpha` 不看前剪枝；
- 用 iris 静态图当作每次拟合的固定结果，注意陈旧 `iris.svg` 与 doctest 不是同一棵树（[[sklearn-decision-trees]] 分歧 4）。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 深度/叶子数/α 路径扫描 | 高 |
| 输出形状与缺失值接口检查 | 高 |
| 树形稳定性与切分解释 | 中 |
| 准则、权重与部署适配 | 中–低 |
| 最终上线与外推判断 | 低 |

运行时路径与数值对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-decision-trees]]
- 概念：[[decision-trees]]、[[regularization-and-sparsity]]、[[ensembles]]
- 实践：[[ensembles-selection-and-tuning]]、[[linear-models-as-baselines]]、[[preprocessing-pipeline-design]]
