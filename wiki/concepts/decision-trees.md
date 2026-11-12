---
title: 决策树（Decision Trees）：递归划分、贪心最优性与代价复杂度剪枝
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - decision-tree
  - cart
  - pruning
sources:
  - "[[sklearn-decision-trees]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.x 的 tree 模块（文档与 API 签名，见 [[sklearn-decision-trees]]）
---

# 决策树（Decision Trees）

## 模型是一组轴对齐规则

决策树通过递归划分把特征空间切成矩形，在每个矩形里输出常数，即 **piecewise constant approximation**（[[sklearn-decision-trees]], L10、L31）。

「非参数」在这里表示参数个数随数据生长，并不表示没有超参。深度、叶子大小、叶子数、最小不纯度下降与 `ccp_alpha` 在拟合前限制容量；拟合后再产生每个节点的特征和阈值。

四个直接推论：

1. **查表式推理**：预测沿根到叶的路径，成本是 $O(\text{depth})$，只有平衡树才近似 $O(\log n)$。
2. **不能外推**：训练范围外仍落在边界叶子的常数预测。
3. **对单调变换免疫**：分裂只比较特征大小，不需要标准化；旋转会改变轴对齐结构。
4. **矩形拼贴**：XOR、parity 等需要联合特征的概念难以用单次分裂表达，通常需要更深树、集成或特征交叉。

## 贪心与不稳定

最优树是 NP-complete，实践算法采用贪心切分，不能保证全局最优（L32）。节点 $m$ 的候选切分按左右子节点不纯度加权和评分（L249–L253）：

$$G(Q_m,\theta)=\frac{n_m^{\text{left}}}{n_m}H(Q_m^{\text{left}})+\frac{n_m^{\text{right}}}{n_m}H(Q_m^{\text{right}})$$

- `splitter='best'`：穷举候选阈值；
- `splitter='random'`：采样随机阈值，ExtraTree 默认走这条路径（L255–L260）。

数据小扰动可能改变整棵树，因此不稳定不是异常，而是 greedy + 高容量结构的正常后果，也是随机化与集成的动机。

## 输出与权重

- 分类：`predict` 返回叶子多数类，平票取 `classes_` 中索引最小者（L57）。
- `predict_proba`：叶内类别比例；加权时应理解为加权比例，而不是样本个数之比（L59，API 外部核查）。
- 回归：MSE 与 Poisson 输出叶内均值，MAE 输出叶内中位数（L292–L314）。
- 多输出：叶子存向量，准则取各输出平均下降，`predict_proba` 返回 list of arrays（L156–L170）。

`min_samples_split` 直接数样本，与 `sample_weight` 无关（L221）；权重感知停止条件需使用相应参数。Poisson 要求 $y\ge0$，MAE 的成本说明随 1.8 的算法优化而变化，不能跨版本引用固定倍数（L316，版本史外部核查）。

## 缺失值

训练时，树对每个候选阈值分别评估 NaN 走左与走右，选择更优方向（L320–L324）。预测时（L326–L366）：

1. 默认按训练时学到的方向；
2. 左右准则评估相同则走右；
3. 训练时该特征未见 NaN，则走样本更多的子节点。

ExtraTree 还会随机分配缺失值候选方向。具体准则是否支持 NaN 随版本变化：1.9 的版本史扩展了 `absolute_error` 与单调约束组合，不能只写「所有树都原生支持缺失值」。

## 前剪枝与后剪枝

常见前剪枝旋钮：

- `max_depth` 与 `max_leaf_nodes`；
- `min_samples_split`、`min_samples_leaf`；
- `min_impurity_decrease`；
- `max_features` 与 splitter 随机性。

后剪枝使用代价复杂度（L368–L378）：

$$R_\alpha(T)=R(T)+\alpha|\tilde T|$$

惩罚的是叶子数；scikit-learn 的 $R(T)$ 是终端节点样本加权不纯度。`ccp_alpha=0.0` 默认不剪枝，`cost_complexity_pruning_path` 可生成 α 路径（API 外部核查）。

树大小也限制交互阶数：深度 $h$ 的树最多表达 $h$ 阶交互（[[sklearn-ensembles]], L337–L343）。

## 常见失效模式

- 把白盒规则当作因果关系；
- 用平衡树的对数推理成本概括退化单链树；
- 忽略类别不平衡与加权对叶子输出的影响；
- 用 `min_samples_split` 期待它读取权重；
- 把训练未见 NaN 时的「多数分支」当成业务规则；
- 忽略 float32 内部转换可能合并近邻阈值（L224）；
- 只调 `ccp_alpha`，不先限制树容量；
- 把某版本的缺失值支持或 MAE 成本外推到其他版本。

## 待解问题

- 多输出平均下降是否受输出量纲支配；
- 叶内概率的校准与可靠性；
- `ccp_alpha` 路径与 CV 的联合选择；
- 前剪枝参数的交互；
- 未见 NaN 的多数分支在分布偏移下的偏置；
- 高维稀疏数据的切分成本。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-decision-trees]]
- 概念：[[ensembles]]、[[regularization-and-sparsity]]、[[feature-selection]]
- 实践：[[decision-trees-selection-and-tuning]]、[[ensembles-selection-and-tuning]]
