---
title: 正则化与稀疏性（Regularization and Sparsity）
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - regularization
  - sparsity
  - shrinkage
  - feature-selection
sources:
  - "[[sklearn-linear-models]]"
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-naive-bayes]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-decision-trees]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-neural-networks]]"
  - "[[sklearn-covariance]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x 的 linear_model 模块（推断）
  - scikit-learn 1.9.x 的 discriminant_analysis 与 naive_bayes 模块（平滑部分）
  - scikit-learn 1.7.x – 1.9.1 的 svm 模块（`C` 与 margin 部分，见 [[sklearn-svm]]）
  - scikit-learn 1.7.x – 1.9.x 的 neighbors 模块（原型收缩部分，`shrink_threshold`，见 [[sklearn-nearest-neighbors]]）
  - scikit-learn 1.9.x 的 tree 模块（树容量与 `ccp_alpha` 部分，见 [[sklearn-decision-trees]]）
  - scikit-learn 1.9.1 的 ensemble 模块（收缩与随机化预算部分，见 [[sklearn-ensembles]]）
  - scikit-learn 1.9.1 的 feature_selection 模块（零系数的下游用法部分，见 [[sklearn-feature-selection]]）
  - scikit-learn 1.8.0 – 1.9.1 的 neural_network 模块（MLP 的 `alpha` 部分，见 [[sklearn-neural-networks]]）
---

# 正则化与稀疏性（Regularization and Sparsity）

## 正则化是限制有效容量

正则化通过惩罚、约束、数据缩减或平均，让模型更难任意拟合训练集。它不只有「在损失后加一项」一种形式，也不必产生稀疏解。

可以用三个问题分类（本库综合）：

1. 限制什么对象：系数、协方差、频率、原型、树结构或候选函数？
2. 用什么机制：罚、删、平均/随机化、收缩？
3. 是否产生 0：系数零、样本零、矩阵边零，还是仅仅更平滑？

## 惩罚项与稀疏来源

| 机制 | 目标 | 是否产生零 |
| --- | --- | --- |
| $ℓ_1$ | $\alpha\lVert w\rVert_1$ | 特征系数可为 0 |
| $ℓ_2$ | $\alpha\lVert w\rVert_2^2$ | 通常只收缩，不置零 |
| ElasticNet | $ℓ_1+ℓ_2$ | 可置零，相关组行为不同 |
| $ℓ_0$ 约束 | 限制非零个数 | 直接限制支撑集 |
| ARD | 逐系数精度先验 | 通过层级贝叶斯产生稀疏 |
| 软阈值 | 对质心/系数做过零截断 | 可置零 |

Lasso 的坐标下降在惩罚项不可微处形成软阈值；Ridge 的 $ℓ_2$ 可解读为高斯先验下的后验估计（[[sklearn-linear-models]]）。ARD 则通过逐系数精度产生另一种稀疏路径。

**稀疏 ≠ 不重要**：相关特征组、尺度变化、样本扰动与求解器都可能改变哪个系数被置零。零系数只说明当前数据、模型与正则路径下的选择。

## 正则化面

### 系数与间隔

- Ridge/Lasso/ElasticNet 的 `alpha` 控制惩罚强度；
- Logistic 与 SVM 的 `C` 是逆正则，越小正则越强；
- 只有特定模型与归一化下才可能写出闭式对应，不能把 $C=1/\alpha$ 当通用换算（[[sklearn-svm]]）。

SVM 的支持向量稀疏是**样本维**稀疏，与 LinearSVC 的 L1 **特征维**稀疏不同。

### 协方差收缩

$$(1-\alpha)\hat\Sigma+\alpha T$$

LW/OAS 用不同规则选择 α，目标是 $\Sigma$ 的 MSE，不是下游准确率（[[sklearn-covariance]]）。LDA/QDA 的 `shrinkage` 与 `reg_param` 是相关但不同的参数化，不能与系数惩罚混用。

### 平滑

朴素贝叶斯的 `alpha` 控制类别条件频率的 Laplace/Lidstone 平滑，避免零概率（[[sklearn-naive-bayes]]）。`var_smoothing` 给 GaussianNB 的方差加下限，不改变非对角协方差，因此不是一般 shrinkage。

### 原型与树结构

`NearestCentroid.shrink_threshold` 对质心做软阈值，可去除噪声特征（[[sklearn-nearest-neighbors]]）。树的 `ccp_alpha` 惩罚叶子数，是结构正则（[[sklearn-decision-trees]]）。

### 集成与约束

- `learning_rate` 收缩每轮更新；
- `subsample`、`max_features`、bootstrap 用随机化换方差；
- `max_bins`、树深与叶子数限制候选函数；
- 单调/交互约束直接删除候选函数；
- HGBT 的 `l2_regularization` 作用于叶子值（[[sklearn-ensembles]]）。

### MLP 的矩阵 L2

`alpha` 只罚 `coefs_`，不罚偏置，且损失按 $n$ 归一化（[[sklearn-neural-networks]]）。它与 Ridge 的机制相似，但跨架构和跨模型不能共用一个数值刻度。

## 内置 CV、信息准则与 OOB

`RidgeCV`/`LassoCV` 把验证藏进 `fit`，`LassoLarsIC` 使用 AIC/BIC，随机森林等提供 OOB。它们是代理指标的不同来源，假设和偏差不同，不等于自动获得无偏选择（[[sklearn-hyperparameter-tuning]] 的 §3.2.5 外部核查；[[leaderboard-vs-cv]]）。

## 常见失效模式

- 把 $C$ 与 `alpha` 当同方向；
- 跨模型比较正则数值而不看目标函数归一化与尺度；
- 把 $ℓ_2$ 当成稀疏机制；
- 把 SVM 的样本稀疏当特征稀疏；
- 把平滑当协方差收缩；
- 把支持向量零当特征零；
- 把 CV 选择的 α 当成结构恢复最优；
- 忽略集成中的删、平均与约束也是正则化；
- 用「稀疏」替代特征稳定性与因果解释。

## 待解问题

- 不同目标函数中 `alpha` 的样本量补偿；
- 相关特征组下 L1 支撑集稳定性；
- 收缩终点与仿射等变；
- `var_smoothing` 与 shrinkage 的边界；
- 多输出正则的尺度；
- 集成中惩罚、随机化与约束的联合预算。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[sklearn-covariance]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]、[[sklearn-decision-trees]]、[[sklearn-ensembles]]、[[sklearn-neural-networks]]
- 概念：[[feature-selection]]、[[covariance-estimation]]、[[support-vector-machines]]、[[multi-layer-perceptron]]
- 实践：[[linear-models-as-baselines]]、[[feature-selection-workflow]]、[[hyperparameter-search-and-tuning]]
