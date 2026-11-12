---
title: 多层感知机（Multi-layer Perceptron, MLP）：参数化非线性、非凸优化与求解器契约
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - neural-network
  - mlp
  - scikit-learn
sources:
  - "[[sklearn-neural-networks]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.0 – 1.9.1 的 neural_network 模块（文档与 API 签名，见 [[sklearn-neural-networks]]）
---

# 多层感知机（Multi-layer Perceptron, MLP）

## 隐藏层把特征工程换成学表示

MLP 在输入与输出之间加入一个或多个非线性隐藏层（[[sklearn-neural-networks]], L15–L21）。它不再只是特征的线性组合，而是学习中间表示后再输出。

这让模型体积由架构决定：`coefs_` 是若干权重矩阵，而不是 KNN 的训练集。非凸损失、多个局部极小与随机初始化使结果依赖优化路径（官方丢失的优缺点折叠块）。

## `coefs_` 与 `intercepts_` 按层边界索引

`coefs_[i]` 是第 $i$ 层到第 $i+1$ 层的权重矩阵，`intercepts_[i]` 是偏置向量（L23）。列表长度是 `n_layers_ - 1`，不是隐藏层数；`coefs_[-1]` 才是输出层。

`hidden_layer_sizes=()` 没有隐藏层，只剩线性输出层；分类时输出面仍由任务决定，不能据此断言与 `LogisticRegression` 数值等价，因为损失归一化、正则与求解器不同。

## 三个输出面

| 任务 | 输出层 | 损失 | 预测形式 |
| --- | --- | --- | --- |
| 二分类 | logistic | 交叉熵 | 两列概率 + 阈值 |
| 多分类 | softmax | 交叉熵 | $K$ 列概率 |
| 多标签 | 每类独立 logistic | 逐类二元交叉熵 | 每类独立阈值 |
| 回归 | identity | 平方误差 | 数值输出，可多输出 |

多标签各类可同时为 1，与 softmax 的概率和为 1 不同（L69–L84）。`predict_proba` 不自动等于校准概率。

`MLPRegressor` 正文只写平方误差，但 1.7 起 API 已有 `loss='poisson'`，采用 log-link 并要求非负目标（API 外部核查）。这是文档落后于 API 的例子。

## `alpha` 是矩阵版 L2

官方数学节把目标写成数据损失 + $\frac{\alpha}{2n}\lVert W\rVert_2^2$。实现外部核查显示只惩罚 `coefs_`，不惩罚 `intercepts_`（[[regularization-and-sparsity]]）。

因此 `alpha` 与 Ridge 的机制相似，但作用对象是一组矩阵且损失按 $n$ 归一化；跨模型与跨架构不能直接比较同一个数值。来源建议网格 `10.0 ** -np.arange(1, 7)`，默认 `alpha=1e-4`（L148，API 外部核查）。

## 三个求解器是三种训练契约

- SGD：按梯度更新，支持在线/小批量；
- Adam：低阶矩自适应，支持在线/小批量；
- L-BFGS：近似 Hessian，不支持在线/小批量，实现中没有 `partial_fit` 入口（L113–L129，外部核查）。

来源建议小数据优先 L-BFGS，较大数据 Adam，调好学习率后 SGD + momentum；默认却是 Adam（L149，API 外部核查）。建议与默认值要分开。

## 缩放、早停与 warm_start

L137 要求缩放并对测试集复用同一个缩放器，L147 推荐 `StandardScaler` + `Pipeline`。第一层线性求和与优化都直接受尺度影响，因此缩放属于模型假设，而不是可选清洗。

`early_stopping` 从训练集内部切验证数据；`n_iter_no_change` 在开启早停时看验证分数，否则看训练损失（实现外部核查）。时间或分组结构下需要检查内部随机划分的泄漏。

`warm_start=True` + `max_iter=1` 是手动训练循环（L153–L162）：`n_iter_` 是本次 fit 的迭代数，不跨 fit 累计；权重从上轮解继续更新。它不等于 `partial_fit` 接收新样本，也不保证优化器状态与一次性训练完全相同。

## 成本与失效模式

反向传播成本为（L133）：

$$O(i\cdot n\cdot(mh+(k-1)h^2+ho))$$

加宽有 $h^2$ 项，加深与迭代是线性乘子；scikit-learn 实现没有 GPU 路径（L11）。

常见坑：

- 未缩放就训练；
- 用训练准确率判断 `alpha`；
- 忽略随机初始化与局部极小；
- 以为默认求解器就是来源建议；
- 把 `n_iter_` 当累计值；
- 把多标签当多分类；
- 忽略 `early_stopping` 的内部切分；
- 用 `predict_proba` 当校准概率；
- 期待 `feature_importances_`，MLP 不提供。

## 待解问题

- Poisson 与 `PoissonRegressor` 的数值关系与 α 尺度；
- 早停内部切分在分组/时序数据上的泄漏；
- 两种 `n_iter_no_change` 停止路径差异；
- 不罚偏置的实际影响；
- `batch_size='auto'` 下 SGD/Adam 与 L-BFGS 的差异；
- Adam 的矩估计是否跨 `warm_start` fit 保留；
- 多标签逐类阈值与概率校准；
- MLP 作为 Bagging/Stacking 基模型的成本收益。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-neural-networks]]
- 概念：[[regularization-and-sparsity]]、[[evaluation-metrics]]、[[feature-preprocessing]]
- 实践：[[mlp-selection-and-tuning]]、[[hyperparameter-search-and-tuning]]
