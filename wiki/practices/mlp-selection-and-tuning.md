---
title: MLP 的选型与调参：先缩放、再定容量、最后选求解器
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - neural-network
  - mlp
  - scikit-learn
  - workflow
sources:
  - "[[sklearn-neural-networks]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.0 – 1.9.1 的 `MLPClassifier` / `MLPRegressor`（文档与 API 签名，见 [[sklearn-neural-networks]]）
---

# MLP 的选型与调参

## 一、先判该不该用

适合：需要参数化非线性表示、在线/小批量训练、多输出或多标签。

不适合：需要 GPU/深度架构、原生缺失值、树状特征重要性或免费概率校准。scikit-learn 明确没有 GPU 路径，也不适合其大规模深度学习场景（[[sklearn-neural-networks]], L10–L11）。

先比较线性模型、树/集成与 KNN 基线，再决定 MLP 的非凸训练成本是否值得。

## 二、缩放是硬前提

来源要求缩放并对测试集复用同一个缩放器（L137），推荐 `StandardScaler` + `Pipeline`（L147）。它直接影响第一层线性求和与优化条件数，应放在训练折内，见 [[preprocessing-pipeline-design]]。

## 三、容量从小于默认开始

- 默认 `hidden_layer_sizes=(100,)`，先尝试更小网络；
- `coefs_`/`intercepts_` 按层边界索引，长度为 `n_layers_ - 1`；
- `hidden_layer_sizes=()` 只有线性输出层，但不等于与 LogisticRegression 数值相同；
- 宽度有 $h^2$ 成本，深度与迭代是线性乘子（L133）。

先检查架构、参数形状和收敛，再扩大容量。

## 四、`alpha` 与求解器

`alpha` 是只罚 `coefs_` 的 L2，不罚偏置；来源建议 `10.0 ** -np.arange(1, 7)`，默认 `1e-4`（L148，API 外部核查）。分别观察权重范数、训练/验证损失与停止轮数，不用训练准确率单独判断正则。

| 场景 | 求解器 |
| --- | --- |
| 小数据、全批量 | 可优先 `lbfgs`，来源建议更快更好 |
| 较大数据、稳健默认 | `adam` |
| 学习率已调好、在线/小批量 | `sgd` + momentum |
| 需要 `partial_fit` | `sgd` 或 `adam`；`lbfgs` 没有该入口 |

默认 `solver='adam'` 与来源的小数据建议不同，必须显式选择。

## 五、停止条件与种子

- 默认 `max_iter=200`、`early_stopping=False`；
- `early_stopping=True` 从训练集内部切验证数据；
- `n_iter_no_change` 开启早停时看验证分数，否则看训练损失；
- 时间/分组数据需检查内部随机划分的泄漏；
- 记录多个种子的损失与验证结果，不把非凸训练当确定性拟合。

`warm_start=True` + `max_iter=1` 是手动循环：`n_iter_` 每次重新计数，权重继续更新；它不是 `partial_fit` 的新数据接口。

## 六、输出与阈值

- 二分类 logistic，多分类 softmax，多标签每类独立 logistic，回归 identity；
- 多标签可以同时多个 1，不是 softmax 多分类；
- 多标签 `predict` 默认逐类 0.5，业务阈值需另行处理；
- `predict_proba` 不自动等于校准概率；
- MLPRegressor 1.7 起支持 `loss='poisson'`，正文只写平方误差，按 API 与版本使用。

## 常见坑

- 未缩放；
- 用默认 100 神经元不检查容量；
- 把 `alpha` 当跨模型可比值；
- 忽略非凸与种子；
- 用 `n_iter_` 当累计值；
- 忽略早停内部泄漏；
- 把多标签当多分类；
- 期待 `feature_importances_`；
- 把输出概率当校准结果；
- 只用训练分数判断收敛。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 架构/α/求解器扫描 | 高 |
| 种子与停止状态报告 | 高 |
| 缩放与泄漏检查 | 中 |
| 校准与阈值设计 | 中 |
| 非凸结果解释与上线 | 低 |

运行时扫描只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-neural-networks]]
- 概念：[[multi-layer-perceptron]]、[[regularization-and-sparsity]]、[[evaluation-metrics]]
- 实践：[[preprocessing-pipeline-design]]、[[hyperparameter-search-and-tuning]]、[[metric-selection-and-thresholding]]
