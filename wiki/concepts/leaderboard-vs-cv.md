---
title: 排行榜与本地验证（Leaderboard vs CV）
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - cross-validation
  - evaluator-design
  - leaderboard
  - kaggle
sources:
  - "[[kaggle-getting-started]]"
  - "[[aide]]"
  - "[[autoresearch]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-cross-validation]]"
status: draft
verified: 2026-11-12
applies_to:
  - 竞赛型 MLE（Kaggle 类公开榜，2016–2019 的经验）
  - AIDE（arXiv:2502.13138v1）
  - karpathy/autoresearch（README 剪藏 2026-03-07）
  - scikit-learn 的 `*CV` 估计器与 `LassoLarsIC`（1.8.x–1.9.x，推断）
  - "`sklearn.model_selection` 的 splitter 与 `cross_*` 函数（1.7.x–1.9.1，API 签名外部核查）"
---

# 排行榜与本地验证（Leaderboard vs CV）

## 三套分数不是一回事

| 分数 | 谁计算 | 覆盖什么数据 | 用途 |
| --- | --- | --- | --- |
| 本地 CV | 参赛者自己 | 训练集内部划分 | 选特征、模型与超参 |
| Public LB | 主办方 | 测试集的一部分 | 赛季中的反馈 |
| Private LB | 主办方 | 测试集的另一部分 | 最终名次 |

[[kaggle-getting-started]] 只给名称和单次比赛的名次/分数，没有说明 Public/Private 的划分比例，因此本页不推断其份额。

来源案例中，作者按 CV 做决策，最终单模型与 Stacking 的 Private LB 名次不同（[[kaggle-getting-started]], L298、L317）。它说明本地分数与榜单分数可能错位，但不能从一次比赛推出普遍的信任规则。

## 「信 CV」的隐含前提

来源在数据不 i.i.d. 时建议相信 CV（L166、L298），但这是一条**动作规则**，不是正确性证明。它成立需要：本地切分复刻真实测试的依赖结构。

来源给出的具体做法（L331）是：

1. 找出训练与测试共享的实体（产品 ID）；
2. 分别构造共享与不共享的划分；
3. 按真实重叠比例混合，逼近 LB 的分布。

**本库综合**：这与现代分组/时序切分属于同一原则——验证集的统计结构必须与部署目标一致。可映射到 `GroupKFold`、`TimeSeriesSplit` 等，但具体映射需按任务，而不是看到非 i.i.d. 就加分层或折数。

## 对齐信号与替代解释

作者赛后观察到，开始 Stacking 后 CV 与 LB 同步提高（L313）。一种解释是折外预测降低了选择对单一切分的过拟合；另一种解释是后期改动幅度变小、噪声占比下降。来源只有一次比赛、无对照，本库保留两种解释。

## 代理指标的三种来源

### 内置 CV 与信息准则

`RidgeCV`/`LassoCV` 等把验证藏进 `fit`，`LassoLarsIC` 使用 AIC/BIC。它们减少手写 CV，但切分、LOO 假设与信息准则的条件变得不可见（[[sklearn-linear-models]]、[[sklearn-hyperparameter-tuning]] 的 §3.2.5 外部核查）。

### 固定 holdout 与单次训练指标

AIDE 在固定验证划分上搜索代码，autoresearch 用固定 5 分钟 `val_bpb`。它们提高吞吐，却会让搜索过程反复适应同一信号（[[aide]]、[[autoresearch]]）。必须区分「搜索分数」与「独立确认分数」。

### 本地 CV 本身

CV 不是天然更可靠：折间 std 不是标准误，分层可能压小方差，选择偏差与泄漏也会进入分数（[[sklearn-cross-validation]]）。

## 给提升定义噪声底

- 固定并记录切分、种子、数据版本和指标；
- 报逐折数组，而不只报均值；
- 区分折间离散、重复划分方差与最终独立测试的不确定性；
- 对小幅改动做重复或嵌套评估；
- 不用排行榜的反复提交替代独立验证；
- 不用单次实验决定方法优劣。

## 常见失效模式

- 用「信 CV」代替检查切分结构；
- 用增加折数修复错误的依赖结构；
- 把 Public LB 当最终目标反复优化；
- 把内置 CV 当无泄漏黑盒；
- 把单次最佳分数当泛化估计；
- 忽略 stacking 与搜索复用同一切分的双重偏差；
- 把折间 std 当标准误。

## 待解问题

- 按真实重叠结构重建 CV 后，CV 与 LB 的背离是否缩小；
- CV 与 LB 同步上升是否真表示指标对齐；
- 人类与 agent 的验证瓶颈是否只有速度差异；
- 如何为固定 holdout 搜索设置独立确认预算。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[kaggle-getting-started]]、[[aide]]、[[autoresearch]]、[[sklearn-linear-models]]、[[sklearn-cross-validation]]
- 概念：[[cross-validation]]、[[hyperparameter-search]]、[[evaluation-metrics]]
- 实践：[[competition-ml-workflow]]、[[evaluator-design-and-reward-hacking]]、[[cross-validation-design-and-tuning]]
