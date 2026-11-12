---
title: 超参数搜索（Hyperparameter Search）：把预算分配给候选
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - hyperparameter-tuning
  - model-selection
  - grid-search
  - randomized-search
  - successive-halving
sources:
  - "[[sklearn-hyperparameter-tuning]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（用户指南 3.2 节与 API 签名，见 [[sklearn-hyperparameter-tuning]]）
---

# 超参数搜索（Hyperparameter Search）

## 五个部件

一次搜索由估计器、参数空间、候选生成、CV 方案与评分函数组成（[[sklearn-hyperparameter-tuning]], L20–L32）。其中只有候选生成属于「搜索」本身；其他四项决定你在回答什么问题。

先冻结评估协议，再谈搜索算法。若评分函数、切分或目标泛函改变，「最优」也变成另一件事。

## 三条预算曲线

| 策略 | 拟合次数口径 | 资源分配 | 主要边界 |
| --- | --- | --- | --- |
| 穷举 | $|\Theta|\cdot k$，默认 refit 再 +1 | 每个候选全资源 | 维数与网格大小乘积膨胀 |
| 随机采样 | $n_{\text{iter}}\cdot k$ | 每个候选全资源 | 采样分布决定覆盖；全列表时不放回且预算受笛卡尔积限制 |
| 逐次减半 | $\sum_i n_{\text{cand},i}\cdot k$ | 存活候选资源递增 | 第一轮也评估全部候选，减少的是单次资源而非拟合次数 |

逐次减半的递推式见 L125–L143。它适合单次拟合成本近似随资源下降的场景；若成本不随样本数或数值参数下降，资源递增不一定节省时间。

## 参数空间比搜索器更先决定结果

- 连续尺度参数宜用连续分布或对数均匀分布（L75–L77）；
- 全列表随机采样不放回，`n_iter` 被限制到笛卡尔积大小；包含分布时才放回采样（API 外部核查）；
- 加入无用参数并不一定「零成本」：全列表会占用候选名额，分布采样也消耗评估预算；
- 先判断参数是否影响结果，再决定是否进入网格（L32）。

`loguniform` 的可移植来源是 `scipy.stats.loguniform`；文档中的 `sklearn.utils.fixes.loguniform` 路径需在目标版本核对。

## `best_score_` 是最大值，不是无偏估计

搜索从 $m$ 个 CV 估计量中取最大值。它衡量「这批候选里谁最高」，不等于该配置的泛化估计。dev/eval 切分与嵌套 CV 是两种把选择过程纳入评估的方式（L284–L290；[[cross-validation]]）。

多指标下「最好」必须显式声明：`refit` 选择用于重训的指标或 callable。官方 1.9.1 默认 `refit=True`，多重指标下不能依赖文档中过时的 `None` 描述。不同 `refit` 形式会改变可用属性，应检查 `best_estimator_`、`best_score_` 等是否存在。

## 失败有三层语义

- 部分折失败：默认可能记 `nan`，其他候选继续；
- 全部失败：具体搜索器行为需在目标版本核对，不能假设会返回全 `nan` 表；
- refit 失败：官方 API 说明 `error_score` 不影响 refit，阶段会抛错。

因此「搜索跑完」与「最终模型可训练」是两个检查点。报告应包含失败候选数、失败折数与最终 refit 状态。

## Halving 的结果表与资源语义

Halving 的 `cv_results_` 是长表：每行是候选 × 一轮，含 `iter` 与 `n_resources` 等列（L225–L227）。两个 halving 搜索器默认 `return_train_score=True`，与另两个搜索器的默认不同（API 外部核查）。

`resource` 默认是样本数，也可以是 `n_estimators` 等数值参数。`min_resources='exhaust'` 旨在最后一轮用满资源；`n_candidates='exhaust'` 与 `min_resources='exhaust'` 的组合有互斥约束（官方折叠块）。

## 常见失效模式

- 在评估协议未冻结时调参；
- 把 `n_iter` 当作所有情况下都能达到的拟合预算；
- 用线性均匀分布搜索跨数量级参数；
- 把 halving 当作「拟合次数更少」；
- 把 `best_score_` 当泛化估计；
- 多指标未声明 refit；
- 忽略失败折与 refit 失败；
- 把长表当普通网格结果表；
- 忘记实验性 halving 的 import 与版本约束。

## 待解问题

- 全列表不放回采样的版本史；
- `sklearn.utils.fixes.loguniform` 的目标版本可用性；
- 全部失败与部分失败的精确分界；
- `refit=callable` 的属性契约；
- halving 与分组/时序切分的资源语义；
- 平票顺序与评分舍入的报告约定。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-hyperparameter-tuning]]、[[sklearn-cross-validation]]、[[kaggle-getting-started]]
- 概念：[[cross-validation]]、[[evaluation-metrics]]、[[leaderboard-vs-cv]]
- 实践：[[hyperparameter-search-and-tuning]]、[[cross-validation-design-and-tuning]]
