---
title: 超参数搜索的落地流程：先冻结协议、再选搜索器、最后按拟合次数算预算
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - hyperparameter-tuning
  - model-selection
  - grid-search
  - randomized-search
  - successive-halving
  - workflow
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

# 超参数搜索的落地流程

## 一、五个准入问题

1. 参数真的影响结果吗？
2. splitter、评分函数、响应方法与数据版本冻结了吗？
3. 只是选一个配置，还是要无偏报告泛化？
4. 单次拟合成本是否近似随样本数/资源下降？
5. 预算是多少，是否包含 refit 与最终确认？

没有冻结协议时，搜索器只会高效地优化错误目标。

## 二、按尺度建空间

- 跨数量级参数用 `scipy.stats.loguniform` 或显式对数网格；
- 普通连续范围用连续分布；
- 全列表随机搜索不放回，`n_iter` 不能超过笛卡尔积；包含分布时才放回采样（API 外部核查）；
- 先删去明显无效或不影响结果的参数，避免浪费评估预算。

文档中的 `sklearn.utils.fixes.loguniform` 路径需在目标版本核对，不作为可移植默认值。

## 三、选搜索器

| 条件 | 选择 |
| --- | --- |
| 网格小、要完整覆盖 | `GridSearchCV` |
| 空间大、要限制候选数 | `RandomizedSearchCV` |
| 单次拟合随资源明显变便宜 | `HalvingGridSearchCV` / `HalvingRandomSearchCV` |
| 估计器自带正则路径 | 对应 `*CV`，见丢失的 §3.2.5 外部核查 |
| 小样本且 AIC/BIC 假设合适 | `LassoLarsIC` |
| 有合适 OOB 结构 | 对应集成估计器的 OOB |

Halving 第一轮也评估全部候选，减少的是单次资源，不保证总拟合次数更少。

## 四、按拟合次数算预算

- 网格：$|\Theta|\cdot k$，默认 refit 再 +1；
- 随机：$n_{\text{iter}}\cdot k$，全列表受组合数限制；
- Halving：$\sum_i n_{\text{cand},i}\cdot k$；
- 最终独立评估与最终 refit 另计。

同时报告 wall-clock、并行方式、实际资源序列与失败成本，不把拟合次数当唯一成本。

## 五、评估与 refit

- `refit=True` 是 1.9.1 默认；
- 多指标必须显式指定用于重训的指标或 callable；
- 不同 `refit` 形式会改变 `best_estimator_`、`best_score_` 等属性；
- `best_score_` 是候选最大值，不是无偏估计，需 dev/eval 或嵌套 CV；
- 部分失败可能记 `nan`，refit 失败会抛错；全部失败的精确行为需目标环境核对。

搜索的 CV 不应未经审视就与 Stacking 的折外预测共用，避免同折选参与造特征的双重偏差。

## 六、报告

至少包含：搜索空间、候选生成、splitter、评分函数、响应方法、实际候选数、拟合次数、最佳参数、`best_score_`、独立评估、失败折、最终 refit 状态与版本。

Halving 的 `cv_results_` 是候选 × 轮次的长表，含 `iter` 与 `n_resources`，不能按普通网格表处理。

## 常见坑

- 用线性均匀分布搜索跨数量级参数；
- 把 `n_iter` 当必然达到的预算；
- 把 `best_score_` 当泛化估计；
- 多指标未指定 refit；
- 把 halving 当拟合次数更少；
- 忽略失败折与最终 refit；
- 忽略 `return_train_score` 默认差异；
- 把实验性 halving 当稳定 API；
- 用同一 CV 选参、报分和 stacking 而不审查复用。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 空间与预算生成 | 高 |
| 搜索与结果汇总 | 高 |
| 失败与属性检查 | 中 |
| 独立评估与偏差审计 | 中 |
| 最终配置上线 | 低 |

运行时搜索只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-hyperparameter-tuning]]、[[sklearn-cross-validation]]、[[kaggle-getting-started]]
- 概念：[[hyperparameter-search]]、[[cross-validation]]、[[evaluation-metrics]]
- 实践：[[cross-validation-design-and-tuning]]、[[svm-selection-and-tuning]]、[[ensembles-selection-and-tuning]]
