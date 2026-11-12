---
title: 评估指标与评分函数（Metrics and Scoring Functions）：先声明目标泛函，再选严格一致的评分，最后才算分数
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - metrics
  - scoring
  - evaluation
  - proper-scoring-rule
  - decision-threshold
sources:
  - "[[sklearn-metrics]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-hyperparameter-tuning]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x（`make_scorer` 的 `response_method` 默认值 1.8 起为 `'predict'`；API 签名外部核查）
---

# 评估指标与评分函数（Metrics and Scoring Functions）

## 先声明目标泛函

来源给出的判据是：若外部已指定评分函数就使用它；否则先考虑最终目标与应用，再选**严格一致（strictly consistent）**于目标泛函的评分函数（[[sklearn-metrics]], L18–L27）。

目标泛函回答「分布要压成哪个数」：

| 泛函 | 对应评分/损失 | 输出含义 |
| --- | --- | --- |
| 均值 | Brier/log loss/平方误差、Poisson/Gamma/Tweedie deviance | 条件均值 |
| 中位数 | absolute error | 条件中位数 |
| 分位数 | pinball loss | 条件分位数 |
| 众数 | zero-one loss；回归不存在一致的对应评分 | 类别众数 |

严格一致性意味着真实报告在期望上是最优策略。来源还建议训练损失与评估指标使用同一个函数（L27）。

## 严格一致的强度不同

0-1 损失对众数只是 consistent，不是 strictly consistent（L52）。它只看 argmax，因此不能区分同一侧的多个概率预报；准确率无法评价概率质量是定义使然，而不是指标「太粗糙」。

$R^2$ 与平方误差在同一数据集上给出相同排序（L54），属于单调变换。它与前者的差别是「换单位不改变排序」，不是「所有损失都能评价同一目标」。

## 三层 API 的返回值与符号不同

| 入口 | 返回什么 | 符号约定 |
| --- | --- | --- |
| 估计器 `score` | 估计器默认准则 | 依估计器 |
| `scoring=` 字符串/callable | 统一为越大越好 | scorer 负责符号转换，距离类指标可取负 |
| `sklearn.metrics` 函数 | 原始量纲 | 不做 `neg_*` 翻转 |

`neg_*` 是 scorer 的约定，不是损失函数自身突然变成「越大越好」（L87）。自定义 `make_scorer` 必须显式声明 `greater_is_better`；API 默认 `True`，直接包装 MSE 会把误差当收益。

响应方法也必须匹配：`make_scorer` 的 `response_method` 在 1.8 起默认 `'predict'`，概率指标需要显式选择 `predict_proba` 或合适的 `decision_function`（API 外部核查）。

## 预测与决策是两层

预测给出分布的泛函，决策把概率或分数压成动作（L18–L21、L31）。阈值属于模型之外：

1. 先检查概率/分数的排序质量；
2. 根据类别先验与误报/漏报成本确定阈值；
3. 在独立数据上评估最终决策；
4. 部署先验变化后重新校准阈值。

不能用默认 `predict` 代替业务阈值设计，也不能只报概率指标就宣称动作成本最优。

## 一个主指标与若干护栏

- 主指标应与目标泛函一致；
- 护栏覆盖失败折、少数类、校准、延迟、覆盖率与业务约束；
- 报告指标算在逐折、池化预测还是最终模型上；
- 搜索时 `best_score_` 是多候选中的最大值，不是无偏泛化估计，见 [[hyperparameter-search]]；
- 失败折可能因 `error_score=np.nan` 静默，必须单独计数，见 [[cross-validation]]。

## 常见失效模式

- 用 accuracy 比较概率质量；
- 用 $R^2$ 评估分位数或中位数模型；
- 混用训练损失与评估泛函；
- 包装误差函数时漏写 `greater_is_better=False`；
- 概率 scorer 未声明响应方法；
- 把指标不匹配造成的 `nan` 当正常分数；
- 把 `predict` 当业务阈值；
- 只报最终模型分数，不报告选择偏差；
- 把字符串评分表当作跨版本完整清单；
- 直接引用缺失正文中的指标定义，忘记当前剪藏只保留 4/54 小节（[[sklearn-metrics]] 转换损失）。

## 待解问题

- 1.9.1 的完整 scorer 注册表与文档表差异；
- `d2_*` 的注册历史；
- 不同响应方法对内置评分的影响；
- 多类 ROC AUC 的 `ovo` 与 `ovr_weighted`；
- `TunedThresholdClassifierCV` 与决策层衔接；
- 训练与评估使用一致评分在树/集成上的收益；
- 指标定义级来源（当前缺失 §3.4.4/§3.4.6）。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-metrics]]、[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]
- 概念：[[cross-validation]]、[[hyperparameter-search]]、[[leaderboard-vs-cv]]
- 实践：[[metric-selection-and-thresholding]]、[[cross-validation-design-and-tuning]]
