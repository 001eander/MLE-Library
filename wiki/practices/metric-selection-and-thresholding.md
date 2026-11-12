---
title: 指标选择与阈值落地的流程：先声明目标泛函，再选严格一致的评分，最后单独定阈值
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - metrics
  - scoring
  - evaluation
  - decision-threshold
  - workflow
sources:
  - "[[sklearn-metrics]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-hyperparameter-tuning]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x（`make_scorer` 的 `response_method` 默认值与 API 签名，见 [[sklearn-metrics]]）
---

# 指标选择与阈值落地的流程

## 一、五个准入问题

1. 分数已由竞赛或业务外部指定吗？有则使用它。
2. 答案是一个数还是完整分布？
3. 要排序/概率，还是要执行动作？
4. 指标在该模型输出上有定义吗？
5. 分数用于训练、CV 比较、最终报告还是上线？

先声明目标泛函，再选严格一致的评分，最后单独定阈值（[[sklearn-metrics]], L18–L33）。

## 二、按泛函选评分

| 目标泛函 | 评分/损失 | 适用提醒 |
| --- | --- | --- |
| 均值 | Brier/log loss/平方误差、Poisson/Gamma/Tweedie deviance | 可比较概率或均值预测 |
| 中位数 | absolute error | 不要改用均值指标评价 |
| 分位数 | pinball loss | 估计器参数常叫 `quantile`，损失参数常叫 `alpha` |
| 众数 | zero-one loss | 只 consistent，不能评价同一侧的概率质量 |

训练损失与评估指标最好使用同一个函数（L27）。不要用 $R^2$ 或 accuracy 去弥补泛函错配。

## 三、符号只还原一次

- `metrics` 函数返回原始量纲；
- `scoring=` 统一为越大越好，`neg_*` 由 scorer 负责；
- `make_scorer` 默认 `greater_is_better=True`，包装误差时要显式 `False`；
- `response_method` 必须与指标匹配，概率指标通常要 `predict_proba`；
- `error_score=np.nan` 可能隐藏失败，应单独报告。

## 四、阈值四步法

1. **看排序**：先评估概率/分数的排序与校准，而不是直接调默认 `predict`；
2. **定先验与成本**：明确类别先验、误报/漏报成本与覆盖率约束；
3. **检查报出数**：在独立数据上同时报分数、混淆矩阵与动作数量；
4. **上线重算**：部署先验变化后重新校准阈值。

阈值属于模型外的决策层。`TunedThresholdClassifierCV` 的官方 §3.3 尚未入库，需要时应单独补来源。

## 五、一个主指标 + 护栏

主指标对应目标泛函；护栏至少覆盖：

- 失败折与 `nan`；
- 少数类/分组表现；
- 概率校准与响应方法；
- 覆盖率、延迟与资源成本；
- 选择偏差与最终独立评估；
- 阈值后的动作数量与业务成本。

## 六、报告格式

记录指标定义、泛函、响应方法、符号、splitter、逐折数组、池化或逐折计算、阈值、失败折、最终模型与版本。`best_score_` 是搜索最大值，不是无偏估计，见 [[hyperparameter-search]]。

## 常见坑

- 用 accuracy 比较概率质量；
- 用 $R^2$ 评价分位数模型；
- 包装 MSE 时漏写 `greater_is_better=False`；
- 概率 scorer 未声明响应方法；
- 把 `nan` 当正常分数；
- 用 `predict` 代替业务阈值；
- 混淆 `quantile` 与 `alpha`；
- 直接引用缺失的指标正文（当前剪藏只有 4/54 小节）；
- 只报模型分数，不报最终决策成本。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 泛函与评分映射 | 高 |
| scorer 与响应方法检查 | 高 |
| 阈值扫描与混淆矩阵 | 高 |
| 校准与独立评估设计 | 中 |
| 业务成本与上线决策 | 低 |

运行时阈值评估只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-metrics]]、[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]
- 概念：[[evaluation-metrics]]、[[cross-validation]]、[[hyperparameter-search]]
- 实践：[[cross-validation-design-and-tuning]]、[[hyperparameter-search-and-tuning]]
