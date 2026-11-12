---
title: "指标选择与阈值落地的流程：先声明目标泛函，再选严格一致的评分，最后单独定阈值"
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - evaluation
  - metrics
  - scoring
  - decision-threshold
  - reporting
sources:
  - "[[sklearn-metrics]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-hyperparameter-tuning]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.6.1（本页实测读数）
  - scikit-learn 1.8.x – 1.9.x（`make_scorer` 的 `response_method` 默认值与 `d2_*` 的注册状态按 1.8/1.9 计）
---

# 指标选择与阈值落地的流程

本页只做一件事：把一个已经能算出分数的模型，接到一个**可辩护的评分口径**上。它假设 [[cross-validation-design-and-tuning]] 定的切分已经冻结（否则先回去修协议）；机制版见 [[evaluation-metrics]]，机制出处是 [[sklearn-metrics]]。

## 一、五个准入问题

在写任何 `scoring=` 之前，按顺序回答：

| # | 问题 | 答「是」时做什么 | 出处 |
| --- | --- | --- | --- |
| 1 | **评分函数是否已被外部给定**（竞赛指标、业务 KPI、合同条款）？ | 直接用，不做替换；本库只需要在本地**复刻**它，见 [[competition-ml-workflow]] | [[sklearn-metrics]], L18 |
| 2 | **预测的答案是一个数还是一整个分布**？ | 一个数 → 必须声明它是哪个泛函（均值/中位数/分位数/众数） | L23 |
| 3 | **要求的是「排序」还是「某个阈值下的动作」**？ | 排序 → AUC / 平均精度 / pinball（与阈值无关）；动作 → 混淆矩阵派生量 + 显式阈值 | L31 |
| 4 | **这个指标在模型上有没有定义**（要不要 `predict_proba` / `decision_function`）？ | 无概率输出的模型不能用 `neg_log_loss`/`neg_brier_score`；多类不能用 `roc_auc` | L87 + 实测 |
| 5 | **报的数要进哪个层级**（训练损失 / 选择分数 / 对外报表）？ | 三个层级可以是同一个函数（来源推荐），但**符号与量纲不同**，必须各写各的 | L27、L87 |

第 1 问是唯一可以让其余四问失效的一问：**竞赛/业务给了指标就用它**，即使它与你的泛函不一致——此时不一致本身要写进报告，而不是私下换掉。

## 二、按泛函选评分（不要按名字选）

| 你要预测的 | 目标泛函 | 训练损失（估计器侧） | 评估指标（`scoring=`） | 与阈值有关？ |
| --- | --- | --- | --- | --- |
| 条件均值 | mean | `squared_error` | `neg_mean_squared_error` / `r2` | 否 |
| 条件均值（计数/正偏） | mean | `poisson` / `gamma` / `tweedie` | `neg_mean_poisson_deviance` / `neg_mean_gamma_deviance` | 否 |
| 条件中位数 | median | `absolute_error` | `neg_median_absolute_error` / `neg_mean_absolute_error` | 否 |
| 某个分位数（SLA、安全库存） | quantile | `quantile` + `quantile=q` | `mean_pinball_loss(alpha=q)` 包成 `make_scorer` | 否 |
| 类别（众数） | mode | `log_loss` | `neg_log_loss` / `accuracy` / `f1` | **是** |
| 概率质量本身 | 分布 | `log_loss` / Brier | `neg_log_loss` / `neg_brier_score` | 否 |
| 排序（正类优先） | 序 | — | `roc_auc`（二类）/ `roc_auc_ovr`、`roc_auc_ovo`（多类）/ `average_precision` | 否 |
| 不平衡下的类别 | mode + 类权重 | `class_weight`（注意它同时改有效正则，见 [[scikit-learn]]） | `balanced_accuracy` / `f1_macro` | 是 |

三处必须检查的地方：

1. **训练损失与评估指标是否是同一个泛函**（来源建议同一个**函数**，L27）。实测反例：用 `squared_error` 训、用 `mean_pinball_loss(alpha=0.99)` 评，得到的结论是「所有模型都差」，因为训练目标根本不是 0.99 分位数。
2. **分位数损失两边的参数名不同**：估计器是 `quantile=`，`mean_pinball_loss` 是 `alpha=`（来源自己写了「we apologize for the different argument names」，L56）。`alpha` 的默认值是 **0.5**，漏传就静默变成中位数损失。
3. **谁在排序、谁在决策**：AUC 只关心序，因此它**不能**告诉你「阈值 0.5 下召回多少」；反过来 accuracy 只在**某个**阈值下有意义。

## 三、符号与报告格式

`scoring` 层是「越大越好」（L87），`metrics` 层是原始量纲。落地做法是**只在报告函数里做一次转换**，并且转换方向写死：

```python
from sklearn.metrics import get_scorer

def report_score(name, value):
    """把 scoring 层的数还原成 metrics 层的量纲。"""
    sign = get_scorer(name)._sign          # neg_* 是 -1，r2/accuracy 是 +1
    return value if sign == 1 else -value
```

报告最少写清六项（与 [[cross-validation-design-and-tuning]] 的报告六项对齐，只把「指标」一项展开）：

1. 泛函（mean / median / quantile q / mode）；
2. 评分函数名（字符串或 callable 的源码位置）；
3. **响应方法**（`predict` / `predict_proba` / `decision_function`）；
4. 符号约定（是否是 `neg_*`，报表里是否已还原）；
5. 决策规则（如果涉及阈值：阈值是多少、在哪份数据上定的）；
6. 退化情形的处理（某类未预测、全 `nan` 折、`k > n_classes`）。

## 四、阈值的四步法

第 5 节的事实在此有直接后果：**同一个模型在不同阈值下的准确率可以差 0.4**（实测 0.946 → 0.546）。因此阈值必须当成一个**要单独定的参数**，而不是 `predict` 的隐含默认。

1. **先看排序质量**：在验证集上算 AUC / 平均精度（与阈值无关的那部分）。若排序本身不行，调阈值只是移动错误的位置。
2. **再定先验**：阈值应由「我们能接受多少误报」或「正例占比的先验」决定，而不是由验证集上的最优准确率决定——后者在小样本上会过拟合到那一折。
3. **检查报出数**：把 `(proba >= t).sum()` 与业务容量对比（实测：t=0.5 报 23 个、t=0.1 报 121 个，样本 500）。这一步能抓住「阈值看似合理但报出量不可执行」。
4. **上线后重算**：阈值与先验绑定，先验漂移就要重算；不要把它写进模型文件当常量。

> [!warning] `predict` 不是阈值无关的
> `predict` 恒等于 `proba >= 0.5`（二类，实测逐位相同）。所以任何「用 `predict` 算的指标」都在隐含声明「0.5 是我们的决策规则」。要报告与阈值无关的质量，请用 AUC / 平均精度 / pinball，而不是 accuracy。

## 五、多指标：一个主指标 + 若干护栏

`cross_validate(..., scoring=['accuracy', 'f1_macro'])` 返回的键是 `test_accuracy` / `test_f1_macro`（实测）。落地建议：

- **主指标只有一个**，且必须与目标泛函一致；
- 其余指标是**护栏**（例：主指标 pinball 0.99，护栏 MAE 与报出率），只用于**否决**不用于排序；
- 搜索/选择只用主指标，护栏在报告阶段检查——否则等价于多目标优化，`best_score_` 的含义会变成「某个未声明的加权和」（[[hyperparameter-search]]）。

## 六、可复制的骨架

```python
import numpy as np
from sklearn.metrics import make_scorer, mean_pinball_loss, get_scorer
from sklearn.model_selection import cross_validate

# 1) 声明泛函与响应方法（分位数 → pinball；概率型 → predict_proba）
q = 0.99
pinball_q = make_scorer(mean_pinball_loss, alpha=q, greater_is_better=False)

# 2) 主指标 + 护栏，一次算出，避免不同折
cv = cross_validate(
    est, X, y, cv=splitter,
    scoring={'pinball': pinball_q, 'mae': 'neg_mean_absolute_error'},
    return_train_score=True, error_score='raise',   # 让失败折显形
)
assert not np.isnan(cv['test_pinball']).any(), "评分在某个模型/折上没有定义"

# 3) 报表只做一次符号还原
print(report_score('neg_mean_absolute_error', cv['test_mae'].mean()))
```

三处**必须显式写出**的默认值：

| 默认值 | 默认 | 为什么要写出来 |
| --- | --- | --- |
| `make_scorer(greater_is_better=…)` | `True` | 包误差函数时必须写 `False`，否则搜索器最大化误差（实测 MSE 6290 vs 775） |
| `make_scorer(response_method=…)` | 1.6/1.7 `'default'`、1.8+ `'predict'` | 概率型指标必须写 `'predict_proba'`，否则静默 `nan` |
| `cross_validate(error_score=…)` | `np.nan` | 不改就看不见「指标没定义」这件事 |

## 七、诊断信号

| 看到的 | 大概率是什么 | 先查什么 |
| --- | --- | --- |
| 分数里出现 `nan` | 指标在该模型上没有定义，或被 `error_score` 吞 | 响应方法（`predict_proba`？）；多类 AUC 是否写了 `ovr`/`ovo` |
| 三个折的分数完全相同且等于常数 | callable 写错（无签名校验） | 用 3 参数签名重写，先单独调用一次 |
| 搜索选出的模型「明显更差」 | `greater_is_better` 没翻 | `make_scorer(...)._sign` |
| 报表数与搜索数符号相反 | 两层符号没还原 | 报告函数是否用了 `_sign` |
| 某类指标是 0.0 且无报错 | `zero_division` 的退化定义 | 加 `labels=`，或显式 `zero_division=…` |
| 分位数模型在所有指标上都差 | 泛函错配（它只对准一个分位数） | 主指标是不是 pinball；护栏是否被当成主指标 |
| `top_k_accuracy_score` = 1.0 | `k` 大于类别数 | 检查 `k < n_classes` |

## 八、18 条坑

1. **按指标名选指标**（「F1 看起来更专业」），不声明泛函——排名会在换口径时反转（实测 $R^2$ 上为负的模型在 pinball 0.9 上最优）。
2. **`make_scorer(mean_squared_error)`**：默认 `greater_is_better=True`，搜索器最大化误差。
3. **忽略 `neg_*` 与原始量纲的差别**，在报表里混用两层数字。
4. **自定义概率指标不写 `response_method='predict_proba'`**：`make_scorer(log_loss)` 三折全 `nan`。
5. **`neg_log_loss` 配无概率输出的模型**：静默 `nan` + 警告，不抛错。
6. **多类用 `roc_auc`**：静默 `nan`；必须 `roc_auc_ovr`/`roc_auc_ovo`（`_weighted` 变体见来源页丢失的小节）。
7. **把 `predict` 当阈值无关的输出**：它就是 0.5。
8. **用 accuracy 评价概率质量**：0-1 损失对众数不严格一致，四个校准不同的预报准确率全相同。
9. **以为 $R^2$ 与 MSE 会给出不同排名**：同一数据集内必然同序；差异只来自跨数据集归一化。
10. **`classification_report` 的 0.0 被当真实值**：默认 `zero_division='warn'`，只给 `UndefinedMetricWarning`。
11. **`balanced_accuracy_score` 默认不做机会校正**：0.4 vs `adjusted=True` 的 −0.2。
12. **漏传 `mean_pinball_loss(alpha=)`**：默认 0.5，且估计器侧参数叫 `quantile`。
13. **相信「文档表就是全部评分名」**：1.6.1 实测 56 个 vs 1.9.1 表 43 行，两个方向都不一致。
14. **用 `matthews_corrcoef` 却发现它不在表里**：它可用、但四版文档都没列。
15. **依赖 callable 的签名报错**：多一个可选参数会被静默接受。
16. **多指标时读 `cv['accuracy']`**：键是 `test_accuracy`。
17. **`top_k_accuracy_score(k > n_classes)`**：静默 1.0。
18. **把「提升 0.3%」当结论**：先问泛函、阈值、响应方法三者是否都没变（[[cross-validation]] 的折间方差另算）。

## 九、可自动化程度

| 环节 | 自动化程度 | 说明 |
| --- | --- | --- |
| 声明泛函并选对应评分 | **中** | 可由「预测目标是什么」的表格推出，但「业务要 99% 分位」这类信息必须由人给 |
| 符号还原与报告格式 | **高** | 一个函数 + 六项字段，可机械检查 |
| 响应方法检查 | **高（必须做）** | 断言「无 `nan`」+ 检查 `_response_method` 即可 |
| 主指标/护栏分离 | 高 | 配置里区分 `primary` 与 `guards` |
| 阈值选择 | **低** | 需要业务代价或容量约束；本库没有带代价矩阵的来源 |
| 「指标变化是否意味着模型变好」 | 低 | 需要泛函、阈值、响应方法三项一致的对照实验 |

## 关联页面

- 概念：[[evaluation-metrics]]（机制：三层 API、严格一致性、响应方法、阈值）、[[cross-validation]]（分数怎么估）、[[hyperparameter-search]]（分数怎么比）、[[leaderboard-vs-cv]]（代理指标与真实目标）、[[clustering]]（ARI/AMI 的机会校正）、[[anomaly-detection]]（`contamination`/`offset_` 的阈值语义）
- 实践：[[cross-validation-design-and-tuning]]（先冻结协议）、[[hyperparameter-search-and-tuning]]（主指标如何进入搜索）、[[outlier-detection-selection-and-tuning]]（阈值五步的另一份写法）、[[competition-ml-workflow]]（外部给定指标的复刻）、[[evaluator-design-and-reward-hacking]]（代理指标被高频优化）
- 来源：[[sklearn-metrics]]（3.4 节，本页全部引文的出处）、[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]
- 工具：[[scikit-learn]]（`scoring`/`metrics` 矩阵、默认值第 217–230 条）
