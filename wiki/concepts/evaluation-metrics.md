---
title: "评估指标与评分函数（Metrics and Scoring Functions）：先声明目标泛函，再选严格一致的评分，最后才算分数"
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - evaluation
  - metrics
  - scoring
  - proper-scoring-rule
  - decision-threshold
  - calibration
sources:
  - "[[sklearn-metrics]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-hyperparameter-tuning]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x（`make_scorer` 的 `response_method` 默认值 1.8 起为 `'predict'`；1.6/1.7 是 `'default'`）
  - scikit-learn 1.6.1（本页全部实测读数）
---

# 评估指标与评分函数（Metrics and Scoring Functions）

## 概要

本库此前把「分数」当成一个既成事实：[[cross-validation]] 讨论**怎么估**它，[[hyperparameter-search]] 讨论**怎么比**它，[[leaderboard-vs-cv]] 讨论它**与真实目标的错位**。三页都缺一个前置问题：**那个分数是哪个函数？**

来源给出的答案不是清单，而是一条**两层结构**的判据（[[sklearn-metrics]], L18–L33）：

1. **先声明目标泛函（target functional）**：预测问题的答案不是 $Y$ 本身，而是 $Y$ 的分布 $F$ 的某个性质——均值、中位数、某个分位数、众数。「Usually, the response variable $Y$ is a random variable, in the sense that there is *no deterministic* function $Y = g(X)$ … Instead, there is a probability distribution $F$ of $Y$」（L23）。
2. **再选严格一致（strictly consistent）于该泛函的评分函数**：「use a **strictly consistent** scoring function for that (target) functional」（L25）。来源把这类函数称作 *truth serum*，因为它们保证「*“that truth telling […] is an optimal strategy in expectation”*」（同上）。
3. **最好让训练损失与评估指标是同一个函数**：「it is best used for both: as loss function for model training and as metric/score in model evaluation and model comparison」（L27）。

**本库综合（一句话）**：指标不是「衡量预测好坏」的中立尺子，而是**对某个泛函的损失**；换泛函就换尺子，尺子与泛函不匹配时，「哪个模型更好」的答案会整个翻转（实测见下文第五节：分位数模型在 pinball 0.9 上比另两个好 1.9 倍，而在 $R^2$ 上是负的）。

## 1. 三层不是同一件事

来源把评分入口分成三处（[[sklearn-metrics]], L62–L66），它们的返回值与符号语义完全不同：

| 层 | 在哪里 | 返回什么 | 符号 | 谁在用 |
| --- | --- | --- | --- | --- |
| **估计器 `score` 方法** | `est.score(X, y)` | 该估计器「为它要解决的问题」预设的准则（分类器通常是 accuracy、回归器是 $R^2$） | 越大越好（按各估计器自己的定义） | 手写 `for` 循环、`cross_val_score(scoring=None)` |
| **`scoring` 参数** | `cross_val_score` / `cross_validate` / `GridSearchCV` / `validation_curve` / `LogisticRegressionCV` | 一个**统一的「越大越好」标量** | **已翻转**：distance 类指标以 `neg_*` 形式出现（L87） | 一切模型选择工具 |
| **`sklearn.metrics` 函数** | `from sklearn.metrics import …` | 原始量纲（MSE 就是 MSE） | **未翻转** | 报表、诊断、自定义评分 |

**关键实测**：`check_scoring(est, None)` 返回 `_PassthroughScorer`——即第一层被直接转调，所以「默认评分」不是一个统一函数，而是**每个估计器各自的方法**。这与 [[hyperparameter-search]] 记录的「`scorer_` 在 `scoring=None` 时是 `_PassthroughScorer`」是同一条事实，后果也相同：`OneClassSVM` 这种没有 `score` 的估计器在 `scoring=None` 下直接 `TypeError`。

**本库综合**：三层混用的典型事故是「用 `metrics.mean_squared_error` 报数（正数、越小越好），却用 `scoring=` 选模型（负数、越大越好），然后在报表里比较两个不同符号的数」。这不是笔误问题，而是**同一件事在两层的表示不同**。

## 2. 严格一致：判据与它的三种强度

来源的四列表（[[sklearn-metrics]], L33–L48）在剪藏里丢了「评分函数」一列，下表按官方 1.9.1 页重建（**外部核查**，见来源页「外部核查」节）：

| 泛函 | 严格一致评分函数 | 响应 `y` | 预测输出 |
| --- | --- | --- | --- |
| 均值（分类） | Brier score | multi-class | `predict_proba` |
| 均值（分类） | log loss | multi-class | `predict_proba` |
| 众数（分类） | zero-one loss（只 consistent，**不** strictly） | multi-class | `predict` |
| 均值（回归） | squared error（$R^2$ 与之同序） | all reals | `predict` |
| 均值（回归） | Poisson deviance | non-negative | `predict` |
| 均值（回归） | Gamma deviance | strictly positive | `predict` |
| 均值（回归） | Tweedie deviance | depends on `power` | `predict` |
| 中位数 | absolute error | all reals | `predict` |
| 分位数 | pinball loss | all reals | `predict` |
| 众数（回归） | 「no consistent one exists」 | reals | — |

三条脚注定义了三个重要的**等价/不等价**关系（[[sklearn-metrics]], L50–L54）：

- **Brier ≡ 一热编码下的平方误差**（分类）。这条在 1.8 起被写成「with one-hot encoded targets」，也是本库判定剪藏版本的那条证据。
- **0-1 损失 ≡ 1 − accuracy**，且「only consistent but not strictly consistent for the mode」。
- **$R^2$ 与平方误差同序**。

三种强度的差别，实测如下（本机 1.6.1）：

| 强度 | 含义 | 可观测后果 |
| --- | --- | --- |
| **consistent** | 真实泛函在期望上最优 | 真实 $p=0.3$ 时，报告 0.3 的 log loss（0.6101）优于 0.2（0.6378）、0.4（0.6321）、0.5（0.6931） |
| **strictly consistent** | 最优解**唯一** | 上例中四个预报可被严格排序；换成 0-1 损失，四者的 argmax 都对 → **全部同分（准确率 1.0000）** |
| **only consistent**（0-1 对众数） | 最优解不唯一 | 所有「在同一侧」的预报无法区分——这就是「accuracy 无法评价概率质量」的正式版本 |

**本库综合**：这条判据解决了一个本库反复出现的困惑——为什么「概率校准（calibration）」这类问题不能用 accuracy 回答。答案不是「accuracy 太粗糙」，而是**accuracy 对众数这个泛函只是 consistent，不 strictly consistent**，所以它对「同一侧的概率大小」结构性地无感（[[sklearn-metrics]], L52）。反过来，$R^2$ 与 MSE 是同一个泛函的两种单位，所以它们在同一数据集上**必然**同序（实测 5 个 `alpha`：MSE 376.076→2249.484 对应 $R^2$ 0.9798→0.8793）——「换指标换来不同结论」在严格一致的同一泛函内**不应该发生**；发生了就说明两个指标在算不同的东西。

## 3. 符号：`neg_*` 约定与自定义评分的默认值

`scoring` 层的约定是「higher return values are better than lower return values」，因此误差类指标以 `neg_*` 出现（[[sklearn-metrics]], L87）。实测（1.6.1）：

- `get_scorer('neg_mean_squared_error')._sign = -1`、`_score_func = mean_squared_error`；
- `get_scorer('r2')._sign = +1`、`get_scorer('accuracy')._sign = +1`。

**自定义评分必须自己声明符号**，而 `make_scorer` 的默认是 `greater_is_better=True`（实测 1.6.1 与 1.9.1 签名一致）。后果是可测量的：

| 写法 | 5 点 `alpha` 网格上选出 | `best_score_` | 最终 MSE |
| --- | --- | --- | --- |
| `scoring=make_scorer(mean_squared_error)` | **`alpha=1e4`（最差）** | 6380.4 | **6290.0** |
| `scoring='neg_mean_squared_error'` | `alpha=1e-4`（最好） | −907.9 | **774.9** |

差 8.1 倍，**没有任何警告**。这不是「负号写反」的小错，而是「把损失当收益」→ 搜索器最大化误差。

## 4. 评分算在哪一层输出上：`response_method`

字符串评分名各自绑定一个**响应方法**（`predict` / `predict_proba` / `decision_function`），实测：

| 评分名 | `_response_method` | 说明 |
| --- | --- | --- |
| `accuracy`、`r2`、`neg_mean_squared_error` | `'predict'` | 只看点预测 |
| `neg_log_loss`、`neg_brier_score`、`d2_log_loss_score` | `'predict_proba'` | 需要概率（表里标「requires `predict_proba` support」） |
| `roc_auc` | `('decision_function', 'predict_proba')` | **按序取第一个存在的**，所以「有没有 `predict_proba`」会改变它算的东西 |

这一层有两个必须知道的失败形态（实测 1.6.1，均为**静默 `nan`**）：

1. `scoring='neg_log_loss'` 用在 `SVC()`（没开 `probability=True`）上 → `[nan, nan, nan]` + 3 条 `UserWarning: Scoring failed … set to nan`，内部异常是 `AttributeError: SVC has none of the following attributes: predict_proba`。
2. `scoring='roc_auc'` 用在 3 类 iris 上 → 同样 `nan`，内部异常是 `ValueError: multi_class must be in ('ovo', 'ovr')`。多类 AUC 必须显式用 `roc_auc_ovr` / `roc_auc_ovo`（及其 `_weighted` 变体）。

**本库综合**：这两条把 [[cross-validation]] 的「失败折 = `nan`」从切分层扩展到了**指标层**——`error_score=np.nan` 不只吞掉「某折训练集只有一个类」，也吞掉「这个指标在这个模型上没有定义」。因此「CV 均值 0.87」在指标不匹配时可能是「三折全 `nan`，均值恰好是 nan-safe 的」这种荒谬情形的一半。

自定义评分还有第二层陷阱：`make_scorer(score_func, response_method=…)` 的默认值在 **1.8.0 起从 `'default'` 改成 `'predict'`**（**外部核查**：1.6/1.7 API 页是 `'default'`，1.8/1.9 是 `'predict'`）。实测（1.6.1，`'default'`）：`make_scorer(log_loss)` 在 iris 三折上给 `[nan, nan, nan]`，只有显式 `response_method='predict_proba'` 才给 `[0.142, 0.152, 0.156]`；显式 `'predict'` 回到 `nan`（内部 `ValueError: y_true and y_pred contain different number of classes 3, 2`）。**换句话说：1.8+ 的默认值对概率型自定义指标依然是错的，必须手写 `response_method='predict_proba'`。**

## 5. 决策是第二步：阈值不是模型的一部分

来源把「预测」与「决策」明确分开（[[sklearn-metrics]], L18、L20–L21、L31）：预测给出 `predict_proba`，决策把它压成一个动作，而 `predict` 就是「在 0.5 处切一刀」的那个动作。L31 把决策类评分函数指向 `confusion_matrix` 及其派生量，并把阈值调优指向官方 §3.3（**该节未入库**）。

实测（1.6.1，500 样本、10% 正类的逻辑回归）：

| 阈值 | 准确率 | 正类召回 | 报出正例数 |
| --- | --- | --- | --- |
| 0.50（= `predict`） | 0.946 | 0.460 | 23 |
| 0.30 | **0.954** | 0.540 | 27 |
| 0.10 | 0.802 | 0.720 | 121 |
| 0.05 | 0.546 | **0.840** | 261 |

同一个模型、同一份概率，四个「决策」。这解释了本库在 [[outlier-detection-selection-and-tuning]] 与 [[competition-ml-workflow]] 里反复遇到的一类问题：**指标是在某个隐式阈值下算出来的**，所以「AUC 0.93 的模型上线后召回很差」不是矛盾，而是两个不同的量。

**本库综合**：这里要区分三个东西——(a) 排序质量（AUC / 平均精度，与阈值无关）；(b) 某个阈值下的混淆矩阵派生量（准确率、F1、precision/recall）；(c) 业务代价。来源只覆盖 (a) 与 (b)，(c) 在剪藏里没有；而实践中把 (b) 的默认阈值（0.5）当 (c) 用，是本库见到的最高频误用。

## 6. 指标的可判定性质

本库从已入库的来源里能确立的性质只有四条，每条都有可验证的后果：

| 性质 | 判据 | 后果 |
| --- | --- | --- |
| **单位不变性** | $R^2$ vs MSE（同一泛函） | 同一数据集内**排序必然相同**（实测 5/5 一致） |
| **严格一致性** | 0-1 损失 vs log loss | 前者对「同一侧的概率预报」完全不可区分（准确率全 1.0000，log loss 0.61–0.69） |
| **退化定义的默认值** | `classification_report` 的 `zero_division='warn'` | 某类从未被预测 → 3 条 `UndefinedMetricWarning` + precision/recall/f1 全 **0.0**（1.6.1 与 1.9.1 默认相同） |
| **机会校正** | `balanced_accuracy_score` 的 `adjusted` | 默认 `adjusted=False`；90/10 不平衡、正类全错 → **0.4**，`adjusted=True` → **−0.2** |

**注意第 4 条与文档的错位**：剪藏保留的残片（[[sklearn-metrics]], L139）描述官方定义时说「adopt the adjusted version to ensure that random predictions have a score of 0」，但 API 默认**不**校正。想对齐「机会水平 = 0」必须显式传 `adjusted=True`。

## 7. 泛函错配会反转排名（本库实测）

这是上面所有论述的合成实验（1.6.1）：同一训练集（5 维、Laplace 噪声），三个 `HistGradientBoostingRegressor` 各对准一个泛函，在 2,000 个测试样本上按六个口径评估：

| 模型 | RMSE | MAE | MedAE | $R^2$ | pinball 0.5 | pinball 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| `squared_error`（均值） | 0.7409 | 0.5397 | 0.3940 | 0.4136 | 0.2698 | 0.2727 |
| `absolute_error`（中位数） | **0.7314** | **0.5232** | **0.3706** | **0.4286** | **0.2616** | 0.2605 |
| `quantile 0.9`（0.9 分位数） | 1.0154 | 0.8284 | 0.7260 | **−0.1012** | 0.4142 | **0.1401** |

- 均值型指标（RMSE/$R^2$/pinball 0.5）都偏好 `absolute_error` 模型（噪声是 Laplace 分布，均值与中位数接近，差异小）；
- **pinball 0.9 独占性地偏好分位数模型**（0.1401 vs 0.26，好 1.9 倍），而同一个模型在 $R^2$ 上是**负的**（比预测均值还差）。

**结论**：单一数字的「模型 A 比 B 好」在没有声明泛函时**没有意义**。这给 [[leaderboard-vs-cv]] 的「代理指标」问题补上了一个更上游的版本：不只是「代理指标与真实目标不一致」，而是**每个指标都只对一个泛函负责**。

## 失效模式清单

1. **不声明泛函就选指标**：先用「RMSE 小」选模型，再用业务要求「99% 分位不超标」验收——两个泛函，实测可让排名反转。
2. **把 `metrics.mean_squared_error` 直接包进 `make_scorer`**：默认 `greater_is_better=True`，搜索器最大化误差（实测选出最差 `alpha`，MSE 6290 vs 775）。
3. **以为 `scoring='neg_*'` 与 `metrics.*` 同号**：一个是翻转后的标量，一个是原始量纲。
4. **自定义概率指标不写 `response_method='predict_proba'`**：`make_scorer(log_loss)` 三折全 `nan`（1.6 默认 `'default'`、1.8+ 默认 `'predict'`，都错）。
5. **用 `scoring='neg_log_loss'` 却没有概率输出**：`SVC()` 上静默 `nan`，不抛异常。
6. **多类数据上用 `scoring='roc_auc'`**：静默 `nan`，必须用 `roc_auc_ovr`/`roc_auc_ovo`。
7. **把 `predict` 当模型输出的一部分**：它只是 0.5 阈值；换阈值准确率 0.946→0.954→0.802→0.546。
8. **用 accuracy 评价概率质量**：0-1 损失对众数只 consistent，不 strictly consistent——四个校准程度不同的预报准确率全相同。
9. **以为 $R^2$ 与 MSE 会给出不同排名**：同一数据集内它们必然同序（实测 5/5）；不同结论只可能在**跨数据集**归一化时出现。
10. **相信「某类没预测到」时会报警**：`classification_report` 默认 `zero_division='warn'`，只给警告并把 precision/recall/f1 置 0.0，汇总平均时被洗掉。
11. **以为 `balanced_accuracy_score` 默认做机会校正**：默认 `adjusted=False`（0.4 vs −0.2）。
12. **漏传 `mean_pinball_loss` 的 `alpha`**：默认 **0.5**，静默变成中位数损失；且估计器侧叫 `quantile`（来源自己都为此道歉，[[sklearn-metrics]], L56）。
13. **写错评分字符串**：`InvalidParameterError`（好消息，这是少数会抛错的地方）；用 `get_scorer_names()` 取权威清单。
14. **照抄文档的「the table below shows all possible values」**：实测 1.6.1 有 56 个名字，而 1.9.1 表只有 43 行；`matthews_corrcoef`、`positive_likelihood_ratio`、`neg_negative_likelihood_ratio` 可用但不在表内；`d2_log_loss_score` 在表内但 1.6.1 的注册表里没有。
15. **依赖 callable 的签名检查**：带第 4 个可选参数的 callable 被静默接受，返回常数 0.5 也能跑完整个 CV。
16. **把 `cross_validate` 的键当裸指标名**：多指标时键是 `test_accuracy`/`test_f1_macro`。
17. **用 `top_k_accuracy_score` 而不检查 `k`**：`k > n_classes` 时静默返回 1.0（实测 3 类数据 `k=4` → 1.0）。
18. **把「指标提升」直接当成「模型变好」**：先要问「提升发生在哪个泛函上、哪个阈值下、哪个响应方法上」——三者任一不同，比较就不成立。

## 关联页面

- 来源：[[sklearn-metrics]]（3.4 节，本页全部引文的出处；转换损失最严重的一份）、[[sklearn-cross-validation]]（分数怎么估）、[[sklearn-hyperparameter-tuning]]（分数怎么比）
- 概念：[[cross-validation]]（分数是分布而非数）、[[hyperparameter-search]]（`best_score_` 是最大值而非估计）、[[leaderboard-vs-cv]]（代理指标与真实目标）、[[clustering]]（ARI/AMI 的机会校正与对称性是最接近本页判据的另一组指标）、[[anomaly-detection]]（阈值 `contamination` 与 `offset_` 的同类问题）、[[regularization-and-sparsity]]（AIC/BIC 作为「不用 CV 的分数」）
- 实践：[[metric-selection-and-thresholding]]（落地流程）、[[outlier-detection-selection-and-tuning]]（阈值五步）、[[competition-ml-workflow]]（竞赛指标与本地代理的错位）、[[evaluator-design-and-reward-hacking]]（代理指标被高频优化）
- 工具：[[scikit-learn]]（`scoring`/`metrics` 矩阵、默认值第 217–230 条、版本边界）
