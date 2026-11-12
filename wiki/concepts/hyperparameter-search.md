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
  - evaluation
sources:
  - "[[sklearn-hyperparameter-tuning]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（用户指南 3.2 节，见 [[sklearn-hyperparameter-tuning]]）
  - scikit-learn 1.6.1（本页运行时读数）
---

# 超参数搜索（Hyperparameter Search）

本页讨论**已经定好评估协议之后**的那一层：有 $m$ 个候选配置、每个配置要花 $k$ 次拟合来估一个分数，预算有限时怎么分配、怎么比较、以及「最好」这个字在一张搜索表里究竟是什么。

与相邻页面的分工：[[cross-validation]] 讨论「一个配置的分数是怎么估出来的」（本页把它当成已解决的输入）；[[hyperparameter-search-and-tuning]] 讨论落地流程；[[sklearn-hyperparameter-tuning]] 是本页的主要依据。三者的关系可以一句话概括：**评估协议决定分数的含义，搜索只决定分数的最大值**——而最大值是一个有偏统计量。

## 一页速查

| 维度 | 内容 | 位置 |
| --- | --- | --- |
| 搜索的部件 | 估计器 × 参数空间 × 候选生成方法 × CV 方案 × **评分函数** | [[sklearn-hyperparameter-tuning]], L20–L32 |
| 三种候选分配 | 穷举 / 采样 / 逐次减半（不是三种 API，是三种预算曲线） | L28、L56–L59、L99–L108 |
| 成本口径 | 拟合次数 = 候选数 × 折数（+1 次 refit）；逐次减半是「每轮候选数 × 折数」的和 | 本页第三节 |
| 输出 | **一个配置 + 一张 `cv_results_` 表**，不是一个模型 | L47、L225–L227 |
| 「最好」的含义 | 折内分数的均值最大的那个候选；`refit` 之后才有一个全量重训的模型 | L241、L47 |
| 最大的风险 | 用同一份数据既选又报（选择偏差），幅度随候选数增长 | L286–L290 |
| 失败语义 | 部分失败 → `nan`；全部失败 → `ValueError`；refit 失败 → 永远抛错 | L296–L297 + 外部核查 |
| 默认值陷阱 | `n_iter` 被夹、`refit=True` 而非 `None`、halving 的 `return_train_score=True` 与 `cv=5` | 见 [[scikit-learn]] |

## 一、搜索 = 在评估协议外面套一层循环

来源把一次搜索拆成五项（L22–L26）：估计器、参数空间、**候选的生成或采样方法**、CV 方案、评分函数。这五项里只有第二、三项属于「搜索」，其余三项都属于「评估」。**本库综合**：这个拆法的价值在于把三类错误分开——

| 你改的是 | 产出的东西变了 | 症状 |
| --- | --- | --- |
| 参数空间（搜什么） | 答案的范围 | 「最优」不在你给的网格里，而且你不会知道 |
| 候选生成方法（怎么选点） | 答案的分辨率与成本 | 网格太粗错过窄峰；随机搜索把预算花在边界上 |
| CV 方案（怎么评分） | **问题本身** | 分数与部署表现无关（见 [[cross-validation]] 第三节） |
| 评分函数（评什么） | 目标函数 | 优化了代理指标（不平衡数据上的 accuracy） |
| 估计器 | 一切 | — |

第 1 行的后果最隐蔽：搜索**永远返回一个「最好」**，即使真实最优在网格外。本库实测过一个具体形态：`Ridge` + `loguniform(1e-3, 1e3)`，20 次与 100 次采样都把 `alpha` 推到 **1e-3 附近**（0.00132 / 0.00107），而两者 `best_score_` 完全相同（0.9948）——说明「最优在下界之外」，但搜索表不会告诉你这件事（[[sklearn-hyperparameter-tuning]] 实测 B-4）。

## 二、三种策略 = 三条预算曲线

来源给的三种工具（L28）不是并列的三个选项，而是三种**「每条候选被评估几次、每次用多少资源」**的方案：

| 策略 | 候选数 | 每条候选的评估 | 拟合次数 | 预算与维数的关系 |
| --- | --- | --- | --- | --- |
| 穷举 `GridSearchCV` | $\prod_j \lvert\Theta_j\rvert$ | $k$ 次，每次全量 | $\lvert\Theta\rvert\cdot k$（+1） | **指数**：加一个参数就乘一档 |
| 采样 `RandomizedSearchCV` | $n_{\text{iter}}$（固定） | $k$ 次，每次全量 | $n_{\text{iter}}\cdot k$（+1） | **解耦**：预算与参数个数无关 |
| 逐次减半 `Halving*SearchCV` | 由网格或 `n_candidates` 定 | 第 $i$ 轮 $n_{\text{res},i}$ 个样本、只有存活者进入下一轮 | $\sum_i n_{\text{cand},i}\cdot k$（+1） | 与候选数的关系变成「对数」 |

三条曲线各自的失效方向：

- **穷举的失效是「维数」**：预算随参数个数指数增长，于是实践里常见的反应是「把每个参数的取值改成 3 个」——而这正好把分辨率压到最粗。
- **采样的失效是「分辨率不均匀」**：均匀分布下的点在数量级上不均匀（实测 `uniform(0.001, 1000)` 100 次采样，`best alpha` 落在 4.696，完全错过 1e-3 附近的真最优）。所以「连续参数要用连续分布」这条（L75–L77）不是优化建议，而是**模型正确性**的一部分：`loguniform(1e-3, 1e3)` 才和 `[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]` 这个网格等价。
- **逐次减半的失效是「比较的时机」**：它用 $n_{\text{res},i}$ 个样本估分数，而不同候选停在不同轮次。官方配图自己就显示排名会翻转（`iter=0` 的第一名在 `iter=1` 被反超，[[sklearn-hyperparameter-tuning]] 图表核查），所以「存活者一直排在前列」只能读成「总体上稳定」。

## 三、成本应该按「拟合次数」记，但 halving 要按「拟合次数 × 样本量」记

来源对成本**一句话都没有**（L292–L294 只讲 `n_jobs`）。本库实测给出两条可用的口径（[[sklearn-hyperparameter-tuning]] 实测 B-1、C）：

$$\text{fits}_{\text{grid}} = \lvert\Theta\rvert\cdot k + [\text{refit}],\qquad \text{fits}_{\text{halving}} = \Big(\sum_i n_{\text{cand},i}\Big)\cdot k + [\text{refit}]$$

| 场景 | $\lvert\Theta\rvert$ | $k$ | grid fits | halving fits | 壁钟 |
| --- | --- | --- | --- | --- | --- |
| 3 候选、KNN、5 折 | 3 | 5 | 16 | **26** | 都不值一提 |
| 8 候选、计数器估计器、5 折 | 8 | 5 | 41 | **76** | — |
| 180 候选、RF(20 树)、3 折、2000×20 | 180 | 3 | 541（24.58s） | **811**（8.62s） | halving 快 **2.85×** |
| 同一网格、`n_jobs=-1` vs `1` | 180 | 3 | 541 | — | 24.73s → 5.80s（**4.27×**） |

两条反直觉的结论：

1. **逐次减半的拟合次数恒 $\ge$ 网格**。第一轮必须评估**全部**候选，所以 $\sum_i n_{\text{cand},i}\ge\lvert\Theta
vert$ 恒成立（实测 3 候选 → 5、8 候选 → 15、180 候选 → 270）。「用更少的候选次数换取更短的总额时间」这个直觉是错的：它做的是「**更多次、但每次更少样本**」的拟合。
2. **它唯一能省的途径是每次拟合的成本随 $n_{\text{res},i}$ 下降**。上表第三行按 `n_resources_` 与折比例推算的总样本量约为网格的 **1/31**，而壁钟只快 2.85×——差的那部分是每轮的固定开销（克隆、索引、折切分）。所以判据不是「候选多不多」，而是「**拟合时间是否近似正比于训练样本量**」；固定开销越强（建索引、编译、大数据拷贝），halving 的收益越接近消失。

因此「halving 更快」这个说法只在**资源可截断 + 候选很多**时成立；在 20–30 个候选的常规场景里，它更可能是「同分但更贵」。

## 四、搜索在验证集上「训练」，所以 `best_score_` 是最大值而不是估计

这是本节与 [[cross-validation]] 的接缝，也是本库反复出现的一条：**CV 分数是一个估计量，而搜索取的是 $m$ 个估计量的最大值**。来源用一句很短的话承认了这件事（L286）：「按各种参数设置评估来做模型选择，可以看作**用带标签的数据去『训练』网格的参数**」；对策是留出 **evaluation set**（L288–L290）。

| 概念 | 是什么 | 不是什么 |
| --- | --- | --- |
| `best_score_` | 胜出候选的**折内分数均值** | 泛化误差的估计（它是 $m$ 个估计量的最大值） |
| `best_estimator_` | 在全量数据上重训的模型（实测：折内 fit 看到 400 样本，refit 看到 500） | 「胜出候选本身」（两者只共享超参） |
| `cv_results_` | 一张「候选 × 折」的明细表 | 一张可以直接排序当结论的表（halving 里它是长表） |
| 平票 | 取**枚举顺序第一个**（实测：常数分数 → 6 行全 rank 1，`best_index_=0`） | 随机取或取最新 |

「用同一份数据既选又报」的乐观偏差幅度是**条件性**的：本库在 [[sklearn-cross-validation]] 实测的两个端点是 breast cancer **+0.0000** 与 80 样本纯噪声 **+0.0500**；本次在本节的数据上取 6 个种子做 dev/eval 切分（180 候选的 RF 网格、`cv=3`、dev 占 70%），六次运行的 `best_score_` 均值 ≈ **0.9912**、eval 分均值 ≈ **0.9846**，平均差 **+0.0067**。这三个数一起说明：偏差不是「一定有」或「一定没有」，而是**候选数 / 有效样本量 / 折数的函数**（[[hyperparameter-search-and-tuning]] 第五节给出判据）。

## 五、「最优」的比较对象是资源序列，不是数据

逐次减半把「候选 $a$ 与候选 $b$ 谁更好」变成「在**不同样本量**下谁更好」。三组本库读数说明这不是细节（[[sklearn-hyperparameter-tuning]] 实测 C）：

| 对照 | 结果 |
| --- | --- |
| 同一 180 网格，5 个种子，`factor=3` | grid 与 halving 的 `best_params_` **0/5 一致**；平均 `best_score_` 0.9187 vs 0.9162 |
| 同一网格，单次运行 | grid 0.9890 / halving 0.9871 / random(60) 0.9885，三者参数互不相同 |
| `aggressive_elimination` False → True | `n_resources_ [20,40] → [20,20,40]`，`best_params_` 从 `C=10` 变 `C=100`，`best_score_` 都是 0.85 |

三点推论：

1. **halving 的答案不是「网格最优的近似」**，而是「在给定 `factor`/`min_resources`/`max_resources` 下的锦标赛冠军」。要把它当网格最优用，需要额外的验证（这正是 [[hyperparameter-search-and-tuning]] 第七条诊断信号）。
2. **平票与近似平票在 halving 里更常见**，因为最后一轮只剩 1–3 个候选、每个候选的分数来自最多 $n_{\text{res}}$ 个样本。第三行显示两种淘汰策略给同一个分数但不同参数——说明分数在那个量级上已经分辨不出差异。
3. **「资源」可以是任意数值参数**（L144–L162 用 `n_estimators`）。这带来一个容易忽略的后果：`best_params_` 里会出现资源参数（实测 `n_estimators: 24`），而 `param_grid` 里**不能**再出现它（该约束在丢失的折叠块里，见来源页「转换损失」）。

## 六、比较多个指标时，「最好」需要被指定

来源对多重指标只有两句要求（L239–L245）：`scoring` 可以是列表或 dict；**`refit` 必须是某个指标名（或 `False`）**。本库实测补齐了三处语义（[[sklearn-hyperparameter-tuning]] 实测 B-6~B-9）：

| 配置 | `best_params_` | `best_score_` | 后果 |
| --- | --- | --- | --- |
| 单指标、`refit=True` | 有 | 有（该指标折均值） | 正常 |
| 单指标、`refit=False` | 有 | 有 | 有参数、无模型 |
| 多指标、默认 `refit` | — | — | **`ValueError`**（报错信息里是 `True was passed`，不是文档说的 `None`） |
| 多指标、`refit='f1'` | 有 | 有（**F1** 的折均值） | `cv_results_` 里没有 `mean_test_score`，只有 `mean_test_f1` |
| 多指标、`refit=False` | **不存在** | 不存在 | 想读参数只能自己从 `cv_results_` 里挑 |
| `refit=callable` | 有 | **不存在** | 自定义「最好」的定义，代价是失去统一的分数 |

这张表的用途不是记 API，而是记一条原则：**「最好」是一个需要被声明的选择，不是 API 的一个默认**。`halving` 干脆不支持多指标（实测直接 `InvalidParameterError`）。

## 七、失败是分层的，`nan` 只是第二层

来源只有一句「失败配置默认得 `np.nan`」（L296–L297）。实测三档（见来源页「关键主张 5」）：**部分失败 → `nan`**、**全部失败 → `ValueError: All the 6 fits failed.`**（与 `error_score` 无关）、**refit 阶段失败 → 永远抛错**（外部核查）。

这条分层结构有一个直接工程后果：**「搜索跑完了」不能推出「模型能训练」**，也不能推出「表里没有 `nan` 就说明配置都合法」——全部失败会直接把流程打断，而部分失败会静默留下 `nan`，随后的 `np.nanmean` 会把失败折算成「更小的评测集」（[[cross-validation]] 失效模式 6）。

## 八、搜索与「训练机制」的边界正在模糊

最后一条**本库综合**：搜索的输出（`best_estimator_`）与 Stacking 的输入（折外预测）是同一套机制的两种用法。[[ensembles]] 里 Stacking 用 `cross_val_predict` 造第二层训练数据；搜索用同样的折内拟合造「候选的分数」。两者的共同前提都是**折必须与最终评估无关**，因此：**一旦你把搜索的 `cv` 与报告的 `cv` 设成同一套划分，Stacking 与调参同时失效**。这是本库目前对「元学习器 + 调参」组合唯一确定的约束。

## 失效模式清单

1. **以为搜索一定返回「最优」**：真实最优可能在参数空间之外；搜索表不会提示（实测：`alpha` 被推到 `loguniform` 的下界）。
2. **网格每维只给 3 个值**：预算压力被转成分辨率损失，且与「随机搜索 + 连续分布」相比是双输。
3. **连续参数用列表**：`RandomizedSearchCV` 的列表是**不放回**采样，`n_iter` 大于网格大小时被静默夹住（实测 20 → 5 行）。
4. **连续参数用均匀分布**：数量级上错过最优（实测 `uniform(0.001, 1000)` 的最佳落在 4.696）。
5. **以为加无用参数「不影响效率」**（L59）：只在分布形态下成立；列表形态下照样按笛卡尔积乘。
6. **`refit` 留默认就写多重指标**：直接 `ValueError`，且报错里说的默认值是 `True` 而文档说 `None`。
7. **把 `best_score_` 当泛化误差报**：它是 $m$ 个估计量的最大值，乐观偏差随候选数增长。
8. **把 `best_estimator_` 当「胜出候选」**：它是全量重训的另一个模型（实测折内 400 样本 vs refit 500 样本）。
9. **`cv` 与报告的 CV 共用同一套划分**：选择偏差直接进入报告（[[cross-validation]] 第五节）。
10. **不看平票**：`best_index_` 取枚举顺序第一个（实测全 rank 1 时取第 0 行），网格顺序因此是结果的一部分。
11. **把 `nan` 当「这个配置差」**：它可能是某个折的训练集退化（单类、参数非法），与模型质量无关。
12. **全部失败时没有预案**：`ValueError` 会在 `fit` 阶段直接抛出，与 `error_score=np.nan` 的预期相反。
13. **在候选很少时改用 halving**：拟合次数反而更多（8 候选 41 → 76；3 候选 16 → 26）。
14. **假设 halving 会找到网格最优**：实测 5 个种子 0/5 参数一致，平均分数低 0.0025。
15. **把 halving 的 `cv_results_` 当短表处理**：它是「候选 × 轮次」的长表（同一候选出现多次，带 `iter`/`n_resources` 列）。
16. **依赖 `resource_` 属性**：halving 没有 `resource_`（只有 `min_resources_`/`max_resources_`），实测 `AttributeError`。
17. **以为 `refit=callable` 之后还能读 `best_score_`**：不存在。
18. **`GridSearchCV` 传 `random_state`**：没有这个参数，直接 `TypeError`；可复现性只能经 `cv=` 与估计器自身。
19. **`n_jobs` 当免费加速**：4.27× 是本机 180 候选网格的实测值；小网格上进程开销会抵消收益。
20. **照抄剪藏 L82 的 `from sklearn.utils.fixes import loguniform`**：1.6.1 上 `ImportError`，应改用 `scipy.stats.loguniform`。

## 关联页面

- 来源：[[sklearn-hyperparameter-tuning]]（3.2 节，本页全部数字的出处）、[[sklearn-cross-validation]]（3.1 节，被搜索的那层）、[[sklearn-svm]]（`C`/`gamma` 的指数网格）、[[sklearn-ensembles]]（预算型参数：树数、学习率）、[[kaggle-getting-started]]（人类选手的调参经验）
- 概念：[[cross-validation]]（分数作为估计量；本页把它当输入）、[[leaderboard-vs-cv]]（本地分数与真实目标的错位）、[[regularization-and-sparsity]]（内置 CV / 信息准则 = 第三种「选正则强度」的路线）、[[feature-selection]]（同样是「在候选集合里选一个」，但它选的是子集而不是配置）
- 实践：[[hyperparameter-search-and-tuning]]（落地流程）、[[cross-validation-design-and-tuning]]（先冻结评估协议）、[[svm-selection-and-tuning]]、[[ensembles-selection-and-tuning]]、[[mlp-selection-and-tuning]]、[[decision-trees-selection-and-tuning]]
- 概念补充：[[evaluation-metrics]]（`scoring` 是谁：目标泛函与严格一致评分；`best_score_` 取的最大值来自哪一层）
- 工具：[[scikit-learn]]（搜索器矩阵、默认值第 201–216 条与第 217–230 条、版本边界）
