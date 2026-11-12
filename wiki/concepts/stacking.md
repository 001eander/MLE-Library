---
title: 叠加（Stacking）与折外预测（OOF）：第二层学的是组合，不是建模
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - ensemble
  - stacking
  - cross-validation
sources:
  - "[[sklearn-stacking-example]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-tfms-llms-tabular]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1 的 `StackingRegressor` / `StackingClassifier`（见 [[sklearn-stacking-example]]、[[sklearn-ensembles]]）
  - 公共 OOF/test pair 池的 stacking 协议（见 [[kaggle-tfms-llms-tabular]]）
---

# 叠加（Stacking）与折外预测（OOF）

## 结构：两层、三种数据

Stacking 把「组合」本身当成一个可训练的问题：第一层是若干基模型，第二层是一个在**基模型的折外预测**上训练的元模型（[[sklearn-stacking-example]], L13）。

```
训练集 ──K 折──► 每折训练基模型 ──► 折外预测矩阵 (n × m) ──► 训练第二层
                                                              │
推理：全量重训的 m 个基模型 ──► 它们的预测 ──► 第二层 ──► 输出
```

三个容易混淆的数据面：

| 数据 | 产生方式 | 用在哪 |
| --- | --- | --- |
| 折外预测（OOF） | 每个样本由「没见过它」的那一折模型预测 | **训练第二层** |
| 全量基模型预测 | 基模型在全部训练数据上重训后预测 | 推理时的第二层输入 |
| 测试侧预测 | 各折基模型分别预测测试集后平均，或全量模型预测 | 提交/部署（做法随实现，见 [[kaggle-getting-started]], L190–L194） |

`cross_val_predict` 是产出第一类数据的标准入口；它在 [[cross-validation]] 里被标注为**不能当泛化误差指标**——在 stacking 里它不是指标，而是特征构造器。

## 为什么必须折外

第二层是在学「当基模型犯错时，该信谁多一点」。若用基模型在训练点上的预测来训练第二层，第二层看到的是**见过标签**的预测：它会把本来不可靠的成员当成可靠，因为它们在训练集上几乎不出错。

折外预测的作用是让第二层的训练输入与它在推理时看到的东西**尽可能同分布**。这条要求与 [[cross-validation]] 里「折外结果才是对未见数据的估计」是同一个原理的两种用法：一个是估计，一个是构造特征。

**它不是无条件的**：折外预测只有在「折划分与未来的数据划分同构」时才近似推理分布。有组、有用户、有时序结构而用随机 K 折时，折外预测本身已经泄漏（[[sklearn-ensembles]], L727–L829；[[cross-validation]]）。

## 第二层的三种约束

第二层的「形状」由一个选择决定：**要不要约束元权重**。scikit-learn 默认给的是最松的那一档。

| 约束 | 怎么得到 | 权重的含义 | 代价 |
| --- | --- | --- | --- |
| **无约束 + 截距** | `StackingRegressor(final_estimator=RidgeCV())`（默认） | 相对贡献 + 纠偏，**可以为负** | 共线性下权重不稳；不可读成「份额」 |
| **非负** | `final_estimator=LinearRegression(fit_intercept=False, positive=True)` | 接近份额，但和不必为 1 | 仍需自己再归一化 |
| **凸（非负 + 和为 1）** | R 的 `SuperLearner`；scikit-learn 无内置 | 严格的分数贡献 | 官方明说要用自定义估计器才能实现（[[sklearn-stacking-example]], L263） |

两个具体后果：

- **只有去掉截距，权重才是「份额」**（[[sklearn-stacking-example]], L230）。带截距时权重与截距可以互相补偿，同一组预测可以由不同的权重+截距组合得到。
- **归一化到 1 带来三条性质**：所有成员输出同一值时集成不放大不衰减；所有成员同加一个常数时集成同加该常数；少一个自由度从而在共线性下更稳（[[sklearn-stacking-example]], L259–L261）。三条都以「和恰好为 1」为前提——示例自己给出的模仿代码权不和为 1（0.9977），正文也承认这一点，所以这三条描述的是理想约束而不是那段代码。

## 元学习器同时是成员选择器

第二层的正则强度直接决定「几个成员在起作用」。同一份折外预测、同一批基模型，换第二层就换结果形态（[[sklearn-stacking-example]], L205、L221）：

| 第二层 | 对弱成员的处理 | 结果 |
| --- | --- | --- |
| `RidgeCV`（$ℓ_2$，默认） | 收缩到很小的值，可为负 | 三项全在，权重可读成「主 + 次 + 纠偏」 |
| `LassoCV`（$ℓ_1$） | **恰好压到 0** | 两项，成员选择完成 |

**不要用一组权重去解释另一组**：把一项压到 0 之后，剩下两项的权重都会重新分配（示例里 0.4488 → 0.4115、0.5476 → 0.5619）。这条提醒与 [[regularization-and-sparsity]] 里「L1 选出来的系数不是 L2 系数的无偏版本」是同一件事。

## 折外与全量的错配

这是 stacking 的固有偏差，几乎所有教程都不讲：

- 第二层是在**折外预测**上拟合的；
- 推理时喂给它的是**全量数据重训**后的基模型预测。

两者的误差结构不同：全量模型见过的数据更多，预测通常更平滑、更接近条件均值，而折外预测更「抖」。所以元权重是在一个略悲观、略嘈杂的输入分布上选的，然后被用于另一个分布。

用户指南只交代了机制（第二层用 `cross_val_predict` 的产物训练，[[sklearn-ensembles]], L769），没有讨论这个错配。scikit-learn 文档里点名相关风险的地方是 `cv="prefit"`：此时基模型不再重训，第二层直接用**全训练集**上的预测训练，而文档明确警告「若模型与被 stack 的模型用同一批数据训练，过拟合风险很高」（[[sklearn-stacking-example]] 的「被吞掉的 API 说明」一节）。

工程含义：**不要把「第二层在 OOF 上的分数」当成最终系统的分数**。要评估整个 stack，只能用外层 CV（本库把这条作为 stacking 的评估纪律，见 [[ensembles-selection-and-tuning]]）。

## 成本

结构决定的成本是「基模型 ×(K+1) + 第二层」：

- 每个基模型要训练 $K$ 次产出折外预测，再训练 1 次供推理；
- 第二层通常很便宜（线性/浅模型），但它吃掉的是全体折外矩阵；
- 调参时如果每个候选配置都重跑整个 stack，成本是乘法。

官方示例给的是同一台机器上的相对量级：HGBT 单独评估 0.24 s，整个 stack 1.16 s（[[sklearn-stacking-example]] 图 3），而结论段把「训练 stack 比选出最好的单模型贵得多」列为本示例唯一要强调的代价（L267）。

## 收益判据

四条判据，按可靠性排序：

1. **差值必须对着折间波动看。** 官方示例自己的数字是 $R^2$ 0.80±0.03（stack）对 0.79±0.03（最强单模型 HGBT）、MAE 1.49±0.14 对 1.53±0.14，正文只概括为「略好」（[[sklearn-stacking-example]] 图 3、L196）；用户指南 1.11 的示例更极端：单层 0.53、多层 0.53（[[sklearn-ensembles]], L779、L822）。**差值小于折间标准差时，不要把它写成「提升」。**
2. **成员之间必须互补。** 误差高度相关的同族模型几乎无收益：9 个同族 TFM 互相 stack 只比最好的单个高 **+5e-5**（[[kaggle-tfms-llms-tabular]], L187）。
3. **加一个「与池内都不像」的成员，价值可以等于加几百个像的。** 245 对公共 pair 的池子上，新增一个建模 $P(x\mid y)$ 的模型带来 **+4.9e-4 CV / +3.7e-4 LB**（[[kaggle-tfms-llms-tabular]], L238）。
4. **报告口径要写清是 pooled OOF 还是 fold-mean**，两者在非线性指标上不同（[[cross-validation]]）。

来源里关于收益的措辞都相当克制：用户指南说 stacking「as good as the best predictor of the base layer and even sometimes outperforms it」（[[sklearn-ensembles]], L793），官方示例说「slightly improves」。

## 与 Voting / Blending 的分工

| | Voting | Blending | Stacking |
| --- | --- | --- | --- |
| 权重 | 固定（可人工指定）/ 均匀 | 固定或简单学习 | **由第二层学** |
| 第二层训练数据 | 无 | 与基模型不相交的留出集 | **全部训练集的折外预测** |
| 数据利用 | 全量训练基模型 | 基模型只用了一部分数据 | 全量（每个样本都参与基模型训练） |
| 主要风险 | 权重与模型强弱不匹配 | 样本少时留出集太小 | 折外不近似推理分布、成本高 |

Voting 与 Stacking 的差别被官方示例一句话点出：一个是固定权重的平均，一个是通过 `final_estimator` 学出组合（[[sklearn-stacking-example]], L13）。Blending 的数据利用问题见 [[kaggle-getting-started]] 对四种集成方法的对照。

## 何时不该用

- **切分无法复刻推理分布**（分组、用户、时序）时，折外预测本身就泄漏；
- 基模型同族、误差高度相关时，收益接近 0，成本照付；
- 样本很少时，第二层的训练数据（折外矩阵）只有 $n$ 行且每行都带折间噪声；
- 需要逐成员解释或审计时，凸约束/无截距才是可读的，而这在 scikit-learn 里要自己写估计器；
- 预算只够跑一两个模型时，「选最好的单模型 + 调参」通常更划算（这是官方示例结论段的意思，[[sklearn-stacking-example]], L267）。

## 常见失效模式

- 用基模型在训练集上的预测（而非折外预测）训练第二层；
- 用同一套折既训练第二层、又用来报告分数，且不区分内层/外层；
- 把 `cross_val_predict` 的输出当成泛化误差（它是特征，不是指标）；
- 把「堆了很多成员」当成「第二层变强」，忽略成员间的误差相关；
- 把第二层在 OOF 上的分数当成系统的分数，忘了推理时喂进去的是全量基模型；
- 用 L1 第二层的权重去解释 L2 第二层的权重（去掉一项会重新分配）；
- 只报均值不报折间波动，把 0.01 的差写成提升；
- 有组/时序结构却用随机 K 折产出折外预测；
- 只在默认 `cv` 上验证，不检查 `cv` 的 splitter 是否洗牌（scikit-learn 默认 `cv=None` → 5 折、`shuffle=False`，[[sklearn-stacking-example]]「被吞掉的 API 说明」）。

## 待解问题

- 折外预测与全量重训预测之间的分布错配，在什么样本量/模型族下会实质改变元权重？本库目前只有机制描述，没有量化。
- 第二层的约束（无约束 / 非负 / 凸）与正则（$ℓ_1$/$ℓ_2$）在什么条件下产生可分辨的差异？
- 大规模 pair 池（数百个高度相关成员）下，权重集中在少数成员是必然还是可改善（正则强度、分组 stack、非负约束）？
- 内层折与外层折不对齐时的泄漏量有多大？（竞赛场景常见：公共 pair 池由他人的折划分产出）
- 分类任务里 `predict_proba` 两列共线只留一列的做法（[[sklearn-ensembles]], L795–L796）与回归的 `predict` 在第二层的可加性上是否等价？

## 关联页面

- 来源：[[sklearn-stacking-example]]、[[sklearn-ensembles]]、[[kaggle-tfms-llms-tabular]]、[[kaggle-getting-started]]
- 概念：[[ensembles]]、[[cross-validation]]、[[regularization-and-sparsity]]、[[leaderboard-vs-cv]]、[[evaluation-metrics]]
- 实践：[[ensembles-selection-and-tuning]]、[[competition-ml-workflow]]、[[cross-validation-design-and-tuning]]
- 工具：[[scikit-learn]]
