---
title: 集成方法（Ensembles）：平均降方差、boosting 降偏差、元学习器与随机化预算
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - ensemble
  - gradient-boosting
  - random-forest
  - bagging
  - stacking
  - adaboost
sources:
  - "[[sklearn-ensembles]]"
  - "[[sklearn-outlier-detection]]"
  - "[[ft-transformer]]"
  - "[[kaggle-tfms-llms-tabular]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1 的 ensemble 模块（文档与 API 签名，见 [[sklearn-ensembles]]）
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13；仅“表格深度模型的集成”一节）
  - 公共 OOF/test pair 池的 stacking 协议（cuML 逻辑回归 `C=1`，见 [[kaggle-tfms-llms-tabular]]）
---

# 集成方法（Ensembles）

## 两条轴：平均与顺序

来源给出的选型判据是：bagging 适合强而复杂的模型，boosting 适合弱模型（[[sklearn-ensembles]], L574）。

| | 平均式 | 顺序式 |
| --- | --- | --- |
| 代表 | Bagging、RF、Extra-Trees、Voting、Stacking | GBDT、HGBT、AdaBoost |
| 主要收益 | 降低方差 | 降低偏差 |
| 基模型 | 强、复杂、可独立训练 | 弱、偏差大、逐轮纠错 |
| 训练 | 可并行 | 每轮依赖上一轮，难并行 |
| 预算 | `n_estimators`、特征/样本随机性 | 轮数 × 学习率、树大小、早停 |

方差不是会降到零：若基模型误差相关，平均能消除的部分有限。Boosting 也不是轮数越多越好，后期会过拟合，需要早停或收缩。

## HGBT 与旧 GBDT：先选实现

`HistGradientBoosting*` 使用直方图分箱，`GradientBoosting*` 使用精确切分（[[sklearn-ensembles]], L22–L28、L249–L261）。

| | HGBT | `GradientBoosting*` |
| --- | --- | --- |
| 单节点切分 | $O(n_f n)$ | $O(n_f n\log n)$ |
| 缺失值 | 原生 | 不支持 |
| 类别特征 | 原生 | 不支持 |
| 单调/交互约束 | 支持 | 不支持 |
| 早停 | 大样本默认开启 | 默认关闭 |
| 小样本 | 分箱可能过粗 | 官方丢失的主题框建议优先考虑 |

`max_bins` 默认 255；源码注释中的 `max_bins + 1` 对应数值箱 + 缺失箱，因此「256 bins」与「255」不是同一计数口径。

## HGBT 的结构能力

- **缺失值**：每个阈值同时评估 NaN 左右方向；训练未见 NaN 时预测走样本更多分支（L76–L105）。
- **类别特征**：支持布尔掩码、索引、列名与 `from_dtype`；基数检查是 `>`，所以允许等于 `max_bins`（L140–L166，源码外部核查）。
- **样本权重**：乘梯度与 Hessian，但分箱阶段的分位数不看权重（L113–L136）。
- **单调约束**：只约束其他特征不变时的偏效应，不支持多类与类别特征（L174–L209）。
- **交互约束**：`interaction_cst` 指定允许交互的特征组，未列出的特征自成一组（L217–L228）。

这些约束不是系数惩罚，而是直接删除部分候选函数。

## RF 与 Extra-Trees

RF 的随机性来自 bootstrap 与节点特征子集，目的在让树间误差解耦并降低方差（L432–L454）。scikit-learn 平均的是 `predict_proba`，不是硬投票（L442），所以概率是树间平均，仍不自动等于校准概率。

Extra-Trees 进一步随机化阈值（L456–L489）。`max_features` 默认随任务变化：RF 分类 `'sqrt'`、回归 `1.0`；ExtraTree 的默认 bootstrap 与 RF 不同（L493，API 外部核查）。

## Bagging、Voting、Stacking、AdaBoost

- **Bagging**：四种抽样变体由样本/特征是否有放回抽取组合而成（L572–L596）。
- **Voting**：硬投票多数类，平票按类标签升序；软投票平均概率，要求基模型提供 `predict_proba`（L598–L693）。
- **Stacking**：用折外预测训练第二层，CV 从评估手段变成训练机制（L727–L829）。二分类时每个基模型的两列概率共线，只保留一列（L795–L796）。
- **AdaBoost**：逐轮调整样本权重；默认分类器是 stump，回归器是 `max_depth=3` 的回归树（L831–L866，API 外部核查）。

Stacking 的关键不是「第二层更复杂」，而是折外预测是否近似未来数据。若训练集有组或时间结构，随机 K 折可能让折外预测泄漏，见 [[cross-validation]]。

## 大规模 stacking：245 个 pair 的工程细节

[[kaggle-tfms-llms-tabular]] 给了一个「第二层极简、第一层极宽」的实例：245 个来自社区共享的 OOF/test pair，全部用同一套 `StratifiedKFold(5, shuffle=True, random_state=42)` 生成，因此可以放进同一个逻辑回归（L160–L168）。它的协议可以整段照抄：

1. **统一到 logit 尺度**：概率先 clip 到 `1e-15`，再 clip 到 $\pm30$；已经是 logit 的 pair 直接用；
2. **折内 refit**：在 4 折的 OOF logit 上拟合 L2 逻辑回归（cuML，`C=1`），预测留出的第 5 折；
3. **测试侧平均 5 个折模型**，顺带抹掉 cuML 的 run-to-run 噪声。

三个可直接迁移的读数：

| 观察 | 数字 | 含义 |
| --- | --- | --- |
| 单对最好 → 245 对 stack | 0.9617 → 0.96218（+4.8e-4） | 宽而强的基模型池确实抬高上限 |
| **单个新模型 → 245 对 stack** | 0.96218 → 0.96266（**+4.9e-4**） | 加**一个**与池内都不像的模型，和加 245 个互相像的模型同量级 |
| 9 个 TFM → 它们自己的 stack | 0.96115（最好单个）→ 0.96120（**+5e-5**） | 同族模型的误差高度相关，stack 几乎不收益 |

作者的两个判断值得记：**a)** 提升来自「准确且与其它一切都很不像」——该模型建模 $P(x\mid y)$ 而不是 $P(y\mid x)$，见 [[generative-classification]]；**b)** 最终 stack 里 10 个 GPT 系 pair 承担约 **80%** 的输出，245 个公共 pair「largely cancel each other」（L240）。也就是说，**宽池的作用主要是提高第二层的下限与稳定性，而不是持续贡献增量**。

**这条经验的边界**：① pair 池的 CV 天生偏乐观——部分公开成员在被打分的那一折上早停（L168，见 [[leaderboard-vs-cv]]）；② 全池共享同一套折划分，第二层无法通过换划分来纠错；③ 「承担 80% 输出」没有给出权重口径，不能直接当「贡献 80% 增益」读。

## 表格深度模型的集成：收益取决于基模型是否已近似集成

[[ft-transformer]] 的协议是 **3 组 × 5 seeds**：把 15 个单模型分成 3 组，组内平均预测，得到 3 个集成（[[ft-transformer]], p. 6）。把表 2 的单模型均值与表 3 的集成均值按 score 方向比较（本库计算，非来源表格）：

| 模型 | 平均相对改进 | 中位相对改进 | 典型数据集 |
| --- | --- | --- | --- |
| FT-Transformer | **0.85%** | **0.73%** | CA 2.4%、HE 1.8%、YA 1.2% |
| ResNet | 0.60% | 0.51% | CA 1.65% |
| NODE | 0.41% | 0.40% | YE 0.77% |

来源的解释是 FT-T/ResNet 从集成中受益更多，NODE 本身已近似 ensemble（[[ft-transformer]], p. 7）。换句话说，**集成收益与基模型的方差/多样性有关，不是所有模型都同等受益**；一个已经带集成结构的模型，再加 seed averaging 的边际收益更小。

另一个边界：表 8/9 中默认 XGBoost 的单模型与集成数值完全相同（±0.0）。若训练在固定种子下确定，5 个“不同种子”可能给出同一模型，集成收益为零（**本库假设**，需按 XGBoost 版本与线程配置验证）。

实践上：分类器集成平均**概率**（先 `expit`/`softmax`）而不是 logits（repo `bin/ensemble.py` L34–L65）；GBDT 本身是 boosting 集成，再做 seed averaging 的收益要看基模型方差。

## 正则化的三种机制

1. **罚**：叶子级 `l2_regularization`、`ccp_alpha`；
2. **删**：树深、叶子数、单调/交互约束；
3. **平均/随机化**：`subsample`、`max_features`、bootstrap、不同阈值。

学习率 $\nu$ 与轮数必须联合看：$F_m=F_{m-1}+\nu h_m$（L353–L363）。HGBT 的 `max_iter` 是上限，早停开启时不保证跑满。

## 常见失效模式

- 把 HGBT 的速度优势当成所有规模上都更合适；
- 忽略小样本分箱过粗；
- 把 RF 当作硬投票；
- 只用 `max_iter`/`n_estimators` 报预算，忽略早停；
- 把 `subsample` 当作 `learning_rate` 的替代；
- Stacking 使用非折外预测；
- 把「加了很多基模型」当成「第二层变强」，忽略同族模型的误差相关（TFM 例子只值 +5e-5）；
- 直接用概率而不先统一到 logit 尺度做线性第二层；
- 软投票平均未校准概率；
- 忽略类别基数、缺失值与版本约束；
- 用 MDI 解释高基数或 ID 特征，忘记其训练集统计与高基数偏好（L511–L526）。

## 待解问题

- 小样本 HGBT 与旧 GBDT 的分界；
- 学习率与轮数的联合预算；
- `interaction_cst` 与 `monotonic_cst` 的组合；
- Stacking 折外预测与最终基模型的分布差异；
- 宽 pair 池（数百个高度相关成员）下，逻辑回归权重集中在少数成员是必然还是可以改善（正则强度、分组 stack、非负约束）；
- AdaBoost 对标签噪声的敏感性；
- IsolationForest 作为 bagging 骨架 + 路径聚合的成本模型（[[sklearn-outlier-detection]]）。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-ensembles]]、[[sklearn-outlier-detection]]、[[ft-transformer]]、[[kaggle-tfms-llms-tabular]]
- 概念：[[decision-trees]]、[[cross-validation]]、[[regularization-and-sparsity]]、[[anomaly-detection]]、[[generative-classification]]、[[leaderboard-vs-cv]]
- 实践：[[ensembles-selection-and-tuning]]、[[hyperparameter-search-and-tuning]]
