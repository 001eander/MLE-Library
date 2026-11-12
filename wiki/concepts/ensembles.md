---
title: "集成方法（Ensembles）：平均降方差、boosting 降偏差、元学习器与随机化预算"
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
  - bias-variance
  - anomaly-detection
sources:
  - "[[sklearn-ensembles]]"
  - "[[sklearn-outlier-detection]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1 的 ensemble 模块（见 [[sklearn-ensembles]] 的版本判定）
---

# 集成方法

本页整理「把多个模型拼成一个」的**机制**：两类集成各自在降什么、为什么随机化能降方差、GBDT 为什么是函数空间的梯度下降、以及 stacking 为什么必须用折外预测。事实来自 [[sklearn-ensembles]]（官方用户指南 1.11 节，1.9.1），引用写作 `[[sklearn-ensembles]], L438`。

## 两条轴，不是七个模型

来源给了 9 组估计器（共 17 个类），但它们只沿两条轴变化：

| | **平均式（averaging）** | **顺序式（boosting）** |
| --- | --- | --- |
| 代表 | Bagging、RandomForest、ExtraTrees、Voting、Stacking | GBDT、HGBT、AdaBoost |
| 基学习器要求 | **强、方差大**（全展开的树） | **弱、偏差大**（浅树 / stump） |
| 主要降什么 | 方差（L438） | 偏差（L729 对 stacking 说「reduce their biases」；boosting 同理） |
| 训练关系 | 相互独立 → 可并行 | 串行，第 $m$ 轮依赖 $F_{m-1}$ |
| 过拟合从哪来 | 单模型太复杂 | 轮数太多 / 步长太大 |
| 关键旋钮 | `n_estimators`、`max_features` | `n_estimators`×`learning_rate`、树大小、早停 |
| 典型失败 | 基模型太弱 → 平均也没用 | 基模型太强 → 几轮就过拟合 |

来源把这条判据写在一句话里：「bagging 方法在强而复杂的模型上效果最好……而 boosting 方法通常最适合弱模型」（L574）。

**本库综合**：这条轴还解释了为什么 boosting **不能**用 `n_jobs` 并行（每轮依赖上一轮），而 RF 可以（L500）——「能不能并行」不是实现细节，是轴的直接后果。

## 平均式：方差 $1/M$ 的前提是「误差不相关」

若 $M$ 个模型的预测误差两两相关系数为 $\rho$、各自方差为 $\sigma^2$，平均后的方差是

$$\operatorname{Var}\!\left(\frac{1}{M}\sum_{m=1}^{M} f_m\right)=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2$$

**本库综合（标准结果，来源未给公式）**：这个式子说明两件事：

1. $M\to\infty$ 时方差**不会降到 0**，而是降到 $\rho\sigma^2$——所以「树越多越好」有天花板；L493 的「results will stop getting significantly better beyond a critical number of trees」正是这个现象的经验表述。
2. 降低 $\rho$ 比增加 $M$ 更有效。集成方法的所有随机化手段，本质都是**去相关**。

于是「随机化」有三个作用位置：

| 位置 | 手段 | 谁在用 |
| --- | --- | --- |
| 样本 | bootstrap / 无放回子采样 / Pasting | Bagging、RF、GBDT 的 `subsample` |
| 特征 | 每**节点**抽子集 vs 每**基模型**抽子集 | RF/ET 的 `max_features`（每节点）vs Bagging 的 `max_features`（每模型） |
| 阈值 | ET 直接随机抽阈值 | `ExtraTrees*`（L458） |

**易混淆点**：`max_features` 在 RF 与 Bagging 里同名不同义。RF 是「每次切分时考虑多少特征」（L436），Bagging 是「每个基模型用多少特征」（L583）。前者制造**树内**多样性，后者制造**树间**多样性。

## Boosting：函数空间的梯度下降

GBDT 的加法形式（官方 1.9.1 数学节，剪藏丢失，见 [[sklearn-ensembles]] 转换损失 A6）：

$$F_M(x)=\sum_{m=1}^{M}h_m(x),\qquad F_m(x)=F_{m-1}(x)+h_m(x)$$

第 $m$ 轮的树在**当前模型**上最小化损失：

$$h_m=\arg\min_{h}\sum_{i=1}^{n} l\!\left(y_i,\,F_{m-1}(x_i)+h(x_i)\right)$$

一阶 Taylor 展开后，$h_m$ 等价于拟合**负梯度** $-g_i=\partial l/\partial F$。所以 boosting 不是「拟合残差」，而是「拟合损失对预测的负梯度」；残差只是平方损失下的特例。

三个由此推出的结构事实：

1. **收缩**：$F_m=F_{m-1}+\nu h_m$（L358），$\nu$ 即 `learning_rate`（L361）。它把每棵树当作「一步」，$\nu$ 是步长。
2. **子采样**：每轮只用 `subsample` 比例的数据（无放回，L367），等价于在随机子集上估计梯度 → 降方差。
3. **树大小 = 交互阶数**：深度 $h$ 的树能表达 $h$ 阶交互（L337），`max_leaf_nodes=k` 则是 $k-1$ 阶（L341）。

HGBT 额外用了**二阶**信息（Hessian）：叶子值取

$$w_k=-\frac{\sum_i g_i}{\sum_i h_i+\lambda}$$

其中 $\lambda$ 即 `l2_regularization`（官方 A2 折叠块；剪藏只剩目标函数 L68–L70）。这就是 Newton boosting（L257），它省掉了线搜索。

**本库综合**：$\lambda$ 放在**分母**里，因此它惩罚的是「Hessian 小（平坦区域）的叶子」——与 Ridge 惩罚系数大小的几何不同（见 [[regularization-and-sparsity]] 第七个面）。这是「同一个 $ℓ_2$ 符号、不同作用对象」的又一例。

## 树为什么适合当基学习器

三个来源里明说的性质：

1. **高方差、低偏差**（全展开时，L438）→ 正好是平均式需要的。
2. **轴对齐 + 分段常数** → 对单调变换免疫，不需要缩放（[[decision-trees]]）。
3. **贪心 + 不稳定** → 数据小扰动就换一棵树，这正是「去相关」想要的（L438：「decoupled prediction errors」）。

代价：树**不能外推**（[[decision-trees]]），所以整个集成也不能；树集成的预测永远落在训练目标范围内。

## GBDT 的代价模型：精确切分 vs 直方图

| | `GradientBoosting*` | `HistGradientBoosting*` |
| --- | --- | --- |
| 单节点切分 | $O(n_{\text{features}}\,n\log n)$（每节点排序，L251） | $O(n_{\text{features}}\,n)$（直方图，L253） |
| 候选切分点 | 每个不同取值 | ≤ `max_bins`（默认 255） |
| 排序次数 | 每节点、每特征 | **全流程一次**（L255） |
| 缺失值 / 类别 / 约束 | 无 | 原生（L76、L140、L187、L217） |
| 小样本 | 精度可能更好（官方 `aside`，剪藏丢失） | 分箱可能过粗 |

**本库综合**：直方图不是「近似算法」，它把问题**换成了另一个离散化问题**——一旦分箱，切分点只能落在箱边界上。这就是「小样本时旧实现可能更准」的机制来源，也是 `max_bins` 同时是「正则化旋钮」与「精度上限」的原因（L64 说「Using less bins acts as a form of regularization」）。

## Stacking：为什么必须折外

若第二层直接用第一层在**训练集**上的预测，第一层会过度自信（尤其树/近邻类模型能完美拟合训练集），第二层学到的映射是错的。所以 `Stacking*` 内部用 `cross_val_predict` 生成折外预测（L769）。

后果清单：

- **训练成本 ≈ 单层 × (K+1)**：K 折折外预测 + 全量重拟合（L793「computationally expensive」）。
- **`transform` 的输出是「基模型预测矩阵」**，二分类 + `predict_proba` 时每模型只留一列（两列共线，L795–L796）。
- **`stack_method='auto'` 的优先级**：`predict_proba` → `decision_function` → `predict`（L771）。混用不同输出类型时，第二层看到的特征尺度可能完全不同。
- **多层 stacking** 只是把 `final_estimator` 再设成一个 `Stacking*`（L798–L823）。

**本库综合**：stacking 与 [[leaderboard-vs-cv]] 是同一个主题的两面——「用训练集内部的切分模拟未见数据」。区别在于 CV 用它**评估**，stacking 用它**构造特征**。这也解释了为什么 stacking 的收益高度依赖「折外预测是否真的近似测试分布」：如果训练集有时间序或分组结构，随机 K 折会让折外预测泄漏（该问题在 3.1 节，未入库）。

## AdaBoost：重加权版 boosting

机制（L835）：初始 $w_i=1/N$；每轮提高错分样本权重、降低正确分类样本权重；下一轮弱学习器被迫关注错例；最后加权多数（和）组合。分类用 SAMME、回归用 R2（L841–L842）。

**本库综合**：AdaBoost 可看作 GBDT 在**指数损失**下的特例（官方分类损失表明确说 `exponential` 与 AdaBoost 相同——该表在剪藏中丢失）。因此它继承了指数损失的性质：对**错标样本**极敏感（权重指数增长），而 log-loss 不会。这条是本库判断，来源正文未展开。

## 聚合函数：四种「怎么合」

| 聚合 | 定义 | 需要什么 | 何时失效 |
| --- | --- | --- | --- |
| 硬投票 | 多数类；平票按类标签升序 | `predict` | 基模型强弱悬殊时无法超过最好的那个 |
| 软投票 | 加权概率和的 argmax（L636–L638） | 每个基模型有 `predict_proba`（L659） | 概率未校准时「平均」没有意义 |
| 平均概率 | 各树 `predict_proba` 的算术平均（L442） | 树 | 概率仍是树间平均，不是校准概率 |
| 元学习器 | 第二层模型在折外预测上拟合（L769） | 内部 CV | 折外预测与测试分布不一致时 |

平均式的四个代表（Bagging / RF / Voting / Stacking）差别主要在**聚合函数**与**随机化位置**；Voting 与 Stacking 还允许**异质**基模型（L600、L729），而 Bagging/RF 的基模型通常同类。

## 特征重要性：训练期统计量，不是模型性质

- **MDI** = 特征贡献的样本比例 × 不纯度下降，跨树平均、归一化到和 1（L513–L520）。
- 两个缺陷（L518）：基于训练集统计（不保证留出集重要）；偏好**高基数**特征。
- 替代：`permutation_importance`（L408、L518）。

**本库综合**：这和 [[decision-trees]] 的「`predict_proba` 是叶内比例」同源——都是把**训练期统计量**当成模型性质。高基数偏好有一个可操作推论：任何 ID 类列（用户编号、交易号）放进特征都会在 MDI 里排前列，而对留出集无用。当 MDI 被用作**选列判据**（`SelectFromModel(RandomForestClassifier())`、`RFE`）时，这两个缺陷直接变成保留/删除的决定，且 1.13 节的默认阈值是「均值」——ID 类列只要高于均值就会被留下（[[feature-selection]]、[[sklearn-feature-selection]], L127–L149）。

## 与其他模型族的定位

| 族 | 参数化方式 | 容量旋钮 | 需要缩放 | 能否外推 | 缺失值 |
| --- | --- | --- | --- | --- | --- |
| 线性 / 惩罚（[[sklearn-linear-models]]） | 全局线性 + 惩罚 | $\alpha$、`C` | 是 | 能 | 需插补 |
| 判别分析（[[sklearn-lda-qda]]） | 类条件高斯 | `shrinkage`、`reg_param` | 否 | 有限 | 需插补 |
| 朴素贝叶斯（[[sklearn-naive-bayes]]） | 条件独立 + 分布族 | `alpha` | 否 | 能 | 需插补 |
| SVM（[[sklearn-svm]]） | 最大间隔 + 核 | $C$、$\gamma$、$\nu$ | 是（硬前提） | 有限 | 否 |
| 最近邻（[[nearest-neighbors]]） | 局部常数（模型=训练集） | $k$ / $r$ | 是（=模型定义） | 否 | 否 |
| 单棵决策树（[[decision-trees]]） | 轴对齐矩形 + 每盒常数 | 深度/叶子/`ccp_alpha` | 否 | 否 | 原生 |
| **树集成（本页）** | **多个树/模型的组合** | **`n_estimators` × `learning_rate`、`max_features`、树大小** | **否** | **否** | **HGBT 原生；RF 需插补** |

**本库综合**：树集成没有引入新的假设形式——它的假设形式继承自基学习器。它新增的是**「用计算换方差/偏差」的维度**。所以在选型表里它不该与「线性 vs 核 vs 近邻」并列，而应作为**这些族的放大器**：弱模型 → boosting，强模型 → bagging。

### 一个反例：`IsolationForest` 用的是 bagging 骨头，但聚合的不是预测

`IsolationForest` 的骨架与本页的 bagging 完全一致：`bootstrap` 可选、每棵树只看 `max_samples` 个点、结果跨树汇总。但三件事都不同（[[sklearn-outlier-detection]], L123–L133）：

| 维度 | 本页的 RF / Bagging | `IsolationForest` |
| --- | --- | --- |
| 基学习器 | `DecisionTree*`（用标签找最优切分） | `ExtraTreeRegressor`（**随机阈值**，不看标签） |
| 聚合 | 投票 / 平均概率 | **平均路径长度**（无投票） |
| 树的深度 | 长到纯或受 `max_depth` 限制 | **写死 $\lceil\log_2 n\rceil$** |
| 目标 | 降低预测方差 | 估计「隔离难度」 |

**本库综合**：所以它验证的是本页「集成 = 骨架 + 聚合函数」这条拆分的可用性——换聚合就换了任务（从分类/回归到异常打分），而骨架不变。这也解释了它为什么**完全尺度不变**（轴对齐切分内比大小）而 RF 不是（虽然 RF 也不需缩放，但那是另一个原因）。默认阈值同样是常数：`contamination='auto'` → `offset_=-0.5`，不是「自动 10%」（实测高斯数据上报 13.9–15.4%）。选型与坑见 [[outlier-detection-selection-and-tuning]]，机制见 [[anomaly-detection]]。

## 失效模式

| 失效 | 机制 | 证据 |
| --- | --- | --- |
| 树越多越好（错） | 方差有 $\rho\sigma^2$ 下限；测试误差平台化 | L493；图 6 读数 |
| 无收缩 + 子采样 | 降方差但不降步长，后期仍过拟合 | L369；图 2 读数（蓝线 ≈0.70） |
| 早停静默生效 | 默认 ≥10,000 样本开启，`n_iter_` < `max_iter` | L72；实测 1.6.1（100→76） |
| 小样本用 HGBT | 分箱使切分点过粗 | 官方 `aside`（剪藏丢失） |
| 类别基数超限 | 基数 > `max_bins` 报错；等于则通过 | L164 与源码口径不一致（分歧 1） |
| 未见类别被当缺失 | 预测期映射为 NaN 分支 | L166；实测 |
| 多类 GBRT 成本爆炸 | 每轮 `n_classes` 棵树 | 官方 A4（剪藏丢失） |
| 单调约束被过度期待 | 只约束「其余特征不变」的偏效应；多类不支持 | L198、L204 |
| 硬投票平票不可控 | 平票按类标签升序，重编码会改结果 | 官方 1.11.4.1（剪藏丢失）；实测 |
| 软投票无概率 | 基模型无 `predict_proba` | L659 |
| Stacking 不涨点 | 第二层只在折外预测上训练，信息少于第一层 | L779、L822（官方示例均 0.53） |
| MDI 误导 | 训练集统计 + 高基数偏好 | L518 |
| RF 概率被当校准概率 | 树间平均仍是模型输出 | L442（本库判断） |
| 模型体积失控 | 默认全展开 → $O(MN\log N)$ | L493、L496 |
| AdaBoost 怕错标 | 指数损失放大错例权重 | L835（本库判断） |

## 待解问题

1. 平均式的 $\rho$ 在真实数据上由什么决定？`max_features` 与 `subsample` 各自把 $\rho$ 降到多少，本节没有量化。
2. 「小样本用 `GradientBoosting*`」的样本量拐点，以及它与 `max_bins` 的交互。
3. $\nu\cdot M$ 是否存在可跨数据的等价预算。
4. HGBT 的 `max_features` 在正文只被描述为「缩短运行时间」（L376）；源码 docstring 说明是每节点抽特征，且受 `interaction_cst` 限制（外部核查）。它与交互约束叠加后的实际多样性收益未知。
5. `interaction_cst` 与 `monotonic_cst` 同时开启的可行性，以及约束对早停验证损失的影响。
6. stacking 的折外预测与最终 `estimators_` 之间的分布差，是否足以解释「多层不涨点」。
7. 软投票/概率平均在未校准模型上是否仍然合理（本节无校准论述）。
8. MDI 的高基数偏好在 GBDT（熵口径，L408）与 RF 上是否同强度。

## 关联页面

- 来源：[[sklearn-ensembles]]（含转换损失、默认值与图表读数）
- 实践：[[ensembles-selection-and-tuning]]（选型顺序与调参次序）
- 上游：[[decision-trees]]（基学习器）、[[sklearn-decision-trees]]
- 正则化：[[regularization-and-sparsity]]（第七个面：收缩与随机化预算）
- 评估：[[leaderboard-vs-cv]]（CV 作为训练机制 vs 评估机制）、[[cross-validation]]（Stacking 为何必须用折外预测、`cross_val_predict` 为何不是分数）
- 下游：[[feature-selection]]（MDI 当选列判据时的偏差）、[[feature-selection-workflow]]（嵌入式选择的工程写法）
- 对照：[[nearest-neighbors]]、[[support-vector-machines]]（另两处「实现选择 ≠ 模型选择」）
- 另一类聚合：[[anomaly-detection]]（`IsolationForest`：bagging 骨架 + 路径长度聚合，任务是异常打分而非预测）、[[outlier-detection-selection-and-tuning]]
- 工具：[[scikit-learn]]
