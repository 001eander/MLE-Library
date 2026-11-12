---
title: "scikit-learn 用户指南 1.11「Ensembles」：GBDT、HGBT、随机森林、Bagging、Voting、Stacking、AdaBoost（stable 文档剪藏，未回填）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - ensemble
  - gradient-boosting
  - random-forest
  - bagging
  - stacking
  - adaboost
  - reference-doc
source_path: raw/articles/sklearn/1.11. Ensembles Gradient boosting, random forests, bagging, voting, stacking.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.11「Ensembles」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking（用户指南第 1.11 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页属项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；29 段代码块、7 处图片引用（6 张唯一图，末段重复 1 次）、1 张表（投票权重示例）、63 个指向 `generated/` API 页的链接 |
| 原文 | <https://scikit-learn.org/stable/modules/ensemble.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/1.11. Ensembles Gradient boosting, random forests, bagging, voting, stacking.md`（898 行、76,566 字节、约 8,133 词；**末行无换行**） |
| SHA-256 | `3859a2d451f4f4edc88bdb3dcf8eb22a603e86e3f2331671596cbd55838cf11d` |
| 完整性 | **8 个内容折叠块 + 1 个 `aside` 主题框 + 1 个整节（1.11.4.1）丢失**，散文合计约 1,750 词 ≈ 本节的 **21%**；19 条参考文献只留下 1 条；另有 16 处引用标记与页内锚点文字被吞。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.1**（173 条句子探针中 138 条可逐字比对，1.9.1 全部命中；1.6/1.7/1.8 分别为 132/134/137。判据见「版本判定」） |
| 插图 | 6 张，已入库 `raw/assets/sklearn-ensembles/`（全部目视核查，读数见「图表核查」） |
| 代码 | 29 段：HGBT 分类（L40–L49）、缺失值两段（L81–L89、L95–L102）、样本权重（L118–L131）、`categorical_features` 三种写法（L147、L153、L159）、单调约束（L190–L193）、交互约束树形图（L219–L224）、GBC 分类（L270–L280）、GBR 回归（L286–L299）、warm_start（L314–L332）、特征重要性（L397–L405）、RF 拟合（L423–L427）、DT/RF/ET 交叉验证对照（L461–L486）、RF warm_start（L548–L560、L566–L567）、Bagging+KNN（L586–L589）、Voting 硬投票（L607–L631）、软投票（L662–L673）、GridSearchCV（L680–L692）、VotingRegressor（L704–L718）、Stacking 三段（L736–L741、L746–L754、L760–L766、L776–L790）、多层 Stacking（L802–L823）、AdaBoost（L849–L857） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-ensembles]], L24`。凡超出剪藏内容的部分（默认值、弃用、源码行为、丢失的折叠块）都显式标注「外部核查」，本机执行结果标注「实测（1.6.1）」。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `3859a2d4…`，末行无换行）。丢失内容在「转换损失」中逐块记录，**未写回原文件**；下文的行号全部指向原始剪藏。

## 概要

这一节覆盖 **9 组估计器（共 17 个类）**，是本库到目前最大的单篇来源，也是第一次出现「模型不是一种假设形式，而是一种组合策略」的类目。它把此前各节的模型族当作**基学习器**，回答一个新问题：*怎样把许多弱/不稳的模型拼成一个更强的模型*。对本库的意义有五层：

1. **第一次给出「集成」的两条正交轴**。平均式（bagging / random forest / extra-trees / voting / stacking）靠**降方差**，顺序式（boosting / AdaBoost）靠**降偏差**。同一节里两类并列，且来源自己写出了判据：「bagging 适合强而复杂的模型（全展开的树），boosting 适合弱模型（浅树）」（L574）。
2. **第一次把「计算预算」写成一等公民**。HGBT 的直方图分箱把单节点切分从 $O(n_{\text{features}}\,n\log n)$ 降到 $O(n_{\text{features}}\,n)$（L251、L253），并且 `max_bins` 直接把候选切分点数从 $n$ 压到 ≤255（L64）。这解释了为什么「树集成」在 2020 年代取代了 SVM / KNN 成为表格数据的默认族。
3. **第一次出现「三种实现算同一个算法、但代价相差数量级」的族**：`HistGradientBoosting*` / `GradientBoosting*` / 外部 LightGBM-XGBoost（L22–L24）。上一处类似情形（[[nearest-neighbors]] 的 KD/ball/brute）是「选错只慢不错」；这里**连支持的功能都不同**（缺失值、类别特征、部分损失函数）。
4. **第一次出现「正则化不靠惩罚系数，而靠随机化与步长」**：`learning_rate`（收缩）、`subsample`、`max_features`、`max_bins`、早停，以及 `l2_regularization`（作用在**叶子值**上，L66–L70）。已折入 [[regularization-and-sparsity]] 的第七个正则化面。
5. **第一次出现「元学习器」**：stacking 把基模型的**折外预测**当作新特征训练第二层（L769），因此它天然依赖交叉验证——这也是本库第一个把「CV 不只是评估手段，还是训练机制」写进 API 的地方。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 集成的定义；两个著名例子（**锚点文字丢失**）；平均式与 boosting 的举例（**丢失**） | L10–L14 |
| 1.11.1 Gradient-boosted trees | GBDT 的适用范围；`Hist…` vs `GradientBoosting` 的取舍（**整个 `aside` 框丢失**） | L16–L18 |
| 1.11.1.1 Histogram-Based Gradient Boosting | 0.21 引入；速度；缺失值；分箱（256 bins）；API 差异 | L20–L33 |
| 1.11.1.1.1 Usage | `max_iter` 取代 `n_estimators`；回归五损失、分类一损失；`max_bins`；`l2_regularization`；早停默认 ≥10,000 样本 | L35–L72 |
| 1.11.1.1.2 Missing values support | 训练时学 NaN 走左/右；预测沿用；**训练未见 NaN → 走样本多的子节点** | L74–L109 |
| 1.11.1.1.3 Sample weight support | 权重乘梯度与 Hessian；**分箱（分位数）不看权重** | L111–L136 |
| 1.11.1.1.4 Categorical Features Support | 原生类别切分优于 one-hot 与序数编码；四种开启方式；基数约束；未见类别→缺失 | L138–L170 |
| （折叠块，已丢）Split finding with categorical features | $2^{K-1}-1$ → $K-1$ 的排序等价切分（Fisher 1958） | 官方页；剪藏无标题残迹 |
| 1.11.1.1.5 Monotonic Constraints | 先验方向约束；两个定义；只边际约束；多类不支持；类别特征不支持 | L172–L213 |
| 1.11.1.1.6 Interaction constraints | `interaction_cst` 限制特征交互；未列出的特征自成一组 | L215–L234 |
| 1.11.1.1.7 Low-level parallelism | OpenMP 并行点清单 | L236–L247 |
| 1.11.1.1.8 Why it's faster | 排序 vs 直方图；只分箱一次；Newton boosting（**1.9 新增句**）；并行 | L249–L261 |
| 1.11.1.2 GradientBoostingClassifier/Regressor | 两个最重要参数；两段示例；warm_start；树大小 = 交互阶数 | L263–L343 |
| （折叠块×2，已丢）Classification / Regression 示例 | `n_estimators`/`max_depth`/`learning_rate` 的语义；`train_score_`、`staged_predict` 早停 | 官方页；剪藏只剩代码 |
| 1.11.1.2.3 Mathematical formulation | 只剩一句引导语（**回归 438 词 + 分类 151 词全部丢失**） | L345–L347 |
| 1.11.1.2.4 Loss Functions | 只剩一句引导语（**两张损失表 122 + 115 词丢失**） | L349–L351 |
| 1.11.1.2.5 Shrinkage via learning rate | $F_m=F_{m-1}+\nu h_m$；$\nu$ 与 `n_estimators` 强耦合；建议 ≤0.1 + 早停 | L353–L363 |
| 1.11.1.2.6 Subsampling | 无放回抽样 `subsample`（典型 0.5）；特征子采样；OOB 偏差估计偏悲观 | L365–L384 |
| 1.11.1.2.7 Interpretation with feature importance | 不纯度重要性 = 各树平均；与 `permutation_importance` 不同 | L386–L414 |
| 1.11.2 Random forests and other randomized ensembles | perturb-and-combine；多输出 | L416–L430 |
| 1.11.2.1 Random Forests | bootstrap + 特征子集 → 解耦误差 → 降方差；**平均概率而非投票**；与 HGBT 的四点对照 | L432–L454 |
| 1.11.2.2 Extremely Randomized Trees | 阈值也随机；DT 0.98 / RF 0.999 / ET >0.999 的 doctest | L456–L489 |
| 1.11.2.3 Parameters | `n_estimators`、`max_features`（分类 `sqrt`/回归 1.0）、全展开、bootstrap 默认差异、`oob_score`、模型体积 $O(MN\log N)$ | L491–L496 |
| 1.11.2.4 Parallelization | `n_jobs`；加速比不线性 | L498–L509 |
| 1.11.2.5 Feature importance evaluation | MDI 的定义与两个缺陷（训练集统计、偏好高基数）；改用 permutation importance | L511–L526 |
| 1.11.2.6 Totally Random Trees Embedding | 叶子索引的 one-of-K 稀疏编码；隐式密度估计 | L528–L541 |
| 1.11.2.7 Fitting additional trees | `warm_start` 与 `random_state` 的交互（一次拟合 = 多次增量） | L543–L570 |
| 1.11.3 Bagging meta-estimator | 四种抽样变体（Pasting/Bagging/Random Subspaces/Random Patches）；`max_samples`/`max_features`/`bootstrap*`；OOB | L572–L596 |
| 1.11.4 Voting Classifier | 硬投票与软投票；加权平均示例 | L598–L600 |
| （整节，已丢）1.11.4.1 Majority Class Labels | 多数票；平票按**升序类标签**取 | 官方页；剪藏直接从 1.11.4 跳到 1.11.4.2 |
| 1.11.4.2 Usage | logistic / RF / NB 与硬投票集成的 acc 读数（0.95/0.94/0.91/0.95） | L602–L632 |
| 1.11.4.3 Soft Voting | argmax 加权概率和；3 类权重表；决策域示意图 | L634–L655 |
| 1.11.4.4 Usage | 软投票要求 `predict_proba`；`weights=[2,5,1]`；与 `GridSearchCV` 组合 | L657–L693 |
| 1.11.5 Voting Regressor | 平均预测值；GBR/RF/Linear 三模型示例 | L695–L725 |
| 1.11.6 Stacked generalization | 两层结构；`estimators`/`final_estimator`/`stack_method`；内部 `cross_val_predict`；二分类概率列丢弃；多层 | L727–L829 |
| 1.11.7 AdaBoost | 1995 Freund & Schapire；逐轮改样本权重；SAMME / R2；默认 stump | L831–L866 |
| （尾部） | 空 `References` + 被复制成脚注的 7 段重复正文 | L868–L898 |

## 关键主张

### 1. 两条正交轴：平均 vs 顺序（来源论述 + 本库综合）

来源把两类方法放在同一节但从不并列比较，只有 L574 一句判据：「bagging 方法在**强而复杂**的模型上效果最好（例如完全展开的决策树），而 boosting 方法通常最适合**弱模型**（例如浅树）」。

**本库综合**：这句话给出了整节的选型主轴——

| | 平均式（bagging / RF / ET / voting / stacking） | 顺序式（GBDT / HGBT / AdaBoost） |
| --- | --- | --- |
| 基学习器 | 强、方差大、可独立并行 | 弱、偏差大、必须串行 |
| 主要收益 | 降**方差**（L438） | 降**偏差**（L729 对 stacking 说「reduce their biases」，boosting 同理） |
| 过拟合来源 | 单棵树太深（L493） | 轮数太多 / 学习率太大（L363） |
| 关键旋钮 | `n_estimators`、`max_features` | `n_estimators`×`learning_rate`、树深、早停 |
| 训练成本 | 随树数线性，可 `n_jobs` 并行 | 随轮数线性，**不可并行**（每轮依赖上一轮） |
| 缺失值 | 需插补（RF/ET 无原生支持） | HGBT 原生（L76–L105） |

注意最后一行不是「算法优劣」，是**实现差异**：RandomForest 走 `DecisionTree*` 的精确切分，没有 NaN 分支；HGBT 走自己的分箱生长器，才有 NaN 分支。选型时这一点常常比准确率更决定性。

### 2. 三个 GBDT 实现：先选实现，再调参（来源论述 + 外部核查）

来源给出**两套** scikit-learn 实现，并把取舍写在丢失的 `aside` 框里（官方 1.9.1 页；剪藏完全没有）：

> 「Scikit-learn provides two implementations… The former can be **orders of magnitude faster** than the latter when the number of samples is larger than tens of thousands of samples. Missing values and categorical data are natively supported by the Hist… version… `GradientBoostingClassifier` and `GradientBoostingRegressor` might be preferred for **small sample sizes** since binning may lead to split points that are too approximate in this setting.」

剪藏只保留了前三句（L22–L28），**丢掉了最后一句「小样本反而应该用旧的 `GradientBoosting*`」**。这是本页最重要的一条转换损失：只读剪藏会得到「HGBT 全面更快」的单边结论。

两者的差别不止速度（L28、L253）：

| | `HistGradientBoosting*` | `GradientBoosting*` |
| --- | --- | --- |
| 切分方式 | 直方图分箱，候选点 ≤ `max_bins`（默认 255） | 每节点精确排序 |
| 单节点代价 | $O(n_{\text{features}}\,n)$ | $O(n_{\text{features}}\,n\log n)$ |
| 缺失值 | 原生（L76） | 不支持 |
| 类别特征 | 原生（L140） | 不支持 |
| 损失函数 | 回归 5 种 + 分类 log_loss（L52–L60） | 回归 4 种 + 分类 `log_loss`/`exponential`（2 个取值；**表已丢**，外部核查） |
| 单调 / 交互约束 | 支持（L187、L217） | 不支持 |
| 二阶信息 | Newton boosting（L257） | 一阶梯度 + 线搜索 |
| 早停 | 默认开启（≥10,000 样本，L72） | 默认关闭（`n_iter_no_change=None`，外部核查） |

**本库综合**：这就是「同一数学对象、不同数值路径」的第三次出现（前两次是 KNN 的三种搜索结构、SVM 的 libsvm/liblinear）。但这次差别**进入了模型定义**（缺失值、类别、约束），所以不能像 KNN 那样「选错只慢不错」。

### 3. HGBT 的五个结构能力（来源论述）

1. **缺失值原生**（L76–L105）：训练时每个切分点同时评估「NaN 走左」与「NaN 走右」的增益；预测时按学到的方向走。若某特征训练时**没出现 NaN**，预测时的 NaN 被送到「样本更多的子节点」（L105）——这是一个与特征值无关的默认方向。
2. **类别特征原生**（L140–L166）：优于 one-hot（等价切分需要更多深度）与序数编码（类别是名义量）。开启方式四种：布尔掩码、索引列表、列名列表、`"from_dtype"`（L147–L162）。基数必须 ≤ `max_bins`（L164 写「less than」，实测是 `<=`，见「分歧」）；未见类别预测时按缺失处理（L166）。
3. **样本权重**（L113–L136）：权重乘梯度与 Hessian，因此 `weight=0` 的样本等价于删除（L118–L134）；但**分箱阶段的分位数计算不看权重**（L136）——所以加权会改变叶子值，不改变候选切分点。
4. **单调约束**（L174–L209）：`monotonic_cst=[1,-1,0]`；只约束「其余特征不变时」的偏效应（L198），不能表达交叉约束；多类分类不支持（L204）；类别特征不支持（L209）。
5. **交互约束**（L217–L228）：`interaction_cst=[{0},{1},{2}]` 禁止一切交互；未列出的特征自动自成一组，所以 `[{0}]` 等价于 `[{0},{1,2}]`（L228）。来源注明与 LightGBM 的重叠组逻辑一致（L226）。

**本库综合**：这五项里只有第 1、3 项是「数据接口」，2、4、5 是**把领域知识写进假设空间**的手段。它们与 [[regularization-and-sparsity]] 的惩罚项不同：不缩小系数，而是**删掉一部分候选函数**。

### 4. 早停是默认行为，不是一个选项（来源论述 + 外部核查）

L72 写明：「**early-stopping is enabled by default if the number of samples is larger than 10,000**」，由 `early_stopping`、`scoring`、`validation_fraction`、`n_iter_no_change`、`tol` 控制；默认用验证损失，用 callable scorer 会显著变慢。

实测（1.6.1）：10,050 样本、`max_iter=100` 时 `do_early_stopping_=True`、`n_iter_=76`——**你以为在跑 100 轮，实际只跑了 76 轮**，而且训练集内部被切出 10% 做验证（`validation_fraction=0.1`）。

**本库综合**：这让 `max_iter` 的含义变成「上限」而非「轮数」，与 `GradientBoosting*` 的 `n_estimators`（默认无早停，跑满）**语义不同**。跨实现比较「树的数量」前必须先确认早停状态，否则比较的是两个不同的预算。

### 5. 学习率 × 轮数是同一个预算的两个刻度（来源论述）

$F_m(x)=F_{m-1}(x)+\nu\,h_m(x)$（L358），$\nu$ 即 `learning_rate`（L361）。来源的三条经验：

- 更小的 $\nu$ 需要更多轮才能达到同样的训练误差（L363）；
- 经验上更小的 $\nu$ 带来更好的测试误差（L363）；
- 建议 $\nu\le0.1$ 且把 `n_estimators` 设到「早停会触发」的程度（L363）。

**本库综合**：这是一条**只能联合调**的超参对，孤立刻度没有意义。若把 `learning_rate` 减半而不动轮数，等价于把模型容量减半；若同时把轮数加倍，才是「同一个预算、更细的步长」。注意 L363 的两处引文（Friedman 2001、HTF）在剪藏中被吞，见「转换损失 C」。

### 6. 树大小 = 可表达的交互阶数（来源论述）

「a tree of depth `h` can capture interactions of order `h`」（L337）。两种控制方式：

- `max_depth=h` → 完全二叉树，(至多) `2**h` 个叶子、`2**h - 1` 个分裂节点（L339）；
- `max_leaf_nodes=k` → best-first 生长，`k-1` 个分裂节点，**交互阶数 ≤ k-1**（L341）。

来源还给了等价关系：`max_leaf_nodes=k` 与 `max_depth=k-1` 结果可比，但训练显著更快、训练误差略高（L343），并对应 R `gbm` 的 `interaction.depth + 1`。这句引用的 Friedman 2001 在剪藏中丢失。

**本库综合**：这条把「调树深」翻译成「决定模型能看几阶交互」，比「防止过拟合」更可操作——如果业务上已知目标只依赖单特征与二阶交叉，把 `max_leaf_nodes` 压到 3 是有信息量的约束，而非盲目正则。

### 7. 收缩与子采样不是一回事（来源论述 + 图 2 读数）

L369 的图（`plot_gradient_boosting_regularization_001`）把五条测试偏差曲线画在一起。本库目视读数（640×480）：

| 曲线 | 400 轮处测试偏差 | 观察 |
| --- | --- | --- |
| No shrinkage（橙） | ≈0.78，且约 80 轮后**回升** | 无收缩先过拟合，后期明显变差 |
| `learning_rate=0.2`（青） | ≈0.50，单调缓降 | 收缩有效 |
| `subsample=0.5`（蓝） | ≈0.70，噪声大、上下抖动 | **无收缩 + 子采样表现最差之一** |
| `learning_rate=0.2, subsample=0.5`（灰） | ≈**0.40**，最低 | 两者叠加最好 |
| `learning_rate=0.2, max_features=2`（品红） | ≈0.46 | 特征子采样也有效，但不如样本子采样 |

**本库综合**：来源原文说「Subsampling without shrinkage, on the other hand, does poorly」（L369），图证实了这一点。所以 `subsample` 不能替代 `learning_rate`——它降方差，不降步长。

### 8. 随机森林降方差，但 scikit-learn 是「平均概率」不是「投票」（来源论述 + 实测）

RF 的两处随机性——bootstrap 抽样（L434）与每节点特征子集（L436）——目的是**让各树的预测误差解耦**，从而在平均时相互抵消（L438）。代价是偏差略升。

关键实现细节：**scikit-learn 平均的是 `predict_proba`，不是硬投票**（L442）。实测（1.6.1）：7 棵树的 `predict_proba` 与「各树 `predict_proba` 的算术平均」最大绝对差为 `0.0`。

**本库综合**：这条常被忽略，但它决定了三件事：(a) RF 的概率是「树间平均」，比单树平滑，但仍不是校准概率；(b) `predict` = `predict_proba` 的 argmax，因此**平票按概率和**而非票数；(c) 想复刻「硬投票 RF」必须自己遍历 `estimators_`。

RF 与 HGBT 的取舍，来源给了三条（L446–L448）：深树 vs 浅树、并行平均 vs 串行纠错、无分箱精确切分 vs 分箱。结论是「在具体数据上两个都试」（L450）。

### 9. Extra-Trees 把随机性推进到阈值（来源论述 + doctest 读数）

RF 仍选**最优阈值**，ET 则「为每个候选特征随机抽阈值，再取其中最好的」（L458）。后果：方差再降一点、偏差再升一点。

剪藏自带一个 doctest（L470–L486）：同一 `make_blobs(n_samples=10000, n_features=10, centers=100)`、5 折 CV——单树 **0.98**、RF(10 棵) **0.999**、ET(10 棵) **>0.999**。**本库判断**：这个数据有 100 个球形簇，属于「轴对齐树极擅长」的分布，因此三者的差距只能说明「集成优于单树」，**不能**用来排序 RF 与 ET。

### 10. 默认值本身就在做选型（来源论述 + 外部核查）

来源只给了少数默认值：`max_features` 的「经验好值」（分类 `"sqrt"`、回归 `1.0`，L493）、`max_bins=255`（L64）、RF `bootstrap=True` 而 ET `bootstrap=False`（L493）、`learning_rate` 建议 ≤0.1（L363）、`subsample` 典型 0.5（L367）。

其余默认值全部来自 1.9.1 签名（**外部核查**，见下表）。其中三条会沉默改变结果：

1. **`max_features` 在 RF 分类里默认 `"sqrt"`，在 RF 回归里默认 `1.0`**（外部核查）——同一份代码换任务就换了随机性强度。
2. **`GradientBoosting*.criterion` 在 1.9 已弃用（1.11 移除）**（外部核查：源码 `.. deprecated:: 1.9`），而**本节正文从头到尾没有出现 `criterion` 参数**（`grep` 仅命中英文单词 "criterion" 一次，L440）。
3. **HGBT 的 `categorical_features` 默认值是 `"from_dtype"`**（1.6 起；外部核查），也就是说**传入带 category dtype 的 DataFrame 时类别支持自动开启**；正文只说「可以传 `"from_dtype"`」（L162），没说这是默认。

### 11. Bagging 是「抽样策略」的元估计器（来源论述）

四种变体只差抽样方式（L578–L581）：样本无放回 = Pasting；样本有放回 = Bagging；特征子集 = Random Subspaces；两者都抽 = Random Patches。`BaggingClassifier/Regressor` 用一个 `estimator` 参数 + `max_samples`/`max_features`/`bootstrap`/`bootstrap_features` 覆盖全部四种（L583）。

来源强调它**不需要改基算法**，且「works best with strong and complex models」（L574）。示例用 `KNeighborsClassifier` 做基模型、`max_samples=0.5, max_features=0.5`（L588–L589）。

**本库综合**：bagging 与 RF 的区别只在「基模型是不是树」以及「特征子集是否在**每次切分**时重抽」——RF 的 `max_features` 是**每节点**抽，bagging 的是**每个基模型**抽一次。二者名字相同、语义不同，是本库最容易混淆的一对参数。

### 12. Voting：硬投票丢信息，软投票要概率（来源论述 + 丢失的平票规则）

- 硬投票 = 多数类（官方 1.11.4.1，**剪藏整节丢失**）；平票时「按类标签**升序**取」（官方页原文，实测 1.6.1：4 个分类器投 2,2,1,1 → 返回 `1`）。
- 软投票 = 加权概率和的 argmax（L636–L638）；3 类示例的加权平均为 `0.37/0.40/0.23`，取类 2（L644–L651）。
- 软投票要求每个基模型都有 `predict_proba`（L659）。

剪藏里的硬投票 doctest（L628–L631）读数：logistic 0.95、RF 0.94、NB 0.91、集成 0.95——**集成只追平了最好的单模型**。

**本库综合**：这个读数不是反例，而是 voting 的正常形态：当基模型强弱分明时，投票/平均无法超过最好的那个；它的价值在「几个模型差不多好但错误不相关」时（L600 的原话）。想让集成稳定超过单模型，要么用 stacking（学权重），要么用 boosting（纠错）。

### 13. Stacking 的「训练机制」是交叉验证（来源论述）

结构：第一层 `estimators` 并行拟合；第二层 `final_estimator` 用第一层的**折外预测**训练（L769，内部调用 `cross_val_predict`）。因此：

- 第一层在 `predict` 时用**全量训练数据**拟合的模型（`estimators_`），而第二层见过的输入是折外预测——两者分布略有差异，这是 stacking 的固有偏差；
- `stack_method='auto'` 按 `predict_proba` → `decision_function` → `predict` 的顺序选（L771）；
- 二分类 + `predict_proba` 时**第一列被丢弃**（两列完全共线，L795–L796）；实测 `transform` 形状为 `(n_samples, n_estimators)`；
- 多层 stacking 就是把 `final_estimator` 再设成一个 `Stacking*`（L798–L823）。

来源对收益的表述很克制：「as good as the best predictor of the base layer and even sometimes outperforms it」，但训练**计算代价高**（L793）。剪藏自带示例的读数值得记住：单层 R2 = **0.53**（L779），多层 R2 = **0.53**（L822）——**官方示例自己没跑出提升**。

### 14. AdaBoost 是「改样本权重」的 boosting（来源论述）

机制（L835）：初始 $w_i=1/N$；每轮提高被错分样本的权重、降低被正确分类样本的权重；下一轮弱学习器被迫关注上一轮错例；最后按加权多数（和）组合。分类用 SAMME、回归用 R2（L841–L842）。

默认弱学习器是**决策 stump**（L860），实测 1.6.1：`AdaBoostClassifier` 的 `estimators_[0]` 是 `max_depth=1` 的 `DecisionTreeClassifier`；`AdaBoostRegressor` 则是 `max_depth=3` 的回归树，`loss='linear'`（外部核查 + 实测）——所以「默认 stump」这句只对分类器成立。

剪藏示例读数：iris 5 折 CV **0.95**（L857）。图 6（`plot_adaboost_multiclass_001`）在 300 棵树上测试误差从 0.62 降到 ≈0.30 后**平台化**，说明「树更多不再更好」在 AdaBoost 上同样成立。

### 15. 特征重要性：MDI 有两个已知缺陷（来源论述）

MDI（mean decrease in impurity）= 每个特征贡献的样本比例 × 不纯度下降，跨树平均后归一化到和为 1（L513–L520）。来源明确列两个缺陷（L518）：

1. 基于**训练集**统计，不保证对留出集重要；
2. **偏好高基数特征**（unique 值多的特征）。

替代品是 `sklearn.inspection.permutation_importance`（L408、L518）。GBDT 的 `feature_importances_` 在来源里被描述为**基于 entropy**（L408，分类示例）；回归下对应的是方差下降，本节没有单独说明。

**本库综合**：这与 [[decision-trees]] 页里「`predict_proba` 是叶内比例、不是校准概率」是同一类问题——**训练期统计量被当成模型性质**。高基数偏好有一个可检验的推论：把 ID 类列（如用户编号）放进特征，MDI 会给它很高的分数，而它对留出集毫无用处。

## 版本判定

方法：把剪藏的散文切成 173 条句子探针（≥14 词、取前 12 词），分别在各版本官方页的正文中查找；未命中的探针逐条人工检查（均为 Markdown 链接/公式渲染差异，非文本差异）。

| 版本 | 命中的探针数 | 结论 |
| --- | --- | --- |
| 1.6 | 132 / 173 | 缺 1.7+ 新增内容 |
| 1.7 | 134 / 173 | 缺 1.8+ 新增内容 |
| 1.8 | 137 / 173 | 缺 1.9 新增内容 |
| **1.9** | **138 / 173** | 与剪藏一致 |
| `/stable/`（2026-11-11） | 138 / 173 | 页标题即 **1.9.1**，与 1.9 文本相同 |

关键判据：**1.9 新增的「Newton boosting」句在剪藏中**（L257：「A second major reason for being faster is the fact that … use second order information about the loss, i.e. the Hessian. This is called Newton boosting…」），1.8 页没有这句。因此版本 ≥ 1.9；`/1.9/` 与 `/stable/` 均为 1.9.1，故判为 **1.9.1**。

旁证：1.9 弃用 `GradientBoosting*.criterion`（外部核查），而剪藏正文根本不提这个参数（见「关键主张 10」），因此这条判据与正文不冲突、也无法进一步细分 1.9.0/1.9.1。**本页所有功能性结论都不依赖小版本号**。

## 转换损失

剪藏的结构性损失分四类。A 类最严重：8 个 `<details>` 折叠块 + 1 个 `<aside class="topic">` 主题框 + 1 个整节全部没有内容。

### A. 丢失的折叠块 / 主题框 / 整节

词数为**散文词数**（剔除代码块与表格后统计，官方 1.9.1 页）。

| # | 官方位置 | 内容 | 词数 | 剪藏现状 |
| --- | --- | --- | --- | --- |
| A1 | 1.11.1 引言 | `GradientBoostingClassifier` vs `HistGradientBoostingClassifier` 的 `aside` 主题框：两套实现、速度差、原生缺失值/类别、**小样本应优先用旧实现** | ≈90 | **整块丢失**；只剩 L24 的速度句与 L26 的缺失值句（重复自正文） |
| A2 | 1.11.1.1 Usage | *Details on l2 regularization*：损失项只是一半；$k$ 是第几棵树（回归/二分类 1 棵/轮，多类 `n_classes` 棵/轮）；叶子值 $w_k=-\text{sum\_gradient}/(\text{sum\_hessian}+\lambda)$；$\lambda$ 惩罚 Hessian 小的叶子；XGBoost 的 $\gamma\sum_k T_k$ 未实现 | 288 | 只剩 L66–L70 的目标函数与两条脚注 |
| A3 | 1.11.1.1.4 | *Split finding with categorical features*：$2^{K-1}-1$ 种划分 → 按目标方差排序类别 → 只需 $K-1$ 个切分；复杂度 $O(K\log K+K)$ vs $O(2^K)$（Fisher 1958） | 159 | **整块丢失**，标题也不在剪藏里 |
| A4 | 1.11.1.2 | *Classification* 示例块：`n_estimators` 控制弱学习器数；`max_depth`/`max_leaf_nodes` 控制树大小；`learning_rate∈(0,1]`；**多类需 `n_classes×n_estimators` 棵树，类多时强烈建议改用 HGBT** | 134 | 只剩代码与 0.913 |
| A5 | 1.11.1.2 | *Regression* 示例块：糖尿病数据 500 棵树的训练/测试误差图；`train_score_` 存每轮训练误差；`staged_predict` 取每轮测试误差 → 用早停定 `n_estimators` | 128 | 只剩代码与 5.00 |
| A6 | 1.11.1.2.3 | *Mathematical formulation — Regression*：加法模型、贪心拟合 $h_m=\arg\min_h\sum_i l(y_i,F_{m-1}+h)$、初始值 $F_0$、一阶 Taylor 展开、负梯度、函数空间梯度下降、`absolute_error` 的叶子值改写为中位数 | 438 | 只剩 L347 一句「We first present GBRT for regression…」 |
| A7 | 1.11.1.2.3 | *Mathematical formulation — Classification*：$F_M$ 不是预测；log-loss 下 $p=\sigma(F_M)$；多类 softmax + 每轮 K 棵树；**子估计器永远是回归树** | 151 | **整块丢失** |
| A8 | 1.11.1.2.4 | *Loss Functions — Regression*：`squared_error`（初始值=均值）、`absolute_error`（中位数）、`huber`（`alpha` 控制离群敏感度）、`quantile`（`0<alpha<1`，可做预测区间） | 122 | 只剩 L351 一句引导语 |
| A9 | 1.11.1.2.4 | *Loss Functions — Classification*：二分类 log-loss（初始值=对数几率）、多类 log-loss（初始值=类先验）、`exponential`（同 AdaBoost，仅二分类，对错标更敏感） | 115 | **整块丢失** |
| A10 | 1.11.4.1 | *Majority Class Labels (Majority/Hard Voting)*：多数票定义；例子；**平票按类标签升序取** | ≈130 | **整节丢失**，剪藏直接从 1.11.4 跳到 1.11.4.2 |
| **合计** | | | **≈1,755** | 以官方本节散文 8,238 词计，**≈21%** |

**结构性后果（本库判断）**：

- 丢了 A1，剪藏会让人以为 HGBT **无条件**优于 `GradientBoosting*`；实际官方给的是「大样本快、小样本精度可能差」。
- 丢了 A6–A9，剪藏**完全没有 GBDT 的数学**：只剩 `max_iter`、损失名、叶子值公式的残片。因此「GBDT 在拟合负梯度」「分类时子估计器仍是回归树」这两条只能从外部核查取得。
- 丢了 A10，剪藏里 `VotingClassifier` **没有平票规则**——而这是唯一会改变 `predict` 结果的边界条件。
- 丢了 A4 的多类警告，读者不会知道 `GradientBoostingClassifier` 在多类问题上会乘上 `n_classes` 倍的树。

### B. 丢失的参考文献（19 / 20 条）

| 官方 References 位置 | 丢失条目 | 剪藏现状 |
| --- | --- | --- |
| 1.11.1.1 | `[Mayer2022]` Mayer, Bourassa, Hoesli & Scognamiglio (2022), *Machine Learning Applications to Land and Structure Valuation*, JRFM 15(5):193 | L234 只有空标题 |
| 1.11.1.1.8 | `[XGBoost]` Chen & Guestrin；`[LightGBM]` Ke et al.；`[Fisher1958]` Fisher, *On Grouping for Maximum Homogeneity* | L261 只有空标题 |
| 1.11.1.2 | `[Friedman2001]` *Greedy function approximation*；`[Friedman2002]` *Stochastic gradient boosting*；`[R2007]` Ridgeway, *Generalized Boosted Models* | L414 只有空标题 |
| 1.11.2 | `[B2001]` Breiman, *Random Forests*；`[B1998]` Breiman, *Arcing Classifiers* | L507–L509 只剩 Geurts et al. 2006 一条 |
| 1.11.2.5 | `[L2014]` Louppe, *Understanding Random Forests*（PhD thesis） | L526 只有空标题 |
| 1.11.3 | `[B1999]` Pasting；`[B1996]` Bagging predictors；`[H1998]` Random subspace；`[LG2012]` Ensembles on Random Patches | L596 只有空标题 |
| 1.11.6 | `[W1992]` Wolpert, *Stacked generalization* | L829 只有空标题 |
| 1.11.7 | `[FS1995]` Freund & Schapire；`[ZZRH2009]` Multi-class AdaBoost；`[D1997]` Drucker；`[HTF]` ESL | L868 之后只有重复正文脚注 |

### C. 被吞的引用标记与页内锚点文字（16 处）

| 行 | 现象 | 官方原文 |
| --- | --- | --- |
| L12 | 「Two very famous examples … are and.」 | gradient-boosted trees / random forests |
| L14 | 「averaging methods such as,, or, or in boosting, as.」 | Bagging methods / model stacking / Voting / AdaBoost |
| L18 | 「see the seminal work of.」 | `[Friedman2001]` |
| L22 | 「inspired by LightGBM (See ).」 | `[LightGBM]` |
| L66 | 「(see equation (2) in )」 | `[XGBoost]` |
| L343 | 「in and is related to」 | `[Friedman2001]` |
| L355 | 「proposed a simple regularization strategy」 | `[Friedman2001]` |
| L363 | 「recommend to set the learning rate」 | `[HTF]` |
| L367 | 「proposed stochastic gradient boosting」 | `[Friedman2002]` |
| L392 | 「(see for more details)」 | `[L2014]` |
| L442 | 「In contrast to the original publication」 | `[B2001]` |
| L448 | 「(see )」 | 页内链接 *Why it's faster* |
| L515 | 「Refer to for more information on MDI」 | `[L2014]` |

另：L418 的 `perturb-and-combine` 后丢 `[B1998]`；L436 的 `(see the for more details)` 丢页内链接；L833 的 `Freund and Schapire.` 丢 `[FS1995]`。

### D. 被复制成脚注的重复正文（转换器痕迹）

L870–L892 的 7 条「脚注」不是参考文献，而是正文段落的复制：`[^1]`=L18、`[^2]`=L66–L70、`[^4]`≈A2 的一句、`[^5]`=L343、`[^7]`=L355–L361、`[^8]`=L363、`[^9]`=L835。原始编号缺 `[^3]`、`[^6]`。**引用时应指向正文行号，不要引这些脚注**（它们本身也是残缺的，例如 `[^3]`/`[^6]` 已不存在）。

## 外部核查（不在剪藏内）

核查对象：`sklearn/ensemble/` @ tag `1.9.1`（`_hist_gradient_boosting/gradient_boosting.py`、`_gb.py`、`_forest.py`、`_bagging.py`、`_voting.py`、`_stacking.py`、`_weight_boosting.py`）。

### 默认值一览（1.9.1 签名）

| 估计器 | 默认值（关键项） | 文档是否提及默认值 |
| --- | --- | --- |
| `HistGradientBoostingRegressor` | `loss='squared_error'`、`quantile=None`、`learning_rate=0.1`、`max_iter=100`、`max_leaf_nodes=31`、`max_depth=None`、`min_samples_leaf=20`、`l2_regularization=0.0`、`max_features=1.0`、`max_bins=255`、`categorical_features='from_dtype'`、`monotonic_cst=None`、`interaction_cst=None`、`warm_start=False`、`early_stopping='auto'`、`scoring='loss'`、`validation_fraction=0.1`、`n_iter_no_change=10`、`tol=1e-7` | 仅 `max_bins=255`（L64） |
| `HistGradientBoostingClassifier` | `loss='log_loss'`，其余同上 + `class_weight=None` | 仅 `max_bins=255` |
| `GradientBoostingClassifier` | `loss='log_loss'`、`learning_rate=0.1`、`n_estimators=100`、`subsample=1.0`、`criterion='deprecated'`（1.9 弃用）、`min_samples_split=2`、`min_samples_leaf=1`、`max_depth=3`、`max_features=None`、`max_leaf_nodes=None`、`validation_fraction=0.1`、`n_iter_no_change=None`、`tol=1e-4`、`ccp_alpha=0.0` | 仅 `learning_rate` 的建议值（L363） |
| `GradientBoostingRegressor` | `loss='squared_error'`、`alpha=0.9`（quantile/huber），其余同上 | 否 |
| `RandomForestClassifier` | `n_estimators=100`、`criterion='gini'`、`max_depth=None`、`max_features='sqrt'`、`bootstrap=True`、`oob_score=False`、`class_weight=None`、`ccp_alpha=0.0`、`max_samples=None` | `max_features='sqrt'`（L493）、`bootstrap=True`（L493） |
| `RandomForestRegressor` | `criterion='squared_error'`、`max_features=1.0`，其余同上（无 `class_weight`） | `1.0`（L493） |
| `ExtraTreesClassifier` | 同 RF，但 `bootstrap=False` | `bootstrap=False`（L493） |
| `ExtraTreesRegressor` | 同 RF 回归，但 `bootstrap=False` | 同上 |
| `RandomTreesEmbedding` | `n_estimators=100`、`max_depth=5`、`sparse_output=True` | 否 |
| `BaggingClassifier/Regressor` | `estimator=None`（→ `DecisionTree*`）、`n_estimators=10`、`max_samples=None`（→1.0）、`max_features=1.0`、`bootstrap=True`、`bootstrap_features=False`、`oob_score=False` | 否 |
| `VotingClassifier` | `voting='hard'`、`weights=None`、`flatten_transform=True`、`n_jobs=None` | 否 |
| `VotingRegressor` | `weights=None`、`n_jobs=None` | 否 |
| `StackingClassifier` | `final_estimator=None`（→`LogisticRegression`）、`cv=None`（→5 折）、`stack_method='auto'`、`passthrough=False`、`n_jobs=None` | `stack_method='auto'` 的优先级（L771） |
| `StackingRegressor` | `final_estimator=None`（→`RidgeCV`）、`cv=None`、`passthrough=False` | 否 |
| `AdaBoostClassifier` | `estimator=None`（→`DecisionTreeClassifier(max_depth=1)`）、`n_estimators=50`、`learning_rate=1.0` | 「默认 stump」（L860，仅对分类器成立） |
| `AdaBoostRegressor` | `estimator=None`（→`DecisionTreeRegressor(max_depth=3)`）、`n_estimators=50`、`learning_rate=1.0`、`loss='linear'` | 否 |

### 弃用与版本变更（源码依据）

| 变更 | 版本 | 依据 |
| --- | --- | --- |
| `GradientBoosting*.criterion` 弃用，将于 **1.11 移除** | 1.9 | `_gb.py` docstring `.. deprecated:: 1.9`；`fit` 中若 `criterion != "deprecated"` 抛 `FutureWarning` |
| `HistGradientBoosting*.categorical_features` 默认值 `None` → `"from_dtype"` | 1.6 | docstring `versionchanged:: 1.6` |
| `categorical_features` 新增 `"from_dtype"` 选项 | 1.4 | docstring `versionchanged:: 1.4` |
| `HistGradientBoosting*.max_features` 存在，且是**每节点**抽特征 | 1.4 | docstring `versionadded:: 1.4`：「Proportion of randomly chosen features in each and every node split」 |
| `AdaBoostClassifier` 无 `algorithm` 参数（SAMME.R 已移除，仅 SAMME） | 1.6 起 | 签名只有 `estimator`/`n_estimators`/`learning_rate`/`random_state` |
| `max_bins` 合法区间 **[2, 255]** | 1.9.1 | `_parameter_constraints` `Interval(Integral, 2, 255)` |

### 源码层面的两个数字

- **`max_bins` 与「256 bins」的关系**：源码注释写 `n_bins = max_bins + 1  # + 1 for missing values`。所以 L28 的「typically 256 bins」与 L64 的「255（默认）」说的是同一件事的两半：255 个数值箱 + 1 个缺失箱。剪藏没有解释这层关系。
- **类别基数检查是 `>` 不是 `>=`**：源码 `if categories.size > self.max_bins` 抛错，错误信息为「cardinality <= max_bins」；而 L164 写「must be **less than** the `max_bins` parameter」。实测（1.6.1）3 个类别 + `max_bins=3` 可以通过，4 个类别才报错。见「分歧」第 1 条。

## 本机运行时复核（scikit-learn 1.6.1）

以下 10 条在 1.6.1 上执行（PyPI 当前最高可安装版本；1.9.1 未安装，机制层结论按源码判断可跨版本，数值结论不跨版本）。

| # | 检查 | 结果 |
| --- | --- | --- |
| 1 | 训练集无 NaN 时，预测 NaN 的走向 | `X=[[0],[1],[2]], y=[0,0,1]` → `predict([[nan]])=[0]`；左子节点 2 个样本，**走样本多的分支**，复现 L105 |
| 2 | 未见类别按缺失处理 | `categorical_features=[True]`，训练类别 {0,1}，`predict([[5]])=[0]`，复现 L166 |
| 3 | `max_bins=256` | `InvalidParameterError`：「must be an int in the range [2, 255]」 |
| 4 | 类别基数 = `max_bins` | **不报错**（3 类 + `max_bins=3`）；4 类才报 `ValueError`（「cardinality <= 3」）。与 L164 的「less than」不符 |
| 5 | `sample_weight=0` 等价删除 | 复现 L118–L134：`predict([[1,0]])=[1]`，`predict_proba=0.99902` |
| 6 | 早停默认开启的后果 | 10,050 样本、`max_iter=100` → `do_early_stopping_=True`、`n_iter_=76` |
| 7 | 硬投票平票 | 4 个分类器投 2,2,1,1 → `predict=1`（类标签升序最小者）；软投票同数据 `proba=[0,0.5,0.5]` → `predict=1` |
| 8 | Stacking 的二分类概率列 | `StackingClassifier`（2 个基模型）`transform` 形状 `(200, 2)` = 每模型 1 列（第一列已丢）；`stack_method_=['predict_proba','predict_proba']` |
| 9 | RF 是平均概率而非投票 | 7 棵树：`rf.predict_proba` 与「各树 `predict_proba` 平均」最大绝对差 **0.0** |
| 10 | RF `warm_start` 等价性 | `n_estimators=10` + `warm_start` 再 fit 到 20，与一次性 20 棵的 `predict_proba` **完全一致**（`allclose=True`），复现 L563 |

另有三条顺带确认：

- `AdaBoostClassifier(n_estimators=3)` 的基学习器是 `DecisionTreeClassifier(max_depth=1)`；`AdaBoostRegressor` 是 `DecisionTreeRegressor(max_depth=3)`、`loss='linear'`。
- `ExtraTreesClassifier(oob_score=True)` 直接抛 `ValueError: Out of bag estimation only available if bootstrap=True`（默认 `bootstrap=False`）。
- `BaggingClassifier()` 的 `estimators_[0]` 是 `DecisionTreeClassifier`，`max_samples` 属性为 `1.0`（参数默认 `None`），`bootstrap=True`、`max_features=1.0`。

**版本警告**：第 3、4 条依赖参数约束实现；第 1、2 条依赖 HGBT 的分箱生长器；第 9、10 条依赖 RF 的聚合实现。这三处 1.6.1 与 1.9.1 源码结构一致，但**未在 1.9.1 上复跑**。

## 图表核查

6 张图于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-ensembles/`，逐张目视核查。

| # | 文件 | 尺寸 / SHA-256 前 16 位 | 目视读数 |
| --- | --- | --- | --- |
| 1 | `sphx_glr_plot_gradient_boosting_regression_001.png` | 640×480 / `457d57539ada7ab9` | 标题 "Deviance"；训练偏差从 ≈5,900 单调降到 ≈950（500 轮），测试偏差约 100 轮降到 ≈3,050 后**基本走平并轻微回升**到 ≈3,000。图上没有早停标记——早停的效果要自己用 `staged_predict` 复现 |
| 2 | `sphx_glr_plot_gradient_boosting_regularization_001.png` | 640×480 / `690e62d6c938739f` | 五条曲线；橙（无收缩）≈80 轮后回升到 0.78；蓝（`subsample=0.5`、无收缩）噪声大且终值 ≈0.70；灰（收缩+子采样）最低 ≈0.40；品红（收缩+`max_features=2`）≈0.46；青（仅收缩）≈0.50。与 L369 的定性结论一致 |
| 3 | `sphx_glr_plot_forest_iris_001.png` | 600×600 / `41d3b7de9ea433c4` | 标题 "Classifiers on feature subsets of the Iris dataset"，3 行 × 4 列（DecisionTree / RandomForest / ExtraTrees / AdaBoost），每行用两个特征。所有面板都是**轴对齐矩形**；单树面板边界最碎、最不规则，RF/ET 的边界相对规整；AdaBoost（stump）在第一行主要表现为**单方向条带**，说明其决策面是许多轴对齐阶梯的叠加 |
| 4 | `sphx_glr_plot_voting_decision_regions_002.png` | 640×480 / `bd6f90a5945dc6fa` | 4 个面板：Splines（constant extrapolation）、Splines（periodic extrapolation）、RBF Nystroem、**Soft Voting**。色标 "Probability estimate" 0.2–0.8。Soft Voting 面板是**块状拼接**的概率面，训练区外（四角）趋向中性 ≈0.5；periodic 面板在数据范围外**重复出现红蓝斑块**，是「样条外推」的反例。该图无坐标刻度标签以外的数值 |
| 5 | `sphx_glr_plot_voting_regressor_001.png` | 640×480 / `32c7cf9b74cd0a73` | 标题 "Regressor predictions and their average"；红五星（VotingRegressor）始终落在三个基模型的预测**之间**（例如某样本 Linear≈210、GBR≈160，红星≈200），符合「平均预测值」（L697）。图的 x 轴无刻度标签 |
| 6 | `sphx_glr_plot_adaboost_multiclass_001.png` | 640×480 / `f0b8acc48b6e4d8c` | 标题 "Convergence of AdaBoost algorithm"；蓝色 AdaBoost 测试误差从 ≈0.62 降到 ≈0.30（约 70 棵）后**平台化并轻微波动**到 300 棵；橙色虚线 `DecisionTreeClassifier` ≈0.475；点线 `DummyClassifier` ≈0.70。与「树数超过临界点后收益消失」一致（对照 L493） |

第 7 处图片引用在 L894，是 L837 的**重复**（转换器把脚注区的图重复输出一次）。因此唯一图 6 张。

## 分歧

1. **类别基数：文档说「less than」，实现允许等于**。L164：「The cardinality of each categorical feature must be less than the `max_bins` parameter」；1.9.1 源码检查 `categories.size > self.max_bins`，错误信息为「cardinality <= max_bins」；实测 1.6.1：3 类 + `max_bins=3` 通过、4 类才报错。按文档只能取 ≤254，按实现可取 ≤255。**保留来源原话 + 记录实现口径**，不修改剪藏。
2. **「256 bins」与「255（默认）」的措辞冲突**。L28 说「typically 256 bins」，L64 说「recommended to use as many bins as possible (255), which is the default」。源码注释 `n_bins = max_bins + 1  # + 1 for missing values` 说明两者是「数值箱 255 + 缺失箱 1」。剪藏未解释，容易读成矛盾。
3. **HGBT 的适用边界被剪掉一半**。L24 只留「orders of magnitude faster … larger than tens of thousands」，官方的同一主题框还写「小样本时 `GradientBoosting*` 可能更好，因为分箱会让切分点过于粗糙」。这不是矛盾，而是**剪藏侧的单边结论**；本页按官方补齐。
4. **`max_features` 的「default」用词**。L493 先给「好默认值」：回归 `1.0`、分类 `"sqrt"`；紧接着说「The default value of `max_features=1.0` is equivalent to bagged trees」。字面上「默认值 1.0」与分类的 `"sqrt"` 冲突；按 API，RF 分类默认 `"sqrt"`、回归默认 `1.0`，`1.0` 只是「等价于 bagged trees」的那个**取值**。本库按 API 口径理解，保留来源原句。
5. **「默认弱学习器是 stump」只对分类器成立**。L860 写在 AdaBoost 一节末尾，而该节同时覆盖分类与回归；实测 `AdaBoostRegressor` 的基学习器是 `max_depth=3` 的回归树（源码同）。文档没有区分。
6. **Stacking 示例的 R2 没有提升**。单层 0.53（L779）、多层 0.53（L822）。来源正文说 stacking「有时能超过最好的基模型」（L793），但**自己的示例没有超过**。两者不矛盾（示例未给单模型基线），但引用「stacking 更好」时不能引这个示例。
7. **`GradientBoosting*` 的 `criterion` 参数在 1.9 已弃用，而正文完全不提**。这不是文档内部矛盾，而是**文档与 API 的时间错位**：正文的「树大小」只讲 `max_depth`/`max_leaf_nodes`，而 API 里 `criterion` 正在被移除（1.11）。引用本节时不应推断「`criterion` 仍可用」。
8. **尾部脚注是转换器痕迹，不是文献**。7 条 `[^n]` 复制了正文段落，且缺 `[^3]`/`[^6]`。任何「按脚注引用」的尝试都会指回正文本身。

## 局限

- **数学部分整体缺失**（A6–A9，共 826 词）。本页对 GBDT 目标函数、负梯度、叶子值更新的描述只能引官方页，**不能引剪藏**。
- **19 条参考文献丢失**，本页无法核对任何原始论文（Friedman 2001/2002、Breiman 1996/1998/1999/2001、Fisher 1958、Wolpert 1992、Freund & Schapire、Louppe 2014、XGBoost、LightGBM 等）。
- **剪藏不含任何分类/回归 GBDT 的损失表**，因此本页不能给出「哪个损失对应哪个初始值」的完整对照（只有 `feature_importances_` 与 HGBT 的五种回归损失名）。
- **未读未入库章节**：1.13 Feature selection、5.2 Permutation importance、3.1/3.2 CV 与调参、8.4 缺失值插补、`sklearn.calibration`。因此 stacking 的 `cv` 语义、OOB 与 CV 的取舍、概率校准只能按本节字面记录。
- **运行复核只做到 1.6.1**。1.9 新增/变更的行为（`criterion` 弃用、`absolute_error` 的 NaN 支持、`categorical_features` 默认值）未实测。
- **未验证**：HGBT 与 RF 在同一数据上的实际速度比与精度比；分箱对精度的影响随样本量的拐点；`interaction_cst` 与 `monotonic_cst` 的组合限制；`max_features` 对 HGBT 的实际作用（正文只说「减小可显著缩短运行时间」，L376）。
- 本节**没有任何关于特征缩放/编码/管道**的说明（属 8.3）；树集成通常不需要缩放，但**本条不是来源论述**，是外部常识，本页不据此下结论。
- 图 1–6 的读数是定性/近似读数（无坐标表格），不要当作精确数值引用。

## 待解问题

1. 「小样本用 `GradientBoosting*`」的拐点在哪里？官方只说「small sample sizes」，与 `max_bins` 的交互（分箱粗糙度 vs 样本量）没有量化。
2. `learning_rate` 与 `n_estimators` 的等价预算：是否存在与数据无关的换算（如 $\nu\cdot M$ 常数）？来源只给「强交互」与一条经验（L363）。
3. `subsample` 无放回（L367）与 RF 的有放回 bootstrap 在方差上的差别有多大？两者都叫「子采样」，语义不同。
4. `max_features` 在 HGBT 里的作用层级：正文只说能缩短运行时间（L376），源码 docstring 明确是「each and every node split」（外部核查），但它与 `interaction_cst` 的交互（只从允许的特征里抽）没有在正文说明。
5. `interaction_cst` 与 `monotonic_cst` 同时开启时的可行性，以及约束会不会让早停的验证损失失去单调性。
6. `StackingClassifier` 的折外预测与最终 `estimators_` 的分布差异，是否足以解释「多层不涨点」（官方示例两次 0.53）。
7. `VotingClassifier(voting='soft')` 在基模型未校准时是否仍然合理——本节没有校准论述，而概率平均的前提是概率可比。
8. `RandomTreesEmbedding` 的编码维度上界 `n_estimators * 2**max_depth`（L530）在默认参数（100×5）下是 3,200 维，实际稀疏度取决于叶子占用；本节没有给稀疏度经验值。
9. AdaBoost 对标签噪声的敏感度（官方在分类损失表里说 `exponential` 不如 `log-loss` 稳健，**该表已丢**），与 `learning_rate` 的关系未量化。
10. MDI 的「偏好高基数特征」在 GBDT 与 RF 上是否同强度——本节把两者写在同一段（L518），但 GBDT 的 `feature_importances_` 口径是熵（L408），未说明是否同样受影响。

## 关联页面

- 概念：[[ensembles]]（本页的机制整理：两条轴、GBDT 的函数空间梯度、Stacking 的折外训练）
- 实践：[[ensembles-selection-and-tuning]]（选型顺序、调参次序、18 条坑与可自动化程度）
- 上游：[[decision-trees]]（基学习器的机制）、[[sklearn-decision-trees]]（树的结构旋钮与缺失值规则）
- 正则化：[[regularization-and-sparsity]]（第七个面：收缩、随机化预算与叶子级 $ℓ_2$）
- 工具：[[scikit-learn]]（`ensemble` 模块矩阵与默认值第 56–76 条）
- 对照：[[nearest-neighbors]]（另一种「实现选择 ≠ 模型选择」）、[[support-vector-machines]]（另一种「同一族、两种实现」）
- 未入库：1.13 Feature selection、5.2 Permutation importance、`sklearn.calibration`
