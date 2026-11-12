---
title: "集成模型的选型与调参：HGBT / 随机森林 / Extra-Trees / Bagging / Voting / Stacking / AdaBoost"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - ensemble
  - gradient-boosting
  - random-forest
  - bagging
  - stacking
  - hyperparameter-tuning
  - scikit-learn
sources:
  - "[[sklearn-ensembles]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1（文档与签名；运行复核在 1.6.1 上完成，见 [[sklearn-ensembles]]）
---

# 集成模型的选型与调参

本页是工程指导：**先选实现，再调预算，最后才调结构**。事实来自 [[sklearn-ensembles]]（1.9.1），默认值来自 1.9.1 签名（外部核查），行为验证在 1.6.1 上执行。引用写作 `[[sklearn-ensembles]], L363`。

## 一、先判「该不该用集成」

**适合**（四条）：

1. **表格数据、混合量纲、含缺失值/类别**。HGBT 原生支持三者（L76、L140），省掉插补与编码链路。
2. **单模型已接近饱和、误差看起来「随机」**。此时集成的收益来自降方差（L438）。
3. **需要非线性 + 高阶交互，但不想手写特征交叉**。深度 $h$ 的树能表达 $h$ 阶交互（L337），`max_leaf_nodes=k` 能表达 $k-1$ 阶（L341）。
4. **有领域方向先验**（如「信用分越高越可能通过」，L174）。`monotonic_cst` 把先验写进假设空间，是本节唯一「用知识换泛化」的接口。

**不适合**（四条）：

1. **样本量很小（几十到几百）**。分箱会让切分点过粗（官方 `aside`，剪藏丢失），此时 `GradientBoosting*` 或线性/近邻基线更合适。
2. **需要外推**。树集成的预测永远落在训练目标范围内（[[decision-trees]] 的分段常数性质），时间序列趋势、物理外推都不行。
3. **需要可解释到「一条规则」**。单棵树可以画出来，几百棵树的集成只能给重要性/偏依赖（L388）。
4. **训练预算极紧、且已有线性基线够用**。集成是「用计算换精度」，`n_estimators` 直接乘训练时间（L493）。

## 二、选型顺序：从「一个实现」开始

| 你的情况 | 首选 | 理由 | 备选 |
| --- | --- | --- | --- |
| 表格数据，样本 ≥ 万级，有缺失/类别 | `HistGradientBoosting*` | 分箱 + 原生缺失/类别 + 早停（L22–L72） | LightGBM / XGBoost（未入库） |
| 表格数据，样本少（< 数千） | `GradientBoosting*` | 精确切分，小样本精度更好（官方 `aside`） | HGBT（调大 `max_bins`） |
| 需要强并行（`n_jobs`）、不想付 boosting 的串行成本 | `RandomForest*` | 树间独立，`n_jobs` 近线性（L500） | `ExtraTrees*` |
| 想要更低的方差、可接受更高偏差 | `ExtraTrees*` | 阈值也随机（L458） | RF（调小 `max_features`） |
| 基模型不是树（如 KNN、线性） | `Bagging*` | 任意黑盒 + 抽样（L583） | 手动多 seed 平均 |
| 几个**异质**模型差不多好 | `Voting*`（soft） | 平衡各自弱点（L600） | `Stacking*` |
| 想要「学权重」而不是等权 | `Stacking*` | 第二层在折外预测上训练（L769） | Voting + 手调 `weights` |
| 需要增量加树 | `Bagging*`、`RandomForest*`、`GradientBoosting*`、HGBT 的 `warm_start=True` | 复用上次解并追加轮数（L311、L545；HGBT 源码 docstring 同） | 增量**不**等于增量学习新数据：必须在同一份数据上重训 |

「万级 / 数千」是本库为了让规则可执行而划的经验线；官方只说「tens of thousands」与「small sample sizes」，没有给数量（L24 + 官方 `aside`）。

**先跑三行基线**（不要一上来就调参）：

```python
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
hgb = HistGradientBoostingClassifier(random_state=0).fit(X_train, y_train)   # 早停自动
rf  = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X_train, y_train)
```

对比 `hgb.score` / `rf.score` 与**训练时间**，再决定往哪条路调。官方对 RF vs HGBT 的结论就是「在你的数据上两个都试」（L450）。

## 三、HGBT 的调参次序

**第 0 步：确认早停状态。** 样本 >10,000 时早停默认开启（L72），`max_iter` 只是上限；实测 1.6.1：`max_iter=100` 实际 `n_iter_=76`。要看真实预算，先读 `n_iter_`。

**第 1 步：`learning_rate` × `max_iter`。** 固定 `learning_rate=0.1`（默认），把 `max_iter` 加到「早停稳定触发」；需要更高精度再降到 0.05 并把 `max_iter` 加倍。来源建议 $\nu\le0.1$ + 早停（L363）。

**第 2 步：树大小。** 三个旋钮（L62）：`max_leaf_nodes`（默认 31）、`max_depth`（默认 `None`）、`min_samples_leaf`（默认 20）。优先动 `max_leaf_nodes`：它直接限制交互阶数（L341），且比 `max_depth` 快。

**第 3 步：正则。** 依次试 `l2_regularization`（默认 0.0，叶子级 $ℓ_2$，L66）、`max_features`（默认 1.0，**每节点**特征子采样，L373；抽样层级见源码 docstring）、`max_bins`（默认 255，**调小即正则**，L64）。

**第 4 步：数据接口。** 缺失值不用插补（L76）；类别用 `categorical_features`（L144–L162）；有方向先验时加 `monotonic_cst`（L187）；已知不该交互的特征加 `interaction_cst`（L217）。

**第 5 步：早停细节。** 若自定义验证集，用 `early_stopping=True` + 传入 `X_val/y_val`（1.9 语义，外部核查）；`scoring` 用 `'loss'` 比 callable 快得多（L72）。

## 四、随机森林 / Extra-Trees 的调参次序

**第 1 步：`n_estimators` 先给足。** 默认 100，工程上直接 300–1000；超过临界点后收益消失（L493），所以用 OOB 或 CV 画一条「树数-误差」曲线找平台。

**第 2 步：`max_features`。** 默认分类 `"sqrt"`、回归 `1.0`（L493）。调小 → 方差更低、偏差更高；官方给的经验值：回归 `1.0`/`None`、分类 `"sqrt"`，文献常用 0.3（L493）。

**第 3 步：叶子大小。** `max_depth=None` + `min_samples_split=2` 是「长到纯」（L493），内存消耗大；先加 `min_samples_leaf`，再加 `max_leaf_nodes`。模型体积是 $O(MN\log N)$（L496）。

**第 4 步：OOB 还是 CV。** RF 的 `bootstrap=True` 才可用 `oob_score=True`（L493）；ET 默认 `bootstrap=False`，开 OOB 会直接报错（实测 1.6.1）。OOB 便宜，但来源只对**随机梯度提升**的 OOB 明确说「通常非常悲观，建议优先用 CV」（L378）；把它推广到 RF 是**本库推断**，本节没有对应论述。关键结论仍用 CV。

**ET 与 RF 的两个实质差异**：ET 的阈值随机（L458），且默认 `bootstrap=False`（L493）。因此同样的 `max_features` 下 ET 方差更低、偏差更高、训练更快；代价是数据子集带来的多样性也没有了。剪藏 doctest 的 0.98 / 0.999 / >0.999（L470–L486）来自 100 个球形簇，**不能用来排序 RF 与 ET**。

## 五、Bagging：只在「基模型不是树」时值得

`Bagging*` 的价值是「不改基算法就能降方差」（L574）。四种抽样（L578–L581）：

| 想要 | 参数 |
| --- | --- |
| Pasting（无放回抽样本） | `bootstrap=False` |
| Bagging（有放回抽样本） | `bootstrap=True`（默认） |
| Random Subspaces（只抽特征） | `bootstrap=False, bootstrap_features=True, max_samples=1.0` |
| Random Patches（样本+特征都抽） | `bootstrap=True, bootstrap_features=True, max_samples<1` |

默认 `n_estimators=10`（外部核查）——**这是本节最容易被忽略的默认值**：10 个基模型往往不够，先加到 50–200。

## 六、Voting：只在「异质且同强」时值得

```python
VotingClassifier([
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=300)),
    ('hgb', HistGradientBoostingClassifier()),
], voting='soft', weights=[1, 1, 1])
```

规则：

- `voting='hard'` 用 `predict`，不需要概率；平票按**类标签升序**取（官方 1.11.4.1，剪藏丢失；实测 1.6.1）。
- `voting='soft'` 要求每个基模型有 `predict_proba`（L659），否则 `fit`/`predict` 报错。
- `weights` 乘在概率上再平均（L638）；权重尺度无所谓，比例才重要。
- **别期待「集成 > 最好的单模型」**：官方 iris 示例是 0.95 vs 0.95（L628–L631）。

## 七、Stacking：预算够、且基模型异质时

```python
StackingClassifier(
    estimators=[('lr', LogisticRegression()), ('hgb', HistGradientBoostingClassifier())],
    final_estimator=LogisticRegression(),   # 默认也是它
    cv=5,                                    # 默认 5 折
    stack_method='auto',                     # predict_proba → decision_function → predict
    passthrough=False,
)
```

三条预算事实：

1. **成本 ≈ 单层 × (K+1)**：K 折折外预测 + 全量重拟合（L793）。
2. **二分类 + 概率输出会丢一列**（L795–L796）；`transform` 的形状是 `(n, n_estimators × 每模型列数)`。
3. **官方示例没有涨点**：单层 R2 = 0.53（L779），多层仍 0.53（L822）。先跑单模型基线，再决定是否值得这层复杂度。

**不要**在 stacking 内部再用同一个 CV 做早停/选参（会与折外切分耦合）；把调参放在外层。

## 八、AdaBoost：现在很少是首选

- 默认分类器是 `max_depth=1` 的 stump（L860；实测），回归器是 `max_depth=3`、`loss='linear'`（外部核查）。
- 对**错标样本**敏感（指数损失，本库判断），标签噪声大的数据不要用。
- 现代表格数据上，HGBT 通常更快更准；AdaBoost 的价值在「极简基模型 + 可解释的加性结构」。

## 九、缺失值、类别与约束的写法

```python
# 缺失值：不需要插补
HistGradientBoostingClassifier().fit(X_with_nan, y)

# 类别：三种写法等价
HistGradientBoostingClassifier(categorical_features=[True, False])
HistGradientBoostingClassifier(categorical_features=[0])
HistGradientBoostingClassifier(categorical_features=["site", "manufacturer"])
# DataFrame + category dtype：默认就是 "from_dtype"（1.6 起，外部核查）

# 方向先验：+1 单调增、-1 单调减、0 无约束
HistGradientBoostingRegressor(monotonic_cst=[1, -1, 0])

# 交互白名单：未列出的特征自成一组
HistGradientBoostingClassifier(interaction_cst=[{0, 1}, {1, 2}])
```

四个边界（都在剪藏里）：

- 类别基数必须 ≤ `max_bins`（L164 写「less than」，实现是「≤」，见 [[sklearn-ensembles]] 分歧 1）；**未见类别预测时按缺失处理**（L166）。
- 训练时该特征没出现 NaN，预测时的 NaN 走「样本更多的子节点」（L105）——这是一个与特征值无关的默认方向。
- 单调约束**只约束其余特征不变时的偏效应**（L198），不能表达 $x_1\le x_1'\Rightarrow F(x_1,x_2)\le F(x_1',x_2')$ 这类交叉约束；**多类分类不支持**（L204）；类别特征不支持（L209）。
- 样本权重会乘进梯度/Hessian，但**分箱阶段的分位数不看权重**（L136）：加权会改叶子值，不改候选切分点。

## 十、特征重要性：不要用 `feature_importances_` 下结论

- `feature_importances_` 是 MDI（不纯度下降），基于**训练集**统计，且**偏好高基数特征**（L518）。
- 用于筛选特征时先做 `permutation_importance`（L408）。
- GBDT 的 `feature_importances_` 在分类下被来源描述为熵口径（L408）；回归下是方差下降，本节未单独说明。它与 `permutation_importance` 的数值含义不同，**不可混用同一阈值**。
- 高基数陷阱的可操作检查：把 ID 类列加入特征，若它排到前列，说明 MDI 在用「唯一值多」冒充重要性。

## 十一、计算预算

| 项 | 事实 | 依据 |
| --- | --- | --- |
| RF/ET 并行 | `n_jobs=k` 分到 k 核，`-1` 用全部；加速比**不线性**（进程通信开销） | L500 |
| HGBT 并行 | OpenMP/Cython；分箱映射、直方图、找切分、样本分流、梯度/Hessian、预测都并行；**找箱阈值是串行的** | L238–L247 |
| 单节点代价 | `GradientBoosting*` $O(n_f n\log n)$ vs HGBT $O(n_f n)$ | L251、L253 |
| 分箱次数 | 只在流程开头排序一次 | L255 |
| 模型体积 | RF 默认参数 $O(MN\log N)$；用 `min_samples_split`/`max_leaf_nodes`/`max_depth`/`min_samples_leaf` 缩小 | L496 |
| 早停 | 默认 ≥10,000 样本开启；`n_iter_` 才是真实轮数 | L72 + 实测 |
| `subsample`/`max_features` 小 | 显著缩短运行时间（但 `subsample` 不能替代 `learning_rate`） | L369、L376 |

## 十二、18 条「踩到才发现」

1. **HGBT 用 `max_iter`，GBRT 用 `n_estimators`**（L37）——跨实现复制参数会静默失效（`set_params` 报错，但配置脚本常吞掉）。
2. **早停默认开启**（≥10,000 样本，L72）：`max_iter=1000` 可能只跑 80 轮；对比实验必须固定 `early_stopping`。
3. **`max_bins` 上限是 255**（源码区间 [2,255]），文档说「255 是默认且推荐」（L64）；`256` 直接 `InvalidParameterError`（实测）。
4. **类别基数检查是 `≤` 不是 `<`**（L164 vs 源码/实测）——按文档设 254 会浪费一档。
5. **未见类别 → 缺失**（L166）；若该特征训练时**从未出现 NaN**，缺失又会走「样本多的分支」（L105）。两个默认行为叠加时，新类别可能被静默归到多数类。
6. **类别特征不能加单调约束**（L209），因为类别无序。
7. **单调约束不支持多类**（L204），多类任务里不要写 `monotonic_cst`。
8. **`interaction_cst` 里未列出的特征自成一组**（L228）：写 `[{0}]` 不是「只允许 0 交互」，而是「0 与其余全部隔离」。
9. **`sample_weight` 不改分箱**（L136）：重加权样本后，切分点位置不变，只有叶子值变。
10. **RF 平均的是概率，不是投票**（L442，实测差值 0.0）：所以 `predict` 是概率 argmax，平票按概率和而非票数。
11. **ET 默认 `bootstrap=False`**（L493），开 `oob_score=True` 会直接报错（实测）。
12. **`max_features` 同名不同义**：RF 每**节点**抽（L436），Bagging 每**基模型**抽（L583）。
13. **`Bagging*` 默认只有 10 个基模型**（外部核查），且默认基模型是决策树（实测）。
14. **硬投票平票按类标签升序**（官方 1.11.4.1，剪藏丢失；实测）：重编码标签会改预测结果。
15. **软投票要求 `predict_proba`**（L659）；`SVC(probability=False)`（默认）不能直接进软投票。
16. **Stacking 的二分类会丢一列概率**（L795–L796），且内部 CV 让成本 ×(K+1)（L793）；官方示例两层都是 0.53（L779、L822）。
17. **`GradientBoosting*.criterion` 在 1.9 弃用、1.11 移除**（外部核查），而正文完全不提这个参数。
18. **`warm_start` 下 `random_state` 的语义与常规不同**（L570）：常规 `random_state` 保证「同一调用重复执行结果相同」，而 `warm_start` 是接着上次的解继续拟合、不重启随机流。等价关系是「一次拟合 20 棵 = 先 10 棵再 10 棵」（L563；实测 1.6.1 的 `predict_proba` 完全一致）。

## 十三、可自动化程度

| 任务 | 可自动化 | 说明 |
| --- | --- | --- |
| 三行基线（HGBT/RF） | **高** | 无需人工判断，只需记录时间与分数 |
| `n_estimators` 平台搜索 | **高** | 单调、可早停、指标稳定 |
| `learning_rate`×`max_iter` 网格 | **中** | 计算量大，且早停会改变有效轮数 |
| 树结构旋钮（`max_leaf_nodes`/`min_samples_leaf`） | **中** | 与数据规模强耦合，需分位数搜索 |
| 单调/交互约束 | **低** | 需要领域知识，写错会静默降精度 |
| Voting 权重 | **低** | 容易在验证集上过拟合；且通常无收益（L628–L631） |
| Stacking 结构与 `cv` | **低** | 成本高、易泄漏、官方示例不涨点 |
| 特征重要性解释 | **低** | MDI 有已知偏差（L518），必须换 permutation |

**给 AutoResearch 的边界**：前两行可以完全自动；第 3–4 行适合半自动（人来定搜索范围，机器扫）；约束类与 stacking 类需要人类先给假设，否则搜索空间既大又容易得到「验证集上的假提升」。

## 关联页面

- 来源：[[sklearn-ensembles]]（转换损失、默认值表、6 张图读数）
- 概念：[[ensembles]]（两条轴、函数空间梯度、折外预测）
- 上游：[[decision-trees]]、[[decision-trees-selection-and-tuning]]（基学习器的结构旋钮）
- 正则化：[[regularization-and-sparsity]]（第七个面：收缩与随机化预算）
- 评估：[[leaderboard-vs-cv]]（CV 的两种用途）
- 工具：[[scikit-learn]]（`ensemble` 模块矩阵与默认值）
- 对照：[[linear-models-as-baselines]]（先有线性基线，再上集成）
- 另一类聚合：[[anomaly-detection]]、[[outlier-detection-selection-and-tuning]]（`IsolationForest`：bagging 骨架 + 路径长度聚合，任务不是预测）
