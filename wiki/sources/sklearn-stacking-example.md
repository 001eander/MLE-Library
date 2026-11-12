---
title: "scikit-learn 示例「Combine predictors using stacking」：一维合成数据上的第二层回归与 SuperLearner 对照"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - ensemble
  - stacking
  - cross-validation
  - reference-doc
source_path: raw/articles/sklearn/Combine predictors using stacking.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 示例「Combine predictors using stacking」（1.9 重写版）

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Combine predictors using stacking |
| 作者 | 剪藏未署名；代码块头部是 `The scikit-learn developers` / BSD-3-Clause（L17–L19） |
| 原始 URL | `https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html` |
| 形态 | Sphinx-Gallery 自动示例页（一个可执行脚本 + 3 张图 + 1 张 HTML 估计器 repr），**不是论文、不是基准报告** |
| 本地文件 | `raw/articles/sklearn/Combine predictors using stacking.md`（298 行、14,705 B、1,369 词，其中散文 ≈873 词） |
| SHA-256 | `9ce0bbe584f676ebc8784f77fa336e760d4a72baa6eb8fae79708d431c962809` |
| 剪藏自带日期 | 只有 frontmatter `created: 2026-11-12`；抓取日期即创建日期 |
| 版本 | **scikit-learn 1.9.1**（判定见「版本判定」） |
| 代码 | 9 个 Python 块 + 5 个输出块（print/repr），全部保留 |
| 插图 | 3 张，已入库 `raw/assets/sklearn-stacking-example/`，**逐张目视核查**（见「图表核查」） |
| 参考文献 | 页面有 1 条 `[Polley2010]`，**剪藏只剩空标题**（L269） |

入库说明：本页依据本地剪藏撰写，行号指向原始剪藏。凡涉及页面 HTML 自带、但被剪藏吞掉的内容（估计器 repr 里的参数表），单独在「被吞掉的 API 说明」一节标出，并注明其来源是同一页面的 HTML，不是外部资料。

## 概要

一个回归示例，把三件事串在一条 1-D 合成数据上：**折外预测怎么训练第二层**、**元权重怎么读**、**`StackingRegressor` 与 R 的 `SuperLearner` 差在哪**。

数据由四项叠加而成（L31–L36）：均匀分布的 $X\in[-3,3]$（$n=500$）、线性趋势 $2.4X$、正弦项 $3.1\sin(3.2X)$、在 $x>2$ 处突降 10 的台阶，再叠加异方差高斯噪声（噪声尺度取 $\sqrt{0.75+0.75X^2}$）。台阶的存在是有意的：线性模型描述不了它，树模型天然能处理（L24）。

三个基模型刻意异质（L55–L77）：`StandardScaler`+`RidgeCV`（只能拟合趋势）、`SplineTransformer(n_knots=6, degree=3)`+`PolynomialFeatures(interaction_only=True)`+`RidgeCV`（平滑非线性）、`HistGradientBoostingRegressor(random_state=0)`（分段常数、能出台阶）。第二层是默认的 `RidgeCV()`（L48）。

来源的结论只有一句，而且是**成本**结论不是精度结论：「stacked regressor 结合了不同回归器的长处，但训练它比选出单个最好的模型贵得多」（L267）。

对本库的意义：这是第一篇**把 stacking 的每一步摊开、连权重和 SuperLearner 对照都算给你看**的官方来源，而且它属于 1.9 的整篇重写（见「版本判定」）。此前本库的 stacking 知识只有用户指南 1.11 的机制段（[[sklearn-ensembles]]）与 Kaggle 竞赛的 245 对 pair 工程细节（[[kaggle-tfms-llms-tabular]]），缺的正是「小规模、可读完、能对着权重看」的中间一层。

## 文档结构

| 行 | 小节 | 内容 |
| --- | --- | --- |
| L10–L15 | 引言 | stacking 的定义；与 `VotingRegressor` 的差别是「学出来的权重」而不是固定权重 |
| L22–L42 | Generate data | 数据生成代码 + 散点图（图 1） |
| L44–L95 | Stack of predictors on a single data set | 三个基模型 + 默认 `RidgeCV` 第二层；`passthrough` 的注意框；纯文本 repr |
| L97–L141 | Measure and plot the results | 四个模型在网格上的预测曲线（图 2） |
| L143–L202 | （续） | `cross_validate` + `cross_val_predict` + `PredictionErrorDisplay`（图 3） |
| L204–L222 | （续） | `final_estimator_.coef_`：`RidgeCV` 与 `LassoCV` 两组权重 |
| L224–L263 | How to mimic SuperLearner with scikit-learn | 凸权重 + 无截距的模仿；三条性质；归一化需自定义估计器 |
| L265–L267 | Conclusions | 贵 |
| L269–L298 | References / 运行时间 / 相关示例 | 文献条目已丢；脚本总运行 5.666 s；4 个相关示例链接 |

## 关键主张

### 1. Stacking 与 Voting 的差别是「第二层是否被训练」（来源论述）

来源的对照很干脆：`VotingRegressor` 用**固定（可由用户指定）的权重平均**预测，`StackingRegressor` 通过 `final_estimator` **学出**组合方式（L13）。两者输入同为基模型的输出，差别在这层是否是一个可训练的模型。

这解释了为什么 stacking 引入额外的泄漏面：一个有参数的组合器需要训练数据，而基模型在训练集上的预测是「已见过标签」的，所以第二层的训练数据只能来自折外预测。

### 2. 三个基模型是按「失效方式」选出来的，不是按「分数」选出来的（来源论述 + 本库综合）

来源给出的理由是：不同模型族可以达到相近表现但强弱项不同，stacking 用它们的互补性纠正单个模型改不了的系统误差（L46）。在这条 1-D 数据上，互补性是**肉眼可见**的：

| 基模型 | 能表达 | 表达不了 | 图 2 的表现 |
| --- | --- | --- | --- |
| 线性 Ridge | 直线 | 正弦、台阶 | 一条直线穿过整个散点 |
| 样条 Ridge | 平滑曲线 | 陡台阶 | 曲线贴合正弦，台阶处被抹平 |
| HGBT | 分段常数 | 平滑趋势 | 台阶清楚，但整个曲线是阶梯 |

**本库综合**：三段合起来覆盖了「趋势 / 平滑 / 台阶」三种结构，第二层才有东西可组合。这与 [[ensembles]] 里「成员误差相关性决定平均式集成的收益上限」是同一条原则，只是换到了精度轴而不是方差轴。

### 3. `passthrough=False` 时第二层看到的是纯预测值，因此不需要预处理（来源论述）

注意框（L50–L51）：基模型各自带了预处理（如 `StandardScaler`），但 `final_estimator` 在默认 `passthrough=False` 下**只收到基模型的预测**，所以不需要额外的预处理；若设 `passthrough=True`，`final_estimator` 应当是带正确预处理的管线。

这句话解释了示例里为什么 `RidgeCV()` 裸露在第二层：`StandardScaler` 装在**基模型的管线里**（L57），元特征的尺度由基模型决定，不再由原始特征决定。

### 4. 元权重把「谁在贡献」写成了数字（来源论述）

`RidgeCV` 作第二层时（L205）：`[-0.00446216, 0.44878552, 0.54762418]`。来源的解读（L208）：

- HGBT 主导（0.548）；
- 样条 ridge 有实质贡献（0.449）；
- 线性模型在另两个存在后不再提供有用信号，但**没有被丢弃**，而是拿到一个小的负权重去纠正残余偏置。

**本库综合**：线性项的权重比另外两项小两个数量级（0.0045 vs 0.45 / 0.55），所以「负权重 = 纠偏」这一读法在数值上是次要项；真正可读的信息是**排序与相对大小**。至于负号是否真的对应「残余偏置」，示例没有做消融，只能记为来源的解释。

### 5. L1 第二层会把无用成员的权重压到恰好 0（来源论述）

换 `LassoCV` 作第二层（L221）：`[0., 0.41148006, 0.56187293]`。来源的说法是「那个小的、无用的贡献被恰好置零，得到一个更简单的样条 ridge + HGBT 混合」（L210）。

这是一个干净的分工示例：

| 第二层 | 对无用成员的处理 | 结果形态 |
| --- | --- | --- |
| `RidgeCV`（$ℓ_2$） | 收缩但不为零，可带负号 | 三项混合，权重可解释为「份额 + 纠偏」 |
| `LassoCV`（$ℓ_1$） | 恰好为 0 | 两项混合，成员选择完成 |

注意两个权重不相同（0.4488 → 0.4115、0.5476 → 0.5619）：去掉一项会重新分配权重，所以**不能用 L1 的结果去解释 L2 的权重**。见 [[regularization-and-sparsity]]。

### 6. `SuperLearner` = 凸权重 + 无截距；`StackingRegressor` 默认是「无约束 + 截距」（来源论述）

来源的对照（L228）：`SuperLearner` 估计一组**凸**元权重（非负且和为 1）并**省掉截距**；`StackingRegressor` 默认用**无约束**的元学习器并**带截距**，还可以通过 `passthrough` 把原始特征也喂进去。

用 `LinearRegression(fit_intercept=False, positive=True)` 模仿（L232–L241）得到 `[2.41599723e-04, 4.48129539e-01, 5.49327451e-01]`；权重和「接近 1.0，但不恰好是 1」（L247–L254，页面印出 `0.9976985896404182`）。

来源明确交代了这个模仿的边界：**用 scikit-learn 强制归一化约束最干净的办法是自定义估计器，但那超出了本教程范围**（L263）。也就是说，上表的「凸」性质在这段代码里**只实现了非负**。

### 7. 无截距让权重成为「份额」，归一化带来三条性质（来源论述）

无截距时元权重可以直接读成对最终预测的**分数贡献**（L230）。把权重归一化到 1.0 换来三条性质（L259–L261）：

| 性质 | 内容 | 成立条件 |
| --- | --- | --- |
| Consensus-preserving | 所有基模型在同一点输出同一值时，集成返回该值，不放大也不衰减 | 权重和**恰好**为 1 |
| Translation-equivariant | 所有基预测同加一个常数，集成输出同加该常数 | 权重和**恰好**为 1 |
| 少一个自由度 | 避免与常数项冗余，在共线性下略微稳定权重 | 无截距 |

**本库综合**：三条性质都以「和恰好为 1」为前提，而模仿代码给出的是 0.9977——所以三条性质描述的是理想 `SuperLearner`，不是纸上这段代码。示例自己在下一句就承认了这一点（L263），但没有把「0.9977 与 1 的差会让三条性质各打多少折扣」写出来。

### 8. 示例自己的分数只出现在图上（来源论述，图 3 坐标轴注记）

正文从头到尾不给一个分数。唯一可读的数字在**图 3 每个子图的图例里**（`$R^2$` 与 MAE，均值 ± 折间标准差）以及子图标题里的评估耗时：

| 模型 | $R^2$ | MAE | 页面上的评估耗时 |
| --- | --- | --- | --- |
| Linear Ridge | 0.35 ± 0.09 | 2.91 ± 0.10 | 0.02 s |
| Spline Ridge | 0.79 ± 0.03 | 1.51 ± 0.12 | 0.02 s |
| HGBT | 0.79 ± 0.03 | 1.53 ± 0.14 | 0.24 s |
| Stacking Regressor | **0.80 ± 0.03** | **1.49 ± 0.14** | **1.16 s** |

正文对这四个数字的概括只有两句：「即使交叉验证后分数重叠相当多，stacked regressor 的预测仍然略好」（L196）、以及结论里的成本提醒（L267）。**「略好」的具体幅度（$R^2$ +0.01、MAE −0.04）比同一张图上印出的折间标准差（±0.03、±0.14）小**，这是示例自身两个组成部分之间的口径问题，记入「分歧与口径」。

耗时是**同一台机器上单次墙钟**，来源没有给硬件；只能读相对量级（stack ≈ 基模型总和的 4 倍以上），不能跨机比较。

### 9. 成本是结论的一半（来源论述）

结论段只讲一件事：训练 stacked regressor 比「选出表现最好的模型」贵得多（L267）。这与图 3 的耗时一致，也与 stacking 的结构一致：每个基模型要额外训练 $K$ 次来产出折外预测（本示例 $K=5$，见 [[stacking]]）。

## 图表核查（3 张，全部入库并逐张目视）

| 文件 | 尺寸 / 字节 | 目视读数 |
| --- | --- | --- |
| `sphx_glr_plot_stack_predictors_001.png` | 640×480 / 19,231 | 生成数据的散点图（`X` 横轴、`y` 纵轴，$y$ 覆盖约 −13.8 至 +6.5）。可见：一个正弦峰（$x\approx0.5$–$1$ 处最高 ≈+6）、一个谷（$x\approx-3$ 处 ≈−9）、$x>2$ 处整体下移约 10 的台阶、以及**随 $|x|$ 增大而变宽的散布**（异方差）。与 L31–L36 逐项吻合 |
| `sphx_glr_plot_stack_predictors_002.png` | 1000×800 / 89,105 | 2×2 共享坐标（`y` 刻度 7.5 / 5.0 / 2.5 / 0.0 / −2.5 / −5.0 / −7.5 / −10.0 / −12.5）的预测曲线：Linear Ridge 是一条直线；Spline Ridge 平滑贴合正弦，但在 $x\approx2$ 处把台阶抹成一个下凹；HGBT 是**阶梯**、在 $x\approx2$ 处出现整段下沉；Stacking 也是阶梯，但台阶更细、更平滑。四格都叠了同一批 `observed (sample)` 散点。横轴只比数据范围多 0.1，看不到 `extrapolation` 的实际形态 |
| `sphx_glr_plot_stack_predictors_003.png` | 900×700 / 161,779 | 2×2 `actual vs predicted`（`PredictionErrorDisplay(kind="actual_vs_predicted")`，红色虚线是 $y=x$），纵轴刻度被数据决定为 6.54 / 1.46 / −3.63 / −8.71 / −13.79。每格图例印 `$R^2$` 与 MAE，子图标题印 `Evaluation in … seconds`：0.02 / 0.02 / 0.24 / 1.16 s。第 4 格标题是 `Stacking Regressor`，即 L145–L191 那个把三个基模型与 `stacking_regressor` 一起遍历的循环 |

**图与正文的互补**：

- 4 个分数、4 个耗时、± 折间标准差**只出现在图 3 里**，正文一个数字都不给（L196 只有「slightly better」）；
- 图 2 是「三个基模型各缺一种结构」这一论断的唯一证据，正文只用文字描述（L99）；
- 图 1 与 L24 的描述一致，但**异方差的方向**（$|x|$ 越大越散）只有看图才知道，正文只说 "heteroscedastic"。

三张图与正文无定量冲突。图 2 里 Stacking 与 HGBT 的曲线关系（更平滑、台阶更细）与 L99 的「stacked regressor 是 HGBT 的平滑版」一致。

## 版本判定

**结论：scikit-learn 1.9.1，证据强，且这一页在 1.9 被整篇重写。**

1. **抓取端点证据**：`/stable/auto_examples/ensemble/plot_stack_predictors.html` 与 `/1.9/` 版本**字节级相同**（254,650 B，sha256 `a6e17c1e82ba7ec74b6c484dee1ac5e4f369cd7625893a5fe935021ca84c6b63`），页标题为 `scikit-learn 1.9.1 documentation`。
2. **内容级证据（重写而非微调）**：把 1.6.1、1.7.2、1.8.0 的同名页面逐一抓取比对，三版都是**另一篇教程**——`fetch_openml(name="house_prices")` 拉 Ames Housing（1,460 行、取 20 个特征、`np.log(y)`）、`OrdinalEncoder`/`SimpleImputer` + `ColumnTransformer` 预处理、`RandomForestRegressor` + `GradientBoostingRegressor` + `StackingRegressor`。剪藏里的合成 1-D 数据、`SplineTransformer`、`LassoCV` 第二层与 `SuperLearner` 对照在这些版本里**一个都没有**。
3. **判别性字符串**：`SuperLearner` 在 1.9 页出现 5 次，在 1.6.1 / 1.7.2 / 1.8.0 页出现 **0 次**；`interaction_only` 在 1.9 页出现 9 次，在 1.8 页 0 次。
4. **旁证（弱）**：剪藏末尾的「Total running time of the script: (0 minutes 5.666 seconds)」（L271）与 1.9.1 页面同字；它只说明剪藏与 1.9 页同源，不能区分小版本。

**对引用的含义**：`plot_stack_predictors.html` 这个 URL 在 1.9 之前教的是「Ames Housing 上 20 个特征的 stacking」，1.9 起教的是「1-D 合成数据的 stacking + SuperLearner」。任何缓存副本、二手引用或"我照着官方示例做的"式描述，**必须先确认它对应哪一版**，否则会在同一 URL 下讨论两个不同的实验。本页所有结论只对 1.9.1 的这一版成立。

## 转换损失

这份剪藏**散文几乎无损**（873 词，逐句比对后所有差异都属于链接/标记级），与 8.3 预处理那篇「接口全在、机制全无」形成对照。损失集中在两处：

### A. 整块 HTML 估计器 repr 被丢

官方页在 `StackingRegressor(...)` 的纯文本 repr 之后还有一个**可展开的 HTML 估计器 repr**（`<div class="sk-container" hidden>`），内含 **8 个 `<details>` 参数表、3,721 词**，覆盖 6 个类：

| 类 | 参数表词数 | 剪藏现状 |
| --- | --- | --- |
| `StackingRegressor` | 391 | 只剩纯文本 repr（L79–L93），参数全丢 |
| `StandardScaler` | 112 | 全丢 |
| `RidgeCV`（出现在 3 处） | 453 × 3 | 全丢 |
| `SplineTransformer` | 471 | 全丢 |
| `PolynomialFeatures` | 188 | 全丢 |
| `HistGradientBoostingRegressor` | 1,200 | 全丢 |

后果具体到阅读：剪藏里**没有** `cv` 的默认与语义（`None` → 5 折 `KFold`，且这些 splitter 以 `shuffle=False` 实例化）、**没有** `passthrough` 的定义、**没有** `stack_method='auto'` 的选择顺序、**没有** `final_estimator` 默认是 `RidgeCV()` 这句、**没有** `RidgeCV.alphas=(0.1, 1.0, 10.0)`。这些恰好是「第二层怎么训练」的全部机制说明。复原清单见下节。

### B. 参考文献与页内链接

- 正文 L226 的引用标记 `[Polley2010]` **被吞**（官方原文为 "The SuperLearner [Polley2010] is a stacking strategy…"），`References` 下只剩标题（L269），条目正文丢失。本库多份 sklearn 剪藏都有同一种失效（[[sklearn-preprocessing]]、[[sklearn-ensembles]] 的文献条目同样被吞），不是本页特有。
- L11 的 note 保留文字但**丢掉链接目标**（跳到下载页 / JupyterLite / Binder 的三个入口）。
- 其余：9 个代码块、5 个输出块、3 张图的直链、4 个相关示例链接全部保留。

## 被吞掉的 API 说明（同一页面 HTML 自带，非外部资料）

把官方 1.9.1 页的 HTML 还原后，可以读出剪藏缺失的那部分。以下每条都出自**这一页自己的估计器 repr**，不是从 API 文档另找的：

| 项 | 页面自带的说明 |
| --- | --- |
| `StackingRegressor.final_estimator` | 默认 `None`，此时用 `RidgeCV()`；"A regressor which will be used to combine the base estimators" |
| `StackingRegressor.cv` | 默认 `None` → 5 折；给整数则用 `(Stratified)KFold`；**这些 splitter 以 `shuffle=False` 实例化，因此跨调用折划分相同**；`cv` 只用于 `cross_val_predict` 产折外预测，**不用于模型评估** |
| `StackingRegressor.cv="prefit"` | 假定 `estimators` 已拟合、不再重训；此时 `final_estimator_` 用**全训练集**上的预测训练，**不是**交叉验证预测；若模型与被 stack 的模型用同一批数据训练，过拟合风险很高 |
| `StackingRegressor.passthrough` | 默认 `False`；为 `True` 时 `final_estimator` 同时收到基模型预测与原始训练数据 |
| `StackingRegressor.stack_method` | 默认 `'auto'` → 按 `predict_proba` → `decision_function` → `predict` 顺序选 |
| `RidgeCV.alphas` | `(0.1, 1.0, 10.0)`；`gcv_mode='auto'`（稠密用 SVD、稀疏退回 eigen）；`store_cv_results=False` |
| `SplineTransformer` | `n_knots=5`、`degree=3`、`knots='uniform'`、`extrapolation='constant'`、`include_bias=True`、`handle_missing='error'` |
| `PolynomialFeatures` | `degree=2`、`interaction_only=False`、`include_bias=True`；`interaction_only=True` 时**同一特征自身的高次项被排除**（保留 $x_0x_1$，排除 $x_0^2$） |
| `HistGradientBoostingRegressor` | `learning_rate=0.1`、`max_iter=100`、`max_leaf_nodes=31`、`min_samples_leaf=20`、`l2_regularization=0`、`max_bins=255`、`early_stopping='auto'`（样本 > 10 000 或给了验证集时开启） |

**这直接说明示例的两处隐含前提**：① 示例的 $n=500$，所以 HGBT 的 `early_stopping='auto'` 实际**关闭**，`max_iter=100` 会跑满；② 示例的 `cv` 用默认值，所以**外层 5 折与第二层内部的 5 折都是不洗牌的 `KFold`**——数据本身是 i.i.d. 均匀抽样，这不构成泄漏，但换到有顺序/分组的任务上就是另一回事（见 [[stacking]]）。

## 分歧与口径

> 本节只记录**来源自己说了什么、以及来源内部两处说法之间的口径差**；不引入外部实测。

1. **分数只在图上，正文只给「slightly better」。** 图 3 印出的 stack−HGBT 差是 $R^2$ +0.01、MAE −0.04，而同一张图印出的折间标准差是 ±0.03、±0.14。正文（L196）把这组数字概括为「略好」，没有提差值小于折间波动，也没有把「折间标准差不是标准误」这一点写出来（该口径见 [[cross-validation]]）。
2. **「correct its residual bias」是解释，不是测量。** L208 说线性模型的小负权重用于纠正残余偏置；示例没有做去掉该成员的对照，也没有给出该权重对预测的实际影响，因此这句话在本来源内不可验证。
3. **SuperLearner 段与它自己的代码不同。** L228 说 `SuperLearner` 的权重非负且和为 1，L259–L261 列出归一化带来的三条性质；但模仿代码只设了 `positive=True`，页面自己印出的权和是 0.9977，并在 L263 承认精确归一化需要自定义估计器。三条性质描述的是理想约束，代码实现的是「非负 + 接近 1」。
4. **`RidgeCV` 与 `LassoCV` 两组权重不可互推。** 去掉线性项后另外两项的权重都变了（0.4488 → 0.4115、0.5476 → 0.5619），正文（L210）只说「得到更简单的混合」，没有提醒权重会重新分配。
5. **`PolynomialFeatures(interaction_only=True)` 出现在样条之后，没有任何说明。** 这是三个基模型里唯一没有解释的设计选择：页面不说它为什么需要、也不给该管线的输出列数。按页面自带的默认值（`n_knots=6`、`degree=3`、`interaction_only=True`、`include_bias=True`）只能读出「基函数的两两乘积 + 偏置」这一形态，用途留给读者。
6. **`passthrough=True` 的建议是措辞，不是约束。** L51 用 "should be a pipeline with proper preprocessing" 描述正确做法，但页面不展示 `passthrough=True` 的行为差异，也不说明不做预处理会怎样。
7. **HGBT 的早停状态在示例里是隐含的。** 正文只说「树模型天然能处理台阶」（L24），没有交代 $n=500$ 使 `early_stopping='auto'` 关闭；这一信息只在被吞掉的参数表里（上节）。
8. **耗时无可比口径。** 图 3 的 0.02 / 0.02 / 0.24 / 1.16 s 是单机单次墙钟，硬件、线程数、`n_jobs` 都未写；L145–L191 的 `cross_validate` 用了 `n_jobs=-1`、`cross_val_predict` 也用了 `n_jobs=-1`，所以这组数字是并行后的结果。

## 局限

- **一维、$n=500$、i.i.d. 抽样、单一随机种子**（`np.random.RandomState(42)`）：没有组结构、没有时序、没有重复实验，所以整页只说明「机制怎么运作」，不构成任何方法优劣的证据。
- **只有一个数据集、一次 5 折**：图 3 的 ± 是折间标准差，不是重复实验的方差。
- **对照不完整**：没有 `VotingRegressor` 的同数据对照、没有「只选最好的单模型 + 调参」的同预算对照、没有 `passthrough=True` 的对照、没有「用非折外预测训练第二层会怎样」的对照。
- **没有报第二层的元特征维度**：样条管线的输出列数、`passthrough` 前后的差异都只能自己按默认值推。
- **`SuperLearner` 部分只有三行性质与一段模仿代码**，没有与真正的 R 包结果比对。
- **成本只给了一次墙钟**，没有随 $K$、基模型数量、样本量的曲线。

## 待解问题

- 折外预测与「全量重训后的基模型预测」之间的分布差在 $n=500$ 上有多大？元权重是在折外分布上拟合、却在全量分布上使用的，来源完全没有讨论这一点（机制见 [[sklearn-ensembles]] L769；本库把它记在 [[stacking]] 的待解问题里）。
- 第二层在 `RidgeCV` / `LassoCV` / 凸约束之间的差异，需要多大样本量才可分辨？本页的 3 组权重不同、但没有配套的分数对照。
- `SplineTransformer` 后接 `PolynomialFeatures(interaction_only=True)` 到底在买什么？它增加了「基函数的两两乘积」，与直接用更高 `degree` 的样条、或直接让第二层看到基函数，三者如何区分？
- 如果基模型改成同族（例如三个 GBDT 变体），这张图会变成什么样？[[ensembles]] 的 245-pair 案例给出「同族只值 +5e-5」的量级，但那是分类赛题、不同尺度。
- `cv="prefit"` 那条过拟合警告在什么条件下会真的咬人？页面只给了一句警告。
- 1.8 及更早版本的同 URL 教程（Ames Housing）与本版共享「stacking 略好」的措辞，但实验、特征与结论都不同；这两个版本的教学意图差异值得单独对照。

## 行号索引

| 主题 | 行 |
| --- | --- |
| stacking 定义、与 `VotingRegressor` 的差别 | L13 |
| 示例目的、合成数据说明 | L15、L24 |
| 数据生成代码（$n=500$、趋势、正弦、台阶、异方差） | L26–L40 |
| 图 1 散点 | L42 |
| 三个基模型与「默认 `RidgeCV` 第二层」 | L48、L53–L77 |
| `passthrough` 注意框 | L50–L51 |
| 纯文本 repr（含 `SplineTransformer(n_knots=6)`） | L79–L93 |
| 图 2 四格预测曲线 | L101–L141 |
| `cross_validate` + `cross_val_predict` + 图 3 | L145–L191 |
| 「分数重叠、stacked 略好」 | L196 |
| `RidgeCV` 元权重与解读 | L199–L208 |
| `LassoCV` 元权重与「置零」 | L210–L222 |
| `SuperLearner` 与无约束第二层的对照 | L224–L230 |
| 模仿代码与权重和 | L232–L254 |
| 三条归一化性质、需自定义估计器 | L257–L263 |
| 结论（成本） | L265–L267 |
| 空 `References` / 运行时间 / 相关示例 | L269、L271、L273–L298 |

## 关联页面

- 概念：[[stacking]]（本来源的主题页）、[[ensembles]]、[[cross-validation]]、[[regularization-and-sparsity]]
- 实践：[[ensembles-selection-and-tuning]]、[[competition-ml-workflow]]
- 工具：[[scikit-learn]]
- 其它来源：[[sklearn-ensembles]]（用户指南 1.11 的机制段）、[[kaggle-tfms-llms-tabular]]（245 对 pair 的大规模 stacking）、[[kaggle-getting-started]]（5-Fold Stacking 的手写实现）
