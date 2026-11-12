---
title: "scikit-learn 用户指南 2.7「Novelty and Outlier Detection」：outlier/novelty 两种问题、One-Class SVM、EllipticEnvelope、Isolation Forest 与 LOF（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - one-class-svm
  - isolation-forest
  - local-outlier-factor
  - robust-statistics
  - reference-doc
source_path: raw/articles/sklearn/2.7. Novelty and Outlier Detection.md
source_kind: article
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档正文与 API 签名）
---

# scikit-learn 用户指南 2.7「Novelty and Outlier Detection」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 2.7. Novelty and Outlier Detection（scikit-learn 用户指南第 2.7 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南 HTML→Markdown 剪藏；**5 个代码块（其中 1 条 `warm_start` doctest）、5 段行内公式、2 个 `[!warning]` 提示、6 张唯一图、1 个 5×3 表（LOF 方法可用性，L53–L59）、6 个 `Examples` 标题（10 条示例链接）、4 个 `References` 标题** |
| 原文 | <https://scikit-learn.org/stable/modules/outlier_detection.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/2.7. Novelty and Outlier Detection.md`（**204 行**（`wc -l`）、25,631 字节、`wc -w` 2,490（含 frontmatter 与 URL）、**正文 2,441 词**；末行有换行；**未做任何回填**） |
| SHA-256 | `8c19a014fd24895d22ae685fe66054dc2868902f05068daadd6885e7fb9262b4` |
| 完整性 | **本库损失最小的一份 sklearn 剪藏**：正文 2,441/2,486 词；84 条句子探针在官方 1.9.1 页命中 78 条，6 条未命中项逐条核为图片与 Markdown 相邻导致的切句差异，**无正文缺失**。丢的是 **1 条参考文献（Liu, Ting & Zhou 2008）与 2 处页内引用/链接标记**，详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.1，证据强**（剪藏含 1.9 才有的 `contamination`/`offset_` 句；`/stable/` 与 `/1.9/` 字节相同；见「版本判定」） |
| 插图 | **6 张唯一图全部下载入库** `raw/assets/sklearn-outlier-detection/`，**6 张均已目视核查**（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-outlier-detection]], L34`。凡超出剪藏内容的部分（API 默认值、`offset_` 的定义、复杂度、版本注记）都显式标注「外部核查」或「官方 1.9.1 页」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `8c19a014…`）。丢失的 1 条参考文献与 2 处引用标记只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库此前只有**消费侧**（`NearestCentroid`、`EllipticEnvelope` 的协方差、`OneClassSVM` 的两句话）而**没有名字的类目**：预测目标不是 $y$，甚至不是「把数据分组」，而是**「这条样本属不属于训练分布」**。它因此与前面十三个 sklearn 文档形成一组对照：

1. **第一次把问题定义本身当成选型依据。** L12–L20 给出两条互不相同的任务：`outlier detection`（训练集**已被污染**，要拟合「最密集的区域」）与 `novelty detection`（训练集**干净**，只判新样本）。两者不是「同一算法的两种用法」，而是**两个不同的统计目标**，并且带一条常被忽略的硬约束：**outlier detection 的异常点不能成簇**（L20），因为所有可用估计器都假设异常落在低密度区。
2. **第一次出现「分数 → 阈值 → 标签」三段式 API，并把它写成契约。** `score_samples` 给原始分数、`contamination` 定阈值（落成 `offset_`）、`decision_function = score_samples − offset_`、`predict = (decision_function < 0) ? −1 : 1`（L34、L36）。这也解释了为什么本节能用一个 5 行的可用性表（L53–L59）就把 LOF 的「哪种方法在哪种任务下可用」讲完。
3. **`contamination` 有三种互不相同的语义，文档只描述了一种。** 官方页只说「按 `contamination` 控制阈值」；外部核查 API 页显示：`IsolationForest`/`LocalOutlierFactor` 的 `'auto'` 是**两个写死的常数偏移**（−0.5 与 −1.5），`EllipticEnvelope` 的默认是 **0.1**（没有 `'auto'`），`OneClassSVM` **根本没有这个参数**（它用 `nu`）。因此「把 `contamination` 当统一旋钮」在 API 层不成立，而这一节从头到尾没有点明。
4. **第一次让「无监督模型的评估」成为必要但缺席的东西。** 本节给了 6 张图、13 个 Examples 链接，却**没有任何指标**：`sklearn.metrics` 里也没有异常检测分数。L71–L72 指向的两个示例（`plot_anomaly_comparison`、`plot_outlier_detection_bench`）用的是**已知真值**下的 ROC 曲线——即把无监督方法当二分类器评。本库据此在实践页把「评估」单列一节：没有标签时只能用稳定性与业务判据。
5. **它把前面几页接了起来**：[[nearest-neighbors]] 的 kNN 图在这里变成 LOF 的密度估计器（表 L53–L59 的第一行就是「LOF 没有 `predict`」）；[[ensembles]] 的 bagging 骨架在这里变成 `IsolationForest`——**同一骨架、不同聚合**（平均概率 → 平均路径长度），并且聚合的是「隔离难易」而不是预测误差；[[support-vector-machines]] 的 $\nu$ 在这里第二次出现，但含义从「训练误差/支持向量比例上界」变成「新样本落在边界外的概率」（L80）；[[covariance-estimation]] 的 MCD 在这里有了下游消费者 `EllipticEnvelope`，并且**同一个估计器既是建模工具又是诊断工具**。

## 文档结构

| 章节 | 解决什么问题 | 代表对象 | 行号 |
| --- | --- | --- | --- |
| 引言 | outlier vs novelty 两个定义；`predict`/`score_samples`/`decision_function` 契约；LOF 特例表 | — | L10–L59 |
| 2.7.1 Overview | 5 个算法的横向对照（图 1 的 5×5 面板） | `IsolationForest`/`LOF`/`OneClassSVM`/`SGDOneClassSVM`/`EllipticEnvelope` | L61–L72 |
| 2.7.2 Novelty Detection | 学一条「包住训练分布」的边界 | `svm.OneClassSVM` | L74–L90 |
| 2.7.2.1 Scaling up the One-Class SVM | 线性复杂度替代品 + 核近似 | `linear_model.SGDOneClassSVM` | L92–L98 |
| 2.7.3 Outlier Detection | 没有干净训练集时怎么定义「核心区」 | — | L100–L102 |
| 2.7.3.1 Fitting an elliptic envelope | 假设高斯 → 稳健椭圆 → Mahalanobis | `covariance.EllipticEnvelope` | L104–L121 |
| 2.7.3.2 Isolation Forest | 用「被隔离所需的切分次数」当异常分数 | `ensemble.IsolationForest` | L123–L156 |
| 2.7.3.3 Local Outlier Factor | 用「邻域密度比」当异常分数 | `neighbors.LocalOutlierFactor` | L158–L185 |
| 2.7.4 Novelty detection with LOF | 同一个类切到归纳模式 | `LocalOutlierFactor(novelty=True)` | L187–L205 |

## 关键主张（按剪藏顺序）

### A. 两个定义、一条硬约束、一份契约（L10–L59）

- **outlier detection = 训练集被污染**（L14）：「拟合训练数据最密集的区域，忽略偏离的观测」。**novelty detection = 训练集干净、只判新样本**（L18）。L20 把两者归到 `anomaly detection` 之下，并给出本库第一条**类目级不对称**：「outlier detection 里异常点不能成簇；novelty detection 里异常点可以成簇，只要它落在训练分布的低密度区」。
- **标签约定是 1/−1，不是 0/1**（L34）：这是与所有分类器 API 的硬差异。`predict` 的输出直接是 `±1`，接 `sklearn.metrics` 前必须重编码。
- **三段式契约**（L34、L36）：`score_samples` 是**原始打分函数**，`contamination` 控制**阈值**，`decision_function` 是**平移后的分数**（负 = 异常，非负 = 正常）。官方没有把阈值参数本身暴露成属性，落地的名字是 `offset_`（L172 首次点名，且只出现在 LOF 小节）。
- **`contamination` 同时决定「报多少个」与「报哪些」**：它按训练集分数的分位数确定阈值（L172）。`'auto'` 则使用 API 页给出的固定 `offset_`，**不是统一的 10% 异常比例**。
- **LOF 是唯一的例外，而且例外方式有两层**（L42、L44、L53–L59）：默认（`novelty=False`）只有 `fit_predict`，训练样本的分数在 `negative_outlier_factor_`；要 `predict`/`decision_function`/`score_samples` 必须 `novelty=True`，而那时 `fit_predict` **不可用**。
- **`novelty=True` 时不能在训练样本上调用这三个方法**（L46–L49，`[!warning]`，且 L198–L201 重复一遍）。文档明确警告：此时训练集上的 `predict` 结果可能与 `fit_predict` 不同，两种分数口径不能混用。

### B. Novelty detection：一条边界与一个概率参数（L74–L98）

- **标准答案是一条「粗糙但贴合」的边界**（L78）：落在边界内 = 来自同一总体，落在边界外 = 异常。来源用「a rough, close frontier」（粗糙而贴近）——这两个词互相拉扯，本库把它读成**边界复杂度是被刻意限制的**（否则它就退化成核密度估计）。
- **`OneClassSVM` 的两个超参没有正解**（L80）：核（默认 RBF）与带宽（$\gamma$）。原文直言「there exists no exact formula or algorithm to set its bandwidth parameter」。这是本库第二个「默认值本身就是任意选择」的案例（第一个是 `ShrunkCovariance.shrinkage=0.1`，见 [[sklearn-covariance]] L34）。
- **$\nu$ 的语义在这里换了**（L80）：在 `SVC` 里 $\nu$ 是「训练误差与支持向量比例的上界/下界」，这里被写成「**新来的、但仍然正常的观测落在边界外的概率**」。两种口径的关系需在目标版本中重新验证，当前只保留来源分歧，不用历史运行时读数裁决。
- **`SGDOneClassSVM` 是「同一目标、另一条求解路径」**（L94）：线性复杂度 $O(n)$，配核近似（Nyström）可逼近核化版本，而核化版本的复杂度「at best quadratic」。近似程度与实际成本需按目标环境评估。

### C. EllipticEnvelope：把「形状」假设写成椭圆（L104–L121）

- **这一小节的全部内容就是一句假设**（L106、L108、L110）：正常数据服从已知分布（例：高斯）→ 定义数据的「形状」→ 用**稳健**协方差估计中心与形状（不被离群点影响）→ 用 **Mahalanobis 距离**当异常度。
- **它与 2.6 节是同一套估计器、不同的用法**（本库综合）：`EllipticEnvelope` 内部就是 `MinCovDet` + 一个阈值；2.6 节把 MCD 当「估计 $\Sigma$ 的工具」，2.7 节把它当「判异常的工具」。因此 [[covariance-estimation]] 里关于 `support_`≠h、`raw_covariance_` 与校正因子的三条警告**在这里全部适用**。
- **`contamination` 的默认是 0.1，且不接受 `'auto'`**（外部核查 1.9.1 签名 `EllipticEnvelope(*, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None)`）。这是本库第一次出现「**同一族估计器的同名参数默认值语义不同**」（IF/LOF 用 `'auto'`，EE 用 0.1，OCSVM 没有）。

### D. Isolation Forest：把「隔离难度」当分数（L123–L156）

- **算法只有三步**（L125–L131）：随机选一个特征 → 在该特征的最小/最大值之间随机选切分点 → 递归。样本被隔离所需的切分次数 = 从根到叶的路径长度；森林上的**平均路径长度**就是正常度。随机划分对异常产生**明显更短**的路径。
- **实现是 `ExtraTreeRegressor` 的集成，且深度被写死**（L133）：每棵树的最大深度为 $\lceil\log_2 n\rceil$（$n$ = 该树的样本数）。
- **`max_samples` 决定「什么算大样本」，默认值是个常数**（外部核查 `max_samples='auto'` → `min(256, n_samples)`；`max_samples_` 属性给出实际值）。单棵树默认最多见 256 个点，实际成本仍需在目标环境评估。
- **`warm_start` 的语义是「只训练新增的树」**（L139–L148）：示例从 10 棵加到 20 棵。更改其他会影响既有树训练方式的参数是否安全，文档没有说明，需在目标环境中单独验证。

### E. LOF：局部密度比、两个 k 判据与一条阈值规则（L158–L205）

- **LOF 的定义是比值，不是距离**（L164）：某点分数 = 其 k 近邻的**平均局部密度** / 该点自身局部密度。正常点 ≈ 1，异常点 ≫ 1（代码里取负号后 ≈ −1 与更小）。因此 LOF 的输出**没有单位、也不可比跨数据集**。
- **$k$ 的两条判据都是「结构性」的**（L166）：要大于「一个簇至少要有的对象数」（否则簇内点会被判为彼此的局部异常），要小于「可能成为局部异常的对象数的最大值」。实践中 $k=20$ 一般够用，**异常比例 > 10% 时应取更大（示例用 35）**。
- **LOF 的强项是「局部 vs 全局」而不是「更准」**（L168）：问题不是「样本多孤立」，而是「相对周围邻域多孤立」。它与其他全局方法的排名差异需要按数据和目标环境评估，不能用单次实验概括。
- **阈值规则终于写清楚了**（L172，**1.9 新增句**）：`contamination` 给定时，`offset_` = 训练集 `negative_outlier_factor_` 的对应百分位；**分数严格小于阈值**才判为异常。`predict` 使用重新计算的推理分数，不能与训练集的 `negative_outlier_factor_` 混为一谈。
- **`n_neighbors` 的上界需要显式检查**：文档只给下界式判据，没有说明超过样本数时的行为；该边界行为留待目标环境验证。

## 版本判定

**1.9.1，证据强。**

1. **剪藏含 1.9 新增句**：L172「When the `contamination` parameter is set, the threshold (called `offset_`) is determined as the corresponding percentile of `negative_outlier_factor_` scores on the training data. Samples with scores strictly below this threshold are classified as outliers.」在 1.8 页**完全没有**，1.8→1.9 的正文 diff 只有这一处（按句切分，7 行 diff 全部属于该段）。
2. **1.7→1.8 的 diff 是「引用格式」而非内容**：L133 的父句在 1.7 是 "(see (Liu et al, 2008) for more details)"、1.8 起是 "(see [1] for more details)"，同时 References 排版变化（1.7 的 "Technometrics" 式换行 → 1.8 的一行式）。剪藏把引用标记整个吞掉，因此这一版差异**不能用于判版本**，只能用于判损失。
3. **1.6→1.7 的三道判据全部命中**：L78「the same population **as** the initial observations」（1.6 是 "than"）、L110「The Mahalanobis distances … **are** used」（1.6 是 "is used"）、L203「Novelty detection with LOF is illustrated below」整句（1.6 无此句）。
4. **句子探针**：84 条 ≥14 词的句子在官方 1.6/1.7/1.8/1.9.1 四版页的命中数是 **75/77/77/78**；6 条未命中项逐条核为「图片 Markdown 与下一段粘连」与「`[!warning]` callout 的标题重复」，**无正文缺失**。
5. **抓取日 `/stable/modules/outlier_detection.html` 与 `/1.9/modules/outlier_detection.html` 字节相同**（79,533 B，sha256 `310cf68b…`），页标题 `scikit-learn 1.9.1 documentation`。
6. **外部旁证**（官方 API 页的 version 注记）：`IsolationForest` 与 `LocalOutlierFactor` 的 `contamination` 默认值 **0.22 起从 0.1 改为 `'auto'`**，`OneClassSVM.gamma` **0.22 起从 `'auto'` 改为 `'scale'`**，`SGDOneClassSVM` 1.0 新增，`offset_` 属性 0.20 新增。

## 转换损失

**本库损失最小的一份 sklearn 剪藏：正文完整，只丢 1 条文献与 2 处标记。**

| 位置 | 官方 1.9.1 原文 | 剪藏 | 影响 |
| --- | --- | --- | --- |
| L133 | `... where n is the number of samples used to build the tree (see **[1]** for more details).` | `... (see for more details).` | 句子变病句；引用标记 [1] 丢失 |
| L156（`References` 标题下） | `[1] F. T. Liu, K. M. Ting and Z.-H. Zhou. “Isolation forest.” 2008 Eighth IEEE International Conference on Data Mining (ICDM), 2008, pp. 413-422.` | **空** | 整条文献丢失（唯一一条） |
| L170 | `... the result of predict may differ from that of fit_predict. See **Novelty detection with Local Outlier Factor**.` | `... See.` | 句子截断；页内链接文字丢失 |
| L82 / L119 / L183 | 三条文献 | **完整** | — |

其他：

- **无 `<details>` 折叠块**（本页官方 HTML 里 0 个），所以本库「数学表述全丢」的常见问题在这里**不存在**；代价是本节本来就没有复杂度表，成本只能参考图标题或在目标环境重测。
- 公式渲染噪声：L133 的 $\lceil \log_2(n) \rceil$ 被渲染成 `$\lceil log_{2} \left(n\right) \rceil$`；L76、L78 的 `$p$` 与后文空格粘连（`$p$ -dimensional`）。
- 5 个代码块里只有 4 段是真 Python（L24–L26、L30–L32、L38–L40 是三行骨架、L141–L149 是 doctest、L191–L194 是 LOF 用法）；doctest **无输出**（`>>>` 无返回行），因此本页只转述「10 棵 → 20 棵」的来源写法，不声称复现。
- 图片全部保留（6/6），且 URL 指向 `/stable/`。

## 外部核查（官方 1.9.1 API 页，不在剪藏内）

5 个估计器的完整签名（1.9.1）：

| 估计器 | 签名关键参数（默认值） |
| --- | --- |
| `svm.OneClassSVM` | `kernel='rbf'`、`degree=3`、`gamma='scale'`、`coef0=0.0`、`tol=1e-3`、`nu=0.5`、`shrinking=True`、`cache_size=200`、`verbose=False`、`max_iter=-1` |
| `linear_model.SGDOneClassSVM` | `nu=0.5`、`fit_intercept=True`、`max_iter=1000`、`tol=1e-3`、`shuffle=True`、`learning_rate='optimal'`、`eta0=0.01`、`power_t=0.5`、`warm_start=False`、`average=False` |
| `covariance.EllipticEnvelope` | `store_precision=True`、`assume_centered=False`、`support_fraction=None`、`contamination=0.1`、`random_state=None` |
| `ensemble.IsolationForest` | `n_estimators=100`、`max_samples='auto'`、`contamination='auto'`、`max_features=1.0`、`bootstrap=False`、`n_jobs=None`、`random_state=None`、`warm_start=False` |
| `neighbors.LocalOutlierFactor` | `n_neighbors=20`、`algorithm='auto'`、`leaf_size=30`、`metric='minkowski'`、`p=2`、`contamination='auto'`、`novelty=False` |

`offset_` 的三个官方定义（这决定了阈值的行为，剪藏只有 L172 一句）：

- `IsolationForest`：「`decision_function = score_samples - offset_`。当 `contamination='auto'` 时 **offset 等于 −0.5**（正常样本的分数接近 0、异常的接近 −1）；当给定 `contamination` 时，offset 定义为使训练集出现**预期数量的异常**（`decision_function < 0`）」。
- `LocalOutlierFactor`：「`negative_outlier_factor` 小于 `offset_` 的观测被判为异常。**offset 取 −1.5**（正常样本的分数约 −1），除非给了 `'auto'` 以外的 `contamination`」。
- `OneClassSVM` / `EllipticEnvelope` / `SGDOneClassSVM`：同一恒等式；`OneClassSVM.offset_` 是 `intercept_` 的相反数（0.20 起提供），`EllipticEnvelope.offset_` 由 `contamination` 决定。

版本注记（同上）：`IsolationForest`/`LocalOutlierFactor` 的 `contamination` 默认 **0.22 起** `0.1 → 'auto'`；`OneClassSVM.gamma` **0.22 起** `'auto' → 'scale'`；`SGDOneClassSVM` **1.0 新增**；`offset_` **0.20 新增**；`IsolationForest.base_estimator_` **1.2 起改名 `estimator_`**。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 图表核查

6 张图（与官方 1.9.1 页的 6 张一一对应）于 2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200）并入库，**逐张目视核查**：

| # | 文件 | 尺寸 | SHA-256 前 16 位 | 读数要点 |
| --- | --- | --- | --- | --- |
| 1 | `sphx_glr_plot_anomaly_comparison_001.png` | 1400×1250 | `74b98b8010f89b98` | 5 列（Robust covariance / One-Class SVM / One-Class SVM (SGD) / Isolation Forest / Local Outlier Factor）× 5 行数据的对照总图；**LOF 列没有黑色决策边界**（与 L63 一致）；逐面板标题带运行时间 |
| 2 | `sphx_glr_plot_oneclass_001.png` | 640×480 | `735dd95886853ef0` | 「Novelty Detection」单面板，一条红色闭合边界包住两团训练点；页脚 `error train: 19/200 ; errors novel regular: 2/40 ; errors novel abnormal: 1/40` |
| 3 | `sphx_glr_plot_mahalanobis_distances_001.png` | 1000×500 | `5e85e45480d15441` | 蓝色虚线（MLE 距离）与橙色点线（MCD 距离）两族椭圆；红点（真异常）落在蓝线内、橙线外——**这正是 EllipticEnvelope 的判据** |
| 4 | `sphx_glr_plot_isolation_forest_003.png` | 640×480 | `fee1dc4388a5bca7` | 「Path length decision boundary of IsolationForest」；色标 −1.00…+1.00，两团点为亮黄（正常）、四角为深紫；**边界是轴对齐的阶梯状**，直观说明 IF 只做轴对齐切分 |
| 5 | `sphx_glr_plot_lof_outlier_detection_001.png` | 640×480 | `d737e0cc87c28500` | 两团密集点 + 红圈（半径 ∝ LOF）；页脚 `prediction errors: 8`——即 `fit_predict` 与真值差 8 个 |
| 6 | `sphx_glr_plot_lof_novelty_detection_001.png` | 640×480 | `78f1c5d0ec2eae7a` | 「Novelty Detection with LOF」：等高线着色的分数场 + 红色 learned frontier；白圈=训练点、紫=新正常、黄=新异常；页脚 `errors novel regular: 8/40 : errors novel abnormal: 0/40` |

要点：

- **图 1 是本节唯一的横向对照**，但它只给「哪列把哪个形状包住」，不给任何数字指标；面板标题的运行时间（每格都带运行时间：Robust covariance 约 `.04`–`.08s`、One-Class SVM `.00s`、One-Class SVM (SGD) `.01s`、Isolation Forest `.14`–`.16s`、LOF `.00s`——LOF 最快、IF 最慢，差约 1 个数量级）是**本节唯一可引的成本证据**。
- **图 5 与图 6 是同一个估计器的两种形态**：图 5（`fit_predict`）只给「哪个点异常」，图 6（`novelty=True`）给整片**分数场**。这正是 L42–L59 那张表的图形版：传导模式能画边界，归纳模式不能。
- **图 3 与 [[sklearn-covariance]] 的第 5 张是同一个文件**（`5e85e454…`，1000×500）：2.6 节用它说明「MCD 比 MLE 稳」，2.7 节用它说明「Mahalanobis 距离就是异常度」。同一个资产在两个叙事里的用法不同，引用时须写清是哪一个论点。
- 5 行数据（图 1）里最值得注意的是**最后一行（均匀噪声）**：`One-Class SVM` 在纯噪声上挖出多个「洞」，`One-Class SVM (SGD)` 几乎不挖洞，`Isolation Forest` 沿四边画出一个矩形边界——**没有任何方法在无结构数据上给出有意义的答案**，与 L67「high-dimension / without assumptions 非常困难」一致。

## 关键判断

1. **选型的第一性判据是「训练集是否被污染」+「要不要给新样本打分」**，不是算法名字。前者决定 outlier vs novelty（L14/L18），后者决定能不能上线（LOF 默认形态只能 `fit_predict`，表 L53–L59）。
2. **`contamination` 不是统一旋钮**：`IF`/`LOF` 的 `'auto'` 是常数额度（−0.5 / −1.5），`EE` 是 0.1，`OCSVM` 用 `nu`。因此「同一份数据上比较五个算法的默认输出」没有意义——它们的默认阈值口径不同（本库综合）。
3. **`offset_` 是本节唯一真正的「模型参数」**：分数函数由算法固定，能调的只有阈值。而阈值一旦由 `contamination` 定成「训练集分位数」，它就成了**对训练集污染率的显式承诺**——`'auto'` 只是把这句承诺藏进常数里。
4. **LOF 的两个模式不是「同一模型的两种调用」，是两个统计量**：文档明确警告 `novelty=True` 的 `score_samples` 在训练点上可与 `negative_outlier_factor_` 不同。工程上应当**把训练集分数与推理分数分别缓存**。
5. **`IsolationForest` 的「高维高效」与 `max_samples=256` 这个默认上限有关**：单棵树只看有限子样本，而不是靠降维保留完整分布。实际扩展性仍需在目标环境测量。
6. **`EllipticEnvelope` 是 2.6 节的下游，不是新算法**：它的判据就是稳健 Mahalanobis 距离，所以 2.6 节关于 `support_`、`raw_covariance_`、`n>p` 的三条警告全部继承。本节唯一新增的是阈值层（`contamination=0.1`）。
7. **本节完全没有评估**：6 张图、13 个示例、0 个指标。而两个官方示例（L71–L72）都用**已知真值的 ROC** 来评估无监督方法——这是本库目前「评估器与目标错位」的最干净样本（见 [[leaderboard-vs-cv]] 的同类张力）。
8. **$\nu$ 的两种读法需要分开**：L80 说「新正常样本落在边界外的概率」，SVM 文档则使用训练误差/支持向量比例口径。当前不在没有目标环境验证的情况下裁决。

## 分歧（9 条）

1. **L80 的 $\nu$ 语义**：文档写「probability of finding a new, but regular, observation outside the frontier」，SVM 文档则采用训练误差/支持向量比例口径。两种定义的关系保留为待验证分歧。
2. **`contamination` 的「预期数量」与推理结果可能不同**：L172 用训练集 `negative_outlier_factor_` 的分位数定义 `offset_`，而 `predict` 使用重新计算的推理分数；LOF 的训练集计数与 `predict` 计数不能直接等同。
3. **`IsolationForest` 与 LOF 的 `'auto'` 阈值没有统一异常比例语义**：API 页分别给出固定 `offset_` −0.5 与 −1.5，但未把它们承诺为某个通用污染率。
4. **L67「OneClassSVM 对异常敏感、检测效果不好」与 L80 的推荐并存**：同一页既推荐用 One-Class SVM 做 novelty detection，又说它在 outlier detection 上表现差。应区分两种任务，不合并为统一的性能结论。
5. **`LocalOutlierFactor` 的 `n_neighbors` 上限没有任何提示**：文档 L166 只给下界式判据，没有说明超过样本数时的行为。
6. **L20 的「异常不能成簇」需要按算法假设理解**：它是本节的任务前提，不应在没有目标环境对照的情况下改写成所有估计器都同样成立的性能排序。
7. **`contamination=0.0` 不可表达**：API 参数约束是 `(0.0, 0.5]`，文档没有说明这个下界。

## 局限

- **本节没有复杂度分析**（无 `<details>` 块），成本只能从图 1 的面板时间（量级）或目标环境复测获得；LOF 的 $O(n^2)$ 最坏情形、IF 的 $O(t\psi\log\psi)$ 都**不能引剪藏**。
- **本节不涉及评估**：没有任何指标、没有交叉验证、没有「污染率如何估」。两个官方示例用 ROC，但那需要真值。
- **本节不涉及特征工程**：缩放只在 2.6 节（协方差）与邻居章节隐含；核方法与局部密度方法对尺度的敏感性需在目标环境中评估。
- **本节不涉及类别不平衡、多类或半监督扩展**，也不涉及 `sklearn.metrics` 里的任何异常指标（该类目在 1.9.1 中也不存在专门的异常检测指标）。
- **剪藏丢了唯一的 Isolation Forest 文献**（Liu, Ting & Zhou 2008），所以「路径长度的调和数校正 $c(n)$」「异常分数的 $2^{-E(h)/c(n)}$ 定义」**不在本库可引范围内**（L129–L133 只有定性描述）。

## 待解问题

1. **`IsolationForest` 的分数与路径长度的精确关系**：官方实现里 `score_samples` = $-2^{-E(h)/c(\psi)}$，$c(\cdot)$ 是 BST 不成功搜索的期望路径长度。剪藏不含该式，需读 Liu et al. 2008 或源码，才能把「−0.5 阈值」翻译成「路径长度 = $c(\psi)$」。
2. **`offset_` 的精确分位数规则**：`IsolationForest` 用 `np.percentile(score_samples, 100*contamination)` 还是排序后取第 $k$ 个？并列分数时如何处理，需在目标版本源码与环境中核对。
3. **`LocalOutlierFactor` 的「自身排除」语义**：`novelty=True` 的训练分数与 `negative_outlier_factor_` 的系统性差异需逐点核对。
4. **`OneClassSVM` 的 $\nu$ 与「新样本落在边界外概率」的关系**：需要按 Schölkopf et al. 2001 的定义与目标环境结果共同核对。
5. **`EllipticEnvelope` 的 `contamination` 与 2.6 节 `MinCovDet.support_fraction` 的交互**：两者都控制「用哪些点」与「报多少个」，是否会在同一数据上给出矛盾的支持集？未验证。
6. **`SGDOneClassSVM` + Nyström 的 `n_components` 与 $\nu$ 的联合调参**：高维下 Nyström 的秩近似会不会让 $\nu$ 的语义失真？未验证。
7. **`IsolationForest` 的 `max_features < 1.0`**：剪藏完全不提（默认 1.0），而它是与 `max_samples` 并列的第二个预算旋钮。未测其对本库数据的 AUC 影响。
8. **无标签时的评估协议**：本库目前只有「用真值算 ROC」这一条（官方示例）。扰动稳定性、跨种子一致性、业务阈值回测都待补。
9. **`LocalOutlierFactor` 在稀疏矩阵、`metric='precomputed'` 下的行为**：与 [[nearest-neighbors]] 的 `algorithm='auto'` 规则同源，但本节未提。
10. **`contamination` 与 `offset_` 外推**：把在一个样本量/污染率上拟合的模型用到另一个先验上，是否需要重算 `offset_`（而不是重训）？API 没有入口，`offset_` 是拟合期属性。

## 行号索引

| 内容 | 行号 |
| --- | --- |
| outlier / novelty 两个定义 | L12–L18 |
| 「异常不能成簇」的硬约束 | L20 |
| `fit` / `predict` / `score_samples` / `decision_function` 契约 | L22–L40 |
| LOF 的特例说明与 5 行可用性表 | L42–L59 |
| 2.7.1 总览与横向对照（图 1） | L61–L72 |
| 2.7.2 边界式 novelty detection | L74–L78 |
| `OneClassSVM` 与 $\nu$ 的语义 | L80 |
| 2.7.2.1 `SGDOneClassSVM`（线性复杂度） | L92–L94 |
| 2.7.3 outlier detection 的前提 | L100–L102 |
| 2.7.3.1 椭圆包络与稳健 Mahalanobis | L104–L110 |
| 2.7.3.2 Isolation Forest 三步与深度 $\lceil\log_2 n\rceil$ | L123–L133 |
| `warm_start` 示例（唯一代码块） | L141–L149 |
| 空 `References`（丢 Liu et al. 2008） | L156 |
| 2.7.3.3 LOF 定义、$k$ 判据、局部 vs 全局 | L158–L168 |
| LOF 的方法可用性与 `offset_` 规则（1.9 新增句） | L170–L172 |
| 2.7.4 `novelty=True` 用法与重复警告 | L187–L201 |

## 关联页面

- 概念：[[anomaly-detection]]、[[covariance-estimation]]、[[nearest-neighbors]]、[[ensembles]]、[[support-vector-machines]]、[[regularization-and-sparsity]]
- 实践：[[outlier-detection-selection-and-tuning]]、[[covariance-estimation-selection-and-tuning]]、[[nearest-neighbors-selection-and-tuning]]、[[ensembles-selection-and-tuning]]、[[svm-selection-and-tuning]]
- 工具：[[scikit-learn]]
