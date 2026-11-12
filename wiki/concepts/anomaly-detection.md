---
title: 异常检测（Anomaly / Novelty / Outlier Detection）
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - one-class-svm
  - isolation-forest
  - local-outlier-factor
  - robust-statistics
  - unsupervised
sources:
  - "[[sklearn-outlier-detection]]"
  - "[[sklearn-covariance]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-svm]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档论述与接口，见 [[sklearn-outlier-detection]]）
  - scikit-learn 1.6.1（本页标「实测」的运行时行为）
---

# 异常检测（Anomaly / Novelty / Outlier Detection）

## 一句话定义与它的三个推论

**异常检测 = 学一个「正常」的判据，然后给每条样本一个「有多不正常」的分数，再用一个阈值把它变成 ±1。**

这句话里有三个名额，正好对应三个必须显式回答的问题：

1. **「正常」由什么假设定义**（形状假设：高斯椭圆 / 支持边界 / 隔离路径 / 局部密度比）；
2. **分数怎么来**（`score_samples`，单位与量纲各不相同，**跨数据集不可比**）；
3. **阈值怎么定**（`offset_`，由 `contamination` 或 `nu` 决定，**这是唯一真正可调的「模型参数」**）。

推论：

- **它是唯一「没有 $y$、也没有簇标签」的监督式问题**：输出是 `±1`（不是 0/1），且第 2、3 步可以分开调——换阈值不重训，换形状假设要重训。[[sklearn-outlier-detection]] L34、L36 把这三段写成了契约。
- **它的「正确答案」依赖训练集是否被污染**，而这在现实中几乎不可知（见下一节）。
- **它天然缺少评估**：分数没有真值可比，`sklearn.metrics` 里也没有异常检测指标（官方示例只能靠已知真值算 ROC，见 [[sklearn-outlier-detection]] L71–L72）。

## 两类问题的断层（outlier vs novelty）

| 维度 | outlier detection（无监督异常检测） | novelty detection（半监督异常检测） |
| --- | --- | --- |
| 训练集 | **已被污染**，要拟合最密集的区域 | **干净**，代表「正常」总体 |
| 判据 | 远离核心区 | 落在训练分布之外 |
| 异常能否成簇 | **不能**（估计器假设异常在低密度区） | **能**，只要它在训练分布的低密度区 |
| API 形态 | 常常只有 `fit_predict`（传导式） | 必须有 `predict`/`decision_function`（归纳式） |
| 典型对象 | `LocalOutlierFactor(novelty=False)`、`EllipticEnvelope` | `OneClassSVM`、`LocalOutlierFactor(novelty=True)` |

来源论述：[[sklearn-outlier-detection]] L12–L20（两张定义 + 「不能成簇」的硬约束）。

**综合（本库）**：这条「能否成簇」的约束被实测检验过，而且**只对密度比类成立**。在「1,000 个内点 + 40 个 σ=0.06 的密集远簇」上，`LocalOutlierFactor(20)` 对该簇的召回是 **0.03**，`OneClassSVM` 是 0.30，而 `IsolationForest` 与 `EllipticEnvelope` 都是 **1.00**（[[sklearn-outlier-detection]], L20 与「本机实测」15）。原因是：LOF 比较的是**局部**密度，密集簇内部的密度比自身邻居还高，于是每个点都「正常」；而 IF 比较的是**隔离难度**（整体上仍在数据范围之外）、EE 比较的是**全局椭圆距离**（远在椭圆外）。

因此这张表在实际使用中要改写成：

- 异常**可能成簇** → 只能用全局型估计器（IF、EE、OCSVM）；
- 异常**是孤立的局部离群点** → LOF 最敏感（同一实验里 LOF 把「距密簇 8σ、全局很近」的点排到第 2/802，而 EE 排最后，见下节）。

## 四种形状假设（一族一个几何对象）

| 假设 | 「正常」的形状 | 估计对象 | 分数 | 对象 | 主导超参 |
| --- | --- | --- | --- | --- | --- |
| 高斯椭圆 | 一个（稳健估计的）椭圆体 | $\mu$、$\Sigma$（MCD） | Mahalanobis 距离 → 对数似然差 | `EllipticEnvelope` | `contamination`、`support_fraction` |
| 支持边界 | 包住训练点的**边界**（可非线性） | 支持向量与 $\nu$ 分位 | 到边界的有符号距离 | `OneClassSVM`、`SGDOneClassSVM` | `nu`、`gamma`、`kernel` |
| 隔离路径 | 无显式形状；「被切开有多容易」 | 随机轴对齐树森林 | 平均路径长度 → $2^{-E(h)/c(n)}$ | `IsolationForest` | `max_samples`、`n_estimators`、`max_features` |
| 局部密度比 | 每个点自己的邻域 | kNN 图 | 邻居平均密度 / 自身密度 | `LocalOutlierFactor` | `n_neighbors`、`metric` |

四个假设的**失效方向完全不同**，这是选型的核心：

- **椭圆**假设单峰、凸、且协方差有意义。多峰数据（两团）会被一个大椭圆包住，中间的空洞被判为正常（图 1 第二行的 Robust covariance 面板就是这个形状——它把两团用一条长椭圆串起来）。$p>n$ 时协方差不可逆（见 [[covariance-estimation]] 的 `pinvh` 伪逆问题）。
- **支持边界**假设存在一个「紧」的边界，代价是 $\nu$ 与 $\gamma$ 耦合：实测固定 `gamma=0.5` 时，单列 ×100 会让 **51.5%** 的标签翻转；`gamma='scale'` 降到 11%（[[sklearn-outlier-detection]]「本机实测」11）。
- **隔离路径**假设异常「少且不同」，因此**在无结构数据上会沿数据范围画一个矩形边界**（图 1 第五行），而且它**完全尺度不变**（整体 ×1000 与单列 ×100 的分数 maxdiff 都是 0.0，实测 5）。
- **局部密度比**假设异常是「相对邻域的稀疏」，因此**看不见密集异常簇**，也**看不见全局异常若不是局部稀疏**。它对距离度量敏感（`cosine` 在同一数据上报 8 个而非 24 个，实测 23）。

## 三段式 API：分数 → 阈值 → 标签

契约（[[sklearn-outlier-detection]] L34、L36、L172）：

```text
score_samples(X)          # 原始分数，越大越正常（IF/LOF 取负号后的约定）
offset_                   # 阈值，拟合期确定
decision_function(X) = score_samples(X) - offset_    # 负 = 异常
predict(X) = -1 if decision_function(X) < 0 else 1   # 非负 = 正常
```

本机实测四类估计器（IF / LOF(novelty) / OCSVM / EE）都严格满足前两行，maxdiff 均为 **0.0**；第三行在含 0 的情况下也成立（0 归为正常）——与 L36 的「non-negative ones are inliers」一致（实测 4）。

**`offset_` 才是模型里唯一「可调而不重训」的量**，而它的来源有**四种互不相同的语义**（这是本页最重要的表）：

| 估计器 | `contamination='auto'` 时 | 给定 `contamination` 时 | 备注 |
| --- | --- | --- | --- |
| `IsolationForest` | **常数 −0.5**（写死） | 训练集分数的对应百分位 | `'auto'` 下异常率随数据变：高斯数据 13.9–15.4%（实测 6） |
| `LocalOutlierFactor` | **常数 −1.5**（写死） | `negative_outlier_factor_` 的百分位 | `'auto'` 下实测只报 0.2–1.4%（实测 22） |
| `EllipticEnvelope` | **不支持 `'auto'`**，默认 `0.1` | 使训练集出现 $c\cdot n$ 个异常 | `0.0` 与 `'auto'` 都抛 `InvalidParameterError`（实测 14） |
| `OneClassSVM` / `SGDOneClassSVM` | **无此参数** | — | 用 `nu` 同时控制边界与训练集异常比例 |

**推论（工程上最容易踩的一脚）**：在同一份数据上跑五个估计器、比较「谁报了多少个异常」**没有意义**——它们的默认阈值口径不同。要比较必须先统一口径：或统一 `contamination`，或统一用 `score_samples` 算 AUC/排序指标。

`contamination` 给定时，IF 与 EE 的「预期数量」是**精确**的（1,000 样本上 0.01/0.05/0.1/0.2 → 10/50/100/200；1,100 样本上 0.05/0.1/0.2/0.5 → 55/110/220/550），而 LOF 的 `predict` 与 `negative_outlier_factor_` 口径**故意不一致**（$c$=0.1 时 32 vs 33，$c$=0.2 时 61 vs 66），因为 `predict` 重新计算分数、自身不在邻域里（实测 3、9）。

## 归纳 vs 传导：一条硬断层

| 能力 | `IsolationForest` | `EllipticEnvelope` | `OneClassSVM` | `LocalOutlierFactor` |
| --- | --- | --- | --- | --- |
| 给训练样本打分 | `score_samples` | `score_samples` | `score_samples` | `negative_outlier_factor_` |
| 给新样本打分 | `decision_function` | `decision_function` | `decision_function` | **仅 `novelty=True`** |
| 训练集上的标签 | `predict` | `predict` | `predict` | **仅 `fit_predict`** |
| 能否上线（一次拟合、多次推理） | 能 | 能 | 能 | **默认不能** |

来源论述：[[sklearn-outlier-detection]] L42–L59 的 5×3 表。**本库综合**：这张表的现实含义不是「LOF 少三个方法」，而是「LOF 默认是**传导式**（transductive）估计器」，它的分数定义依赖训练集本身的邻域结构；切成归纳模式后，它计算的其实是**另一个统计量**——同一批训练点上，`novelty=True` 的 `score_samples` 与 `negative_outlier_factor_` 最大差 **0.586**（秩相关仍 0.9967），`predict` 与 `fit_predict` 有 1.2% 的标签不同（实测 8）。

因此「训练集分数」与「推理分数」必须**分开缓存、分别解释**，不能互相校验。

## 局部 vs 全局：两种不可互换的读法

来源论述：[[sklearn-outlier-detection]] L168「问题不是样本多孤立，而是相对周围邻域多孤立」。

**实测证据**（密簇 σ=0.1 在原点、疏簇 σ=2 在 $x$=20、外加一个「距密簇 8σ 但全局尺度上很近」的点）：

| 估计器 | 该点排名（1 = 最异常，共 802） | 是否报出 |
| --- | --- | --- |
| `LocalOutlierFactor(20)` | **2** | 是（df −6.008） |
| `LocalOutlierFactor(50)` | 2 | 是 |
| `IsolationForest` | 53 | 是（较弱） |
| `OneClassSVM(nu=0.005)` | 405 | **否** |
| `EllipticEnvelope(0.005)` | **802（最不异常）** | **否** |

这张表解释了「为什么需要 LOF」：全局型估计器被疏簇的尺度主导，密簇旁边的局部离群点**完全看不见**。反过来，LOF 看不见成簇的异常（上节 0.03）。

**待解问题**：两种读法能否用「多尺度 LOF」（多个 $k$ 的分数取最大）合并？本库未测。

## 与其他类目的定位

| 类目 | 有 $y$ | 输出 | 假设 | 本库页面 |
| --- | --- | --- | --- | --- |
| 分类/回归 | 有 | 标签/数值 | $p(y\mid x)$ | [[linear-models-as-baselines]] 等 |
| 聚类 | 无 | 簇标签（可多类） | 簇结构（凸/密度/图割） | [[clustering]] |
| 降维/分解 | 无 | 新坐标/成分 | 低维结构 | [[manifold-learning]]、[[matrix-factorization]] |
| **异常检测** | 无 | **±1 + 分数** | **「正常」的形状** | 本页 |
| 协方差估计 | 无 | $\Sigma$、$K$ | 二阶结构 | [[covariance-estimation]] |

三个关键差别：

- **与聚类的差别是「类数」**：聚类假设数据里有几个组，异常检测假设数据里有**一个**「正常」的组（多峰时要靠单峰假设硬包）。因此聚类的结果（簇标签）可以喂给异常检测当特征，但两者不能互相替代。
- **与协方差估计的差别是「有没有下游」**：`EllipticEnvelope` 就是「MCD + 一个阈值」，所以 2.6 节关于 `support_`、`raw_covariance_`、$n>p$ 的警告全部继承（见 [[covariance-estimation]]）。
- **与集成的关系是「同一骨架、不同聚合」**：`IsolationForest` 用 bagging 骨架，但聚合的不是预测而是**路径长度**，基学习器是 `ExtraTreeRegressor`（随机阈值），深度被写死为 $\lceil\log_2 n\rceil$（[[sklearn-outlier-detection]] L133）。所以它既不是随机森林（不投票）也不是 GBDT（不纠错），见 [[ensembles]]。

## 预处理：四种尺度行为

| 估计器 | 整体等比缩放 | 单列缩放 | 距离度量 |
| --- | --- | --- | --- |
| `IsolationForest` | 不变 | **不变**（标签一致率 1.000） | 无（只比较同列内的相对大小） |
| `OneClassSVM(gamma='scale')` | 不变 | **变**（一致率 0.89） | 核 |
| `OneClassSVM(gamma=0.5)` | 不变 | **变**（一致率 0.485） | 核 |
| `LocalOutlierFactor` | 变 | 变 | `metric`/`p`（`cosine` 报 8 vs `euclidean` 24） |
| `EllipticEnvelope` | **不变** | **不变**（实测秩相关 1.0） | Mahalanobis（自估协方差 → 仿射等变） |

**综合（本库）**：尺度不变性只有 IF 有；EE 的 Mahalanobis 是**尺度等变**（它自己估协方差，所以线性变换不改变排序）；OCSVM 与 LOF 都需要人工缩放。这条与 [[nearest-neighbors]] 的「距离是超参」是同一件事，但异常检测里更危险——因为**没有标签能告诉你缩放错了**。

## 评估：一个缺席的环节

来源论述：本节 **6 张图、10 条示例链接、0 个指标**；两个官方示例（`plot_anomaly_comparison`、`plot_outlier_detection_bench`）都用**已知真值**画 ROC（[[sklearn-outlier-detection]] L71–L72）。

**综合（本库）**：可用三条替代协议，按可靠性排序——

1. **有真值时**：用 `decision_function` 算 ROC-AUC / PR-AUC（阈值无关），别用 `predict` 的准确率（`±1` 且受 `contamination` 支配）。实测同一份污染训练集上，AUC 的排序是 EE 0.9872 ≈ IF 0.9866 > LOF(35) 0.9822 > OCSVM(0.5) 0.9809 > … > OCSVM(0.1) 0.9610（[[sklearn-outlier-detection]] 实测 12）。
2. **无真值时**：看**稳定性**（换种子/换子样本，前 $k$ 个异常的重合率）与**可解释性**（被判异常的点是否有业务解释）。
3. **上线前**：用历史标注或人工抽检回测**固定阈值下的精度**——因为 `contamination` 一旦入库就是一句「我承诺污染率是 $c$」的断言。

## 失效模式清单

1. **把 `'auto'` 当「自动定污染率」**：IF 是常数 −0.5、LOF 是常数 −1.5，两者与 10% 无关（高斯数据上分别是 13.9–15.4% 与 0.2–1.4%）。
2. **在同一数据上比较五个估计器的默认输出**：默认阈值口径不同，`contamination` 甚至有的不存在（OCSVM）。
3. **用 `contamination=0.0` 表达「不报异常」**：非法（区间 `(0.0, 0.5]`），只能改 `offset_`。
4. **在训练集上调用 `novelty=True` 的 `predict`**：结果与 `fit_predict` 不同（1.2% 标签差、分数最大差 0.586），且没有任何警告。
5. **期待 LOF 检出成簇异常**：召回 0.03（密集远簇实验）。
6. **期待全局型估计器检出局部异常**：EE 把「距密簇 8σ」的点排到 802/802，OCSVM 不报。
7. **`n_neighbors` 大于样本数**：静默夹到 $n-1$，结果退化成「全正常」（330 点上 400 → 329、0 个异常）。
8. **`n_neighbors` 太小而污染率高**：训练集 17% 污染时 $k=5$ 的 AUC 是 **0.5183**（随机水平），$k=35$ 才 0.9766。
9. **未缩放就上 OCSVM/LOF**：固定 `gamma` 时单列 ×100 使 51.5% 标签翻转。
10. **用 `EllipticEnvelope` 处理多峰或 $p>n$**：单椭圆串多团；$n\le p$ 时内部 MCD 直接抛 `ValueError: kth(=7) out of bounds`。
11. **把 LOF 的分数当跨数据集可比的「异常度」**：它是比值，量纲依赖邻居密度。
12. **用 `predict` 的准确率当评估**：受阈值支配；应优先用 `decision_function`。
13. **`IsolationForest(warm_start=True)` 改 `max_samples` 后重训**：只有新树用新值，旧树保留旧值，**无警告**。
14. **把 `OneClassSVM` 的 $\nu$ 读成「新样本越界概率」**：实测 $\nu=0.01$ 给出 6.8% 的训练集异常率（训练集统计量，不是泛化概率）。
15. **在离群点存在时用非稳健的协方差当椭圆**：见 [[covariance-estimation]] 的 MCD vs MLE（Frobenius 误差 0.313 vs 9.394）。

## 待解问题

1. IF 的分数式 $2^{-E(h)/c(n)}$ 与 −0.5 阈值的等价关系（剪藏丢了 Liu et al. 2008，只有定性描述）。
2. `offset_` 在并列分数（tie）时的分位数规则；`contamination` 给定时「精确数量」是否只在连续分数下成立。
3. 多尺度 LOF（多个 $k$ 取最大）能否同时兼顾局部与成簇异常。
4. OCSVM 的 $\nu$ 与「新样本越界概率」的精确关系（Schölkopf et al. 2001 的定义与实现的差距）。
5. 无标签时的稳定性指标应该怎么定义（种子间 top-$k$ 重合率？子样本重抽？）。
6. `IsolationForest(max_features<1.0)` 对本库数据的 AUC 影响（剪藏完全不提这个预算旋钮）。
7. 阈值跨数据集迁移：在一个污染率先验上定的 `contamination` 用到另一个先验时，是否只需重算 `offset_`（API 无入口）。

## 关联页面

- 来源：[[sklearn-outlier-detection]]、[[sklearn-covariance]]、[[sklearn-nearest-neighbors]]、[[sklearn-ensembles]]、[[sklearn-svm]]
- 实践：[[outlier-detection-selection-and-tuning]]
- 相邻概念：[[covariance-estimation]]、[[nearest-neighbors]]、[[ensembles]]、[[support-vector-machines]]、[[clustering]]、[[matrix-factorization]]、[[regularization-and-sparsity]]
- 工具：[[scikit-learn]]
