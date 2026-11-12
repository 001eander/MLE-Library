---
title: "协方差估计（Covariance Estimation）：Σ 与 K 的四种估计器、三种损失口径与 Mahalanobis 作为共同货币"
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - covariance
  - shrinkage
  - precision-matrix
  - robust-statistics
  - estimation
  - anomaly-detection
sources:
  - "[[sklearn-covariance]]"
  - "[[sklearn-outlier-detection]]"
status: draft
---

# 协方差估计（Covariance Estimation）

本页不是「又一个模型族」，而是本库第一个**把估计量本身当研究对象**的类目：不预测 $y$、不分组、不降维，只估计数据的二阶结构 $\Sigma$，以及它的逆 $K=\Sigma^{-1}$（精度矩阵）。前十二个 sklearn 来源都在问「参数怎么拟合」，这一页问的是「**这个矩阵怎么估、估错会怎样、用什么损失衡量**」。

## 一页速查

| 估计器 | 结构假设 | 强度参数 | 能否产生 0 | 对离群点 | 需要 $n>p$ |
| --- | --- | --- | --- | --- | --- |
| `EmpiricalCovariance` | 无 | 无 | 否 | **极敏感** | 否（但会奇异） |
| `ShrunkCovariance` | 向 $\frac{\mathrm{Tr}}{p}I$ 收缩 | `shrinkage`（0–1，默认 0.1） | 否 | 敏感 | 否 |
| `LedoitWolf` | 同上，α 由数据定 | 无（α̂ 解析） | 否 | 敏感 | 否 |
| `OAS` | 同上 + 高斯假设 | 无（α̂ 解析） | 否 | 敏感 | 否 |
| `GraphicalLasso` | $K$ 稀疏（L1） | `alpha`（默认 0.01） | **是（$K$ 的非对角元）** | 敏感 | 否（但易不收敛） |
| `MinCovDet` | 修剪 + 重加权 | `support_fraction`（默认 $\lceil(n+p+1)/2\rceil$） | 否 | **稳健** | **是** |

来源：[[sklearn-covariance]], L16、L28–L32、L79–L83、L124–L128；默认值与 $n>p$ 由**外部核查**（1.9.1 签名 + 1.6.1 实测）补齐。

## 一、为什么协方差是一个特殊的估计对象

三个结构性原因，全部来自来源 L28 那一句话的不同侧面：

1. **它必须半正定**。这是 $\Sigma$ 作为「散布形状」的定义约束。样本协方差在半正定这一点上确实满足，但它**不是满秩**：$n\le p$ 时秩最多 $n-1$。于是「特征值估不准」不是统计效率问题，而是**结构性缺秩**。
2. **下游要的往往是它的逆**。LDA/QDA 的决策面、Mahalanobis 距离、GaussianNB 的似然，用的都是 $K$。而「$\Sigma$ 的特征值小」= 「$K$ 的特征值大」——**求逆把估计误差放大，而不是缩小**（本库综合，来源只给结果「precision matrix … is not accurate」，L28）。
3. **参数个数是 $p(p+1)/2$**。它是本库所有估计对象里增长最快的一个：$p=100$ 时 5,050 个参数。这与 [[gaussian-discriminant-analysis]] 里 QDA 的 $K\cdot d(d+1)/2$ 是同一个计数，只是那里按类再乘一遍。

**由此得到本页的第一条判据（本库综合）**：协方差估计的错误不是「偏差」问题，而是**特征值谱与秩**的问题。所以本节的四种方法都不是「加更多数据」能解决的——它们分别改变**秩**（收缩）、**条件数/结构**（稀疏）与**对个别点的敏感度**（稳健）。

## 二、一个算子、三种 α、两种结构

### 2.1 收缩是凸组合，不是「加噪声」

来源 L32 给的形式是：

$$\Sigma_{\text{shrunk}}=(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$$

三个等价读法（来源自己给的，L32）：压小「最小/最大特征值之比」、给每个特征值加同一偏移、求 $L_2$ 惩罚的 MLE。本机实测逐位复现该式（[[sklearn-covariance]] 实测 4）：$\alpha=0$ 是经验协方差，$\alpha=1$ 是 $\frac{\mathrm{Tr}\hat\Sigma}{p}I$（**非对角全零、对角全部相等**）。

**关键综合（本库）**：`EmpiricalCovariance`、`ShrunkCovariance`、`LedoitWolf`、`OAS` 在实现上是**同一个算子**，差别只在 α 是谁：

| 估计器 | α 的来源 | 目标函数 |
| --- | --- | --- |
| `EmpiricalCovariance` | α ≡ 0 | 无（MLE） |
| `ShrunkCovariance` | 用户给定（默认 0.1，任意值） | 无（α 是超参） |
| `LedoitWolf` | 解析，最小化 $\lVert\hat\Sigma_s-\Sigma\rVert^2$ 的期望 | **$\Sigma$ 的 MSE**（L42） |
| `OAS` | 解析，高斯假设下比 LW 的 MSE 更小 | **$\Sigma$ 的 MSE**（L61） |

本机实测：`LedoitWolf().covariance_` 与 `ShrunkCovariance(shrinkage=LedoitWolf().shrinkage_)` **逐位相同**，OAS 同样（实测 5）。所以选型问题应该重述为：**谁来定 α、定多少、按什么损失定**——而不是「选哪个估计器」。

### 2.2 「optimal」指的是 $\Sigma$ 的 MSE，不是下游指标

这是本节最重要、也最容易被读漏的一句口径。L42 说 LW 的 α「minimizes the Mean Squared Error between the estimated and the real covariance matrix」，L61 说 OAS 在高斯下 MSE 更小。**两者都不是**分类准确率、回归误差或 held-out 似然。

本机实测把它量化为一张表（[[sklearn-covariance]] 实测 10，8 个种子均值，held-out 负对数似然越低越好）：

| 设定 | Empirical | LW | OAS | 手设 α=0.1 | oracle |
| --- | --- | --- | --- | --- | --- |
| $p=5,n=40$（$n/p=8$） | 7.344 | 7.295 | 7.304 | **7.267** | 6.999 |
| $p=10,n=30$（$n/p=3$） | 16.018 | 14.740 | 14.749 | 14.999 | 13.900 |
| $p=20,n=20$（$n/p=1$） | `inf` | **30.608** | **30.608** | 39.776 | 27.801 |

三条读数：

1. **收缩确实有用**，但收益随 $n/p$ 下降而放大：$n/p=8$ 时只值 0.08 个 nats，$n/p=3$ 时值 1.3，$n/p=1$ 时从 `inf` 变成有限值。
2. **数据驱动的 α 不一定赢手设的 α**：第一行里 α=0.1 优于 LW 与 OAS。原因是 LW/OAS 优化的是 $\Sigma$ 的 MSE，而 NLL 对 $\Sigma$ 的**小特征值方向**更敏感——两者不是同一个损失（本库综合）。
3. **「不收缩」在 $n=p$ 时不是「差一点」，而是不可用**（`score` = `inf`）。这解释了 L28 为什么把「不可逆」单列一句。

### 2.3 三个不同的收缩终点（这里有一个真实的文档冲突）

「收缩」这个词在 sklearn 的三个地方指向**三种不同的目标矩阵**：

| 位置 | 终点（α=1 或 ρ=1） | 依据 |
| --- | --- | --- |
| `ShrunkCovariance` / `LedoitWolf` / `OAS` | $\frac{\mathrm{Tr}\hat\Sigma}{p}I$（**标量乘单位阵**） | L32 公式；本机实测 α=1 时对角全等、非对角 0 |
| `LDA(shrinkage=float)` | 同上（`shrunk_covariance`，本机实测**逐位相同**） | 外部核查：`_cov` → `shrunk_covariance(empirical_covariance(X), shrinkage)` |
| `QDA(reg_param=float)` | $I$（**单位阵**） | 外部核查：`S2 = (1-ρ)S2 + ρ` 在 SVD 基上等价于 $(1-\rho)\Sigma+\rho I$ |
| 1.2 节**文档**对 LDA `shrinkage=1` 的描述 | 「the **diagonal matrix of variances**」（**对角阵**） | [[sklearn-lda-qda]], L99 |

**分歧（保留，不覆盖）**：1.2 节用户指南说 `shrinkage=1` 用「方差对角阵」；而 1.6.1 的 `LinearDiscriminantAnalysis(solver='lsqr'|'eigen', shrinkage=1.0, store_covariance=True)` 实测 `covariance_ = 133.3276\,I`，而经验对角是 `[1.06, 24.25, 383.12]`——**对角元全部相等，是标量乘单位阵，不是方差对角阵**。`shrinkage=0.5` 与 `shrunk_covariance(pooled, 0.5)` 逐位相同（maxdiff 0.0），进一步确认走的是本节 §2.6.2 的公式。因此：**文档那句「对角阵」在 1.6.1 上不成立**；它可能是旧实现的描述，也可能只是作者的简化。本库保留两种说法并标注版本，见 [[regularization-and-sparsity]] 的「第二个正则化面」与 [[gaussian-discriminant-analysis]]。

**这条分歧的实际后果**：三个终点的**尺度行为不同**。$\frac{\mathrm{Tr}}{p}I$ 依赖特征尺度，$I$ 不依赖，对角阵也不依赖跨特征的尺度。所以「LDA 加收缩后是否还尺度不变」取决于 `shrinkage` 的取法（见 2.4）。

### 2.4 等变性：`'auto'` 等变，手设 float 不等变

[[gaussian-discriminant-analysis]] 曾把「收缩终点不是仿射等变的」列为**待验证假设**。本机实测给出了明确答案（[[sklearn-covariance]] 外核实测）：

| 配置 | 对特征缩放 $X\mapsto XD$ 的等变性 | 证据 |
| --- | --- | --- |
| `LDA(shrinkage='auto')` | **等变**（decision function 差 8.9e-15） | 源码 `_cov`：先 `StandardScaler`，再 `ledoit_wolf`，再按 `scale_` 还原 |
| `LDA(shrinkage=0.5)` | **不等变**（decision function 差 7.22） | 目标 $\frac{\mathrm{Tr}}{p}I$ 随尺度变化；实测把一列×20 后，反变换回来的对角是 65.8 vs 0.57 |
| `LedoitWolf` 单独用 | **不等变**（α̂ 从 0.0814 变 0.0303） | 同上 |
| `QDA(reg_param=0.5)` | 不等变 | 目标 $I$ |

**结论（本库综合）**：`shrinkage='auto'` 之所以等变，不是因为它用了 LW，而是因为它在**标准化后的空间**里用 LW——标准化的存在把尺度信息消掉了。要「尺度无关的收缩」，正确做法是**先在相关阵上估，再按标准差还原**；`ShrunkCovariance`/`LedoitWolf`/`OAS` 本身都没有这个步骤。这也是 L94「用相关阵比协方差阵更容易恢复结构」的同一个道理。

## 三、从 Σ 到 K：精度矩阵与 covariance selection

### 3.1 $K$ 是图，不是矩阵

L79 的一句话把整节的重心挪了位置：

> The matrix inverse of the covariance matrix, often called the precision matrix, is proportional to the partial correlation matrix. It gives the partial independence relationship. In other words, if two features are independent conditionally on the others, the corresponding coefficient in the precision matrix will be zero.

于是 **$K_{ij}=0$ ⇔ 给定其余变量时 $x_i\perp x_j$**，而「让 $K$ 稀疏」＝「学条件独立图」＝*covariance selection*（L79）。这是本库第一次出现「模型参数就是一张图」的对象：

| 无监督产物 | 输出 | 页 |
| --- | --- | --- |
| 删列 | 列掩码 | [[feature-selection]] |
| 换坐标 | 一套新坐标 | [[manifold-learning]] |
| 分组 | 每个样本一个标签 | [[clustering]] |
| 分解 | 编码 + 成分 | [[matrix-factorization]] |
| **条件独立图** | **变量之间的边** | 本页 |

**关键区分（本库综合）**：**$\Sigma$ 稀疏与 $K$ 稀疏是两件相反的事**。$\Sigma_{ij}=0$ 是「边际独立」（两个变量直接无关），而 $K_{ij}=0$ 是「条件独立」（去掉别人之后无关）。一般的稀疏 $\Sigma$（如 FA 的低秩 + 对角噪声）对应一个**稠密**的 $K$。因此「稀疏协方差」与「稀疏精度矩阵」不能混用同一个名词。

### 3.2 稀疏 vs 收缩：来源的三句话与实测

L81 给了三条判据：

1. $n$ 与 $p$ 同量级或更小时，稀疏估计器**往往比收缩好**；
2. 反过来（$n\gg p$）或高度相关数据，稀疏估计器**数值不稳定**；
3. 收缩**恢复不了非对角结构**，稀疏可以。

本机实测把三条拆开（合成 $p=20$、真值 12 条边，[[sklearn-covariance]] 实测 14–19）：

| 设定 | 经验 | LW | GLasso（手调 α） | GLassoCV |
| --- | --- | --- | --- | --- |
| $n=60$（$n/p=3$） | 0.1096 | 0.0559 | **0.0217**（α=0.4，F1 0.857） | 0.0214（F1 0.522） |
| $n=200$（$n/p=10$） | 0.0625 | 0.0470 | 0.0186（F1 1.000） | **0.0134**（F1 0.375） |
| $n=2000$（$n/p=100$） | 0.0205 | 0.0196 | 0.0096（α=0.02） | 0.0055 |

（表中为协方差平均绝对误差。）

- 判据 1 **成立且幅度大**（$n/p\ge3$ 时 GLasso 好 2–3 倍），在 $n=p$ 时只是持平（0.0384 vs 0.0357）。
- 判据 2 **未复现**：$n=2000$ 时只要 α≥0.02 就没有任何收敛警告，F1 反而最高。真正的不稳定条件是**近共线列 + 极小 α**（$\alpha=0.001$ 触发 125 条 `ConvergenceWarning`，α=0.01 降到 3 条，α=0.1 为 0）。
- 判据 3 **成立**，且图 3 是它的可视化版本：LW 的协方差几乎只剩对角（非对角被 $\alpha$ 压扁），GLasso 的精度矩阵仍有稀疏的蓝/白非对角点。
- **`GraphicalLassoCV` 的 α 系统性偏小**：$n=60$ 时 34 条边（真值 12，F1 0.522）、$n=200$ 时 52 条边（F1 0.375）。但它的**协方差误差却最小**（0.0134 vs 手调 0.0186）。→ **「边数正确」与「$\Sigma$ 误差小」是两个不同的目标**，CV 优化的是后者（似然），不是前者。

### 3.3 标准化是结构恢复的硬前提

L94 的第一条是「Recovery is easier from a correlation matrix than a covariance matrix: standardize your observations before running `GraphicalLasso`」。本机实测（实测 17）：$n=60$、$\alpha=0.2$ 时，尺度相近的数据 F1 从 0.286（协方差）升到 0.308（相关阵）；但**把一列乘 1000** 后，原始尺度 F1 掉到 **0.207**（104 条边），标准化后回到 **0.308**（66 条边）。

**机制（本库综合）**：$\alpha$ 是一个**绝对**阈值，作用在 $K$ 的元素上；而 $\Sigma$ 的元素带着量纲。量纲差异大时，同一个 $\alpha$ 对不同变量施加的有效惩罚不同。标准化把 $\Sigma$ 换成相关阵 $R$，$\alpha$ 才有统一含义。这与 [[nearest-neighbors]] 的「换缩放 = 换模型」同构——只是这里换的是**惩罚的尺度**。

### 3.4 目标函数与惩罚范围

L101–L103：

$$\hat K=\arg\min_K\ \mathrm{tr}SK-\log\det K+\alpha\|K\|_1,\qquad \|K\|_1=\sum_{i\ne j}|K_{ij}|$$

L105 明确 $\|K\|_1$ 只算**非对角**。外部核查源码 `_objective` 一致（`abs(K).sum() - abs(diag(K)).sum()`）。**这是本库第一处「L1 作用在矩阵上但排除对角」的例子**：Lasso 罚系数、SparsePCA 罚成分、这里罚「边」。对角的 $K_{ii}=1/\sigma^2_{i\mid\cdot}$ 由 $\log\det$ 项唯一确定，罚它会与约束冲突。

`alpha=0` 是合法的，且等价于经验协方差（实测 18：maxdiff 0.0）——所以「稀疏」可以从「关」连续调到「全零」，不需要换类。

## 四、稳健性：Mahalanobis 是共同的货币

L118 的论证结构值得单独抄下来：

1. 经验与收缩估计器**对离群点极敏感**；
2. 所以要改用稳健估计器；
3. 或者反过来，用稳健估计器**做离群点检测**（按 Mahalanobis 距离丢/降权）。

第 3 条是本库第一次出现「估计器同时是诊断工具」。而它们共享的货币就是 **Mahalanobis 距离** $d_i^2=(x_i-\hat\mu)^\top\hat K(x_i-\hat\mu)$：MCD 用它做重加权（阈值 $\chi^2_p(0.975)$），`EllipticEnvelope` 用它做判定，[[nearest-neighbors]] 的 NCA 学的是同一个 $M$，[[gaussian-discriminant-analysis]] 的线性决策面也是它。

### `EllipticEnvelope` = MCD + 一个阈值（2.7 节已入库）

2.7 节把上面的第 3 条写成了一个估计器：`EllipticEnvelope` 就是「稳健估一个椭圆，再用 Mahalanobis 距离当异常度」（[[sklearn-outlier-detection]], L104–L110）。因此本页关于 MCD 的三条警告**全部继承**：`support_` 不是 h、校正使 det 变 9.7 倍、$n\le p$ 报内部下标错误。

它只新增了两样东西：

1. **一个阈值层**：`contamination=0.1`（**默认是 float，不接受 `'auto'`**；实测 `0.0` 与 `'auto'` 都抛 `InvalidParameterError`）→ `offset_`，使训练集上恰好出现 $c\cdot n$ 个异常（实测 55/550、28/550、165/550，[[sklearn-outlier-detection]]「本机实测」14）。
2. **一套「分数 → `offset_` → ±1」的公共契约**：`decision_function = score_samples − offset_`（实测 maxdiff 0.0），与 `IsolationForest`/`OneClassSVM` 同一个接口。

**本库综合**：这使本页成为异常检测的「基础设施页」——四种形状假设里，高斯椭圆这一支完全由 $\Sigma$、$\hat K$ 与 MCD 的稳健性决定。反过来，异常检测也把本页的一个抽象问题变成了可测的：「协方差估得好不好」在异常检测里的后果是**能否兑现一个指定的污染率**，而不是 frobenius 误差。

### MCD 三步与两个「h」

L124 的三步是：① 取 $h$ 个「好」点算经验协方差；② 重标定（*consistency step*）；③ 按 Mahalanobis 距离重加权（*reweighting step*）。L126 补上 FastMCD，并说明它**同时**估位置。

本机实测（$n=200,p=3$、10% 污染，[[sklearn-covariance]] 实测 20–21）：

| 量 | 值 |
| --- | --- |
| $h=\lceil(n+p+1)/2\rceil$ | 102（= `raw_support_.sum()`） |
| `support_.sum()` | **178**（重加权后的掩码，**不等于 h**） |
| $\det$：校正前 → 校正后 | 0.0762 → 0.7365（≈9.7×） |
| Frobenius 误差（对 $\Sigma=I$） | MCD **0.313** vs 经验 **9.394** |
| Mahalanobis：污染点 / 内点 | MCD **124.34 / 3.07**；经验 10.33 / 2.19 |

**两个坑（本库综合 + 外部核查）**：

- **`support_` 不是 h**。它被 `reweight_covariance` 覆盖成重加权掩码（178），h 只在 `raw_support_` 里。官方用户指南不提这一点，docstring 也不提。
- **`support_fraction` 的默认值公式在 docstring 里写错了**（写成 `(n+p+1)/2 * n_samples`，源码是 `ceil(0.5*(n+p+1))`）。引默认值时以源码为准。
- **校正不是微调**：det 差近 10 倍，跨估计器比较时必须说明用的是 `raw_covariance_` 还是 `covariance_`。

## 五、和本库其他页的关系

| 页面 | 这里提供什么 | 那边提供什么 |
| --- | --- | --- |
| [[gaussian-discriminant-analysis]] | `shrinkage` / `covariance_estimator` / `reg_param` 的**实体**与三种终点 | 生成式模型的假设嵌套、Mahalanobis 决策面 |
| [[regularization-and-sparsity]] | 收缩的精确公式、$\frac{\mathrm{Tr}}{p}I$ 终点、$K$ 上的 L1 | 「第几个正则化面」的定位与横向对照表 |
| [[naive-bayes]] | 对角极限 $\Sigma=\mathrm{diag}$ 的估计层 | `var_smoothing` 是**方差下限**，不是收缩 |
| [[nearest-neighbors]] | Mahalanobis 度量的估计版本（NCA 学 $M$） | 零参数模型、kNN 图 |
| [[clustering]] | KMeans＝等协方差 GMM 的 $\Sigma$ 退化、GMM 的 $\Sigma_k$ | 簇的几何假设、$k$ 的选择 |
| [[matrix-factorization]] | FA 的 $\Sigma=WW^\top+\Psi$ 是**结构化协方差**的另一条路 | 成分、编码、可辨识性 |
| [[linear-models-as-baselines]] | 「小样本下不收缩直接失效」的量化版本 | 何时把 LDA/QDA 当基线 |
| [[anomaly-detection]] | $\Sigma$ 与 $\hat K$ 作为「高斯椭圆」假设的全部要素 | 四种形状假设的对照、阈值的四种语义 |

## 六、失效模式清单

1. **在未居中的数据上用 `assume_centered=True`**：对角多算 $\bar x^2$（实测 1.348 → 2.550），`location_` 静默置零。这是本节最贵的错误，因为不报错。
2. **以为「`precision_` 一定存在」**：`store_precision=False` 时属性存在但值为 `None`；`get_precision()` 才算。
3. **以为奇异协方差会报错**：实现用 `pinvh` 静默给伪逆，`score` 才返回 `-inf`。
4. **把 `shrinkage='auto'` 的 α 当作「分类最优」**：它优化的是 $\Sigma$ 的 MSE，实测在 $n/p=8$ 时输给手设 α=0.1。
5. **把三个估计器当成三个不同算法**：它们是同一个算子 + 三种 α 来源，`ShrunkCovariance` 可以直接复现 LW/OAS。
6. **忘了 $n=2$ 时 LW 的 α 恒为 0**：样本最少时收缩恰好关闭（代数恒等式）。
7. **把 `MinCovDet.support_` 当作 h**：它是重加权掩码；h 在 `raw_support_`。
8. **比较 `raw_covariance_` 与 `covariance_` 的数字**：校正让 det 变 9.7 倍。
9. **在未标准化的数据上跑 `GraphicalLasso`**：量纲差 1000 倍时 F1 从 0.308 掉到 0.207。
10. **忽略 `GraphicalLasso` 的 `ConvergenceWarning`**：近共线 + 小 α 会刷出上百条，结果不可信；它同时是「α 太小」的诊断信号。
11. **把 CV 选的边数当结构恢复结果**：CV 目标与结构恢复目标不同，实测边数多 2.8–6.8 倍，但协方差误差更小。
12. **对 LDA 的 `shrinkage=1` 期待「只保留方差」**：1.6.1 实测是标量乘单位阵（对角全等），与 1.2 节文档描述冲突。
13. **忘了收缩不等变**：手设 `shrinkage` 会让结果随特征缩放变化；只有 `'auto'`（内部标准化）等变。
14. **在 $n\le p$ 上直接用 `EmpiricalCovariance` 做下游**：$p=20,n=20$ 的 held-out NLL 实测为 `inf`。
15. **把 `EllipticEnvelope` 当「和 IF/LOF 一样可调 `contamination='auto'`」**：它只接受 float，且默认是 0.1（[[sklearn-outlier-detection]] 实测 14）。
16. **以为稳健估计器自动等于「更好的异常检测器」**：MCD 的椭圆会拉长到包住多峰数据；实测在多峰数据上 EE 把两团用一长椭圆串起来（[[sklearn-outlier-detection]] 图 1 第二行）。

## 待解问题

1. **LW/OAS 的解析式**：剪藏只给口径不给公式（文献已丢）。若要用「α̂ 的可解释性」做诊断，必须先补这两篇。
2. **MSE 与 NLL 的差距能不能预测**？实测第一行 α 的 MSE 最优 ≠ NLL 最优，但差距随 $n/p$ 变化的规律未知。
3. **相关阵收缩 vs 协方差阵收缩**：LDA 的 `'auto'` 已经在做「先标准化再收缩」，`sklearn.covariance` 的四个估计器都没有。这是 API 层的不一致，还是有意分工？
4. **收缩与稀疏的组合**：能否先 `MinCovDet` 再 `GraphicalLasso`？两者都要求同一份数据，但稳健化的样本加权会不会破坏 L1 的尺度假设？本库未验证。
5. **$p\gg n$ 时 GLasso 的可解释性**：本次只测到 $p=20$，更高维未验证。
6. **`EllipticEnvelope` 与 `MinCovDet` 的关系**：已部分回答（2.7 节入库后）——EE = MCD + `contamination` 定的 `offset_`，阈值使训练集恰好 $c\cdot n$ 个异常；但 **`support_fraction`（选哪些点）与 `contamination`（报多少个）的交互未测**，两者可能在同数据上给出矛盾的支持集。
7. **`EmpiricalCovariance.score` 的常数项**：跨实现比较 NLL 的绝对量级前必须核对。

## 关联页面

- 来源：[[sklearn-covariance]]、[[sklearn-lda-qda]]、[[sklearn-outlier-detection]]（`EllipticEnvelope` 的下游用法）
- 概念：[[regularization-and-sparsity]]（第二个正则化面）、[[gaussian-discriminant-analysis]]、[[naive-bayes]]、[[nearest-neighbors]]、[[clustering]]、[[matrix-factorization]]、[[anomaly-detection]]
- 实践：[[covariance-estimation-selection-and-tuning]]、[[outlier-detection-selection-and-tuning]]、[[linear-models-as-baselines]]
- 工具：[[scikit-learn]]
