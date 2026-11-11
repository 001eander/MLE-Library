---
title: "高斯判别分析（Gaussian Discriminant Analysis）：Mahalanobis 几何、LDA/QDA/GaussianNB 的嵌套与参数代价"
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - discriminant-analysis
  - generative-model
  - covariance
  - classifier
sources:
  - "[[sklearn-lda-qda]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.x 的 discriminant_analysis 模块（据 1.2 节剪藏判定）
---

# 高斯判别分析

本页整理「**先建模 $p(x\mid y)$，再用贝叶斯得到 $p(y\mid x)$**」这一族分类器：它们的共同结构、决策面的几何、三个成员（`GaussianNB` / `LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis`）的嵌套关系，以及每一层「多出的自由度」要付多少样本代价。事实来自 [[sklearn-lda-qda]]（写作 `[[sklearn-lda-qda]], L67`），凡本库推导或推断都显式标注。

## 与判别式模型的根本差别

| | 生成式（本页） | 判别式（[[sklearn-linear-models]] 那一族） |
| --- | --- | --- |
| 建模对象 | $p(x\mid y=k)$ 与 $P(y=k)$ | 直接建模 $p(y\mid x)$ 或决策函数 |
| 决策面来源 | 两类的密度相等处**自动**产生 | 由损失函数与假设空间**直接优化**得到 |
| 训练方式 | 闭式（均值 + 协方差 + 类比例） | 迭代优化（除 OLS/岭） |
| 超参形态 | 协方差怎么估（收缩/对角/共享） | 正则强度（`alpha`/`C`） |
| 多类 | 天生（每类一套参数） | 需要 multinomial 或 OvR 策略 |

**本库综合**：这个差别的工程含义是——生成式的"超参"作用在**协方差估计**上，判别式的"超参"作用在**系数**上，两者不可互相替代，也不能用同一套调参直觉（见 [[regularization-and-sparsity]] 的「第二个正则化面」）。

## 一个模型，三层嵌套

三者是同一个高斯生成模型的三级约束（[[sklearn-lda-qda]], L60–L67）：

| 成员 | 类协方差假设 | 决策面 | 来源 |
| --- | --- | --- | --- |
| `GaussianNB` | $\Sigma_k$ 为**对角**阵（各类、各特征独立） | 二次（但轴向受限） | L60–L63 |
| `LinearDiscriminantAnalysis` | **所有类共享**同一个 $\Sigma$ | 线性 | L67、L83 |
| `QuadraticDiscriminantAnalysis` | 每类自己的 $\Sigma_k$，无约束 | 二次 | L52–L58 |

约束越少、自由度越多，但**每一层多出的自由度都要用样本去买**。下图是官方示例的读数（**本库目视核查** `raw/assets/sklearn-lda-qda/sphx_glr_plot_lda_qda_001.png`）：第 1 行（球状等协方差）与第 2 行（等协方差、非球状）里 QDA 的边界与 LDA 几乎重合；只有第 3 行协方差不同时，QDA 才出现真正不同的双曲线边界。

**本库判断**：这是"更灵活的模型不一定更好"的最干净的反例——**当 $\Sigma_k$ 真的相等时，QDA 学到的二次项只是噪声**。

## 几何：Mahalanobis 距离与线性决策面

LDA 的 log 后验（[[sklearn-lda-qda]], L69–L71）：

$$\log P(y=k\mid x) = -\frac{1}{2}(x-\mu_k)^{T}\Sigma^{-1}(x-\mu_k) + \log P(y=k) + Cst$$

来源点明：那个二次型就是 $x$ 与 $\mu_k$ 之间的 **Mahalanobis 距离**的平方，因此 LDA 可读作「把 $x$ 判给 Mahalanobis 意义下最近的类均值，同时兼顾先验」（L73）。

注意 $\log|\Sigma|$ 项在 LDA 里**消失了**（所有类共享 $\Sigma$，与 $k$ 无关，被吸进常数），这正是决策面从二次降为线性的代数原因；而 QDA 保留 $-\frac12\log|\Sigma_k|$（L55），所以每个类有一个"体积惩罚"。

仿射形式与实现属性的对应（L75–L81）：

$$\omega_k = \Sigma^{-1}\mu_k,\qquad \omega_{k0} = -\frac{1}{2}\mu_k^{T}\Sigma^{-1}\mu_k + \log P(y=k)$$

$\omega_k$ 就是 `coef_`，$\omega_{k0}$ 就是 `intercept_`。**本库综合**：这也解释了为什么 LDA 的 `coef_` 可以直接当线性模型读，但**不能当"特征重要性"读**——它的方向由 $\Sigma^{-1}$ 旋转过，是"判别方向"而不是"预测方向"。

## 降维：白化 + 投影到类均值张成的子空间

LDA 同时是一个降维器，其几何链条（[[sklearn-lda-qda]], L87–L93）：

1. $K$ 个类均值落在**至多 K−1 维**的仿射子空间里（两点成线、三点成面）；
2. LDA 等价于先把数据 sphering（协方差变成单位阵），再取欧氏最近均值 + 先验（L89）；
3. 白化空间里，与 $H$ 正交的那些维度对所有类贡献相同，可以忽略 → 分类隐含了一次到 $K-1$ 维的投影（L91）；
4. 想再降到 $L<K-1$ 维，就投影到"投影后类均值方差最大"的子空间——「a form of PCA for the transformed class means」（L93）。这就是 `n_components`。

**本库综合**：第 4 步的 PCA 作用在**类均值**上（样本数 = 类数），不是作用在原始数据上。这带来两个直接后果：(a) 几乎不可能过拟合；(b) 完全看不见类内结构，所以它和 PCA 的取舍永远不该用"降维后谁更省"来比，而要看下游任务。

## 参数代价：多一个自由度就是多一份样本

**本库推导**（不是来源论述）。自由参数个数：

| 成员 | 协方差参数 | 均值参数 | 先验 | 合计 |
| --- | --- | --- | --- | --- |
| `GaussianNB` | $K d$（对角） | $K d$ | $K-1$ | $2Kd + K - 1$ |
| LDA | $d(d+1)/2$ | $K d$ | $K-1$ | $\frac{d(d+1)}{2} + Kd + K - 1$ |
| QDA | $K\,d(d+1)/2$ | $K d$ | $K-1$ | $K\frac{d(d+1)}{2} + Kd + K - 1$ |

代入 $d=100$、$K=10$：`GaussianNB` ≈ 2,009；LDA ≈ 6,059；QDA ≈ **51,509**（约 LDA 的 8.5 倍）。而 QDA 还要求**每类**样本数大于 $d$ 才能得到满秩的样本协方差。

**两条实践判据（本库综合）**：
1. 只要 $d$ 与每类样本数同一量级，QDA 的协方差估计就不可靠——这正是 `shrinkage` / `reg_param` 存在的理由（见下节与 [[sklearn-lda-qda]], L97–L101）。
2. 类数很少但特征很多时，LDA 的 `transform` 只能给 $K-1$ 维，作为特征压缩器会**损失绝大部分信息**；此时应当直接用正则化线性模型或树模型。

## 协方差怎么估：三个假设层与两个收缩旋钮

| 手段 | 作用对象 | 语义 | 依据 |
| --- | --- | --- | --- |
| 对角假设（`GaussianNB`） | 每个 $\Sigma_k$ | 特征条件独立，参数从 $O(d^2)$ 降到 $O(d)$ | [[sklearn-lda-qda]], L60–L63 |
| 共享假设（LDA） | 全部类 | 只要一套协方差，自由度从 $K$ 套降到 1 套 | L67 |
| `shrinkage` | 每个 $\Sigma$（LDA 是共享的那个） | 向"对角方差"收缩；`'auto'` = Ledoit-Wolf 解析解 | L97–L99 |
| `covariance_estimator` | 同上 | 换任意具备 `fit` + `covariance_` 的估计器（如 OAS） | L103 |
| QDA `reg_param` | 每个类协方差 | $S^2\leftarrow(1-\rho)S^2+\rho I$（**外部核查**，API 页） | 见 [[sklearn-lda-qda]] 转换损失第 4 条 |

**关键口径（本库综合）**：`shrinkage='auto'` 的「optimal」指的是**协方差估计误差**，不是分类准确率——来源自己用"MSE 更小"来论证 OAS 更好（L101）。因此收缩强度本身应被当作一个待验证的建模选择，而不是已知最优。

**假设（待验证）**：`shrinkage` 与 `reg_param` 的收缩终点（对角阵 / 单位阵）**不是仿射等变的**。纯 LDA/QDA 理论上对特征缩放的线性变换等变，但一旦启用收缩，结果就会随特征尺度变化。文档完全未讨论这一点，需要用实验确认（见 [[sklearn-lda-qda]] 待解问题 3）。

## 边界与失效模式

1. **高斯假设是全部推导的起点**。类条件分布非高斯（多峰、重尾、类别内异方差结构复杂）时，LDA/QDA 的决策面形状就没有理由是最优的；本来源不提供任何失效诊断方法。
2. **"无需调参"是错的读法**。基础模型确实没有正则强度，但协方差怎么估（对角/共享/收缩/自定义估计器）是**结构选择**，且默认值（`shrinkage=None`）在小样本下直接失效（图 2：比值越过 ≈1 后准确率塌到 0.65 以下）。
3. **先验默认等于训练集类比例**（**外部核查**，API 页 `priors`）。因此重采样之后再 fit，阈值会跟着移动——这在正文里完全看不出来（见 [[sklearn-lda-qda]] 关键主张 7）。
4. **`coef_` 不是特征重要性**：它被 $\Sigma^{-1}$ 旋转过（见上节）。

## 待解问题

- LDA 与 **logistic 回归**在什么条件下等价/更优？ESL §4.3 是标准答案，但本库没有该文献（剪藏的参考文献丢失）。需要另找来源。
- 收缩强度与分类准确率的一致性在什么数据规模下成立？（见 [[sklearn-lda-qda]] 待解问题 1）
- `GaussianNB` 的完整行为（`var_smoothing`、部分拟合）在同批未入库的 1.9 节里，本页对它的描述只到"对角协方差"为止。

## 关联页面

- 来源：[[sklearn-lda-qda]]
- 工具：[[scikit-learn]]
- 概念：[[regularization-and-sparsity]]（系数惩罚 vs 协方差收缩）、[[leaderboard-vs-cv]]（收缩强度该怎么选，最终仍是验证问题）
- 实践：[[linear-models-as-baselines]]（线性/生成式基线的定位）
