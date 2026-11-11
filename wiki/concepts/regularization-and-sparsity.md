---
title: "正则化与稀疏性（Regularization and Sparsity）"
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - regularization
  - sparsity
  - linear-model
  - feature-selection
  - margin
sources:
  - "[[sklearn-linear-models]]"
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-naive-bayes]]"
  - "[[sklearn-svm]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.x – 1.9.x 的 linear_model 模块（推断）
  - scikit-learn 1.9.x 的 discriminant_analysis 与 naive_bayes 模块（平滑部分）
  - scikit-learn 1.7.x – 1.9.1 的 svm 模块（`C` 与 margin 部分，见 [[sklearn-svm]]）
---

# 正则化与稀疏性

本页整理线性模型里"惩罚项"这一族概念：$ℓ_2$ 与 $ℓ_1$ 各自改变什么、稀疏性从哪里来、以及为什么"系数为 0"不能直接读成"该特征不重要"。事实来自 [[sklearn-linear-models]]，引用写作 `[[sklearn-linear-models]], L146`；协方差收缩一节来自 [[sklearn-lda-qda]]（引用写作 `[[sklearn-lda-qda]], L97`）。

## 五种惩罚，一张表

| 名称 | 目标函数 | 对系数的影响 | 依据 |
| --- | --- | --- | --- |
| 无惩罚（OLS） | $\lVert Xw-y\rVert_2^2$ | 不解唯一时方差极大；共线时对噪声敏感 | L24、L40 |
| $ℓ_2$（Ridge） | $\lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_2^2$ | 整体收缩，接近共线时更稳，**不置零** | L67、L70 |
| $ℓ_1$（Lasso） | $\frac{1}{2n}\lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_1$ | **可以把系数恰好设为 0** → 隐式特征选择 | L141、L146 |
| 混合（Elastic-Net） | $\frac{1}{2n}\lVert Xw-y\rVert_2^2 + \alpha\rho\lVert w\rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2$ | 组内相关特征"倾向于都选"，兼有 Ridge 的部分旋转稳定性 | L247–L256 |
| 组稀疏（MultiTask） | Frobenius + $ℓ_{2,1}$ | 非零是**整列**，即所有任务共享同一支撑 | L231、L235 |

注意两个目标函数的**归一化不同**：Ridge 没有 $\frac{1}{n}$，Lasso/ElasticNet 有（L67 vs L146、L256）。**本库综合**：这使同一个 `alpha` 数值在两者中不可比；跨估计器比较必须重扫网格（实践细节见 [[linear-models-as-baselines]]）。

## 稀疏性从哪来：软阈值

Lasso 的坐标下降解有闭式：

$$w_j = \frac{S\left(x_j^{T}\left(y - X_{-j}w_{-j}\right), \alpha\right)}{\|x_j\|_2^2}, \qquad S(z,\alpha) = \operatorname{sign}(z)\max\left(0, |z|-\alpha\right)$$

来源直接点明关键性质：**「软阈值函数在 $\alpha \ge |z|$ 时恰好为零」**（L190）。这就是 Lasso 产生精确 0 的机制，而不是"优化到很小再截断"。同一机制也解释了为什么 `alpha` 越大越稀疏（L196）。

第二种稀疏来源是 **ARD**：它把 Ridge 的球形高斯先验换成中心椭圆高斯，每个系数有自己的精度 $\lambda_i$（L413–L421），因此可以在拟合中把某些坐标的精度推向极大、系数压到 0。文档给出的别名是 Sparse Bayesian Learning / Relevance Vector Machine（L423）。

两种稀疏的差别（**本库综合**）：软阈值是**逐坐标的硬判决**，且在相关特征组里由优化路径决定谁被留下；ARD 的稀疏来自**先验的逐坐标尺度**，属于模型选择式的机制。二者都不产生"显著性"，只产生"在这组数据与这个正则强度下的 0"。

## 惩罚项的贝叶斯读法

来源只对 $ℓ_2$ 给出了先验解释：Ridge 的 $ℓ_2$ 正则「等价于在系数的高斯先验（精度 $\lambda^{-1}$）下求最大后验估计」（L345）。贝叶斯版本的更进一步：

- `BayesianRidge`：把 $\alpha$（噪声精度）与 $\lambda$（系数精度）当随机变量，用 gamma 先验（高斯的精度的共轭先验），在 fit 中通过最大化**对数边缘似然**联合估计（L366–L378）。
- `ARDRegression`：先验从球形改成椭圆，逐系数精度（L413–L421）。

**注意不要外推**：来源没有给 Lasso 的 $ℓ_1$ 惩罚一个先验解释（拉普拉斯先验是常见教材说法，但本来源未提，本库不作断言）。因此"Lasso = MAP + Laplace 先验"这条在本库中标记为**未收录**。

## 收缩、选择、分组：三种不同的"简化"

| 想要的效果 | 机制 | 代价 |
| --- | --- | --- |
| 降低方差、稳住共线性 | $ℓ_2$ 收缩（L70） | 系数不会变成 0，模型仍然"用满"所有特征 |
| 减少依赖的特征数 | $ℓ_1$ 的软阈值（L190） | **相关特征组内选择不稳定**（L247） |
| 组内一起留、一起去 | Elastic-Net（L247）或 MultiTask（L231） | 需要额外超参 `l1_ratio` / 需要多任务结构 |
| 给定非零个数的上限 | OMP 的 $ℓ_0$ 约束（L321） | 贪心，非凸，无全局最优保证 |
| 沿正则强度看整个过程 | LARS 的分段线性路径（L300） | 对噪声敏感（L294–L296，转述 Weisberg 的批评） |

由此得出一条**实践判据（本库综合）**：如果目的是"解释哪几个特征在起作用"，Lasso 的单次解不足以支撑结论；如果目的是"减少特征数以便训练更快/更稳"，Lasso 足够，因为不稳定只影响"选了哪一个"，不影响"这组特征里有信号"。

## 第二个正则化面：协方差收缩（covariance shrinkage）

前面所有惩罚都作用在**系数** $w$ 上。LDA / QDA 提供了另一个面：惩罚作用在**协方差矩阵** $\Sigma$ 上（[[sklearn-lda-qda]], L97–L103）。两个面的动机完全不同：

| | 系数惩罚（Ridge/Lasso…） | 协方差收缩（LDA/QDA） |
| --- | --- | --- |
| 作用对象 | $w$（判别方向） | $\Sigma$ 或 $\Sigma_k$（类内散布） |
| 想解决的问题 | 过拟合、共线性、特征筛选 | **小样本下协方差估计不可靠**（$n$ 与 $d$ 同量级） |
| 强度参数 | `alpha`、`C` | `shrinkage`（`'auto'` 或 0–1）、QDA 的 `reg_param` |
| 能否产生 0 | Lasso 能（软阈值） | **不能**——收缩从不把参数置零 |

来源对小样本问题的描述很直接：「the empirical sample covariance is a poor estimator, and shrinkage helps improving the generalization performance」（L97）。

### 收缩的三个端点与两个目标

来源给的端点语义（L99）：`shrinkage=0` = 用经验协方差；`shrinkage=1` = 完全收缩，用**对角方差矩阵**当协方差估计；中间值 = 两者的收缩版。`'auto'` = 按 Ledoit-Wolf 的 lemma **解析地**定出收缩系数（L97）。

**本库综合（重要口径）**：「optimal」在这里指的是**协方差估计误差**，不是分类准确率。证据在同一段：来源用「OAS yields a smaller Mean Squared Error」（L101）来论证 OAS 更好——如果优化目标是准确率，MSE 就不构成理由。所以 `shrinkage='auto'` 是「一个定义明确的默认值」，不是「分类任务上的最优值」。

外部核查（官方 API 页，不在剪藏内）：QDA 还有一个 `reg_param`，做 $S^2\leftarrow(1-\rho)S^2+\rho I$。它与 `shrinkage` **不是同一个旋钮**：`shrinkage=1` 的终点是「只保留方差」，`reg_param=1` 的终点是「完全球化」。

### 稀疏与收缩的对比

| 手段 | 几何效果 | 参数是否变 0 | 什么时候用 |
| --- | --- | --- | --- |
| $ℓ_1$（Lasso） | 坐标轴向的软阈值 | 是 | 要减少特征数 |
| $ℓ_2$（Ridge） | 球形收缩，全体变小 | 否 | 共线性、降方差 |
| 协方差收缩 | 把 $\Sigma$ 拉向对角/单位阵 | 否（协方差不会变 0） | 样本少、维度高、矩阵病态 |

**实践含义（本库综合）**：协方差收缩不能拿来“选特征”。它改变的是模型对类内散布形状的信任度，不是“用不用某个特征”。

## 第三个正则化面：平滑（smoothing）

第三个对象的正则化出现在朴素贝叶斯里（[[sklearn-naive-bayes]]）：它不惩罚系数、也不收缩协方差，而是**给频率或方差加一个正的下界**，目的是防零。

| | 系数惩罚 | 协方差收缩 | 平滑 |
| --- | --- | --- | --- |
| 作用对象 | $w$ | 协方差阵 | 频率 $\hat\theta$ / 方差 $\sigma_y^2$ |
| 解决的问题 | 过拟合/共线/筛选 | 协方差估计不可靠 | **零概率**与零方差 |
| 强度参数 | `alpha`、`C` | `shrinkage`、`reg_param` | 离散族的 `alpha`；`GaussianNB.var_smoothing` |
| 能否产生 0 | Lasso 能 | 不能 | 不能（恰恰相反：它防止 0） |

两个具体口径：

1. **`alpha` 是按特征数缩放的**。公式分母是 $N_y+\alpha n$（[[sklearn-naive-bayes]], L78–L82），所以同一个 `alpha` 在 1k 与 50k 词表上的有效强度不同——**本库综合**，与“跨估计器 alpha 不可比”同构。
2. **`var_smoothing` 加的是方差下限**，不是向对角收缩（源码 1.9.1：`epsilon_ = var_smoothing × max(var(X, axis=0))`）。把它当“`GaussianNB` 的 shrinkage”是错的。

**未收录**：平滑到底对应什么先验（离散族通常读作 Dirichlet、`var_smoothing` 读作方差的弱先验）本来源**未讨论**，因此本库不作断言，与 Lasso 的 Laplace 先验同列为“未收录”。

## 第四个正则化面：C 与 margin（SVM）

前三个面都在改**模型参数**的分布（系数、协方差、频率）；SVM 的正则化在**约束的松弛量**上（[[sklearn-svm]], L317–L323）：

$$\min_{w,b,\xi}\ \frac{1}{2}w^{T}w + C\sum_{i}\xi_{i}\quad \text{s.t.}\ y_{i}\left(w^{T}\phi(x_{i})+b\right)\ge 1-\xi_{i},\ \xi_{i}\ge 0$$

这里同时有两个旋钮在拉扯：最小化 $\lVert w\rVert^{2}$ 是**最大化间隔**，而 $C\sum\xi_i$ 是**为越界样本付钱**。来源的措辞是“惩罚项 `C` 控制惩罚强度，因此充当**逆正则化参数**”（L323）。也就是说：**小 `C` = 强正则 = 决策面更平滑；大 `C` = 弱正则 = 力求全部训练样本正确**（L242）。

| | 系数惩罚（Ridge/Lasso） | 协方差收缩 | 平滑 | **margin/松弛（SVM）** |
| --- | --- | --- | --- | --- |
| 作用对象 | $w$ | $\Sigma$ | 频率 / 方差下限 | **约束的松弛量 $\xi$（间接压 $\lVert w\rVert$）** |
| 强度参数 | `alpha` | `shrinkage`、`reg_param` | `alpha`；`var_smoothing` | **`C`（`NuSVC` 的 ν 是它的重参数化）** |
| 大强度意味着 | 系数变小 | 向对角/单位阵靠 | 更保守的频率 | **`C` 更小** |
| 能否产生 0 | Lasso 能（特征维） | 不能 | 不能 | 能（**样本维**：$\alpha_i=0$） |
| 跨模型可比性 | 不可比 | 不可比 | 不可比 | **仅对 Ridge 有闭式：$C=1/\alpha$** |

三条具体口径：

1. **跨模型换算只有一条**：来源明确给出 $C=\frac1{\alpha}$（仅当对方是 `Ridge` 回归时成立），并声称“精确等价取决于各自优化的目标函数”（L344）。对 Lasso、logistic 一律**重扫网格**。
2. **`C` 与样本权重叠乘**：`class_weight` 与 `sample_weight` 都是“把该样本/类别的 `C` 乘一个系数”（L146、L150），所以加权会改变有效正则强度——这是与 logistic 完全同构的陷阱（[[scikit-learn]] 默认值第 2、28 条）。
3. **这里的稀疏在样本维，不在特征维**：$\alpha_i=0$ 的样本被丢出决策函数（L339），但特征权重仍稠密；要特征维稀疏只能用 `LinearSVC(penalty='l1', dual=False)`（L216）。两者名字相同，工程用途完全不同。

**本库综合**：四个面里只有 SVM 的调节旋钮**同时改变模型的容量与训练成本**——大 `C` 不只是更弱的正则，还可能是 10 倍的训练时间（L202）。因此 SVM 的网格搜索不应只记分数，还要记耗时（见 [[svm-selection-and-tuning]]）。

## 正则强度与自由度：为什么信息准则会失效

来源对 `LassoLarsIC`（AIC/BIC）的失效条件写得很具体：这类准则「需要正确的自由度估计、来自大样本的渐近结果、假设真实模型在候选集内」，并且「在病态问题（如特征多于样本）上往往会失效」（L212）。同一段还说它们在**训练集内**计算（L210）。

**本库综合**：这三条限制不是实现缺陷，而是"用训练集内指标替代留出验证"的必然代价。它们与交叉验证的差别不在精度，而在**假设强度**——AIC/BIC 假定候选集内含真模型，CV 不假定。这条与 [[leaderboard-vs-cv]] 的"代理指标必须有正确的结构假设"是同一件事。

## 边界：稀疏 ≠ 不重要

三条不能越过的界线：

1. **相关组内的 0 是任意的**。Lasso「大概率领随机选一个」，Elastic-Net「倾向于都选」（L247）。所以"系数为 0"只说明"在其余特征给定、这个 alpha 下它被牺牲了"。
2. **惩罚强度与尺度耦合**。同一批特征在不同缩放下的稀疏模式不同；文档对求解器尺度的警告（L499、L938）是同一问题的数值侧证据。
3. **没人检验显著性**。来源只提供了一条路径：无惩罚回归的 p 值与置信区间要用 statsmodels 或 bootstrap，sklearn 内没有（L515）。稀疏解不附带任何推断保证。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]
- 工具：[[scikit-learn]]
- 实践：[[linear-models-as-baselines]]（尺度、共线、五个坑）、[[svm-selection-and-tuning]]（`C`/`gamma` 网格与加权后重扫）、[[competition-ml-workflow]]（特征选择在竞赛流程里的位置）
- 概念：[[gaussian-discriminant-analysis]]（协方差为何需要收缩：参数个数与样本量的仗）、[[naive-bayes]]（平滑与“坏估计量”）、[[support-vector-machines]]（`C` 与 margin 的机制）、[[leaderboard-vs-cv]]（代理指标的假设强度）
- 对照：[[kaggle-getting-started]] 里「特征越多越好」与「特征多反而变差」的张力（`raw/articles/Kaggle 入门指南.md:77`、`:290`）——正则化是"让模型挑特征"的一种实现，但不解决"新特征是否引入噪声"的问题
