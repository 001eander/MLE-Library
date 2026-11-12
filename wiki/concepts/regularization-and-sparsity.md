---
title: "正则化与稀疏性（Regularization and Sparsity）"
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - regularization
  - sparsity
  - linear-model
  - feature-selection
  - margin
  - prototype
  - ensemble
  - shrinkage
  - precision-matrix
  - neural-network
  - mlp
sources:
  - "[[sklearn-linear-models]]"
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-naive-bayes]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-decision-trees]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-neural-networks]]"
  - "[[sklearn-covariance]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x 的 linear_model 模块（推断）
  - scikit-learn 1.9.x 的 discriminant_analysis 与 naive_bayes 模块（平滑部分）
  - scikit-learn 1.7.x – 1.9.1 的 svm 模块（`C` 与 margin 部分，见 [[sklearn-svm]]）
  - scikit-learn 1.7.x – 1.9.x 的 neighbors 模块（原型收缩部分，`shrink_threshold`，见 [[sklearn-nearest-neighbors]]）
  - scikit-learn 1.9.x 的 tree 模块（树容量与 `ccp_alpha` 部分，见 [[sklearn-decision-trees]]）
  - scikit-learn 1.9.1 的 ensemble 模块（收缩与随机化预算部分，见 [[sklearn-ensembles]]）
  - scikit-learn 1.9.1 的 feature_selection 模块（零系数的下游用法部分，见 [[sklearn-feature-selection]]）
  - scikit-learn 1.8.0 – 1.9.1 的 neural_network 模块（MLP 的 `alpha` 部分，见 [[sklearn-neural-networks]]）
---

# 正则化与稀疏性

本页整理「正则化」这一族概念在不同模型里的**共同结构**：$ℓ_2$ 与 $ℓ_1$ 各自改变什么、稀疏性从哪里来、为什么“系数为 0”不能直接读成“该特征不重要”，以及**同一个正则化思想在七个不同对象上的投影**。事实来自 [[sklearn-linear-models]]，引用写作 `[[sklearn-linear-models]], L146`；协方差收缩一节来自 [[sklearn-lda-qda]]（`[[sklearn-lda-qda]], L97`），平滑来自 [[sklearn-naive-bayes]]，`C` 与 margin 来自 [[sklearn-svm]]，原型收缩来自 [[sklearn-nearest-neighbors]]，树容量与代价复杂度剪枝来自 [[sklearn-decision-trees]]，收缩与随机化预算来自 [[sklearn-ensembles]]。MLP 的 `alpha` 是 $ℓ_2$ 面的**矩阵版**，不新增一个面，见 [[sklearn-neural-networks]]。

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

**外部核查（2.6 节，不在 1.2 剪藏内）给出了精确式子**（[[sklearn-covariance]], L32）：

$$\Sigma_{\text{shrunk}}=(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$$

即终点是**标量乘单位阵** $\frac{\mathrm{Tr}\hat\Sigma}{p}I$，不是「方差对角阵」。本机实测（[[sklearn-covariance]] 实测 4）逐位复现该式，且 $\alpha=1$ 时对角全部相等、非对角全零。

**分歧（保留，不覆盖）**：1.2 节文档说 LDA 的 `shrinkage=1` 用「对角方差阵」，而 1.6.1 实测 `LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1.0, store_covariance=True).covariance_ = 133.3276\,I`（经验对角是 `[1.06, 24.25, 383.12]`）；`shrinkage=0.5` 与 `shrunk_covariance(pooled, 0.5)` 逐位相同。因此**手工给定的 float 走的是上式的标量乘单位阵**，不是文档说的对角阵。两种说法并存：本库按实现记录，同时保留文档原话，见 [[covariance-estimation]] §2.3。

**本库综合（重要口径）**：「optimal」在这里指的是**协方差估计误差**，不是分类准确率。证据在同一段：来源用「OAS yields a smaller Mean Squared Error」（L101）来论证 OAS 更好——如果优化目标是准确率，MSE 就不构成理由。所以 `shrinkage='auto'` 是「一个定义明确的默认值」，不是「分类任务上的最优值」。2.6 节的实测把这条口径量化了：$p=5,n=40$ 上 LW 的 α（0.207）在 held-out 负对数似然上**输给**手设 α=0.1（7.516 vs 7.450，8 种子均值 7.295 vs 7.267）。

外部核查（官方 API 页，不在剪藏内）：QDA 还有一个 `reg_param`，做 $S^2\leftarrow(1-\rho)S^2+\rho I$。它与 `shrinkage` **不是同一个旋钮**：`shrinkage=1` 的终点是 $\frac{\mathrm{Tr}}{p}I$，`reg_param=1` 的终点是 $I$。

**两个新发现（均为实测）**：

1. **`shrinkage='auto'` 与手设 float 的尺度行为不同**。`'auto'` 内部先 `StandardScaler` → `ledoit_wolf` → 按 `scale_` 还原，所以**对特征缩放等变**（decision function 差 8.9e-15）；手设 float 直接用上式，**不等变**（同一数据缩放后 decision function 差 7.22）。[[gaussian-discriminant-analysis]] 曾把「收缩终点不是仿射等变的」列为待验证假设——现在可以结论：**取决于走哪条路**。
2. **四个 estimator 是一个算子**：`LedoitWolf`/`OAS` 的 `covariance_` 与 `ShrunkCovariance(shrinkage=<该估计器>.shrinkage_)` 逐位相同。所以「选估计器」实际上是「选 α 的来源」（0 / 手设 / 解析）。

### 同一个 Σ 上的另一个算子：稀疏精度矩阵（**不是**新的正则化面）

2.6 节还给了第三个操作：对**精度矩阵** $K=\Sigma^{-1}$ 加 $ℓ_1$（[[sklearn-covariance]], L101–L105）：

$$\hat K=\arg\min_K\ \mathrm{tr}SK-\log\det K+\alpha\|K\|_1,\qquad \|K\|_1=\sum_{i\ne j}|K_{ij}|$$

为什么它**不构成第八个面**：作用对象仍然是二阶结构（与收缩同属一个对象），只是把「向单位阵拉」换成「把非对角元压到 0」。它与收缩的关系是**互补而非并列**：

| | 收缩（shrinkage） | 稀疏精度（GraphicalLasso） |
| --- | --- | --- |
| 作用对象 | $\Sigma$ | $K=\Sigma^{-1}$ |
| 几何效果 | 把特征值谱压向同一值 | 把 $K$ 的非对角元压到 **0** |
| 能否产生 0 | 否 | **是**（在 $K$ 上） |
| 能否恢复非对角结构 | **不能**（图 3：非对角被洗白） | **能**（图 3：留下稀疏的非对角点） |
| 惩罚范围 | 全矩阵（对角也被拉伸） | **只罚非对角**（源码 `_objective`：`abs(K).sum() - abs(diag(K)).sum()`） |
| 尺度敏感 | **是**（目标含 $\mathrm{Tr}/p$） | **是**（$\alpha$ 是绝对阈值，实测量纲差 1000 倍时 F1 0.308→0.207） |

两点必须记住（本库综合）：① **$\Sigma$ 稀疏与 $K$ 稀疏是相反的事**——$\Sigma_{ij}=0$ 是边际独立，$K_{ij}=0$ 是条件独立；FA 的低秩 + 对角噪声对应**稠密**的 $K$。② **CV 选的 $\alpha$ 与结构恢复目标冲突**：实测 $n=60,p=20$、真值 12 条边时 CV 给 34 条边（F1 0.522），但它的 $\Sigma$ 误差最小——两个目标不可兼得。

### 稀疏与收缩的对比

| 手段 | 几何效果 | 参数是否变 0 | 什么时候用 |
| --- | --- | --- | --- |
| $ℓ_1$（Lasso） | 坐标轴向的软阈值 | 是 | 要减少特征数 |
| $ℓ_2$（Ridge） | 球形收缩，全体变小 | 否 | 共线性、降方差 |
| 协方差收缩 | 把 $\Sigma$ 拉向 $\frac{\mathrm{Tr}}{p}I$ | 否（协方差不会变 0） | 样本少、维度高、矩阵病态 |
| 稀疏精度 $ℓ_1$ | 把 $K$ 的非对角元压到 0（＝学条件独立图） | **是（$K$ 上）** | 要结构/偏相关，不要稠密 $\hat\Sigma$ |

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

**本库综合**：前四个面里只有 SVM 的调节旋钮**同时改变模型的容量与训练成本**——大 `C` 不只是更弱的正则，还可能是 10 倍的训练时间（L202）。第五、第六个面（原型收缩、树容量）也都同时改容量与耗时：`shrink_threshold` 只是阈值运算（便宜），而树的 `max_depth`/`ccp_alpha` 直接改变模型大小与拟合成本。因此网格搜索不应只记分数，还要记耗时（见 [[svm-selection-and-tuning]]）。

## 第五个正则化面：原型收缩（prototype shrinkage）

第五个面出现在最近邻（[[sklearn-nearest-neighbors]], L182–L186）：`NearestCentroid(shrink_threshold=r)` 把每个类的每个特征值**先除以该特征的类内方差、再减去阈值 $r$、过零置零**（L184）。源码里的形式就是标准软阈值算子（外部核查：`sklearn/neighbors/_nearest_centroid.py` @ 1.9.1）。

$$S(z,r)=\operatorname{sign}(z)\max(0,|z|-r)$$

**这意味着它与 Lasso 的稀疏机制是同一个算子**（本页「稀疏性从哪来」一节的 $S(z,\alpha)$），只是作用对象不同：

| | Lasso | 最近收缩质心 |
| --- | --- | --- |
| 阈值作用在 | 回归系数 $w$ | **标准化后的类质心偏移** |
| 先除什么 | 列范数 $\lVert x_j\rVert_2^2$（坐标下降步长） | **该类该特征的类内标准差** |
| 置零的含义 | 特征不进入模型 | **该特征不参与最近质心比较** |
| 能否产生特征维的精确 0 | 能 | 能 |
| 官方量化收益 | 未在本节给 | 0.81 → 0.82（L186） |

三个必须区分的点：

1. **同样的「零」，语义不同**。Lasso 的 0 是「模型不用这个特征」；这里的 0 是「分类时不比这个特征」。两者都**不能**读成「该特征不重要」（见本页「边界：稀疏 ≠ 不重要」）。
2. **它自带一层标准化，但不等于尺度无关**（本库判断，未实测）。除以类内方差使阈值与特征量级解耦，但**最终距离仍在原始空间里算**——所以外层缩放依然必要（[[nearest-neighbors-selection-and-tuning]] 第二节）。这一点与 Lasso 的尺度敏感性是同一类问题，不是同一层问题。
3. **它是前五个面里唯一作用在「类原型」上的**。前四个面分别作用在系数、协方差、频率、约束松弛量；原型收缩直接改输入到分类规则里的那个向量（第六个面则作用在模型结构上）。

## 第六个正则化面：树容量与代价复杂度剪枝（tree capacity）

第六个面出现在决策树（[[sklearn-decision-trees]], L368–L378）：最小代价复杂度剪枝把模型大小写成一个**可加的惩罚项**

$$R_\alpha(T)=R(T)+\alpha\,|\tilde T|$$

其中 $|\tilde T|$ 是终端节点（叶子）个数，$\alpha\ge0$ 是复杂度参数。这与 Lasso/Ridge 的形式同构（残差 + 惩罚），但**被罚的不是系数，是模型结构的尺寸**。

与前面五个面的关键区别：

| 面 | 罚什么 | 惩罚单位 | 是否连续 | 产生“零”吗 |
| --- | --- | --- | --- | --- |
| $ℓ_2$ / $ℓ_1$ | 系数 | 系数大小 / L1 质量 | 连续 | 仅 $ℓ_1$ |
| 协方差收缩 | 协方差矩阵 | 向对角/单位阵的比例 | 连续 | 不产生零 |
| 平滑 | 条件频率 | 伪计数 $\alpha$ | 连续 | 不产生零 |
| $C$ / margin | 松弛量 | 违反间隔的代价 | 连续 | 不产生零 |
| 原型收缩 | 类质心偏移 | 软阈值 $r$ | 连续 | 产生零（特征不参与比较） |
| **树容量** | **叶子个数** | **每个叶子换来的不纯度下降** | **离散**（$\alpha$ 连续，但可行树集合离散） | **删叶子 = 合并区域** |

三个必须区分的点：

1. **它是后剪枝，不是前剪枝**。$R_\alpha$ 要先有一棵完整的树才能算（L368–L378），而 `max_depth`/`min_samples_leaf` 是在长树过程中阻止切分。两者不是同一个旋钮的两个刻度（见 [[decision-trees-selection-and-tuning]]）。
2. **$R(T)$ 的定歪了**。经典 CART 用误分类率，scikit-learn 用**终端节点的样本加权不纯度**（L376）。因此 $R_\alpha$ 的尺度与 `criterion`（Gini/entropy/MSE）绑定：同一个 $\alpha$ 在不同准则下不是同一个强度。
3. **$\alpha$ 有可解释的单位**：$\alpha_{\text{eff}}(t)=\frac{R(t)-R(T_t)}{|T|-1}$（L378）就是“每多一个叶子需要换来多少不纯度下降”。这是本库目前唯一一个**用验证集可以直接读成“每个叶子的价格”**的正则参数（前提是 $R$ 的定与准则一致）。

**本库综合（假设，待验证）**：因为被罚量是**离散的叶子数**，$\alpha$ 的验证曲线应该是**阶毯状**而非平滑 U 形：$\alpha$ 在一个区间内不改变树，越过临界值才整枝消失。这意味着扫 $\alpha$ 时“最优值”通常是区间而不是点，与 $ℓ_2$ 的平滑最优解不同。

## 第七个正则化面：收缩与随机化预算（ensemble）

第七个面出现在树集成（[[sklearn-ensembles]]）。它不是一个惩罚项，而是**四个改变容量/方差的旋钮**：

| 旋钮 | 定义 | 作用机制 | 位置 |
| --- | --- | --- | --- |
| `learning_rate`（$\nu$） | $F_m=F_{m-1}+\nu h_m$ | **收缩步长**：每棵树只贡献一部分 | L358–L361 |
| `subsample` | 每轮无放回抽一部分样本 | **随机化**：降梯度估计的方差 | L367 |
| `max_features` | 每节点/每模型抽特征子集 | **随机化**：去相关 | L373、L436 |
| `max_bins` | 分箱数（默认 255，可调小） | **量化**：减少候选切分点 | L64 |

此外还有三个容易被当成「工程参数」的容量旋钮：

- **`l2_regularization`**（默认 0.0）：叶子级 $ℓ_2$，但出现在**分母**上——$w_k=-\sum g_i/(\sum h_i+\lambda)$（官方 A2 折叠块；剪藏只剩 L68–L70 的目标函数）。因此它惩罚的是「Hessian 小（平坦区域）的叶子」，与 Ridge 罚系数大小的几何不同。
- **早停**（HGBT 默认 ≥10,000 样本开启，L72）：把「轮数」变成一个**由验证损失决定的随机变量**。它不是显式惩罚，但效果是截断模型容量。
- **单调约束 / 交互约束**（L187、L217）：不是缩小参数，而是**删掉一部分候选函数**。`monotonic_cst` 删掉非单调的树，`interaction_cst` 删掉跨组交互。

与前面六个面的对照（新增一行）：

| 面 | 罚/控什么 | 单位 | 是否连续 | 产生“零”吗 |
| --- | --- | --- | --- | --- |
| $ℓ_2$ / $ℓ_1$ | 系数 | 系数大小 / L1 质量 | 连续 | 仅 $ℓ_1$ |
| 协方差收缩 | 协方差矩阵 | 向对角/单位阵的比例 | 连续 | 不产生零 |
| 平滑 | 条件频率 | 伪计数 $\alpha$ | 连续 | 不产生零 |
| $C$ / margin | 松弛量 | 违反间隔的代价 | 连续 | 不产生零 |
| 原型收缩 | 类质心偏移 | 软阈值 $r$ | 连续 | 产生零（特征不参与比较） |
| 树容量 | 叶子个数 | 每个叶子的不纯度下降 | 离散 | 删叶子 = 合并区域 |
| **集成预算** | **步长 / 随机性 / 分箱粒度** | **$\nu$、抽样比例、箱数** | **连续（除分箱）** | **不产生零（`max_bins` 可合并切分点）** |

三个必须区分的点：

1. **收缩与随机化不是惩罚项**。它们不改目标函数，而改**估计量的方差**；因此「训练误差」对它们的反应与 $ℓ_2$ 不同：$\nu$ 减半会直接抬高训练误差，而 `subsample` 减小主要抬高方差。
2. **$\nu$ 与 `n_estimators` 是同一个预算的两个刻度**（L363）。单独谈「学习率小就更正则」不成立：小 $\nu$ + 大 $M$ 可以等价于大 $\nu$ + 小 $M$（直到早停介入）。这与 $ℓ_2$ 的 $\alpha$ 可以单维度比较完全不同。
3. **约束型正则（单调/交互）与惩罚型正则不可互换**。前者改变**可行集**，后者改变**同一可行集上的偏好**。一个非单调的树用再大的 $ℓ_2$ 也变不成单调的树。

**本库综合（分类，供后续页面引用）**：正则化在本库已出现**三种机制**——① **罚**（shrink：$ℓ_1$/$ℓ_2$、收缩、平滑、原型、$C$）；② **删**（constraint：叶子数、单调/交互约束、深度/叶子大小）；③ **平均**（variance reduction：bagging/RF/子采样）。三者对「过拟合」的作用路径不同：罚与删降低假设空间/有效容量，平均降低估计方差。把「集成」当成「又一种正则化」容易出错：它不缩小单模型的容量，而是买「多个模型的平均」。

## MLP 的 alpha：同一个 $ℓ_2$ 的矩阵版（不新增面）

MLP 的 `alpha` 看起来像又一个正则化面，其实不是。来源只说两个估计器都用 `alpha` 做 L2 正则、惩罚大权重（[[sklearn-neural-networks]], L101）；官方丢失的数学节把目标写成

$$\text{Loss}=\text{数据损失}+\frac{\alpha}{2n}\lVert W\rVert_2^2$$

外部核查 1.9.1 的默认 `alpha=1e-4`；1.6.1 实现（`_backprop`）只对 `coefs_` 求和，**不含 `intercepts_`**，因此偏置不被收缩。

与 Ridge 的 $ℓ_2$ 对照：

| | Ridge | MLP |
| --- | --- | --- |
| 作用对象 | 一个系数向量 $w$ | **一组权重矩阵** `coefs_`（逐层） |
| 目标项 | $\alpha\lVert w\rVert_2^2$ | $\frac{\alpha}{2n}\lVert W\rVert_2^2$ |
| 是否罚偏置 | 截距默认不罚 | **不罚** `intercepts_` |
| 默认强度 | `alpha=1.0` | `alpha=1e-4`（外部核查） |
| 跨模型可比性 | 不可比 | **更不可比**：还随架构的宽度/深度变化 |

三条必须区分的点：

1. **它不是新机制，而是新对象**。$ℓ_2$ 收缩的数学形式没变；变的是「被罚的参数是一组矩阵」以及「损失被 $1/n$ 归一化」。所以它归入本页第一个面，不另起一面。
2. **`alpha` 不能跨架构比较**。同一个 `alpha` 在 `(10,)` 与 `(100,100)` 上对应的有效强度不同，因为 $\lVert W\rVert^2$ 的量级随参数个数增长；跨估计器（Ridge/Lasso/MLP）更不可比。
3. **早停/迭代预算不是惩罚项**。MLP 还有 `max_iter`、`tol`、`n_iter_no_change`、`early_stopping` 等预算旋钮（外部核查 1.9.1 签名），它们通过「截断优化」改变有效容量，但不改目标函数。按本页的分类，它们属于「预算层」，既不是第八个面，也不等于「删」或「平均」。

**本库综合**：把 MLP 的 `alpha` 写成「第八个正则化面」会掩盖真正的新东西——新东西不是正则项，而是**非凸优化 + 随机初始化**（见 [[multi-layer-perceptron]]）。正则化在这里仍是熟悉的 $ℓ_2$，只是被放在一个更难优化的模型里。

## 零系数的下游用法：特征选择（**不是**第八个面）

$ℓ_1$ 产生零（[[sklearn-linear-models]]）之后有两条路，本库此前的页只写了第一条：

1. **把稀疏模型当模型用**：零系数就是不参与预测的列，预测成本随之下降，但**保留的系数仍被收缩**（有偏）。
2. **把稀疏模型当选择器用**（1.13 节）：`SelectFromModel(LinearSVC(penalty="l1"))` 用 `threshold=1e-5` 的默认值把「非零」翻成「保留」，再用选中的列**重拟合一个无惩罚（或弱惩罚）模型**（[[sklearn-feature-selection]], L91–L119）。

第二条不是「另一种正则化」，而是**把软阈值换成硬选择**：系数不再被缩小，但「留下了哪几列」变成了一个二值决策，其误差无法用系数大小衡量。本库不把它列为第八个面，因为它不改变任何目标函数——它改变的是「谁进入下一轮拟合」。

三点必须与本节前面的面区分：

| | 惩罚（$ℓ_1$/$ℓ_2$） | 特征选择 |
| --- | --- | --- |
| 作用对象 | **系数**（连续收缩） | **列**（二值保留） |
| 是否改变后续拟合 | 不改变（同一个模型） | 改变（重拟合的模型只见到子集） |
| 偏差来源 | 收缩偏置 | 选错列的偏置（无偏但高方差） |
| 强度旋钮 | `alpha` / `C` | `alpha`/`C`（嵌入式）、`k`/`percentile`/`alpha`（过滤式）、`n_features_to_select`（包裹式） |
| 方差估计 | 有解析/自助法路径 | **无官方入口** |

三条工程后果：

- **方向一致、语义不同**：$C$ 越小、`alpha` 越大 → 特征越少（L119）。但「$C=0.01$ 的 L1 逻辑回归」与「`Lasso(alpha=0.01)`」的惩罚强度不可比（[[support-vector-machines]] 的 $C=1/\alpha$ 只在 Ridge 上闭式成立）。
- **本节只覆盖「嵌入式选择」的来源**。过滤式（$F$/互信息/方差）与包裹式（CV 分数）的目标函数与惩罚项无关，它们的正则性来自「少用几个特征」而非「罚系数」（[[feature-selection]]）。
- **「稀疏 ≠ 不重要」在选择上更强**：相关组里被 L1 留下的是**任意的**，而选择把这种任意性固定成「这几列」。下游若把这组列名当结论（「这三个指标是驱动因素」），就把 $ℓ_1$ 的随机选择读成了领域事实（[[feature-selection]] 的失效模式表）。

## 邻域权重上的正则：LLE 的 `reg`（也不是第八个面）

流形学习（[[sklearn-manifold-learning]], L78）给出一个与前面七个面都不同的作用对象：**局部重构权重矩阵**。

- 标准 LLE 在每个邻域解 $\sum_j w_{ij}x_j\approx x_i$；当 $k > D$ 时该线性系统**秩亏**，实现于是加一个正则项（`reg=1e-3`，相对局部权重矩阵的迹选取）。来源明确说：$r\to0$ 时收敛到目标嵌入，但 $r>0$ 时**没有最优性保证**，表现为「embeddings which distort the underlying geometry」。
- 用实测（S-curve、$k=10$、$d=2$）看：$10^{-6}$ → trustworthiness 0.9104、$10^{-3}$（默认）→ **0.9861**、$0.1$ → 0.9521、$1.0$ → 0.9519。**两个方向的失效机制不同**：过小 = 秩亏（数值不稳），过大 = 几何被抹平（真实结构被罚掉）。这与 Ridge 的「$\alpha$ 越大越平滑、单调变差」不同。
- **为何不列为第八个面**：它不作用在模型系数上、不作用在协方差上、不产生稀疏、不改变假设空间，只保证**局部线性系统可解**——它是数值正则（stabilizer），不是容量正则。MLLE/HLLE 更彻底：它们用「多个权重向量」与「Hessian 二次型」直接把秩补回来，**不再需要 `reg`**，代价是各自新增下界（$k>n_c$、$k>n_c(n_c+3)/2$）与新的 `*_tol`。
- 这条与「收缩」家族的关系：形式上都缩小了某个量的范数，但**目的相反**——协方差收缩是为了降低方差/改善条件数（[[sklearn-lda-qda]]），LLE 的 `reg` 只是为了不让矩阵奇异；它不改善泛化，只保证算得下去。

## 正则强度与自由度：为什么信息准则会失效

来源对 `LassoLarsIC`（AIC/BIC）的失效条件写得很具体：这类准则「需要正确的自由度估计、来自大样本的渐近结果、假设真实模型在候选集内」，并且「在病态问题（如特征多于样本）上往往会失效」（L212）。同一段还说它们在**训练集内**计算（L210）。

**本库综合**：这三条限制不是实现缺陷，而是"用训练集内指标替代留出验证"的必然代价。它们与交叉验证的差别不在精度，而在**假设强度**——AIC/BIC 假定候选集内含真模型，CV 不假定。这条与 [[leaderboard-vs-cv]] 的"代理指标必须有正确的结构假设"是同一件事。

## 边界：稀疏 ≠ 不重要

三条不能越过的界线：

1. **相关组内的 0 是任意的**。Lasso「大概率领随机选一个」，Elastic-Net「倾向于都选」（L247）。所以"系数为 0"只说明"在其余特征给定、这个 alpha 下它被牺牲了"。
2. **惩罚强度与尺度耦合**。同一批特征在不同缩放下的稀疏模式不同；文档对求解器尺度的警告（L499、L938）是同一问题的数值侧证据。
3. **没人检验显著性**。来源只提供了一条路径：无惩罚回归的 p 值与置信区间要用 statsmodels 或 bootstrap，sklearn 内没有（L515）。稀疏解不附带任何推断保证。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]、[[sklearn-nearest-neighbors]]、[[sklearn-decision-trees]]、[[sklearn-ensembles]]、[[sklearn-feature-selection]]、[[sklearn-neural-networks]]、[[sklearn-manifold-learning]]
- 工具：[[scikit-learn]]
- 实践：[[linear-models-as-baselines]]（尺度、共线、五个坑）、[[svm-selection-and-tuning]]（`C`/`gamma` 网格与加权后重扫）、[[nearest-neighbors-selection-and-tuning]]（`shrink_threshold` 需 CV、且外层仍要缩放）、[[decision-trees-selection-and-tuning]]（`ccp_alpha` 与前剪枝的先后顺序）、[[ensembles-selection-and-tuning]]（学习率×轮数、子采样、约束的工程写法）、[[feature-selection-workflow]]（把零系数当选择器时的写法与泄漏）、[[mlp-selection-and-tuning]]（MLP 的 `alpha` 网格、缩放与早停）、[[competition-ml-workflow]]（特征选择在竞赛流程里的位置）
- 概念：[[feature-preprocessing]]（**缩放 = 改每个特征的有效惩罚强度**，因此同一个 `alpha` 只在给定缩放下可比；实测缩放器把 `SVC` 从 0.7767 救回 0.9833）
- 概念：[[gaussian-discriminant-analysis]]（协方差为何需要收缩：参数个数与样本量的仗）、[[naive-bayes]]（平滑与“坏估计量”）、[[support-vector-machines]]（`C` 与 margin 的机制）、[[nearest-neighbors]]（最近质心与最近收缩质心的位置）、[[decision-trees]]（贪心、不稳定与容量旋钮）、[[ensembles]]（收缩、随机化与平均的三种机制）、[[feature-selection]]（三种选择家族、代理指标的假设、选择的方差）、[[multi-layer-perceptron]]（`alpha` 与非凸优化）、[[manifold-learning]]（LLE 的 `reg` 作为邻域权重上的数值正则）、[[leaderboard-vs-cv]]（代理指标的假设强度）
- 对照：[[kaggle-getting-started]] 里「特征越多越好」与「特征多反而变差」的张力（`raw/articles/Kaggle 入门指南.md:77`、`:290`）——正则化是"让模型挑特征"的一种实现，但不解决"新特征是否引入噪声"的问题
