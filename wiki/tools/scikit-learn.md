---
title: "scikit-learn（工具页）：linear_model / discriminant_analysis / naive_bayes 的估计器矩阵、默认值与版本边界"
type: tool
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - linear-model
  - discriminant-analysis
  - naive-bayes
  - tooling
sources:
  - "[[sklearn-linear-models]]"
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-naive-bayes]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.x – 1.9.x（linear_model：据官方 1.8.0 / 1.9.1 页比对推断，剪藏未记录版本）
  - scikit-learn 1.9.x（discriminant_analysis：QDA 协方差公式写 1/n，1.8 及更早写 1/(n-1)，见 [[sklearn-lda-qda]]）
  - scikit-learn 1.9.x（naive_bayes：仅两个措辞级证据，推断，见 [[sklearn-naive-bayes]] 的版本判定）
  - 本文只覆盖 `sklearn.linear_model`、`sklearn.discriminant_analysis` 与 `sklearn.naive_bayes` 三个模块
---

# scikit-learn（工具页）

本页记录 scikit-learn 的**当前行为**：模块里有哪些估计器、默认值是什么、哪些默认值会沉默地改变结果。事实来自 [[sklearn-linear-models]]（官方用户指南 1.1 节）、[[sklearn-lda-qda]]（1.2 节）与 [[sklearn-naive-bayes]]（1.9 节），引用写作 `[[sklearn-linear-models]], L146`。

## 收录范围与覆盖状态

`raw/articles/sklearn/` 下现有 18 篇同批剪藏。本页只对已入库的模块负责：

| 文档 | 模块／主题 | 状态 |
| --- | --- | --- |
| 1.1 Linear Models | `sklearn.linear_model` | **已入库**（本页 + [[linear-models-as-baselines]] + [[regularization-and-sparsity]]） |
| 1.2 LDA / QDA | `sklearn.discriminant_analysis` | **已入库**（本页 + [[gaussian-discriminant-analysis]]，来源页 [[sklearn-lda-qda]]） |
| 1.4 Support Vector Machines | `sklearn.svm` | 未入库 |
| 1.6 Nearest Neighbors | `sklearn.neighbors` | 未入库 |
| 1.9 Naive Bayes | `sklearn.naive_bayes` | **已入库**（本页 + [[naive-bayes]]，来源页 [[sklearn-naive-bayes]]） |
| 1.10 Decision Trees | `sklearn.tree` | 未入库 |
| 1.11 Ensembles | `sklearn.ensemble` | 未入库 |
| 1.13 Feature selection | `sklearn.feature_selection` | 未入库 |
| 1.17 Neural network models | `sklearn.neural_network` | 未入库 |
| 2.2 Manifold learning | `sklearn.manifold` | 未入库 |
| 2.3 Clustering | `sklearn.cluster` | 未入库 |
| 2.5 Decomposing signals in components | `sklearn.decomposition` | 未入库 |
| 2.6 Covariance estimation | `sklearn.covariance` | 未入库 |
| 2.7 Novelty and Outlier Detection | `sklearn.covariance` / 异常检测 | 未入库 |
| 3.1 Cross-validation | `sklearn.model_selection` | 未入库（部分概念散见于 [[leaderboard-vs-cv]]） |
| 3.2 Tuning hyper-parameters | `sklearn.model_selection` | 未入库 |
| 3.4 Metrics and scoring | `sklearn.metrics` | 未入库 |
| 8.3 Preprocessing data | `sklearn.preprocessing` | 未入库（仅 `PolynomialFeatures` 在本页涉及） |

## 估计器矩阵（linear_model）

按**目标函数**分类。行号指向 [[sklearn-linear-models]]，公式已改写为规范 LaTeX。

| 估计器 | 目标／损失 | 正则 | 关键超参 | 位置 |
| --- | --- | --- | --- | --- |
| `LinearRegression` | $\min_w \lVert Xw-y\rVert_2^2$ | 无 | `positive=True` 切到 NNLS | L24–L50 |
| `Ridge` | $\min_w \lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_2^2$ | $ℓ_2$ | `alpha`、`solver="auto"` | L66–L84 |
| `RidgeClassifier` | 同上，标签先转 $\{-1,1\}$ | $ℓ_2$ | `alpha`；多类取最大输出 | L102–L110 |
| `Lasso` | $\min_w \frac{1}{2n}\lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_1$ | $ℓ_1$ | `alpha`、`selection`、`tol` | L141–L192 |
| `ElasticNet` | $\min_w \frac{1}{2n}\lVert Xw-y\rVert_2^2 + \alpha\rho\lVert w\rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2$ | $ℓ_1$+$ℓ_2$ | `alpha`、`l1_ratio`（$\rho$） | L243–L256 |
| `MultiTaskLasso` | 同 Lasso，$Y$ 为 $(n_{\text{samples}}, n_{\text{tasks}})$，共享支撑 | $ℓ_{2,1}$ | `alpha` | L229–L241 |
| `MultiTaskElasticNet` | Frobenius + $ℓ_{2,1}$ + Frobenius | 混合 | `alpha`、`l1_ratio` | L266–L278 |
| `Lars` / `LassoLars` | 前向等角路径；`LassoLars` 为 LARS 版 Lasso | $ℓ_1$（后者） | `alpha`；解为分段线性 | L280–L317 |
| `OrthogonalMatchingPursuit` | $\arg\min_w \lVert y-Xw\rVert_2^2$ s.t. $\lVert w\rVert_0 \le n_{\text{nonzero\_coefs}}$ | $ℓ_0$ 约束 | `n_nonzero_coefs` 或 `tol` | L319–L339 |
| `BayesianRidge` | Gaussian 似然 + 系数球形 Gaussian 先验，$\alpha,\lambda$ 由边缘似然估计 | 隐式 $ℓ_2$ | `alpha_init`、`lambda_init`、四个 gamma 超参 | L364–L407 |
| `ARDRegression` | 同上，但逐系数精度 $\lambda_i$（椭圆先验） | 隐式、更强稀疏 | 同上 | L409–L429 |
| `LogisticRegression` | 交叉熵 + $r(w)/(SC)$ | 四种（由 `C`、`l1_ratio` 表达） | `C`、`solver`、`class_weight` | L431–L517 |
| `TweedieRegressor` | 单位偏差 + $\frac{\alpha}{2}\lVert w\rVert_2^2$ | $ℓ_2$ | `power`、`link`、`alpha` | L565–L572 |
| `PoissonRegressor` / `GammaRegressor` | 等价于 `TweedieRegressor(power=1/2, link='log')` | $ℓ_2$ | `alpha` | L570–L571 |
| `SGDClassifier` / `SGDRegressor` | 可选 loss（`log_loss`/`hinge`/`huber`/`epsilon_insensitive`…） | 可选 | `loss`、`penalty`、`learning_rate`、`eta0` | L594–L618 |
| `Perceptron` | perceptron loss + 常数学习率（`SGDClassifier` 的包装） | 无 | — | L602–L612 |
| `RANSACRegressor` | 随机子集 + 内点共识 | 无 | `max_trials`、`residual_threshold`、`min_samples` | L664–L694 |
| `TheilSenRegressor` | 空间中位数（多变量中位数的推广） | 无 | `max_subpopulation` | L703–L739 |
| `HuberRegressor` † | $\sum_i\left(\sigma + H_\epsilon\!\left(\frac{X_iw-y_i}{\sigma}\right)\sigma\right) + \alpha\lVert w\rVert_2^2$ | $ℓ_2$ | `epsilon`（建议 1.35）、`alpha` | L741–L782 |
| `QuantileRegressor` † | $\min_w \frac{1}{n}\sum_i PB_q(y_i - X_iw) + \alpha\lVert w\rVert_1$ | $ℓ_1$ | `quantile`、`alpha` | L784–L798 |

† 这两行的目标函数出自官方 1.8 页的 `Mathematical details` 折叠块——**剪藏里对应的公式是空的**（见 [[sklearn-linear-models]] 的「转换损失与外部核查」第 1、3 条），因此属于外部核查所得，不是剪藏内容。

## 估计器矩阵（discriminant_analysis）

| 估计器 | 生成模型假设 | 决策面 | 能否 `transform` | 关键超参 | 位置 |
| --- | --- | --- | --- | --- | --- |
| `LinearDiscriminantAnalysis` | 类条件高斯，**各类共享** $\Sigma$ | 线性 | 能（$\le \min(n_{\text{classes}}-1,\, n_{\text{features}})$ 维） | `solver`（默认 `'svd'`）、`shrinkage`、`n_components`、`priors`、`covariance_estimator` | [[sklearn-lda-qda]], L22–L26、L67–L93 |
| `QuadraticDiscriminantAnalysis` | 类条件高斯，**逐类** $\Sigma_k$ | 二次 | 不能 | `solver`（默认 `'svd'`，或 `'eigen'`）、`shrinkage`、`reg_param`、`priors`、`covariance_estimator` | [[sklearn-lda-qda]], L50–L58、L97–L103 |
| `GaussianNB`（`sklearn.naive_bayes`，**未入库**） | 类条件高斯且**对角** $\Sigma_k$ | 二次（轴向受限） | 不能 | — | 官方 1.2 页只给等价关系：[[sklearn-lda-qda]], L60–L63 |

三者是嵌套关系（`GaussianNB` ⊂ LDA ⊂ QDA），机制与参数代价见 [[gaussian-discriminant-analysis]]。

求解器能力（[[sklearn-lda-qda]], L115–L119）：

| 求解器 | 适用 | 分类 | `transform` | shrinkage / 自定义协方差估计器 | 显式构造 $\Sigma$ |
| --- | --- | --- | --- | --- | --- |
| `'svd'`（**两者默认**） | LDA、QDA | 是 | LDA | **否** | 否（高维友好） |
| `'lsqr'` | 仅 LDA | 是 | 否 | 是 | 是（解 $\Sigma\omega=\mu$，不求逆） |
| `'eigen'` | LDA、QDA | 是 | LDA | 是（QDA 唯一可用） | 是（高维不适用） |

## 估计器矩阵（naive_bayes）

五个估计器共享同一套“条件独立 + MAP”框架，差别只在 $p(x_i\mid y)$ 的分布（[[sklearn-naive-bayes]], L36）。构造参数出自官方 API 页与 1.9.1 源码（**外部核查**，不在剪藏内），文档位置指回剪藏。

| 估计器 | $p(x_i\mid y)$ | 输入要求 | 构造参数（默认值） | `partial_fit` | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `GaussianNB` | 高斯，$\mu_y,\sigma_y$ 用 MLE | 连续特征 | `priors=None`、`var_smoothing=1e-9` | 支持（自有实现） | L51–L57 |
| `MultinomialNB` | 多项（计数） | 非负计数 / tf-idf | `alpha=1.0`、`fit_prior=True`、`class_prior=None`、`force_alpha=True` | 支持（继承） | L74–L84 |
| `ComplementNB` | 多项，统计量取**类补集** | 同上 | 同上 + `norm=False` | 支持（继承） | L88–L112 |
| `BernoulliNB` | 伯努利（二值） | 0/1，或由 `binarize` 转换 | 同上 + `binarize=0.0` | 支持（继承） | L122–L132 |
| `CategoricalNB` | 逐特征类别分布 | **已编码**为 $0..n_i-1$ | 同上 + `min_categories=None` | 支持（覆写） | L147–L161 |

两点结构性事实：

- **`GaussianNB` 是 QDA 的对角特例**（[[sklearn-lda-qda]], L60–L63），参数 $2Kd+K-1$（[[gaussian-discriminant-analysis]] 的推导）——三者嵌套关系不变。
- **五个都有 `partial_fit`**（外部核查：`_BaseDiscreteNB.partial_fit` 被三个离散类继承，`CategoricalNB` 自行覆写），而文档 L165 仍只列了三类——已知分歧，记录在 [[sklearn-naive-bayes]]。首次调用必须传 `classes`（L167）。

## 会沉默改变结果的默认值

按"踩到才会发现"的顺序排：

1. **logistic 默认带正则**。「Regularization is applied by default」(L440)；要无正则就把 `C` 设到极大。
2. **`class_weight` 与 `C` 耦合**。目标里的 $s_i$ 是 `sample_weight`×`class_weight`，损失项除以 $S$、正则项除以 $SC$；因此「样本权重乘以 $b$ 等价于把 $C$ 乘以 $b$」(L468–L471, L484)。改类别权重就会改有效正则强度。
3. **`Ridge` 默认 `solver="auto"`**，按 `positive=True` → `lbfgs`、非稀疏 → `cholesky`、其余 → `sparse_cg` 的顺序选 (L78–L84)。默认不等于最快的那个。
4. **`RidgeCV` 默认不是 K 折而是 LOO**，且 LOO 公式下 `alpha` 不能取 0；要 K 折必须显式给 `cv` (L122, L137)。
5. **`alpha` 在不同估计器里含义不同**：Ridge 目标没有 $1/n$，Lasso/ElasticNet 有 (L67 vs L146)。跨模型比同一个 `alpha` 无意义。
6. **`liblinear` 会惩罚截距**（文档直接标 "bad"），其余求解器不会 (L499)；此外它在 `fit_intercept=False`、`decision_function` 恰为 0 时预测的类与 sklearn 封装相反 (L505)。
7. **`sag` / `saga` 对未缩放数据不鲁棒**（表格里的 "Robust to unscaled datasets: no"），`lbfgs` 在未良好缩放或稀有 one-hot 上性能下降 (L499, L938)。
8. **`Lasso` 的停止条件是 duality gap 而非系数变化**（"It stops if the duality gap is smaller than the provided tolerance `tol`"，L190）。
9. **`HuberRegressor` 与 `SGDRegressor(loss="huber")` 的 `epsilon` 语义不同**：前者尺度不变，后者在缩放 X/y 后需要重设 (L777–L779)。

LDA / QDA 一侧的默认值陷阱（引用 `[[sklearn-lda-qda]], L…`；标「外部核查」的取自官方 API 页，不在剪藏内）：

10. **`solver` 默认 `'svd'`，而 `svd` 不支持 `shrinkage`**。文档只说“shrinkage 只在 `lsqr`/`eigen` 下有效”（L97），实现层直接抛 `NotImplementedError`（外部核查：`sklearn/discriminant_analysis.py`，1.9.X 分支）——必须显式换成 `'lsqr'` 或 `'eigen'`；QDA 只有 `'eigen'`。
11. **`shrinkage` 与 `covariance_estimator` 互斥**：两者都作用在同一个协方差估计上，文档要求另一个留 `None`（L103；外部核查 API 页同述）。
12. **`n_components` 对预测毫无影响**（L26）：它只改 `transform` 的输出，因此“调 `n_components` 提升准确率”是无效操作。
13. **`priors` 默认取训练集类比例**（外部核查）：过采样/欠采样后 fit 会连带移动决策阈值，正文完全没提这个参数。
14. **OAS 与 Ledoit-Wolf 是两个可换的协方差估计器**，不是同一个东西：`shrinkage='auto'` 用 Ledoit-Wolf；要 OAS 得走 `covariance_estimator=OAS()`（L101–L103）。
15. **QDA 的 `reg_param` 与 `shrinkage` 是两个旋钮**（外部核查）：收缩终点分别是单位阵与对角方差，默认都是“不收缩”（`None` / `0.0`）。
16. **`store_covariance` 默认 `False`**（外部核查）：LDA 在 `svd` 下 `covariance_` 不存在，QDA 任何情况下都要显式开启——想检查协方差形状时容易 AttributeError。

朴素贝叶斯一侧（引用 `[[sklearn-naive-bayes]], L…`；标「外部核查」的取自 API 页与 `sklearn/naive_bayes.py` @ 1.9.1）：

17. **`alpha` 默认 `1.0`，即 Laplace 平滑**。文档定义了 Laplace/Lidstone 两种名字（L84）但没说默认就是前者；`alpha=0` 是合法的（`force_alpha=True`），会真的产生零概率。
18. **`fit_prior=True` 默认从数据学类先验**（外部核查），与文档 L34 的“类比例”一致——所以重采样后再 `fit`，决策阈值会跟着漂。这与 LDA/QDA 的 `priors` 是同一个陷阱。
19. **`GaussianNB.var_smoothing=1e-9` 给方差加下限**（外部核查）：`epsilon_ = var_smoothing × max(var(X, axis=0))` 被加到每个方差上。文档完全不提这个参数，却默认开着——小样本 / 低方差特征上会改变结果。
20. **`BernoulliNB(binarize=0.0)` 默认把 >0 的值二值化**（外部核查）。`binarize=None` 才表示“输入已二值”；把 tf-idf 或计数直接喂进去会被静默压成 0/1，从而变成与 `MultinomialNB` 完全不同的模型。
21. **`ComplementNB(norm=False)` 默认不做权重归一化**（外部核查），而文档 L103 的三步公式里包含按 $\sum_j\lvert w_{cj}\rvert$ 归一化——按文档实现和用默认参数跑不是同一件事。
22. **`CategoricalNB` 要求每个特征已编码为 $0..n_i-1$**（L161），且 `min_categories=None`（外部核查）时类别数从训练集推断；未见类别与编码不一致的行为文档未说明。

## 版本边界

- `linear_model` 部分对应**官方 1.8–1.9.1 的 stable 文档**。1.9 重写了 1.1.3.1 gap safe screening 的正文，1.7.2 及更早没有这一小节 → 引用该小节时应注明版本。
- `discriminant_analysis` 部分对应 **1.9.x**：1.9 把 QDA 的 SVD 推导从 $\frac{1}{n-1}X_k^{T}X_k$ 改为 $\frac{1}{n}X_k^{T}X_k$（与实现的最大似然估计一致），1.8 及更早仍是 $\frac{1}{n-1}$。两份剪藏因此可互证版本：1.1 判到 ≥1.8、1.2 判到 1.9.x（见 [[sklearn-lda-qda]] 的版本判定表）。
- `naive_bayes` 部分判为 **1.9.x，但证据弱**：1.8→1.9.1 在本节只改了 `one dimensional` → `one-dimensional`、`through \(x_n\), :` → `through \(x_n\):` 两处措辞。功能层面两版无差别，因此上面 17–22 条的默认值结论不依赖这条判定。
- 剪藏里 logistic 的惩罚只通过 `C` 与 `l1_ratio` 表达（L473），没有出现 `penalty=` 参数的说明；这与旧版本 docstring 的写法不同。**本文不对 `penalty=` 的弃用状态下结论**，需要时查目标版本的 API 文档。
- `np.float64(...)` 形式的 doctest 输出（L84）要求 NumPy ≥ 2 的 repr 习惯，与代码行为无关。

## 使用边界

- 本页只覆盖线性模型、判别分析与朴素贝叶斯。缺失值、编码、缩放、Pipeline、CV 切分、指标都在未入库的章节里（见上面覆盖表）。特征工程与向量化（8.2/8.3）未读，所以本页**不提供** CountVectorizer / TfidfVectorizer 的用法。
- 1.9 节的用户指南**不覆盖** `alpha`、`fit_prior`、`class_prior`、`var_smoothing`、`binarize`、`norm`、`min_categories` 七个参数；本页第 17–22 条全部来自 API 页或源码（外部核查），不是剪藏论述。
- 1.2 节的用户指南**不覆盖** `priors`、`reg_param`、`store_covariance`、`tol` 四个参数；本页第 13、15、16 条与估计器矩阵中这些项全部来自官方 API 页（外部核查），不是剪藏的论述（第 14 条的两个估计器来自 L101–L103）。
- 官方页有 29.3% 的正文在折叠块里，剪藏全缺（详见 [[sklearn-linear-models]] 的「转换损失与外部核查」）。本页因此**不声称穷尽了文档的所有约束**：`Solvers’ details`、`Practical considerations` 里的细则（如 `l1_min_c`、GLM 的 exposure 用法）本页没有。
- 稳健回归的取舍（RANSAC / Theil-Sen / Huber）不宜按一句话规则选，见 [[linear-models-as-baselines]] 里由官方示例图读数得到的反例。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]
- 实践：[[linear-models-as-baselines]]
- 概念：[[regularization-and-sparsity]]、[[gaussian-discriminant-analysis]]、[[naive-bayes]]、[[leaderboard-vs-cv]]
- 对照来源：[[kaggle-getting-started]]（2016 年的 sklearn 用法，多数 API 已移除）
