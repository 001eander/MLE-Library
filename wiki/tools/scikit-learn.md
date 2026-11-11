---
title: "scikit-learn（工具页）：linear_model 模块的估计器矩阵、默认值与版本边界"
type: tool
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - linear-model
  - tooling
sources:
  - "[[sklearn-linear-models]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.x – 1.9.x（据官方 1.8.0 / 1.9.1 页比对推断，剪藏未记录版本）
  - 本文只覆盖 `sklearn.linear_model` 一个模块
---

# scikit-learn（工具页）

本页记录 scikit-learn 的**当前行为**：模块里有哪些估计器、默认值是什么、哪些默认值会沉默地改变结果。事实全部来自 [[sklearn-linear-models]]（官方用户指南 1.1 节），引用写作 `[[sklearn-linear-models]], L146`。

## 收录范围与覆盖状态

`raw/articles/sklearn/` 下现有 18 篇同批剪藏。本页只对已入库的模块负责：

| 文档 | 模块／主题 | 状态 |
| --- | --- | --- |
| 1.1 Linear Models | `sklearn.linear_model` | **已入库**（本页 + [[linear-models-as-baselines]] + [[regularization-and-sparsity]]） |
| 1.2 LDA / QDA | `sklearn.discriminant_analysis` | 未入库 |
| 1.4 Support Vector Machines | `sklearn.svm` | 未入库 |
| 1.6 Nearest Neighbors | `sklearn.neighbors` | 未入库 |
| 1.9 Naive Bayes | `sklearn.naive_bayes` | 未入库 |
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

## 版本边界

- 本页描述对应**官方 1.8–1.9.1 的 stable 文档**。1.9 重写了 1.1.3.1 gap safe screening 的正文，1.7.2 及更早没有这一小节 → 引用该小节时应注明版本。
- 剪藏里 logistic 的惩罚只通过 `C` 与 `l1_ratio` 表达（L473），没有出现 `penalty=` 参数的说明；这与旧版本 docstring 的写法不同。**本文不对 `penalty=` 的弃用状态下结论**，需要时查目标版本的 API 文档。
- `np.float64(...)` 形式的 doctest 输出（L84）要求 NumPy ≥ 2 的 repr 习惯，与代码行为无关。

## 使用边界

- 本页只覆盖线性模型。缺失值、编码、缩放、Pipeline、CV 切分、指标都在未入库的章节里（见上面覆盖表）。
- 官方页有 29.3% 的正文在折叠块里，剪藏全缺（详见 [[sklearn-linear-models]] 的「转换损失与外部核查」）。本页因此**不声称穷尽了文档的所有约束**：`Solvers’ details`、`Practical considerations` 里的细则（如 `l1_min_c`、GLM 的 exposure 用法）本页没有。
- 稳健回归的取舍（RANSAC / Theil-Sen / Huber）不宜按一句话规则选，见 [[linear-models-as-baselines]] 里由官方示例图读数得到的反例。

## 关联页面

- 来源：[[sklearn-linear-models]]
- 实践：[[linear-models-as-baselines]]
- 概念：[[regularization-and-sparsity]]、[[leaderboard-vs-cv]]
- 对照来源：[[kaggle-getting-started]]（2016 年的 sklearn 用法，多数 API 已移除）
