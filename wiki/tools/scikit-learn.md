---
title: "scikit-learn（工具页）：linear_model / discriminant_analysis / naive_bayes / svm / neighbors / tree / ensemble / feature_selection / neural_network / manifold / cluster / decomposition / covariance 与异常检测面、model_selection 切分器与搜索器、metrics 评分入口、preprocessing 变换器的矩阵、默认值与版本边界"
type: tool
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - linear-model
  - discriminant-analysis
  - naive-bayes
  - svm
  - nearest-neighbors
  - decision-tree
  - ensemble
  - feature-selection
  - neural-network
  - mlp
  - manifold-learning
  - dimensionality-reduction
  - clustering
  - decomposition
  - covariance
  - cross-validation
  - model-selection
  - preprocessing
  - feature-scaling
  - categorical-encoding
  - target-encoding
  - discretization
  - hyperparameter-tuning
  - grid-search
  - randomized-search
  - successive-halving
  - metrics
  - scoring
  - evaluation
  - decision-threshold
  - proper-scoring-rule
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - isolation-forest
  - local-outlier-factor
  - one-class-svm
  - shrinkage
  - precision-matrix
  - pca
  - nmf
  - ica
  - topic-model
  - unsupervised
  - gradient-boosting
  - random-forest
  - tooling
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
  - "[[sklearn-manifold-learning]]"
  - "[[sklearn-clustering]]"
  - "[[sklearn-decomposition]]"
  - "[[sklearn-covariance]]"
  - "[[sklearn-outlier-detection]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-hyperparameter-tuning]]"
  - "[[sklearn-metrics]]"
  - "[[sklearn-preprocessing]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x（linear_model：据官方 1.8.0 / 1.9.1 页比对推断，剪藏未记录版本）
  - scikit-learn 1.9.x（discriminant_analysis：QDA 协方差公式写 1/n，1.8 及更早写 1/(n-1)，见 [[sklearn-lda-qda]]）
  - scikit-learn 1.9.x（naive_bayes：仅两个措辞级证据，推断，见 [[sklearn-naive-bayes]] 的版本判定）
  - scikit-learn 1.7.x – 1.9.1（svm：1.7/1.8/1.9.1 三版该节字面相同，剪藏与 1.9.1 逐句一致，见 [[sklearn-svm]] 的版本判定）
  - scikit-learn 1.7.x – 1.9.x（neighbors：该节 1.7/1.8/1.9 正文无实质差异，见 [[sklearn-nearest-neighbors]]；默认值与 `auto` 规则另在 1.6.1 实测，与 1.9.1 一致）
  - scikit-learn 1.9.x（tree：1.8→1.9 有两处文本改动可判定，且 `/stable/` 与 1.9.1 页逐行相同，见 [[sklearn-decision-trees]]）
  - scikit-learn 1.9.1（ensemble：138/173 条句子探针在 1.9.1 页全部命中，1.8 缺 Newton boosting 句，见 [[sklearn-ensembles]] 的版本判定；默认值取自 1.9.1 签名）
  - scikit-learn 1.9.1（feature_selection：四条 1.6→1.9 的措辞判据，见 [[sklearn-feature-selection]] 的版本判定；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测）
  - scikit-learn 1.8.0 – 1.9.1（neural_network：1.7→1.8 有公式括号改动、1.8→1.9 正文无 diff，见 [[sklearn-neural-networks]] 的版本判定；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测）
  - scikit-learn 1.8.0 – 1.9.1（manifold：1.7→1.8 新增 ClassicalMDS 段、1.8→1.9 正文 diff 为空，见 [[sklearn-manifold-learning]] 的版本判定；`MDS` 的默认值在 1.8/1.9 之间变过，必须带版本）
  - scikit-learn 1.9.1（cluster：126 条句子探针在 1.6/1.7/1.8/1.9 的命中数是 107/108/110/111，且剪藏含 1.9 才有的 "assigning" 措辞，见 [[sklearn-clustering]] 的版本判定；默认值取自 1.9.1 签名，行为在 1.6.1 实测）
  - scikit-learn 1.9.1（decomposition：1.8→1.9 的唯一正文差异「SVMs」→「kernel approximation」已出现在剪藏 L69，见 [[sklearn-decomposition]] 的版本判定；默认值取自 1.9.1 签名，行为在 1.6.1 实测）
  - scikit-learn 1.7.0 – 1.9.1（covariance：1.6→1.7 四道措辞判据全部命中，1.7→1.8→1.9 正文 diff 为空，见 [[sklearn-covariance]] 的版本判定；默认值取自 1.9.1 签名，1.6.1 实测签名一致）
  - scikit-learn 1.9.1（异常检测面：剪藏含 1.9 新增的 `contamination`/`offset_` 句，1.8→1.9 的正文 diff 只有这一处；`/stable/` 与 `/1.9/` 字节相同，见 [[sklearn-outlier-detection]] 的版本判定；默认值取自 1.9.1 签名，行为在 1.6.1 实测）
  - scikit-learn 1.7.0 – 1.9.1（`model_selection`：133 条句子探针在 1.6/1.7/1.8/1.9 的命中数是 101/122/122/122，且 1.7→1.8→1.9 正文 diff 为空；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测，见 [[sklearn-cross-validation]] 的版本判定）
  - scikit-learn 1.7.0 – 1.9.1（`model_selection` 搜索器：50 条句子探针在四版全命中，判别性来自 1.9 节的「see Which scoring function should I use?」（1.6 无）、1.7→1.8 正文 diff 为空、1.8→1.9 的唯一差异落在剪藏已丢失的 3.2.5 节；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测，见 [[sklearn-hyperparameter-tuning]] 的版本判定）
  - scikit-learn 1.8.0 – 1.9.1（`metrics`：唯一判别证据是脚注「with one-hot encoded targets」（1.6/1.7 无此半句），覆盖区 1.8→1.9 正文逐句 diff 为空；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测，见 [[sklearn-metrics]] 的版本判定）
  - scikit-learn 1.9.1（`preprocessing`：七条判别性证据——小节编号 8.3 排除 1.6/1.7/1.8、`This diagram shows` 与三句 TargetEncoder 措辞排除 1.7、pandas repr `Categories (5, str)` 排除 1.8；`/stable/` 与 `/1.9/` 字节相同，见 [[sklearn-preprocessing]] 的版本判定。**本次在 1.9.1 上实测**，1.6.1 并列给出差异）
  - 本文只覆盖 `sklearn.linear_model`、`sklearn.discriminant_analysis`、`sklearn.naive_bayes`、`sklearn.svm`、`sklearn.neighbors`、`sklearn.tree`、`sklearn.ensemble`、`sklearn.feature_selection`、`sklearn.neural_network`、`sklearn.manifold`、`sklearn.cluster`、`sklearn.decomposition`、`sklearn.covariance` 与 `sklearn.preprocessing` 十四个模块，外加跨模块的**异常检测面**（2.7 节：`svm` / `linear_model` / `covariance` / `ensemble` / `neighbors` 五个入口）、`sklearn.model_selection` 的**切分器面**（3.1 节）与**搜索器面**（3.2 节：`GridSearchCV` / `RandomizedSearchCV` / `Halving*SearchCV`）与 `sklearn.metrics` 的**评分入口面**（3.4 节：估计器 `score` / `scoring=` / `metrics` 函数 / `make_scorer`）与 `sklearn.preprocessing` 的**变换器面**（8.3 节：16 个变换器 + 9 个函数级入口）
---

# scikit-learn（工具页）

本页记录 scikit-learn 的**当前行为**：模块里有哪些估计器、默认值是什么、哪些默认值会沉默地改变结果。事实来自 [[sklearn-linear-models]]（官方用户指南 1.1 节）、[[sklearn-lda-qda]]（1.2 节）、[[sklearn-naive-bayes]]（1.9 节）、[[sklearn-svm]]（1.4 节）、[[sklearn-nearest-neighbors]]（1.6 节）、[[sklearn-decision-trees]]（1.10 节）、[[sklearn-ensembles]]（1.11 节）、[[sklearn-feature-selection]]（1.13 节）、[[sklearn-neural-networks]]（1.17 节）与 [[sklearn-manifold-learning]]（2.2 节）、[[sklearn-clustering]]（2.3 节）、[[sklearn-decomposition]]（2.5 节）、[[sklearn-covariance]]（2.6 节）、[[sklearn-outlier-detection]]（2.7 节）、[[sklearn-cross-validation]]（3.1 节）、[[sklearn-hyperparameter-tuning]]（3.2 节）、[[sklearn-metrics]]（3.4 节）、[[sklearn-preprocessing]]（8.3 节），引用写作 `[[sklearn-linear-models]], L146`。

## 收录范围与覆盖状态

`raw/articles/sklearn/` 下现有 18 篇同批剪藏，**18 篇已全部入库**（2026-11-12 补齐 8.3 Preprocessing data）。本页只对已入库的模块负责：

| 文档 | 模块／主题 | 状态 |
| --- | --- | --- |
| 1.1 Linear Models | `sklearn.linear_model` | **已入库**（本页 + [[linear-models-as-baselines]] + [[regularization-and-sparsity]]） |
| 1.2 LDA / QDA | `sklearn.discriminant_analysis` | **已入库**（本页 + [[gaussian-discriminant-analysis]]，来源页 [[sklearn-lda-qda]]） |
| 1.4 Support Vector Machines | `sklearn.svm` | **已入库**（本页 + [[support-vector-machines]] + [[svm-selection-and-tuning]]，来源页 [[sklearn-svm]]） |
| 1.6 Nearest Neighbors | `sklearn.neighbors` | **已入库**（本页 + [[nearest-neighbors]] + [[nearest-neighbors-selection-and-tuning]]，来源页 [[sklearn-nearest-neighbors]]） |
| 1.9 Naive Bayes | `sklearn.naive_bayes` | **已入库**（本页 + [[naive-bayes]]，来源页 [[sklearn-naive-bayes]]） |
| 1.10 Decision Trees | `sklearn.tree` | **已入库**（本页 + [[decision-trees]] + [[decision-trees-selection-and-tuning]]，来源页 [[sklearn-decision-trees]]） |
| 1.11 Ensembles | `sklearn.ensemble` | **已入库**（本页 + [[ensembles]] + [[ensembles-selection-and-tuning]]，来源页 [[sklearn-ensembles]]） |
| 1.13 Feature selection | `sklearn.feature_selection` | **已入库**（本页 + [[feature-selection]] + [[feature-selection-workflow]]，来源页 [[sklearn-feature-selection]]） |
| 1.17 Neural network models | `sklearn.neural_network` | **已入库**（本页 + [[multi-layer-perceptron]] + [[mlp-selection-and-tuning]]，来源页 [[sklearn-neural-networks]]） |
| 2.2 Manifold learning | `sklearn.manifold` | **已入库**（本页 + [[manifold-learning]] + [[manifold-learning-selection-and-tuning]]，来源页 [[sklearn-manifold-learning]]） |
| 2.3 Clustering | `sklearn.cluster` | **已入库**（本页 + [[clustering]] + [[clustering-selection-and-tuning]]，来源页 [[sklearn-clustering]]） |
| 2.5 Decomposing signals in components | `sklearn.decomposition` | **已入库**（本页 + [[matrix-factorization]] + [[decomposition-selection-and-tuning]]，来源页 [[sklearn-decomposition]]） |
| 2.6 Covariance estimation | `sklearn.covariance` | **已入库**（本页 + [[covariance-estimation]] + [[covariance-estimation-selection-and-tuning]]，来源页 [[sklearn-covariance]]） |
| 2.7 Novelty and Outlier Detection | 异常检测面（`sklearn.svm` / `linear_model` / `covariance` / `ensemble` / `neighbors`） | **已入库**（本页 + [[anomaly-detection]] + [[outlier-detection-selection-and-tuning]]，来源页 [[sklearn-outlier-detection]]） |
| 3.1 Cross-validation | `sklearn.model_selection` | **已入库**（本页 + [[cross-validation]] + [[cross-validation-design-and-tuning]]，来源页 [[sklearn-cross-validation]]） |
| 3.2 Tuning hyper-parameters | `sklearn.model_selection` 搜索器 | **已入库**（本页 + [[hyperparameter-search]] + [[hyperparameter-search-and-tuning]]，来源页 [[sklearn-hyperparameter-tuning]]） |
| 3.4 Metrics and scoring | `sklearn.metrics` | **部分入库**（方法论 4 小节；官方 54 小节中的其余 50 个——全部具体指标定义——不在剪藏内）→ 本页 + [[evaluation-metrics]] + [[metric-selection-and-thresholding]]，来源页 [[sklearn-metrics]] |
| 8.3 Preprocessing data | `sklearn.preprocessing` | **已入库**（本页 + [[feature-preprocessing]] + [[preprocessing-pipeline-design]]，来源页 [[sklearn-preprocessing]]；缺 11 个折叠块与 5 条文献，机制层须引官方 1.9.1 页） |

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
| `GaussianNB`（`sklearn.naive_bayes`，**已入库**） | 类条件高斯且**对角** $\Sigma_k$ | 二次（轴向受限） | 不能 | `var_smoothing`、`priors` | 等价关系：[[sklearn-lda-qda]], L60–L63；参数见 [[sklearn-naive-bayes]] |

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

## 估计器矩阵（svm）

`sklearn.svm` 里是**两套实现**（libsvm 与 liblinear，[[sklearn-svm]], L396），不是同一模型的后端。默认值来自 `sklearn/svm/_classes.py` @ 1.9.1（**外部核查**）；“文档是否提及”指剪藏 [[sklearn-svm]]。

| 估计器 | 实现 | 目标／损失 | 关键超参（默认值） | 多类策略 | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `SVC` | libsvm | hinge（原问题）+ 松弛 | `C=1.0`、`kernel='rbf'`、`gamma='scale'`、`degree=3`、`coef0=0.0`、`probability`**（1.9 弃用）**、`cache_size=200`、`decision_function_shape='ovr'`、`break_ties=False` | 内部 ovo，接口 ovr | L28–L65、L75 |
| `NuSVC` | libsvm | ν 参数化（与 SVC 等价） | `nu=0.5`，其余同 `SVC`；`class_weight` 见下方说明 | 内部 ovo，接口 ovr | L146、L356–L358 |
| `LinearSVC` | liblinear | `squared_hinge` + `ℓ2`/`ℓ1` 惩罚 | `penalty='l2'`、`loss='squared_hinge'`、`dual='auto'`、`C=1.0`、`intercept_scaling=1`、`multi_class='ovr'` | 真 ovr | L32、L92、L346–L352 |
| `SVR` | libsvm | ε-insensitive | `C=1.0`、`epsilon=0.1`、`kernel='rbf'`、`gamma='scale'` | — | L362–L384 |
| `NuSVR` | libsvm | ν 参数化 | `nu=0.5`、`C=1.0` | — | L165 |
| `LinearSVR` | liblinear | ε-insensitive | `epsilon=0.0`、`C=1.0`、`dual='auto'` | — | L165、L386–L392 |
| `OneClassSVM` | libsvm | 单类（无监督） | `nu=0.5`、`kernel='rbf'`、`gamma='scale'` | — | L186–L188 |

三点结构性事实：

- **`SVC(kernel='linear')` ≠ `LinearSVC`**：前者 hinge、截距**不进**正则；后者 `squared_hinge`、截距**进**正则（可选 `intercept_scaling` 削弱）。同一 `C` 下结果不同，官方直接警告（[[sklearn-svm]], L32）。
- **`LinearSVC`/`LinearSVR` 只支持线性核**，因为它们的原问题不含样本内积，核技巧用不上（L352、L392）。
- **三个 `dual_coef_` 语义各不相同**：分类存 $y_i\alpha_i$（L341），回归存 $\alpha_i-\alpha_i^*$（L384），且多类 ovo 下按列（支持向量）× 行（其余类）排布（L113）。
- **`NuSVC` 的 `class_weight` 有分歧**：文档说“SVC（but not NuSVC）实现了 `class_weight`”（L146），但 1.9.1 的源码里 `NuSVC` 签名接受该参数并在 `_validate_targets` 计入 `class_weight_`（外部核查，1.6.1 起已如此）。本页不对此下结论，详见 [[sklearn-svm]] 的分歧节第 2 条。

## 估计器矩阵（neighbors）

`sklearn.neighbors` 与前面四个模块不同：这里的估计器**不拟合参数**，只存训练集；「超参」调的是**邻域的定​​义**（$k$ 或 $r$、权重、度量）和**搜索结构**（后者只改速度）。行号指向 [[sklearn-nearest-neighbors]]；默认值来自 1.9.1 签名，并已在 1.6.1 本机实测复核（**外部核查**，见该页转换损失 E 节）。

| 估计器 | 任务 | 邻域定义 | 关键超参（默认值） | 输出形态 | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `NearestNeighbors` | 只找邻居 | 固定 $k$ | `n_neighbors=5`、`algorithm='auto'`、`leaf_size=30`、`metric='minkowski'`、`p=2`、`radius=1.0`、`n_jobs=None` | `kneighbors` / `kneighbors_graph` | L20–L25、L27–L67 |
| `KNeighborsClassifier` | 分类 | 固定 $k$ | `n_neighbors=5`、`weights='uniform'`、其余同上 | 多数投票；`predict_proba` ＝加权投票占比 | L97–L113 |
| `RadiusNeighborsClassifier` | 分类 | 固定半径 $r$ | `radius=1.0`、`outlier_label=None` | 空邻域时报 `ValueError` | L101–L105 |
| `KNeighborsRegressor` | 回归 | 固定 $k$ | 同 `KNeighborsClassifier` | 邻域均值（支持多输出） | L115–L132 |
| `RadiusNeighborsRegressor` | 回归 | 固定半径 $r$ | 同 `RadiusNeighborsClassifier` | 邻域均值 | L115–L119 |
| `NearestCentroid` | 分类 | 到类质心的距离 | `metric='euclidean'`、`shrink_threshold=None`、`priors='uniform'` | 线性边界；`deviations_` 可读 | L166–L192 |
| `KNeighborsTransformer` | 变换 | 固定 $k$ | `mode='distance'`、`n_neighbors=5` | 稀疏邻居图（可缓存） | L194–L236 |
| `RadiusNeighborsTransformer` | 变换 | 固定半径 | 同 `RadiusNeighborsClassifier` + `mode` | 稀疏邻居图 | L198 |
| `NeighborhoodComponentsAnalysis` | 度量学习 / 降维 | —（学 $L$） | `n_components=None`（＝不降维）、`max_iter=50`、`tol=1e-5`、`init='auto'` | `transform` → $LX^{T}$；配 KNN 用 | L238–L320 |
| `KDTree` / `BallTree` | 底层索引 | 固定 $k$ | `leaf_size=30`；`BallTree` 另支持 `pyfunc` | 直接 `query` | L69–L95 |

五点结构性事实：

- **只有两个可用度量面**：`KDTree.valid_metrics` 9 个（Minkowski 家族）、`BallTree.valid_metrics` 24 个（多出 `mahalanobis`、`hamming`、`haversine` 等），后者含 `pyfunc`（L91、L93）。`NearestCentroid` 自 1.5 起只接受 `'euclidean'`/`'manhattan'`（实测：其余报 `InvalidParameterError`）。
- **`algorithm` 与 `leaf_size` 不改预测**：三种后端算同一个对象（L136–L156）。这是本库唯一能干净分开「实现选择」与「模型选择」的地方。
- **`n_neighbors` 同时是模型参数与性能开关**：$k \ge N/2$ 时 `auto` 直接选暴力（官方 1.9 页丢失的折叠块；实测 1.6.1 同）。
- **`NearestCentroid.priors`** 是 1.6 新增参数（docstring `versionadded:: 1.6`），默认 `'uniform'`，但 docstring 紧接着写「By default, the class proportions are inferred from the training data」——**自相矛盾**（1.6.1 实测，1.9.1 docstring 同），详见 [[sklearn-nearest-neighbors]] 分歧节。
- **`KNeighborsTransformer(mode='distance')` 会多算 1 个邻居**（L232；实测：请求 3 个时实际报 `n_neighbors = 4`），自定义 ANN 变换器必须跟这个约定。

## 估计器矩阵（tree）

`sklearn.tree` 的参数结构与前五个模块不同：**拟合产生参数，但容量在拟合前就由结构旋钮锁定了**。行号指向 [[sklearn-decision-trees]]；默认值取自 1.9.1 API 页签名（**外部核查**），部分行为在 1.6.1 实测。

| 估计器 | 任务 | 切分策略 | 关键超参（默认值） | 叶子输出 | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `DecisionTreeClassifier` | 分类 | 贪心穷举（`'best'`） | `criterion='gini'`、`max_depth=None`、`min_samples_split=2`、`min_samples_leaf=1`、`min_weight_fraction_leaf=0.0`、`max_features=None`、`max_leaf_nodes=None`、`min_impurity_decrease=0.0`、`class_weight=None`、`ccp_alpha=0.0`、`monotonic_cst=None` | 类比例（`predict_proba`） | L36–L132 |
| `DecisionTreeRegressor` | 回归 | 同上 | `criterion='squared_error'`，其余同上（**无 `class_weight`**） | 均值 / 中位数（按准则） | L134–L154、L292–L316 |
| `ExtraTreeClassifier` | 分类 | 每特征**一个随机阈值**（`'random'`） | `splitter='random'`、**`max_features='sqrt'`**，其余同上 | 同上 | L260、L361–L366 |
| `ExtraTreeRegressor` | 回归 | 同上 | `splitter='random'`、**`max_features=1.0`**，其余同上 | 同上 | L260、L361–L366 |

六点结构性事实：

- **`criterion` 的取值集合在 1.9 缩过一次**：`friedman_mse` 于 1.9 弃用（外部核查：API 页 “Changed in version 1.9”），回归准则一节只剩 `squared_error` / `poisson` / `absolute_error`（L296–L316）。
- **`splitter='random'` 不是“随机选特征”**：它给每个特征采一个随机阈值（L258）；限制特征数的是另一个参数 `max_features`（剪藏 grep 为 0）。
- **`ExtraTree*` 与 `DecisionTree*` 共享同一实现**，差别只在默认参数（L260，1.9 新增句）；其中最要命的是 `max_features`（`'sqrt'` / `1.0` vs `None`）。
- **缺失值不是插补**：训练时每个阈值同时评估 NaN 走左/走右（L324）；预测三条规则见 [[decision-trees-selection-and-tuning]] 第六节。哪些准则支持 NaN **随版本变化**（1.9 起含 `absolute_error`，外部核查）。
- **`ccp_alpha=0.0` 等于默认不剪枝**（外部核查 API 页）；剪枝用**终端节点的样本加权不纯度**而不是误分类率（L376）。
- **内部一律 `float32`**（L224）：阈值被 float32 舍入，相邻高精度值会塌成同一个候选（本机实测，见下第 47 条）。

## 估计器矩阵（ensemble）

`sklearn.ensemble` 是本库目前**估计器最多**的模块，且它不引入新的假设形式——所有估计器都是「把基学习器组合起来」的策略。行号指向 [[sklearn-ensembles]]；默认值取自 1.9.1 源码签名（**外部核查**），行为在 1.6.1 实测。

| 估计器 | 组合方式 | 关键超参（默认值） | 文档位置 |
| --- | --- | --- | --- |
| `HistGradientBoostingClassifier` | 直方图 GBDT（Newton boosting） | `loss='log_loss'`、`learning_rate=0.1`、`max_iter=100`、`max_leaf_nodes=31`、`max_depth=None`、`min_samples_leaf=20`、`l2_regularization=0.0`、`max_features=1.0`、`max_bins=255`、`categorical_features='from_dtype'`、`monotonic_cst=None`、`interaction_cst=None`、`early_stopping='auto'`、`scoring='loss'`、`validation_fraction=0.1`、`n_iter_no_change=10`、`tol=1e-7`、`class_weight=None` | L20–L261 |
| `HistGradientBoostingRegressor` | 同上（回归） | `loss='squared_error'`、`quantile=None`，其余同分类器（无 `class_weight`） | L20–L261 |
| `GradientBoostingClassifier` | 精确切分 GBDT（一阶梯度 + 线搜索） | `loss='log_loss'`、`learning_rate=0.1`、`n_estimators=100`、`subsample=1.0`、`criterion='deprecated'`（1.9 弃用，1.11 移除）、`min_samples_split=2`、`min_samples_leaf=1`、`max_depth=3`、`max_features=None`、`max_leaf_nodes=None`、`n_iter_no_change=None`、`tol=1e-4`、`ccp_alpha=0.0` | L263–L414 |
| `GradientBoostingRegressor` | 同上（回归） | `loss='squared_error'`、`alpha=0.9`（huber/quantile），其余同上 | L263–L414 |
| `RandomForestClassifier` | bootstrap + 每节点特征子集，平均概率 | `n_estimators=100`、`criterion='gini'`、`max_depth=None`、`max_features='sqrt'`、`bootstrap=True`、`oob_score=False`、`class_weight=None`、`max_samples=None` | L432–L454 |
| `RandomForestRegressor` | 同上（回归） | `criterion='squared_error'`、`max_features=1.0`，其余同分类器（无 `class_weight`） | L432–L454 |
| `ExtraTreesClassifier` | 阈值也随机的森林 | 同 RF 分类，但 **`bootstrap=False`**、`max_features='sqrt'` | L456–L489 |
| `ExtraTreesRegressor` | 同上（回归） | 同 RF 回归，但 **`bootstrap=False`**、`max_features=1.0` | L456–L489 |
| `RandomTreesEmbedding` | 无监督：叶子索引的 one-of-K 编码 | `n_estimators=100`、`max_depth=5`、`sparse_output=True` | L528–L541 |
| `BaggingClassifier` / `BaggingRegressor` | 任意基模型的样本/特征子集集成 | `estimator=None`（→`DecisionTree*`）、`n_estimators=10`、`max_samples=None`（→1.0）、`max_features=1.0`、`bootstrap=True`、`bootstrap_features=False`、`oob_score=False` | L572–L596 |
| `VotingClassifier` | 硬投票 / 软投票 | `voting='hard'`、`weights=None`、`flatten_transform=True`、`n_jobs=None` | L598–L693 |
| `VotingRegressor` | 预测值平均 | `weights=None`、`n_jobs=None` | L695–L725 |
| `StackingClassifier` | 折外预测 + 元分类器 | `final_estimator=None`（→`LogisticRegression`）、`cv=None`（→5 折）、`stack_method='auto'`、`passthrough=False` | L727–L829 |
| `StackingRegressor` | 折外预测 + 元回归器 | `final_estimator=None`（→`RidgeCV`）、`cv=None`、`passthrough=False` | L727–L829 |
| `AdaBoostClassifier` | 重加权 boosting（SAMME） | `estimator=None`（→`DecisionTreeClassifier(max_depth=1)`）、`n_estimators=50`、`learning_rate=1.0` | L831–L866 |
| `AdaBoostRegressor` | 重加权 boosting（R2） | `estimator=None`（→`DecisionTreeRegressor(max_depth=3)`）、`n_estimators=50`、`learning_rate=1.0`、`loss='linear'` | L831–L866 |

五点结构性事实：

- **两个 GBDT 实现不是同一个模型的不同后端**：HGBT 支持缺失值/类别/单调约束/交互约束与 Newton boosting，`GradientBoosting*` 都不支持（L76–L257）。“小样本时旧实现可能更准”写在官方 `aside` 主题框里，**剪藏丢失**（见 [[sklearn-ensembles]] 转换损失 A1）。
- **`max_features` 在 RF 与 Bagging 里同名不同义**：RF 每**节点**抽（L436），Bagging 每**基模型**抽一次（L583）。
- **RF/ET 的聚合是概率平均，不是投票**（L442；实测 1.6.1 差值 0.0）。因此 `predict` = 概率 argmax，平票按概率和而非票数。
- **Stacking 的 `cv` 是训练机制**，不是评估参数：内部 `cross_val_predict` 生成折外预测（L769），成本 ≈ 单层 ×(K+1)。
- **AdaBoost 的默认弱学习器不对称**：分类器是 stump，回归器是 `max_depth=3` 的树（外部核查 + 实测）；正文只写「默认 stump」（L860）。

## 变换器矩阵（feature_selection）

`sklearn.feature_selection` 与本页前面七个模块不同：它的输出仍然是**原始列的子集**，因此全部是变换器（`transform`），不是预测器。行号指向 [[sklearn-feature-selection]]；默认值取自 1.9.1 API 页签名（**外部核查**），行为在 1.6.1 实测。

| 变换器 | 家族 | 目标函数 | 关键超参（默认值） | 文档位置 |
| --- | --- | --- | --- | --- |
| `VarianceThreshold` | 过滤式（无监督） | 每列方差 | `threshold=0.0`（只删零方差列） | L12–L37 |
| `SelectKBest` | 过滤式 | 每列打分函数值 | `score_func=f_classif`、`k=10` | L39–L78 |
| `SelectPercentile` | 过滤式 | 同上 | `score_func=f_classif`、`percentile=10` | L39–L78 |
| `SelectFpr` / `SelectFdr` / `SelectFwe` | 过滤式（多重检验） | p 值 + 校正方式 | `score_func=f_classif`、`alpha=0.05` | L45 |
| `GenericUnivariateSelect` | 过滤式 | 可配置策略 | `score_func=f_classif`、`mode='percentile'`、`param=1e-05` | L46 |
| `RFE` | 包裹式 | 模型重要性（`coef_`/`feature_importances_`/callable） | `n_features_to_select=None`（→一半）、`step=1`、`importance_getter='auto'` | L80–L89 |
| `RFECV` | 包裹式 | 折平均 CV 分数 | `step=1`、`min_features_to_select=1`、`cv=None`（→5 折）、`scoring=None` | L84 |
| `SelectFromModel` | 嵌入式 | 模型重要性 + 阈值 | `threshold=None`（L1→`1e-5`，否则 `"mean"`）、`prefit=False`、`norm_order=1`、`max_features=None` | L91–L149 |
| `SequentialFeatureSelector` | 包裹式 | CV 分数（贪心） | `n_features_to_select='auto'`（→一半）、`direction='forward'`、`cv=5`、`tol=None` | L151–L161 |

五点结构性事实：

- **三个家族的目标函数不同，因此三者的「盲点」不同**：过滤式看不见冗余（重复列会同时入选）、嵌入式继承模型的偏好（L1 在相关组里随机、MDI 偏好高基数）、包裹式的目标函数就是 CV 分数（因此继承 CV 的方差）。文档按对象平铺，**从不点出家族名**（见 [[feature-selection]]）。
- **`SelectFromModel` 是唯一「一次拟合 + 按重要性截断」的入口**，`RFE`/`RFECV`/`SFS` 都要反复拟合；成本口径是「估计器拟合次数」，不是「特征数」。
- **本节没有任何「降维/抽取」方法**（无 PCA/投影），所以引言里的 `feature selection/dimensionality reduction`（L10）只能读作「删列后的维数下降」。
- **本节是唯一明文推荐 `Pipeline` 的一节**（L163–L173），但没有给理由；实测泄漏代价见第 86 条。
- **本节没有稳定性选择的入口**（无 bootstrap 选择频率、无「重复 CV 取交集」），所以「选出来的列」应视为一次抽样（见 [[feature-selection]] 的「缺失的一层」）。

## 估计器矩阵（neural_network）

`sklearn.neural_network` 只有两个估计器，且共享绝大多数超参数；行号指向 [[sklearn-neural-networks]]，默认值取自 1.9.1 API 页签名（**外部核查**），行为在 1.6.1 实测。

| 估计器 | 任务 | 损失 | 输出激活 | 文档位置 |
| --- | --- | --- | --- | --- |
| `MLPClassifier` | 二分类/多分类/多标签 | `log_loss`（固定） | 二分类/多标签 logistic；多分类 softmax | L25–L91 |
| `MLPRegressor` | 回归/多输出回归 | `squared_error`（默认）；`poisson`（1.7+） | identity（`poisson` 时为指数/log-link） | L93–L97 |

五点结构性事实：

- **`coefs_`/`intercepts_` 按层边界索引**，列表长度 = `n_layers_ - 1`，不是「隐藏层数」。实测 `(5,2)` 二分类 → `[(2,5),(5,2),(2,1)]`；`()` → `n_layers_=2`、只有一个线性输出矩阵。
- **三个输出面不是同一个模型**：多分类 softmax 强制互斥，多标签每个类独立 logistic（0.5 阈值），回归 identity。把多标签当多分类会得到互斥预测。
- **三个求解器是三种训练契约**：SGD/Adam 支持 `partial_fit`，L-BFGS 对象没有该方法（L125–L129；实测 `hasattr` 为 False）。默认 `solver='adam'`，而来源建议小数据用 L-BFGS（L149）。
- **`alpha` 只罚 `coefs_`，不罚 `intercepts_`**（外部核查 1.6.1 源码）；默认 `1e-4`，来源建议的网格是 `10**-np.arange(1,7)`（L148）。
- **本节的 3 个内容折叠块全丢**（622 词 ≈ 36%）：优缺点（非凸/局部极小/随机初始化）、数学表述（三个输出面与两种损失）、5 条文献。非凸性与损失公式**不能引剪藏**，只能引官方 1.9.1 页。

## 估计器矩阵（manifold）

`sklearn.manifold` 里有 **6 个估计器 + 3 个函数**：`Isomap`、`LocallyLinearEmbedding`（4 种 `method`）、`SpectralEmbedding`、`MDS`（metric/non-metric）、`ClassicalMDS`、`TSNE`；函数为 `locally_linear_embedding`、`spectral_embedding` 与评估用的 `trustworthiness`。行号指向 [[sklearn-manifold-learning]]，默认值取自 1.9.1 API 页签名（**外部核查**），行为在 1.6.1 实测。

| 估计器 | 保什么 | 有 `transform`？ | 关键超参（默认值） | 文档位置 |
| --- | --- | --- | --- | --- |
| `Isomap` | 测地距离（kNN 图最短路） | 有（近似） | `n_neighbors=5`、`n_components=2`、`eigen_solver='auto'`、`tol=0`、`max_iter=None`、`path_method='auto'`、`neighbors_algorithm='auto'`、`metric='minkowski'`、`p=2` | L56 |
| `LocallyLinearEmbedding` | 局部重构权重 | 有 | `n_neighbors=5`、`n_components=2`、`reg=0.001`、`method='standard'`、`eigen_solver='auto'`、`tol=1e-06`、`max_iter=100`、`hessian_tol=0.0001`、`modified_tol=1e-12` | L66、L80、L90、L108 |
| `SpectralEmbedding` | 图上的局部邻近（Laplacian Eigenmaps） | **无** | `n_components=2`、`affinity='nearest_neighbors'`、`gamma=None`、`n_neighbors=None`（**= $n/10$**）、`eigen_solver=None`、`eigen_tol='auto'`、`random_state=None` | L100 |
| `MDS` | 距离（metric）/ 距离的序（non-metric） | **无** | `metric_mds=True`、`n_init=1`（**1.9 前是 4**）、`init='warn'`、`max_iter=300`、`eps=1e-06`、`dissimilarity='deprecated'`、`metric='euclidean'`、`normalized_stress='auto'` | L118–L126 |
| `ClassicalMDS` | 成对**内积**（strain） | **无** | `n_components=2`、`metric='euclidean'`、`metric_params=None`（无迭代参数、无 `random_state`） | L128、L138 |
| `TSNE` | 邻域概率（KL） | **无** | `perplexity=30.0`、`early_exaggeration=12.0`、`learning_rate='auto'`、`max_iter=1000`、`n_iter_without_progress=300`、`min_grad_norm=1e-07`、`init='pca'`、`method='barnes_hut'`、`angle=0.5`、`metric='euclidean'` | L149–L164 |
| `locally_linear_embedding` / `spectral_embedding` | 同上（函数版） | 返回数组，无对象状态 | 同对应类的核心参数 | L66、L100 |
| `trustworthiness` | **评估**：嵌入里的近邻有多少在原始空间也近 | — | `n_neighbors=5`、`metric='euclidean'` | **文档未提** |

五点结构性事实：

- **同一类名挂四个模型**：`LocallyLinearEmbedding(method='standard'|'modified'|'hessian'|'ltsa')` 的 `n_neighbors` 下界分别是无、`>= n_components`、`> n_components*(n_components+3)/2`、无校验（前两个是硬报错，`ltsa` 会在内部报一个与参数无关的 broadcast 错误）。
- **`transform` 存在与否把这一族劈成两半**：`SpectralEmbedding`/`TSNE`/`MDS`/`ClassicalMDS` **只有 `fit_transform`**（实测 1.6.1），因此不能当预处理步骤用在需要「训练一次、推理多次」的管线里；`Isomap`/`LLE` 有 `transform`，但对新样本只是外推近似。
- **`Isomap` 有 `reconstruction_error()` 方法，`LocallyLinearEmbedding` 有 `reconstruction_error_` 属性**，`SpectralEmbedding`/`TSNE`/`MDS` 都没有（后者是 `stress_`、`kl_divergence_`）。写成同一个名字必错一个。
- **`MDS` 在 1.8/1.9 之间改了三处默认值**（`metric`→`metric_mds`、`n_init` 4→1、`eps` 1e-3→1e-6），`dissimilarity` 1.8 弃用、1.10 移除。引用任何 `stress_` 数字都必须带版本。
- **本节 10 个内容折叠块全丢**（1,669 词）：6 段复杂度、metric/non-metric MDS 的细节、t-SNE 的五个优化参数、Barnes-Hut 的四条限制。所以「三阶段复杂度」「isotonic regression」「learning_rate 的启发式」都不能引剪藏。

## 估计器矩阵（cluster）

`sklearn.cluster` 里有 **11 个估计器 + 3 个函数**：`KMeans`、`MiniBatchKMeans`、`BisectingKMeans`、`AffinityPropagation`、`MeanShift`、`SpectralClustering`、`AgglomerativeClustering`、`DBSCAN`、`HDBSCAN`、`OPTICS`、`Birch`（以及 2.1 节的 `GaussianMixture`，未入库）；函数为 `kmeans_plusplus`、`estimate_bandwidth` 与 `cluster_optics_dbscan`。行号指向 [[sklearn-clustering]]，默认值取自 1.9.1 API 页签名（**外部核查**），行为在 1.6.1 实测。

| 估计器 | 假设的簇形状 | 有 `predict`？ | 关键超参（默认值） | 文档位置 |
| --- | --- | --- | --- | --- |
| `KMeans` | 凸 + 各向同性 + 等方差（inertia） | **有** | `n_clusters=8`、`init='k-means++'`、`n_init='auto'`、`max_iter=300`、`tol=1e-4`、`algorithm='lloyd'` | L45–L82 |
| `MiniBatchKMeans` | 同上（同一目标） | **有** | 同 `KMeans`，另 `batch_size=1024`、`tol=0.0`、`max_no_improvement=10`、`init_size=None`、`reassignment_ratio=0.01`、`max_iter=100` | L93–L107 |
| `BisectingKMeans` | 同上，**分裂式** | **有** | `n_clusters=8`、`init='random'`、`n_init=1`、`bisecting_strategy='biggest_inertia'`、`algorithm='lloyd'` | L259–L280 |
| `AffinityPropagation` | exemplar（真实样本点） | **有** | `damping=0.5`、`max_iter=200`、`convergence_iter=15`、`preference=None`（= 相似度中位数）、`affinity='euclidean'`（负平方距离） | L109–L122 |
| `MeanShift` | 平滑密度的众数 | **有** | `bandwidth=None`（→ `estimate_bandwidth(quantile=0.3)`）、`bin_seeding=False`、`cluster_all=True`、`min_bin_freq=1`、`max_iter=300` | L124–L138 |
| `SpectralClustering` | 图上割（normalized cut） | **无** | `n_clusters=8`、`affinity='rbf'`、`gamma=1.0`、`n_neighbors=10`、`assign_labels='kmeans'`、`n_init=10`、`eigen_solver=None`、`eigen_tol='auto'` | L140–L183 |
| `AgglomerativeClustering` | 由 `linkage` 决定（Ward 方差 / complete 最远 / average 平均 / single 最近） | **无** | `n_clusters=2`、`linkage='ward'`、`metric='euclidean'`、`connectivity=None`、`distance_threshold=None`、`compute_distances=False` | L185–L257 |
| `DBSCAN` | 任意形状的高密度连通区 | **无** | `eps=0.5`、`min_samples=5`、`metric='euclidean'`、`algorithm='auto'`、`leaf_size=30` | L282–L298 |
| `HDBSCAN` | 同上，**允许密度不同** | **无** | `min_cluster_size=5`、`min_samples=None`（**= `min_cluster_size`**）、`cluster_selection_epsilon=0.0`、`alpha=1.0`、`cluster_selection_method='eom'`、`store_centers=None`、`leaf_size=40`、`copy='warn'` | L300–L349 |
| `OPTICS` | 同上，`eps` 放松成区间 | **无** | `min_samples=5`、`max_eps=inf`、`cluster_method='xi'`、`xi=0.05`、`predecessor_correction=True`、`metric='minkowski'`、`p=2` | L351–L361 |
| `Birch` | CF 子簇（阈值内的球） | **有** | `threshold=0.5`、`branching_factor=50`、`n_clusters=3`（**默认会做一次全局聚类**）、`compute_labels=True` | L363–L377 |
| `kmeans_plusplus` / `estimate_bandwidth` / `cluster_optics_dbscan` | 函数版 | 返回数组 | `estimate_bandwidth(quantile=0.3, n_samples=None, random_state=0)`；`cluster_optics_dbscan(..., eps)` | L75、L128、L353 |

五点结构性事实：

- **「有没有 `predict`」把这一族劈成两半**：`SpectralClustering`/`AgglomerativeClustering`/`DBSCAN`/`HDBSCAN`/`OPTICS` 都没有，文档明说这属于「传导式」，"not designed to be applied to new, unseen data"（L39）——与 `manifold` 模块里「有没有 `transform`」是同一类断层。
- **「$k$ 从哪来」有三种模式**：显式 `n_clusters`（KMeans 系、Spectral、Agglomerative）、由尺度阈值间接决定（`eps`/`min_cluster_size`/`xi`/`threshold`）、算法自定（`preference`/`bandwidth`）。第三种不是「不用调」，而是把自由度移到了那个参数上。
- **`Birch` 的默认 `n_clusters=3` 是最危险的默认值**：正文只讲 `n_clusters=None` 时读叶子子簇（L377），读者会以为默认不聚类；实测 iris 上 `None` → 13 个子簇、默认 → 3 个全局簇。
- **`SpectralClustering.affinity` 默认 `'rbf'`，而正文示例全是 `'precomputed'`**（L176–L183）；`gamma=1.0` 配未缩放数据容易撞上 L153 的「相似度分布不好 → 问题奇异」警告。
- **`cluster` 这一节有 16 个 References 块、30 个内容折叠块全丢**（3,073 词）：4 个 `Mathematical formulation`（Rand/MI/homogeneity/CH/DB）、AP 与 BIRCH 的算法描述、DBSCAN 的 `Implementation` 与内存节、OPTICS 的对比与复杂度、BIRCH 的两个用法问答。所以「ARI 的机会校正公式」「互达距离的剪枝细节」「BIRCH 何时该换 MiniBatchKMeans」都**不能引剪藏**。

## 估计器矩阵（decomposition）

`sklearn.decomposition` 里有 **13 个可拟合估计器 + 1 个固定字典变换器**：`PCA`、`IncrementalPCA`、`SparsePCA`、`MiniBatchSparsePCA`、`KernelPCA`、`TruncatedSVD`、`DictionaryLearning`、`MiniBatchDictionaryLearning`、`FactorAnalysis`、`FastICA`、`NMF`、`MiniBatchNMF`、`LatentDirichletAllocation`，以及不学习字典的 `SparseCoder`。行号指向 [[sklearn-decomposition]]，默认值取自 1.9.1 API 页签名（**外部核查**），行为在 1.6.1 实测。

| 估计器 | 约束/目标 | 输出 | 关键超参（默认值） | 文档位置 |
| --- | --- | --- | --- | --- |
| `PCA` | 正交、最大方差 | scores + `components_` | `n_components=None`（→`min(n,p)`）、`whiten=False`、`svd_solver='auto'`、`tol=0.0`、`iterated_power='auto'`、`random_state=None` | L12–L30 |
| `IncrementalPCA` | 同 PCA，分块 | scores + `components_` | `n_components=None`、`whiten=False`、`batch_size=None` | L32–L49 |
| `SparsePCA` | 重构 + $ℓ_{1,1}$ | 码 + 稀疏成分 | `n_components=None`、`alpha=1`、`ridge_alpha=0.01`、`max_iter=1000`、`tol=1e-08`、`method='lars'`、`random_state=None` | L79–L108 |
| `MiniBatchSparsePCA` | 同 SparsePCA，按特征分块 | 码 + 稀疏成分 | `n_components=None`、`alpha=1`、`batch_size=3`、`shuffle=True`、`tol=0.001`、`max_no_improvement=10`、`random_state=None` | L79–L108 |
| `KernelPCA` | 核空间方差 | scores + `components_`（可反投影） | `n_components=None`、`kernel='linear'`、`gamma=None`、`alpha=1.0`、`fit_inverse_transform=False`、`eigen_solver='auto'`、`random_state=None` | L110–L130 |
| `TruncatedSVD` | 不中心化的低秩重构 | $U\Sigma$ + `components_` | `n_components=2`、`algorithm='randomized'`、`n_iter=5`、`n_oversamples=10`、`random_state=None` | L132–L144 |
| `SparseCoder` | 固定字典稀疏编码 | 码 | `transform_algorithm='omp'`、`transform_n_nonzero_coefs=None`、`transform_alpha=None`、`split_sign=False`、`positive_code=False` | L148–L166 |
| `DictionaryLearning` | 重构 + 稀疏，学过完备字典 | 码 + 字典 | `n_components=None`、`alpha=1`、`max_iter=1000`、`fit_algorithm='lars'`、`transform_algorithm='omp'`、`split_sign=False`、`positive_code=False`、`positive_dict=False`、`random_state=None` | L168–L192 |
| `MiniBatchDictionaryLearning` | 同字典学习，小批量 | 码 + 字典 | `n_components=None`、`alpha=1`、`batch_size=256`、`shuffle=True`、`transform_algorithm='omp'`、`tol=0.001`、`max_no_improvement=10`、`random_state=None` | L194–L210 |
| `FactorAnalysis` | 高斯潜变量 + 对角噪声 | 因子得分 + 载荷 | `n_components=None`、`tol=0.01`、`max_iter=1000`、`noise_variance_init=None`、`svd_method='randomized'`、`rotation=None`、`random_state=0` | L212–L264 |
| `FastICA` | 最大独立成分 | 独立成分 + 解混矩阵 | `n_components=None`、`algorithm='parallel'`、`whiten='unit-variance'`、`fun='logcosh'`、`max_iter=200`、`tol=0.0001`、`random_state=None` | L266–L282 |
| `NMF` | 非负、重构（Frobenius/beta） | W + H | `n_components='auto'`（1.6 起）、`init=None`、`solver='cd'`、`beta_loss='frobenius'`、`tol=0.0001`、`max_iter=200`、`alpha_W=0.0`、`alpha_H='same'`、`l1_ratio=0.0`、`shuffle=False` | L284–L375 |
| `MiniBatchNMF` | 同 NMF，小批量 | W + H | `n_components='auto'`、`batch_size=1024`、`forget_factor=0.7`、`fresh_restarts=False`、`max_iter=200`、`random_state=None` | L365–L375 |
| `LatentDirichletAllocation` | 主题模型（Dirichlet 先验） | 文档-主题 + 主题-词 | `n_components=10`、`doc_topic_prior=None`、`topic_word_prior=None`、`learning_method='batch'`、`learning_decay=0.7`、`max_iter=10`、`batch_size=128`、`random_state=None` | L377–L407 |

五点结构性事实：

- **`components_` 的方向与 `fit_transform` 的返回值每个模型都不同**：PCA 的 `fit_transform` 是 scores、`components_` 是 $k\times p$；NMF 的 `fit_transform` 是 **W**（$n\times k$）、`components_` 是 **H**（$k\times p$）；LDA 的 `transform` 是文档-主题、`components_` 是主题-词。把三者的 `components_` 当同一种东西会读错形状（L349–L358、L398）。
- **`transform` 不是 `fit_transform` 的缓存**：NMF 的 `transform(X)` 用 H 重新求 W，与 `fit_transform(X)` 不逐位相同（本机 3.72e-4/7.68e-4）；训练集编码要保存 `fit_transform` 的返回值。
- **约束就是模型**：PCA 正交（L16）、SparsePCA 的 $ℓ_1$（L91–L99）、NMF 非负（L286）、ICA 独立（L268）、FA 对角噪声（L246–L250）、LDA Dirichlet（L379–L388）。**换约束 = 换模型**，不是同一个模型的不同求解器。
- **非凸/随机模型集中在右半边**：SparsePCA、DictionaryLearning、NMF、LDA 都有局部极小/随机初始化；NMF 的 `init` 比 `random_state` 更能改变结果（本机 4 成分误差 4.4395/2.3878/1.4714/2.0252）。PCA 有解析解（full），但 randomized/arpack 仍依赖 `random_state`。
- **文档里的两个参数名与 API 不一致**（外部核查）：正文说 `split_code`（L160），API 是 `split_sign`；正文说 `transform_method`（L152），API 是 `transform_algorithm`。按正文抄会 `TypeError`。

## 估计器矩阵（covariance）

`sklearn.covariance` 里有 **6 个可拟合估计器 + 4 个函数**：`EmpiricalCovariance`、`ShrunkCovariance`、`LedoitWolf`、`OAS`、`GraphicalLasso`、`GraphicalLassoCV`、`MinCovDet`（另有 `EllipticEnvelope` 属已入库的 2.7 节，见 [[sklearn-outlier-detection]]），函数 `empirical_covariance`、`shrunk_covariance`、`ledoit_wolf`、`oas`。行号指向 [[sklearn-covariance]]，默认值取自 1.9.1 API 页签名（**外部核查**，1.6.1 实测签名一致）。

| 估计器 | 估计对象 | 结构假设 | 强度旋钮（默认值） | 输出属性 | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `EmpiricalCovariance` | $\Sigma$（MLE，$1/n$） | 无 | 无；`assume_centered=False`、`store_precision=True` | `covariance_`、`precision_`、`location_` | L14–L22 |
| `ShrunkCovariance` | $\Sigma$，向 $\frac{\mathrm{Tr}}{p}I$ 收缩 | 标量乘单位阵 | `shrinkage=0.1`（0–1，**任意默认值**） | 同上 | L26–L38 |
| `LedoitWolf` | 同上，α 由**MSE**解析定 | 同上 | `block_size=1000`（仅内存） | 同上 + `shrinkage_` | L40–L57 |
| `OAS` | 同上，高斯下 MSE 更小 | 同上 | 无 | 同上 + `shrinkage_` | L59–L75 |
| `GraphicalLasso` | $K=\Sigma^{-1}$（**稀疏**） | $K$ 的非对角 L1 | `alpha=0.01`、`mode='cd'`、`tol=1e-4`、`enet_tol=1e-4`、`max_iter=100`、`eps=2.22e-16`、`assume_centered=False` | `precision_`、`covariance_`、`location_` | L77–L114 |
| `GraphicalLassoCV` | 同上的 α 由 CV 定 | 同上 | `alphas=4`（**路径点数**）、`n_refinements=4`、`cv=None`（→5 折） | 同上 + `alpha_` | L83 |
| `MinCovDet` | 修剪 + 重加权后的 $\Sigma$、$\mu$ | h 个「好」点 | `support_fraction=None`（→$\lceil(n+p+1)/2\rceil$）、`random_state=None`、`store_precision=True` | `covariance_`、`precision_`、`location_`、`support_`、`raw_support_`、`raw_covariance_`、`raw_location_`、`dist_` | L122–L135 |

五点结构性事实：

- **四个收缩估计器是同一个算子**：`LedoitWolf`/`OAS` 的 `covariance_` 与 `ShrunkCovariance(shrinkage=<该估计器>.shrinkage_)` **逐位相同**（本机实测）。差别只在 α 的来源：0 / 手设 / 解析-MSE。
- **`precision_` 是伪逆**（docstring 原文「Estimated pseudo inverse matrix」）：`pinvh` 计算，$n\le p$ 时不报错，`score` 才返回 `-inf`。`store_precision=False` 时 `precision_` 属性存在但值为 `None`，`get_precision()` 仍能算。
- **`GraphicalLasso` 的输出是图**：`K_{ij}=0` ⇔ 条件独立；L1 只罚**非对角**（源码 `_objective` 与 L105 一致）。`alpha=0` 合法且等价于 `EmpiricalCovariance`（本机 maxdiff 0.0）。
- **`MinCovDet` 的 `support_` 不是 h**：它被重加权掩码覆盖（实测 178 vs h=102），h 在 `raw_support_`；校正使 $\det$ 从 0.0762 变 0.7365（≈9.7×）。
- **`EllipticEnvelope`、`log_likelihood`、`fast_mcd`、`ledoit_wolf_shrinkage` 不在 2.6 节里**：`EllipticEnvelope` 的用法属 2.7 节（已入库，见下方「异常检测面」），后三者是本节实现细节的外部可见面。

## 估计器矩阵（异常检测面：2.7 节）

2.7 节**没有自己的模块**：五个估计器分散在 `svm`、`linear_model`、`covariance`、`ensemble`、`neighbors` 里，靠同一套「分数 → `offset_` → ±1」契约统一。行号指向 [[sklearn-outlier-detection]]，默认值取自 1.9.1 API 页签名（外部核查），行为在 1.6.1 实测。

| 估计器（模块） | 形状假设 | 分数（`score_samples`） | 阈值旋钮（默认值） | 能否给新样本 `predict` | 文档位置 |
| --- | --- | --- | --- | --- | --- |
| `svm.OneClassSVM` | 高斯核边界 + $\nu$ 分位 | 到边界的有符号距离 | `nu=0.5`、`gamma='scale'`、`kernel='rbf'`（**无 `contamination`**） | 能 | L74–L90 |
| `linear_model.SGDOneClassSVM` | 线性边界（配核近似） | 同上 | `nu=0.5`、`max_iter=1000`、`learning_rate='optimal'` | 能 | L92–L94 |
| `covariance.EllipticEnvelope` | 稳健高斯椭圆（MCD） | Mahalanobis → 对数似然差 | `contamination=0.1`（**不支持 `'auto'`**）、`support_fraction=None` | 能 | L104–L121 |
| `ensemble.IsolationForest` | 平均隔离路径长度 | $-2^{-E(h)/c(n)}$（≈ −1 异常 … 0 正常） | `contamination='auto'`（→ `offset_=-0.5`）、`n_estimators=100`、`max_samples='auto'`（→`min(256,n)`）、`max_features=1.0` | 能 | L123–L156 |
| `neighbors.LocalOutlierFactor` | kNN 局部密度比 | 邻域平均密度 / 自身密度（取负） | `contamination='auto'`（→ `offset_=-1.5`）、`n_neighbors=20`、`metric='minkowski'`、`novelty=False` | **仅 `novelty=True`** | L158–L205 |

五点结构性事实：

- **`offset_` 是唯一「可调而不重训」的量**，而它的默认语义有**四种**：IF 常数 −0.5、LOF 常数 −1.5、EE 数值 0.1、OCSVM 无此参数。所以「同一数据上五个默认输出谁报得多」不可比（本机实测：高斯数据上 IF 报 13.9–15.4%、LOF 只报 0.2–1.4%）。
- **`decision_function = score_samples - offset_` 四类全成立**（1.6.1 实测 maxdiff 0.0），`predict = (decision_function < 0) ? -1 : +1`，**0 归为正常**。
- **`contamination` 的合法区间是 `(0.0, 0.5]`**：IF/LOF 还接受 `'auto'`，EE 只接受 float。「一个异常也不报」**没有合法写法**。
- **IF/EE 的报出数精确等于 $c\cdot n$**（1,100 样本上 0.05/0.1/0.2/0.5 → 55/110/220/550）；**LOF 的 `predict` 会少于 `nof < offset_`**（32 vs 33），因为它重算分数、自身不在邻域里。
- **尺度行为分两档**：`IsolationForest`（秩比较）与 `EllipticEnvelope`（仿射等变）**完全不变**——整体 ×1000、单列 ×100 都实测 maxdiff 0.0 / 标签一致率 1.0；`OneClassSVM(gamma='scale')` 对整体等比不变、对单列敏感（一致率 0.89，`gamma` 固定时 0.485）；LOF 依赖 `metric`（`cosine` 报 8 个 vs `euclidean` 24 个）。

## 切分器矩阵（model_selection：3.1 节）

3.1 节**没有估计器**：它提供的是「把数据切成训练/评估两份」的 14 个输入（11 个 splitter + 1 个 `train_test_split` + 3 个 `cross_*` 函数）。行号指向 [[sklearn-cross-validation]]，默认值取自 1.9.1 API 页签名（外部核查），行为在 1.6.1 实测。

| 对象 | 赌的依赖假设 | 关键参数（默认值） | 它不管什么 | 文档位置 |
| --- | --- | --- | --- | --- |
| `KFold` | 样本互换不变 | `n_splits=5`、`shuffle=False`、`random_state=None` | 类别、分组 | L228–L256 |
| `RepeatedKFold` | 同上 + 抗划分运气 | `n_splits=5`、`n_repeats=10`、`random_state=None` | 同上 | L258–L279 |
| `LeaveOneOut` | 同上，极端版 | 无参数 | 同上 | L281–L304 |
| `LeavePOut` | 同上 | `p`（必填） | 同上；$p>1$ 时测试集重叠 | L306–L325 |
| `ShuffleSplit` | 同上，但**不要求划分** | `n_splits=10`、`test_size=None`、`train_size=None`、`random_state=None` | 同上；**不能给 `cross_val_predict` 用** | L327–L352 |
| `StratifiedKFold` | 类比例稳定 | `n_splits=5`、`shuffle=False`、`random_state=None` | 分组 | L367–L399 |
| `RepeatedStratifiedKFold` | 同上 + 重复 | `n_splits=5`、`n_repeats=10` | 分组 | L258–L279 |
| `StratifiedShuffleSplit` | 同上，非划分 | `n_splits=10`、`test_size=None` | 分组 | L401–L407 |
| `GroupKFold` | 组内相关、组间独立 | `n_splits=5`、`shuffle=False`（**1.6 新增**）、`random_state=None` | 类比例 | L425–L454 |
| `StratifiedGroupKFold` | 组 + 类比例 | `n_splits=5`、`shuffle=False`、`random_state=None`（**两者不是 keyword-only**） | 两者都可能退化 | L456–L477 |
| `LeaveOneGroupOut` / `LeavePGroupsOut` | 同上 | `n_groups`（前者无参数） | 类比例 | L479–L521 |
| `GroupShuffleSplit` | 同上，抽样版 | `n_splits=5`、`test_size=None`、`train_size=None` | 类比例 | L523–L551 |
| `TimeSeriesSplit` | 时间顺序不可交换 | `n_splits=5`、`max_train_size=None`、`test_size=None`、`gap=0`（后三项 **0.24 新增**） | 分组、类比例 | L579–L604 |
| `PredefinedSplit` | 划分由外部给定 | `test_fold`（必填；0 = 验证、−1 = 训练） | 一切 | L409–L413 |
| `train_test_split` | `ShuffleSplit` 的包装 | `test_size=None`（→**0.25**）、`train_size=None`、`shuffle=True`、`stratify=None` | **分组（无 `groups` 参数）** | L553–L573 |
| `cross_val_score` / `cross_validate` / `cross_val_predict` | — | `cv=None`（→5 折）、`error_score=nan`、`method='predict'` | — | L58–L213 |

六点结构性事实：

- **`cv=int` 在分类器上等于 `StratifiedKFold`**（外部核查：`check_cv`，及 1.6.1 实测）：`cv=5` 与 `cv=StratifiedKFold(5)` 逐位相同，与 `cv=KFold(5)` **不同**（iris 上 0.98 vs 0.94）。文档只写了「when the `cv` argument is an integer」（L89）。
- **`cv=None` 也是 5 折**（0.22 起；之前 3 折，外部核查）。剪藏**完全不提默认折数**。
- **默认不洗牌**（L613）：`KFold`/`StratifiedKFold`/`GroupKFold` 在 `shuffle=False` 时传 `random_state` 会直接抛 `ValueError`，所以「种子设了没生效」在 1.6.1 上是硬错误。
- **`groups` 只在 splitter 是分组型时生效**：`cv=int` + `groups` 只给 `UserWarning: The groups parameter is ignored by StratifiedKFold`；`cv=GroupKFold(...)` 而不传 `groups` 才抛 `ValueError`（L423）。
- **`TimeSeriesSplit` 的测试折大小 = $\lfloor n/(k+1)\rfloor$**，剩余样本全部塞进第一个训练折（L581）；$n=5,k=5$ 时报「folds=6」（内部 +1）。
- **`cross_val_predict` 只接受「划分」**：测试集必须恰好覆盖每个样本一次，`ShuffleSplit`/`GroupShuffleSplit` 直接 `ValueError: cross_val_predict only works for partitions`（L192 正文的「Only cross-validation strategies that assign all elements to a test set exactly once」）。

## 搜索器矩阵（model_selection：3.2 节）

3.2 节给出**四个搜索器 + 两个候选生成器**，它们不是四个 API，而是**三条预算曲线**（穷举 / 采样 / 逐次减半）。行号指向 [[sklearn-hyperparameter-tuning]]，默认值取自 1.9.1 API 页签名（外部核查），行为在 1.6.1 实测。

| 对象 | 预算曲线 | 关键参数（1.9.1 默认值） | 能不能给新样本 `predict` | 文档位置 |
| --- | --- | --- | --- | --- |
| `GridSearchCV` | 每条候选 × $k$ 折，全量资源 | `param_grid`（必填）、`scoring=None`、`refit=True`、`cv=None`（→5 折）、`error_score=nan`、`return_train_score=False`、`n_jobs=None`、`pre_dispatch='2*n_jobs'` | 能（仅 `refit` 为 `True`/指标名/callable 时；`refit=False` → `AttributeError`） | L34–L52 |
| `RandomizedSearchCV` | `n_iter` 条候选 × $k$ 折 | `param_distributions`（必填）、`n_iter=10`、`random_state=None`、其余同上 | 同上 | L54–L95 |
| `HalvingGridSearchCV` | 全部候选 × 小资源 → 按 `factor` 淘汰、资源递增 | `factor=3`、`resource='n_samples'`、`max_resources='auto'`、`min_resources='exhaust'`、`aggressive_elimination=False`、**`cv=5`**、**`return_train_score=True`**、`random_state=None` | 同上 | L97–L227 |
| `HalvingRandomSearchCV` | 同上，候选数由 `n_candidates` 定 | `n_candidates='exhaust'`、**`min_resources='smallest'`**、其余同上 | 同上 | L97–L227 |
| `ParameterGrid` | 笛卡尔积枚举（可传 list of dict = 多个网格） | 无参数；值必须是 list/array，标量 → `TypeError` | — | L39–L45（未点名） |
| `ParameterSampler` | 从分布/列表抽样 | `n_iter`、`random_state`；**全列表 → 不放回**，`n_iter` 被夹 | — | L61–L86（未点名） |

五点结构性事实：

- **拟合次数是可数的**：`GridSearchCV`/`RandomizedSearchCV` 恒为「候选数 × 折数 + 1」（`refit=False` 时不留那 1 次），实测 3 候选 × 5 折 → 16 / 15，180 候选 × 3 折 → 541。**逐次减半恒为 $\big(\sum_i n_{\text{cand},i}\big)\times k+1$，永远不少于网格**（实测 8 候选 × 5 折：76 vs 41；180 候选 × 3 折：811 vs 541）。
- **halving 的 `return_train_score` 与 `cv` 默认值和另两个搜索器不同**：前者 `True`、`cv=5`（另两个是 `False`、`None`）。所以换到 halving 之后 `cv_results_` 的列数从 15 涨到 25。
- **只有 `RandomizedSearchCV` 与两个 halving 有 `random_state`**。`GridSearchCV` 传入即 `TypeError`；它的可复现性只能经 `cv=`（splitter 的 `random_state`）与估计器自身。
- **`refit` 有三种取值、四种后果**：`True` → `best_estimator_` + `best_score_`；指标名字符串 → 同上（多指标时唯一合法）；`False` → 只有 `best_params_`/`best_index_`（**多指标时连这两个也没有**），无 `predict`；callable → 有 `best_index_`/`best_estimator_`、**没有 `best_score_`**。
- **`cv_results_` 两种形状**：`GridSearchCV`/`RandomizedSearchCV` 是「一候选一行」的短表；两个 halving 是「候选 × 轮次」的长表，多出 `iter`、`n_resources` 与 `split*_train_score`。

## 评分入口矩阵（metrics：3.4 节）

3.4 节同样**不是估计器**：它给出的是「分数本身怎么定义」的一层。三个入口返回的东西不同（估计器 `score` 给默认准则、`scoring=` 给统一「越大越好」的标量、`metrics` 函数给原始量纲），而 `make_scorer` 是前两者之间的桥。行号指向 [[sklearn-metrics]]（**该剪藏只保留官方 54 个小节中的 4 个**：字符串评分表、`Usage examples` doctest 与全部具体指标定义都不在剪藏内），签名与默认值取自 1.9.1 API 页（外部核查），行为在 1.6.1 实测。

| 对象 | 返回/作用 | 关键参数（默认值） | 它不管什么 | 位置 |
| --- | --- | --- | --- | --- |
| 估计器的 `.score` | 该估计器预设的准则（分类器通常 accuracy、回归器 $R^2$） | 无参数 | 不翻转符号、不声明泛函 | L64（「accuracy」一词已丢）；[[sklearn-cross-validation]] L56 |
| `scoring=` 字符串 | 统一为**越大越好**的标量 | 1.6.1 实测 **56 个**名字 vs 1.9.1 表 **43 行** | 不改响应方法、不管多类 AUC 必须写 `ovr`/`ovo` | L85–L87（表本身已丢） |
| `scoring=` callable | 自定义 `(estimator, X, y) -> float` | **无签名校验**（实测多一个可选参数也被静默接受） | 不声明符号、不声明响应方法 | L80–L81（两个列表标签丢） |
| `make_scorer` | 把 `metrics` 函数包成 scorer | `response_method='predict'`（**1.8+**；1.6/1.7 是 `'default'`）、`greater_is_better=True` | 不检查函数是否吃概率 | 剪藏外（§3.4.3.2.1） |
| `get_scorer` / `get_scorer_names` / `check_scoring` | 取 scorer 对象 / 合法名字清单 / 默认 scorer | `check_scoring(estimator=None, scoring=None, *, allow_none=False, raise_exc=True)` | — | 剪藏外（`get_scorer_names` 只在丢失的 Note 里） |
| `metrics.*` 函数 | 原始量纲（MSE 就是 MSE） | 各自参数：`average`、`zero_division='warn'`、`multioutput`、`labels` | **不**翻转符号 | 剪藏外（§3.4.4 与 §3.4.6 全丢） |
| `confusion_matrix` 及派生量 | 决策层的评分 | `labels`、`normalize`、`threshold`（在 `predict` 里隐含 0.5） | 不告诉你怎么选阈值 | L31 |

六点结构性事实：

- **三层的返回值不同，符号也不同**：`get_scorer('neg_mean_squared_error')._sign = -1`、`_score_func = mean_squared_error`；`get_scorer('r2')._sign = +1`；而 `metrics.mean_squared_error` 直接返回正数。跨层混用是本页最高频的报告错误（第 217、219 条）。
- **每个字符串名都绑定一个响应方法**：`accuracy`/`r2` → `'predict'`；`neg_log_loss`/`neg_brier_score` → `'predict_proba'`；`roc_auc` → `('decision_function', 'predict_proba')`（**按序取第一个存在的**）。所以「模型有没有概率输出」会改变同一个评分名的含义（第 220 条）。
- **符号与响应方法只在字符串名上自动成立**：自定义 `make_scorer` 必须自己写 `greater_is_better` 与 `response_method`，两个默认值都偏向「直接包一个误差函数」的错用法（第 217、218 条）。
- **指标不匹配不会抛错，只会变 `nan`**：`neg_log_loss` 配无 `predict_proba` 的 `SVC`、`roc_auc` 配多类数据，都返回 `[nan, nan, nan]` + `UserWarning`（第 221 条）。
- **字符串清单在两个方向上都不完整**：1.6.1 的 56 个名字里有 3 个（`matthews_corrcoef`、`positive_likelihood_ratio`、`neg_negative_likelihood_ratio`）四版文档表都没列；表里也有 2 个（`d2_log_loss_score`、`d2_brier_score`）在 1.6.1 上不可用（第 222–223 条）。
- **阈值是模型外的参数**：`predict` 恒等于 `proba >= 0.5`；同一个逻辑回归在四个阈值下准确率 0.946/0.954/0.802/0.546、正类召回 0.460→0.840（第 224 条与 [[evaluation-metrics]] 第五节）。

## 变换器矩阵（preprocessing：8.3 节）

8.3 节**不是估计器**：它提供的对象不吃 `y`（`TargetEncoder` 除外）、不输出分数，只把 `X` 换成另一个 `X`。真正有用的分界线不是「缩放/编码/离散化」，而是 **`fit` 有没有学到状态**（12 : 4）与**作用在哪个轴上**（逐列 / 逐行 / 全矩阵 / 列之间）。行号指向 [[sklearn-preprocessing]]，签名与默认值取自 1.9.1 API 页（外部核查），行为在 **1.9.1 与 1.6.1 双跑**。

| 对象 | 状态（`fit` 学到什么） | 关键参数（默认值） | 作用轴 | 它不管什么 | 位置 |
| --- | --- | --- | --- | --- | --- |
| `StandardScaler` | `mean_`、`var_`/`scale_` | `with_mean=True`、`with_std=True` | 逐列 | 不做稀疏（`with_mean=True` 直接 `ValueError`）；常数列 `scale_` 被换成 1 | L14–L77 |
| `MinMaxScaler` | `data_min_`/`data_max_`/`scale_`/`min_` | `feature_range=(0,1)`、`clip=False` | 逐列 | 不裁剪越界值（默认）；输出范围随训练集变 | L79–L117 |
| `MaxAbsScaler` | `scale_`（每列最大绝对值） | `clip=False`（**1.8+**） | 逐列 | 不中心化（因此对「未居中」数据只能压到 $[-1,1]$） | L119–L147 |
| `RobustScaler` | `center_`、`scale_`（IQR） | `quantile_range=(25,75)`、`unit_variance=False` | 逐列 | **不能 `fit` 稀疏**（只能 `transform`）；`scale_` 不是标准差 | L160–L162 |
| `KernelCenterer` | `K_fit_rows_`、`K_fit_all_` | 无参数 | 全矩阵 | 只接受**方阵**；不处理非 PSD 核 | L164–L166 |
| `QuantileTransformer` | `quantiles_`、`references_` | `n_quantiles=1000`、`output_distribution='uniform'`、`subsample=10000` | 逐列 | 不保相关/距离；训练范围外退化成 0/1 或 ±5.1993 | L176–L252 |
| `PowerTransformer` | `lambdas_` | `method='yeo-johnson'`、`standardize=True` | 逐列 | Box-Cox 不做非正值；`standardize` 会再标准化 | L211–L231 |
| `Normalizer` | **无** | `norm='l2'` | 逐**行** | 不解决零向量；不保证列尺度 | L254–L296 |
| `OrdinalEncoder` | `categories_`（+`infrequent_categories_`） | `handle_unknown='error'`、`encoded_missing_value=nan` | 逐列（1 列） | 不告诉你序是否合理（强加字典序） | L298–L341 |
| `OneHotEncoder` | `categories_` | `sparse_output=True`、`handle_unknown='error'`、`drop=None` | 逐列 → $n_{\text{cat}}$ 列 | 不管基数成本；`drop`+`'ignore'` 时未知 = 被 drop 类 | L355–L599 |
| `TargetEncoder` | `encodings_`、`target_mean_`、`target_type_` | `smooth='auto'`、`cv=5`、`shuffle='deprecated'`（1.9） | 逐列（1 列） | **不能**用 `fit().transform()` 代替 `fit_transform`；不防外层 CV 泄漏 | L601–L626 |
| `KBinsDiscretizer` | `bin_edges_`、`n_bins_` | `n_bins=5`、`encode='onehot'`、`strategy='quantile'` | 逐列 → `n_bins` 列 | 不管越界值（静默落边界箱）；不保证跨版本同边界 | L634–L688 |
| `Binarizer` | **无** | `threshold=0.0` | 逐列 | 不做「$\ge$」（严格大于 threshold 才为 1） | L690–L725 |
| `PolynomialFeatures` | `powers_`、`n_output_features_` | `degree=2`、`interaction_only=False`、`include_bias=True` | 列之间 | 不管列数爆炸（20 维 3 阶 = 1771） | L735–L775 |
| `SplineTransformer` | 结点位置、`n_features_out_` | `n_knots=5`、`degree=3`、`extrapolation='constant'`、`handle_missing='error'`（**1.8+**） | 逐列 → `n_knots+degree-1` 列 | 不给交互项（L781）；外推被钉在边界，不是线性 | L777–L818 |
| `FunctionTransformer` | **无** | `validate=False`、`check_inverse=True` | 任意 | 不检查 NaN（`validate=False` 时静默穿过）；只在给了 `inverse_func` 时检查互逆 | L821–L843 |

函数级入口（同样在 1.9.1 签名核实）：`scale(X, *, axis=0, with_mean=True, with_std=True)`、`minmax_scale(X, feature_range=(0,1), *, axis=0)`、`maxabs_scale(X, *, axis=0)`、`robust_scale(X, *, axis=0, with_centering=True, with_scaling=True, quantile_range=(25,75), unit_variance=False)`、`quantile_transform(X, *, axis=0, n_quantiles=1000, output_distribution='uniform', subsample=**100000**, random_state=None)`、`power_transform(X, method='yeo-johnson', *, standardize=True)`、`normalize(X, norm='l2', *, axis=**1**)`、`binarize(X, *, threshold=0.0)`、`add_dummy_feature(X, value=1.0)`。

五点结构性事实：

- **12 : 4 的有状态比值决定「能不能不放进管线」**：`Normalizer`/`Binarizer`/`FunctionTransformer` 的 `fit` 是 no-op，`PolynomialFeatures` 只从 `degree` 推出 `powers_`（与数据无关），其余 12 个都在 `fit` 学到数据相关的量。但即使是 no-op 的三个，也应该放进管线（顺序一致性），因为「不属于模型的一部分」并不等于「可以随意提前」。
- **两个轴不可互换**：`scale`/`StandardScaler` 是 `axis=0`（逐列），`normalize(X, norm='l2')` 是 `axis=1`（逐行）。前者保证「同一特征在所有样本上同一尺度」，后者保证「每个样本的单位范数」，**后者不保证前者**。
- **稀疏的活路只有三条**：`MaxAbsScaler`、`StandardScaler(with_mean=False)`、显式 `toarray()`（L154–L158）。`RobustScaler` 是半合法状态：`fit` 稀疏报 `ValueError: Cannot center sparse matrices: use 'with_centering=False' instead.`，但 `transform` 稀疏可以。
- **吃 `y` 的只有 `TargetEncoder`**：它的 `fit_transform` 内部走 $k$ 折交叉拟合（默认 `cv=5`）、`fit` 不走；`encodings_` 存的是全量训练集编码，折内编码「temporary and not saved」（L616）。1.9 起 `shuffle`/`random_state` 弃用（1.11 移除），改用 `cv` 传 splitter。
- **列数只有四类**：1 列（缩放器、归一化、序数/目标编码、二值化）、固定小倍数（样条 `n_knots+degree-1`、分箱 `n_bins`）、基数倍数（one-hot $n_{\text{cat}}$）、组合爆炸（多项式 $\binom{d+\text{deg}}{\text{deg}}$）。20 维特征上的实测列数：231 / 211 / 1771 / 1351（多项式 2/2-interaction/3/3-interaction）、140（默认样条）、200（10 箱 one-hot）、120 与 29（120 类 one-hot 全量 / `min_frequency=20`）。

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

svm 一侧（引用 `[[sklearn-svm]], L…`；标「外部核查」的取自 `sklearn/svm/_classes.py` @ 1.9.1 与 `whats_new/v1.9`，不在剪藏内）：

23. **`SVC(probability=True)` 与 `NuSVC(probability=True)` 在 1.9 已弃用**（外部核查，1.11 移除），原因是**非线程安全**；替代写法 `CalibratedClassifierCV(SVC(), ensemble=False)`，`probA_`/`probB_` 同时弃用。**剪藏正文（就是 1.9.1 那一页）完全没反映这一点**——已知分歧，记录在 [[sklearn-svm]]。
24. **`probability` 默认 `False`**（外部核查）。所以默认的 `SVC` 没有 `predict_proba`；需要概率就得付内部交叉验证的代价（L22、L133）。
25. **`gamma='scale'`**（外部核查，即 $1/(n_{\text{features}}\cdot\mathrm{Var}(X))$）。**正文完全不提 gamma 的默认值**，只描述其语义（L242、L224）——所以“用默认 gamma 跑一次”在文档里查不到。
26. **`decision_function_shape='ovr'` 但内部永远 ovo**（L75）。改这个参数只改**输出形状**，不重训；且它可以在 `fit` 之后改（L86）。
27. **`break_ties=False`**：多类 ovr 下 `predict` **默认不破平局**（返回第一个并列类），与 `np.argmax(decision_function(...), axis=1)` 不等（L140）。报准确率必须声明这个开关。
28. **`class_weight` 就是乘入 `C`**（L146）；`sample_weight` 同理（L150）。加权后必须重扫 `C`。另：文档说 `class_weight` 仅 `SVC` 有，但 1.9.1 的 `NuSVC` 签名也接受（外部核查）——该句存档，见 [[sklearn-svm]] 的分歧节。
29. **`cache_size=200`**（MB，外部核查）：不改结果只改时间，且无任何提示；大问题上建议 500–1000（L200）。
30. **`LinearSVC` 默认 `loss='squared_hinge'`，不是 hinge**（外部核查）：与 `SVC` 的根本差异之一，叠加“截距进正则”（L32）后，同一 `C` 的结果不同。`dual='auto'`（外部核查）决定随机性是否生效（L215）。
31. **`shrinking=True`**：迭代多时有益，但**松散求解（大 `tol`）时不开反而快**（L211，引 [12]）。
32. **`SVR.epsilon=0.1`、`LinearSVR.epsilon=0.0`**（外部核查）：两个默认值不同，前者是容忍带宽度，后者对 `LinearSVR` 意味着零容忍带。
33. **`nu=0.5`** 是 `NuSVC`/`NuSVR`/`OneClassSVM` 的共同默认（外部核查）；文档只给语义（“近似训练误差与支持向量的比例”，L212）。

neighbors 一侧（引用 `[[sklearn-nearest-neighbors]], L…`；标「外部核查」的取自 1.9.1 签名与官方丢失折叠块，**另有「实测」项是在本机 1.6.1 上执行**）：

34. **`n_neighbors=5`**（外部核查 + 实测）：用户指南只讨论 $k$ 的语义与取舍（L103），**完全不提默认值**——「用默认 5 跑一下」在文档里查不到。
35. **`weights='uniform'`**，而 `weights='distance'` 在**命中训练点时退化成 1-NN**（实测 1.6.1：训练点处 `predict_proba` 变 one-hot，回归返回该点标签而非邻域均值）。留一交叉验证下每个查询点都是训练点，因此这个组合会**恒定触发**该分支。
36. **投票平票取最小标签**（实测 1.6.1）：`predict_proba` 给 `[[0.5, 0.5]]` 但 `predict` 返回 `0`。重编码标签会改变结果，行顺序不会。
37. **`predict_proba` 是投票占比，不是校准概率**（外部核查）：`uniform` 下就是 `count/k`。在需要概率的场合必须另做校准。
38. **`algorithm='auto'` 的隐性转向**（官方 1.9 页丢失折叠块 + 实测）：输入稀疏、`metric='precomputed'`、$D>15$、$k \ge N/2$、或度量不在两棵树的 `VALID_METRICS` 里 → 选 `brute`。所以你在 `D>15` 或 `k` 大时**本来就不在用树**。
39. **`leaf_size=30`**（外部核查）：只在树路线上生效；极大时查询退化成暴力，极小时节点遍历开销吞掉收益；`BallTree` 的存储约 $1/\text{leaf\_size}$ 倍训练集。用户指南**只在 doctest 里出现过这个参数**（L77），无解释。
40. **`RadiusNeighbors*` 的 `radius=1.0` 与 `outlier_label=None`**（外部核查 + 实测）：默认半径与数据尺度无关（**未缩放下毫无意义**）；遇到空邻域**直接报错**而不是跳过。
41. **`NearestCentroid(metric='euclidean', priors='uniform', shrink_threshold=None)`**（实测 1.6.1）：默认构造确实是零超参基线；但 `metric` 只允许两个取值（1.5 起），`priors` 的 docstring 自相矛盾（见上方估计器矩阵）。
42. **`KNeighborsTransformer(mode='distance')` 多算 1 个邻居**（实测 1.6.1）：L232 明说了，但下游若按「恰好 $k$ 个」预期就会收到 $k+1$ 条。
43. **`NeighborhoodComponentsAnalysis(n_components=None)` 不降维**（外部核查）：默认取 `n_features`，即全维变换；`max_iter=50`、`tol=1e-5` 且用 L-BFGS-B 全梯度（L308）——小样本上很容易碰迭代上限。
44. **三种后端的成本不写在默认值里**：$N \lesssim 30$ 时暴力更快；$D$ 大时 KD 树可能**比暴力更慢**；查询点很少时树反而亏（官方 1.9 页丢失折叠块）。剪藏只留了「总共 $O[DN\log N]$」一句（L142），容易误读为「树总是更快」。

tree 一侧（引用 `[[sklearn-decision-trees]], L…`；默认值取自 1.9.1 API 页，标「实测」的在 1.6.1 上执行）：

45. **`max_depth=None` + `min_samples_leaf=1` + `ccp_alpha=0.0` = 长到每个叶子纯（或无法再分）**（外部核查）。Tips 一节**从不说默认值**，所以「用默认参数跑一下」实际上是在最大化过拟合（L29、L213–L225）。
46. **`min_samples_split=2` 按未加权样本数算**（L221；实测：6 样本、权重 `[1,1,1,1,1,100]` 时 `=6` 仍切、`=7` 才停）。加权场景要换成 `min_weight_fraction_leaf`（L223）。
47. **`np.float32` 内部转换**（L224）：实测 0.30000001 与 0.30000002 塌成同一阈值 → 4 个样本只切出 3 叶，`predict(0.300000015)` 返回两目标均值 `1.5`；阈值本身带 float32 舍入痕迹（`0.25000001`）。
48. **`ExtraTree*` 的 `max_features` 默认不是 `None`**：分类 `'sqrt'`、回归 `1.0`（外部核查）。实测同一 200×20 数据：`DecisionTreeClassifier` 13 叶 vs `ExtraTreeClassifier` 79 叶。
49. **`predict` 平票取 `classes_` 中最小索引**（L57；实测 `y=['b','a']` → 返回 `'a'`）。重编码标签会改预测，行顺序不会。
50. **`predict_proba` 是叶内（加权）比例**（L59）：实测三个不可分样本默认 `[1/3, 2/3]`，`class_weight='balanced'` 后变 `[0.5, 0.5]`——类别平衡会连带改概率。
51. **缺失值的「训练未见 NaN」分支走样本多的子节点**（L349；实测复现）。这是一个与特征值无关的默认方向。
52. **`poisson` 要求 $y\ge0$**（L308；实测 `y=[-1,2]` → `ValueError`），且实现最小化**半** deviance（L308）。
53. **`absolute_error` 的成本随版本变**：文档写「1.8 起是 MSE 的 3–6×」（L316）；1.8 把 MAE 从 $O(n^2)$ 改成 $O(n\log n)$（外部核查 `whats_new/v1.8`）。本机 1.6.1 实测 ≈23×——**这个数字不能跨版本引**。
54. **`monotonic_cst=None` 存在但不被本节覆盖**（外部核查：1.4 页的签名里已含该参数；剪藏 grep 为 0）。1.9 起可与缺失值组合（`whats_new/v1.9`）。
55. **`iris.svg` 那张导出图是 2014 年的静态资产**：1.6/1.7/1.8/1.9 四版字节相同，根切分 `petal length ≤ 2.45`，而同页 `export_text` doctest 与本机运行时都是 `petal width ≤ 0.80`（两个候选根的加权 Gini 相同，都是 $1/3$）。抄图里的阈值前先自己跑一遍。

ensemble 一侧（引用 `[[sklearn-ensembles]], L…`；默认值取自 1.9.1 源码签名，标「实测」的在 1.6.1 上执行）：

56. **HGBT 的早停默认开启**：`early_stopping='auto'` 在样本 >10,000 时生效（L72；源码 docstring 同）。实测 1.6.1：10,050 样本、`max_iter=100` → `n_iter_=76`。所以「跑满 100 轮」是一个需要显式关闭早停才成立的前提。
57. **HGBT 的默认树大小是「浅而宽」**：`max_leaf_nodes=31`、`min_samples_leaf=20`、`max_depth=None`、`max_iter=100`（外部核查）。正文只说「树大小由三个参数控制」（L62），**一个默认值都没给**。
58. **`max_bins=255`，合法区间 [2,255]**（外部核查）。L64 说「推荐用满 255，这是默认」，但没说这是**硬上限**（实测 `max_bins=256` 报 `InvalidParameterError`）；源码注释 `n_bins = max_bins + 1  # + 1 for missing values` 解释了 L28 的「typically 256 bins」——数值箱 255 + 缺失箱 1。
59. **`categorical_features='from_dtype'` 是默认值**（1.6 起，外部核查）。也就是说：传一个带 `category` dtype 列的 DataFrame，**类别支持会自动开启**；正文只说「可以传 `"from_dtype"`」（L162）。
60. **HGBT 也有 `max_features`，默认 1.0，且是「每节点」抽特征**（外部核查：docstring `versionadded:: 1.4`，原文「Proportion of randomly chosen features in each and every node split」，并说明只从 `interaction_cst` 允许的特征里抽）。正文提到它时只说「用小的 `max_features` 可显著缩短运行时间」（L376），既没给默认值，也没说抽样层级。
61. **`l2_regularization` 罚的是叶子值，且分母带 Hessian**：叶子值 $w_k=-\sum g_i/(\sum h_i+\lambda)$（官方 A2 折叠块，剪藏只剩目标函数 L68–L70）。因此平坦区域（Hessian 小）的叶子被罚得更重——与 Ridge 罚系数的几何不同。另：**分箱阶段的分位数不看 `sample_weight`**（L136）。
62. **HGBT 的损失默认值**：回归 `'squared_error'`、分类 `'log_loss'`（外部核查）；`quantile` 需同时给 `quantile` 参数（源码 `quantile=None`）。正文列了五种回归损失（L52–L58），但没说哪个是默认（只在列表里写「which is the default loss」）。
63. **`GradientBoosting*.criterion` 在 1.9 弃用、1.11 移除**（外部核查：`.. deprecated:: 1.9`；传非默认值会 `FutureWarning`）。而正文**完全不提 `criterion` 参数**，只讲 `max_depth`/`max_leaf_nodes`（L337–L343）——按本节读代码不会知道这个参数正在消失。
64. **`GradientBoosting*` 默认不早停**：`n_iter_no_change=None`、`tol=1e-4`（外部核查），与 HGBT 的 `n_iter_no_change=10` 相反。这使「同样的 `n_estimators=100`」在两个实现里意味着不同预算（L72 vs 外部核查）。
65. **`GradientBoosting*` 默认 `max_depth=3`**（外部核查），而正文示例用 `max_depth=1`（L278、L295）。把示例参数误当默认值，会把容量高估 2 阶交互（L337）。
66. **`RandomForest*` 的 `max_features` 按任务分叉**：分类 `'sqrt'`、回归 `1.0`（L493 作为「经验好值」给出；源码默认同）。同一份代码换任务就换了随机性强度，且 `1.0` 等价于 bagged trees（L493）。
67. **RF 平均的是概率，不是投票**（L442）。实测 1.6.1：7 棵树的 `predict_proba` 与各树 `predict_proba` 的算术平均**完全相等**（差值 0.0）。因此 `predict` 是概率 argmax，平票看概率和。
68. **`ExtraTrees*` 默认 `bootstrap=False`**（L493），所以 `oob_score=True` 会直接报 `ValueError`（实测 1.6.1）。想用 OOB 必须显式 `bootstrap=True`，但那就不再是标准 ET。
69. **`Bagging*` 默认只有 10 个基模型**，且 `estimator=None` → `DecisionTree*`，`max_samples=None` → 1.0（外部核查；实测 `max_samples` 属性为 1.0）。正文只说「用 `max_samples`/`max_features` 控制子集大小」（L583），不给默认值。
70. **`VotingClassifier` 默认 `voting='hard'`**（外部核查）：不开 `voting='soft'` 就没有概率平均。另有三个正文未提的默认值：`weights=None`（等权）、`flatten_transform=True`（`transform` 把各模型概率拼成 `(n, n_classifiers×n_classes)`，设 False 则返回 list）、`n_jobs=None`。
71. **硬投票平票按类标签升序取**（官方 1.11.4.1，**剪藏整节丢失**；实测 1.6.1：4 个分类器投 2,2,1,1 → 返回 `1`）。重编码标签会改变预测结果，行顺序不会。
72. **`Stacking*` 的三个默认值**（外部核查）：`cv=None` → 5 折（实测 1.6.1 的报错信息即 `n_splits=5`）；`final_estimator=None` → 分类 `LogisticRegression`、回归 `RidgeCV`；`passthrough=False`（基模型原始特征不进入第二层）。
73. **Stacking 二分类会丢一列概率**（L795–L796）：`stack_method='predict_proba'` 时两列共线，只保留一列。实测 `transform` 形状为 `(n, n_estimators)`。若下游按「每模型两列」估特征数，会错。
74. **AdaBoost 的默认值不对称**：`n_estimators=50`（不是 100）；分类器基学习器是 `max_depth=1`、回归器是 `max_depth=3`、回归 `loss='linear'`（外部核查 + 实测）；`algorithm` 参数已不存在（SAMME.R 已移除，只剩 SAMME）。正文只写「默认是 decision stump」（L860），对回归器不成立。
75. **`RandomTreesEmbedding` 默认 `max_depth=5`**（外部核查）：编码维度上界 `n_estimators * 2**max_depth` = 100×32 = **3,200**（L530），`sparse_output=True`。正文给了维度公式，但没给默认深度。
76. **`warm_start=True` 的语义是「复用解并追加树/轮」，不是「增量学新数据」**：RF/ET（L545）、GBRT（L311）、Bagging（签名）、HGBT（源码 docstring）都如此；且 RF 会保留内部随机状态，所以「一次 20 棵 = 10+10」（L563；实测 1.6.1 完全一致）。L570 提醒这与 `random_state` 的常规语义（重复调用结果相同）不同。

feature_selection 一侧（引用 `[[sklearn-feature-selection]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

77. **`SelectFromModel` 的 `threshold=None` 有两套语义**（外部核查 + 实测）：估计器带 L1 惩罚（显式或隐式，如 `Lasso`）→ 阈值 `1e-5`（等于「保留所有非零系数」）；否则 → `"mean"`（树上约留一半）。正文只说「below the provided `threshold`」（L93），**完全不说默认值**。
78. **`SelectKBest` 的 `k` 默认 10、`SelectPercentile` 的 `percentile` 默认 10**（外部核查）：正文的两个 doctest 都显式传了 `k=2`（L57），看不出默认。`k > d` 时静默全留（实测 `k=10`、4 列 → 4 列），`k=0` 合法并返回 0 列。
79. **`GenericUnivariateSelect()` 的默认 `mode='percentile'`、`param=1e-05`**（外部核查 + 实测）：iris 上只保留 **1 列**——不传参等于近乎全删。正文只在 L46 说它「allows to perform univariate feature selection with a configurable strategy」，不给默认策略。
80. **`SelectFpr`/`SelectFdr`/`SelectFwe` 的 `alpha` 同名不同义**（外部核查源码）：`SelectFpr` 是 `pvalues < alpha`（**逐特征**，不校正）；`SelectFdr` 是 Benjamini–Hochberg；`SelectFwe` 是 `pvalues < alpha/n_features`（Bonferroni）。实测 200 个特征（198 噪声 + 2 真信号）：Fpr 留 11 列（9 列噪声）、Fdr/Fwe 留 2 列。正文只给三个名字（L45），不给公式也不给默认 `alpha=0.05`。
81. **`VarianceThreshold` 的阈值单位是方差**（L14 只说「variance doesn't meet some threshold」）：`StandardScaler` 之后各列方差为 1，`VarianceThreshold()` 一个都不删（实测）。默认 `threshold=0.0` 只删零方差列。
82. **`RFE(n_features_to_select=None)` → 保留一半；`RFECV` 没有 `n_features_to_select` 参数**（外部核查）：`RFECV` 只有 `min_features_to_select=1`，个数由折平均分 argmax 决定（L84）。实测 iris + `LogisticRegression` → `n_features_=4`（**全部保留**）。
83. **`SequentialFeatureSelector` 默认 `n_features_to_select='auto'` → 一半**（外部核查 + 实测 4→2），`direction='forward'`、`cv=5`；前向与后向结果一般不同（实测 iris 前向 `[0,1]` vs 后向 `[1,3]`）。正文对这两个参数的默认值**一个都没提**（L155–L157）。
84. **`prefit=True` 时不 `fit` 就读 `threshold_` 会报错**（实测 1.6.1）：`transform` 走 `_get_support_mask`（有 `estimator_`→`estimator` 回退），而 `threshold_` 直接用 `self.estimator_` → `AttributeError`。正文的 doctest 正是这种写法（L113、L140）。
85. **常量特征的打分是 `NaN`，互信息是随机估计量**（实测）：`f_classif` 对常量列打 `Features [i] are constant.` 并给 `NaN`（不会被选入、不报错）；`mutual_info_classif` 默认 `random_state=None`，两次调用最大差 0.058——比较「MI 选中了谁」必须固定种子。
86. **`Pipeline` 内外差 0.4 个准确率**（实测）：120 样本 × 20,000 特征、纯噪声标签，选择放进 `Pipeline` 的 5 折 CV = 0.508（机会水平），先在全量数据上选择再 CV = **0.900**；有信号时 0.792 vs 0.950。L165 只说「recommended way is to use a `Pipeline`」，**不给理由**（不提泄漏）。

neural_network 一侧（引用 `[[sklearn-neural-networks]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

87. **`hidden_layer_sizes=(100,)` 是默认架构**（外部核查）：一个隐藏层、100 个神经元，**不是「无隐藏层」**。用户指南只给 `(5,2)`、`(15,)` 两个示例，不给默认值。
88. **`solver='adam'` 是默认，而来源建议小数据用 L-BFGS**（L149；外部核查签名）。所以小数据上必须显式写 `solver='lbfgs'`；默认参数不会替你做这个选择。
89. **`alpha=1e-4`，且只罚 `coefs_`、不罚 `intercepts_`**（外部核查 1.9.1 签名与 1.6.1 源码）。来源建议的网格 `10**-np.arange(1,7)`（L148）包含默认值；`alpha` 不能跨架构/跨估计器比较。
90. **`max_iter=200`、`tol=1e-4`**（外部核查）：默认 `early_stopping=False`，所以 200 是硬上限。实测默认参数在 1000×20 数据上 `n_iter_=200` 并抛 `ConvergenceWarning`；`tol=1e-1` → 12 轮、`1e-7` → 2000 轮。
91. **`early_stopping=False`、`validation_fraction=0.1`、`n_iter_no_change=10`**（外部核查）：早停从**训练集内部**再切 10%，不是测试集；`n_iter_no_change` 在无早停时看训练损失、有早停时看验证分数。实测同一数据 `early_stopping=False` → 395 轮，`True` → 34 轮。
92. **`batch_size='auto'` 等价于 `min(200, n_samples)`，`shuffle=True`**（外部核查源码）：小数据上等于全批量；`batch_size` 大于样本数会被截断并 warning。
93. **SGD/Adam 专属参数不少**（外部核查）：`learning_rate='constant'`、`learning_rate_init=0.001`、`power_t=0.5`（仅 invscaling）、`momentum=0.9`、`nesterovs_momentum=True`（仅 SGD）、`beta_1=0.9`、`beta_2=0.999`、`epsilon=1e-8`（仅 Adam）。正文一个默认值都没给。
94. **`MLPRegressor` 在 1.7 起有 `loss={'squared_error','poisson'}`，默认 `squared_error`**（外部核查 API 页；`poisson` 用 log-link 且要求 $y\ge0$）。用户指南 1.17.3 仍只说平方误差（L95–L97），这是文档落后于 API 的一处。
95. **`partial_fit` 只在 SGD/Adam 下存在**（L125–L129；实测 `lbfgs` 对象 `hasattr(...,'partial_fit') == False`）。`partial_fit` + `early_stopping=True` 直接 `ValueError`（外部核查源码）。
96. **多标签用逐类 logistic + 固定 0.5 阈值**（L69；实测 `predict` 返回 0/1 矩阵，`classes_=[0,1]`、`n_outputs_=2`、`out_activation_='logistic'`）。多分类才是 softmax；不要用 `predict_proba` 的列和判断多标签。
97. **`warm_start=True` + `max_iter=1` 的 `n_iter_` 每次归 1**（L153–L162；实测连续 10 次 `fit`，`n_iter_` 都是 1，但 `coefs_` 每次都在变）。`n_iter_` 是本次 fit 的迭代数，不是累计值。
98. **`coefs_`/`intercepts_` 的形状**（L23；实测）：`(5,2)` 二分类 → `[(2,5),(5,2),(2,1)]`、`n_layers_=4`；`()` → `n_layers_=2`、`[(m,1)]`；三分类时最后一维是 3，`out_activation_='softmax'`。
99. **`max_fun=15000` 只用于 L-BFGS**（外部核查 API 页）：它是函数调用上限，不是迭代上限；SGD/Adam 用 `max_iter` 与 `n_iter_no_change`。
100. **MLP 没有 `feature_importances_`，`predict_proba` 也不是校准概率**（本库判断）：解释要走 permutation importance/部分依赖，校准要走 `sklearn.calibration`；两者都在未入库章节。另有 L10–L11 的定位：无 GPU、不适合大规模应用。

manifold 一侧（引用 `[[sklearn-manifold-learning]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

101. **`n_neighbors=5` 是本模块全部估计器的默认**（外部核查），而用户指南**一个默认值都没给**。$k=5$ 正是官方 digits 示例的取值，也正是 MLLE/HLLE/LTSA 塌成细丝的配置（图 7–9）。
102. **`n_components=2` 是全部估计器的默认**（外部核查）：这是个**可视化**默认值，不是建模选择；下游若需要 $d$ 维特征，不改这个参数就会一直拿到 2 维。
103. **`LocallyLinearEmbedding.reg=0.001`**（外部核查）：正文只说「an arbitrary regularization parameter $r$, chosen relative to the trace」（L78），**不提参数名与默认值**。实测（S-curve、$k=10$、$d=2$）trustworthiness：$10^{-6}$→0.9104、$10^{-3}$→**0.9861**、$0.1$→0.9521、$1.0$→0.9519。默认值恰好在甜点上，但两个方向都会掉。
104. **`SpectralEmbedding.n_neighbors=None` 等于 $n/10$**（外部核查源码 + 实测：$n=200\to20$、$500\to50$、$1000\to100$）：**邻居数随数据量增长**，与“固定 $k$”的直觉相反。正文完全不提该参数。
105. **`SpectralEmbedding(affinity='nearest_neighbors')` 遇稀疏矩阵会静默回退 `rbf`**（实测 1.6.1，带 `UserWarning: … falling back to rbf affinity`）：模型名没变、图被换掉。
106. **`MDS` 的三处默认值都在近两个版本里变过**（外部核查 API 页）：`metric`→`metric_mds`（1.8 改名）、`n_init` 4→**1**（1.9）、`eps` 1e-3→**1e-6**（1.7，伴随收敛判据 bugfix）；`dissimilarity` 1.8 弃用、1.10 移除；`init` 1.8 新增（默认值将在 1.10 变 `classical_mds`）。
107. **`MDS.stress_` 的量纲随配置变**（实测 1.6.1，同一组 3D 距离）：metric+`'auto'` → 780.03（raw）、non-metric+`'auto'` → **0.29784**（Stress-1）、non-metric+`False` → 140.16。跨配置、跨版本比较 `stress_` 都会得出错误结论；报告时必须同时写 `metric_mds` 与 `normalized_stress`。
108. **`MDS.n_init` 在 1.6.1 仍默认 4，且它真的分高低**（实测）：同一数据 `n_init=1` → stress 431.06（100 轮），`n_init=4` → **419.09**（66 轮）。1.9 的默认 1 会让同一份代码的结果变差。
109. **`ClassicalMDS` 没有 `random_state`、`max_iter`、`n_init`**（外部核查签名）：它是特征分解，不是迭代法。同一数据的图标题时间是 0.105s vs metric MDS 2.393s（快 23×）——能当基线就别先上迭代法。
110. **`TSNE.learning_rate='auto'` 的实现是 $\max(N/\text{early\_exaggeration}/4,\ 50)$**（1.6.1 源码 + 实测：$N=500$→50、$N=8000$→166.67），**不是官方折叠块写的 $N/\text{ee}$**；小样本上被 50 的下限卡住。
111. **`TSNE.method='barnes_hut'` 默认 + `n_components<=3` 硬限**（实测 `n_c=4` → `ValueError: 'n_components' should be inferior to 4 …`）；精确法在 500 样本上 KL 更低（0.33716 vs 0.39931），但只支持稠密输入。
112. **`TSNE.init='pca'` 是默认（1.2 起）**，因此**稀疏输入会在 `init` 处报错**，而不是在 `method` 处（实测 10）。要喂稀疏就必须 `init='random'`；`random_state=None` 则完全不可复现。
113. **`Isomap` 允许 `n_components > n_features`**（实测 64 维数据上 `n_c=70` 返回 `(500,70)`），`LocallyLinearEmbedding` 不允许（`ValueError: output dimension must be less than or equal to input dimension`）；`n_c == n_samples` 时报的是「negative eigenvalues」，不提示真实原因。
114. **`LocallyLinearEmbedding` 的坐标不可位级复现**（实测：arpack 两次 `fit` 差 3.39e-06，arpack vs dense 差 0.2755，但 trustworthiness 0.9196 vs 0.9197）；`Isomap` 与固定 `random_state` 的 `SpectralEmbedding` 则逐位相同。
115. **`trustworthiness` 的 `n_neighbors` 默认 5**（外部核查）：**评估指标自带一个 $k$**，它不一定等于建模用的 `n_neighbors`；两个数不同就会得到不同的「保真度」。

聚类一侧（`sklearn.cluster` + `sklearn.metrics` 的聚类指标；引用 `[[sklearn-clustering]], L…`；标「外部核查」的取自官方 API 页，不在剪藏内）：

116. **`KMeans.n_init='auto'` 等于「只跑一次」**（外部核查：`init='k-means++'` → 1 次、`init='random'` → 10 次；1.4 起默认 `'auto'`）。实测 iris、$k=8$、10 个种子：`n_init=1` 的 inertia 32.77±2.06，`n_init=10` 为 30.21±0.19。`MiniBatchKMeans` 的 `'auto'` **取反**：`random` → 3 次、`k-means++` → 1 次。
117. **`KMeans.tol` 是「按数据方差归一」的相对量**（实测 + 源码）：源码 `_tolerance(X, tol) = tol * mean(var(X, axis=0))`，比较量是各质心**位移平方和**。X 乘 1000 后 `n_iter` 不变，证明尺度不变；`tol=1.0` 时 1 轮就停（inertia 78.86→82.26）。L71 的「centroids move less than the tolerance」省略了这一层。
118. **`MiniBatchKMeans.tol=0.0` 是默认**（外部核查）：早停只由 `max_no_improvement=10` 控制；`batch_size` 的默认 **1.0 起从 100 改为 1024**。
119. **`Birch.n_clusters=3` 是默认**（外部核查），而正文只讲 `n_clusters=None`（L377）。实测 iris：`None` → 13 个子簇、默认 → 3 个全局簇；`subcluster_centers_` 形状 `(13, 4)`。
120. **`DBSCAN.eps=0.5` 是默认，而文档自己说不能留**（L290）。实测 iris（`min_samples=5`）：`eps` 0.05/0.5/1.0/10 → 簇数 0/2/2/1、噪声 150/17/0/0。
121. **DBSCAN 的标签受样本顺序影响**（这条只在官方丢失的 `Implementation` 折叠块里，本机已复现）：两条间隔 0.1 的链 + 一个边界点，`eps=0.35`、`min_samples=4`，顺序 `A,M,B` 与 `B,M,A` 给出不同的边界点归属。
122. **`HDBSCAN.min_samples=None` 等于 `min_cluster_size`**（外部核查）：实测 `min_cluster_size=10` 与 `min_cluster_size=10, min_samples=10` 标签逐位相同。`min_cluster_size` 5/20/50 → 噪声点 0/11/24（两团数据）。
123. **`HDBSCAN.store_centers=None` 是不存中心**（外部核查）：要 `centroids_` 必须显式设 `'centroid'`；`copy` 在 1.9.1 显示 `'warn'`，**1.10 起默认变 `True`**。
124. **`OPTICS.max_eps=inf` 是默认**（外部核查），此时可用 `cluster_optics_dbscan(eps=…)` 对任意 `eps` 事后抽取；实测 iris 上 `eps=0.5` 与 `DBSCAN(eps=0.5, min_samples=5)` **完全一致**（2 簇 + 17 噪声）。`cluster_hierarchy_` 是属性（1.6.1 为 ndarray），L357 称其为「parameter」。
125. **`SpectralClustering.affinity='rbf'` + `gamma=1.0` 是默认**（外部核查），而正文示例全用 `'precomputed'`/`nearest_neighbors`；`n_init=10` 只在 `assign_labels='kmeans'` 时生效。
126. **`assign_labels='discretize'` 不是「100% 可复现」**（与 L168 冲突）：1.6.1 的 `discretize(vectors, *, random_state=None)` 吃 `random_state`；同一嵌入 rs=0 vs 1 给不同标签，均匀噪声 $k=8$ 上 6 个种子出现 3 种划分。`'cluster_qr'` 则给出逐位相同的标签值。
127. **`AgglomerativeClustering.n_clusters=2`、`linkage='ward'`、`metric='euclidean'` 是默认**（外部核查）；Ward 配曼哈顿直接 `ValueError`（L204 是硬约束）。`compute_distances=True` 要求 `n_clusters=None`。
128. **四种 linkage 的「富者愈富」可量化**（实测 iris、$k=3$）：Ward `[64,50,36]`、average `[64,50,36]`、complete `[72,50,28]`、**single `[98,50,2]`**；silhouette 0.5543/0.5542/0.5136/0.5121。
129. **`connectivity` 会同时改结果与速度**（L227 主张 + 实测）：iris 3 次 `fit` 均值 8.68ms → 5.49ms（$n=150$）。但配 single/average/complete 会放大「富者愈富」（L234 的 `[!warning]`）。
130. **`BisectingKMeans` 的 `init` 默认 `'random'`、`n_init=1`、`bisecting_strategy='biggest_inertia'`**（外部核查），而 L265 说它「by design」无法享受 `k-means++`。实测 digits、$k=10$：BKMeans inertia 1,238,977（`biggest_inertia`，0.104s）vs 1,237,640（`largest_cluster`，0.086s）vs KMeans 1,165,189——「largest_cluster 更准更快」成立，但「簇更均匀」（L278）在 3 个种子上方向不稳定。
131. **`MeanShift.bandwidth=None` 会调用 `estimate_bandwidth`**（L128 + 外部核查，`quantile=0.3`）；**没有 `bandwidth_` 属性**（1.6.1），估出的值事后读不到。实测 iris：`quantile=0.3` → 带宽 1.2021 → 2 簇；`quantile=0.5` → 2.3404；`bandwidth=3` → 1 簇。
132. **`AffinityPropagation.preference=None` = 相似度中位数**（外部核查 + 实测）：iris 上 `preference=-2.3302` → **7 簇**（`n_iter_=28`）；`preference=-10` → 5 簇。`affinity='euclidean'` 内部用的是**负平方**欧氏距离。
133. **AMI/NMI 的 `average_method` 默认 0.22 起是 `'arithmetic'`**（外部核查），而剪藏的 §2.3.11.2 doctest 仍是旧口径：文档 0.22504 / −0.10526，1.6.1 实测 **0.29879 / −0.16667**（用 `average_method='max'` 才得回 0.22504）。
134. **`v_measure_score` 与 `normalized_mutual_info_score`（算术平均）是同一个数**（实测逐位相同 0.5158037429793889）：同时报两个等于重复；`v_measure_score` 的 `beta=1.0` 是默认。
135. **`silhouette_score.sample_size=None` 是不采样**（外部核查）：$n$ 大时是 $O(n^2)$ 距离矩阵；`silhouette_score`/`calinski_harabasz_score`/`davies_bouldin_score` 都**不做机会校正**，只有 ARI/AMI 做（L395、L448）。

decomposition 一侧（引用 `[[sklearn-decomposition]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

136. **`PCA` 只中心化、不缩放**（L16）。`mean_` 是训练集均值；要缩放必须显式 `StandardScaler`。实测 iris 上 2 维 PCA + logistic 的 5 折 CV：原始 0.96、标准化 0.9133——缩放是模型选择，不是必然改进。
137. **`PCA(n_components=None)` 等于 `min(n_samples, n_features)`**（外部核查 1.9.1 签名），不是 2。默认 `whiten=False`、`svd_solver='auto'`；`whiten=True` 让各轴方差为 1（实测 0.9933/0.9933 vs 不白化 4.2001/0.2411）。
138. **`TruncatedSVD` 默认 `n_components=2` 且不中心化**（外部核查 + 实测）：对已中心化数据与 `PCA` 投影最大差 0.0；未中心化时前两个「解释方差比」0.5288/0.4485，与 PCA 完全不同。文本 LSA 用它是**因为**不中心化。
139. **`IncrementalPCA(batch_size=None)` 不是 1，也不是全量**：1.6.1 源码取 `5 * n_features`（外部核查）；实测 batch=150（全量）与 PCA 逐位相同，batch=10/30 坐标最大差 0.01271/0.00970。内存随 batch，不随 $N$。
140. **`SparsePCA(alpha=1)`、`MiniBatchSparsePCA(batch_size=3)`**（外部核查）：`alpha` 越大越稀疏但重构越差（实测 0.01/0.1/1/10 → 零 3/3/8/10，误差 1.903/1.911/2.4/10.721）。`MiniBatchSparsePCA` **没有 `partial_fit`**（实测 False），它的「online」是沿特征方向。
141. **`KernelPCA` 默认 `kernel='linear'`、`eigen_solver='auto'`、`fit_inverse_transform=False`**（外部核查）：不显式设 `fit_inverse_transform=True` 就**不能** `inverse_transform`。反投影用 kernel ridge，是近似（L118；实测 RBF 重建误差 6.9775 vs 线性 PCA 3.8993）。
142. **`SparseCoder` 的 `fit` 是空操作**（实测 1.6.1）：`hasattr` 为 True，源码是「Do nothing and return the estimator unchanged」。默认 `transform_algorithm='omp'`、`split_sign=False`；它**不学习字典**，不是「不能进 Pipeline」。
143. **`DictionaryLearning` / `MiniBatchDictionaryLearning` 默认 `alpha=1`、`transform_algorithm='omp'`、`split_sign=False`**（外部核查）：正文写作 `split_code`（L160），API 只有 `split_sign`；`split_sign=True` 把编码从 `(n, k)` 变成 `(n, 2k)` 且非负（实测 `(3,8)`→`(3,16)`）。
144. **`MiniBatchDictionaryLearning` 有 `partial_fit`，但默认没有停止条件**（L198；外部核查 `tol=0.001`、`max_no_improvement=10`）：轮数靠 `max_iter`，不是「自动收敛」。`batch_size=256`（1.3 起；更早是 3）。
145. **`FactorAnalysis(random_state=0)` 默认固定种子**（外部核查）：`rotation=None`、`svd_method='randomized'`；噪声方差是**逐特征**的（实测 `[0.016,0.098,0.004,0.041]`），这是它与 PCA 的核心差别（L246–L258）。FA 的载荷**不保证正交**（L254）。
146. **`FastICA` 的 `whiten='unit-variance'` 是默认（1.3 起）**（外部核查）：`whiten=True` 在 1.6.1 直接 `InvalidParameterError`；`whiten=False` 会**忽略 `n_components`**，实测输出回到 4 维、`components_` `(4,4)`，尽管 `n_components` 属性仍是 2。要降维必须显式白化。
147. **`NMF(n_components='auto')` 自 1.6 起是默认**（外部核查 API 页）：`init=None`、`solver='cd'`、`beta_loss='frobenius'`、`max_iter=200`。`n_components='auto'` 如何选，1.9.1 文档没有解释——这是一个高影响、低透明度的默认值。
148. **NMF 的 `alpha_W=0.0`、`alpha_H='same'`、`l1_ratio=0.0`**（外部核查）：L1/L2 先验按 `n_features`/`n_samples` 缩放；`alpha_H='same'` 表示与 `alpha_W` 相同。不同 `beta_loss` 的 `reconstruction_err_` 不可比（实测 β=2/1/1.5/0 → 3.9579/2.475/3.091/1.6254）。
149. **NMF 的 `init` 与 `solver` 联合决定零**（L304 + 实测）：`solver='mu'` 无法更新初始化中的零；`nndsvd` + `mu` 留 5 个零、误差 4.4395，`nndsvda`/`nndsvdar`/`random` 无零、误差 2.3878/1.4714/2.0252。**不要只固定 `random_state` 而不固定 `init`。**
150. **`MiniBatchNMF` 的 `batch_size=1024`、`forget_factor=0.7`、`fresh_restarts=False`**（外部核查）：有 `partial_fit`（实测连续 5 次后 `components_` `(3,4)`）。`LatentDirichletAllocation` 默认 `n_components=10`、`learning_method='batch'`、`max_iter=10`；`components_` 是主题-词、`transform` 是文档-主题（实测 `(2,4)`/`(6,2)`，每行和 1）。

covariance 一侧（引用 `[[sklearn-covariance]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

151. **`ShrunkCovariance.shrinkage=0.1` 是任意默认值**（外部核查），而正文从不给默认值；公式为 $(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$（L32，实测逐位复现）。$\alpha=1$ 时对角全部等于 $\mathrm{Tr}\hat\Sigma/p$、非对角 0（实测 0.88836 = 0.88836）。
152. **经验协方差是 $1/n$ 而不是 $1/(n-1)$**（实测）：对角 = `np.var(ddof=0)`，与 `np.cov` 的比是 49/50 = 0.98（$n=50$）。与 1.2 节 QDA 的 $\frac{1}{n}X_k^TX_k$ 同一约定。
153. **`assume_centered=True` 的意思是「不减均值」**，且 `location_` 被**静默置零**（实测）：在均值 1–4 的数据上误设，对角 1.348→**2.550**，`precision_`/`score` 全跟着错。文档 L18 只说要零均值，不说这层后果。
154. **`store_precision=True` 是默认，`precision_` 是 `pinvh` 伪逆**（外部核查 docstring 原文）：$n=10,p=20$（秩 9）时 fit 不报错、`score` = `-inf`。`store_precision=False` 时 `precision_` 属性存在但值为 `None`，`get_precision()` 仍可算。
155. **`LedoitWolf.block_size=1000` 只是内存参数**（实测）：1000/100/17 三种分块结果**逐位相同**（maxdiff 0.0）。
156. **`LedoitWolf`/`OAS` 精确等于某个 `ShrunkCovariance`**（实测）：`allclose(cov, ShrunkCovariance(shrinkage=shrinkage_).covariance_)` 为 `True`。→ 三个估计器 = 一个算子 + 三种 α 来源。
157. **LW 的两个退化点**（实测）：$p=1$ → `shrinkage_=0`（源码分支）；$n=2$ → **α 恒为 0**（6/6 种子，居中后 $x_2=-x_1$ 使 $\hat\beta\equiv0$，是代数恒等式）；$n=1$ → 两条 `UserWarning`、协方差全零、`score` = `-inf`。
158. **`GraphicalLasso.alpha=0.01` 是默认，且 `alpha=0` 合法**（实测）：`alpha=0` 的 `covariance_` 与 `EmpiricalCovariance` **逐位相同**；`alpha<0` 抛 `InvalidParameterError`（报错文字写 `[0.0, inf]`）。
159. **GLasso 的 $\|K\|_1$ 只罚非对角**（外部核查源码）：`alpha * (abs(K).sum() - abs(diag(K)).sum())`，与 L105 一致。这是本库唯一「L1 作用在矩阵上但排除对角」的例子。
160. **`GraphicalLassoCV.alphas=4` 是路径点数，不是候选列表**（外部核查源码）：$\alpha_{\max}=$ 经验协方差非对角元素最大绝对值，在 $[10^{-2}\alpha_{\max},\alpha_{\max}]$ 上取 4 个对数点，再 refine 4 轮；`cv=None` → 5 折。选出的 α 系统性偏小（实测 34–82 条边 vs 真值 12）。
161. **`MinCovDet.support_fraction=None` 的默认 h 是 $\lceil(n+p+1)/2\rceil$**（外部核查源码），而 docstring 写成 `(n+p+1)/2 * n_samples`（**写错了**，按字面算是 $n^2$ 量级）。$n=200,p=3$ → h=102。
162. **`MinCovDet.support_` 不是 h**（实测 + 源码）：`reweight_covariance` 把它覆盖成重加权掩码（阈值 $\chi^2_p(0.975)$），实测 178/200；h 只在 `raw_support_` 里。校正因子 = $\mathrm{median}(\mathrm{dist})/\chi^2_p(0.5)$，使 $\det$ 从 0.0762 变 0.7365。
163. **`MinCovDet` 有 `n>p` 的硬前提，但报错不友好**（实测）：$n=4,p=10$ 抛 `ValueError: kth(=7) out of bounds (4)`；1.9.1 签名已无 `correction` 参数（移除），只剩 `support_fraction`/`random_state`。
164. **`LDA.shrinkage` 的两条路径语义不同**（外部核查源码 + 实测）：`'auto'` = `StandardScaler` → `ledoit_wolf` → 按 `scale_` 还原（**对特征缩放等变**，decision function 差 8.9e-15）；手设 float = `shrunk_covariance`（目标是 $\frac{\mathrm{Tr}}{p}I$，**不等变**，同一数据缩放后 decision function 差 7.22）。`shrinkage=1` 实测是 $133.3276\,I$，**不是** 1.2 节文档所说的「方差对角阵」。

异常检测面（2.7 节：`svm` / `linear_model` / `covariance` / `ensemble` / `neighbors`；引用 `[[sklearn-outlier-detection]], L…`；默认值取自 1.9.1 API 页签名，标「实测」的在 1.6.1 上执行）：

165. **`IsolationForest(contamination='auto')` 的 `offset_` 是常数 −0.5**（外部核查 API 页），不是「按 10% 分位」。实测高斯数据上报出 13.9%/14.9%/15.4%（$n$=1e3/1e4/1e5）；给定 `contamination=c` 后才变成**精确**的 $c\cdot n$（1,100 样本上 55/110/220/550）。
166. **`LocalOutlierFactor(contamination='auto')` 的 `offset_` 是常数 −1.5**（外部核查），实测在高斯数据上只报 **0.2–1.4%**。`'auto'` 与 10% 无关。
167. **`EllipticEnvelope(contamination=0.1)` 是唯一的 float 默认，且不接受 `'auto'`**（外部核查 + 实测抛 `InvalidParameterError`）。三个同族估计器的同名参数语义不同——这是本节最容易错的一处。
168. **`contamination` 的合法区间是 `(0.0, 0.5]`**（实测）：`0.0` 与 `0.6` 都抛 `InvalidParameterError`。想「一个不报」只能手改 `offset_`。
169. **`OneClassSVM` 没有 `contamination`**（实测 `TypeError: __init__() got an unexpected keyword argument`）；阈值语义由 `nu` 承担，实测 `nu`=0.01/0.1/0.3/0.5 → 训练集异常率 **6.8/11.4/30.2/50.2%**、支持向量 6.8/12.8/31.0/50.8%。
170. **`OneClassSVM.nu` 的合法区间是 `(0, 1]`**（实测）：`nu=0` 与 `nu=1.5` 都抛 `InvalidParameterError`。
171. **`OneClassSVM.offset_` 实际是 `(1,)` 数组**（实测 1.6.1），而文档写 float；`decision_function`/`score_samples` 才是 `(n,)`。`offset_ = -intercept_`。
172. **`OneClassSVM.gamma` 默认 `'scale'`（0.22 起；之前是 `'auto'`）**（外部核查）：整体等比缩放不变，但**单列**缩放会改结果（一致率 0.89；固定 `gamma=0.5` 时 0.485）。
173. **`IsolationForest.max_samples='auto'` = `min(256, n_samples)`**（外部核查 + 实测：50/200/256/300/5000 → 50/200/256/**256**/**256**）。这是 IF 成本几乎与 $n$ 无关的原因（$10^3$→$10^5$：0.045s→0.140s）。
174. **`IsolationForest(max_features='auto')` 在 1.6.1 已非法**（实测 `InvalidParameterError`，只接受 int 或 `(0,1]` 的 float）；老代码的迁移点。
175. **`IsolationForest` 的每棵树深度固定为 $\lceil\log_2 n\rceil$**（L133 + 实测：6 样本全部 `get_depth()=3`）。
176. **`IsolationForest(warm_start=True)` 只训练新增的树**（实测 10→20）；**`n_estimators` 不变时重复 `fit` 只给 `UserWarning` 且不增树**；**改 `max_samples` 后 `fit` 无任何警告**（`max_samples_` 变 200，旧树仍按 300 训练）。
177. **`LocalOutlierFactor(novelty=False)` 没有 `predict`/`decision_function`/`score_samples`**（实测 `AttributeError: This 'LocalOutlierFactor' has no attribute '…'`，不是 `NotFittedError`）。
178. **`novelty=True` 时 `fit_predict` 抛 `AttributeError`**，且训练集上的 `predict` 与 `fit_predict` 有 1.2% 标签不同、`score_samples` 与 `negative_outlier_factor_` 最大差 0.586。
179. **`LocalOutlierFactor.n_neighbors` 超出 $n$ 时被静默夹到 $n-1$**（实测 330 点上 400 → 329），结果是**一个异常也报不出**。合法下限是 1（`0` 抛 `InvalidParameterError`）。

180. **`LocalOutlierFactor.offset_` 与 `predict` 的计数故意不一致**（实测 $c$=0.1：`nof < offset_` 33 个、`predict` 32 个；$c$=0.2：66 vs 61）——`predict` 重算 `score_samples`。
181. **`LocalOutlierFactor.metric` 换一个就等于换模型**（实测：`euclidean`/`manhattan` 报 24 个，`cosine` 报 8 个）。

`model_selection`（3.1 节）一侧（引用 `[[sklearn-cross-validation]], L…`；默认值取自 1.9.1 API 页签名，行为在 1.6.1 实测）：

182. **`cv=None` 默认 5 折**（0.22 起；之前 3 折）。文档只描述 `cv` 为整数时的行为（L89），**从未写默认折数**。
183. **`cv=int` 在分类器上选 `StratifiedKFold`、在回归器上选 `KFold`**（`check_cv`，实测）。所以同一份数据上 `cv=5` 与 `cv=KFold(5)` 是两个不同的估计量（iris：0.98 vs 0.94）。
184. **所有 splitter 默认 `shuffle=False`**，且此时传 `random_state` 会抛 `ValueError: Setting a random_state has no effect since shuffle is False`（`KFold`/`StratifiedKFold`/`GroupKFold`，实测）。
185. **`KFold(n_splits=n)` ≡ `LeaveOneOut`**（L230 + 实测：150 折与 LOO 的索引与分数逐位相同）——所以「LOO 与 $k$ 折是两种方法」在 $k=n$ 处不成立。
186. **`error_score=np.nan`**（外部核查，0.20 新增）：失败折的分数变成 `nan` + 一条 `FitFailedWarning`，不中断。`error_score='raise'` 才会抛出真正原因（实测：`The number of classes has to be greater than one; got 1 class`）。
187. **`cross_validate.return_train_score` 默认 `False`**（0.19 新增，**0.21 起由 `True` 改为 `False`**，外部核查）。训练分与测试分的 gap 默认不可见。
188. **`cross_validate.return_indices=True` 返回的是 `indices` 字典**（`{'train': [...], 'test': [...]}`），不是 `train_indices`/`test_indices`。按剪藏 L144 的字面理解会 `KeyError`（实测）。
189. **`cross_val_predict` 默认 `method='predict'`**（外部核查）：要概率/分数必须显式传 `method='predict_proba'`/`'decision_function'`，且模型必须真有该方法（默认 `SVC` → `AttributeError`，实测）。
190. **`cross_val_predict` 拒绝非划分的 splitter**（实测 `ShuffleSplit` → `ValueError: cross_val_predict only works for partitions`）。
191. **`groups` 与 `cv=int` 同时传时被静默忽略**，只有一条 `UserWarning: The groups parameter is ignored by StratifiedKFold`（实测）——分组泄漏最常见的来源。
192. **`train_test_split` 默认 `test_size=0.25` 且 `stratify=None`（不分层）**（外部核查 + 实测：150 样本 3 类上测试集比例 0.342/0.421/0.237）。`shuffle=True` 是默认值，所以结果**不可复现**除非给 `random_state`。
193. **`train_test_split` 没有 `groups` 参数**（实测传 `groups=` 抛 `TypeError`），与剪藏 L553「cannot account for groups」一致。
194. **`TimeSeriesSplit` 的 `gap`/`max_train_size`/`test_size` 是 0.24 新增**（外部核查）；默认 `gap=0` 意味着训练末端与测试起点**相邻**，自相关仍在。`max_train_size` 会把「扩张窗口」变成「滑动窗口」（实测 `n_splits=3, max_train_size=3` → `[0,1,2]/[2,3,4]/[4,5,6]`）。
195. **`PredefinedSplit` 只给 0/−1 时只有 1 个划分**（实测 `get_n_splits()` = 1），不是 `KFold` 式的多折；全 `-1` 在构造时不报错。
196. **`permutation_test_score.n_permutations=100` 默认、下界是 ≥1**（外部核查 + 实测 `0` → `InvalidParameterError`）；$p$ 的最小可达值是 $1/(n_{\text{perm}}+1)$（实测 50 次置换 → 0.0196）。
197. **`StratifiedGroupKFold` 的 `shuffle`/`random_state` 不是 keyword-only**，而 `KFold`/`StratifiedKFold`/`GroupKFold` 都是（外部核查）。
198. **`GroupKFold.shuffle=True` 平衡的是「组数」不是「样本数」**（L454 + 实测两种模式都给 `[4,3,3]`）；`StratifiedGroupKFold` 在单类组上开洗牌反而会**加大**类比例偏离（实测极差 0.064 → 0.375）。
199. **`RepeatedKFold(n_splits=1)` 构造时不报错**，`StratifiedKFold(n_splits=1)` 构造即抛 `ValueError`（实测）——校验点不一致。
200. **`cross_val_score` 在没有 `score` 方法的估计器上必须给 `scoring`**（实测 `OneClassSVM` → `TypeError: … should have a 'score' method`）；剪藏 L77 只说默认用 `score` 方法。

搜索器一侧（引用 `[[sklearn-hyperparameter-tuning]], L…`；默认值取自 1.9.1 API 页，标「实测」的在 1.6.1 上执行）：

201. **`refit` 的默认值是 `True`，不是 `None`**（外部核查 + 实测）。用户指南 L241 说「留成默认值 `None` 会报错」，实际报错信息里写的是 `True was passed`。多重指标 + 默认 `refit` → `ValueError`。
202. **`RandomizedSearchCV` 全列表参数时是「不放回」采样，`n_iter` 被静默夹到笛卡尔积大小**（API 页写明 + 实测：5 个候选配 `n_iter=20` → `cv_results_` 只有 5 行；`ParameterSampler({'a':[1,2,3]}, n_iter=4)` 只给 3 条）。用户指南 L61 只说「预算 = 采样次数」。
203. **`error_score=nan` 只覆盖「部分失败」**（实测）：全部候选的全部折失败 → `ValueError: All the 6 fits failed.`，与 `error_score` 取值无关；**refit 阶段失败永远抛错**（外部核查）。
204. **`cv=None` 在四个搜索器里都是 5 折**（实测 `n_splits_=5`，`check_cv` 在分类器上给 `StratifiedKFold(5)`）；而两个 halving 的签名直接把默认写成 `cv=5`。剪藏从不提默认折数。
205. **`RandomizedSearchCV.n_iter=10` 与 `HalvingGridSearchCV.factor=3`**（外部核查）：前者意味着「10 次预算」，后者意味着「每轮资源 ×3、候选 ÷3」。剪藏只在 L61 与 L108 提到这两个数字的语义，没写它们就是默认值。
206. **`min_resources='exhaust'` 是 `HalvingGridSearchCV` 的默认值**（外部核查）。它只出现在剪藏 L181–L186 的 doctest 里，因此「没显式给 `min_resources` 时会发生什么」在剪藏里查不到；实测 200 样本 4 候选 → `min_resources_=66`、`n_candidates_ [4, 2]`。
207. **`HalvingRandomSearchCV` 的 `n_candidates='exhaust'` 与 `min_resources='exhaust'` 互斥**（外部核查 + 实测 `ValueError: n_candidates and min_resources cannot both be set to 'exhaust'.`）；其默认 `min_resources='smallest'` 与 `HalvingGridSearchCV` 的 `'exhaust'` 不同。
208. **两个 halving 的 `cv=5`、`return_train_score=True`**（外部核查 + 实测）：与 `GridSearchCV`（`cv=None`、`False`）不一致，直接导致 `cv_results_` 列数 15 → 25。
209. **`GridSearchCV` 没有 `random_state`**（实测 `TypeError`）；两个 halving 有（它们要抽样/排序）。
210. **`refit=False` 时 `predict` 抛 `AttributeError` 而不是 `NotFittedError`**（实测：`This 'GridSearchCV' has no attribute 'predict'`）；多指标 + `refit=False` 时 `best_params_` 与 `best_index_` **都不存在**。
211. **`refit=callable` 时 `best_score_` 不存在**（实测 1.6.1，`refit=lambda cv: 0`）：`best_index_` 听 callable 的，分数「没有唯一含义」所以干脆不给。
212. **平票取「枚举顺序第一个」**（实测：常数分数估计器 → 6 行全 `rank_test_score=1`、`best_index_=0`）。`ParameterGrid` 的枚举顺序因此是结果的一部分。
213. **两个 halving 的拟合次数永远不少于网格**（实测 8 候选 × 5 折：76 vs 41；3 候选 × 5 折：26 vs 16），且 `best_params_` 与网格只 0/5 一致（5 个种子）。引用「halving 更快」时必须带「候选数 $\gg$ 折数」与「拟合成本正比于样本量」两个前提。
214. **halving 的 `cv_results_` 是长表**（实测多出 `iter`/`n_resources`，同一候选出现多次）；剪藏 L227 只说「与 `GridSearchCV` 类似，只是多些信息」。
215. **halving 没有 `resource_` 属性**（实测 `AttributeError`），只有 `min_resources_`/`max_resources_`/`n_resources_`/`n_candidates_`/`n_possible_iterations_`/`n_required_iterations_`/`n_iterations_`/`n_remaining_candidates_`。
216. **剪藏 L82 的 `from sklearn.utils.fixes import loguniform` 在 1.6.1 上 `ImportError`**（实测）：可移植写法是 `scipy.stats.loguniform`。而 1.9.1 文档页仍在用它——示例与实现脱节，引用 `loguniform` 时必须写来源。
217. **`make_scorer` 的 `greater_is_better` 默认 `True`**（实测 1.6.1 与 1.9.1 签名一致）：包误差函数时必须显式写 `False`，否则搜索器最大化误差。5 点 `alpha` 网格上 `make_scorer(mean_squared_error)` 选出 **`alpha=1e4`**（MSE 6290.0），`'neg_mean_squared_error'` 选出 `alpha=1e-4`（MSE **774.9**）——差 8.1 倍且无警告。
218. **`make_scorer` 的 `response_method` 默认值 1.8 起从 `'default'` 改为 `'predict'`**（外部核查 1.6/1.7/1.8 三版 API 页）：两个默认值对概率型自定义指标都不对——实测 `make_scorer(log_loss)` 给 `[nan, nan, nan]`，只有 `response_method='predict_proba'` 才给 `[0.142, 0.152, 0.156]`。
219. **`scoring=` 层的符号是 scorer 的私有 `_sign`，`metrics` 层不翻转**：`neg_*` 一律 −1。报表函数必须只做一次还原，否则「搜索选出的分」与「报表里的分」符号相反。
220. **字符串评分名绑定响应方法**：`roc_auc` 的偏好序是 `('decision_function', 'predict_proba')`，所以给不给 `probability=True` 会改变它算的东西；`neg_log_loss`/`neg_brier_score` 强制 `predict_proba`。
221. **指标不匹配只变 `nan`，不抛错**（实测）：`scoring='neg_log_loss'` 配 `SVC()` → `[nan, nan, nan]` + 3 条 `UserWarning: Scoring failed … set to nan`（内部 `AttributeError: SVC has none of the following attributes: predict_proba`）；`scoring='roc_auc'` 在三类数据上同样 `nan`（内部 `ValueError: multi_class must be in ('ovo', 'ovr')`）。
222. **`get_scorer_names()` 在 1.6.1 上返回 56 个名字，而 1.9.1 的表只有 43 行**（21 分类 + 9 聚类 + 13 回归）：`matthews_corrcoef`、`positive_likelihood_ratio`、`neg_negative_likelihood_ratio` 可用但 1.6–1.9 四版文档表都没列；另有 12 个 `precision_*`/`recall_*`/`jaccard_*` 由「etc.」后缀行覆盖。
223. **`d2_log_loss_score` 在 1.6 的表里就有，但 1.6.1 的注册表没有它**（函数可 `import`、`get_scorer` 抛 `ValueError: 'd2_log_loss_score' is not a valid scoring value`）；`d2_brier_score` 1.8 才进表、1.6.1 连 `import` 都失败。
224. **`mean_pinball_loss` 的参数叫 `alpha`（默认 0.5），而估计器侧叫 `quantile`**（来源自己为此道歉，[[sklearn-metrics]], L56）：漏传 `alpha` 就静默变成中位数损失。实测 300 样本上 `mean_pinball_loss(alpha=0.99)`：HGB(`quantile=.99`) 1.0367 vs `QuantileRegressor(quantile=.99)` 1.9742。
225. **`balanced_accuracy_score` 默认 `adjusted=False`**：90/10 不平衡、正类全错时默认给 **0.4**，`adjusted=True` 给 **−0.2**；而 1.9 文档的 note 说官方定义「adopt the adjusted version」（[[sklearn-metrics]], L139 的残片）。
226. **`classification_report` 的 `zero_division` 默认仍是 `'warn'`**（1.6.1 与 1.9.1 签名相同）：某类从未被预测 → 3 条 `UndefinedMetricWarning` + precision/recall/f1 全 **0.0**，汇总时被平均掉。
227. **`top_k_accuracy_score` 的 `k > n_classes` 静默返回 1.0**（实测 3 类分数矩阵上 `k=4`）。该行为属剪藏外小节（§3.4.4.3），本页只记现象。
228. **读符号与响应方法只能用私有属性**：`_sign`/`_score_func`/`_response_method` 都是下划线属性，没有公开 API；跨版本可能改名，自动化代码要带 try/except 并在升级时复查。
229. **多指标时键带 `test_` 前缀**：`cross_validate(..., scoring=['accuracy','f1_macro'])` 给出 `test_accuracy`/`test_f1_macro`（+ `fit_time`/`score_time`），没有裸 `accuracy`。
230. **callable scorer 没有签名校验**（实测）：3 参数 `(est, X, y)` 正常；带第 4 个可选参数 `(est, X, y, extra=None)` 也被静默接受并返回常数 0.5——写错的 scorer 不报错，只给常数。
231. **`StandardScaler` 把常数列的 `scale_` 换成 1**（实测 1.9.1/1.6.1：`[[1,5],[1,3],[1,7]]` → `scale_ = [1.0, 1.633]`）：好处是没有 nan，代价是「方差为 0」的信息消失，因此 `VarianceThreshold` 放在缩放之后**一列都不删**。
232. **`MinMaxScaler` 默认 `clip=False`**（1.9.1 签名）：训练集在 `[0,1]` 内时，测试点 `[-3,-1,4]` 变换后是 `[[-1.5, 0, 1.6667]]`（L105–L106 的 doctest 自己就是证据）。`clip=True` 才给 `[[0,0,1]]`。`MaxAbsScaler.clip` **1.8 才加**（1.6.1/1.7 只有 `copy`）。
233. **`RobustScaler` 的 `scale_` 是 IQR，不是标准差**（`quantile_range=(25,75)`、`unit_variance=False` 默认）：实测 95+5 个含离群点样本上 `StandardScaler.scale_ = [8.53, 11.03]` 而 `RobustScaler.scale_ = [1.29, 1.35]`——差 6–8 倍，两者不可互换解释。
234. **`RobustScaler` 能 `transform` 稀疏但不能 `fit` 稀疏**（实测两版相同）：`ValueError: Cannot center sparse matrices: use 'with_centering=False' instead.`。所以它不能进 `Pipeline`，「离线稠密训练 + 线上稀疏推理」会突然失败。
235. **`QuantileTransformer` 的 `n_quantiles=1000` 会被样本量静默夹住**（实测：112 行训练集 → `UserWarning` + `n_quantiles_ = 112`）：小数据上它退化成「每样本一个分位」。
236. **`QuantileTransformer.subsample=10000` 而 `quantile_transform.subsample=100000`**（1.6.1 与 1.9.1 都是如此）：同一算法的类入口与函数入口对「用多少样本估分位」的默认承诺差 10 倍。
237. **分位数变换在训练范围外退化成常数**（实测）：均匀输出给 `0`/`1`，正态输出给 `±5.1993`（对应 L252 的 1e-7 截断）。所以「分位数变换抗异常值」的准确含义是「把所有极端值压成同一个值」。
238. **`PowerTransformer` 的 `standardize=True` 是默认**（外部核查）：实测同一数据的输出均值/标准差为 `0/1`（默认）vs `[-0.155,-0.022,0.045]`/`[0.87,0.93,1.00]`（`standardize=False`）。要按 λ 解释必须显式关掉。**Box-Cox 对非正数据直接 `ValueError`**（Yeo-Johnson 不会）。
239. **`normalize` 的 `axis` 默认 1，而 `scale` 的 `axis` 默认 0**（外部核查两个签名）：前者逐行、后者逐列。写成 `normalize(X, axis=0)` 会按列归一，结果与默认完全不同。
240. **`OrdinalEncoder` 按字典序编号，并把 NaN 直通**（实测 `categories_ = ['female','male']`、含 `np.nan` 的列输出 `nan`；`encoded_missing_value=nan` 是默认）：下游模型会收到 NaN 而不报错；要编码缺失必须显式 `encoded_missing_value=-1`。
241. **`OrdinalEncoder` 强加的顺序会改变模型分数**（本库实测）：非单调效应下 `OrdinalEncoder`+`Ridge` 的 CV $R^2$ = **0.2236** vs `OneHotEncoder`+`Ridge` **0.9845**；**把一个水平改名**（数据不变）后掉到 **0.0236**。
242. **`OneHotEncoder` + `drop` + `handle_unknown='ignore'` 的碰撞**（L450 文档自认、实测确认）：未知类别映射为**全零**，与被 drop 的类别**不可区分**，`inverse_transform` 把它还原成被 drop 的那类（`[['female', None, None]]`）。
243. **`OneHotEncoder` 默认 `sparse_output=True`**（正文从不写）：`KBinsDiscretizer` 默认 `encode='onehot'` 同理，输出是 `csr_matrix`（实测）。接需要稠密输入的环节时必须显式 `sparse_output=False` / `encode='onehot-dense'`。
244. **`KBinsDiscretizer` 的 `strategy` 默认是 `'quantile'`**（外部核查）：正文 L664 只描述三种策略，从不说默认值，而默认并不是 `'uniform'`。另外 `quantile_method` 是 **1.7 新增**（默认 `'warn'`）、**1.9 改默认为 `'averaged_inverted_cdf'`**，于是同一代码在 1.6.1 与 1.9.1 上得到不同 `bin_edges_`（实测 `[-3,-1,2,6]` vs `[-3,-1.5,3,6]`），而剪藏 L647–L649 印出的区间是 **1.6 口径**。
245. **`SplineTransformer` 的默认外推是 `'constant'`，含义是「钉在边界值」而不是「外推为 0」**（实测：`X=10` → `[0,0,0,0,0.167,0.667,0.167]`）；`extrapolation='linear'` 才给线性外推。`handle_missing='error'`（**1.8 新增**）使 NaN 直接抛错。
246. **`FunctionTransformer` 默认 `validate=False`**（外部核查 + 实测）：`[[0, nan]]` 原样穿过，`validate=True` 才抛 `ValueError: Input X contains NaN.`。而剪藏 L828 的 doctest 写的正是 `validate=True`，容易让人以为默认开启。

## 版本边界

- `linear_model` 部分对应**官方 1.8–1.9.1 的 stable 文档**。1.9 重写了 1.1.3.1 gap safe screening 的正文，1.7.2 及更早没有这一小节 → 引用该小节时应注明版本。
- `discriminant_analysis` 部分对应 **1.9.x**：1.9 把 QDA 的 SVD 推导从 $\frac{1}{n-1}X_k^{T}X_k$ 改为 $\frac{1}{n}X_k^{T}X_k$（与实现的最大似然估计一致），1.8 及更早仍是 $\frac{1}{n-1}$。两份剪藏因此可互证版本：1.1 判到 ≥1.8、1.2 判到 1.9.x（见 [[sklearn-lda-qda]] 的版本判定表）。
- `naive_bayes` 部分判为 **1.9.x，但证据弱**：1.8→1.9.1 在本节只改了 `one dimensional` → `one-dimensional`、`through \(x_n\), :` → `through \(x_n\):` 两处措辞。功能层面两版无差别，因此上面 17–22 条的默认值结论不依赖这条判定。
- `svm` 部分对应 **1.7.x – 1.9.1**：本节在 1.7 改过 ovo 措辞，之后 1.7/1.8/1.9.1 **字面完全相同**，剪藏与 1.9.1 逐句一致（见 [[sklearn-svm]]）。
- `neighbors` 部分对应 **1.7.x – 1.9.x，证据弱**：整节唯一可用的版本地标是 `re-used`（1.6）→ `reused`（1.7），以后无改动。已回查官方 1.6/1.7/1.8/1.9 四页确认这一点。本文第 34–44 条的默认值结论**不依赖**该判定：它们来自 1.9.1 签名，并在 1.6.1 上实测一致。
- `tree` 部分对应 **1.9.x（可定为 1.9.1）**：1.8→1.9 有两处可判定的文本改动——新增 ExtraTree「共享同一实现」句（L260）、删除「支持缺失值的准则清单」；且抓取日的 `/stable/` 页与 `1.9/modules/tree.html` 文本逐行相同（页标题 1.9.1）。外部旁证：1.9 弃用 `friedman_mse`、`absolute_error` 开始支持稠密 NaN（见 [[sklearn-decision-trees]] 的版本判定）。
- `feature_selection` 部分对应 **1.9.1**：四条独立措辞判据——「a F-test」→「an F-test」（1.9 才改）、「different number」→「numbers」与「selector are evaluated」→「is evaluated」与「importance … are below」→「is below」（均为 1.6→1.7）、`array([ 0.04..., …])`→`array([ 0.04, …])`（1.6→1.7 的 repr 变化）——全部指向 ≥1.7 且「an F-test」指向 1.9。1.7 与 1.8 两版本节**逐词相同**，故无法再细分，取抓取日 `/stable/` 的页标题 1.9.1（见 [[sklearn-feature-selection]] 的版本判定表）。
- **`feature_selection` 的默认值结论不依赖该判定**：第 77–86 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 79、83、85 条涉及默认策略与随机数（`param=1e-05`、`n_features_to_select='auto'`、`random_state=None`），引用时须带版本。
- `ensemble` 部分对应 **1.9.1**：173 条句子探针中 138 条可逐字比对，1.9.1 页全部命中；1.6/1.7/1.8 分别只有 132/134/137，缺的正是 1.9 新增的 Newton boosting 句（L257）。旁证：1.9 弃用 `GradientBoosting*.criterion`，而正文完全不提该参数（见 [[sklearn-ensembles]]）。
- **`ensemble` 的默认值结论不依赖该判定**：它们取自 1.9.1 源码签名，并在 1.6.1 上实测机制层行为；但 `criterion` 弃用、`categorical_features` 默认值变更（1.6 起 `'from_dtype'`）、AdaBoost 移除 `algorithm`（1.6 起）三条**是版本敏感项**，引用时必须写明版本。
- `ensemble` 的 `max_bins` 是**硬上限 255**（源码约束区间 [2,255]），不是「建议值」；`max_bins=256` 在 1.6.1 上实测报 `InvalidParameterError`，1.9.1 约束相同。
- `neural_network` 部分对应 **1.8.0 – 1.9.1**：剪藏 L15 的集合花括号排除 1.6，L63 的新 repr 与 L67 的 `represent` 排除 1.6，L116 的 `\left[...\right]` 排除 1.7；1.8→1.9 正文逐词 diff 为空，抓取日 `/stable/` 页标题为 1.9.1，故取 1.9.1（见 [[sklearn-neural-networks]] 的版本判定）。外部旁证：`MLPRegressor.loss` 于 1.7 新增，而剪藏正文未更新。
- **`neural_network` 的默认值结论不依赖该判定**：第 87–100 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 90、91、94、99 条涉及默认预算、早停、`poisson` 与 `max_fun`，引用时须带版本。
- `manifold` 部分对应 **1.8.0 – 1.9.1**：82 条句子探针在 1.6/1.7/1.8/1.9.1 四版页的命中数是 61/68/75/75；剪藏含 1.8 新增的 `ClassicalMDS`、三种 MDS、「classical MDS ≈ PCA」与 strain 公式，而 1.8→1.9 的正文逐词 diff 为空，抓取日 `/stable/` 页标题为 1.9.1（见 [[sklearn-manifold-learning]] 的版本判定）。外部旁证：`ClassicalMDS` 是 1.8 新增（#31322）、`MDS` 支持任意度量与 `init` 也是 1.8（#32229）。
- **`manifold` 的默认值结论**来源于 1.9.1 签名，但第 106、108、110 条**涉及版本敏感的默认值**（`n_init` 1.9 变 1、`eps` 1.7 变、`learning_rate` 1.2 起 `'auto'`），引用时必须写明版本；行为在 1.6.1 实测，而 1.6.1 里 `ClassicalMDS` 尚不存在。
- `manifold` 的**复杂度、metric/non-metric 的机制与 t-SNE 的五个参数全在剪藏丢失的折叠块里**（10 块、1,669 词）。因此「三阶段复杂度」「isotonic regression / Stress-1 公式」「`learning_rate='auto'` 的启发式」「Barnes-Hut 四条限制」**不能引剪藏行号**，只能引官方 1.9.1 页。
- `ensemble` 的**数学节（回归 438 词 + 分类 151 词）与两张损失表在剪藏中全丢**，因此第 61、63 条涉及的叶子值公式、负梯度与损失初始值均属外部核查，不能引剪藏行号。
- `neighbors` 的 `algorithm='auto'` 规则与 `leaf_size` 语义**不在剪藏里**（它们在官方页的折叠块中，该剪藏丢了 6 个折叠块中的 5 个）。引用第 38–39、44 条时应引官方 1.9 页，**不能引剪藏**。
- **`svm` 存在文档与 API 错位**：1.9 弃用了 `probability` 参数（外部核查），而用户指南正文仍以 `probability=True` 为正规写法。引用第 23–24 条时必须以 API 页/源码为准，不能引剪藏。
- `gamma` 默认值（`'scale'`）与 `LinearSVC.dual='auto'` 的选取规则**在任何版本的这一节里都没有**，属外部核查所得；它们的行为随版本可能变化，引用时应注明取自 1.9.1。
- 剪藏里 logistic 的惩罚只通过 `C` 与 `l1_ratio` 表达（L473），没有出现 `penalty=` 参数的说明；这与旧版本 docstring 的写法不同。**本文不对 `penalty=` 的弃用状态下结论**，需要时查目标版本的 API 文档。
- `np.float64(...)` 形式的 doctest 输出（L84）要求 NumPy ≥ 2 的 repr 习惯，与代码行为无关。
- `cluster` 部分对应 **1.9.x（可定为 1.9.1）**：126 条句子探针在 1.6/1.7/1.8/1.9 四版页的命中数是 **107/108/110/111**；1.8→1.9 的正文 diff 只有三处（"Allows to assign"→"Allows assigning"、"Egineering"→"Engineering"、页脚版权年），剪藏含 "assigning" 故 ≥1.9；描取日 `/stable/` 与 `/1.9/modules/clustering.html` **字节级相同**（245,702 B），页脚 1.9.1（见 [[sklearn-clustering]] 的版本判定）。
- **`cluster` 的默认值结论不依赖该判定**：第 116–135 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 116、118、119、122、123、130、131、133 条涉及版本敏感的默认值（`n_init` 1.4 变、`batch_size` 1.0 变、`HDBSCAN` 1.3 新增与 `copy` 1.10 变、`average_method` 0.22 变），引用时必须写版本。
- `cluster` 一节丢了 **30 个内容折叠块（3,073 词）+ 16 个 References 块 + 总览表首列**，所以「ARI 的机会校正公式」「AP 的 responsibility/availability」「DBSCAN 的实现与内存」「OPTICS 的复杂度」「BIRCH 的三个用法问答」**不能引剪藏行号**，只能引官方 1.9.1 页。
- `decomposition` 部分对应 **1.9.1**：1.8→1.9 的唯一正文差异是「SVMs」→「kernel approximation」（剪藏 L69 已是后者）；1.7→1.8 的逗号与 LDA 记号（`N_d`/`z_{dn}`/`w_{dn}`）、1.6→1.7 的 `preserving`/`yields`/`datasets` 也都已出现（见 [[sklearn-decomposition]] 的版本判定）。
- **`decomposition` 的默认值结论不依赖该判定**：第 136–150 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 138、143、146、147、150 条涉及版本敏感的默认值（`TruncatedSVD` 的 `n_components`、`MiniBatchDictionaryLearning.batch_size`、`FastICA.whiten`、`NMF.n_components='auto'`、`MiniBatchNMF` 的参数），引用时必须写版本。
- `decomposition` 一节丢了 **4 个内容折叠块（833 词）+ 约 14 条参考文献**，所以「KernelPCA 的 `eigen_solver` 自动规则」「LSA 的 $U_k\Sigma_k V_k^\top$」「NMF 的 `cd`/`mu` 能力边界」「LDA 的生成过程/ELBO」**不能引剪藏行号**，只能引官方 1.9.1 页。
- `covariance` 部分对应 **1.7.0 – 1.9.1**：93 条句子探针在 1.6/1.7/1.8/1.9 四版页的命中数是 **50/54/54/54**，差的 4 条正好是 1.6 的旧措辞（`assume_centered=False` 的旧句、`a simple a convex`、`number of sample`、`is multiple of the identity`）；1.7→1.8→1.9 正文逐词 diff 为空，抓取日 `/stable/` 页标题 1.9.1（见 [[sklearn-covariance]] 的版本判定）。
- **`covariance` 的默认值结论不依赖该判定**：第 151–164 条来自 1.9.1 API 页签名与 1.6.1 实测（两版签名逐个一致）；但第 160、161、163 条涉及版本敏感项（`GraphicalLassoCV` 的网格生成、`MinCovDet` 的 `correction` 移除），引用时必须写版本。
- `covariance` 一节**正文几乎完整、损失集中在文献层**：丢了 [1] Ledoit & Wolf 2004、[2] Chen et al. 2010、[3] Rousseeuw 1984、[4] Rousseeuw & Van Driessen 1999 四条文献与 4 处页内引用标记（L42/L61/L124/L126），其中 **L124 被截成病句**（「introduced by P.J. Rousseeuw in.」）。因此 α 的解析式、MCD 的 breakdown point **不能引剪藏**。
- `covariance` 的剪藏还有 **5 个脚注重复了 631 词**（L143–L182，占 `wc -w` 的 25%），且第二次出现的末尾表被截断；行号引用应取第一份（L44、L46–L51、L139–L141）。
- `model_selection` 部分对应 **1.7.0 – 1.9.1**：133 条句子探针在 1.6/1.7/1.8/1.9 四版页的命中数是 **101/122/122/122**，差的 21 条正好是 1.6 的旧措辞（`5- or 10- fold`、`equal number`、`similar as`、`tests set`、分层族整段旧版、置换检验里的 `classifiers`）；1.7→1.8 与 1.8→1.9 的正文逐句 diff **为空**，抓取日 `/stable/` 与 `/1.9/` **字节相同**（169,672 B）、页标题 1.9.1（见 [[sklearn-cross-validation]] 的版本判定）。
- **`model_selection` 的默认值结论不依赖该判定**：第 182–200 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 182、186、187、194、196、197 条**涉及版本敏感的默认值与参数存在性**（`cv=None` 的 3→5 折、`return_train_score` 的 0.19/0.21 变更、`TimeSeriesSplit` 三参数 0.24 新增、`GroupKFold.shuffle` 1.6 新增、`StratifiedGroupKFold` 的签名不统一），引用时必须写版本。
- `model_selection` 一节的**四个折叠块全丢（442 词）**，其中一个就是 `StratifiedGroupKFold` 的 `Implementation notes`（154 词）。因此「洗牌为何不完整」「为何可能产生不平衡划分」**不能引剪藏行号**，只能引官方 1.9.1 页；第 198 条是用 1.6.1 实测补上这段机制（类比例极差 0.064 vs 0.375）。另：L226 的两个「safer to use」链接文字丢成「a.」，该句引时应引官方页。
- `model_selection` 的**主 References 块（6 条，含 Kohavi 与 Rao et al.）与置换检验的 Ojala & Garriga 2010 全丢**（连标题都没有）。因此「5 或 10 折优于 LOO」只能引剪藏 L302–L304 的断言，**引用实证文献必须另找**。
- `model_selection` 的**搜索器面（3.2 节）对应 1.7.0 – 1.9.1**：50 条 ≥14 词句子探针在 1.6/1.7/1.8/1.9 四版的命中数是 **45/45/45/45**（非判别性，未命中的 5 条是 Markdown 伪影）；定版靠 1.6→1.7 的一处措辞——剪藏 L233 是 1.7 的 `…, see Which scoring function should I use? for some guidance.`（1.6 把该句拆成两句、无此链接）；1.7→1.8 正文逐句 diff 为空，1.8→1.9 的唯一差异（`LassoCV(*[, eps, n_alphas, …])` → `alphas`）落在**已丢失的 3.2.5 节**；抓取日 `/stable/` 与 `/1.9/` **字节相同**（137,995 B）、页标题 1.9.1（见 [[sklearn-hyperparameter-tuning]] 的版本判定）。
- **搜索器面的默认值结论不依赖该判定**：第 201–216 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 201、202、204、205、206、207、208 条涉及版本敏感的默认值与行为（`refit` 的默认语义、`RandomizedSearchCV` 的「不放回」规则、`cv` 默认折数、halving 的实验性 API），引用时必须写版本。
- **3.2 节的损失集中在「为什么」上**：7 个 `<details>` 折叠块里 3 个整块丢失（*Advanced examples*、*Choosing min_resources and the number of candidates*、*cv_results_ 的输出示例*），4 个只剩代码（资源递推式、`resource='n_estimators'`、`min_resources` 两种写法、`aggressive_elimination` 两组 doctest）；**整个 §3.2.5（正则化路径 / AIC-BIC / OOB）丢失**。因此「`min_resources` 怎么选」「最后一轮为什么不能多于 `factor` 个候选」「不能拿网格里的参数当 `resource`」**不能引剪藏行号**，只能引官方 1.9.1 页；第 210、213 条的行为是用 1.6.1 实测补上的。
- **3.2 节的 halving References（Jamieson & Talwalkar 2016、Li et al. Hyperband 2018）全丢**，L99 的引用标记 `[1] [2]` 也被吞，所以「逐次减半的最优性」在本页**没有任何文献支撑**，只有机制与实测读数。
- **`*CV` 估计器（`RidgeCV`/`LassoCV`/`LogisticRegressionCV` 等 10 个）与 `LassoLarsIC` 出自已丢失的 3.2.5 节**：它们的本体在 1.1 节（`linear_model`），本页只在 [[regularization-and-sparsity]] 与 [[linear-models-as-baselines]] 里转述过用法，**不能声称「用户指南 3.2 节这样说」**。
- **3.2 节的示例有一处照抄跑不起来**：L82 的 `from sklearn.utils.fixes import loguniform` 在 1.6.1 上 `ImportError`（第 216 条）。引用该示例必须换成 `scipy.stats.loguniform`。
- `metrics`（评分入口面）对应 **1.8.0 – 1.9.1**：54 条 ≥14 词句子探针在 1.6/1.7/1.8/1.9 四版的命中数是 **37/37/37/37**（非判别性，未命中的 17 条全是 Markdown 伪影），定版靠一处：剪藏的脚注 1 是「… in case of classification **with one-hot encoded targets**.」，而 1.6.1/1.7.2 页只有「… in case of classification.」（**外部核查**，逐版 grep）。官方覆盖区的逐句 diff：1.6→1.7 为空、1.8→1.9 为空、**1.7→1.8 只有这一条脚注**。拓取日 `/stable/` 与 `/1.9/` **字节相同**（438,145 B，sha256 `7e740e14…`）、页标题 1.9.1（见 [[sklearn-metrics]] 的版本判定）。
- **`metrics` 的默认值结论不依赖该判定**：第 217–230 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 217–218、222–223、227 条涉及版本敏感项（`make_scorer.response_method` 1.8 变、`d2_*` 的入表与注册、`top_k_accuracy_score` 的 `k` 校验），引用时必须写版本。
- **3.4 节是本库损失最严重的剪藏**：官方 54 个编号小节里只有 4 个进来（约 92% 的正文、即 17,063/18,573 词不在），且保留的四节里仍丢了「评分函数」整列、43 行字符串评分表、`Usage examples` doctest、4 条参考文献（Gneiting 2007/2009/2014、Fissler 2022）。因此本页的评分入口矩阵只到「入口」这一层，**不提供**任何具体指标的定义、参数与公式。
- `metrics` 的**唯一判别性版本证据是一条脚注**，比前十六篇都弱；引用「1.9.1 这样说」时必须记得：它同样可能是 1.8.0。
- `preprocessing` 部分对应 **1.9.1**：七条判别性证据——小节编号 **8.3**（1.6/1.7 是 6.3、1.8 是 7.3）、L610 的 `This diagram shows`（1.7 是 `The following`）、L608/L614/L616 三句 TargetEncoder 新措辞（1.7 全不同）、pandas repr `Categories (5, str)`（1.8 是 `object`）；撕取日 `/stable/` 与 `/1.9/` **字节相同**（205,574 B，sha256 `efe09a2f…`）、页标题 1.9.1（见 [[sklearn-preprocessing]] 的版本判定）。
- **`preprocessing` 的默认值结论**：第 231–246 条来自 1.9.1 API 页签名，并在 **1.9.1 与 1.6.1 上双跑**；但第 232、240、244、245 条**涉及版本敏感项**（`MaxAbsScaler.clip` 1.8 新增、`encoded_missing_value` 的默认值、`quantile_method` 的 1.7/1.9 两次变更、`handle_missing` 1.8 新增），引用时必须写版本。
- **`preprocessing` 一节丢了 11 个折叠块（≈933 词散文）+ 3 处 `References`（5 条文献）**，丢的正好是核中心化推导、Box-Cox/Yeo-Johnson 公式、`TargetEncoder` 的收缩与经验贝叶斯、两处稀疏说明、两段 FAQ。因此「为什么稀疏不能中心化」「λ 怎么估」「目标编码怎么收缩」**不能引剪藏行号**，只能引官方 1.9.1 页。
- **`preprocessing` 的文档自身有三处与实现/自身不一致**（1.9.1 上可复现）：`KBinsDiscretizer` 印出的分箱区间是 1.6 口径但 doctest 的分箱编号恰好不变；`max_categories` 并列取舍的说明把 `3` 写成 `2`、把「字典序更小」说成「更大」；`QuantileTransformer` 的 iris 分位数 doctest 在 1.6.1 与 1.9.1 上都不复现（≤0.02）。因此本页对这三处均标「实测」而非「文档」。

## 使用边界

- 本页只覆盖线性模型、判别分析、朴素贝叶斯、SVM、最近邻、tree、ensemble、feature_selection、neural_network、manifold、cluster、decomposition、covariance 与 preprocessing 十四个模块，外加跨模块的**异常检测面**（2.7 节，五个入口分布在 svm/linear_model/covariance/ensemble/neighbors）、`sklearn.model_selection` 的**切分器面**（3.1 节：11 个 splitter + `train_test_split` + 3 个 `cross_*` 函数）与**搜索器面**（3.2 节：4 个搜索器 + 2 个候选生成器）、`sklearn.metrics` 的**评分入口面**（3.4 节：4 个入口 + `make_scorer`）与 `sklearn.preprocessing` 的**变换器面**（8.3 节：16 个变换器 + 9 个函数）。缺失值填补（8.3.6 只有一句指针，正文在 `sklearn.impute`）与文本向量化（8.2）**仍未入库**，所以本页**不提供** `SimpleImputer`/`CountVectorizer`/`TfidfVectorizer` 的用法；`sklearn.compose`（`ColumnTransformer`）只有本页与 [[preprocessing-pipeline-design]] 的骨架级用法。
- **8.3 节（`sklearn.preprocessing`）的选型与顺序不在本页**：见 [[preprocessing-pipeline-design]]；机制与几何见 [[feature-preprocessing]]。本页只负责「有哪些变换器、默认值是什么、版本边界在哪」。
- **8.3 节的机制层引用必须绕开剪藏**：11 个折叠块（核中心化、两套幂变换公式、目标编码三套收缩式、两处稀疏、两段 FAQ）与 5 条文献全丢，本页与来源页对此均只能引官方 1.9.1 页或本机实测。
- `model_selection` 的**选型与调参不在本页**：见 [[cross-validation-design-and-tuning]]；机制与失效模式见 [[cross-validation]]。本页只负责「有哪些 splitter、默认值是什么、版本边界在哪」。
- **3.4 节（`sklearn.metrics`）只入了方法论四小节**：本页给的字符串清单是 **1.6.1 运行时实测的 56 个**（而非 1.9.1 文档表的 43 行），两者双向不一致（第 222–223 条）；多指标的报告写法见 [[cross-validation-design-and-tuning]]；「搜索器如何复用同一套洗牌」见 [[sklearn-cross-validation]] L614。
- **3.2 节的搜索器选型与预算不在本页**：见 [[hyperparameter-search-and-tuning]]；机制见 [[hyperparameter-search]]。本页只负责「有哪些搜索器、默认值是什么、版本边界在哪」。
- 1.9 节的用户指南**不覆盖** `alpha`、`fit_prior`、`class_prior`、`var_smoothing`、`binarize`、`norm`、`min_categories` 七个参数；本页第 17–22 条全部来自 API 页或源码（外部核查），不是剪藏论述。
- 1.2 节的用户指南**不覆盖** `priors`、`reg_param`、`store_covariance`、`tol` 四个参数；本页第 13、15、16 条与估计器矩阵中这些项全部来自官方 API 页（外部核查），不是剪藏的论述（第 14 条的两个估计器来自 L101–L103）。
- 官方页有 29.3% 的正文在折叠块里，剪藏全缺（详见 [[sklearn-linear-models]] 的「转换损失与外部核查」）。本页因此**不声称穷尽了文档的所有约束**：`Solvers’ details`、`Practical considerations` 里的细则（如 `l1_min_c`、GLM 的 exposure 用法）本页没有。
- 稳健回归的取舍（RANSAC / Theil-Sen / Huber）不宜按一句话规则选，见 [[linear-models-as-baselines]] 里由官方示例图读数得到的反例。
- 1.4 节的用户指南**不覆盖** `sklearn.calibration`（`CalibratedClassifierCV` 的 `ensemble` 语义）、`sklearn.kernel_approximation`（Nyström / RBFSampler）、`sklearn.multiclass` 与 `l1_min_c` 的细节，因此本页**不提供**核近似或校准曲线的用法。
- `svm` 对 `OneClassSVM` 只有两句话（[[sklearn-svm]], L186–L188）；它的实际用法在已入库的 2.7 节（见 [[sklearn-outlier-detection]] 与 [[outlier-detection-selection-and-tuning]]），但 **1.4 节的 $\nu$ 语义（训练误差/支持向量比例）与 2.7 节的 $\nu$ 语义（新样本越界概率）不一致**，本页保留两种说法，实测支持前者。
- svm 的 `gamma`、`dual`、`probability` 三个关键默认值都不在用户指南里；本页把它们标为外部核查而不是文档论述，引用时应保留该区分。
- neighbors 部分的**选型内容约 20% 在剪藏丢失的折叠块里**（共缺 959 词正文 + 34 词 References）。因此「何时用 KD / ball / brute」「`leaf_size` 怎么设」在本页均为外部核查。
- neighbors 的**用户指南完全不提特征缩放**，而官方示例自己套了 `StandardScaler`（外部核查：`plot_classification.py`、`plot_nca_classification.py`）。本页不因此断言「文档认为不需要缩放」——只能说文档缺了这句话。具体做法见 [[nearest-neighbors-selection-and-tuning]]。
- neighbors 的**度量与距离函数细节**在未入库的 `sklearn.metrics.pairwise`；下游消费者（`DBSCAN`、`SpectralClustering`、`TSNE`、`Isomap`）已在 2.2/2.3 节入库，见 [[sklearn-manifold-learning]] 与 [[sklearn-clustering]]。本页只负责它们**吃哪种图**（`connectivity` vs `distance`，L198）。
- `NearestCentroid.priors` 的 docstring 矛盾（`default="uniform"` vs 「默认从训练集推断比例」）**未做运行时验证**：本页只断言两句话不能同时成立，不断言实现采用了哪一个。
- tree 部分的**特征重要性、`tree_` 遍历 API、`cost_complexity_pruning_path` 与 `export_graphviz` 参数细节全部不在剪藏内**，所以本页**不提供**重要性解释或剪枝路径的用法；第 45–55 条只覆盖树结构与缺失值。
- tree 一节丢掉了**全部 3 个内容折叠块**（530 词散文：导出树、ID3/C4.5/C5.0 谱系、熵 = log loss 的推导）。因此本页对「Gini vs entropy」只说「叶子输出相同、只影响切分排序」，**不引用**那篇丢失的推导。
- 官方 `iris.svg` 是**陈旧静态资产**（2014 年 graphviz 2.36.0 生成，1.6–1.9 字节相同），与同页 doctest 的根切分不同。引用该图的阈值时必须注明，或改用自己拟合的结果。
- tree 的**运行时复核只做到 1.6.1**（PyPI 当前最高可安装版本），而文档是 1.9.1。机制层结论（平票、float32、`min_samples_split` 忽略权重、缺失值三规则）可跨版本；`absolute_error`+NaN、`monotonic_cst`+NaN 只在 1.9 起支持，本页未实测。
- ensemble 一节丢了 **8 个内容折叠块 + 1 个 `aside` 主题框 + 整个 1.11.4.1 节**，散文约 1,750 词 ≈ 本节的 21%。因此 «小样本用 `GradientBoosting*`»、«GBDT 拟合负梯度»、«多类 GBRT 每轮 K 棵树»、«硬投票平票规则» 四条**都不在剪藏里**，本页标为外部核查。
- ensemble 的**选型与调参细节**不在本页：见 [[ensembles-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么」。
- `ensemble` 一节**不覆盖** `sklearn.inspection.permutation_importance`、`sklearn.calibration`、`sklearn.model_selection` 的 CV 切分，也不提供 LightGBM/XGBoost 的迁移对照。
- ensemble 的**特征重要性结论只到「MDI 有两个缺陷」**（L518）；具体用 `permutation_importance` 的写法在未入库的 5.2 节。
- ensemble 的**运行复核只做到 1.6.1**；`criterion` 弃用与 `categorical_features` 默认值变更属于 1.6/1.9 的版本边界，本次未在 1.9.1 上复跑。
- feature_selection 一节丢了 **1 个 `aside` + 2 个 `<details>` + 整段 References（386 词 ≈ 19%）**，丢的正好是「稀疏输入不被稠密化」、「Lasso 精确恢复支撑集的条件与 `alpha` 的三种取法」、「前向/后向的成本算术与 SFS 不要求重要性属性」。因此本页第 77–86 条中涉及「哪个方法能吃稀疏矩阵」「为什么 SFS 比 RFE 贵」的部分均为外部核查，**不能引剪藏**。
- `feature_selection` 的**选型内容不在本页**：见 [[feature-selection-workflow]]。本页只负责「有哪些变换器、默认值是什么」。
- `feature_selection` 覆盖表里的 `sklearn.decomposition`（PCA/NMF 等真正的降维）已在 2.5 节入库（见 [[matrix-factorization]] 与 [[decomposition-selection-and-tuning]]），因此「删列 vs 投影」的对照见这两页；`sklearn.inspection.permutation_importance` 仍未入库，`sklearn.model_selection` 的 CV 切分已入库（见 [[sklearn-cross-validation]] 与 [[cross-validation-design-and-tuning]]），本页不提供 MDI 的替代写法。
- `feature_selection` 的**运行复核只做到 1.6.1**；第 78–83、86 条的行为在 1.9.1 上未复跑（签名已核，行为未核）。
- `neural_network` 一节丢了 **3 个内容 `<details>` 折叠块（622 词 ≈ 36%）**：优缺点（非凸/局部极小/随机初始化）、数学表述（单隐藏层公式、tanh/logistic/softmax、交叉熵/MSE、反向传播）、5 条 References。因此非凸性、损失公式与文献**不能引剪藏**，只能引官方 1.9.1 页（见 [[sklearn-neural-networks]] 转换损失）。
- `neural_network` 的**默认值几乎全不在用户指南里**：正文只给 `alpha` 的网格、三个求解器的经验、缩放建议与 `warm_start` 写法；`hidden_layer_sizes=(100,)`、`solver='adam'`、`max_iter=200`、`early_stopping=False`、`batch_size='auto'` 等均为外部核查。
- `neural_network` 的**选型与调参不在本页**：见 [[mlp-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么、版本边界在哪」。
- `neural_network` 的**运行复核只做到 1.6.1**；`loss='poisson'`、`max_fun` 与早停细节未在 1.9.1 上复跑（签名已核，行为未核）。
- `manifold` 一节丢了 **10 个内容折叠块（1,669 词，相当于保留正文 2,214 词的 75%）**，另有 1 张图（`plot_mds_001.png`）未入库。因此本页的「三阶段复杂度」「Stress-1」「t-SNE 五个参数」「Barnes-Hut 四限」均为外部核查，不能引剪藏。
- `manifold` 的 `transform` 语义不在本节：`SpectralEmbedding`/`TSNE`/`MDS`/`ClassicalMDS` **没有 `transform`**（实测 1.6.1，1.9.1 签名同）；`Isomap`/`LocallyLinearEmbedding` 的 `transform` 对**新样本**做了什么外推，官方 2.2 节不提，本页不予断言。
- `manifold` 的**评估指标（`trustworthiness`）本节完全不提**：第 115 条与「全局 Spearman」都是外部核查或本库补充，不是文档论述。
- `manifold` 的**选型与调参不在本页**：见 [[manifold-learning-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么、版本边界在哪」。
- `manifold` 的**运行复核只做到 1.6.1**，且 1.6.1 **没有 `ClassicalMDS`**（`import` 就失败）；第 106–108 条的 `MDS` 默认值差异在 1.9.1 上未复跑（签名已核，行为未核）。
- `cluster` 的**运行复核只做到 1.6.1**；第 123 条的 `copy='warn'`（1.9.1）与第 126 条的 `discretize` 实现未在 1.9.1 上核过（签名已核、行为未核）。
- `cluster` 一节的**缩放、Pipeline 用法、选型与调参全部不在剪藏内**：见 [[clustering-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么」。
- `cluster` 的 **9 个评估指标在本节里与选型完全脱节**（§2.3.11 前面一次也没提「怎么知道这次聚类好不好」），也不说内部指标随 $k$ 的单调性。本页因此**不提供**「哪个指标能选 $k$」的结论（见 [[clustering]] 第二节）。
- `cluster` 覆盖表里的 `sklearn.mixture`（GMM，仅以一行表项出现）、`sklearn.metrics.pairwise`（`affinity='precomputed'` 的矩阵怎么造）与 2.4 Biclustering **均未入库**，所以本页不提供 GMM 的参数矩阵，也不提供软/硬分配对照。
- `manifold` 覆盖表里的 `sklearn.decomposition`（PCA/NMF 等）已在 2.5 节入库，因此「classical MDS ≈ PCA」之外，PCA 自身的参数矩阵见上面的 `decomposition` 节与 [[matrix-factorization]]。
- `cluster` 的**30 个内容折叠块（其中 16 个是 References）与总览表首列全丢**，因此上面第 116–135 条中凡涉及「机会校正公式」「消息传递细节」「实现/内存」「复杂度」的部分均为外部核查，**不能引剪藏**。
- `decomposition` 的**参数名有两处文档与 API 不一致**：正文的 `split_code` 在 API 是 `split_sign`（L160），正文的 `transform_method` 在 API 是 `transform_algorithm`（L152）。引用这两个参数时必须以签名为准。
- `decomposition` 的**运行复核只做到 1.6.1**：`NMF.n_components='auto'`（1.6 起）、`FastICA.whiten`（1.3 起字符串）、`MiniBatchDictionaryLearning.batch_size`（1.3 起 256）都未在 1.9.1 上复跑（签名已核，行为未核）。
- `decomposition` 的**选型与调参不在本页**：见 [[decomposition-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么」。
- `decomposition` 的**4 个折叠块（KernelPCA 求解器、LSA 数学、NMF 求解器、LDA 生成过程）全丢**，所以本页不提供这四块的公式；需要时引官方 1.9.1 页。
- `covariance` 是**唯一没有任何代码/doctest 的 sklearn 来源**（0 个代码块、0 条 doctest），所以本页的用法示例全部来自 API 页与实测，不是文档示例。
- `covariance` 一节丢了 **4 条参考文献与 4 个页内引用标记**，所以「LW 的 α 公式」「OAS 的闭式解」「MCD 的 breakdown point」**不能引剪藏行号**；文献条目本身已按官方 1.9.1 页抄录在 [[sklearn-covariance]] 的「转换损失」表中。
- `covariance` 的 **2.7 节（novelty/outlier detection）已入库**：`EllipticEnvelope` 就是「MCD + 阈值」，所以第 161–163 条的 `support_`/`correction`/$n>p$ 三条结论对它全部适用；它的 `contamination=0.1` 默认与 IF/LOF 的 `'auto'` 不同（见上表 165–167）。
- `covariance` 的**运行复核只做到 1.6.1**；`GraphicalLasso` 的 `eps`/`covariance=`、`MinCovDet` 无 `correction` 等签名差异已核，但 1.9.1 的行为未复跑。
- `异常检测面` 的**运行复核只做到 1.6.1**：`contamination` 与 `gamma` 的默认值在 1.6.1 上已是 `'auto'`/`'scale'`（0.22 起），但 1.9.1 的行为未复跑（签名已核，行为未核）。
- `异常检测面` 的**评估完全不在剪藏内**：2.7 节有 6 张图、10 条示例链接、**0 个指标**。本页因此不提供任何「异常检测指标」，评估协议见 [[outlier-detection-selection-and-tuning]] 第六节。
- `异常检测面` 的**选型与调参不在本页**：见 [[outlier-detection-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么、版本边界在哪」。
- `异常检测面` 的**复杂度、分数公式与 IF 的 $c(n)$ 校正都不在剪藏内**（该页无 `<details>` 折叠块，本节本来就没写），所以不能说「文档给出了复杂度」；第 173、175 条只到「行为如上」。
- **1.2 节文档与 1.6.1 实现关于 LDA `shrinkage=1` 终点的说法冲突**（文档：方差对角阵；实测：标量乘单位阵 133.3276·I）。本页按实现记录，冲突保留在 [[covariance-estimation]] §2.3 与 [[regularization-and-sparsity]] 的「第二个正则化面」。
- `covariance` 的**选型与调参不在本页**：见 [[covariance-estimation-selection-and-tuning]]。本页只负责「有哪些估计器、默认值是什么、版本边界在哪」。
- `异常检测面` 部分对应 **1.9.1**：84 条句子探针在 1.6/1.7/1.8/1.9 的命中数是 **75/77/77/78**，差的那一条正是 1.9 新增的 `contamination`/`offset_` 句（剪藏 L172）；`/stable/` 与 `/1.9/modules/outlier_detection.html` **字节相同**（79,533 B）。旁证：`IsolationForest`/`LocalOutlierFactor` 的 `contamination` 默认 0.22 起才变 `'auto'`，`OneClassSVM.gamma` 0.22 起才变 `'scale'`。
- **`异常检测面` 的默认值结论不依赖该判定**：第 165–181 条来自 1.9.1 API 页签名与 1.6.1 实测；但第 165、166、172 条涉及版本敏感的默认值（`contamination` 与 `gamma` 的 0.22 变更），引用时必须写版本。
- **2.7 节的剪藏是本库损失最小的一份**：只丢了 Liu/Ting/Zhou 2008 这一条文献与 2 处引用标记（L133 的 `[1]`、L170 的页内链接文字），因此「IF 的分数式 $2^{-E(h)/c(n)}$」「$c(n)$ 的调和数定义」**仍不能引剪藏**，只能引官方 1.9.1 页或 Liu et al. 2008。
- **3.4 节已入库，但只入了方法论四小节**：本页因此只给 `scoring`/`metrics` 的**入口与默认值**，不给 F1 的 `average` 语义、ROC 的多类定义、MAPE 在 0 附近的行为、Brier 与 log loss 的分解——它们全在丢失的 §3.4.4/§3.4.6 里，需另引 API 页或新增来源。
- 「某个指标怎么算」不在本页：机制见 [[evaluation-metrics]]，落地流程见 [[metric-selection-and-thresholding]]。本库目前**没有**指标定义级来源，因此本页不说「官方推荐 macro 还是 micro」。
- 阈值调优（官方 §3.3 *Tuning the decision threshold for class prediction*）与概率校准（`sklearn.calibration`）都未入库，本页不提供 `TunedThresholdClassifierCV` 的用法；只能给「`predict` = `proba >= 0.5`」这条实测事实。
- §3.4.3.2（callable scorers / `make_scorer` / 自定义 scorer 对象，730 词）全在剪藏外，第 217–218、230 条的行为均来自实测，不是文档论述。
- `sklearn.metrics.pairwise`（度量与核）未入库：剪藏 L71 的 See also 只提到 pairwise 存在，聚类指标本体在已入库的 2.3 节（见 [[clustering]]）。
- `metrics` 的**运行复核只做到 1.6.1**：`d2_log_loss_score`/`d2_brier_score` 的注册状态、`positive_likelihood_ratio` 一族、`make_scorer` 的 1.8 默认值均未在 1.9.1 上复跑。
- **`preprocessing` 是第一个在 1.9.1 上实测的模块**（此前各篇均止步于 1.6.1）：本页第 231–246 条的机制与数值均在 1.9.1（Python 3.14.7 / NumPy 2.5.4）与 1.6.1 上双跑，差异已逐条标出。
- 8.3 节的 **10 条官方示例脚本全部未跑**（`plot_scaling_importance`、`plot_all_scaling`、`plot_target_encoder_cross_val` 等），所以本页与两个新页**不提供**「七种缩放器在离群点数据上的形状对比」或「官方目标编码交叉拟合的分数差」这类读数。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]、[[sklearn-nearest-neighbors]]、[[sklearn-decision-trees]]、[[sklearn-ensembles]]、[[sklearn-feature-selection]]、[[sklearn-neural-networks]]、[[sklearn-manifold-learning]]、[[sklearn-clustering]]、[[sklearn-decomposition]]、[[sklearn-covariance]]、[[sklearn-outlier-detection]]、[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]、[[sklearn-metrics]]、[[sklearn-preprocessing]]
- 实践：[[linear-models-as-baselines]]、[[svm-selection-and-tuning]]、[[nearest-neighbors-selection-and-tuning]]、[[decision-trees-selection-and-tuning]]、[[ensembles-selection-and-tuning]]、[[feature-selection-workflow]]、[[mlp-selection-and-tuning]]、[[manifold-learning-selection-and-tuning]]、[[clustering-selection-and-tuning]]、[[decomposition-selection-and-tuning]]、[[covariance-estimation-selection-and-tuning]]、[[outlier-detection-selection-and-tuning]]、[[cross-validation-design-and-tuning]]、[[hyperparameter-search-and-tuning]]、[[metric-selection-and-thresholding]]、[[preprocessing-pipeline-design]]
- 概念：[[regularization-and-sparsity]]、[[gaussian-discriminant-analysis]]、[[naive-bayes]]、[[support-vector-machines]]、[[nearest-neighbors]]、[[decision-trees]]、[[ensembles]]、[[feature-selection]]、[[multi-layer-perceptron]]、[[manifold-learning]]、[[clustering]]、[[matrix-factorization]]、[[covariance-estimation]]、[[anomaly-detection]]、[[cross-validation]]、[[hyperparameter-search]]、[[leaderboard-vs-cv]]、[[evaluation-metrics]]、[[feature-preprocessing]]
- 对照来源：[[kaggle-getting-started]]（2016 年的 sklearn 用法，多数 API 已移除）
