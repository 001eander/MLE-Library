---
title: scikit-learn（工具页）：模块矩阵、官方默认值与版本边界
type: tool
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - api
  - reference
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
  - "[[sklearn-stacking-example]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.8.x – 1.9.x（linear_model，见 [[sklearn-linear-models]]）
  - scikit-learn 1.9.x（discriminant_analysis 与 naive_bayes，见对应来源页）
  - scikit-learn 1.7.x – 1.9.1（svm，见 [[sklearn-svm]]）
  - scikit-learn 1.7.x – 1.9.x（neighbors，见 [[sklearn-nearest-neighbors]]）
  - scikit-learn 1.9.x（tree，见 [[sklearn-decision-trees]]）
  - scikit-learn 1.9.1（ensemble、feature_selection、cluster、decomposition、异常检测面、preprocessing，见对应来源页）
  - scikit-learn 1.8.0 – 1.9.1（neural_network、manifold、metrics，见对应来源页）
  - scikit-learn 1.7.0 – 1.9.1（covariance、model_selection 切分与搜索，见对应来源页）
  - 运行时行为须在 `uv` 创建的目标版本环境中另行验证；本页不包含历史本机实验
---

# scikit-learn（工具页）

本页汇总已入库用户指南的**接口、官方默认值与版本边界**。事实优先引用对应来源页；API 签名与版本史标为「外部核查」。不包含历史本机运行时实验、性能读数或旧环境复现结果。

## 收录范围

`raw/articles/sklearn/` 下 18 篇剪藏均已建立来源页。3.4 Metrics 只保留方法论 4 小节，其他章节的转换损失见各来源页。

| 文档 | 模块/主题 | 概念与实践 |
| --- | --- | --- |
| 1.1 Linear Models | `linear_model` | [[regularization-and-sparsity]]、[[linear-models-as-baselines]] |
| 1.2 LDA/QDA | `discriminant_analysis` | [[gaussian-discriminant-analysis]] |
| 1.4 SVM | `svm` | [[support-vector-machines]]、[[svm-selection-and-tuning]] |
| 1.6 Nearest Neighbors | `neighbors` | [[nearest-neighbors]]、[[nearest-neighbors-selection-and-tuning]] |
| 1.9 Naive Bayes | `naive_bayes` | [[naive-bayes]] |
| 1.10 Decision Trees | `tree` | [[decision-trees]]、[[decision-trees-selection-and-tuning]] |
| 1.11 Ensembles | `ensemble` | [[ensembles]]、[[ensembles-selection-and-tuning]] |
| 1.13 Feature selection | `feature_selection` | [[feature-selection]]、[[feature-selection-workflow]] |
| 1.17 Neural networks | `neural_network` | [[multi-layer-perceptron]]、[[mlp-selection-and-tuning]] |
| 2.2 Manifold learning | `manifold` | [[manifold-learning]]、[[manifold-learning-selection-and-tuning]] |
| 2.3 Clustering | `cluster` | [[clustering]]、[[clustering-selection-and-tuning]] |
| 2.5 Decomposition | `decomposition` | [[matrix-factorization]]、[[decomposition-selection-and-tuning]] |
| 2.6 Covariance | `covariance` | [[covariance-estimation]]、[[covariance-estimation-selection-and-tuning]] |
| 2.7 Outlier detection | 异常检测面 | [[anomaly-detection]]、[[outlier-detection-selection-and-tuning]] |
| 3.1 Cross-validation | `model_selection` 切分 | [[cross-validation]]、[[cross-validation-design-and-tuning]] |
| 3.2 Hyperparameters | `model_selection` 搜索 | [[hyperparameter-search]]、[[hyperparameter-search-and-tuning]] |
| 3.4 Metrics | `metrics` 评分入口 | [[evaluation-metrics]]、[[metric-selection-and-thresholding]] |
| 8.3 Preprocessing | `preprocessing` | [[feature-preprocessing]]、[[preprocessing-pipeline-design]] |

## 1. 模型与估计器矩阵

### `linear_model`

| 对象 | 用途 | 关键参数/默认 |
| --- | --- | --- |
| `LinearRegression` | OLS | `fit_intercept=True` |
| `Ridge` / `RidgeClassifier` | $ℓ_2$ | `alpha=1.0`、`solver='auto'` |
| `RidgeCV` | 内置 LOO/K 折选 α | `alphas=(0.1,1.0,10.0)`、`cv=None` |
| `Lasso` / `ElasticNet` | L1 / L1+L2 | `alpha=1.0`、`l1_ratio=0.5` |
| `LassoCV` / `ElasticNetCV` | CV 选正则 | `cv=None`、`n_alphas=100` |
| `Lars` / `LassoLars` | 分段线性路径 | `n_nonzero_coefs=500`、`eps=2.22e-16` |
| `LassoLarsIC` | AIC/BIC 选 α | `criterion='aic'` |
| `OrthogonalMatchingPursuit` | $ℓ_0$ 约束 | `n_nonzero_coefs=None`、`tol=None` |
| `BayesianRidge` | 球形高斯先验 | `alpha_1=alpha_2=lambda_1=lambda_2=1e-6` |
| `ARDRegression` | 逐系数精度 | 同类初始化参数 |
| `LogisticRegression` | 交叉熵 | `C=1.0`、`penalty='l2'`、`solver='lbfgs'`、`max_iter=100` |
| `TweedieRegressor` | GLM | `power=0.0`、`alpha=1.0`、`link='auto'` |
| `SGDClassifier` / `SGDRegressor` | 流式线性 | `loss='hinge'` / `'squared_error'`、`penalty='l2'`、`alpha=0.0001` |
| `RANSACRegressor` | 共识内点 | `min_samples=None`、`residual_threshold=None`、`max_trials=100` |
| `TheilSenRegressor` | 空间中位数 | `max_subpopulation=10000` |
| `HuberRegressor` | 稳健回归 | `epsilon=1.35`、`alpha=0.0001` |
| `QuantileRegressor` | 分位数 + L1 | `quantile=0.5`、`alpha=1.0`、`solver='highs'` |

**契约**：`C` 与 `alpha` 方向相反；$C=1/\alpha$ 只在特定模型与归一化下成立。L1 产生特征稀疏，L2 通常只收缩。内置 CV/IC/OOB 是不同代理指标，不等于自动无偏选择。

### `discriminant_analysis` 与 `naive_bayes`

| 对象 | 关键默认/约束 |
| --- | --- |
| `LinearDiscriminantAnalysis` | `solver='svd'`、`shrinkage=None`、`priors=None`、`n_components=None`；`svd` 不支持 shrinkage，`lsqr`/`eigen` 才支持 |
| `QuadraticDiscriminantAnalysis` | `reg_param=0.0`、`store_covariance=False`、`tol=1e-4` |
| `GaussianNB` | `priors=None`、`var_smoothing=1e-9` |
| `MultinomialNB` / `ComplementNB` / `BernoulliNB` / `CategoricalNB` | `alpha=1.0`、`fit_prior=True`、`class_prior=None`、`force_alpha=True`；Complement `norm=False`；Bernoulli `binarize=0.0` |
| `NearestCentroid` | `metric='euclidean'`、`shrink_threshold=None`、`priors='uniform'`（docstring 默认说明存在矛盾） |

`partial_fit` 的覆盖以 1.9.1 源码/API 为准：五个 NB 估计器都有，尽管剪藏只列三个（[[sklearn-naive-bayes]] 分歧）。

### `svm`

| 对象 | 关键默认/约束 |
| --- | --- |
| `SVC` / `NuSVC` | `C=1.0` / `nu=0.5`、`kernel='rbf'`、`gamma='scale'`、`probability=False`、`break_ties=False`、`cache_size=200` |
| `LinearSVC` | `penalty='l2'`、`loss='squared_hinge'`、`dual='auto'`、`C=1.0`、`intercept_scaling=1` |
| `SVR` / `NuSVR` | `C=1.0` / `nu=0.5`、`epsilon=0.1` |
| `LinearSVR` | `epsilon=0.0`、`loss='epsilon_insensitive'`、`dual='auto'` |
| `OneClassSVM` | `kernel='rbf'`、`gamma='scale'`、`nu=0.5`；没有 `contamination` |

`SVC(kernel='linear')` 与 `LinearSVC` 的损失和截距正则不同，换实现必须重扫 `C`。`probability=True` 在 1.9 弃用，1.11 将移除；替代方案见 [[sklearn-svm]] 分歧。

### `tree`

| 对象 | 关键默认/约束 |
| --- | --- |
| `DecisionTreeClassifier` | `criterion='gini'`、`splitter='best'`、`max_depth=None`、`min_samples_split=2`、`min_samples_leaf=1`、`max_features=None`、`ccp_alpha=0.0` |
| `DecisionTreeRegressor` | `criterion='squared_error'`，其余结构默认相同 |
| `ExtraTreeClassifier` / `ExtraTreeRegressor` | `splitter='random'`、分类 `max_features='sqrt'`、回归 `max_features=1.0` |

`min_samples_split` 直接数样本，不看权重；`predict_proba` 在加权时是加权比例；`ccp_alpha=0` 默认不剪枝。float32 内部转换与 NaN 规则须按版本检查。

### `ensemble`

| 对象 | 关键默认/约束 |
| --- | --- |
| `HistGradientBoosting*` | `learning_rate=0.1`、`max_iter=100`、`max_leaf_nodes=31`、`max_bins=255`、`early_stopping='auto'`、`categorical_features='from_dtype'` |
| `GradientBoosting*` | `n_estimators=100`、`learning_rate=0.1`、`max_depth=3`、`subsample=1.0`、`n_iter_no_change=None`；`criterion` 在 1.9 弃用 |
| `RandomForest*` | `n_estimators=100`、`max_depth=None`、`bootstrap=True`；分类 `max_features='sqrt'`、回归 `1.0` |
| `ExtraTrees*` | 同 RF 结构，`bootstrap=False` |
| `Bagging*` | `n_estimators=10`、`max_samples=None`（→1.0）、`max_features=1.0`、`bootstrap=True` |
| `VotingClassifier` | `voting='hard'`、`weights=None`、`flatten_transform=True` |
| `VotingRegressor` | `weights=None` |
| `StackingClassifier` | `cv=None`（→5 折 `StratifiedKFold`）、`stack_method='auto'`（`predict_proba` → `decision_function` → `predict`）、`passthrough=False`、`final_estimator=None`（→`LogisticRegression`） |
| `StackingRegressor` | `cv=None`（→5 折 `KFold`）、`passthrough=False`、`stack_method='auto'`（回归→`predict`）、`final_estimator=None`（→`RidgeCV()`） |
| `AdaBoostClassifier` | `n_estimators=50`、`learning_rate=1.0`、默认 stump |
| `AdaBoostRegressor` | 默认 `max_depth=3` 回归树、`loss='linear'` |

RF 平均概率而非硬投票；Stacking 必须用折外预测；`max_bins` 上限 255 且数值箱 + 缺失箱是另一计数口径。

Stacking 的两个实现细节以 1.9.1 页面自带的参数表为准（[[sklearn-stacking-example]]）：① `cv=None` 时用 5 折，且这些 splitter **以 `shuffle=False` 实例化**，因此跨调用折划分相同；`cv` 只用于 `cross_val_predict` 产折外预测，**不用于模型评估**；② `cv="prefit"` 假定基模型已拟合、不再重训，此时 `final_estimator_` 用的是**全训练集**上的预测，而非折外预测，文档明写此时有过拟合风险。机制与评估纪律见 [[stacking]]。

### `neighbors`

| 对象 | 关键默认/约束 |
| --- | --- |
| `KNeighborsClassifier` / `KNeighborsRegressor` | `n_neighbors=5`、`weights='uniform'`、`algorithm='auto'`、`leaf_size=30`、`p=2`、`metric='minkowski'` |
| `RadiusNeighbors*` | `radius=1.0`、`outlier_label=None` |
| `NearestNeighbors` | `n_neighbors=5`、`algorithm='auto'`、`leaf_size=30` |
| `KNeighborsTransformer` | `mode='distance'`、`n_neighbors=5` |
| `NearestCentroid` | `metric='euclidean'`、`shrink_threshold=None` |
| `NeighborhoodComponentsAnalysis` | `n_components=None`、`max_iter=50`、`tol=1e-5`、`init='auto'` |

零距离时 `weights='distance'` 退化为 1-NN（源码外部核查）。`algorithm='auto'` 的官方折叠块给出 $D>15$ 或 $k\ge N/2$ 时倾向 brute，与正文 $D<20$ 不是同一阈值。

### `neural_network`

| 对象 | 关键默认/约束 |
| --- | --- |
| `MLPClassifier` | `hidden_layer_sizes=(100,)`、`activation='relu'`、`solver='adam'`、`alpha=0.0001`、`batch_size='auto'`、`learning_rate='constant'`、`max_iter=200`、`early_stopping=False` |
| `MLPRegressor` | 同上；`loss='squared_error'`，1.7 起可加 `'poisson'` |

`alpha` 只罚 `coefs_`，不罚偏置；SGD/Adam 有 `partial_fit`，L-BFGS 没有；`n_iter_` 是本次 fit 的轮数，不跨 warm_start 累计。

### `manifold`

| 对象 | 关键默认/约束 |
| --- | --- |
| `Isomap` | `n_neighbors=5`、`n_components=2`、`neighbors_algorithm='auto'` |
| `LocallyLinearEmbedding` | `n_neighbors=5`、`n_components=2`、`reg=1e-3`、`method='standard'`、`eigen_solver='auto'` |
| `SpectralEmbedding` | `n_components=2`、`affinity='nearest_neighbors'`、`n_neighbors=None`（约 $n/10$） |
| `MDS` | `n_components=2`、`metric=True`、`n_init=1`（1.9 改）、`eps=1e-6`（1.7 改）、`normalized_stress='auto'` |
| `ClassicalMDS` | 1.8 新增；预计算距离下的 classical 解法 |
| `TSNE` | `n_components=2`、`perplexity=30.0`、`learning_rate='auto'`、`init='pca'`、`method='barnes_hut'` |

`SpectralEmbedding`/`TSNE`/`MDS`/`ClassicalMDS` 没有统一的 `transform`。`learning_rate='auto'` 是 `max(N/early_exaggeration/4,50)`，不是简单 $N/\text{ee}$。

### `cluster`

| 对象 | 关键默认/约束 |
| --- | --- |
| `KMeans` | `n_clusters=8`、`init='k-means++'`、`n_init='auto'`、`algorithm='lloyd'` |
| `MiniBatchKMeans` | `batch_size=1024`、`n_init='auto'`、`max_no_improvement=10` |
| `BisectingKMeans` | `init='random'`、`n_init=1`、`bisecting_strategy='biggest_inertia'` |
| `AffinityPropagation` | `damping=0.5`、`preference=None`、`convergence_iter=15` |
| `MeanShift` | `bandwidth=None`、`bin_seeding=False`、`cluster_all=True` |
| `SpectralClustering` | `n_clusters=8`、`affinity='rbf'`、`gamma=1.0`、`assign_labels='kmeans'` |
| `AgglomerativeClustering` | `n_clusters=2`、`metric='euclidean'`、`linkage='ward'`、`distance_threshold=None` |
| `DBSCAN` | `eps=0.5`、`min_samples=5`、`algorithm='auto'` |
| `HDBSCAN` | `min_cluster_size=5`、`min_samples=None`（→`min_cluster_size`）、`copy='warn'` |
| `OPTICS` | `min_samples=5`、`max_eps=inf`、`cluster_method='xi'`、`xi=0.05` |
| `Birch` | `threshold=0.5`、`branching_factor=50`、`n_clusters=3` |

Ward 只能欧氏距离；BIRCH 默认做全局聚类；`discretize` 的可复现性边界待目标环境验证；AMI/NMI 的 `average_method='arithmetic'` 是 0.22 起默认。

### `decomposition`

| 对象 | 关键默认/约束 |
| --- | --- |
| `PCA` | `n_components=None`、`whiten=False`、`svd_solver='auto'` |
| `IncrementalPCA` | `batch_size=None`、`whiten=False` |
| `SparsePCA` | `alpha=1`、`ridge_alpha=0.01`、`method='lars'` |
| `MiniBatchSparsePCA` | `batch_size=3`、`method='lars'`；无样本方向 `partial_fit` |
| `KernelPCA` | `kernel='linear'`、`fit_inverse_transform=False`、`eigen_solver='auto'` |
| `TruncatedSVD` | `n_components=2`、`algorithm='randomized'` |
| `SparseCoder` | `transform_algorithm='omp'`、`split_sign=False`、`positive_code=False` |
| `DictionaryLearning` | `alpha=1`、`fit_algorithm='lars'`、`split_sign=False` |
| `MiniBatchDictionaryLearning` | `batch_size=256`（1.3 改） |
| `FactorAnalysis` | `rotation=None`、`random_state=0` |
| `FastICA` | `whiten='unit-variance'`（1.3 起）、`fun='logcosh'` |
| `NMF` | `n_components='auto'`（1.6 起）、`solver='cd'`、`beta_loss='frobenius'` |
| `MiniBatchNMF` | `batch_size=1024`、`forget_factor=0.7` |
| `LatentDirichletAllocation` | `n_components=10`、`learning_method='batch'`、`learning_decay=0.7` |

PCA 只中心化；TruncatedSVD 不中心化；NMF 非负；ICA 必须白化。`split_sign` 与 `transform_algorithm` 是 API 名，不是正文的 `split_code` 与 `transform_method`。

### `covariance`

| 对象 | 关键默认/约束 |
| --- | --- |
| `EmpiricalCovariance` | `store_precision=True`、`assume_centered=False` |
| `ShrunkCovariance` | `shrinkage=0.1`（任意起点） |
| `LedoitWolf` | `block_size=1000` |
| `OAS` | 无额外强度参数 |
| `GraphicalLasso` | `alpha=0.01`、`mode='cd'`、`tol=1e-4`、`max_iter=100` |
| `GraphicalLassoCV` | `alphas=4`（路径点数）、`n_refinements=4`、`cv=None` |
| `MinCovDet` | `support_fraction=None`、`random_state=None` |
| `EllipticEnvelope` | `contamination=0.1`、`support_fraction=None` |

`raw_support_` 与重加权 `support_` 不同；`assume_centered=True` 不是自动居中；LW/OAS 的 MSE 目标不等于下游最优。

### 异常检测面

| 对象 | 关键默认/约束 |
| --- | --- |
| `IsolationForest` | `n_estimators=100`、`max_samples='auto'`（→`min(256,n)`）、`contamination='auto'`（offset −0.5） |
| `LocalOutlierFactor` | `n_neighbors=20`、`contamination='auto'`（offset −1.5）、`novelty=False` |
| `OneClassSVM` | `nu=0.5`、`gamma='scale'`；没有 `contamination` |
| `EllipticEnvelope` | `contamination=0.1` |
| `SGDOneClassSVM` | `nu=0.5`、`max_iter=1000` |

LOF 的训练与 novelty 模式使用不同统计量；不同模型的默认阈值不能直接横向比较。

## 2. 评估、切分与搜索

### `model_selection` 切分器

| 对象 | 默认/关键约束 |
| --- | --- |
| `KFold` / `StratifiedKFold` / `GroupKFold` | `n_splits=5`、`shuffle=False`、`random_state=None`（不洗牌时不可传） |
| `StratifiedGroupKFold` | 同折数；`shuffle`/`random_state` 不是 keyword-only |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | `n_splits=5`、`n_repeats=10` |
| `ShuffleSplit` / `StratifiedShuffleSplit` | `n_splits=10`、`test_size=None` |
| `GroupShuffleSplit` | `n_splits=5` |
| `TimeSeriesSplit` | `n_splits=5`、`max_train_size=None`、`test_size=None`、`gap=0` |
| `LeaveOneOut` / `LeavePOut` | LOO 每样本一折；LPO 组合划分 |
| `LeaveOneGroupOut` / `LeavePGroupsOut` | 留 1 个 / P 个组 |
| `PredefinedSplit` | `test_fold` 外部指定 |
| `train_test_split` | `test_size=None`（→0.25）、`shuffle=True`、`stratify=None` |
| `cross_val_score` | `cv=None`（→5 折）、`error_score=np.nan` |
| `cross_validate` | `return_train_score=False`、`return_estimator=False`、`return_indices=False` |
| `cross_val_predict` | `method='predict'`；只适合每样本恰好预测一次的划分 |
| `permutation_test_score` | `n_permutations=100`、`random_state=0` |

`cv=int` 在分类器上使用 `StratifiedKFold`；`cross_val_predict` 不是泛化误差度量；折间 std 不是标准误。

### 搜索器

| 对象 | 关键默认/约束 |
| --- | --- |
| `GridSearchCV` | `refit=True`、`cv=None`、`error_score=np.nan`、`return_train_score=False` |
| `RandomizedSearchCV` | `n_iter=10`、`random_state=None`；全列表不放回且 `n_iter` 受组合数限制 |
| `HalvingGridSearchCV` | `factor=3`、`resource='n_samples'`、`min_resources='exhaust'`、`cv=5`、`return_train_score=True` |
| `HalvingRandomSearchCV` | `n_candidates='exhaust'`、`min_resources='smallest'`、`cv=5`、`return_train_score=True` |

Halving 第一轮也评估全部候选；`cv_results_` 是长表；`refit` 决定重训与部分属性是否存在；`best_score_` 不是无偏估计。

### `metrics` 评分入口

| 入口 | 契约 |
| --- | --- |
| 估计器 `score` | 估计器自己的默认准则 |
| `scoring=` 字符串/callable | 统一为越大越好，符号由 scorer 处理 |
| `metrics` 函数 | 原始量纲，不做 `neg_*` 翻转 |
| `make_scorer` | `response_method='predict'`（1.8 起）、`greater_is_better=True` |
| `classification_report` | `zero_division='warn'` |
| `balanced_accuracy_score` | `adjusted=False` |
| `mean_pinball_loss` | `alpha=0.5` |
| `top_k_accuracy_score` | `k=2`；越界行为需按当前指标正文/目标环境核对 |

当前 3.4 剪藏只有 4/54 小节，不能据此回答具体指标定义。完整 scorer 注册表与文档表的差异待在 `uv` 目标版本复核。

## 3. 预处理矩阵

| 对象 | 关键默认/约束 |
| --- | --- |
| `StandardScaler` | `with_mean=True`、`with_std=True`；稀疏中心化受限 |
| `MinMaxScaler` | `feature_range=(0,1)`、`clip=False` |
| `MaxAbsScaler` | `clip=False`（1.8 新增） |
| `RobustScaler` | `quantile_range=(25,75)`、`unit_variance=False` |
| `QuantileTransformer` | `n_quantiles=1000`、`subsample=10000` |
| `quantile_transform` | `subsample=100000` |
| `PowerTransformer` | `method='yeo-johnson'`、`standardize=True` |
| `Normalizer` / `normalize` | `norm='l2'`；函数 `axis=1` |
| `OrdinalEncoder` | `handle_unknown='error'`、`encoded_missing_value=nan` |
| `OneHotEncoder` | `drop=None`、`sparse_output=True`、`handle_unknown='error'` |
| `TargetEncoder` | `smooth='auto'`、`cv=5`；`shuffle`/`random_state` 在 1.9 弃用 |
| `KBinsDiscretizer` | `n_bins=5`、`encode='onehot'`、`strategy='quantile'` |
| `Binarizer` / `binarize` | `threshold=0.0` |
| `PolynomialFeatures` | `degree=2`、`interaction_only=False`、`include_bias=True` |
| `SplineTransformer` | `n_knots=5`、`degree=3`、`extrapolation='constant'`、`handle_missing='error'` |
| `FunctionTransformer` | `validate=False`、`check_inverse=True` |
| `KernelCenterer` | 在特征空间中心化 Gram 矩阵 |

`fit` 状态只能从训练折学习。`TargetEncoder.fit` 不做交叉拟合，`fit_transform` 才做；`drop` 与未知类别可共用向量；`FunctionTransformer` 示例中的 `validate=True` 不是默认。

## 4. 版本边界汇总

- `SVC(probability=True)`：1.9 弃用，1.11 移除；
- `GradientBoosting*.criterion`：1.9 弃用；
- `MLPRegressor(loss='poisson')`：1.7 新增；
- `ClassicalMDS`：1.8 新增；
- `MDS.n_init`：1.9 从 4 改 1；`MDS.eps`：1.7 改 1e-6；
- `FastICA.whiten`：1.3 起字符串默认 `'unit-variance'`；
- `NMF.n_components`：1.6 起默认 `'auto'`；
- `MiniBatchDictionaryLearning.batch_size`：1.3 起默认 256；
- `KMeans.n_init='auto'`：1.4 起默认；
- `HDBSCAN.copy`：1.9.1 为 `'warn'`，1.10 默认将变 `True`；
- `adjusted_mutual_info_score` / NMI：`average_method` 0.22 起默认 `'arithmetic'`；
- `TargetEncoder.shuffle`/`random_state`：1.9 弃用，1.11 移除，`cv` 传 splitter；
- `KBinsDiscretizer.quantile_method`：1.7 新增，1.9 默认改 `'averaged_inverted_cdf'`；
- `SplineTransformer.handle_missing` 与 `MaxAbsScaler.clip`：1.8 新增；
- `GroupKFold.shuffle`/`random_state`：1.6 新增；
- **示例页 `plot_stack_predictors` 在 1.9 被整篇重写**：1.6–1.8 是 Ames Housing（`fetch_openml` + `ColumnTransformer` + RF/GBDT）版本，1.9 换成 1-D 合成数据 + 样条/HGBT + `SuperLearner` 对照（[[sklearn-stacking-example]]）。同一 URL 在不同版本下教不同的课，引用二手资料前需先确认版本；
- `TimeSeriesSplit.gap`/`max_train_size`/`test_size`：0.24 新增；
- `cross_validate.return_train_score`：0.21 起默认 `False`；
- `make_scorer.response_method`：1.8 起默认 `'predict'`。

## 5. 使用边界与验证纪律

- 本页不覆盖 `sklearn.impute`、文本向量化、`sklearn.calibration`、`sklearn.inspection.permutation_importance` 与 `sklearn.metrics.pairwise` 的完整用法。
- 3.4 Metrics 的指标定义级正文缺失，不能将本页当作指标大全。
- 官方示例图的读数属于来源资产，不等于本机复现或跨机基准。
- 运行时行为、报错文本、性能与实际数据收益必须在 `uv` 创建的目标版本环境中重新验证，并记录 Python、scikit-learn、NumPy/SciPy 版本、数据、种子与脚本。
- 旧实验支持的结论已删除或改为待验证，不应从 Git 历史继续当作当前事实引用。

## 关联页面

- 索引：[[index]]
- 动态目录：[[catalog]]
- 预处理与管线：[[feature-preprocessing]]、[[preprocessing-pipeline-design]]
- 评估：[[cross-validation]]、[[evaluation-metrics]]、[[hyperparameter-search]]
- 全部来源见上方 frontmatter 与各章节链接。
