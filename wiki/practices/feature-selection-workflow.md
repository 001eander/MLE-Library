---
title: "特征选择的落地流程：三类方法怎么选、阈值与个数怎么定、16 条坑与可自动化程度"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - feature-selection
  - pipeline
  - scikit-learn
  - workflow
sources:
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1（接口与文档论述；默认值取自 1.9.1 API 页签名）
  - scikit-learn 1.6.1（本页标「实测」的行为，机制层可跨版本）
---

# 特征选择的落地流程

本页回答四个问题：**要不要选、用哪类、阈值怎么定、怎么验**。机制与假设在 [[feature-selection]]，来源与版本在 [[sklearn-feature-selection]]。引用写作 `[[sklearn-feature-selection]], L119`。

## 一、先判「该不该选」

三个前置条件，任何一个不成立就不必做：

1. **特征数相对样本量确实大**。$d \ll n$ 时删列几乎只损失信息。官方给的两个用途是「提高准确率」与「在超高维数据上提高性能」（L10），后者是更可靠的那个。
2. **你愿意为「跑得快/可解释」付掉一点精度**。选择是信息论的恒等运算，它只能通过「降方差」间接帮忙（[[feature-selection]] 的定义段）。
3. **下游有人会读列名**。如果下游是黑箱模型且不需要解释，用 L1/正则化「软压制」通常比硬删列更省事（[[regularization-and-sparsity]]）。

反例（实测）：120 样本 × 20,000 个**纯噪声**特征，把 `SelectKBest(k=20)` 放进管线后 5 折 CV 仍是 **0.508**（机会水平）。选择不会造出信号；它只在「模型被噪声维度拖累」时有用。

## 二、按「目标函数从哪来」选类

| 你的情形 | 用哪类 | 具体入口 | 成本（拟合次数） |
| --- | --- | --- | --- |
| 先做一次廉价体检（常量列、近零方差列） | 过滤式（无监督） | `VarianceThreshold` | 0 |
| 想要「按单个特征的统计强度」快筛，$d$ 很大 | 过滤式 | `SelectKBest` / `SelectPercentile` | 0 |
| 想控制**假阳性率**而不是个数 | 过滤式（多重检验） | `SelectFpr` / `SelectFdr` / `SelectFwe` | 0 |
| 已有一个你信任的稀疏模型 | 嵌入式 | `SelectFromModel` + `Lasso`/`LogisticRegression`/`LinearSVC(l1)` | **1** |
| 下游是树/森林，想在它的重要性口径上筛 | 嵌入式 | `SelectFromModel` + `RandomForest*`/`ExtraTrees*` | 1 |
| 需要「恰好 $k$ 个」且 $k$ 小，$d$ 中等 | 包裹式 | `RFE(k=...)` / `SequentialFeatureSelector` | $O(d-k)$ / $O(kd\cdot\text{folds})$ |
| 不知道几个合适，想让 CV 定 | 包裹式 | `RFECV` / `SequentialFeatureSelector` + `GridSearchCV` | ×折数 |
| 模型**没有** `coef_`/`feature_importances_`（如 KNN、朴素贝叶斯以外的大多数） | 包裹式 | `SequentialFeatureSelector`（唯一不要求重要性属性） | 最贵 |

三条选型判据（**本库综合**，文档未给）：

- **先问「模型有没有内生重要性」**：有 → 嵌入式（1 次拟合）；没有 → 只能过滤式或包裹式。
- **再问「你信不信这个重要性度量」**：MDI 有高基数偏好、L1 在相关组里随机（[[ensembles]], L518；[[regularization-and-sparsity]]），不信就换 `SFS`（用 CV 分数当判据）。
- **最后问「$k$ 是不是硬约束」**：是 → 包裹式或 `SelectKBest(k=...)`；否 → 用统计阈值（`alpha`/`threshold="mean"`）更稳。

## 三、四类写法与各自的默认值陷阱

### 过滤式（无监督）：`VarianceThreshold`

```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.0)   # 默认值：只删零方差列
```

- 阈值单位是**方差**，与特征尺度绑定。`StandardScaler` 之后所有方差为 1，**一个都不删**（实测）。
- 布尔特征的经典用法是 $p(1-p)$：要删掉「80% 以上样本取同一值」的列，阈值取 `.8*(1-.8)`（L16–L22）。
- 它**不看 `y`**，所以不会泄漏；也因此不会帮你找「与目标相关」的列。

### 过滤式（有监督）：`SelectKBest` / `SelectPercentile` / `SelectFpr|Fdr|Fwe`

```python
from sklearn.feature_selection import SelectKBest, f_classif
sel = SelectKBest(f_classif, k=20)
```

四个默认值陷阱：

1. **`SelectKBest` 的 `k` 默认 10**（API 1.9.1），不是「全部」。`k > d` 时静默全留，不报错（实测 `k=10`、4 列 → 4 列）。
2. **`GenericUnivariateSelect()` 的默认 `mode='percentile'`、`param=1e-05`** → iris 上**只留 1 列**（实测）。它的价值在于 `mode`/`param` 可以进 `GridSearchCV`（L46 原文明说），不在默认值。
3. **`SelectFpr(alpha=0.05)` 是逐特征检验水平，不做多重校正**：200 个特征（198 噪声 + 2 真信号）的期望噪声保留数是约 10（实测共保留 11 列，其中 9 列噪声）。同一数据 `SelectFdr`/`SelectFwe` 只留 2 个（实测）。「用 `SelectFpr` 初筛」实际是在按 5% 的假阳性率灌噪声进来。
4. **打分函数与任务必须匹配**：回归打分函数用在分类上不报错，只是没用（L69–L70）；`chi2` 遇负值直接 `ValueError: Input X must be non-negative.`。

另两条安静行为：

- **常量特征打分是 `NaN`**（`Features [i] are constant.` 警告），NaN 不会被选中，排名不报错（实测）。
- **互信息是随机估计量**：`mutual_info_classif` 默认 `random_state=None`，两次调用结果不同（实测最大差 0.058）。要跨运行比较「MI 选了谁」，必须固定种子。

### 嵌入式：`SelectFromModel`

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
sel = SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.01))
```

- **默认阈值是两套语义**（API 1.9.1 + 实测）：估计器带 L1 惩罚 → `threshold_ = 1e-5`（等价「保留非零系数」）；否则 → `"mean"`（树上约留一半）。所以 `SelectFromModel(RandomForestClassifier())` 默认**不是**「只留重要的」，而是「留高于平均的一半」。
- 想「只按个数」：`threshold=-np.inf, max_features=k`（API 页原文写法；实测 `max_features=1` → 1 列）。两者同时给时先过阈值再截断。
- 三个字符串启发式：`"mean"`、`"median"`、`"0.1*mean"`（L93）。`"0.1*mean"` 在 L1 模型上通常比默认的 `1e-5` 更强（实测 `LinearSVC(l1)`：`mean` → 0.233、`median` → 0.221、`0.1*mean` → 0.023）。
- **L1 前必须缩放**。$ℓ_1$ 惩罚作用在原始尺度上，未缩放时「重要」等价于「数值大」（[[linear-models-as-baselines]] 的三条硬约束）。
- **$C$ 与 `alpha` 是同一根旋钮的反向两端**：$C$ 越小、`alpha` 越大 → 特征越少（L119）。跨模型比较「$C=0.01$ 与 $\alpha=0.01$」没有意义。
- **`prefit=True` 的坑**：文档的 doctest 用 `SelectFromModel(fitted_est, prefit=True)` 再 `transform`（L113、L140），`transform` 可用，但**不 `fit` 就读 `threshold_` 会抛 `AttributeError`**（实测 1.6.1：`threshold_` 走 `self.estimator_`，而 `_get_support_mask` 才有回退）。要看阈值就显式 `fit`。
- **树的示例数值不可复现**：L139 的 `[0.04, 0.05, 0.4, 0.4]` 来自未固定 `random_state` 的 `ExtraTreesClassifier(n_estimators=50)`，实测同一行代码得 `[0.082, 0.072, 0.386, 0.460]`。只读量级（两个花瓣特征主导），不要抄数字。
- 想避开 MDI 的偏差：官方示例自己就用 *Permutation Importance vs Random Forest Feature Importance (MDI)* 做对照（L149），但 `permutation_importance` 在未入库的 5.2 节。

### 包裹式：`RFE` / `RFECV` / `SequentialFeatureSelector`

```python
from sklearn.feature_selection import RFECV
sel = RFECV(estimator, cv=5, scoring="accuracy", min_features_to_select=1)
```

- **`RFE(n_features_to_select=None)` → 留一半**；`RFECV` **没有** `n_features_to_select` 参数（只有 `min_features_to_select=1`，API 1.9.1）。
- **`RFECV` 可能返回「全部特征」**：iris + `LogisticRegression` 实测 `n_features_=4`，因为 `mean_test_score` 从 0.953 单调升到 0.973。别把「跑过 RFECV」当成「一定降维」。
- **`RFE` 的 `step` 改的是排序粒度，不只是速度**：`step=1` 的 `ranking_` 是 `[3,2,1,1]`，`step=2` 是 `[2,2,1,1]`（实测）。报「第几重要」时要带 `step`。
- **SFS 的前向 ≠ 后向**：iris 上前向得 `[0,1]`、后向得 `[1,3]`（实测），与官方丢失折叠块的「generally … do not yield equivalent results」一致。成本不对称：要 $k$ 个时，$k>d/2$ 用后向、$k<d/2$ 用前向更省。
- **SFS 的默认 `n_features_to_select='auto'` → 一半**、`cv=5`（实测）。它是唯一**不要求模型有重要性属性**的方法，因此也是唯一能给 KNN、自定义评分器用的包裹式入口。

## 四、阈值/个数怎么定：三条路

| 路径 | 写法 | 代价 | 适用 |
| --- | --- | --- | --- |
| 内置 CV | `RFECV(...)` | 每次 CV 都跑一条 RFE 路径 | 想要「CV 折平均分最大」的那个 $k$ |
| 把个数当超参 | `GridSearchCV(SequentialFeatureSelector(...), {"n_features_to_select": [...]})` 或 `GenericUnivariateSelect` 的 `mode`/`param` | 外层 × 内层 | 需要和模型超参**联合**搜索 |
| 统计阈值 | `SelectFdr(alpha=...)` / `SelectFromModel(threshold="mean")` | 最便宜 | 只要「一个合理的子集」，不追求最优 $k$ |

注意 `RFECV` 的 `cv_results_` 也是在**同一批折**上取 argmax，和「用 CV 选超参后再报同一个 CV 分数」是同一类偏差（[[leaderboard-vs-cv]]）；需要无偏估计就套嵌套 CV。

## 五、必须放进 `Pipeline`（附实测代价）

官方推荐写法（L163–L173）：

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

clf = Pipeline([
    ("feature_selection", SelectFromModel(LinearSVC(penalty="l1"))),
    ("classification", RandomForestClassifier()),
])
clf.fit(X, y)
```

实测（120 样本、20,000 特征、5 折）：

| 场景 | 先在全量数据上选择再 CV | 选择在 `Pipeline` 内 |
| --- | --- | --- |
| 纯噪声 $y$ | **0.900** | 0.508 |
| 有信号（2 个有效特征） | 0.950 | 0.792 |

所以「先选特征、再交叉验证」不是保守做法，而是**把选择阶段的信息泄漏进评估**。判据：任何碰过 `y` 的步骤（选择、目标编码、用全量数据算的缩放统计量、插补）都在 CV 内侧。

`Pipeline` 的额外好处：`get_feature_names_out()` 能按名字返回选中的列（`SelectKBest` 实测可用），避免「第 3 列到底是哪个特征」的混乱。

## 六、16 条坑

1. **`SelectFromModel` 的 `threshold=None` 两套语义**（L1 → `1e-5`；其他 → 均值）。以为「默认只留重要的」会实际留一半。
2. **`GenericUnivariateSelect()` 默认只留 1 列**（`percentile=1e-05`）。
3. **`SelectKBest` 的 `k` 默认 10**，且在 $d<10$ 时静默全留；`k=0` 合法并返回 0 列。
4. **`SelectFpr(alpha=0.05)` 不做多重校正**：200 个特征（198 噪声）期望留约 10 个噪声列（实测共 11 列、9 列噪声）。
5. **`chi2` 遇负值报错**（`Input X must be non-negative.`），不是「取绝对值」。
6. **常量特征给 `NaN` 分数**，且只打警告。
7. **`mutual_info_*` 不固定种子** → 每次「重要特征」略有不同。
8. **特征列顺序改变选中集合**（平票取低索引，实测 `[pl,pl,pw,pw]` → 选 `[0,1]`）。
9. **`VarianceThreshold` 以方差为单位**，缩放后一个都不删。
10. **L1 前不缩放**＝按数值大小选列。
11. **`prefit=True` 下 `threshold_` 抛 `AttributeError`**（未 `fit` 时）。
12. **`RFECV` 可能留满**；且它没有 `n_features_to_select` 参数。
13. **`step` 改变 `ranking_` 的粒度**，两次运行的「第几重要」不可比。
14. **前向 vs 后向 SFS 结果不同**，且成本差一个量级。
15. **先选后 CV 会高估**（实测 0.508 → 0.900），且在弱信号、高维时最严重。
16. **选出的集合是 (数据, 模型, 超参) 的函数**：换模型必须重跑，别把 `LinearSVC` 选出的列直接喂给树并假定它「最优」。

## 七、可自动化程度（面向智能体）

| 步骤 | 可否自动 | 需要的判断 |
| --- | --- | --- |
| 删常量/近零方差列 | 完全可以 | 阈值取值（是否在缩放前） |
| 单变量快筛（`SelectKBest`/`SelectPercentile`） | 完全可以 | 打分函数与任务的匹配、固定 MI 种子 |
| 用 `GridSearchCV` 联合搜「模型超参 × 特征数」 | 完全可以 | 计算预算；外层折是否嵌套 |
| L1 嵌入式选择 | 可以 | 缩放、$C$/`alpha` 的搜索区间 |
| MDI 嵌入式选择 | 可以但要复核 | 高基数/ID 列是否被误留；是否能换成 `permutation_importance` |
| `RFECV`/`SFS` 的路径选择 | 可以 | 成本上限（$O(kd\cdot\text{folds})$ 可能不可承受） |
| 「该不该选」「选出来的列是否可信」 | **不可以** | 需要领域判断 + 外部验证；本页只给判据不给结论 |

## 关联页面

- 来源：[[sklearn-feature-selection]]（1.13 节本体、版本判定与三项转换损失）
- 概念：[[feature-selection]]（三个家族、代理指标的假设、成本与泄漏）、[[regularization-and-sparsity]]（L1 的来源与边界）、[[ensembles]]（MDI 的缺陷）、[[decision-trees]]（`feature_importances_` 的语义）、[[leaderboard-vs-cv]]（同一批折既选又评）
- 实践：[[linear-models-as-baselines]]（L1 估计器当筛选器时的缩放与共线性约束）、[[competition-ml-workflow]]（特征工程阶段的位置与「特征越多越好」的张力）、[[svm-selection-and-tuning]]（`LinearSVC` 的 `C`/`dual`）、[[cross-validation-design-and-tuning]]（泄漏的机制解释与 `Pipeline` 骨架）
- 工具：[[scikit-learn]]（`feature_selection` 模块矩阵与默认值清单）
- 相邻实践：[[preprocessing-pipeline-design]]（预处理的七阶段顺序；`VarianceThreshold` 在缩放之前、选择在展开之后）与 [[feature-preprocessing]]（变换是否吃 $y$ 决定泄漏的量级：≤0.02 / +0.055 / +0.39）
