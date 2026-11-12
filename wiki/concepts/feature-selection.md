---
title: "特征选择（Feature Selection）：过滤式/嵌入式/包裹式，代理指标的假设与「选了哪些」的模型依赖"
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - feature-selection
  - dimensionality-reduction
  - model-selection
  - scikit-learn
sources:
  - "[[sklearn-feature-selection]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-decision-trees]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1 的 `sklearn.feature_selection`（接口与文档论述，见 [[sklearn-feature-selection]]）
  - scikit-learn 1.6.1（本页标「实测」的运行时行为，机制层可跨版本）
---

# 特征选择（Feature Selection）

## 一句话定义与两个推论

**定义**：在拟合最终模型之前**删掉一部分原始列**，用剩下的子集继续训练（[[sklearn-feature-selection]], L10、L165）。

两个推论必须写在前面，因为它们决定了「什么时候值得做」：

1. **它是信息论的恒等运算**。删列不会给数据增加任何关于 $y$ 的信息；它能带来的收益只有两种——**降低方差**（更少参数、更少噪声维度）与**降低采集/推理成本**。因此「用了特征选择准确率就上去」只在「原先的模型被噪声维度拖累」时成立。本机实测的反例：120 样本 × 20,000 个纯噪声特征、5 折 CV，把 `SelectKBest(k=20)` 放进管线后准确率停在 **0.508**（机会水平）——选择不会凭空造出信号（[[sklearn-feature-selection]] 的「本机实测」14）。
2. **它保留列，不换坐标**。这与 [[sklearn-linear-models]] 之外的另一条降维路线（PCA/矩阵分解，见 [[matrix-factorization]]）根本不同：

| | 特征选择 | 特征抽取（PCA 等） |
| --- | --- | --- |
| 输出 | 原始列的子集 | 原始列的线性/非线性组合 |
| 可解释性 | 「保留了 `petal width`」 | 「保留了第 1 主成分」 |
| 下游稀疏性 | 保持（稀疏矩阵仍稀疏） | 一般破坏 |
| 能否表达「两个特征之差」 | 不能 | 能 |

本节标题用 "feature selection/dimensionality reduction" 混称（L10），但页内没有抽取方法——本库据此把两者分开记。

## 三个家族：判据是「是否调用估计器、是否迭代」

文档按对象组织（`VarianceThreshold`、`SelectKBest`、`RFE`、`SelectFromModel`、`SFS` 平铺在同一层），**从不点出家族名**。本库按「目标函数从哪来」重排：

| | 目标函数 | 拟合估计器 | 迭代 | 本节入口 |
| --- | --- | --- | --- | --- |
| **过滤式** filter | 每个特征的单变量统计量（$F$、$\chi^2$、互信息、方差） | 否 | 0 | `VarianceThreshold`（L12）、`SelectKBest`/`SelectPercentile`/`SelectFpr`/`SelectFdr`/`SelectFwe`/`GenericUnivariateSelect`（L39） |
| **嵌入式** embedded | 全模型拟合后的 $|\beta_j|$ 或 MDI | 是（1 次） | 1 | `SelectFromModel`（L91）、L1 线性模型（L101）、树/森林（L125） |
| **包裹式** wrapper | 子集在 CV 下的**模型分数** | 是（很多次） | $O(d)$–$O(kd)$ | `RFE`/`RFECV`（L80）、`SequentialFeatureSelector`（L151） |

三个家族各自的**盲点**不同，这才是选型的真正依据：

- **过滤式看不见冗余**。两个完全相同的强特征会得到同一个分数、同时入选；本机实测：把 iris 两列各复制一份，`SelectKBest(k=2)` 保留的是 `[0,1]`（前一份），即**冗余没有被处理，只是被平票规则决定了顺序**（[[sklearn-feature-selection]] 的「本机实测」5）。
- **嵌入式继承模型的偏好**。L1 在相关组里「大概率领随机选一个」，MDI 偏好高基数（[[regularization-and-sparsity]] 的「稀疏 ≠ 不重要」、[[ensembles]], L518）。「树选出来的重要特征」不是中立事实。
- **包裹式最贴近目标，也最容易过拟合 CV**。它的目标函数就是 CV 分数，而 CV 分数本身有方差（[[leaderboard-vs-cv]]）。

## 「选了哪些」不是数据的性质

同一份 iris、同一批 4 个特征，三种方法给出三个答案（本机实测，1.6.1）：

| 方法 | 选中的列 | 依据 |
| --- | --- | --- |
| `SelectKBest(f_classif, k=2)` | `[2, 3]`（petal length/width） | F 值排序（1180、960） |
| `LinearSVC(C=0.01, penalty="l1")` + `SelectFromModel` | `[0, 1, 2]` | 非零系数（第 3 列系数恰为 0） |
| `RFE(LogisticRegression(), n_features_to_select=2)` | `[2, 3]` | 递归删最弱系数 |

**结论**：「选中集合」是 **(数据, 模型, 超参)** 三元组的函数，不是数据的属性。因此——

1. 「这个数据集的重要特征是哪几个」这个问题**没有唯一答案**，必须写成「在 $M$ 模型、$\lambda$ 强度下被保留的是哪几个」。
2. 换了模型就必须重跑选择；用 A 模型选出的列喂给 B 模型，是一个**未经验证的假设**（[[sklearn-feature-selection]] L165–L173 的 `Pipeline` 例子恰好就是这么做的：`LinearSVC(l1)` 选、`RandomForestClassifier` 用）。
3. 下游系数/重要性的索引是**新索引**。要用 `get_feature_names_out()` 或 `get_support(indices=True)` 对回原列名（实测可用）。

## 代理指标的假设强度阶梯

过滤式的核心是「用一个便宜的量代替模型效用」。文档给了三个假设，但没排成阶梯（[[sklearn-feature-selection]], L67）：

| 打分函数 | 能测什么 | 假设 | 代价 |
| --- | --- | --- | --- |
| `r_regression` / `f_regression` / `f_classif` | 线性依赖 | 线性 + （F 检验的）高斯/独立性近似 | 最低 |
| `chi2` | 与类别的偏离（列联表） | 特征非负、语义像频次 | 低 |
| `mutual_info_*` | 任意统计依赖 | 非参数，无形式假设 | 高（需要更多样本、且是随机估计量） |

两条工程推论：

- **「无假设」要付样本代价**。互信息估计量在本机实测中**两次调用结果不同**（最大差 0.058/1.0），默认 `random_state=None`（[[sklearn-feature-selection]] 的「本机实测」6）。跨运行比较「MI 选中了谁」必须固定种子——这是一个**比模型随机性更容易被忽略**的随机源，因为 `SelectKBest` 本身看起来是确定性的。
- **打分函数选错不会被报错**。回归打分函数用在分类上「you will get useless results」（L69–L70）——不抛异常，只是没用。

## 「选多少」是超参，而且默认值不中立

三种表达个数的方式：`k`（绝对个数）、`percentile`（比例）、`threshold`/`alpha`（统计阈值）。**没有一个有「中立」默认值**：

| 类 | 默认 | 后果（实测/文档） |
| --- | --- | --- |
| `SelectKBest` | `k=10` | 特征少于 10 时静默全留；多于 10 时留 10（与信号量无关） |
| `VarianceThreshold` | `threshold=0.0` | 只删零方差列；**且阈值是方差单位**，缩放后全部方差为 1 → 一个都不删 |
| `SelectFpr`/`SelectFdr`/`SelectFwe` | `alpha=0.05` | `SelectFpr` **逐特征**不校正 → 200 个特征（198 噪声 + 2 真信号）期望留约 10 个噪声列（实测共 11 列、9 列噪声）；`Fdr`/`Fwe` 同数据只留 2 个 |
| `GenericUnivariateSelect` | `mode='percentile', param=1e-05` | 在 iris 上**只留 1 列**——不传参等于近乎全删 |
| `SelectFromModel` | `threshold=None` | L1 模型 → `1e-5`（「非零即留」）；其他 → `"mean"`（约留一半）。**两套语义共用一个默认值** |
| `RFE` | `n_features_to_select=None` | 留一半 |
| `SequentialFeatureSelector` | `n_features_to_select='auto'` | 留一半；`cv=5` |

（默认值取自官方 1.9.1 API 页签名，行为在本机 1.6.1 实测，见 [[sklearn-feature-selection]] 的「外部核查」与「本机实测」。）

因此**个数本身要用 CV 选**，官方提供的三个入口：`RFECV`（折平均分的 argmax）、`SequentialFeatureSelector`（同一个 `n_features_to_select` 参与调参）、`GenericUnivariateSelect`（`mode`/`param` 可进 `GridSearchCV`，L46 原文明说这一点）。注意 `RFECV` **没有 `n_features_to_select` 参数**，只有 `min_features_to_select`。

还有两个「选了也没用」的陷阱：

- `k > n_features` **不报错**，静默全留（实测 `k=10`、4 列 → `(150, 4)`）；`k=0` 合法，返回 `(150, 0)` 并把错误留给下游。
- **常量特征的分数是 `NaN`**（`f_classif` 打警告 `Features [i] are constant.`），`NaN` 不会被选入，但也不会让排名报错——它只是从候选里消失（实测）。

## 成本模型：以「估计器拟合次数」为单位

剪藏丢掉了唯一把成本写成公式的一段（官方 `<details>` *Details on Sequential Feature Selection*，157 词），要点如下：

| 方法 | 拟合次数 | 说明 |
| --- | --- | --- |
| 过滤式 | 0（统计量 $O(nd)$） | 一次扫描全列 |
| `SelectFromModel` | **1** | 单次拟合后按重要性截断，**不迭代** |
| `RFE`（到 $k$ 个，`step=1`） | $d-k$ 次左右 | 每删一个特征重拟合一次 |
| `RFECV` | $(d-k)\times$ 折数 | 每个特征数都要在每折上评估 |
| SFS（前向到 $k$ 个，$k$ 折） | 约 $\sum_{i<k}(d-i)\times k$ 次（本库按折叠块的「每步 $m\times k$ 次拟合」推算） | 官方例子：10 个特征取 7 个，前向 7 轮、后向 3 轮 |

**成本不对称是本节的实用判据**：要 $k$ 个特征时，若 $k > d/2$ 用**后向**更省（删的比加的多），若 $k < d/2$ 用**前向**更省。文档原话只在丢失的折叠块里，剪藏只留了「前向是贪心加、后向是贪心删」（L155、L157）。

另一个成本事实：**SFS 不需要模型有 `coef_`/`feature_importances_`**（丢失折叠块），所以它是唯一能包住任意评分器的包裹式方法——代价就是上表的拟合次数。

## 泄漏：选择必须在 CV 内侧

这是本节唯一被官方**明文推荐**的工程做法（L165–L173：把选择器放进 `Pipeline`），但文档没有给理由。本机实测（120 样本、20,000 特征、5 折）：

| 场景 | 先在全量数据上选择再 CV | 选择放进 `Pipeline` |
| --- | --- | --- |
| 纯噪声 $y$ | **0.900** | 0.508 |
| 有信号（2 个有效特征） | 0.950 | 0.792 |

两个方向都高估。「先选后 CV」的错误不是「太乐观一点」，而是**让评估集参与了选择**：`SelectKBest` 用到了全部样本的 $y$（它算 $F$ 值时用了 $y$），于是每个折的验证样本已经「帮」筛过特征。

同一个错误在小 $d$、强信号时可以忽略，在大 $d$、弱信号时会到 0.4 个准确率的量级。判据很简单：**任何碰过 `y` 的步骤（选择、缩放用的统计量、目标编码、缺失值插补）都必须在 CV 内侧**（[[leaderboard-vs-cv]] 的同一主题）。

## 缺失的一层：选择的方差

本节所有方法都是**一次性确定性输出**（除了 `mutual_info_*` 的内部随机数）。这带来一个与模型不对称的局面：

- 模型有 `random_state`、有集成、有 `warm_start`，可以估计自己的方差；
- **选择结果没有方差估计**——API 里没有「选择频率」「bootstrap 支撑集」「stability selection」的入口。

本库判断：这意味着「选出来的列」应被当作**一次抽样**而不是定论。可用的近似做法（本节未提供）：在外层重采样里重复选择并统计每列被选中的频率；或对相关特征组做整组进出（本节也无此机制）。

## 与其他概念页的关系

| 概念页 | 关系 |
| --- | --- |
| [[regularization-and-sparsity]] | 嵌入式选择的来源：L1 的零来自软阈值，$\alpha$/$C$ 是同一根旋钮；「稀疏 ≠ 不重要」直接约束「系数为 0 就删列」 |
| [[ensembles]] | MDI 的两个缺陷（训练期统计、高基数偏好）在这里变成**选列的偏差**；`permutation_importance` 是官方给出的替代方案（未入库的 5.2 节） |
| [[decision-trees]] | 树是本节最常用的嵌入式估计器；`feature_importances_` 的语义（不纯度下降 × 样本比例）决定了 `SelectFromModel` 的截断点 |
| [[support-vector-machines]] | `LinearSVC(penalty='l1')` 是分类侧的三个推荐估计器之一（L103）；$C$ 越小特征越少（L119） |
| [[nearest-neighbors]] | 反例：零参数模型没有 `coef_`，因此**不能**做嵌入式选择，只能用过滤式或 SFS（后者不要求重要性属性） |
| [[leaderboard-vs-cv]] | 泄漏与「用同一批折既选又评」的问题同源 |
| [[manifold-learning]] | **另一条降维路线：删列 vs 换坐标**。选择保住原列与稀疏性，流形学习保住几何但不给新样本定位 |
| [[matrix-factorization]] | **第三条路线：换坐标但保留可解释成分**。PCA/NMF 输出编码与 `components_`，与「删列」在目标上不同（投影 vs 子集） |
| [[naive-bayes]] | 另一个「分类器好、属性坏」的例子：`feature_log_prob_` 不是重要性度量，不能当 `SelectFromModel` 的输入 |

## 失效模式

| 失效 | 机制 | 现象 |
| --- | --- | --- |
| 冗余特征同时入选 | 过滤式逐特征打分，看不见相关 | 选 20 个特征，实际只有 2 个独立信息源 |
| 相关组里任意挑一个 | L1 的软阈值在相关组内随机 | 重跑换随机种子/换缩放，列变了 |
| ID 类列排前面 | MDI 的高基数偏好 | 用户编号/交易号被保留，留出集无收益 |
| 选完再 CV | 选择阶段用了验证集的 $y$ | CV 分数比真实高 0.1–0.4（实测 0.4） |
| 排序换列 | 平票按索引顺序取 | 特征列顺序变化改变选中集合 |
| 阈值语义误判 | `SelectFromModel(threshold=None)` 在 L1 与树上不同 | 以为「只留重要的」，实际留了一半 |
| 缩放后方差阈值失效 | `VarianceThreshold` 以方差为单位 | `StandardScaler` 之后一个都不删 |
| 不传参近全删 | `GenericUnivariateSelect()` 的 `param=1e-05` | iris 只剩 1 列 |
| 随机估计量被当确定量 | `mutual_info_*` 默认不固定种子 | 每次运行「重要特征」略变 |

## 待解问题

1. 选择结果的稳定性没有官方入口；本库未验证「重复 CV 的选择频率」在实际表格数据上是否比单次选择带来可测收益。
2. 嵌入式选择用 `permutation_importance`（未入库的 5.2 节）替代 MDI 后，`SelectFromModel` 的 `threshold` 尺度是否仍可比？未验证。
3. 多类/多输出的 `coef_`（2D）用 `norm_order` 压成 1D 时，不同 `norm_order` 如何改变选中集合？未实测。
4. one-hot 展开后的**整组进出**在 sklearn 里没有原生支持；「先分组聚合再选」是否是唯一可行路径？未验证。
5. 本节完全没讨论**选择与重采样的顺序**（先采样再选 vs 先选再采样），而这在类别不平衡时可能改变选中集合。

## 关联页面

- 来源：[[sklearn-feature-selection]]（1.13 节本体，含版本判定与三项转换损失）、[[sklearn-linear-models]]（L1 惩罚与稀疏性）、[[sklearn-ensembles]]（MDI 的缺陷）、[[sklearn-decision-trees]]（重要性的语义）
- 实践：[[feature-selection-workflow]]（选型、写法、16 条坑、可自动化程度）、[[linear-models-as-baselines]]（L1 估计器当特征筛选器）、[[competition-ml-workflow]]（特征工程阶段的位置）
- 概念：[[regularization-and-sparsity]]、[[ensembles]]、[[decision-trees]]、[[support-vector-machines]]、[[nearest-neighbors]]、[[manifold-learning]]（删列 vs 换坐标）、[[leaderboard-vs-cv]]
- 工具：[[scikit-learn]]（`feature_selection` 模块矩阵与默认值）
- 相邻机制：[[feature-preprocessing]]（`VarianceThreshold` 必须排在缩放之前，否则常数列被判为方差 1；见该页「失效模式」表第 6 条）、[[preprocessing-pipeline-design]]（顺序：「缩放 → 展开 → 选择」）
