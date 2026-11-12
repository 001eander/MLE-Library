---
title: "scikit-learn 用户指南 1.13「Feature selection」：方差/单变量筛选、RFE 与 SFS、嵌入式选择（stable 文档剪藏，未回填）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - feature-selection
  - dimensionality-reduction
  - reference-doc
source_path: raw/articles/sklearn/1.13. Feature selection.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.13「Feature selection」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 1.13. Feature selection（scikit-learn 用户指南第 1.13 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；4 段 doctest 代码块 + 1 段非 doctest 片段、1 个行间公式、2 个 admonition（`[!warning]`、`[!note]`）、**0 张图、0 张表** |
| 原文 | <https://scikit-learn.org/stable/modules/feature_selection.html>（剪藏 frontmatter `source`，未钉版本；2026-11-11 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/1.13. Feature selection.md`（175 行、18,227 字节、`wc -w` 1,545 词；**未做任何回填**，末行有换行） |
| SHA-256 | `45bf420dc7f0e33be6380d3863fb55d9e22326692f2598fb3465f33580e1f8c7` |
| 完整性 | **3 个内容块全丢（1 个 `aside` 主题框 35 词 + 2 个 `<details>` 折叠块 194 + 157 词 = 386 词）**、整段 References（`[sfs]` Ferri et al.）、1 处引用标记 `[sfs]`；正文其余段落逐句对齐。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x，可进一步定为 1.9.1**（四条独立证据，见「版本判定」） |
| 插图 | 无（官方 1.9.1 页本节 `<article>` 内 `<img>` 计数为 0），因此**未新增 `raw/assets/` 目录** |
| 代码 | 4 段 doctest：`VarianceThreshold`（L24–L35）、`SelectKBest`+`f_classif`（L50–L60）、`LinearSVC(l1)`+`SelectFromModel`（L105–L117）、`ExtraTreesClassifier`+`SelectFromModel`（L129–L144）；1 段非 doctest 的 `Pipeline` 片段（L167–L173） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-feature-selection]], L43`。凡超出剪藏内容的部分（API 默认值、多假设检验语义、版本史、运行时行为）都显式标注「外部核查」「官方 1.9.1 页」或「本机实测」。

> [!warning] 本文件未被修改
> 与 [[sklearn-svm]]、[[sklearn-naive-bayes]] 不同，本次**没有取得所有者对 `raw/` 回填的授权**，因此剪藏保持抓取时的原始字节（sha256 `45bf420d…`）。丢失的 3 个内容块只在「转换损失」一节里按官方 1.9.1 页逐句记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库第一个**「选择」类目**（此前十三个来源都是「模型族」或「组合策略」）：它不引入新的假设形式，而是**在拟合前删掉列**。对本库的意义有五层。

1. **第一次把「降维」拆成两条互不等价的路**。本节标题叫 feature selection，但引言写的是 "feature selection/dimensionality reduction"（L10）——而页内全部方法都是**保留原始列的子集**，没有一个做线性/非线性组合（那是 2.5 节 `sklearn.decomposition`，未入库）。因此「少了几列」与「换了一套坐标」在可解释性与下游稀疏性上是两件事（[[feature-selection]]）。
2. **三个家族在一页里全齐，但文档从不点出它们的名字**。1.13.1–1.13.2 是**过滤式**（只看每个特征与 $y$ 的单变量统计量，不看模型）；1.13.3 与 1.13.5 是**包裹式**（用模型的 CV 分数当目标函数）；1.13.4 是**嵌入式**（用模型自己的 `coef_`/`feature_importances_` 当重要性）。三者的成本差一个数量级以上（后文「成本」）。
3. **「选了哪些」不是数据的性质，而是 (数据, 模型, 超参) 三元组的性质**。同一份 iris 数据：`f_classif` 选 `[2,3]`，`LinearSVC(C=0.01, penalty="l1")` 选 `[0,1,2]`，`RFE(LogisticRegression)` 选 `[2,3]`（本机实测）。文档给了机制但没给这条判据，本页把它写成显式结论。
4. **本节把「正则化」与「特征选择」接上了**。L119 说明 $C$ 与 `alpha` 是同一根稀疏性旋钮的两个方向（$C$ 越小、$\alpha$ 越大 → 特征越少），这让 [[regularization-and-sparsity]] 里的「稀疏 ≠ 不重要」直接变成选型约束。
5. **唯一一节把「泄漏」写成推荐做法的**：1.13.6 直接给出 `Pipeline([('feature_selection', ...), ('classification', ...)])`（L165–L173），这是本库第一条**官方明文要求把选择放进 Pipeline**的论述。本机实测：120 样本 × 20,000 纯噪声特征，管线内 5 折 CV 0.508（机会水平），先在全量数据上选择再 CV 0.900。

## 文档结构

| 章节 | 解决什么问题 | 家族 | 行号 |
| --- | --- | --- | --- |
| 引言 | 两个用途：提准确率、在超高维上提性能 | — | L10 |
| 1.13.1 Removing features with low variance | 删掉方差低于阈值的列；默认删零方差；布尔特征的 $p(1-p)$ 阈值算法 | 过滤式（无监督） | L12–L37 |
| 1.13.2 Univariate feature selection | `SelectKBest`/`SelectPercentile`/`SelectFpr`/`SelectFdr`/`SelectFwe`/`GenericUnivariateSelect`；6 个打分函数；F 检验 vs 互信息 vs $\chi^2$ | 过滤式（有监督） | L39–L78 |
| （丢失的 `aside`）Feature selection with sparse data | `chi2`/`mutual_info_*` 不把稀疏矩阵稠密化 | — | 官方页；剪藏无 |
| 1.13.3 Recursive feature elimination | `RFE`（递归删最不重要）+ `RFECV`（CV 自动定个数） | 包裹式 | L80–L89 |
| 1.13.4 Feature selection using SelectFromModel | 元变换器：按 `threshold`（数值或 `"mean"`/`"median"`/`"0.1*mean"`）与 `max_features` 过滤 | 嵌入式 | L91–L99 |
| 1.13.4.1 L1-based feature selection | `Lasso`（回归）/`LogisticRegression`/`LinearSVC`（分类）的零系数；$C$ 与 `alpha` 的方向 | 嵌入式 | L101–L123 |
| （丢失的 `<details>`）L1-recovery and compressive sensing | Lasso 何时能**精确**恢复支撑集；为何没有通用 `alpha` 规则；`LassoCV`/`LassoLarsIC` 的偏差方向 | 嵌入式（理论） | 官方页；剪藏无 |
| 1.13.4.2 Tree-based feature selection | 不纯度重要性（MDI）当重要性；`ExtraTreesClassifier` 示例 | 嵌入式 | L125–L149 |
| 1.13.5 Sequential Feature Selection | 前向/后向贪心，用 CV 分数当判据；`n_features_to_select`、`direction` | 包裹式 | L151–L161 |
| （丢失的 `<details>`）Details on Sequential Feature Selection | 前向 ≠ 后向；迭代次数的算术；与 RFE/`SelectFromModel` 的成本对比 | 包裹式（成本） | 官方页；剪藏无 |
| 1.13.6 Feature selection as part of a pipeline | 把选择放进 `Pipeline` 的写法与理由 | 工程 | L163–L175 |
| （丢失的 References） | `[sfs]` Ferri et al. | — | 官方页；剪藏无 |

## 关键主张

### 1. 三人家族的判据：**是否看模型、是否迭代**（本库综合）

文档按「对象」组织，不按「家族」组织，因此三者混排在同一层（L43–L46 与 L93、L153 并列）。本库按**是否调用估计器**重排：

| | 目标函数 | 是否拟合模型 | 迭代次数 | 本节入口 |
| --- | --- | --- | --- | --- |
| 过滤式 | 每个特征的 $t$/$F$/$\chi^2$/MI 统计量 | 否（只看单变量统计） | 0 | L12、L39 |
| 嵌入式 | 全模型拟合后的 $|\beta_j|$ 或 MDI | 是（一次） | 1 | L91、L101、L125 |
| 包裹式 | 子集在 CV 下的模型分数 | 是（很多次） | $O(d)$–$O(kd)$ | L80、L151 |

这条重排解释了三个实际后果：过滤式**看不见冗余**（两个完全相同的强特征得同一个分数，见「本机实测」5）；嵌入式**继承模型的偏好**（L1 偏好相关组里随机留一个，MDI 偏好高基数，见 [[ensembles]], L518）；包裹式**最贵但最贴近目标**，也最容易过拟合 CV（[[leaderboard-vs-cv]]）。

### 2. 单变量统计量是**代理**，它的假设写在函数名里（来源论述 + 本库综合）

L67 三句话给了三个假设：F 检验只测**线性**依赖（"estimate the degree of linear dependency"）；互信息能抓任意依赖但"being nonparametric, they require more samples"；$\chi^2$ 只适用于非负特征。加上 L69 的警告（回归打分函数用在分类问题上"you will get useless results"）与 L72–L73 的 note（`SelectKBest`/`SelectPercentile` 可以在 `score_func` 接受 `y=None` 时做无监督选择）。

**本库综合**：这四句构成一个**假设强度阶梯**，而文档没有把它写成阶梯——`f_classif` 最强（线性 + 高斯近似）、`chi2` 次之（非负+频次语义）、`mutual_info_*` 最弱（无参数形式，但估计量方差大）。选择打分函数等于选择「我认为信号长什么样」，而这一点在文档里只以警告形式出现。

### 3. 稀疏性旋钮有两个方向，$C$ 与 `alpha` 反向（来源论述）

L119 一句给足两个方向：「With SVMs and logistic regression, the parameter `C` controls the sparsity: the smaller `C` the fewer features selected. With Lasso, the higher the `alpha` parameter, the fewer features selected.」

**本库综合**：这不是两个规则，而是同一个（惩罚强度的两个参数化）。跨模型比较「$C=0.01$ 与 $\alpha=0.01$ 哪个更稀疏」没有意义——与 [[support-vector-machines]] 里「$C=1/\alpha$ 只对 Ridge 有闭式对应」是同一处陷阱。而 L103 只说 L1 惩罚的线性模型「have sparse solutions」（零系数），没有说**零是软阈值的产物**（[[regularization-and-sparsity]] 的「稀疏性从哪来」）。

### 4. `SelectFromModel` 的默认阈值不在正文里（外部核查）

正文只说「below the provided `threshold` parameter」，然后列出三个字符串启发式（"mean"、"median"、"0.1\*mean"，L93）。**正文完全没有说 `threshold=None` 时用哪个**。

外部核查（官方 1.9.1 API 页）：`threshold=None` 时，**若估计器带 L1 惩罚（显式或隐式，如 `Lasso`）→ 阈值取 `1e-5`；否则取 `"mean"`**。本机实测 1.6.1 与该描述一致：`LinearSVC(C=0.01, penalty="l1")` 的 `threshold_` = `1e-05`；`ExtraTreesClassifier` 的重要性 `[0.078, 0.066, 0.420, 0.436]`、`threshold_` = 0.25（= 均值）、保留 2 列。

后果：**同一个 `SelectFromModel` 在 L1 模型上等于「保留所有非零系数」，在树上等于「保留高于平均值的一半左右」**——两种完全不同的语义共用一个默认值，而正文没说。

### 5. `prefit=True` 的 doctest 能跑，但 `threshold_` 不能读（本机实测）

L113 与 L140 的 doctest 都用 `SelectFromModel(estimator, prefit=True)` 再直接 `transform`。本机 1.6.1 复现：`transform` 正常（`(150, 3)` / `(150, 2)`），但**不调用 `fit` 就读 `threshold_` 会抛 `AttributeError: 'SelectFromModel' object has no attribute 'estimator_'`**——因为 `threshold_` 走 `self.estimator_`，而 `_get_support_mask` 有 `getattr(self, "estimator_", self.estimator)` 的回退（源码 `feature_selection/_from_model.py`）。

即：文档的写法在**推理路径**上可用，在**诊断路径**上不可用。想知道阈值/保留掩码的数值，要么显式 `fit`，要么从 `get_support()` 反推。

### 6. RFE 的「重要性」来自任意属性或 callable（来源论述）

L82 明确：重要性可以来自「any specific attribute (such as `coef_`, `feature_importances_`) or callable」。`SelectFromModel` 同（L93 的 `importance_getter`）。

**本库综合**：这使得「RFE」不是一个算法，而是一个**外壳**——换成什么重要性，就换成什么模型的偏好。RFE + MDI 会继承 MDI 的高基数偏好，RFE + L1 会继承相关组的随机选择。文档把「估计器可选」写成了灵活性，代价是**结果不可复现地依赖估计器选择**。

### 7. `RFECV` 的「最优个数」是折平均分的 argmax（来源论述）

L84 三句把机制说全了：在每个 CV 划分上拟合一个 `RFE`、用 `scorer` 评估不同特征数、跨折平均、取最大值的那个数。

**本库综合**：这是本节唯一**把特征数当超参并用 CV 选**的路径，也正因如此它继承了 CV 的全部问题（折数、分层、分组结构，见 [[leaderboard-vs-cv]]）。本机实测 iris + `LogisticRegression`：`cv_results_['mean_test_score']` 从 0.953（1 个特征）单调升到 0.973（4 个特征）→ `n_features_=4`，即**RFECV 可以返回「全部特征」**，这与「特征选择总能减维」的直觉相反。

### 8. 树的重要性是训练期统计量，`SelectFromModel` 直接继承了它的偏差（来源论述 + 本库综合）

L127 把 MDI 与「丢弃无关特征」接起来，L148–L149 两个示例链接中，第二个标题就写着 *Permutation Importance vs Random Forest Feature Importance (MDI)*——**文档自己指出了 MDI 的 caveat，却把它放进示例链接而不是正文**。

**本库综合**：这条与 [[ensembles]], L513–L520 的 MDI 两条缺陷（训练集统计、高基数偏好）是同一件事；本节的新增信息是**它在选择场景里的具体后果**——用 MDI 筛 ID 类列，会把 ID 排在前面并保留下来（[[feature-selection]] 的失效模式表）。

### 9. 找不到「采样/重复」这一层（剪藏的结构性缺口）

本节所有方法都假定**一次性确定性选择**：没有 bootstrap 选择频率、没有 stability selection、没有「重复 CV 取交集」。文档唯一提到随机性的地方是 `ExtraTreesClassifier` 的示例（L136，且没写 `random_state`）。

**本库判断**：这是「选择」类目与「模型」类目最大的工程差异——模型有 `random_state` 与集成来降方差，而**选择结果本身没有方差估计**，除非自己套一层重采样。本节的 API 里也没有这个入口（见「待解问题」）。

### 10. 推荐做法是 Pipeline，理由却没写（来源论述 + 本库实测）

L165 说「The recommended way to do this in scikit-learn is to use a `Pipeline`」，L167–L173 给了两行例子。文档**没有解释为什么**（没有提泄漏、没有提 CV）。

本机实测把这句推荐变成可量化结论（1.6.1，120 样本、20,000 特征）：

| 场景 | 选择放在 CV 外（先选再 CV） | 选择放在 CV 内（`Pipeline`） |
| --- | --- | --- |
| 纯噪声标签（$y$ 与 $X$ 独立） | **0.900** | 0.508 |
| 真实信号（2 个有效特征，$y$ 由它们决定） | 0.950 | 0.792 |

两行的方向相反：噪声上「先选」大幅**高估**（0.508 → 0.900），信号上「先选」也**高估**（0.792 → 0.950）。所以「先选特征再交叉验证」不是保守做法，而是**把选择阶段的信息泄漏进评估**。

## 版本判定

结论：**1.9.x，可进一步定为 1.9.1**。四条独立证据（均来自把剪藏逐句与官方 1.6/1.7/1.8/1.9 四页比对）：

| # | 剪藏 | 官方页差异 | 排除的版本 |
| --- | --- | --- | --- |
| 1 | L48「we can use **an** F-test」 | 1.6/1.7/1.8 均为「**a** F-test」，1.9 改为「an」 | 1.6、1.7、1.8 |
| 2 | L84「different **numbers** of selected features」 | 1.6 为「different **number** of selected features」 | 1.6 |
| 3 | L84「The performance of the RFE selector **is** evaluated」 | 1.6 为「**are** evaluated」 | 1.6 |
| 4 | L139 `array([ 0.04,  0.05,  0.4,  0.4])` | 1.6 为 `array([ 0.04...,  0.05...,  0.4...,  0.4...])`（旧 repr 截断） | 1.6 |

补充：L93–L94「importance of the feature values **is** below」在 1.6 为「**are** below」；L119「logistic regression」（无连字符）在 1.6 为「logistic-regression」——两条同向支持「≥1.7」。

官方 1.7 与 1.8 两版本节**逐词相同**（word-level diff 为空），1.8→1.9 只有两处改动（「a」→「an」；删除 Baraniuk 条目末尾的一个 URL），因此**无法用措辞把 1.9 与 1.9.1 分开**；抓取日 `/stable/` 页标题为 "scikit-learn 1.9.1 documentation"，故取 1.9.1。本节所有功能性结论都不依赖这条判定。

## 转换损失

官方 1.9.1 页本节正文约 1,999 词（含代码与公式 token），剪藏 1,545 词。丢失的是 **3 个内容块 + 整段 References**：

| 丢失项 | 形态 | 词数 | 内容要点（据官方 1.9.1 页） |
| --- | --- | --- | --- |
| *Feature selection with sparse data* | `aside class="topic"` | ≈35 | 用稀疏矩阵（sparse matrices）时，`chi2`、`mutual_info_regression`、`mutual_info_classif` **不会把数据稠密化**；其余方法未提 |
| *L1-recovery and compressive sensing* | `<details>` | 194 | Lasso 在特定条件下能**精确恢复**非零变量集合；「sufficiently large」取决于非零系数个数、$\log$ 特征数、噪声量、最小非零绝对值、设计矩阵结构与相关性；**没有通用 `alpha` 规则**，`LassoCV`/`LassoLarsCV` 可能欠惩罚（多留几个无关变量对预测无害）、`LassoLarsIC`（BIC）倾向过大 `alpha`；附 `[Baraniuk2007]` |
| *Details on Sequential Feature Selection* | `<details>` | 157 | 前向/后向**一般不等价**；成本不对称（10 个特征取 7 个：前向 7 轮、后向 3 轮）；SFS **不需要**模型有 `coef_`/`feature_importances_`，但更慢——后向从 $m$ 到 $m-1$ 用 $k$ 折要拟合 $m\times k$ 个模型，而 `RFE` 一步只要 1 次拟合、`SelectFromModel` 只要 1 次且不迭代 |
| References | 尾段 | 1 条 | `[sfs]` Ferri et al., *Comparative study of techniques for large-scale feature selection* |

合计 **386 词散文 ≈ 官方本节 19%**，另丢 1 条文献与 1 处引用标记（L153 的 `Sequential Feature Selection (SFS)` 原文带 `[sfs]`）。

结构性后果（本库判断）：

- 丢的三块恰好是**三类「读起来像背景」的内容**：稀疏支持、Lasso 恢复的理论前提、包裹式选择的成本算术。因此剪藏在**接口层**完整（所有类名、参数名、doctest 都在），但失去了：为什么 `chi2` 能吃稀疏输入、「Lasso 选出的集合」在什么条件下可信、以及**为什么 SFS 比 RFE 贵一个量级**（后者是唯一把成本写成公式的地方）。
- 第 194 词那块的理论前提是本库目前**唯一**能回答「L1 选择的支撑集可信吗」的文档论述，剪藏丢失后只能引官方页。

公式与文本的转换噪声（不影响语义，但影响搜索）：L18–L20 的 `\[Var[X]=p(1-p)\]` 被转成 `$$Var \left[X\right] = p \left(1 - p\right)$$`；L67 的 `$\chi^{2}$ -test` 多了一个空格（原文 `\(\chi^2\)-test`）。

## 外部核查（不在剪藏内）

### A. 估计器默认值（官方 1.9.1 API 页签名）

| 估计器 | 签名（1.9.1） |
| --- | --- |
| `VarianceThreshold` | `(threshold=0.0)` |
| `SelectKBest` | `(score_func=f_classif, *, k=10)` |
| `SelectPercentile` | `(score_func=f_classif, *, percentile=10)` |
| `SelectFpr` / `SelectFdr` / `SelectFwe` | `(*, score_func=f_classif, alpha=0.05)` |
| `GenericUnivariateSelect` | `(score_func=f_classif, *, mode='percentile', param=1e-05)` |
| `RFE` | `(estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter='auto')` |
| `RFECV` | `(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto')` |
| `SelectFromModel` | `(estimator, *, threshold=None, prefit=False, norm_order=1, max_features=None, importance_getter='auto')` |
| `SequentialFeatureSelector` | `(estimator, *, n_features_to_select='auto', tol=None, direction='forward', scoring=None, cv=5, n_jobs=None)` |

两条与正文直接冲突/补充的项：

1. **`SelectKBest` 的 `k` 默认 10、`SelectPercentile` 的 `percentile` 默认 10**——正文两个 doctest 都显式传了 `k=2`，读者可能以为无默认。
2. **`RFECV` 没有 `n_features_to_select`**（1.9.1 签名里只有 `min_features_to_select=1`）；正文说「个数由 CV 自动定」（L84）与之一致，但如果按 `RFE` 的参数名去配 `GridSearchCV` 会直接 TypeError。

### B. 多假设检验语义（源码 `feature_selection/_univariate_selection.py` @ 1.6.1，1.9.1 同构）

正文只把 `SelectFpr`/`SelectFdr`/`SelectFwe` 分别描述为「false positive rate / false discovery rate / family wise error」（L45），**没有一个公式**。源码三行给出全部语义：

- `SelectFpr`：`pvalues_ < alpha` —— **逐特征**检验水平，不做任何校正。
- `SelectFdr`：Benjamini–Hochberg（排序后 `p_(i) <= alpha/n * i`，再取满足者中的最大 $p$ 对应的全体）。
- `SelectFwe`：`pvalues_ < alpha / n_features` —— Bonferroni。

因此 `alpha` 在三个类里**同名不同义**：`SelectFpr(alpha=0.05)` 在 200 个特征上（198 个纯噪声 + 2 个真信号）**期望保留约 $0.05\times198\approx10$ 个噪声列**（本机实测：共保留 11 列，其中 9 列是噪声），而 `SelectFdr`/`SelectFwe` 同一数据只保留 2 个（正好是两个真信号）。

### C. `prefit` 相关实现细节（源码 `_from_model.py` @ 1.6.1）

- `_get_support_mask` 用 `getattr(self, "estimator_", self.estimator)` 回退 → `prefit=True` 且未 `fit` 时 `transform` 可用。
- `threshold_` 属性**不做回退**，直接用 `self.estimator_` → 未 `fit` 时 `AttributeError`（本机实测）。
- `SelectFromModel(threshold=-np.inf, max_features=k)` 是「只用 `max_features`、不看阈值」的官方写法（API 页原文）；本机实测 `max_features=1` → 保留 1 列。
- `SelectFromModel` 的 `threshold` 与 `max_features` **同时生效时取交集后截断**（API 页：「If … max_features is not None, then all features are kept… To only select based on max_features, set threshold=-np.inf」）。

## 本机实测（scikit-learn 1.6.1；**不是** 1.9.1 的数值）

1. **`VarianceThreshold` doctest 逐字复现**（L24–L35）：`threshold=.8*(1-.8)` → 删掉第 0 列；`variances_ = [0.1389, 0.2222, 0.25]`（阈值 0.16）。
2. **`SelectKBest(f_classif, k=2)` 在 iris 上选 `[2,3]`**，`scores_ = [119.26, 49.16, 1180.16, 960.01]`。注意第 1 个特征的 F 值只有 49，与第 2、3 个差一个量级 → 「k 选 2」与「F 值有断崖」是两件事。
3. **`chi2` 遇负值直接 `ValueError: Input X must be non-negative.`**（L67 的「non-negative」是硬约束，不是建议）。
4. **常量特征的 `f_classif` 打分为 `NaN`**：产生 `Features [4] are constant.` 与 `invalid value encountered in divide` 两条警告，`scores_` 该位为 `nan`；`SelectKBest(k=4)` 因此**只保留 4 个有效特征中的 4 个**——NaN 不会被选入，且没有额外提示。
5. **平票取低索引**：把 iris 的第 2、3 列各复制一份（`[pl,pl,pw,pw]`），`SelectKBest(k=2)` 保留 `[0,1]`（前一份），分数量级相同（1180.16 / 960.01）。**特征顺序会改变选中的列**。
6. **`mutual_info_classif` 默认 `random_state=None`，两次调用结果不同**：`[0.5235, 0.2164, 0.9795, 0.9762]` vs `[0.5012, 0.2746, 0.9861, 0.9811]`（最大差 0.058）。跨运行比较「MI 选中了谁」必须固定种子。
7. **`SelectFromModel` + L1 的默认阈值是 `1e-5`**：`threshold_ = 1e-05`，保留 `[0,1,2]`（与 L112–L116 的 `(150, 3)` 一致）。
8. **`SelectFromModel` + 树的默认阈值是均值**：`threshold_ = 0.25`（= `feature_importances_` 均值），保留 2 列；`ExtraTreesClassifier` 不加 `random_state` 时重要性为 `[0.082, 0.072, 0.386, 0.460]`，**与 L139 的 `[0.04, 0.05, 0.4, 0.4]` 不同**（文档值不可复现，且其和 0.89 ≠ 1）。
9. **`GenericUnivariateSelect()` 的默认参数几乎删光特征**：`mode='percentile'`、`param=1e-05` → 在 iris 上**只保留 1 列**（索引 2）。这是一个「不传参就近乎全删」的默认值。
10. **`SelectKBest(k > n_features)` 静默保留全部**（`k=10`、4 个特征 → 形状 `(150, 4)`，不报错）；`k=0` → 形状 `(150, 0)`（合法，下游会炸）。
11. **`RFE` 的 `n_features_to_select=None` → 保留一半**（iris 4 列 → 2 列，`ranking_` 形如 `[3,2,1,1]`）；`step=2` 时 `ranking_` 变成 `[2,2,1,1]`——**`step` 改变的是排序粒度，不只是速度**。
12. **`RFECV` 可以选满**（`LogisticRegression` + iris：`n_features_ = 4`）。
13. **`SequentialFeatureSelector` 默认 `n_features_to_select='auto'` → 一半**（4 → 2），且默认 `cv=5`；`direction='backward'` 得到 `[1,3]`，与前向 `[0,1]` 不同——这是丢失折叠块里「前向/后向一般不等价」的实证。
14. **`Pipeline` vs 先选后 CV**（120×20,000，5 折）：纯噪声 0.508 vs 0.900；有信号 0.792 vs 0.950（见「关键主张 10」）。
15. **`feature_names_in_` / `get_feature_names_out()` 在 `SelectKBest` 上可用**：DataFrame 输入 → `feature_names_in_ = ['sl','sw','pl','pw']`，选择后 `['pl','pw']`。这使「选中了哪些列」可以按名字核对，而不必读索引。
16. **`StandardScaler` 之后 `VarianceThreshold()` 一个都不删**（方差全为 1.0）：方差阈值是**尺度相关**的，放在缩放之后等于失效。

**版本警告**：本机是 1.6.1，只能证明机制存在，不能替代对 1.9.1 的数值核对。以上 16 条中，第 4、9、10、11、12、13 条依赖实现细节（评分函数的 NaN 分支、默认 `param`/`auto` 的启发式、CV 折数），引用时须带版本。

## 分歧与待解问题

### 分歧

1. **标题与内容的口径不一致**：引言写 "feature selection/dimensionality reduction"（L10），但页内**没有任何降维/抽取方法**（无 PCA、无投影）。「dimensionality reduction」在这里只能是「删列后的维数下降」，与 2.5 节的矩阵分解不是同一件事（本库判断）。
2. **`SelectFromModel` 的默认阈值两套语义**：L1 模型取 `1e-5`（等于「非零即留」），其余取 `"mean"`（等于「留一半左右」）。正文从未说明，两种语义共用同一个 `threshold=None`（外部核查 + 实测，见「关键主张 4」）。
3. **文档称 `ExtraTreesClassifier` 示例的重要性是 `[0.04, 0.05, 0.4, 0.4]`**（L139），但该 doctest 没有 `random_state`，本机复跑得到 `[0.082, 0.072, 0.386, 0.460]`，且文档值之和为 0.89（未归一化到 1）。**该 doctest 的数值不可复现**，只能当「量级：两个特征主导」读。
4. **`prefit=True` 的 doctest 与 `threshold_` 的不可读**：文档写法（L113、L140）在推理路径上成立、在诊断路径上抛 `AttributeError`（本机实测）。属文档与实现的**接口错位**，非矛盾，但会让人以为 `threshold_` 可在未 `fit` 时读。
5. **`SelectFpr`/`Fdr`/`Fwe` 的 `alpha` 同名不同义**：正文只给三个名字（L45），没有给出「`SelectFpr` 不做多重校正」这一关键差别；`alpha=0.05` 在 200 个特征（198 噪声 + 2 真信号）上期望留约 10 列噪声（实测共 11 列、9 列是噪声），这对「用 `SelectFpr` 做初筛」是一个反直觉的默认行为。
6. **「特征选择提高准确率」这一前提在本节只有一句无条件的宣称**（L10 "either to improve estimators' accuracy scores or to boost their performance on very high-dimensional datasets"），而本机实测显示：纯噪声上管线内选择只能到机会水平（0.508），即**选择本身不创造信息**。文档没有这个限定。

### 待解问题

1. `RFECV` 的折内选择是否真的避免了乐观偏差？它的 `cv_results_` 仍是在同一批折上取 argmax（本机实测 `n_features_=4` 时 `mean_test_score` 单调），这与「用 CV 选超参后再报同一个 CV 分数」是同一类问题（[[leaderboard-vs-cv]]）。未做嵌套 CV 对照。
2. `SelectFromModel` 的 `importance_getter='auto'` 在**多输出**回归（`coef_` 为 2D）与多类分类（`coef_` 为 `(n_classes, n_features)`）上按 `norm_order=1` 取行范数——`norm_order` 的其他取值会怎样改变选择？（未实测）
3. 稀疏输入下 `VarianceThreshold` / `SelectKBest` 是否也保持稀疏（丢失的 `aside` 只提了 `chi2` 与 `mutual_info_*`）；未实测 `scipy.sparse` 输入下各选择器的返回类型。
4. `SequentialFeatureSelector` 的 `tol` 语义（「分数不提升至少 `tol` 就停」）在官方 API 页有、剪藏无；与 `n_features_to_select` 同时给出时谁优先？未实测。
5. 选择结果的**方差**：本节没有稳定性选择（stability selection）的入口。用 `BaggingClassifier` 包 `SelectFromModel` 或重复 CV 取交集，是否能得到可用的「选择频率」？（未做）
6. `LassoCV`/`LassoLarsIC` 与「恢复支撑集」的取舍在丢失的折叠块里只有定性描述；本库未读 Baraniuk (2007) 与 Ferri et al.，也未验证「`LassoCV` 欠惩罚」在合成数据上的表现。
7. 本节示例链接指向 5 个官方示例（`plot_feature_selection.py`、`plot_f_test_vs_mi.py`、`plot_rfe_digits.py`、`plot_rfe_with_cross_validation.py`、`plot_select_from_model_diabetes.py`、`plot_permutation_importance.py`），**均未入库**；其中 F-test vs MI 的对比图是目前唯一能给出「何时 MI 有优势」的定量证据。
8. 与 8.3 Preprocessing（未入库）的交互：`VarianceThreshold` 与 `StandardScaler` 的先后、`PolynomialFeatures` 之后的选择，本节完全没有讨论。

## 局限

- 本节**不覆盖** `sklearn.decomposition`（PCA/ICA/NMF 等真正的降维）、`sklearn.inspection.permutation_importance`（MDI 的替代方案）、`sklearn.model_selection`（`RFECV`/`SFS` 内部用的 CV 切分）。因此「该用 PCA 还是该删列」在本页**无法回答**。
- 本节**没有任何多输出/多标签选择**的讨论，也没有「分组特征（同源）必须整组进出」的机制（例如 one-hot 展开后的整组保留），而这在工程上是 `SelectFromModel` 最常见的需求。
- 丢失的 3 个内容块（386 词 ≈ 19%）里包含**唯一**关于稀疏输入、Lasso 恢复条件与包裹式成本公式的论述；本页对这些内容的引用都指向官方 1.9.1 页，**不能引剪藏行号**。
- 官方页**没有给出任何一条「该选多少个特征」的经验规则**，也没有「选择前后模型性能差多少」的示例读数（示例都是外部脚本）。本页因此不提供任何推荐 k 值。
- 运行时复核只做到 1.6.1（PyPI 当前最高可安装版本）；`GenericUnivariateSelect` 的 `param=1e-5`、`SFS` 的 `n_features_to_select='auto'` 等默认值在 1.9.1 上**未复跑**（签名已核，行为未核）。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 引言的两种用途 | L10 |
| `VarianceThreshold` 与 $p(1-p)$ 阈值 | L14–L22 |
| 布尔特征 doctest | L24–L37 |
| 单变量选择的五个类 | L43–L46 |
| `SelectKBest` + `f_classif` doctest | L48–L60 |
| 6 个打分函数 | L62–L65 |
| F 检验 vs 互信息 vs $\chi^2$ 的假设 | L67 |
| 回归打分函数用错任务的警告 | L69–L70 |
| 无监督选择（`y=None`） | L72–L73 |
| `RFE` 的递归过程 | L82 |
| `RFECV` 的折平均 argmax | L84 |
| `SelectFromModel` 的阈值与 `max_features` | L93 |
| L1 稀疏性与三个稀疏估计器 | L103 |
| `LinearSVC(l1)` + `SelectFromModel` doctest | L105–L117 |
| $C$ 与 `alpha` 的稀疏方向 | L119 |
| MDI + `SelectFromModel` 的说明 | L127 |
| `ExtraTreesClassifier` doctest | L129–L144 |
| MDI caveat 的示例链接 | L149 |
| SFS 的前向/后向 | L153–L157 |
| `Pipeline` 推荐写法 | L163–L173 |
