---
title: "scikit-learn 用户指南 3.4「Metrics and scoring: quantifying the quality of predictions」：只保留前 4 节、丢掉 92% 正文的一份剪藏，与它保留的那条主张（评分函数必须对准目标泛函）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - metrics
  - scoring
  - evaluation
  - proper-scoring-rule
  - decision-threshold
  - reference-doc
source_path: raw/articles/sklearn/3.4. Metrics and scoring quantifying the quality of predictions.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 用户指南 3.4「Metrics and scoring: quantifying the quality of predictions」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 3.4. Metrics and scoring: quantifying the quality of predictions（scikit-learn 用户指南第 3.4 节） |
| 作者 | 剪藏 `author` 字段是一个页内链接文字（multilabel sample），与官方页署名无关（疑为剪藏工具抓到的页内链接文字）；官方页脚署名为 scikit-learn developers、BSD License——**本库核查所得**，故本页不采用剪藏的 `author` |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**0 个代码块、0 条 doctest、0 张图、2 个 `> [!…]` callout** |
| 原文 | <https://scikit-learn.org/stable/modules/model_evaluation.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/3.4. Metrics and scoring quantifying the quality of predictions.md`（**163 行**、18,587 字节、`wc -w` 2,434 词，其中约 750 词是同一段被脚注重复两次的副本；末行无换行，停在半句 "as described in"） |
| SHA-256 | `d2e889f4c4cc58ba7aee62795e2f64c90611494248598681ae93312407f247c2` |
| 完整性 | **只保留 §3.4.1 – §3.4.3.1 四节（官方 54 个编号小节中的 4 节）**：官方正文 18,573 词里约 17,063 词（92%）不在剪藏内。已覆盖的四节里，最关键的 **`scoring or loss function` 整列、43 行字符串评分表、`Usage examples` doctest、4 条参考文献**仍然丢失。全文未回填（未获授权），详见「转换损失」；本次另做了本机 1.6.1 运行时复核（「本机实测」节） |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.8.0 – 1.9.1，工作版本 1.9.1**，依据见「版本判定」 |
| 插图 | **0 张**（官方 1.9.1 页有 6 张 `<img>`），故 `raw/assets/sklearn-metrics/` 不存在 |
| 代码 | **0 段**（官方 1.9.1 页有 122 个 `highlight` 代码块、382 处 `>>>` doctest 提示符） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-metrics]], L56`。凡超出剪藏内容的部分（API 签名、未保留章节的原文、运行时行为）都显式标注「外部核查」或「本机实测」。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `d2e889f4…`）。丢失内容只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节是本库**第三个方法学章节**（前两个是 3.1 交叉验证、3.2 超参数搜索），但它与前两个的方向相反：3.1 问「分数怎么估」、3.2 问「$m$ 个分数怎么比」，3.4 问的是**「那个分数本身该是谁」**。它不引入新模型、不引入新切分器，而是给前两节里那个被当成既成事实的 `scoring=` 参数补上定义域。

对本库的意义有五层：

1. **它把「评分」拆成三层不同东西**：估计器的 `score` 方法（一个默认准则）、`scoring` 参数（一套**统一的「越大越好」约定**加上符号翻转）、`sklearn.metrics` 里的函数（原样返回真实量纲）。三者的返回值与符号完全不同，而 L62 的「There are 3 different APIs」是全节唯一一句把这件事说清的话——剪藏正好保住了它（L62–L66）。
2. **它给出本库第一个「分数可判定正确性」的判据**：先声明**目标泛函**（均值 / 中位数 / 分位数 / 众数），再选**严格一致（strictly consistent）**于该泛函的评分函数（L25、L33）。文档称这类评分函数是 *truth serum*，并给出「训练损失与评估指标应当用同一个」这条建议（L27）。这是本库第一次看到「指标选择」有可证伪的形式化标准，而不是经验清单。
3. **它把「预测」与「决策」明确分开**（L18、L20–L21、L31）：预测是给出分布的一个泛函，决策是把 `predict_proba` 压成一个动作。二者各有各的评分函数族，而混淆它们会得到「指标看起来很好、业务上不可用」的模型（本机实测：同一个逻辑回归在四个阈值下的准确率 0.946/0.954/0.802/0.546，正类召回 0.460→0.840）。
4. **它是本库目前转换损失比例最高的一份来源**：官方 54 个编号小节里只有 4 节进来，`sklearn.metrics` 的 40 多个函数（准确率、混淆矩阵、F1、ROC/AUC、log loss、Brier、MAE/MSE/RMSE/MAPE、pinball、$D^2$、聚类四指标、`dummy` 基线）**一个都没有正文**。因此本库关于「某个具体指标怎么算」的结论**不能引这一页**，只能引这一页保留的那条方法论主张。
5. **它给 [[cross-validation]] 与 [[hyperparameter-search]] 补上缺失的输入端**：那两页反复说「分数是估计量」，但从未定义「分数是哪个函数」。本节之后，本库的评估链条才闭合——目标泛函 → 评分函数 → CV 估计 → 搜索取最大值 → 决策阈值。

## 文档结构

剪藏保留的四节（行号指向剪藏）：

| 章节 | 解决的问题 | 行号 | 官方词数 |
| --- | --- | --- | --- |
| 3.4.1 Which scoring function should I use? | 目标泛函 → 严格一致评分；预测 vs 决策；严格一致评分函数表；网络可靠性假想例 | L11–L58 | 942 |
| 3.4.2 Scoring API overview | 三套 API（`score` 方法 / `scoring` 参数 / `metrics` 函数）与 `dummy` 基线 | L60–L71 | 179 |
| 3.4.3 The scoring parameter | `scoring` 的三种写法（`None` / 字符串 / callable）与多指标 | L73–L83 | 98 |
| 3.4.3.1 String name scorers | 「越大越好」约定与 `neg_*`；**43 行字符串表（全丢）** | L85–L87（正文）+ L89 起为脚注副本 | 291 |

整段不在剪藏内的（官方 1.9.1 页的节名与词数，**外部核查**）：

| 官方节 | 词数 | 本库现有覆盖位置 |
| --- | --- | --- |
| 3.4.3.2 Callable scorers / .1 `make_scorer` / .2 自定义 scorer 对象 | 34 + 228 + 468 | 无（本页只在「本机实测」记录实测行为） |
| 3.4.3.3 Using multiple metric evaluation | 273 | [[cross-validation]]（`cross_validate` 的键） |
| 3.4.4 Classification metrics（20 个小节：accuracy、top-k、balanced accuracy、kappa、混淆矩阵、classification report、hamming、precision/recall/F、Jaccard、hinge、log loss、MCC、多标签混淆矩阵、ROC、DET、0-1、Brier、likelihood ratio、$D^2$） | ≈7,900 | 只有 [[clustering]]、[[anomaly-detection]] 里零散用到 ARI/AMI/ROC AUC |
| 3.4.5 Multilabel ranking metrics（coverage error、LRAP、ranking loss、NDCG） | ≈1,230 | 无 |
| 3.4.6 Regression metrics（$R^2$、MAE、MSE、MSLE、MAPE、MedAE、max error、explained variance、Poisson/Gamma/Tweedie deviance、pinball、$D^2$、残差可视化） | ≈4,800 | 无（[[linear-models-as-baselines]] 只用到 $R^2$/MSE 的名字） |
| 3.4.7 Clustering metrics（本节只有一句，指向 2.3.11） | 31 | [[clustering]]（已入库的 2.3 节正文） |
| 3.4.8 Dummy estimators | 477 | 无 |

## 关键主张

### 1. 评分函数的选择有判据：先定目标泛函，再选「严格一致」于它的评分（来源论述）

来源把整节的前提写成两步判断（L18–L21）：「if the scoring function is given, e.g. in a kaggle competition or in a business context, use that one. If you are free to choose, it starts by considering the ultimate goal and application」，并把应用侧拆成 **Predicting** 与 **Decision making**。

接着是本节的核心定义（L25）：「use a **strictly consistent** scoring function for that (target) functional」，其含义是「a scoring function that is aligned with *measuring the distance between predictions* `y_pred` *and the true target functional using observations of* $Y$」（同上）。来源把这类函数称作 *truth serum*，因为它保证「*“that truth telling […] is an optimal strategy in expectation”*」（同上），并直接给出使用建议：「it is best used for both: as loss function for model training and as metric/score in model evaluation and model comparison」（L27）。

**本库综合**：这三句话把「指标选择」从口味问题变成了**两层结构化问题**——先在「分布 → 一个数」这一层上声明泛函（均值/中位数/分位数/众数），再在「预测 → 分数」这一层上选与该泛函一致的损失。本库此前所有「A 比 B 高 0.3%」的读数都隐含了某个泛函，但从没写出来；这正是 [[leaderboard-vs-cv]] 里「代理指标与真实目标错位」的最上游来源。

### 2. 严格一致评分函数表：泛函 → 评分，一一对应（来源论述，但表格内容在剪藏里只剩一半）

来源用一张表把「统计泛函 / 评分或损失函数 / 响应 `y` 的类型 / 用哪个输出做预测」四列对应起来（L33–L48），并在三条脚注里补了三个等价关系：

- 分类里一热编码下的平方误差就叫 **Brier score**（L50）；
- **0-1 损失对众数只是 consistent，不是 strictly consistent**，且「equivalent to one minus the accuracy score, meaning it gives different score values but the same ranking」（L52）；
- **$R^2$ 与平方误差给出同样的排序**（L54）。

**关键**：剪藏把第二列（评分函数名）整列吞掉了（L38–L48 的第二格只剩 `<sup>1</sup>`/空），所以从剪藏里读不出「均值对应 Brier/平方误差、分位数对应 pinball」这些对应关系，只能从 L56 的假想例反推出一条（99% 分位数 → pinball loss）。完整表在「外部核查」节按官方 1.9.1 页重建。

**本库综合**：脚注 2 与脚注 3 是这张表的两个「排序不变性」声明，性质完全不同——0-1 损失不是严格一致，所以它**无法在同一侧的概率预报之间排序**（本机实测：真实 $p=0.3$ 时报告 0.2/0.3/0.4/0.5 的四个预报，log loss 0.6378/**0.6101**/0.6321/0.6931，而准确率全部 1.0000）；而 $R^2$ 与 MSE 是**单调变换**，所以在同一数据集内的排序必然相同（本机实测 5 个 `alpha` 的 MSE 376.076→2249.484 与 $R^2$ 0.9798→0.8793 完全同序）。前者是「损失选错，信息全丢」，后者是「换单位，排序不变」。

### 3. 三套 API 的返回值与符号不同（来源论述 + 本库实测）

L62–L66 把评分分成三处入口：

| 入口 | 返回什么 | 符号 | 本机实测（1.6.1） |
| --- | --- | --- | --- |
| 估计器的 `score` | 该估计器的默认准则 | 依估计器 | `SVC().score(iris)` = 0.9733（= accuracy）；`check_scoring(est, None)` 返回 `_PassthroughScorer`，直接转调 `score` |
| `scoring=` 参数（字符串 / callable） | 统一为**越大越好**，distance 类指标被取负（L87） | 已按约定翻转 | `get_scorer('neg_mean_squared_error')._sign = -1`、`_score_func = mean_squared_error`；`get_scorer('r2')._sign = +1` |
| `sklearn.metrics` 函数 | 原始量纲（误差越小越好） | **未翻转** | `mean_squared_error` 直接返回 376.08 这样的正数 |

L87 的约定句是剪藏里唯一保住「符号」这件事的地方：「All scorer objects follow the convention that **higher return values are better than lower return values**. Thus metrics which measure the distance between the model and the data … are available as ‘neg_mean_squared_error’ which return the negated value」。**本库综合**：这条约定是本节最贵的一条——它把「符号」从「用户记忆」变成了「API 保证」，但也意味着**任何自定义 `make_scorer` 都必须自己声明 `greater_is_better`**，而它的默认值是 `True`（实测），于是把 MSE 直接包进去就等于告诉搜索器「误差越大越好」。

### 4. 预测与决策是两个评分族，混用会让指标与业务同时失真（来源论述）

L31 把决策定义成「the result of `predict_proba` is turned into a single outcome」，并指出「most of them are covered with or derived from the `metrics.confusion_matrix`」，同时把阈值调优指向另一节（官方 §3.3 *Tuning the decision threshold for class prediction*，**该节未入库**）。

**本库综合 + 实测**：阈值是**模型之外**的参数，`predict` 只是「在 0.5 处切一刀」的别名。本机实测（500 样本、10% 正类的逻辑回归）：`predict` 与 `proba >= 0.5` 逐位相同；把阈值移到 0.3，准确率 0.946 → **0.954**、正类召回 0.460 → 0.540；移到 0.1，召回 0.720 但准确率掉到 0.802、报出正例 23 → 121。**同一个模型、同一份预测概率，四个决策**。这正是「用 accuracy 选模型、再用另一个阈值上线」时指标与业务脱节的机制。

### 5. 本节保住了「方法论」，丢掉了「指标本体」（本库判断）

剪藏保留的四节加起来 1,510 词（官方口径），而官方全节 18,573 词。**因此本页只能对「如何选评分函数」下结论，不能对「某个指标怎么算」下结论**——例如 balanced accuracy 的多类定义（L136–L147 的脚注残片）、多类 ROC AUC 的 `ovo`/`ovr` 加权公式（L149–L163 的公式残片）虽然留下了碎片，但**上下文全丢**，连它们各自属于哪个小节都看不出来。

## 版本判定

**结论：1.8.0 – 1.9.1，工作版本 1.9.1，证据中等偏强（一条判别性证据 + 一条排除性证据 + 一条上限证据）。**

| # | 判据 | 证据 |
| --- | --- | --- |
| 1 | **脚注 1 的措辞排除 1.6 与 1.7** | 剪藏 L50 与 L126 都是「… in case of classification **with one-hot encoded targets**.」；官方 1.6.1 与 1.7.2 页只有「… in case of classification.」，1.8.0 与 1.9.1 才有后半句（**外部核查**，逐版 grep 得到） |
| 2 | 覆盖区正文「非判别性」 | 剪藏 54 条 ≥14 词散文句在官方 1.6/1.7/1.8/1.9 四版页的命中数是 **37/37/37/37**；未命中的 17 条逐条核为 Markdown 侧伪影（`$…$` vs `\(…\)`、链接文字被吞后的病句、脚注重复块、LaTeX `\frac`） |
| 3 | 覆盖区 1.7→1.8 的唯一差异正是判据 1 | 官方 §3.4.1–§3.4.3.1 的逐句 diff：1.6→1.7 **为空**、1.8→1.9 **为空**，1.7→1.8 **只有那一条脚注**（57 句 vs 57 句） |
| 4 | 上限：1.8 与 1.9 在覆盖区不可分 | 判据 3 的 diff 为空 → 只能用抓取日的 `/stable/` 定版本 |
| 5 | 抓取日 `/stable/` 就是 1.9.1 | **外部核查（2026-11-12 抓取）**：`/stable/modules/model_evaluation.html` 与 `/1.9/` **字节级相同**（438,145 B，sha256 `7e740e14…`），页标题 `scikit-learn 1.9.1 documentation` |

**为什么不是「1.9.1，证据强」**：覆盖区在 1.8 与 1.9 之间**完全没有文本差异**，所以「≥1.8」是硬结论，「=1.9.1」只是抓取时间的旁证。凡引用本页的版本敏感结论（例如 `make_scorer` 的 `response_method` 默认值、`d2_brier_score` 是否可用）都必须写具体版本。

## 转换损失

**这是本库目前损失比例最高的一份 sklearn 剪藏**（此前最高是 2.3 Clustering 的 24% 与 3.2 超参搜索的 52%）。分四类：

### A. 整节丢失（最大的一类）

官方 1.9.1 页有 **54 个编号小节**（h2 8 + h3 39 + h4 7）（含 `3.4.4.1`–`3.4.4.20`、`3.4.6.1`–`3.4.6.12` 等），剪藏只有 **4 个**（L11、L60、L73、L85）。按「去标签后的正文词数」计：**18,573 词中约 17,063 词（92%）不在剪藏内**。丢失的最大块是 §3.4.4 分类指标（≈7,900 词）与 §3.4.6 回归指标（≈4,800 词）——也就是说，**本库目前对「分类/回归指标怎么算」没有任何直接来源**。

### B. 已保留四节内部的丢失

| 丢失项 | 证据 | 后果 |
| --- | --- | --- |
| **严格一致评分表的第二列（`scoring or loss function`）整列** | 剪藏 L38–L48 的第二格只剩 `<sup>1</sup>`/`<sup>2</sup>`/`<sup>3</sup>`/空；官方对应 Brier score、log loss、zero-one loss、squared error、Poisson deviance、Gamma deviance、Tweedie deviance、absolute error、pinball loss | 表从「四列对应关系」退化为「三列」，读者无法从剪藏得知均值/分位数对应哪个损失 |
| **3.4.3.1 的 43 行字符串评分表** | 官方表有 47 个 `<tr>`（4 个分组行 + **43 个名字行**，其中 4 行是 `‘precision’ etc.` 这样的后缀族）；剪藏 L87 之后直接转成脚注副本 | 正文 L87 的「the table below shows all possible values」在剪藏里**指向空处** |
| **`usage examples` doctest** | 官方 3.4.3.1 末尾的 `cross_val_score(..., scoring='recall_macro')` → `array([0.96, 0.96, 0.96, 0.93, 1. ])` | 剪藏 0 条 doctest（官方全页 382 个 `>>>`） |
| **「非法名字」的 Note** | 官方：「If a wrong scoring name is passed, an `InvalidParameterError` is raised. You can retrieve the names of all available scorers by calling `get_scorer_names`」 | 剪藏完全没有错误处理路径 |
| **4 条参考文献** | 剪藏 L58 与 L134 的 `References` 块**为空**；正文对应 4 处引用标记被吞（L25 的「see.」、L25 的「and, coincide」、L33 的「see.」、L39/L43 处的定义出处） | 见下方「丢失的文献」 |

丢失的文献（按官方 1.9.1 页抄录，**外部核查**，不是剪藏内容）：

- **[Gneiting2009]** T. Gneiting. *Making and Evaluating Point Forecasts*. JASA 106 (2009): 746–762.（对应 L25 的「see.」）
- **[Gneiting2007]** T. Gneiting and A. E. Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA 102 (2007): 359–378.（对应 L25 的「and, coincide」）
- **[Gneiting2014]** T. Gneiting and M. Katzfuss. *Probabilistic Forecasting*. Annual Review of Statistics and Its Application 1.1 (2014): 125–151.（对应 L25 的 *truth serum* 引语）
- **[Fissler2022]** T. Fissler, C. Lorentzen and M. Mayer. *Model Comparison and Calibration Assessment: User Guide for Consistent Scoring Functions in Machine Learning and Actuarial Practice*.（对应 L33 的「For further criteria on how to select a specific one, see.」）

### C. 三处「链接文字被吞 → 病句」

| 行 | 剪藏原文 | 官方 1.9.1 原文 |
| --- | --- | --- |
| L23 | 「there is *no deterministic* function $Y = g \left(X\right)$ of the features $X$」 | `\(Y = g(X)\)`（剪藏保留了旧 MathJax 展开形式，属渲染噪声） |
| L68 | 「Finally, **are useful** to get a baseline value of those metrics for random predictions.」 | 「Finally, **`Dummy estimators`** are useful …」（链接文字丢，主语消失） |
| L80–L81 | 「- ：common metrics can be passed via a string name.」/「- ：more complex metrics …」 | 「- **String name**：…」/「- **Callable**：…」（两个列表项标签丢） |
| L66 | 「These metrics are detailed in sections on**,,** and.」 | 三个 `sklearn.metrics` 子节链接文字（分类/回归/聚类）全丢 |
| L83 | 「See for details.」 | 「See **`Using multiple metric evaluation`** for details.」 |

### D. 反向异常：脚注被当成分节内容

剪藏 L89–L134 **原样重复了 L13–L56 的全部内容**（3.4.1 的正文、表、三条脚注、假想例），并以 `[^1]:`、`[^2]:` 标号；随后 L136–L163 又收了四条**来自别的小节**的脚注残片：

- `[^3]`（L136–L141）：一个 `> [!note]` 块，讨论 **balanced accuracy 的多类定义**有三种（官方 §3.4.4.4），并说「Our definition: , and, where … adopt the adjusted version to ensure that random predictions have a score of 0」——**三处文献链接与三个写法名全丢**，句子成病句。
- `[^4]`（L145）与 `[^undefined]`（L147）：class balanced accuracy 与 balanced accuracy 的两句定义（官方 §3.4.4.4）。
- `[^5]`（L149–L155）与 `[^6]`（L157–L163）：多类 ROC AUC 的 `ovo` 平均式与 prevalence 加权式（官方 §3.4.4.15.2），末尾停在「as described in」，无下文。

**后果**：这四条残片是剪藏里唯一涉及「具体指标定义」的内容，但它们**丢掉了所属小节、符号说明与前后文**。引用它们时必须回到官方页，不能当成本节的论述。

## 外部核查（官方 1.9.1 页，不在剪藏内）

### 1. 重建「严格一致评分函数表」的第二列

| 泛函 | 严格一致评分函数 | 响应 `y` | 预测输出 |
| --- | --- | --- | --- |
| 均值（分类） | **Brier score**（脚注 1） | multi-class | `predict_proba` |
| 均值（分类） | **log loss** | multi-class | `predict_proba` |
| 众数（分类） | **zero-one loss**（脚注 2：只 consistent，不 strictly） | multi-class | `predict`, categorical |
| 均值（回归） | **squared error**（脚注 3：$R^2$ 同序） | all reals | `predict`, all reals |
| 均值 | **Poisson deviance** | non-negative | `predict`, strictly positive |
| 均值 | **Gamma deviance** | strictly positive | `predict`, strictly positive |
| 均值 | **Tweedie deviance** | depends on `power` | `predict`, depends on `power` |
| 中位数 | **absolute error** | all reals | `predict`, all reals |
| 分位数 | **pinball loss** | all reals | `predict`, all reals |
| 众数（回归） | 「no consistent one exists」 | reals | — |

### 2. 字符串评分表的形状

- 官方 1.9.1 表 43 个名字行，分三组：**Classification 21 行**（含 `‘precision’ etc.`/`‘recall’ etc.`/`‘jaccard’ etc.` 三个后缀族）、**Clustering 9 行**、**Regression 13 行**。
- 其中 4 行标注了能力要求：「requires `predict_proba` support」（`neg_brier_score`、`neg_log_loss`、`d2_log_loss_score`、`d2_brier_score`）；`f1` 系列标注 binary/micro/macro/weighted/samples。
- `d2_log_loss_score` 在 1.6/1.7/1.8/1.9 四版表里都有；`d2_brier_score` **1.8 才进表**（1.6/1.7 的表没有）。

### 3. 相关 API 签名与版本注记

| 对象 | 1.9.1 签名（节选） | 版本注记 |
| --- | --- | --- |
| `make_scorer` | `(score_func, *, response_method='predict', greater_is_better=True, **kwargs)` | **`response_method` 默认值 1.6/1.7 是 `'default'`，1.8 起改为 `'predict'`**；该参数 1.4 新增 |
| `get_scorer` | `(scoring)` | 接受字符串或 callable |
| `check_scoring` | `(estimator=None, scoring=None, *, allow_none=False, raise_exc=True)` | — |
| `classification_report` | `(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')` | `zero_division` 默认仍为 `'warn'`（1.9.1 未变） |
| `balanced_accuracy_score` | `(y_true, y_pred, *, sample_weight=None, adjusted=False)` | 默认**不是** adjusted |

## 本机实测（scikit-learn **1.6.1**、NumPy 2.0.2、SciPy 1.13.1，18 条）

> **版本警告**：目标文档是 1.9.1，本机最高可安装 1.6.1，以下只证明机制存在。

1. **3.4.3.1 的 doctest 复现**（丢在剪藏外，按官方 1.9.1 页抄写）：`cross_val_score(SVC(random_state=0), iris, cv=5, scoring='recall_macro')` → `[0.9667, 0.9667, 0.9667, 0.9333, 1.0000]`，官方打印 `array([0.96, 0.96, 0.96, 0.93, 1. ])`（四舍五入一致）。
2. **`get_scorer_names()` = 56**，而 1.9.1 的表是 43 个名字行。两个方向的不一致：
   - **表里有、1.6.1 运行时没有（2 个）**：`d2_log_loss_score`（函数 `from sklearn.metrics import d2_log_loss_score` **可导入**，但 `get_scorer('d2_log_loss_score')` 抛 `ValueError: 'd2_log_loss_score' is not a valid scoring value`）、`d2_brier_score`（1.6.1 连导入都 `ImportError`）。
   - **运行时有、1.6–1.9 四版表里都没有（3 个）**：`matthews_corrcoef`、`positive_likelihood_ratio`、`neg_negative_likelihood_ratio`。另有 12 个名字（`precision_*`/`recall_*`/`jaccard_*` 各 4 个）由表里的「etc.」后缀行覆盖。
   - 也就是说，「the table below shows all possible values」（L87）在两个方向上都不严格成立。
3. **符号约定是 scorer 对象的私有属性**：`get_scorer('neg_mean_squared_error')` 的 `_sign = -1`、`_score_func = mean_squared_error`、`_response_method = 'predict'`；`get_scorer('r2')`/`get_scorer('accuracy')` 的 `_sign = +1`；`neg_log_loss` 的 `_response_method = 'predict_proba'`；**`roc_auc` 的 `_response_method = ('decision_function', 'predict_proba')`**（按序取第一个存在的）。
4. **`make_scorer` 的默认符号陷阱**：`make_scorer(mean_squared_error)` 的 `_sign = +1`（`greater_is_better=True` 是默认）。5 点 `alpha` 网格上它选出 **`alpha=1e4`**、`best_score_` = 6380.4、最终 MSE **6290.0**；换成 `scoring='neg_mean_squared_error'` 选出 `alpha=1e-4`、MSE **774.9**。相差 8.1 倍，且**没有任何警告**。
5. **`make_scorer` 的 `response_method` 默认值对自定义概率指标是错的**：`make_scorer(log_loss)`（1.6.1 默认 `'default'`）在 iris 三折上给出 `[nan, nan, nan]`；显式 `response_method='predict_proba'` 才给 `[0.142, 0.152, 0.156]`；显式 `'predict'` 回到 `nan`。内部原因是 `ValueError: y_true and y_pred contain different number of classes 3, 2`，被 `error_score=np.nan` 吞成警告。**1.8 起默认值变成 `'predict'`，即新版本仍然错。**
6. **字符串名字写错 → `InvalidParameterError`**（1.6.1），错误消息里直接列出全部合法字符串。剪藏缺少这条 Note。
7. **丢掉的字符串名不会抛错，只会变 `nan`**：`scoring='neg_log_loss'` 用在无 `probability=True` 的 `SVC` 上，返回 `[nan, nan, nan]` + 3 条 `UserWarning: Scoring failed … set to nan`（内部 `AttributeError: SVC has none of the following attributes: predict_proba`）。同样 `scoring='roc_auc'` 在三类 iris 上也是 `nan`（内部 `ValueError: multi_class must be in ('ovo', 'ovr')`）——**表里的 `roc_auc` 行在多类上不能直接用**。
8. **阈值是模型外的参数**：500 样本、10% 正类的逻辑回归，`predict` 与 `proba >= 0.5` 逐位相同；t = 0.50/0.30/0.10/0.05 → 准确率 **0.946/0.954/0.802/0.546**、正类召回 **0.460/0.540/0.720/0.840**、报出正例数 **23/27/121/261**。
9. **严格一致性的数值演示**（真实 $p=0.3$、20 万样本）：报告 q = 0.2/0.3/0.4/0.5 时 log loss = 0.6378/**0.6101**/0.6321/0.6931、Brier = 0.2195/**0.2096**/0.2198/0.2500，而四者的 argmax 准确率**都是 1.0000**。→ 脚注 2 的「0-1 损失只 consistent」在这里变成可读的数字。
10. **`zero_one_loss = 1 − accuracy_score`**：200 样本 3 类随机标签上 0.725 = 0.725（脚注 2 的等价性成立）。
11. **$R^2$ 与 MSE 同序**：同一数据集 5 个 `alpha` 的 MSE 376.076/376.080/376.487/413.764/2249.484 对应 $R^2$ 0.9798/0.9798/0.9798/0.9778/0.8793，排序完全一致（脚注 3 成立）。
12. **目标泛函错配会反转排名**（§3.4.1 假想例的实体版）：同一训练集（5 维、Laplace 噪声）、三个目标各训一个 `HistGradientBoostingRegressor`（`squared_error`/`absolute_error`/`quantile 0.9`），在 2,000 个测试样本上：

    | 模型 | RMSE | MAE | MedAE | $R^2$ | pinball 0.5 | pinball 0.9 |
    | --- | --- | --- | --- | --- | --- | --- |
    | squared_error（均值） | 0.7409 | 0.5397 | 0.3940 | 0.4136 | 0.2698 | 0.2727 |
    | absolute_error（中位数） | **0.7314** | **0.5232** | **0.3706** | **0.4286** | **0.2616** | 0.2605 |
    | quantile 0.9（0.9 分位数） | 1.0154 | 0.8284 | 0.7260 | **−0.1012** | 0.4142 | **0.1401** |

    分位数模型的 pinball 0.9 是 0.1401（比另两个好 1.9 倍），而在其余五个口径上全部最差（$R^2$ 为负）。**「哪个模型更好」完全由泛函决定**。
13. **`mean_pinball_loss` 的参数名是 `alpha`**（签名 `(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput='uniform_average')`），而 HGB/`QuantileRegressor` 用 `quantile`——正是 L56 那句「we apologize for the different argument names」的内容。实测 300 样本上 `mean_pinball_loss(alpha=0.99)`：HGB(quantile=.99) **1.0367** vs `QuantileRegressor(quantile=.99)` **1.9742**（同一个损失，拟合质量差 1.9 倍）。
14. **`balanced_accuracy_score` 默认 `adjusted=False`**：90/10 不平衡、正类全部错判时默认给 **0.4**，`adjusted=True` 给 **−0.2**。剪藏 L139 的残片说官方定义「adopt the adjusted version」，但 API 默认是**不**校正的版本 → 想对齐 1.9 的「机会水平 = 0」口径必须显式传参。
15. **`classification_report` 的退化情形被静默定义**：某类从未被预测时给 3 条 `UndefinedMetricWarning`，并把 precision/recall/f1 都置 **0.0**（`zero_division='warn'` 是 1.6.1 与 1.9.1 的共同默认）。
16. **callable scorer 没有签名校验**：3 参数 `(est, X, y)` 正常（iris 三折 `[0.96, 0.98, 0.94]`）；带第 4 个可选参数 `(est, X, y, extra=None)` 也被静默接受并返回常数 0.5——**写错的 scorer 不会报错，只会给常数**。
17. **多指标返回的键带前缀**：`cross_validate(..., scoring=['accuracy','f1_macro'])` 的键是 `test_accuracy`/`test_f1_macro`（另有 `fit_time`/`score_time`），没有裸 `accuracy`。
18. **`top_k_accuracy_score` 的 `k` 可以大于类别数**：3 列分数矩阵上 `k=2` 给 0.8、`k=4` 给 **1.0**（不报错）。该行为属于丢失的 §3.4.4.3，本页只记录实测现象，不作文档归属。

## 图表核查

剪藏 **0 张图**（官方 1.9.1 页 6 张 `<img>`，全部来自示例页的静态输出），因此本次**没有**可目视核查的对象，也没有新增 `raw/assets/sklearn-metrics/`。本页因此不对任何图形趋势下结论。

## 分歧

1. **4 处引用标记被吞，句子成病句**：L25 的「… for that (target) functional, see.」（← `see [Gneiting2009]`）、L25 的「see [Wikipedia entry for Scoring rule] and, coincide with…」（← `and [Gneiting2007]`）、L33 的「see.」（← `see [Fissler2022]`）、L27 的 *truth serum* 引语后无出处（← `[Gneiting2014]`）。文献条目本身可从官方页补全，但**剪藏内没有任何一条**。
2. **表的关键列丢失**：第二列（评分函数名）整列没了（L38–L48），而 L33 的正文承诺「Here, we list … corresponding strictly consistent scoring functions」。
3. **「the table below shows all possible values」（L87）与实现不一致（双向）**：见本机实测 2——表外有 3 个可用名字，表内有 2 个在 1.6.1 上不可用（其中 `d2_log_loss_score` 甚至连 1.6/1.7 的表里都列了）。
4. **脚注 2 的「等价」与它自己的「不严格一致」是同一件事的两面**，文档没有把这个联系说破：0-1 损失只在乎 argmax，所以它**结构上**无法区分同一侧的概率预报（实测四个预报的准确率全 1.0000，log loss 0.61–0.69）。
5. **L139 说官方定义「adopt the adjusted version」，但 `balanced_accuracy_score` 的默认是 `adjusted=False`**——文档措辞与默认值口径不一致（实测 0.4 vs −0.2）。
6. **L56 的假想例要求「训练与评估用同一个评分」，但它自己承认两个 API 的参数名不同**（`quantile` vs `alpha`），且 `mean_pinball_loss` 的 `alpha` 默认 **0.5**——漏传就静默变成中位数损失。
7. **L87 的 `neg_*` 约定只覆盖字符串名**：自定义 `make_scorer` 必须自己声明 `greater_is_better`，而默认是 `True`（实测第 4 条，误差被当成收益，选出最差模型）。文档在保留的四节里**从不提这个默认值**。
8. **`scoring` 的失败被 `error_score=np.nan` 静默**：`neg_log_loss` 用在无 `predict_proba` 的模型上、`roc_auc` 用在多类上，都变成 `nan` 而不是异常（实测 7）。这与 3.1 节记录的失败折语义是同一机制，但 3.4 的保留部分完全不提。
9. **L64 的「Most commonly this is for classifiers」漏了 `accuracy` 一词**（链接文字被吞）：官方是「Most commonly this is **accuracy** for classifiers and the $R^2$ for regressors」——恰好漏掉的是三套 API 里最常用的那一个指标名。
10. **L23 的公式保留了旧 MathJax 展开**（`$Y = g \left(X\right)$` vs 官方 `\(Y = g(X)\)`）：属于剪藏管线的渲染噪声，但会让公式与正文的 `$…$` 记法混用，复制到代码时会带上 `\left`/`\right`。

## 局限

1. **本页只覆盖官方 3.4 节的 4/54 小节**。分类/回归/聚类指标的**定义、参数与退化行为**（`average` 的五种、`zero_division`、`multioutput`、`labels`、阈值化）在本页**没有**来源支撑，只能引 API 页或实测。
2. **本页的「本机实测」全部在 1.6.1 上**。目标文档是 1.9.1，而 1.6.1 与 1.9.1 之间至少有：`make_scorer` 的 `response_method` 默认值（1.8 变）、`d2_brier_score`/`d2_log_loss_score` 的注册状态、`positive_likelihood_ratio` 一族的存在性。凡涉及这些的结论都必须带版本。
3. **没有运行时证据说明「严格一致性」在真实模型选择上的收益**：第 12 条只证明「泛函错配会反转排名」，没有证明「用一致评分训练 + 一致评分评估」比「用平方误差训练 + 用 MAE 评估」更好。
4. **`sklearn.metrics.pairwise` 与 `metrics.cluster` 的部分语义在别处**：`pairwise` 指向未入库的 `metrics.html`（剪藏 L71 的 See also 只提到它存在），聚类指标本体在已入库的 2.3 节（见 [[clustering]]）。

## 待解问题

1. **`get_scorer_names()` 在 1.9.1 上是否包含 `positive_likelihood_ratio` 一族？** 1.6.1 有、而 1.6–1.9 四版文档表都没有——需要在新版本上直接跑一次，才能判定「表不全」还是「名字后来被移除」。
2. **`d2_log_loss_score` 为何在 1.6.1 上「函数存在但未注册」**？这决定了它是 1.6 的注册遗漏还是 1.7 才正式加入 scoring 面。
3. **`make_scorer(response_method='predict')`（1.8+ 默认）对哪些内置函数有实际影响**？本次只测了 `log_loss` 一个。
4. **多类 ROC AUC 的 `ovo` 与 `ovr_weighted` 在类别不平衡下的差别**：剪藏 L149–L163 的公式残片没有上下文，本库无法判断官方推荐哪一个。
5. **3.4.1 的「决策」与 §3.3 阈值调优如何衔接**？§3.3（`TunedThresholdClassifierCV`）未入库，而 3.4.1 只给了一个链接。
6. **「用同一个严格一致评分训练与评估」在树模型/集成上是否成立**？本次只做了 3 个目标泛函的对照，没有做「训练损失 A + 评估损失 A」与「训练 A + 评估 B」的四格对照。
7. **`zero_division` 从 `'warn'` 改默认值的计划**：1.6.1 与 1.9.1 都是 `'warn'`，未来版本的行为未知。
8. **`top_k_accuracy_score` 的 `k > n_classes` 静默返回 1.0** 是否是刻意设计（丢失的 §3.4.4.3 可能有说明）。
9. **`classification_report` 的 `output_dict=True` 在退化类上返回 0.0 而非 `nan`**：这条选择会让「某类完全没预测到」在汇总时被平均掉，官方是否有讨论在丢失的 §3.4.4.7。
10. **本库是否需要单独一份「指标定义」来源**：如果后续不引入 §3.4.4/§3.4.6，本库将长期无法对「F1 的 macro/micro 差额」「MAPE 在 0 附近爆炸」这类常见问题给出带出处的答案。

## 行号索引（剪藏）

| 行 | 内容 |
| --- | --- |
| L1–L9 | frontmatter（`source` 指向 `/stable/`；`author` 是误抓的 wikilink；`published` 空） |
| L11 | `## 3.4.1. Which scoring function should I use?` |
| L13 | 引言：先看分数与评估指标，「inspired by statistical decision theory」 |
| L15–L16 | 两个 "*Which scoring function …*" 条目（链接目标丢失） |
| L18 | 「if the scoring function is given … use that one」；两个步骤 Predicting / Decision making |
| L23 | Predicting 段：$Y$ 是随机变量、分布 $F$、点预测 = 选一个泛函 |
| L25 | **核心**：strictly consistent scoring function；「see.」「and, coincide」两处引用丢失 |
| L27 | 同一个评分同时用于训练损失与评估/比较 |
| L29 | 回归用 `predict`、分类用 `predict_proba` |
| L31 | Decision Making 段：`predict_proba` → 单一结果；`confusion_matrix`；阈值调优链接 |
| L33 | 「List of strictly consistent scoring functions」；Fissler2022 引用丢失 |
| L35–L48 | **四列表**（第二列全空）：分类三行、回归七行 |
| L50 | 脚注 1：Brier = 一热编码下的平方误差（**1.8 起才有「one-hot」**） |
| L52 | 脚注 2：0-1 损失只 consistent、且 = 1 − accuracy |
| L54 | 脚注 3：$R^2$ 与平方误差同序 |
| L56 | 假想例：网络可靠性、99% 分位数、pinball、`quantile` vs `alpha` |
| L58 | `References`（空） |
| L60–L71 | 3.4.2：三套 API；`dummy` 基线（主语丢失）；pairwise 的 See also |
| L73–L83 | 3.4.3：`None` / 字符串 / callable 三种写法（后两个标签丢失）；多指标入口丢失 |
| L85–L87 | 3.4.3.1：字符串名 + 「越大越好」约定 + `neg_mean_squared_error` |
| L89–L134 | **3.4.1 全文的脚注副本**（`[^1]`/`[^2]`），含表与假想例的重复 |
| L136–L141 | `[^3]`：balanced accuracy 多类定义的三种写法（note 块，三处引用丢失） |
| L143 | 第二个空的 `References` |
| L145–L147 | `[^4]`/`[^undefined]`：class balanced accuracy 与 balanced accuracy 的定义句 |
| L149–L163 | `[^5]`/`[^6]`：多类 ROC AUC 的 `ovo` 与 prevalence 加权公式，末尾断在「as described in」 |

## 关联页面

- 概念：[[evaluation-metrics]]（本页保留的那条主张展开成机制页）、[[cross-validation]]（分数怎么估）、[[hyperparameter-search]]（分数怎么比）、[[leaderboard-vs-cv]]（代理指标与真实目标的错位）、[[clustering]]（少数几个已入库的具体指标）、[[regularization-and-sparsity]]（AIC/BIC 作为另一种「分数」）
- 实践：[[metric-selection-and-thresholding]]（落地流程：泛函 → 评分 → 报告 → 阈值）、[[cross-validation-design-and-tuning]]、[[hyperparameter-search-and-tuning]]、[[evaluator-design-and-reward-hacking]]（代理指标被高频优化时的防线）
- 工具：[[scikit-learn]]（`scoring` 与 `metrics` 的矩阵、默认值第 217–230 条、版本边界）
- 同批来源：[[sklearn-cross-validation]]（3.1）、[[sklearn-hyperparameter-tuning]]（3.2）
