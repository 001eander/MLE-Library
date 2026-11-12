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
| 完整性 | **只保留 §3.4.1 – §3.4.3.1 四节（官方 54 个编号小节中的 4 节）**：官方正文 18,573 词里约 17,063 词（92%）不在剪藏内。已覆盖的四节里，最关键的 **`scoring or loss function` 整列、43 行字符串评分表、`Usage examples` doctest、4 条参考文献**仍然丢失。全文未回填（未获授权），详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.8.0 – 1.9.1，工作版本 1.9.1**，依据见「版本判定」 |
| 插图 | **0 张**（官方 1.9.1 页有 6 张 `<img>`），故 `raw/assets/sklearn-metrics/` 不存在 |
| 代码 | **0 段**（官方 1.9.1 页有 122 个 `highlight` 代码块、382 处 `>>>` doctest 提示符） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-metrics]], L56`。凡超出剪藏内容的部分（API 签名、未保留章节的原文）都显式标注「外部核查」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `d2e889f4…`）。丢失内容只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节是本库**第三个方法学章节**（前两个是 3.1 交叉验证、3.2 超参数搜索），但它与前两个的方向相反：3.1 问「分数怎么估」、3.2 问「$m$ 个分数怎么比」，3.4 问的是**「那个分数本身该是谁」**。它不引入新模型、不引入新切分器，而是给前两节里那个被当成既成事实的 `scoring=` 参数补上定义域。

对本库的意义有五层：

1. **它把「评分」拆成三层不同东西**：估计器的 `score` 方法（一个默认准则）、`scoring` 参数（一套**统一的「越大越好」约定**加上符号翻转）、`sklearn.metrics` 里的函数（原样返回真实量纲）。三者的返回值与符号完全不同，而 L62 的「There are 3 different APIs」是全节唯一一句把这件事说清的话——剪藏正好保住了它（L62–L66）。
2. **它给出本库第一个「分数可判定正确性」的判据**：先声明**目标泛函**（均值 / 中位数 / 分位数 / 众数），再选**严格一致（strictly consistent）**于该泛函的评分函数（L25、L33）。文档称这类评分函数是 *truth serum*，并给出「训练损失与评估指标应当用同一个」这条建议（L27）。这是本库第一次看到「指标选择」有可证伪的形式化标准，而不是经验清单。
3. **它把「预测」与「决策」明确分开**（L18、L20–L21、L31）：预测是给出分布的一个泛函，决策是把 `predict_proba` 压成一个动作。二者各有各的评分函数族，而混淆它们会得到「指标看起来很好、业务上不可用」的模型。
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
| 3.4.3.2 Callable scorers / .1 `make_scorer` / .2 自定义 scorer 对象 | 34 + 228 + 468 | 无（运行时行为待 `uv` 目标环境验证） |
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

**本库综合**：脚注 2 与脚注 3 是这张表的两个「排序不变性」声明，性质完全不同——0-1 损失不是严格一致，只看 argmax，所以它**无法在同一侧的概率预报之间排序**；而 $R^2$ 与 MSE 是**单调变换**，所以在同一数据集内的排序必然相同。前者是「损失选错，信息全丢」，后者是「换单位，排序不变」。

### 3. 三套 API 的返回值与符号不同（来源论述 + 外部核查）

L62–L66 把评分分成三处入口：

| 入口 | 返回什么 | 符号约定 |
| --- | --- | --- |
| 估计器的 `score` | 该估计器的默认准则 | 依估计器；`check_scoring(est, None)` 直接转调 `score` |
| `scoring=` 参数（字符串 / callable） | 统一为**越大越好**，distance 类指标被取负（L87） | 由 scorer 对象负责符号转换 |
| `sklearn.metrics` 函数 | 原始量纲（误差通常越小越好） | **不做 `neg_*` 翻转** |

L87 的约定句是剪藏里唯一保住「符号」这件事的地方：「All scorer objects follow the convention that **higher return values are better than lower return values**. Thus metrics which measure the distance between the model and the data … are available as ‘neg_mean_squared_error’ which return the negated value」。**本库综合**：这条约定把「符号」从「用户记忆」变成了「API 保证」，但也意味着**任何自定义 `make_scorer` 都必须自己声明 `greater_is_better`**；官方 1.9.1 API 签名的默认值是 `True`，于是把 MSE 直接包进去就等于告诉搜索器「误差越大越好」。

### 4. 预测与决策是两个评分族，混用会让指标与业务同时失真（来源论述）

L31 把决策定义成「the result of `predict_proba` is turned into a single outcome」，并指出「most of them are covered with or derived from the `metrics.confusion_matrix`」，同时把阈值调优指向另一节（官方 §3.3 *Tuning the decision threshold for class prediction*，**该节未入库**）。

**本库综合**：阈值是**模型之外**的参数，分类器的 `predict` 通常对应在默认阈值处把概率或分数转成动作。改变阈值会同时改变误报、漏报与业务成本，因此「用某个指标选模型、再用另一个阈值上线」时必须重新评估最终决策，而不能只报告概率质量。

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

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 图表核查

剪藏 **0 张图**（官方 1.9.1 页 6 张 `<img>`，全部来自示例页的静态输出），因此本次**没有**可目视核查的对象，也没有新增 `raw/assets/sklearn-metrics/`。本页因此不对任何图形趋势下结论。

## 分歧

1. **4 处引用标记被吞，句子成病句**：L25 的「… for that (target) functional, see.」（← `see [Gneiting2009]`）、L25 的「see [Wikipedia entry for Scoring rule] and, coincide with…」（← `and [Gneiting2007]`）、L33 的「see.」（← `see [Fissler2022]`）、L27 的 *truth serum* 引语后无出处（← `[Gneiting2014]`）。文献条目本身可从官方页补全，但**剪藏内没有任何一条**。
2. **表的关键列丢失**：第二列（评分函数名）整列没了（L38–L48），而 L33 的正文承诺「Here, we list … corresponding strictly consistent scoring functions」。
3. **「the table below shows all possible values」（L87）需要按版本复核**：字符串表本身未入剪藏；1.9.1 注册表与文档表的完整差异应在 `uv` 目标版本环境中重新核对。
4. **脚注 2 的「等价」与它自己的「不严格一致」是同一件事的两面**，文档没有把这个联系说破：0-1 损失只在乎 argmax，所以它**结构上**无法区分同一侧的概率预报。
5. **L139 说官方定义「adopt the adjusted version」，但 `balanced_accuracy_score` 的默认是 `adjusted=False`**——文档措辞与 API 默认值口径不一致。
6. **L56 的假想例要求「训练与评估用同一个评分」，但它自己承认两个 API 的参数名不同**（`quantile` vs `alpha`），且 `mean_pinball_loss` 的 `alpha` 默认 **0.5**——漏传就静默变成中位数损失。
7. **L87 的 `neg_*` 约定只覆盖字符串名**：自定义 `make_scorer` 必须自己声明 `greater_is_better`，而 API 默认是 `True`，误差函数若照默认包装就会被当成收益。文档在保留的四节里**从不提这个默认值**。
8. **`scoring` 的失败可能被 `error_score=np.nan` 静默**：3.1 节与 API 签名给出失败折的默认语义，而 3.4 的保留部分完全不提指标与模型输出不匹配时的后果。
9. **L64 的「Most commonly this is for classifiers」漏了 `accuracy` 一词**（链接文字被吞）：官方是「Most commonly this is **accuracy** for classifiers and the $R^2$ for regressors」——恰好漏掉的是三套 API 里最常用的那一个指标名。
10. **L23 的公式保留了旧 MathJax 展开**（`$Y = g \left(X\right)$` vs 官方 `\(Y = g(X)\)`）：属于剪藏管线的渲染噪声，但会让公式与正文的 `$…$` 记法混用，复制到代码时会带上 `\left`/`\right`。

## 局限

1. **本页只覆盖官方 3.4 节的 4/54 小节**。分类/回归/聚类指标的**定义、参数与退化行为**（`average` 的五种、`zero_division`、`multioutput`、`labels`、阈值化）在本页**没有**来源支撑，只能引 API 页或在 `uv` 目标版本环境中复核。
2. **不包含历史本机运行时实验**。`make_scorer` 的 `response_method` 默认值、`d2_*` 的注册状态与 scorer 名称集合都按 1.9.1 文档或签名记录；运行时差异待复核。
3. **没有目标环境证据说明「严格一致性」在真实模型选择上的收益**：需要比较「训练与评估使用一致评分」和混用不同泛函的结果，不能只凭定义推断实际收益。
4. **`sklearn.metrics.pairwise` 与 `metrics.cluster` 的部分语义在别处**：`pairwise` 指向未入库的 `metrics.html`（剪藏 L71 的 See also 只提到它存在），聚类指标本体在已入库的 2.3 节（见 [[clustering]]）。

## 待解问题

1. **`get_scorer_names()` 在 1.9.1 上的完整集合是什么？** 需要在 `uv` 目标版本环境中直接核对，不能用旧环境注册表推断。
2. **`d2_log_loss_score` 的注册历史**：需用版本史与目标环境确认它何时正式进入 scoring 面。
3. **`make_scorer(response_method='predict')`（1.8+ 默认）对哪些内置函数有实际影响**？需按响应方法逐类验证。
4. **多类 ROC AUC 的 `ovo` 与 `ovr_weighted` 在类别不平衡下的差别**：剪藏 L149–L163 的公式残片没有上下文，本库无法判断官方推荐哪一个。
5. **3.4.1 的「决策」与 §3.3 阈值调优如何衔接**？§3.3（`TunedThresholdClassifierCV`）未入库，而 3.4.1 只给了一个链接。
6. **「用同一个严格一致评分训练与评估」在树模型/集成上是否成立**？需要「训练损失 A + 评估损失 A」与「训练 A + 评估 B」的对照。
7. **`zero_division` 从 `'warn'` 改默认值的计划**：1.9.1 API 仍是 `'warn'`，未来版本的行为未知。
8. **`top_k_accuracy_score` 的 `k > n_classes` 行为**是否是刻意设计（丢失的 §3.4.4.3 可能有说明）。
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
