---
title: "scikit-learn 用户指南 3.2「Tuning the hyper-parameters of an estimator」：网格、随机与逐次减半（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - hyperparameter-tuning
  - model-selection
  - grid-search
  - randomized-search
  - successive-halving
  - reference-doc
source_path: raw/articles/sklearn/3.2. Tuning the hyper-parameters of an estimator.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 用户指南 3.2「Tuning the hyper-parameters of an estimator」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 3.2. Tuning the hyper-parameters of an estimator（scikit-learn 用户指南第 3.2 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**8 段 doctest**（16 个围栏代码块中带 `>>>` 的 8 个）、1 张外链插图、2 个 `> [!…]` callout（warning + note） |
| 原文 | <https://scikit-learn.org/stable/modules/grid_search.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/3.2. Tuning the hyper-parameters of an estimator.md`（**297 行**、25,226 字节、`wc -w` 2,384；去掉代码块后正文 **1,888 词**） |
| SHA-256 | `a7547413bb494d8d787903f457e47df2f8c1dc85ea3737806852d7412d58db5a` |
| 完整性 | **7 个 `<details>` 折叠块里 3 个整块丢失、4 个只剩代码；整个 §3.2.5「Alternatives to brute force parameter search」丢失；3 处页内链接文字被吞、2 条 halving 文献全丢**。官方正文（去代码）3,898 词 → 剪藏 1,888 词（**保留约 48%**）。全文未回填（未获授权），详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.7.0 – 1.9.1，工作版本 1.9.1**，依据见「版本判定」 |
| 插图 | 1 张，已入库 `raw/assets/sklearn-hyperparameter-tuning/`（已目视核查，读数见「图表核查」） |
| 代码 | 8 段 doctest：`enable_halving_search_cv` 的 import（L112–L116）、`HalvingGridSearchCV` + `resource='n_estimators'`（L146–L162）、`min_resources=20`（L165–L179）与 `'exhaust'`（L181–L186）、`aggressive_elimination=False`（L194–L211）与 `True`（L213–L223）、嵌套参数 `CalibratedClassifierCV`（L251–L266）、`Pipeline` 双层嵌套（L270–L280）；另有 1 段非 doctest 的 `param_grid`（L38–L43）、2 段抽样字典（L63–L66、L81–L87）、4 条 halving 公式（L127–L143） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-hyperparameter-tuning]], L99`。凡超出剪藏内容的部分（API 默认值、折叠块里的原文）都显式标注「外部核查」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，因此剪藏保持抓取时的原始字节（sha256 `a7547413…`）。丢失内容只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节不是模型族、不是变换器、不是「选择」，也不是「如何估一个分数」——它是**本库第一个「如何在一个已经定好的评估协议之上做搜索」的方法学章节**。[[sklearn-cross-validation]]（3.1 节）回答「一个配置的分数怎么估」；本节回答「有 $m$ 个配置时怎么分配预算、怎么比较、以及『最好』这个字在这张表里意味着什么」。

对本库的意义有五层：

1. **它把「调参」写成一个带预算的搜索问题**。搜索有五个部件（L22–L26）：估计器、参数空间、候选的生成/采样方法、CV 方案、**评分函数**（这一项在剪藏里被吞成「a.」，外部核查补全）。这五项的任意一项写错，产出的都不再是「最优超参」，而是**另一个问题的解**。本库此前各页的「默认值陷阱」都是这一句的特例。
2. **它给出三种候选分配策略，而不是三种 API**：穷举（`GridSearchCV`，成本 $=|\Theta|\cdot k$）、采样（`RandomizedSearchCV`，成本 $=n_{\text{iter}}\cdot k$，与维数解耦）、逐次减半（`HalvingGridSearchCV`/`HalvingRandomSearchCV`，成本 $=\sum_i n_{\text{cand},i}\cdot k$，每条候选看到的资源递增）。三者的差别不在「找得准不准」，而在**每条候选被评估几次、每次用多少资源**。
3. **它暴露了随机采样规则被正文简化的问题**。`RandomizedSearchCV` 的文档说「列表会被均匀采样」；API 页进一步说明：**当所有参数都是列表时不放回采样，`n_iter` 被限制到笛卡尔积大小**；只有包含分布时才采用放回采样。用户指南这一节没写完整规则。
4. **它把「选择偏差」从一句提醒变成两个可执行的写法**：dev/eval 切分（L286–L290）与多重指标的 `refit` 语义（L237–L245）。[[cross-validation]] 第六节列的三种乐观偏差里，本节负责回答第一种（同一份数据既选又报）在 API 层怎么落地。
5. **它第一次让「拟合次数」成为可数的一等成本**。网格的成本是 $|\Theta|\cdot k$，默认 refit 再加一次；逐次减半第一轮也要评估全部候选，之后还有存活轮次，因此它减少的是**每次拟合使用的资源**，而不是拟合次数。这条与 [[cross-validation-design-and-tuning]] 的成本口径必须一起读。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 超参的定义；为什么必须用 CV 搜；`get_params()`；搜索的五个部件；三种策略与「专业搜索」指向 §3.2.5 | L10–L32 |
| 3.2.1 Exhaustive Grid Search | `param_grid`（list of dict = 多个网格）；`GridSearchCV` 就是 estimator；2 个示例链接 | L34–L52 |
| 3.2.2 Randomized Parameter Optimization | 两个好处；`n_iter`；分布或列表；`rvs` 契约；scipy 随机状态的 warning；连续/对数均匀；Bergstra & Bengio | L54–L95 |
| 3.2.3 Successive halving | 锦标赛式淘汰；`factor`/`resource`/`min_resources`/`aggressive_elimination`；**实验性 API** + `enable_halving_search_cv` | L97–L123 |
| （折叠块）Amount of resource and number of candidates | 四条递推式：$n_{\text{res},i}$、$n_{\text{cand},i}$ | L125–L143（**散文丢失，仅存公式**） |
| （折叠块）Choosing a resource | `resource='n_estimators'` 的 doctest | L144–L162（**散文丢失**） |
| （折叠块）Exhausting the available resources | `min_resources=20` vs `'exhaust'` 的 doctest | L163–L187（**散文丢失**） |
| 3.2.3.1 Aggressive elimination | `aggressive_elimination` 的两种 `n_resources_`/`n_candidates_` | L188–L190 |
| （折叠块）Code example of aggressive elimination | 两组 doctest | L192–L223（**散文丢失**） |
| 3.2.3.2 Analyzing results with `cv_results_` | 可转 DataFrame；`iter`/`n_resources` 两列 | L225–L227（**示例表丢失**） |
| 3.2.4 Tips | — | L229 |
| 3.2.4.1 Objective metric | 默认用 `estimator.score`（分类 accuracy、回归 R²）；`scoring` 覆盖 | L231–L233 |
| 3.2.4.2 多重指标 | 列表或 dict；**`refit` 必须指定或 `False`**；halving **不支持**多指标 | L235–L245 |
| 3.2.4.3 复合估计器 | `<estimator>__<parameter>` 语法；多层嵌套 | L247–L282 |
| 3.2.4.4 开发集/评估集 | 「选参＝用标签训练网格的参数」；留出 evaluation set | L284–L290 |
| 3.2.4.5 并行 | `n_jobs=-1`；每条候选 × 每折相互独立 | L292–L294 |
| 3.2.4.6 失败鲁棒性 | 失败配置默认得分 `np.nan`；`error_score='raise'` / 数值 | L296–L297 |
| **3.2.5 Alternatives to brute force parameter search（整节丢失）** | 3.2.5.1 正则化路径（10 个 `*CV`）、3.2.5.2 信息准则（`LassoLarsIC`）、3.2.5.3 OOB（RF/ET/GB 六个类） | **剪藏无此节**（官方 1.9.1 页 244 词散文 + 2 张 API 表） |

## 关键主张

### 1. 搜索的五个部件（L20–L32）

剪藏把一次搜索拆成五项（L22–L26），最后一项被吞成「and a.」，官方原文是 **「a score function」**（外部核查）。这一句是本节的骨架：**参数空间**（搜什么）× **候选生成方法**（怎么选点）× **CV 方案**（怎么评分）× **评分函数**（评什么），再乘上估计器。[[cross-validation-design-and-tuning]] 的「五个准入问题」正是把前四项逐个落地。

紧跟着的一句是本节最实用的取舍（L32）：**少数参数影响巨大，其余可以留在默认值**，并且「读 docstring、必要时读它引的文献」。这句话解释了为什么本库几乎所有实践页都以「先判断该不该调这个参数」开头——搜索空间的形状比搜索算法更决定结果。

### 2. 三种策略是三种预算分配（L28、L56–L59）

- **穷举**：`GridSearchCV` 遍历 `param_grid` 的笛卡尔积；`param_grid` 可以是 **list of dict**，表示「多个网格」而不是一个网格（L39–L45：linear 核 4 个 `C`，rbf 核 4×2 的叉积）。
- **采样**：`RandomizedSearchCV` 用 `n_iter` 个候选，声明两个好处（L58–L59）：预算与参数个数/取值个数**解耦**；加入无用参数不会降低效率。还有一句关键但容易被略过的（L75–L77）：**连续参数必须用连续分布**，这样「加大 `n_iter` 一定细化搜索」；`loguniform(1,100)` 是 `[1,10,100]` 这一网格的连续版本。
- **逐次减半**：把「资源」（默认训练样本数，也可以是 `n_estimators` 这类数值参数）按 `factor` 递增分配给存活候选（L99–L108）。`factor=3` 「通常就够用」（L108）。本节明确了三条纪律：资源增幅要大到能跨过统计显著（L101–L102 的 note）；上一轮的排名决定下一轮谁能拿到更多资源（L104）；这套估计器**仍是实验性的**，要 `from sklearn.experimental import enable_halving_search_cv`（L110–L116）。

### 3. 逐次减半的算术（L125–L143）

剪藏保留了四条递推式：

$$n_{\text{res},i} = \text{factor}^{i}\cdot \text{min\_resources},\qquad n_{\text{res},i+1}=n_{\text{res},i}\cdot\text{factor}$$

$$n_{\text{cand},i} = n_{\text{cand}} \parallel \text{factor}^{i},\qquad n_{\text{cand},i+1}=n_{\text{cand},i}\parallel\text{factor}$$

（`//` 整除在剪藏的代码块里写作 `//`，本页改记为 $\parallel$ 以示「向下整除」。）

**原文的散文丢了，公式还在**——因此「什么时候停」「为什么最后一轮要 $\le$ `factor` 个候选」「`n_resources_i` 为什么同时是 `factor` 与 `min_resources` 的倍数」这三条解释**不能引剪藏**，只能引官方 1.9.1 页（见「转换损失」）。`min_resources='exhaust'` 的目标是最后一轮用满资源，实际资源序列应按递推式计算，不能假设第一轮就是总资源除以候选数。

### 4. `refit` 是「选择」与「建模型」之间唯一的开关（L237–L245、L247–L282）

节内对 `refit` 只有两句（L241）：多重指标时必须给**指标名字符串**（作为 `best_params_` 与 `best_estimator_` 的依据），不想 refit 就 `refit=False`，「留成默认值 `None` 会报错」。官方 1.9.1 签名里 `refit` 的默认值实际是 **`True`**，不是 `None`；多重指标下必须显式选择指标或 callable。`refit` 也可以是 **callable**（1.5+，外部核查；剪藏未提），下游属性是否存在应按所用形式检查，不能假设与字符串形式完全相同。

`<estimator>__<parameter>` 的嵌套语法见 L249–L282：`CalibratedClassifierCV(estimator=...)` 的 `estimator__max_depth` 与 `Pipeline` 的 `select__k` × `model__estimator__max_depth` 都是来源示例。注意剪藏用的是 `estimator=`（1.2 起，`base_estimator` 已弃用）——**这条说明剪藏不早于 1.2**，与后面的版本判定一致。

### 5. 失败、`nan` 与「全失败」是三种不同的结局（L296–L297）

剪藏只有一句：失败的配置默认得 `np.nan`，可用 `error_score='raise'` 或数值改写。应区分三种情形：

| 情形 | 当前记录 |
| --- | --- |
| 部分候选的某些折失败 | 按来源记 `nan` 并发出警告，其余候选继续 |
| 全部候选的全部折都失败 | 具体报错与搜索器行为待在 `uv` 目标版本环境中复核 |
| 选中后 **refit** 阶段失败 | 官方 1.9.1 API 页明确说明 `error_score` 不影响 refit，refit 阶段会抛错 |

因此「搜索跑完」与「最终模型可训练」是两个不同的检查点。

## 版本判定

**结论：`1.7.0 – 1.9.1`，工作版本 1.9.1。证据中等偏强。**

| 证据 | 内容 | 判定 |
| --- | --- | --- |
| ① 句子探针（非判别性） | 剪藏 50 条 ≥14 词散文句，在官方 1.6/1.7/1.8/1.9 四版页的命中数是 **45/45/45/45**；5 条未命中项逐条核为 Markdown 侧伪影（`\[1, 10…\]` 的转义、`<estimator>` 与 `<` 粘连、`**experimental**` 的粗体标记） | 只说明「散文与四版都一致」，**不能定版** |
| ② 1.6→1.7 的三处措辞（**判别性**） | 1.6：`…uninformative). An alternative scoring function…`；1.7 起：`…uninformative), see Which scoring function should I use? for some guidance.`。剪藏 **L233 是 1.7 写法** → 排除 1.6。另两处（`adds a certain amount`、`candidate parameters`）落在丢失的折叠块里，不可用 | **≥ 1.7.0** |
| ③ 1.7→1.8 diff | 正文逐句 diff **为空**（338 vs 338 句） | 无法区分 1.7 / 1.8 |
| ④ 1.8→1.9 diff | 只有 2 行：`LassoCV(*[, eps, n_alphas, ...])` → `LassoCV(*[, eps, alphas, ...])`，位于**已丢失的 §3.2.5** | 无法区分 1.8 / 1.9 |
| ⑤ 抓取日页面 | `/stable/modules/grid_search.html` 与 `/1.9/modules/grid_search.html` **字节级相同**（137,995 B，sha256 `eef80a6e…`），页标题 `scikit-learn 1.9.1 documentation` | 工作版本 = **1.9.1** |

因此：**内容可判定的范围是 1.7.0–1.9.1**；1.7→1.8 与 1.8→1.9 的唯一差异分别落在「无差异」与「已丢失的小节」上，所以细粒度只能靠抓取日的 `/stable/` 指向。

## 转换损失

**按去代码正文词数计，这是本库目前损失比例第二高的 sklearn 剪藏**（仅次于 3.4 Metrics 的约 92%；2.3 Clustering 的约 24% 属另一种结构性损失口径）：官方正文（去代码）**3,898 词** → 剪藏去代码 **1,888 词**，保留约 **48%**。按块计（块内词数含代码）：

| 丢失单元 | 官方词数 | 剪藏在剪藏里的残留 | 丢的到底是什么 |
| --- | --- | --- | --- |
| `<details>` *Advanced examples* | 126 | **无** | 4 个进阶示例链接（`refit=callable` 的「平衡复杂度与 CV 分」、`GridSearchCV` 输出的统计比较、多指标示例）——**剪藏里 `refit=callable` 完全不存在** |
| `<details>` *Choosing min_resources and the number of candidates* | 513 | **无** | 本节最核心的一段散文：$n$ 个样本、`min_resources=10`、`factor=2` 时最多 7 轮；候选太少会浪费资源、太多则最后一轮退化成普通搜索；`HalvingRandomSearchCV` 默认把候选数定成「最后一轮用满资源」；**「资源少时难区分好坏 → 用大 `min_resources`」**；最后一轮资源必须是 `min_resources` 与 `factor` 的公倍数 |
| `<details>` *Amount of resource and number of candidates at each iteration* | 386 | **只剩 4 条公式**（L127–L143） | 全部解释：三轮的取舍、`min_resources=3`/`factor=2`/70 候选的示例表、「**进程在第一个只评 `factor` 个候选的轮次停止**」、「最后一个 `n_resources_i` 是 `factor` 与 `min_resources` 的公倍数」、「`n_resources_` 属性在哪」 |
| `<details>` *Choosing a resource* | 187 | **只剩 doctest**（L146–L162） | 「资源默认是样本数」「**不能拿参数网格里的参数当资源**」两句关键约束 |
| `<details>` *Exhausting the available resources* | 339 | **只剩两组 doctest**（L163–L187） | 「默认 `min_resources='exhaust'`」「`HalvingRandomSearchCV` 有 `min_resources='exhaust'` 与 `n_candidates='exhaust'` 两条路且互斥」「耗尽资源通常给出更好的最终候选，且略贵」 |
| `<details>` *Code example of aggressive elimination* | 267 | **只剩两组 doctest**（L192–L223） | 「理想情况最后一轮恰好 `factor` 个候选」「资源相对候选太少时最后一轮会剩多于 `factor` 个」「`aggressive_elimination` 用 `min_resources` 反复淘汰」 |
| `<details>` *Example of a (truncated) output dataframe* | 236 | **无** | `cv_results_` 的 20 行示例表 + 「每行 = 一个候选 × 一轮」「`iter` 与 `n_resources` 两列的含义」——**`cv_results_` 在 halving 里是「候选 × 轮次」的长表**，这条结构信息在剪藏里只剩一句断言（L227） |
| **§3.2.5 全节** | 401（散文 244 + 2 张 API 表） | **无** | 3.2.5.1 正则化路径（`ElasticNetCV`/`LarsCV`/`LassoCV`/`LassoLarsCV`/`LogisticRegressionCV`/`MultiTask*CV`/`OrthogonalMatchingPursuitCV`/`RidgeCV`/`RidgeClassifierCV` 共 10 个）、3.2.5.2 AIC/BIC（`LassoLarsIC`）、3.2.5.3 OOB（`RandomForest*`/`ExtraTrees*`/`GradientBoosting*` 共 6 个）。**引言 L30 的「outlined in ___」指向的就是这一节**，链接文字一丢，整节的入口也没了 |
| halving 的 References | — | **无** | `[1] K. Jamieson, A. Talwalkar, Non-stochastic Best Arm Identification and Hyperparameter Optimization, 2016`；`[2] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar, Hyperband, JMLR 18, 2018`。剪藏 L99 的引用标记 `[1] [2]` 也一并被吞 |
| 3 处页内链接文字 | — | 变成病句 | ① L26 `and a.` ← `and a score function`；② L30 `we detail applicable to these approaches` ← `we detail **best practices** applicable…`；③ L30 `outlined in.` ← `outlined in Alternatives to brute force parameter search` |

反向核对：剪藏 50 条散文句里 5 条在官方页找不到，全部是格式伪影（见「版本判定」①），**没有「剪藏多出官方没有的正文」**。公式噪声：`\[1, 10, 100, 1000\]`（转义反斜杠留在正文里，L45）、`n_candidates // (factor ** i)` 的 markdown 代码块（L136）。

## 外部核查

以下取自官方 1.9.1 API 页（不在剪藏内）：

| 对象 | 1.9.1 签名（关键默认值加粗） |
| --- | --- |
| `GridSearchCV` | `(estimator, param_grid, *, scoring=None, n_jobs=None, **refit=True**, **cv=None**, verbose=0, pre_dispatch='2*n_jobs', **error_score=nan**, **return_train_score=False**)` |
| `RandomizedSearchCV` | `(estimator, param_distributions, *, **n_iter=10**, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)` |
| `HalvingGridSearchCV` | `(estimator, param_grid, *, **factor=3**, **resource='n_samples'**, **max_resources='auto'**, **min_resources='exhaust'**, **aggressive_elimination=False**, **cv=5**, scoring=None, refit=True, error_score=nan, **return_train_score=True**, random_state=None, n_jobs=None, verbose=0)` |
| `HalvingRandomSearchCV` | `(estimator, param_distributions, *, **n_candidates='exhaust'**, factor=3, resource='n_samples', max_resources='auto', **min_resources='smallest'**, aggressive_elimination=False, cv=5, scoring=None, refit=True, error_score=nan, return_train_score=True, random_state=None, n_jobs=None, verbose=0)` |

三条与剪藏直接冲突的默认值：

1. **两个 halving 的 `return_train_score` 默认是 `True`**，而 `GridSearchCV`/`RandomizedSearchCV` 是 `False`（后者 0.21 起改的）。所以「换到 halving 之后 `cv_results_` 多出训练分列」是默认值，不是 bug。
2. **两个 halving 的 `cv` 默认是 `5`，不是 `None`**。`None` 与 `5` 在分类器上恰好都落到 `StratifiedKFold(5)`，所以现象一致，但语义不同（`None` 还要经 `check_cv` 判断估计器类型）。
3. **`GridSearchCV` 没有 `random_state`**（`n_jobs` 之外唯一与随机性相关的入口是 `cv=` 与估计器自身）。

另有两条 API 页写明、剪藏未写的规则：

- `RandomizedSearchCV`：**「若所有参数都以列表给出则不放回采样；只要有一个是分布就放回采样」**，且「强烈建议连续参数用连续分布」。
- `error_score`：**「不影响 refit 步骤，refit 阶段永远抛错」**。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 图表核查

官方页只有 **1 张图**，已入库 `raw/assets/sklearn-hyperparameter-tuning/sphx_glr_plot_successive_halving_iterations_001.png`（640×480、42,847 B、sha256 `ef6b7b981f2fd00e…`，2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取，HTTP 200）。

目视读数：

- 标题 `Scores of candidates over iterations`；纵轴 `mean test score`，刻度 0.55–0.90；横轴 5 个刻度，文字分别是 `iter=0 / n_samples=20 / n_candidates=20`、`iter=1 / n_samples=40 / n_candidates=10`、`iter=2 / n_samples=80 / n_candidates=5`、`iter=3 / n_samples=160 / n_candidates=3`、`iter=4 / n_samples=320 / n_candidates=2`。即 `factor=2`、`min_resources=20`、20 个候选、5 轮 —— 与 L127–L143 的算术一致。
- 图中约 6 条折线，**每条在被淘汰的那一轮终止**：两条（≈0.72 与 ≈0.795 的端点）止于 `iter=1`，一条在 `iter=2` 回落到 ≈0.82 后终止，只有两条到达 `iter=4`。
- **排名会翻转**：`iter=0` 时最高的一条是浅蓝色（≈0.80），到 `iter=1` 被一条粉色线（≈0.875）反超；粉色线在 `iter=2` 回落到 ≈0.862，浅蓝线升到 ≈0.845 并在 `iter=3` 到 ≈0.865；两条在 `iter=4` 收敛到 ≈0.885（相差不到 0.005，图上几乎重合）。另有两条线从 ≈0.70 起、`iter=1` 都到 ≈0.845（灰色与绿色，彼此难以区分）。
- **对正文的意义**：L104 说「存活者是那些在所有轮次里都稳定排在前列的候选」，而这张图自己就显示「小资源下的第一名不是最终第一名」。因此 L101–L102 那条 note（资源增幅要足以跨过统计显著）不是客套话：排名翻转正是「资源不足时分数不可比」的另一面（**本库综合**，图本身不给出显著性判断）。

## 分歧（文档 vs 实现 / 文档内部）

1. **`refit` 的默认值**：L241 说「留成默认值 `None` 会报错」，官方 1.9.1 签名是 `refit=True`。**多重指标必须显式指定，默认值描述为旧**。
2. **`n_iter` 的语义**：L61 说预算就是「采样次数」，L58 说「预算可与参数个数无关」。API 页写明：**全列表时是不放回采样，`n_iter` 被限制到笛卡尔积大小**；包含分布时才放回采样。**用户指南这一节完全没写完整规则**。
3. **「加入无用参数不降低效率」（L59）**：只在参数是**分布**时成立；若它们以列表形式出现，无用参数照样乘进候选集，效率按笛卡尔积下降。
4. **`error_score` 的「默认记 `nan`」（L297）**：部分失败才记 `nan`；**全部失败抛 `ValueError`**；**refit 阶段永远抛错**。三档行为剪藏只说了第一档。
5. **halving 不支持多重指标（L245）**：不是静默忽略，而是参数校验直接 `InvalidParameterError`。
6. **`min_resources` 的默认值不在剪藏里**：`'exhaust'` 只出现在 L181–L186 的 doctest 里（作为显式参数），而它其实是 `HalvingGridSearchCV` 的**默认值**——原文那句「For HalvingGridSearchCV, by default, `min_resources` is set to `'exhaust'`」在丢失的折叠块里。
7. **「不能拿参数网格里的参数当资源」缺失**：这条约束在丢失的 *Choosing a resource* 折叠块里，剪藏只剩 doctest，所以「`resource` 与 `param_grid` 冲突会怎样」在剪藏里查不到。
8. **L72–L73 的 scipy warning 是十年前的**：讲的是 `scipy < 0.16`（2015 年）无法指定随机状态。当前 SciPy 1.13.1 上这条约束早已不存在，但剪藏把它当成读者需要知道的注意事项。
9. **L82 的 `loguniform` 导入路径需要目标环境复核**：文档使用 `sklearn.utils.fixes.loguniform`，可移植写法是 `scipy.stats.loguniform`；不再保留旧环境的 `ImportError` 作为当前版本结论。
10. **L233 的「默认是 accuracy_score / r2_score」**：实现外部核查为 `_PassthroughScorer`（直接调 `estimator.score`），**不是**把 `accuracy_score` 包一层。对分类器/回归器的数值通常相同，但「默认评分器是哪个函数」取决于估计器的 `score` 方法（例如 `OneClassSVM` 没有 `score`，必须显式给 `scoring`）。
11. **L104「存活者在所有轮次都排在前列」与官方配图自相矛盾**（图中第一名在 `iter=1` 被反超）。这句话只能读成「总体上稳定」，不能读成「每轮都是第一」。
12. **`cv_results_`「与 `GridSearchCV` 类似，只是多些信息」（L227）**：halving 的结果是**长表**（每行 = 一个候选 × 一轮，同一候选可出现多次），并增加 `iter`/`n_resources` 等列。对下游代码这是结构差异，不是「多几列」。

## 局限

- **§3.2.5 整节缺失**，所以本库现在**不能**引用「正则化路径 / 信息准则 / OOB 是网格搜索的三种替代」。本页只能记录它们的存在（外部核查：10 个 `*CV` 估计器、`LassoLarsIC`、6 个 OOB 类），具体机制需另找来源（[[regularization-and-sparsity]] 与 [[leaderboard-vs-cv]] 各有部分内容，但都不是从这一节来的）。
- **halving 的「为什么」（`min_resources` 怎么选、什么时候最后一轮会退化、`aggressive_elimination` 的淘汰算法）全在丢失的折叠块里**，本页只能引用官方 1.9.1 页的递推式，不能引用剪藏的解释。
- **本节完全没有「搜索空间怎么选」的方法论**：L32 只有一句「少数参数影响大」。参数重要性的排序、先验，以及平票时的报告约定都不在文档里。
- **本节没有任何「搜索成本」的定量说明**：`n_jobs`、`pre_dispatch` 一句话（L294），没有拟合次数或复杂度公式；本页只保留按候选数与折数推导的成本口径。
- **运行时行为尚未在 `uv` 目标版本环境中复核**：1.7→1.9 期间 halving 的实验性 API 是否有 breaking change、`refit=callable` 与 `best_score_` 的关系是否改过，均未验证。
- **未读 Bergstra & Bengio 2012、Jamieson & Talwalkar 2016、Li et al. Hyperband 2018**（后两条在剪藏里连条目都没有）。因此「随机搜索优于网格」的原始论证与「逐次减半的最优性界」**本页不转述**，只记录文档的两条断言（L58–L59、L99–L108）。

## 待解问题

1. `RandomizedSearchCV` 的「全列表不放回」是哪个版本引入的？需要核对 `whats_new` 才能定版本边界。
2. 1.9.1 上 `sklearn.utils.fixes.loguniform` 是否还导出？需在 `uv` 目标版本环境中核对；可移植写法是 `scipy.stats.loguniform`。
3. `refit=callable` 时 `best_score_` 缺失，是有意设计（「回调自己决定谁是最优，分数没有唯一含义」）还是实现遗漏？1.9.1 未验证。
4. 「部分失败记 `nan`、全部失败抛 `ValueError`」这条不对称的分界是 1.0 的 `_warn_or_raise_about_fit_failures` 引入的，但其**阈值语义**（是「所有候选」还是「所有折」）需要读源码与目标环境共同确认。
5. 逐次减半在**哪些参数**上真正优于随机搜索？本库只测了 180 候选的 RF 网格（2.85× 加速、`best_score_` 低 0.0019）。需要在小样本、高维、昂贵模型（SVM/GBDT）上重测，才能给出「什么时候值得换 halving」。
6. `factor` 的选值（文档说 3 通常够）与「资源增幅要跨过统计显著」之间没有定量关系。能否用折间 std 反推 `factor`？这直接关系到 [[cross-validation]] 里那条「折间 std ≠ 标准误」的缺口。
7. `HalvingGridSearchCV` 与 `GroupKFold`/`TimeSeriesSplit` 组合时的语义：资源取「样本数」在分组数据上意味着「组的子集」还是「组的随机子集样本」？文档未说，本页未测。
8. 搜索空间里的**平票顺序与评分器舍入**是否应一起报告？本库尚无约定。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 超参定义、`get_params()` | L10–L18 |
| 推荐的 CV 搜索、搜索五部件 | L12、L20–L26 |
| 三种策略总述、`enable_halving_search_cv` 预告 | L28 |
| 断裂句（best practices / Alternatives 链接丢失） | L30 |
| 「少数参数影响大」 | L32 |
| `param_grid` 与多网格语法 | L36–L45 |
| `GridSearchCV` 是 estimator | L47 |
| 示例链接（含丢失的 Advanced examples） | L49–L52 |
| 随机搜索的两个好处 | L56–L59 |
| `n_iter`、分布/列表写法 | L61–L68 |
| `rvs` 契约、scipy 随机状态 warning | L70–L73 |
| 连续分布 / `n_iter` 细化 / `loguniform` | L75–L87 |
| Bergstra & Bengio | L93–L95 |
| 逐次减半的定义与 `factor` | L97–L108 |
| 实验性 + import | L110–L116 |
| 资源/候选数递推式 | L125–L143 |
| `resource='n_estimators'` doctest | L144–L162 |
| `min_resources` 20 / `'exhaust'` doctest | L163–L187 |
| `aggressive_elimination` | L188–L190、L192–L223 |
| `cv_results_`（halving 版） | L225–L227 |
| 默认评分器、`scoring` | L231–L233 |
| 多重指标与 `refit` | L235–L245 |
| 嵌套参数（`__` 语法） | L247–L282 |
| dev set / evaluation set | L284–L290 |
| `n_jobs` | L292–L294 |
| `error_score` | L296–L297 |

## 关联页面

- 概念：[[hyperparameter-search]]（本页的机制层提炼：三种预算分配、`best_score_` 是什么、选择偏差）、[[cross-validation]]（搜索的分数量本身怎么估）、[[leaderboard-vs-cv]]（本地分数与排行榜的错位）、[[regularization-and-sparsity]]（内置 CV 与信息准则——即丢失的 §3.2.5 的另一半）
- 实践：[[hyperparameter-search-and-tuning]]（准入问题、空间设计、预算与报告）、[[cross-validation-design-and-tuning]]（评估协议必须先冻结）、[[svm-selection-and-tuning]]、[[ensembles-selection-and-tuning]]、[[mlp-selection-and-tuning]]（三张具体的 C/预算/架构搜索表）
- 来源：[[sklearn-cross-validation]]（3.1 节，被搜索的那层）、[[kaggle-getting-started]]（人类选手的「XGB 调参五步」）
- 工具：[[scikit-learn]]（搜索器矩阵与默认值清单第 201–216 条）
