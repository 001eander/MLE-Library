---
title: "scikit-learn 用户指南 3.1「Cross-validation: evaluating estimator performance」：三种调用式、九族切分器与置换检验（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - cross-validation
  - model-selection
  - evaluation
  - reference-doc
source_path: raw/articles/sklearn/3.1. Cross-validation evaluating estimator performance.md
source_kind: article
status: draft
verified: 2026-11-12
---

# scikit-learn 用户指南 3.1「Cross-validation: evaluating estimator performance」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 3.1. Cross-validation: evaluating estimator performance（scikit-learn 用户指南第 3.1 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**26 段 doctest**、10 张外链插图、4 个 `> [!…]` callout |
| 原文 | <https://scikit-learn.org/stable/modules/cross_validation.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/3.1. Cross-validation evaluating estimator performance.md`（**641 行**、51,416 字节、`wc -w` 5,569 词——该数含大量链接 URL，不是正文词数；末行无换行） |
| SHA-256 | `9b2541ca912de6e1be5f85cf9f25872c7f4a56fd6e22c1d806e1f3016c4d53d6` |
| 完整性 | **4 个 `<details>` 折叠块全丢（442 词）**、两处页内链接文字丢成「a.」（L226）、公式有转换噪声；26 段 doctest 全在、10 张图全在。全文未回填（未获授权），详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.7.0 – 1.9.1，工作版本 1.9.1**，依据见「版本判定」 |
| 插图 | 10 张，已入库 `raw/assets/sklearn-cross-validation/`（全部目视核查，读数见表） |
| 代码 | 26 段 doctest：`train_test_split`（L12–L40）、`cross_val_score`（L58–L87）、自定义 `cv` 迭代器（L91–L115）、held-out 变换（L117–L136）、`cross_validate` 三种 `scoring`（L151–L188）、`KFold`（L230–L256）、`RepeatedKFold`（L262–L279）、`LeaveOneOut`（L283–L296）、`LeavePOut`（L308–L325）、`ShuffleSplit`（L329–L346）、`StratifiedKFold` vs `KFold`（L371–L391）、`GroupKFold`（L427–L444）、`StratifiedGroupKFold`（L458–L477）、`LeaveOneGroupOut`（L481–L499）、`LeavePGroupsOut`（L505–L521）、`GroupShuffleSplit`（L525–L547）、用 splitter 做单次 train/test 划分（L553–L573）、`TimeSeriesSplit`（L583–L604） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-cross-validation]], L58`。凡超出剪藏内容的部分（API 默认值、折叠块里的原文）都显式标注「外部核查」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 与 [[sklearn-svm]]、[[sklearn-naive-bayes]] 不同，本次**没有取得所有者对 `raw/` 回填的授权**，因此剪藏保持抓取时的原始字节（sha256 `9b2541ca…`）。丢失内容只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节不是模型族、不是变换器、也不是「选择」类目——它是**本库第一个「如何得到一个可信分数」的方法学章节**，也是前二十个来源共同默认、却从未系统化处理的那一层：前面每一页都在讨论「某个模型在某个分数上更好」，而这一节讨论**那个分数本身是怎么估出来的、它的误差来自哪里、以及切分结构错了会怎样**。

对本库的意义有五层：

1. **它把「分数」变成一个带误差的估计量**。$k$ 折 CV 报出的不是一个数而是一个**均值 + 一列折内分数**（L52：报告值是循环中数值的平均），因此比较 CV 结果时必须同时检查折间离散程度。这条要求也是 [[leaderboard-vs-cv]] 里那条「给提升定义噪声底」的机制解释。
2. **它第一次把「切分」写成可枚举的对象**。九族 splitter 不是九种写法，而是**九条关于数据依赖结构的假设**：i.i.d.（`KFold`/`RepeatedKFold`/`LeaveOneOut`/`LeavePOut`/`ShuffleSplit`）、类比例（`StratifiedKFold`/`StratifiedShuffleSplit`）、分组独立（`GroupKFold`/`StratifiedGroupKFold`/`LeaveOneGroupOut`/`LeavePGroupsOut`/`GroupShuffleSplit`）、时间顺序（`TimeSeriesSplit`）、外部给定（`PredefinedSplit`）。选错 splitter 不是「分数噪声变大」，而是**在回答另一个问题**。
3. **它把「泄漏」落实到具体写法**。L117 那个折叠块（正文已丢，代码还在）用 `StandardScaler` 演示「先 `fit` 再 CV」与「把变换塞进 `Pipeline` 再 CV」的差别。 supervised 变换和选择都必须在训练折内拟合，否则测试折信息会参与训练过程。
4. **它区分了三种「用 CV 做事」的调用式**：`cross_val_score`（只要分数）、`cross_validate`（要时间、训练分、折内模型、折内索引）、`cross_val_predict`（要**每个样本**的折外预测）。三者的返回值不同，**语义也不同**：只有前两者报的是「泛化误差的估计」，第三个明确被文档判为「不适合作为泛化误差的度量」（L199）。本库的 Stacking/Blending 与 AutoResearch 的 OOF 特征都依赖这个区分（[[ensembles]]、[[evaluator-design-and-reward-hacking]]）。
5. **它给出本库第一个「显著性」工具**。`permutation_test_score` 把「这个分数是不是碰运气」变成一个 p 值（L625–L639），并明确说明它只能回答「模型是否稳定地优于随机猜测」，不能回答「模型好不好」。这是 [[automated-research-open-questions]] 里「代理指标被高频优化」问题的最朴素对策。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 为什么不能在同一份数据上训练与测试；hold-out 与 `train_test_split`；为什么还要第三个集合；$k$ 折的定义 | L10–L54 |
| 3.1.1 Computing cross-validated metrics | `cross_val_score`；默认 `scoring` 是估计器的 `score`；`cv=int` → `KFold`/`StratifiedKFold`；传 splitter 或自定义迭代器 | L56–L115 |
| （折叠块）Data transformation with held-out data | 预处理必须在划分内 `fit`；`Pipeline` 写法 | L117–L136（**散文丢失，仅存代码**） |
| 3.1.1.1 `cross_validate` 与多指标 | 返回 dict；`return_train_score` / `return_estimator` / `return_indices`；多指标两种写法 | L138–L188 |
| 3.1.1.2 `cross_val_predict` | 折外预测；**只允许能把每个样本恰好判一次的策略**；warning：不是泛化误差度量 | L190–L213 |
| 3.1.2 CV iterators | 生成索引的工具总述 | L215–L217 |
| 3.1.2.1 i.i.d. 族 | i.i.d. 假设的含义；note：实践里很少成立 | L219–L226 |
| 3.1.2.1.1 K-fold | $k=n$ 时等价 LOO；不受类别/分组影响 | L228–L256 |
| 3.1.2.1.2 Repeated K-Fold | `RepeatedKFold` / `RepeatedStratifiedKFold` | L258–L279 |
| 3.1.2.1.3 LOO | $n$ 个模型；代价更高；**方差大**；5/10 折优先 | L281–L304 |
| 3.1.2.1.4 LPO | $\binom{n}{p}$ 个划分；$p>1$ 时测试集重叠 | L306–L325 |
| 3.1.2.1.5 Shuffle & Split | 独立重采样；`random_state`；比 `KFold` 更细的规模控制 | L327–L352 |
| 3.1.2.2 分层族 | 稀有类导致某折无该类 → 指标未定义 / `fit` 抛错 / 输出缺列；note：分层是工程补丁且**会低估方差** | L354–L365 |
| 3.1.2.2.1 Stratified K-fold | 类比例近似保持（45/5 的 3 折对照） | L367–L399 |
| 3.1.2.2.2 Stratified Shuffle Split | 分层 + 独立重采样 | L401–L407 |
| 3.1.2.3 PredefinedSplit | 已有划分时直接喂 `test_fold`（0 / −1） | L409–L413 |
| 3.1.2.4 分组族 | 组内相关 → i.i.d. 破裂；`groups` 参数；测试折的组不得出现在训练折 | L415–L423 |
| 3.1.2.4.1 GroupKFold | 同一组不跨训练/测试；折大小可能不齐；`shuffle` 的两种平衡目标 | L425–L454 |
| 3.1.2.4.2 StratifiedGroupKFold | 同时保类比例与组的整体性；不平衡数据上的动机 | L456–L477（**实现说明折叠块丢失**） |
| 3.1.2.4.3 LeaveOneGroupOut | 每折留一个组；可用于「按实验」或「按年份」 | L479–L503 |
| 3.1.2.4.4 LeavePGroupsOut | 留 $P$ 个组的全部组合；$P>1$ 时测试集重叠 | L505–L521 |
| 3.1.2.4.5 GroupShuffleSplit | 随机留出一部分组；`LeavePGroupsOut` 组合爆炸时的抽样替代 | L523–L551 |
| 3.1.2.5 用 splitter 做单次划分 | `train_test_split` 是 `ShuffleSplit` 的包装，**不支持 `groups`** | L553–L573 |
| 3.1.2.6 时序切分 | 自相关使 i.i.d. 假定失效；必须在「未来」上验证 | L575–L577 |
| 3.1.2.6.1 TimeSeriesSplit | 前 $k$ 折训练、第 $k+1$ 折测试；训练集逐次扩张；折须等长才可比 | L579–L604 |
| 3.1.3 A note on shuffling | 数据有序时洗牌可能必要；时序数据洗牌会**高估**分数；默认不洗牌；`random_state` 语义 | L606–L617 |
| 3.1.4 CV and model selection | 指向 3.2 节（网格搜索） | L619–L621 |
| 3.1.5 Permutation test score | 零分布、p 值定义、两种高 p 值解释、只能证明「优于随机」、成本 $(n_{\text{perm}}+1)\cdot n_{\text{cv}}$ | L623–L639 |

## 关键主张

### 1. 「用一个数报 CV」在来源里就不成立（来源论述）

来源在讲完 $k$ 折流程后立刻给出报告方式：「The performance measure reported by *k*-fold cross-validation is then the average of the values computed in the loop」（L52）。示例代码把均值与标准差一起打印（L70–L74）：`0.98 accuracy with a standard deviation of 0.02`。

**本库综合**：这句话定了整节的基调——CV 的产物是**一个分布**。但它也留下一个至今未解的歧义（见「待解问题」1）：折间标准差**不是**「CV 估计量的标准误」，两个东西的换算需要 $\sqrt{k}$ 之类的假设，文档从未给出。

### 2. 三种调用式的返回值决定了它们能回答什么问题（来源论述 + 本库综合）

| 调用式 | 返回 | 文档给的用途 |
| --- | --- | --- |
| `cross_val_score` | 长度 = 折数的分数数组 | 估计性能；L64–L74 的 doctest 同时打印均值与标准差 |
| `cross_validate` | dict：`test_score` / `fit_time` / `score_time`，加 `train_score` / `estimator` / `indices` | 多指标 + 成本 + 折内对象（L140–L149）；`return_indices=True` 返回 **`indices` 字典** |
| `cross_val_predict` | 长度 = 样本数的折外预测 | 可视化、**模型融合**（L201–L202）；明确不适合当泛化误差度量（L197–L199） |

第三行与第二行的差别不是「多一个数组」：`cross_val_predict` 把每个样本的折外预测池化后再计算指标，和逐折计算再平均在非线性指标上可能不同，不能相互替代。

### 3. 「$k$ 折」不是一个算法，而是一族（来源论述）

来源把 splitter 分成五组，并在两处显式提示「不受类别或分组影响」（L246、L348）——这是本库第一次看到官方文档**用否定句定义切分器的作用域**。三组关键对照：

| 对照 | 剪藏的位置 | 含义 |
| --- | --- | --- |
| `cv=int` 在分类器上使用 `StratifiedKFold`，**不是** `KFold` | L89 | 显式传 `KFold` 与只写整数不是同一协议 |
| 45/5 不平衡数据上分层与否 | L371–L391 | 分层测试折近似保持类比例；普通 KFold 可能出现缺类测试折 |
| `KFold(n_splits=n)` ≡ `LeaveOneOut` | L230 | 每个样本轮流做一次测试集 |

第二行意味着：**在不平衡数据上用 `KFold` 可能在某些折上得到缺类训练集或测试集**，此时 ROC AUC 未定义、`fit` 可能抛「只有一个类」、多类 `predict_proba` 的列会缺（L356）。

### 4. 分层是工程补丁，而且它会**低估**方差（来源论述，罕见地自我拆台）

L361–L365 的 note 是本页最反直觉的一段，来源自己承认：

> Stratified sampling was introduced in scikit-learn to workaround the aforementioned engineering problems rather than solve a statistical one. … As a result, stratification can artificially shrink the spread of the metric measured across cross-validation iterations: the inter-fold variability does no longer reflect the uncertainty in the performance of classifiers in the presence of rare classes.（L361–L365）

**本库综合**：这把「折间标准差」的可信度又降了一级。$\sigma_{\text{fold}}$ 在分层下被人为压小，因此「按折间方差判断改动是否显著」在稀有类问题上**偏保守**（更容易把真改动判成噪声）——而 [[leaderboard-vs-cv]] 恰恰把这个量当作噪声底。

### 5. 分组切分改变的是问题，不是噪声（来源论述）

来源对分组族的动机写得很清楚：模型可能学到「高度个体化的特征」而无法泛化到新个体（L427），因此要保证**同一组不同时出现在训练与测试折**。普通 KFold 不提供这个保证，GroupKFold 则按组划分，所以折大小可能不齐（L446）。`GroupKFold(shuffle=True)` 平衡的是组数而不是样本数（L454）。

### 6. 失败可能只表现为 `nan`：`error_score=np.nan` 是默认值（外部核查）

剪藏 L356 说某折缺类会「leads to undefined classification metrics (e.g. ROC AUC), exceptions raised when attempting to call fit, or missing columns」——但**没说这些异常去哪了**。

**外部核查**：`error_score` 的官方默认值就是 `np.nan`（1.9.1 API 页，`Added in version 0.20`）。因此失败折可能只在分数数组里留下 `nan`；`np.mean` 会把结果变成 `nan`，而 `np.nanmean` 会把失败折隐藏起来。需要显式失败时应使用 `error_score='raise'` 并检查每折计数。

### 7. 洗牌默认关闭，而「该洗 / 不该洗」是两种相反的错（来源论述 + 外部核查）

L608–L615 给了互相矛盾的两条建议，且都成立：数据有序（同类标签连续）时**必须**洗牌；数据按时序排列时洗牌会**高估**分数。

三处细节：
- 外部核查显示，`KFold(shuffle=False, random_state=0)` 会抛 `ValueError`，因为不洗牌时随机状态不起作用；同类 splitter 也应显式检查这一组合。
- `cv=int` 走的是**不洗牌**的 `StratifiedKFold`（L613），因此同一份数据连续调用结果可重复；但 `train_test_split` 默认仍返回随机划分（L613）。
- `GridSearchCV` 在一次 `fit` 内对所有参数组用**同一套**洗牌（L614），所以参数间比较不受洗牌噪声污染，而跨调用比较会受。

### 8. 置换检验只回答一个问题（来源论述）

L625–L639 的定义很克制：零假设是「估计器未能利用特征与目标之间的任何统计依赖」，$p$ 值是「CV 分数**不低于**真实分数的置换所占比例」。两个后果文档都写明了：

1. 高 $p$ 值有**两种**解释——数据无依赖，或估计器没能利用依赖（欠拟合）（L627–L631）。
2. 这个检验「只看是否稳定优于随机猜测」，即使数据里只有微弱结构也容易给低 $p$ 值（L636）。

成本公式也在正文里：内部拟合 $(n_{\text{perm}} + 1) \cdot n_{\text{cv}}$ 个模型（L638）。由定义还可知，$n_{\text{perm}}$ 次置换能达到的最小 p 值是 $1/(n_{\text{perm}}+1)$；API 要求 `n_permutations >= 1`。

## 版本判定

**结论：1.7.0 – 1.9.1，工作版本 1.9.1。证据四路，强。**

1. **133 条句子探针**（剪藏散文切句、≥14 词、归一化后取前 90 字符）在官方 1.6 / 1.7 / 1.8 / 1.9 页的命中数是 **101 / 122 / 122 / 122**。未命中的 11 条逐条检查全是 Markdown 侧伪影（callout 标记、图片 alt 与下一段粘连、`$…$` 与 `\(…\)` 差异），**没有一条是正文缺失**。
2. **1.7→1.8 与 1.8→1.9 的正文逐句 diff 为空**（349 句对 349 句，无 replace/insert/delete）。因此 1.7 / 1.8 / 1.9 无法用正文区分。
3. **1.6→1.7 的 diff 里 21 条与剪藏相关，剪藏全部采用 1.7 的新写法**：`5- or 10- fold` → `5 or 10-fold`（L302、L304）、`equal number` → `an equal number`（L454）、`similar as` → `similar to`（L505）、`tests set` → `test sets`（L283）、分层族整段重写（L356–L365）、`permutation_test_score offers another way to evaluate the performance of a predictor`（L625，1.6 是 `classifiers`）、`target values` / `estimator` / `left-out data` 等替换。另：1.6 的 doctest 输出是 `array([0.96..., 1. ...])`，1.7 起是 `array([0.96, 1., ...])`，剪藏是后者。
4. **抓取日 `/stable/modules/cross_validation.html` 与 `/1.9/modules/cross_validation.html` 字节级相同**（169,672 B，sha256 `78d7587f…`），页标题 `scikit-learn 1.9.1 documentation`。

## 转换损失

**本库第三完整的一份 sklearn 剪藏**（前两名是 [[sklearn-outlier-detection]] 的 2,441/2,486 词与 [[sklearn-covariance]] 的 1,804/1,848 词）：接口与代码全在（26 段 doctest、10 张图一个不缺），丢的集中在**四个折叠块（442 词）**与**两处链接文字**。官方页正文（去掉 `<pre>` 代码）4,324 词，保留率约 **89.8%**。

| 丢失项 | 官方 1.9.1 页的原文 | 影响 |
| --- | --- | --- |
| `<details>` **Data transformation with held-out data**（186 词） | 「Just as it is important to test a predictor on data held-out from training, preprocessing (such as standardization, feature selection, etc.) and similar data transformations similarly should be learnt from a training set and applied to held-out data for prediction…A Pipeline makes it easier to compose estimators, providing this behavior under cross-validation…See Pipelines and composite estimators.」 | **剪藏只剩标题 + 两段代码**（L117–L136）。段落里唯一的论证句（为什么必须这样做）与 `Pipeline` 的解释全丢，读者只看到「两段代码给出 0.9333 与 0.977」而不知道为什么 |
| `<details>` **References**（87 词） | FAQ `neural-nets/part3/section-12.html`；Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Springer 2009；Breiman & Spector, *Submodel selection and evaluation in regression: The X-random case*, International Statistical Review 1992；Kohavi, *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*, IJCAI；Rao, Fung & Rosales, *On the Dangers of Cross-Validation. An Experimental Evaluation*, SIAM 2008；James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Springer 2013 | **整个 References 块消失（连标题都没有）**，而 Kohavi 与 Rao et al. 正是「$k$ 取 5 还是 10」「CV 有陷阱」两条结论的出处 |
| `<details>` **Implementation notes**（StratifiedGroupKFold，154 词） | 洗牌只发生在「类分布标准差相同的组之间」；算法按类频率方差从大到小贪心分配组；因此「即使完美分层可行，也可能产生不平衡划分」；组内类分布接近时应改用 `GroupKFold` | **本页最严重的一条**：这是 `StratifiedGroupKFold` 唯一的机制说明，丢了以后 L458 只剩一句「结合 StratifiedKFold 与 GroupKFold」 |
| `<details>` **References**（置换检验，15 词） | Ojala & Garriga, *Permutation Tests for Studying Classifier Performance*, JMLR 2010 | 3.1.5 整节无任何出处 |
| L226 页内链接文字 | 「it is safer to use **a time-series aware cross-validation scheme**. Similarly, … it is safer to use **a group-aware cross-validation scheme**.」 | 剪藏变成「it is safer to use **a**. Similarly … it is safer to use.」——**两个关键短语都丢了**，这句话读起来像残句，而它正是「何时不该用 i.i.d. 切分器」的判据 |

其他转换噪声：

- 公式：`\left(\right. \genfrac{}{}{0pt}{}{n}{p} \left.\right)`（L308 的 $\binom{n}{p}$）、`$k$ -fold`、`*k* -fold`、`$n - 1$` 与 `\left(k - 1\right) n / k` 的空格与括号混杂。
- 官方页 **0 个 `seealso`、3 个 admonition**（1 个 `Warning` + 2 个 `Note`），剪藏以 `> [!warning]` / `> [!note]` 保留了全部文字（L194–L199、L225–L226、L360–L365）。
- 图片 Markdown 有 `![../_images/…](https://…)` 的路径重复（L24、L54 等处），不影响内容。
- **反向核查**：官方正文里 149 条 ≥14 词的句子中有 13 条（285 词）在剪藏里找不到，逐条核为「标题与首段粘连」「`\(n-1\)` 的 LaTeX 差异」「上表四个折叠块」——**没有额外的正文缺失**。

## 外部核查（官方 1.9.1 API 页，不在剪藏内）

### 20 个对象的签名与默认值

| 对象 | 1.9.1 签名（关键项） |
| --- | --- |
| `KFold` | `KFold(n_splits=5, *, shuffle=False, random_state=None)` |
| `StratifiedKFold` | `StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)` |
| `GroupKFold` | `GroupKFold(n_splits=5, *, shuffle=False, random_state=None)`（`shuffle`/`random_state` **1.6 新增**） |
| `StratifiedGroupKFold` | `StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)`（**`shuffle` 与 `random_state` 不是 keyword-only**） |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | `(*, n_splits=5, n_repeats=10, random_state=None)` |
| `ShuffleSplit` / `StratifiedShuffleSplit` | `(n_splits=10, *, test_size=None, train_size=None, random_state=None)` |
| `GroupShuffleSplit` | `(n_splits=5, *, test_size=None, train_size=None, random_state=None)` |
| `LeaveOneOut` / `LeaveOneGroupOut` | 无参数 |
| `LeavePOut` / `LeavePGroupsOut` | `(p)` / `(n_groups)` |
| `TimeSeriesSplit` | `(n_splits=5, *, max_train_size=None, test_size=None, gap=0)`（后三项 **0.24 新增**） |
| `PredefinedSplit` | `(test_fold)` |
| `train_test_split` | `(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)`（**没有 `groups`**） |
| `cross_val_score` | `(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', error_score=nan)` |
| `cross_validate` | 同上 + `return_train_score=False, return_estimator=False, return_indices=False` |
| `cross_val_predict` | 同上 + `method='predict'` |
| `permutation_test_score` | `(estimator, X, y, *, groups=None, cv=None, n_permutations=100, n_jobs=None, random_state=0, verbose=0, scoring=None, params=None)` |

### 版本注记（官方页的 `Changed in version` 注）

- **`n_splits` 默认值 0.22 起由 3 变 5**（`KFold`/`GroupKFold`/`TimeSeriesSplit` 与 `cv=None` 的默认折数）。剪藏**完全不提默认折数**——`cross_val_score(..., cv=None)` 会跑 5 折这件事只能从 API 页得到。
- `cross_validate.return_train_score`：0.19 新增，**0.21 起默认值由 `True` 改为 `False`**。
- `cross_validate.return_estimator`：0.20 新增。`return_indices`：**1.3 新增**。
- `error_score`：0.20 新增，默认 `np.nan`。
- `params`（元数据路由）：`cross_validate`/`cross_val_score`/`cross_val_predict` 1.4 新增，`permutation_test_score` 1.6 新增；`groups` 在元数据路由开启时不能再直接传（1.4 / 1.6 的 `Changed in version`）。
- `GroupKFold.shuffle` / `GroupKFold.random_state`：**1.6 新增**。这是本节能判到 ≥1.7 之外最强的一条「功能层」旁证（剪藏正文提到 `shuffle=True` 的行为，见 L454，但该行为在 1.6 才存在）。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 图表核查

10 张图（= 官方页全部）已入库 `raw/assets/sklearn-cross-validation/`，逐张目视并记录尺寸与 SHA-256 前 16 位。

| # | 文件 | 尺寸 | sha256（前 16） | 目视读数 |
| --- | --- | --- | --- | --- |
| 1 | `grid_search_workflow.png` | 2031×1362 | `45f77a54331377c4` | 8 个灰蓝色矩形：`Parameters`、`Dataset`、`Cross-validation`、`Training data`、`Test data`、`Best parameters`、`Retrained model`、`Final evaluation`。**连线在当前渲染尺寸下不可辨**，故不描述箭头方向（剪藏 L12 只说它是「typical CV workflow in model training」的流程图） |
| 2 | `grid_search_cross_validation.png` | 885×613 | `f57377e201381e15` | 上排 `All Data` 分成 `Training data`（绿，宽）与 `Test data`（蓝）；下排 `Fold 1..5` 五个格，`Split 1..5` 逐行把一格染蓝、其余绿；右侧大括号标 `Finding Parameters`；最后一次 `Final evaluation` 用橙色标 `Test data`。**与 L47–L52 的 $k$ 折定义逐格对应** |
| 3 | `sphx_glr_plot_cv_indices_005.png` | 600×300 | `39e27221c5c7bc20` | `StratifiedGroupKFold`：4 次迭代、测试块**非连续且分散**（因为组的样本散布在索引上），`class` 行显示类 0（0–49）与类 1（50–99）两段，`group` 行 10 种颜色 |
| 4 | `sphx_glr_plot_cv_indices_006.png` | 600×300 | `56a887b1ff1a0d69` | `KFold`：4 次迭代，测试块是**连续区间**（0–25、25–50、50–75、75–100），因此每一折都切过 `class` 行的两段——**直观解释为什么 `KFold` 在有序标签上失效** |
| 5 | `sphx_glr_plot_cv_indices_007.png` | 600×300 | `8a4680c6a641d4d7` | `GroupKFold`：4 次迭代，测试块由多个不连续片段组成（同一组的样本聚在一起），折大小不等 |
| 6 | `sphx_glr_plot_cv_indices_008.png` | 600×300 | `7c9b8514fe795da2` | `ShuffleSplit`：4 次迭代，测试点是**随机散点**（约 20 个/次），四次之间无互补关系——与 `KFold` 的「划分」区别一眼可见 |
| 7 | `sphx_glr_plot_cv_indices_009.png` | 600×300 | `b2cc8ec5300c4d03` | `StratifiedKFold`：4 次迭代，每折测试块都是**多个小区间**（按类分别切），`class` 行的两段在每折里都被覆盖 |
| 8 | `sphx_glr_plot_cv_indices_011.png` | 600×300 | `2e2ab2095ccd9c04` | `GroupShuffleSplit`：4 次迭代，测试块大（约 50%）、由整组构成、**同组不裂开** |
| 9 | `sphx_glr_plot_cv_indices_012.png` | 600×300 | `1f3896e5b21e7792` | `StratifiedShuffleSplit`：4 次迭代，测试点是**随机散点但两类各占固定比例**（每个类段内均匀分布） |
| 10 | `sphx_glr_plot_cv_indices_013.png` | 600×300 | `3e57bcd142a9cd0a` | `TimeSeriesSplit`：4 次迭代，训练块从索引 0 起**单调变长**（0–20 → 0–40 → 0–60 → 0–80），测试块紧随其后、等宽（20 个/次）——L581「successive training sets are supersets」的图示 |

一致性说明：这 9 张 `plot_cv_indices_*` 图来自同一个官方示例（`plot_cv_indices.py`），因此 `class` 与 `group` 两行在 9 张图里完全相同，这是同一份合成数据的标记，不是错误。

## 分歧与待解问题

### 分歧

1. **L144 的 `return_indices=True` 被描述成「retain the training and testing indices」，而实际返回是一个 `indices` 字典**（`{'train': [...], 'test': [...]}`），不是 `train_indices`/`test_indices` 两个键。照字面写代码会 `KeyError`。
2. **L89 只说「当 `cv` 是整数时」，未说 `cv=None` 的行为**。API 页显示 `cv=None` 等价于 5 折（0.22 起；此前 3 折），且分类器用 `StratifiedKFold`。剪藏读者会以为默认是无切分。
3. **L356 说某折缺类会产生「undefined metrics / exceptions / missing columns」，但没说这些异常默认被 `error_score=np.nan` 吞掉**。剪藏读者会以为程序会崩。
4. **L302–L304 的 LOO 结论（「5 或 10 折优先」）没有给出方差依据的量纲**。LOO 的单样本分数与 K 折的折级分数不是同一种统计量，折间标准差不能直接比较。
5. **L197–L199 的 warning 与 L201–L202 的推荐并存**：「不是泛化误差度量」与「适合做模型融合」看似矛盾，实际是两个用途（诊断 + 造特征）。
6. **L226 的「it is safer to use a.」是链接文字丢失造成的病句**（官方是「a time-series aware cross-validation scheme」与「a group-aware cross-validation scheme」）。
7. **L361–L365 的自我拆台没有进入任何 API 或建议**：分层会压小折间方差，但文档在别处仍把折间方差当作噪声指标。
8. **嵌套 CV 的偏差方向与幅度依赖数据和搜索过程**：文档在 L619–L621 只把读者导向 3.2 节，没有讨论「选了参数以后再报同一个 CV 分数」的偏差，不能假设任何数据集都有固定量级的乐观偏差。
9. **L553 说 `train_test_split` 「only allows for stratified splitting (using the class labels) and cannot account for groups」**，措辞把「用标签分层」当成限制，实际 `stratify` 是可选参数、默认 `None`（不进行分层）。
10. **`StratifiedGroupKFold` 的 `shuffle`/`random_state` 不是 keyword-only**，而 `GroupKFold`/`KFold`/`StratifiedKFold` 都是。这个签名差异在 1.9.1 仍在，剪藏当然不提。

### 待解问题

1. **折间标准差与「CV 估计量的标准误」是什么关系？** 文档只给 $\bar s$ 与 $\sigma_{\text{fold}}$，没给换算。$k$ 折的折间方差本身就是 $\sigma/\sqrt{k}$ 的有偏估计（且折间不正交），这是本库所有「提升是否显著」判断的未闭合处。
2. **`RepeatedKFold` 的 $n_{\text{repeats}}$ 该取多少？** 重复降低的是划分选择的方差，不保证折间离散度变小。文档未讨论。
3. **分层与分组同时存在时该保哪个？** L454 说 `GroupKFold(shuffle=True)` 保「组数相等」，`StratifiedGroupKFold` 保「类比例」，但单类组等情形下可能退化。三者的优先序没有官方建议。
4. **`TimeSeriesSplit` 的「折须等长才可比」（L583）为什么只在时序出现？** 交叉折的折大小不等同样影响可比性，但只有这一句提到。
5. **`gap` 该设多大？** 官方只给「排除末尾 $g$ 个样本」的定义（0.24 新增），没有给「自相关长度」与 $g$ 的关系。
6. **`permutation_test_score` 的 $n_{\text{perm}}$ 与 $p$ 分辨率**：正文说「typically larger than 100」，API 只要求 ≥1。由 $1/(n_{\text{perm}}+1)$ 可知，「$p<0.01$」需要至少 100 次置换。这个算术关系文档未写明。
7. **`error_score` 该不该改成 `'raise'`？** 剪藏不提这个参数；本库倾向在 AutoResearch 里设 `'raise'`（[[evaluator-design-and-reward-hacking]]），但没有官方依据。
8. **`cross_val_predict` 的折外预测在 Stacking 里的正确用法**：文档只说「model blending」一句（L202），没有讨论「用同一 CV 做选型 + 造 OOF 特征」的双重使用偏差（见 [[ensembles]] 的 Stacking 小节）。
9. **`PredefinedSplit` 如何表达「多折」？** L411 说可以表示「several cross-validation folds」，L413 的示例只给 0/−1 两值（等于单折）。用递增整数能否得到 `KFold` 式的多折，文档未演示。
10. **`ShuffleSplit` 与 `train_test_split` 的关系**：L553 说前者是后者的包装，但 `train_test_split` 的 `shuffle=True` 与 `ShuffleSplit` 的默认 `n_splits=10` 之间没有可对照的说明。

### 局限

- 本节**不给指标**（指标见已入库的 [[sklearn-metrics]]；该剪藏只保留方法论四小节），**不给超参搜索**（搜索见 [[sklearn-hyperparameter-tuning]]），**不给数据泄漏的完整清单**（散在 [[sklearn-preprocessing]] 与各估计器页）。因此「怎么算分」与「怎么选参」都不在本节范围内。
- 本节**不提 `cross_validate` 与 `GridSearchCV` 的 `refit`**、不提 `cross_validate` 与元数据路由的组合细节。
- 本节对 `StratifiedGroupKFold` 的机制说明**全在丢失的折叠块里**，正文只剩一句动机，具体行为应引官方实现说明或在 `uv` 目标版本环境中复核。
- 本节**没有任何复杂度分析**（除了置换检验那一条成本公式），也没有「$k$ 该取 5 还是 10」的证明（只有 L304 一句「most authors and empirical evidence suggest」）。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| overfitting / hold-out / `train_test_split` | L10–L40 |
| 为什么需要 validation set；三集合的代价 | L43–L45 |
| $k$ 折定义；报告值 = 均值 | L47–L52 |
| `cross_val_score` doctest；默认 `scoring`；`cv=int` 规则 | L58–L89 |
| 传 splitter / 自定义迭代器 | L91–L115 |
| held-out 变换（**仅代码**） | L117–L136 |
| `cross_validate` 的两种差别；`return_train_score`/`return_estimator`/`return_indices`；多指标两种写法 | L138–L188 |
| `cross_val_predict`；warning；两个正当用途 | L190–L213 |
| i.i.d. 假设与 note | L219–L226 |
| `KFold` | L228–L256 |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | L258–L279 |
| `LeaveOneOut`；三个 caveat；5/10 折优先 | L281–L304 |
| `LeavePOut` | L306–L325 |
| `ShuffleSplit` | L327–L352 |
| 稀有类问题 + 分层 note | L354–L365 |
| `StratifiedKFold` 对照 doctest | L367–L399 |
| `StratifiedShuffleSplit` | L401–L407 |
| `PredefinedSplit` | L409–L413 |
| 分组族总述 + `groups` 参数 | L415–L423 |
| `GroupKFold` | L425–L454 |
| `StratifiedGroupKFold`（**实现说明丢失**） | L456–L477 |
| `LeaveOneGroupOut` | L479–L503 |
| `LeavePGroupsOut` | L505–L521 |
| `GroupShuffleSplit` | L523–L551 |
| 用 splitter 做单次划分；`train_test_split` 的限制 | L553–L573 |
| 时序数据的两段判据 | L575–L577 |
| `TimeSeriesSplit` | L579–L604 |
| A note on shuffling | L606–L617 |
| CV and model selection（指向 3.2） | L619–L621 |
| `permutation_test_score` | L623–L639 |

## 关联页面

- 概念：[[cross-validation]]（本页的机制抽象）、[[leaderboard-vs-cv]]（CV / Public LB / Private LB 三套分数）、[[regularization-and-sparsity]]（信息准则是另一种代理指标）、[[ensembles]]（Stacking 依赖折外预测）
- 实践：[[cross-validation-design-and-tuning]]（选 splitter、嵌套、泄漏、坑清单）、[[competition-ml-workflow]]、[[feature-selection-workflow]]（选择器必须折内拟合）、[[evaluator-design-and-reward-hacking]]
- 工具：[[scikit-learn]]（`model_selection` 矩阵与默认值）
- 后续来源：[[sklearn-hyperparameter-tuning]]（超参搜索）与 [[sklearn-metrics]]（指标）均已入库，本页不重复其内容。
