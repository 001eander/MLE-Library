---
title: "Kaggle 入门指南：竞赛型 MLE 的通用流程与一次完整赛题复盘（知乎专栏剪藏）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - kaggle
  - competition-ml
  - cross-validation
  - ensembling
  - feature-engineering
source_path: raw/articles/Kaggle 入门指南.md
source_kind: article
published: 2016
status: draft
---

# Kaggle 入门指南：竞赛型 MLE 的通用流程与一次完整赛题复盘（2016 年初稿，2019-06 最后编辑）

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Kaggle 入门指南（剪藏未保留原标题之外的副标题） |
| 作者 | 剪藏未署名。正文多处链到 `dnc1994.com` 与 `github.com/dnc1994/Kaggle-Playground`，Kaggle 用户名应为 **dnc1994**（**本库推断**，非来源陈述） |
| 形态 | 知乎专栏 [Human Learning](https://www.zhihu.com/column/AbsTF) 的长文剪藏；**不是论文、不是基准报告** |
| 本地文件 | `raw/articles/Kaggle 入门指南.md`（375 行） |
| SHA-256 | `74549fbf97e8528e485cba3dc83b60e3c9d4549b5abf2de2e1d5df452f1aad7e` |
| 写作时间 | 自述「2016 年 4 月参加完第一次 Kaggle 比赛并拿到前 5% 的成绩后写的总结」，并称「一年过去了」（L5），即正文在 2017 年前后有过修订 |
| 平台最后编辑 | 2019-06-02 17:53（L374） |
| 热度 | 2321 赞同（L3、L376） |
| 代码 | 无独立仓库；正文片段为 IPython Notebook 截图与粘贴代码，其它工作见 `github.com/dnc1994/Kaggle-Playground`（L244 链接） |
| 插图 | 4 张（L53、L63、L73、L192），均托管于 `pic*.zhimg.com`；已抓取入库 `raw/assets/kaggle-guide/`，**逐张目视核查**（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用给到章节名与行号。作者身份、原始发布日与文章英文版的存在均无法从剪藏独立核实。

## 概要

文章分两部分。第一部分（I. General Approach）给出一次竞赛的通用流程：EDA → 预处理 → 特征工程 → 模型选择与调参 → 交叉验证 → Ensemble → Pipeline（L7–L240）。第二部分（II. Home Depot Search Relevance）是作者自己那场比赛的完整复盘，末尾附 Takeaways、Issues Raised 与 Beginner Tips（L242–L364）。第三部分是 3 条参考文献（L366–L370）。

来源的自我定位是「方法至今仍然有效」（L5），但从工程角度它是一份**2016–2017 年的快照**：模型清单、调参直觉与全部代码片段都绑定在那个时代的库版本上，其中若干 API 已被移除（见「代码与 API 时效性」）。它的耐久部分是**流程顺序、验证设计与 Stacking 的实现细节**，而不是具体的模型选择。

对本库的意义：现有来源（[[aide]]、[[alphaevolve]]、[[autoresearch]]、[[hyra-1-0]]、[[hyra-2-0]]）都在谈**智能体**怎么做 MLE，而 AIDE 与 MLE-Bench 的评估标尺恰恰是「超过多少比例的人类 Kaggle 选手」（[[aide]], p. 5–7）。本来源补上了标尺另一侧的描述：人类在竞赛里实际怎么工作、分数为什么可信、以及什么情况下连人类也判断不了自己的改进是否真实。

## 关键主张

### 1. 流程是有顺序的，特征优先于调参与集成

来源给出一句被反复引用的说法：「有人总结 Kaggle 比赛是『Feature 为主，调参和 Ensemble 为辅』，我觉得很有道理」（L69），并在 Takeaways 里重申「**Feature 为王**。我花在 Feature 上的时间还是太少」（L339）。

- 顺序：EDA（L11）→ 预处理（L37）→ 特征工程（L67）→ 模型选择（L93）→ 调参（L112）→ CV（L164）→ Ensemble（L172）→ Pipeline（L230）。
- 「**生成尽量多的 Feature，相信 Model 能够挑出最有用的 Feature**」（L77）；但先做一轮 Feature Selection 仍有三个好处：训练更快、去掉线性相关的特征、以及**把重要特征之间做运算后作为新特征**（L79–L81）。
- 特征有效性的最低验证方式是画图直观看（L71）。

**但来源自己给出了反例**：比赛最后一批新增的特征让模型「反而比之前的要差，而且还差不少」，作者归因为特征间相互作用，并把它列为未解决的开放问题（L290、L350）。也就是说「特征越多越好」与「特征多反而变差」在同一篇文章里并存，来源未调和（见 [[competition-ml-workflow]] 的张力记录）。

### 2. Feature Selection 的实用做法：看树模型的 Feature Importance

- 「Feature Selection 最实用的方法也就是看 Random Forest 训练完以后得到的 **Feature Importance**」（L83）。理由之一是树的数量增加可以增强对噪声数据的鲁棒性。
- 在**脱敏**比赛里看 Feature Importance 尤其重要，可以避免把时间花在琢磨一个不重要变量的含义上（L85）。
- 高基数类别变量的编码策略：按 Importance 或取值频率保留前 95% 的取值各建哑变量，其余归入「其他」类（L91）。
- 哑变量（one-hot）只适用于取值数不多的变量；成百上千类时不适用（L65）。

### 3. 模型清单是 2016 年的表格赛事共识

| 层次 | 来源列出的模型 | 位置 |
| --- | --- | --- |
| 最常用（基于树） | Gradient Boosting、Random Forest、Extra Randomized Trees | L97–L99 |
| 性能稍逊但适合作 Base Model | SVM、Linear Regression、Logistic Regression、Neural Networks | L103–L106 |

来源同时注明「跟图像有关的比赛中神经网络的重要性还是不能小觑」（L101），并把 XGBoost 单列：Gradient Boosting 的性能 + 高效实现，使「几乎每场比赛的获奖者都会用 Xgboost 作为最终 Model 的重要组成部分」（L110）。作者在这次的比赛中实测到 XGB「同样的数据它只需要不到一半的时间就能跑完」（L294）。

**时效性**：这段清单描述的是表格型竞赛的 2016 年格局。本库其它来源的 Weco-Kaggle 已覆盖表格、图像分类与时间序列三类任务（[[aide]], p. 5），不再是单一表格场景；本文不覆盖这两类的工作流。条目应按时点引用，不宜当作现状。

### 4. XGBoost 调参：六个关键参数与五步流程

参数（L128–L133）：`eta`（步长）、`num_round`（迭代数）、`subsample`（行采样）、`colsample_bytree`（列采样，类同 `max_features`）、`max_depth`（深度上限）、`early_stopping_rounds`（早停）。其中两条附有理由：`eta` 越小训练越慢；「Gradient Boosting 如果不对深度加以限制，最终是会 Overfit 的」，这与 Random Forest 不同。

五步流程（L135–L141）：

1. 划出验证集；
2. `eta` 先设高（如 0.1），`num_round` 设 300–500；
3. 用 Grid Search 搜其它参数；
4. 逐步降低 `eta` 找最佳值；
5. 以验证集为 watchlist 重新训练，观察每轮迭代的验证分数，得到最佳 `early_stopping_rounds`。

另给两条经验值：Random Forest 的 `max_features` 「一般在 Feature 数量的平方根附近得到最佳结果」（L124）；有随机性的模型务必记录 `seed`/`random_state` 以便复现（L162）。

### 5. 交叉验证：CV 与 LB 冲突时信 CV，前提被后文悄悄补上

来源的核心启发式很直接：「当你改进了 Feature 或是 Model 得到了一个更高的 CV 结果，提交之后得到的 LB 结果却变差了，**一般认为这时应该相信 CV 的结果**」（L166）。理由：Public LB 会因各种原因不可靠，而且在第二节里被具体化为「数据分布不 i.i.d.」（L35、L254）。

配套的量级建议：分布随机均衡时 5-Fold 够用，不放心可提到 10-Fold，但折数越多越慢（L168）；「靠谱的 CV 方法是 Case By Case 的」（L170）。

作者在复盘里实际做的两件事比这条启发式更有信息量：

- 因为 LB 与 CV 反复背离，他把 5-Fold 提到 10-Fold，「以 CV 为标准继续前进」（L298）；
- 在 Lessons Learned 里补上真正的做法：**针对数据不 i.i.d.，手工构造「产品 ID 不重叠」与「重叠」两种切分，并按与实际训练/测试集相同的比例混合，以逼近 LB 的得分分布**（L331）。

即：**「信 CV」的前提是 CV 切分复刻了真实划分的结构**。来源没有把这个前提写进第 4.2 节，是在第二节末尾补上的（本库归纳，见 [[leaderboard-vs-cv]]）。

### 6. Ensemble：diversity 与 accuracy 的权衡，以及 5-Fold Stacking 的实现

四种方法的来源定义（L178–L181）：

| 方法 | 来源描述 | 训练数据利用 |
| --- | --- | --- |
| Bagging | 不同随机子集训练各 Base Model，等权投票（=Random Forest 原理） | 见下文 |
| Boosting | 迭代训练，按上一轮的错误调整样本权重（=Gradient Boosting 原理） | — |
| Blending | 用**不相交**的数据训练不同 Base Model，输出取（加权）平均 | 「对训练数据利用少了」 |
| Stacking | 折外预测作为第二层特征（L188–L228） | 优于 Blending |

来源给出的两个成功要素（L185–L186）：**Base Model 之间相关性要尽可能小**（这是为什么非树模型性能不佳也要放进 Ensemble——多样性越大最终偏差越低）；**性能不能差距太大**，且这是 Trade-off：「即使这种情况下 Ensemble 还是能大幅提高成绩」。

5-Fold Stacking 的机制（L190–L194，代码 L198–L226）：

1. 训练集切 5 份；每个 Base Model 训练 5 次，每次在 4 份上训练、在留出那份上预测；
2. 5 次预测拼成 `#训练行 × #Base Model` 的矩阵，作为第二层 Model 的训练数据（标签用真实 `y`）；
3. 每个 Base Model 同时对**全部测试数据**预测 5 次，对 5 次取平均，得到与第二层训练矩阵同形状的测试矩阵；
4. 第二层 Model 在测试矩阵上预测即得最终输出。

作者注明获奖队伍会用三到五层、以及把不同预处理/特征工程的模型组合起来，但「对于新手来说，稳稳当当地实现一个正确的 5-Fold Stacking 已经足够了」（L228）。

来源对「Ensemble 同时降低 Bias 和 Variance」给了一篇 Springer 章节作为依据（L174），**本库未读该文献**，故该论断按来源论述记录，不当作已验证结论。

### 7. 结果：单模型前 10%、Stacking 前 5%

| 口径 | 分数 | 名次 | 位置 |
| --- | --- | --- | --- |
| 最好的单个 Model（Private LB） | 0.46378 | 174 | L317 |
| 最终 Stacking（Private LB） | 0.45849 | 98 | L317 |

作者据此表述为「单靠 Feature 和调参进到了前 10%，而 Stacking 使我进入了前 5%」（L317）。**参赛队总数未给出**，百分比无法复算（见「待解问题」）。

Ensemble 构成（L302–L309）：RandomForest、ExtraTrees、GradientBoosting、XGB 四个 Base Model，第二层仍是 XGB。放弃更大多样性的原因是实测：四者两两相关「最低的一对也有 0.9」；想加入的 `gblinear` XGB 的 RMSE 比其他模型高 0.02（作者换算为「LB 上几百名的差距」），SVR 训练太慢（L311）。

另一个细节值得单独记：开始 Stacking 之后，「我的 CV 和 LB 成绩的提高就是完全同步的了」（L313）。这是全文唯一一处「代理指标与目标重新对齐」的观察，但只有单次比赛、无对照。

### 8. 20 个种子的加权平均：一次自觉可疑的 +0.0004

最后两天作者用 20 个不同随机种子各生成一次 Ensemble，再取加权平均（L315）。给出的机制解释很具体：

> 按我实现 Stacking 的方式，我在训练 Base Model 时只用了 80% 的训练数据，而训练第二层的 Model 时用了 100% 的数据……每次更改随机种子可以确保每次用的是不同的 80%，这样在多次训练取平均以后就相当于逼近了使用 100% 数据的效果。（L315）

收益约 **0.0004**，作者自己的结论是「也很难受说是真的有效还是随机性了」（L315）。

这条对评估器设计有跨时代的价值：在没有 agent 的 2017 年，人类选手已经遇到「改进幅度与随机性同量级、无法判定真假」的问题，并诚实记录了下来（**综合**，与 [[autoresearch]] 进度图把 `random seed 42→137` 计为一次保留是同一现象，见 [[evaluator-design-and-reward-hacking]]）。

### 9. Pipeline 与「自动化记录」是中等价值、高必要性的工程

来源把自动化分三级（L234–L236）：模块化 Feature Transform、自动化 Grid Search、自动化 Ensemble。作者对新手优先级的判断是：第一项意义不大（特征数量人脑管得过来）、第三项到比赛末尾做几次即可，**第二项最有意义**——「手工记录每个 Model 的表现不仅浪费时间而且容易产生混乱」（L238）。

参考文献之一是 Crowdflower 冠军 Chenglong Chen 公开的 Pipeline 仓库，作者评价「非常具有参考和借鉴意义」，但也承认看懂并抽出框架「还是比较困难」（L240）。

## 图表核查

4 张图均于 2026-11-11 抓取（需 `Referer: https://zhuanlan.zhihu.com/` 与浏览器 UA，否则返回 403/567），存入 `raw/assets/kaggle-guide/`，**逐张目视核查**。

| 图 | 文件 | 尺寸 / SHA-256（前 8 位） | 位置 | 核查结论 |
| --- | --- | --- | --- | --- |
| 1 离群点 | `fig1-outlier.png` | 606×892 / `e6ab46ad…` | L53 | IPython Notebook 截图（`In [5]`/`In [7]`），非正式图表 |
| 2 哑变量 | `fig2-dummy-variables.png` | 1054×718 / `edf94a5d…` | L63 | Notebook 截图，`pd.get_dummies` 前后对照 |
| 3 特征直方/箱线 | `fig3-feature-example.png` | 381×271 / `b15b9b66…` | L73 | seaborn 风格箱线图，无标题无图注 |
| 4 Stacking 示意 | `fig4-stacking.jpg` | 1297×326 / `93678d12…` | L192 | 手绘流程图（JPEG，实际为 JPG 转存） |

**图 1（离群点，L51–L55）**：上图 x 轴 −10~70、y 轴 −20~120，可见 y≈0 的密集带与右上角一个 y≈117、x≈63 的点；下方一行注释「We can see that there are some erroneous y values. Drop them.」；`In [5]` 的代码是 `trainDF = trainDF[abs(trainDF['Y'])<100]`；下图重绘后 y 收窄到 [−0.15, 0.15]，x 轴也随之缩到 [−3, 2]。**目视可确认：大 x 值点同时也具有大 y 值**，故单靠 `abs(Y)<100` 过滤即可解释 x 轴范围的变化。这与正文「经过 Scaling 的坐标数据，右上角存在一些离群点，去除以后分布比较正常」一致。

**图 2（哑变量，L57–L65）**：`In [31]` 显示原列 `DayOfWeek` 的三行样例（262468 Saturday / 217465 Sunday / 223264 Friday）；`In [32]` 为 `pd.get_dummies(trainDF['DayOfWeek'], prefix='DayOfWeek')` 后 `drop` 原列并 `join`；`In [34]` 的 `columns` 输出确认生成 **7 个列**：Friday、Monday、Saturday、Sunday、Thursday、Tuesday、Wednesday；`In [33]` 的表头可见 Friday/Saturday/Sunday/Thursday 四列，行内取值 1/0 与样例一一对应。正文「将原本有 7 种可能取值的 Weekdays 变量转换成 7 个 Dummy Variables」与图一致。

**补注（综合，来源未提）**：7 类保留全部 7 个哑变量会产生完全共线的一组列（dummy variable trap）。对树模型无影响，但在线性/逻辑回归上需要丢弃一个参考类；来源的语境是树模型为主的赛场，故未讨论。

**图 3（特征示例，L71–L73）**：纵轴 `edit_dist_st_pt_avg`（0–14），横轴 `relevance` 的 7 个取值：1.0、1.33、1.67、2.0、2.33、2.67、3.0。各档中位数随 relevance 上升而下降（约 2.0 → 1.0），离群点上界约 8（低 relevance 档最远到 ≈10）。这直观支撑了正文「当一个变量从直觉上来说对所要完成的目标有帮助，就可以将其作为 Feature」（L71）以及 L286 中打星号的 Edit Distance 特征（该特征的方向与该图一致：相关度越高，编辑距离越小）。

**交叉印证（综合）**：横轴的 7 档恰好是三位评分者各给 1–3 分取均值后的全部可能取值，与 L246「相关度是由 3 个人类打分取平均得到」一致。

**图 4（Stacking 示意，L192）**：左上是 Training Data 的 5 行 × Model 1–5 的网格，每列一个 `Predict`（橙色）其余为 `Learn`（蓝色），橙色格沿对角线错开一格；箭头从各模型的 Predict 格指向右侧 `New Feature` 块；`New Feature` 块（Predictions）即 `Model 6` 的训练数据（Learn）。下半部分是 Test Data 的 5 个绿色 `Predict` 块，经 `Average` 汇总为 `Predictions` 并标注 **Variant A**；右侧另有一支箭头从单个绿色 `Predict` 块直接指向 `Predictions`（Test Data），标注 **Variant B**。

与正文对照：图示结构与 L190–L194 的文字描述（5 折、折外预测组装第二层、测试侧 5 次取平均）一致。**但图文有一处不一致**：图中把测试侧标出两种方案（Variant A 取 5 次平均 / Variant B 单模型直接预测），正文与随后的代码只实现了 Variant A（`S_test[:, i] = S_test_i.mean(1)`，L222）。来源未解释 Variant B 的适用条件。

## 代码与 API 时效性

正文代码片段写于 2016–2017 年，**多处 API 已被移除**。下表为现代等价物（**综合**，非来源陈述；未实测）：

| 来源写法 | 位置 | 现状与等价写法 |
| --- | --- | --- |
| `sklearn.grid_search.GridSearchCV` | L120 | 已移除；`from sklearn.model_selection import GridSearchCV` |
| `sklearn.cross_validation.train_test_split` | L143 | 已移除；`from sklearn.model_selection import train_test_split` |
| `KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016)` | L208 | `n_folds` 已弃用；`KFold(n_splits=..., shuffle=True, random_state=2016)` |
| `xgb` 的 `objective='reg:linear'` | L149 | 已废弃并会告警；回归用 `reg:squarederror` |
| `xgb` 的 `'silent': 0` | L155 | 已弃用；`verbosity` |
| 「在 Windows 上安装 Xgboost 很容易遇到问题」（L110） | — | 已过时：当前有官方 wheel，无需自行编译 |
| `grid_search.GridSearchCV(..., scoring=RMSE)` 中自定义 RMSE | L120 | 现代写 `scoring='neg_root_mean_squared_error'`（sklearn 内部约定「越大越好」，需取负号） |

同理，正文提到的 `iPython Notebook`（L47）现已并入 Jupyter。

## 局限

- **单例证据**：全部实战结论来自一次比赛（Home Depot，2016 年 4 月前已结束）与作者个人的第一次参赛经历。没有任何对照实验、多次运行或方差报告。
- **数字缺分母**：174 → 98 的名次换算成「前 10% → 前 5%」但未给参赛队数（L317）；+0.0004 未给方差（L315）。
- **「Ensemble 同时降低 Bias 和 Variance」超出可核查范围**：来源引用一篇 Springer 章节（L174），本库未读原文，不知道其条件（是否限定 Base Model 类型、是否要求独立性假设）。
- **合并训练集与测试集做预处理**（L265）在当时被解释为省去重复代码，但按现代口径，把 `fit` 暴露给测试数据会让 CV 分数的语义变化（无监督变换下的弱泄漏）；来源未讨论这一点。
- **GitHub 链接与截图未复核**：`github.com/dnc1994/Kaggle-Playground` 的 notebook 本次未下载；4 张图是转存副本，可能与原文当前版本不同。
- **时间绑定**：模型清单、Windows 安装经验、sklearn/xgboost API 均属该时点；L5 称方法「依然行之有效」，这一自我评价已超出可核查范围。

## 待解问题

1. **「前 10% → 前 5%」的分母是多少？** 若 174 名为前 10%、98 名为前 5%，反推参赛队约 1,740–1,960 支；来源未给数字，无法确认是否为同一榜单时点（L317）。
2. **+0.0004 是真实收益还是噪声？** 作者自己也说不清（L315）。20 个种子的加权平均在标签上属于「同一配置的重复采样」，缺少同分布下的标准差。
3. **特征变多为什么会变差，且 Grid Search 救不回来？** 来源只给了归因（特征相互作用）与一个未实施的方案（把不同特征版本的模型 Ensemble 起来，L290、L330）。这是否等价于「模型容量对特征噪声的敏感度」问题？需要对照实验。
4. **「信 CV」在什么条件下会错？** 来源给出「CV 切分要复刻真实划分结构」这一条（L331），但没有说明如何量化「结构相同」（ID 重叠比例？时间顺序？分组？），也没有失败案例的对照。
5. **Stacking 的 Variant B（测试侧用单模型预测）何时更优？** 图上标了，正文没写（见「图表核查」图 4）。
6. **Base Model 相关性 ≥0.9 时 Ensemble 的收益来自哪里？** 来源的机制解释是多样性，但实测的四个模型却高度相关（L311），仍带来 174→98 的名次变化；两种说法需要进一步辨析。

## 行号索引

| 主题 | 位置 |
| --- | --- |
| 自述写作背景与「方法仍有效」的判断 | L5 |
| EDA 与可视化清单 | L11–L29 |
| Public LB 与 Local CV 可能背离 | L35 |
| 预处理清单（含浮点精度噪声） | L37–L47 |
| Outlier 与 Dummy Variables 两例 | L51–L65 |
| Feature Engineering 与 Feature Selection | L67–L85 |
| 高基数变量的编码策略 | L87–L91 |
| 模型清单（树模型 / 非树模型）与 XGBoost | L93–L110 |
| Grid Search 代码与 `max_features` 经验值 | L116–L124 |
| XGBoost 参数与五步调参 | L126–L141 |
| 调参代码（含已废弃参数） | L142–L160 |
| 记录随机种子 | L162 |
| CV：信 CV、5/10-Fold、Case by Case | L164–L170 |
| Ensemble 四种方法、两个成功要素 | L172–L186 |
| 5-Fold Stacking 原理与代码 | L188–L228 |
| Pipeline 三级自动化 | L230–L240 |
| Home Depot 任务与评分口径 | L242–L248 |
| 赛题 EDA 的四点启发 | L252–L257 |
| 预处理四步（含合并 train/test） | L263–L266 |
| 特征清单（打星号者为最后一批） | L268–L290 |
| 模型与从 5-Fold 提到 10-Fold 的决定 | L292–L298 |
| Ensemble 构成、相关性 0.9、CV/LB 同步 | L300–L313 |
| 20 种子加权平均与 +0.0004 | L315 |
| 单模型 0.46378（174 名）/ Stacking 0.45849（98 名） | L317 |
| Lessons Learned（含 CV 切分复刻、特征交互） | L319–L333 |
| Takeaways / Issues Raised / Beginner Tips | L337–L364 |
| 参考文献 3 条 | L366–L370 |

## 关联页面

- 概念：[[leaderboard-vs-cv]]（排行榜与本地验证的张力，本来源的核心贡献）
- 实践：[[competition-ml-workflow]]（七段流程与时效性对照）、[[evaluator-design-and-reward-hacking]]（第六类：验证切分与真实划分结构不一致）
- 研究：[[automated-research-open-questions]]（新增「Kaggle 入门指南新增的数字疑点」与对应假设、风险、待补资料）
- 对照来源：[[aide]]（以人类 Kaggle 分位为标尺）、[[autoresearch]]（同样遇到噪声与种子问题）
