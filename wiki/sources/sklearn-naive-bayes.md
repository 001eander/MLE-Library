---
title: "scikit-learn 用户指南 1.9「Naive Bayes」：条件独立、五种类条件假设与平滑（stable 文档剪藏，含所有者授权的 raw 回填）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - naive-bayes
  - generative-model
  - reference-doc
source_path: raw/articles/sklearn/1.9. Naive Bayes.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.9「Naive Bayes」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Naive Bayes（scikit-learn 用户指南第 1.9 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；含 1 段 doctest（`GaussianNB` + iris）与 7 个公式块；**无任何插图** |
| 原文 | <https://scikit-learn.org/stable/modules/naive_bayes.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/1.9. Naive Bayes.md`（**回填后** 171 行、13,172 字节；**抓取时** 115 行、11,112 字节） |
| SHA-256（回填后） | `85c4050b201718a7877050532715deebc2019d21f59c05a32e737244c8131fa8` |
| SHA-256（抓取时原始字节） | `5e0e5299a7d0447f039e522c7b8c4a3b3f7615bf11ab6da786ddb36e9c84bd66` |
| 完整性 | **抓取时缺 5 个折叠块（318 词 ≈ 21.4% 正文）**，经所有者明确授权已回填；回填后正文与官方 1.9.1 逐句可对，仅剩公式排版噪声（详见「转换损失、回填与外部核查」） |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x（推断，证据弱）**：判据只有两个连字符级字符串，见「版本判定」 |
| 插图 | 无 |
| 代码 | 1 段 doctest：`GaussianNB` 在 iris 上 50/50 切分，输出「Number of mislabeled points out of a total 75 points : 4」（L59–L70，回填后行号） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-naive-bayes]], L42`。凡超出剪藏内容的部分（API 参数、版本差异、`partial_fit` 的实现层事实）都在句子里显式标注「外部核查」或「本库」。

> [!warning] 本文件被修改过（本库第二例）
> `raw/` 的规则是不可变，但本次由所有者明确授权回填。抓取时的原始字节 sha256 为 `5e0e5299…`（115 行）；**当前文件不再是抓取时的原始字节**（`85c4050b…`，171 行）。回填内容全部来自官方 1.9.1 页的同一批折叠块，未改写任何原文句子。**本页及所有引用本来源的页面，行号一律按回填后的新文件计。**

## 概要

这是用户指南里结构最"教科书"的一节：先用 6 段把朴素贝叶斯写成**贝叶斯定理 + 条件独立 + MAP** 的统一框架（L10–L42），再声明「不同变体的差别只在 $P(x_i\mid y)$ 的分布假设」（L36），最后用 6 个小节给出五个估计器与外存拟合。

对本库的意义有三层：

1. **它补上了「第三个零成本基线族」**。[[linear-models-as-baselines]] 里已有判别式线性模型与生成式 LDA/QDA；NB 是同一"闭式 + 快 + 少样本"象限里的第三种，但**只估一维边缘分布**，因此参数代价比 LDA 还低一个量级（见 [[naive-bayes]] 与 [[gaussian-discriminant-analysis]] 的参数表）。
2. **它是本库第一个明确警告「输出概率不可信」的来源**（L42）。此前所有页面讨论的都是"分数好不好用"，这一句第一次把「分类器」与「概率估计量」拆成两件事——直接关系到 [[leaderboard-vs-cv]] 的指标选择和 [[evaluator-design-and-reward-hacking]] 里"把 `predict_proba` 当分数或阈值"的用法。
3. **它把"平滑"变成与 `alpha`、`C`、`shrinkage` 并列的第四个正则化旋钮**（$\alpha$，L84），并顺手暴露了 `GaussianNB` 的方差下限（外部核查）——见 [[regularization-and-sparsity]] 新增小节。

## 文档结构

| 章节 | 解决的问题 | 行号（回填后） |
| --- | --- | --- |
| 引言 | 贝叶斯定理 → 条件独立 → MAP 判定规则；快、少样本、缓解维度灾难；**「bad estimator」** | L10–L42 |
| References | Zhang (2004) *The optimality of Naive Bayes*——「为什么错假设还能 work」的指定答案 | L44–L47 |
| 1.9.1 Gaussian Naive Bayes | 高斯似然；$\mu_y,\sigma_y$ 用 MLE；iris doctest | L49–L70 |
| 1.9.2 Multinomial Naive Bayes | 计数向量/tf-idf；平滑后的相对频率；Laplace 与 Lidstone | L72–L84 |
| 1.9.3 Complement Naive Bayes | 补集统计、不平衡数据；**Weights calculation**（三步权重 + $\arg\min$ 判定） | L86–L112 |
| References | Rennie et al. (2003)，CNB 的原始论文 | L114–L118 |
| 1.9.4 Bernoulli Naive Bayes | 二值特征；显式惩罚"未出现"；短文档可能更好 | L120–L132 |
| References | Manning/Raghavan/Schütze、McCallum & Nigam、Metsis et al. | L134–L143 |
| 1.9.5 Categorical Naive Bayes | 逐特征类别分布；**Probability calculation**（类别型平滑）；ordinal 编码要求 | L145–L161 |
| 1.9.6 Out-of-core fitting | `partial_fit`、首次须传 `classes`、chunk 尽量大、样本权重 | L163–L172 |

## 关键主张

### 1. 统一框架：条件独立把联合密度压成一维乘积（来源论述）

贝叶斯定理（L13）→ 条件独立假设 $P(x_i\mid y,x_1..x_{i-1},x_{i+1}..x_n)=P(x_i\mid y)$（L19）→ 分类规则（L31）：

$$\hat{y}=\arg\max_y P(y)\prod_{i=1}^{n}P(x_i\mid y)$$

且 $P(y)$ 与 $P(x_i\mid y)$ 用 **MAP 估计**，其中 $P(y)$ 就是训练集里类 $y$ 的相对频率（L34）。

来源对这一结构的三个"好处"断言（L38–L40）：小样本即可估参数（"They require a small amount of training data"）；比复杂方法**快得多**；类条件分布解耦成**独立的一维估计**，从而缓解维度灾难。

**本库综合**：第三条是这一族真正的定义性特征——参数个数从 $O(d^2)$（LDA 的共享协方差）或 $O(Kd^2)$（QDA）降到 $O(Kd)$，代价是**所有特征间相关性被写成 0**。因此"NB 什么时候能用"这个问题等价于"特征间相关性是否会改变类别的相对排序"。

### 2. 最重的一句：分类器好、估计量坏（来源论述 + 本库推论）

> On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously.（L42）

这是本节唯一的"使用警告"，也是全库第一次出现「**分类准确率**与**概率估计质量**是两个可分离的属性」。

**本库推论（标注为综合，不是来源论述）**：
1. 任何把 `predict_proba` 当**分数**做阈值、排序合并、期望收益计算的流程，都不能直接套用 NB 的输出；
2. 但它**不妨碍** NB 作为 `predict` 级别的基线，也不妨碍用 AUC 一类只依赖排序的指标；
3. 来源没有给校准方法，也没有量化"多坏"——**待补资料**，见「待解问题」。

### 3. 五个变体的差别只在 $P(x_i\mid y)$（来源论述）

来源明说"differ mainly by the assumptions they make regarding the distribution of $P(x_i \mid y)$"（L36）。逐项：

| 变体 | 假设 | 来源给出的定量内容 | 位置 |
| --- | --- | --- | --- |
| `GaussianNB` | 逐特征高斯 | $P(x_i\mid y)=\frac{1}{\sqrt{2\pi\sigma_y^2}}\exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)$，$\mu_y,\sigma_y$ **用 MLE** | L51–L57 |
| `MultinomialNB` | 多项分布（词频/计数；tf-idf 在实践中也好用） | $\hat\theta_{yi}=\frac{N_{yi}+\alpha}{N_y+\alpha n}$；$N_{yi}$ 是类 $y$ 下特征 $i$ 的计数和 | L74–L82 |
| `ComplementNB` | 用**类补集**的统计量 | $\hat\theta_{ci}=\frac{\alpha_i+\sum_{j:y_j\ne c}d_{ij}}{\alpha+\sum_{j:y_j\ne c}\sum_k d_{kj}}$ → $w_{ci}=\log\hat\theta_{ci}$ → 按 $\sum_j\lvert w_{cj}\rvert$ 归一化；判定 $\hat c=\arg\min_c\sum_i t_iw_{ci}$ | L88、L90–L112 |
| `BernoulliNB` | 逐特征二值（伯努利） | $P(x_i\mid y)=P(x_i=1\mid y)x_i+(1-P(x_i=1\mid y))(1-x_i)$ | L122–L128 |
| `CategoricalNB` | 逐特征独立类别分布 | $P(x_i=t\mid y=c;\alpha)=\frac{N_{tic}+\alpha}{N_c+\alpha n_i}$ | L147–L159 |

### 4. Multinomial：平滑 $\alpha$ 的两种名字（来源论述）

来源给出的语义（L84）：$\alpha\ge 0$ 用于**照顾训练集中没出现的特征并防止零概率**；$\alpha=1$ 叫 **Laplace smoothing**，$\alpha<1$ 叫 **Lidstone smoothing**。

**本库综合**：分母里的 $\alpha n$ 说明这个平滑**按特征数缩放**，因此 $\alpha$ 不是"加几个伪计数"，而是"每个特征加 $\alpha$ 个伪计数"——跨词表大小的 $\alpha$ 不可直接比较，这一点来源没写。

### 5. Complement：为不平衡数据设计（来源论述，含未能核验的实证主张）

- 动机：CNB 是 MNB 的改造，"particularly suited for imbalanced data sets"（L88）。
- 机制：用**每个类的补集**统计量算权重（L88、L90–L112）。
- 实证主张："parameter estimates for CNB are more stable than those for MNB"，且"regularly outperforms MNB (often by a considerable margin)"（L88）——转述自 Rennie et al. 2003（L114–L118），**本库未读该论文**，按来源论述记录。
- 一个来源没解释的点：$\hat\theta$ 取 log 后**按该行的 L1 范数归一化**（L103），判定是 $\arg\min$（L109–L112）。**本库综合**：$\min$ 而不是 $\max$ 是因为权重来自"补集"——文档把这一步直接写成"assigned to the class that is the poorest complement match"（L112），但**为什么归一化不会破坏"补集越差越可能是该类"的单调性**，文档没有论证。

### 6. Bernoulli vs Multinomial：不是替换关系，而是"是否惩罚缺席"（来源论述）

差别写得非常干净（L130）：伯努利规则**显式惩罚某个"该类指示特征"的缺席**，而多项变体"would simply ignore a non-occurring feature"。实践建议（L132）：文本分类可用**出现向量**而非计数向量；`BernoulliNB` 在**较短文档**上可能更好，"It is advisable to evaluate both models"。

### 7. Categorical：自带编码契约（来源论述）

`CategoricalNB` 要求 $X$ 已编码为 $0,\dots,n_i-1$（例如 `OrdinalEncoder`），每个特征有自己的类别数 $n_i$（L161），平滑分母因此是 $N_c+\alpha n_i$（L159）。**本库综合**：这意味着**类别集合是特征级契约**——训练集没见过的类别索引在预测时的行为、以及 `min_categories`（外部核查）如何影响"未知类别"的处理，文档都没说，是使用这个估计器最容易沉默出错的地方。

### 8. 外存拟合：三个类 → 实际五个（来源论述 + 外部核查分歧）

来源（L165）：`MultinomialNB`、`BernoulliNB`、`GaussianNB` 暴露 `partial_fit`；"All naive Bayes classifiers support sample weighting"；与 `fit` 不同，**首次调用必须传全部预期类别标签**（L167）；note 建议 chunk 尽量大（L171–L172）。

外部核查发现**这句已经过时**：1.9.1 里五个估计器都有 `partial_fit`——`ComplementNB` 继承 `_BaseDiscreteNB.partial_fit`，`CategoricalNB` 覆写以处理 `min_categories`，五个 API 页也都记录了该方法。详见下节「分歧」。

## 版本判定

剪藏 frontmatter 的 `source` 指向 `/stable/`，未钉版本，因此只能靠比对。本次可用的信号**只有两个字符串**，都很弱：

| 信号 | 剪藏 | 1.8 / 1.7 / 1.6 | 1.9.1 |
| --- | --- | --- | --- |
| 连字符 | `one-dimensional`（L40） | `one dimensional` | `one-dimensional` |
| 标点残留 | `through $x_{n}$:`（L10） | `through \(x_n\), :` | `through \(x_n\):` |

→ 判定 **1.9.x**，旁证是抓取日（2026-11-11）`/stable/` 指向 1.9.1。**本库明确标注这是薄弱证据**：与 1.2 节那种"公式口径变化"级别的判据不可同日而语，不应据此断言具体小版本。两版之间内容实质无变化（1.8→1.9.1 只改了这两处措辞），因此**功能层面的结论不依赖版本判定**。

## 转换损失、回填与外部核查

### A. 抓取时丢失的 5 个折叠块（≈20% 正文）

根因与 [[sklearn-linear-models]] 相同：官方页把这些内容放在 `<details>` 折叠块里，转换器整块丢弃。本节 5 个折叠块的特征：

| 折叠块 | 类型 | 词数 | 丢失后果 |
| --- | --- | --- | --- |
| References（主） | 文献 | 11 | 「为什么 NB 有效」的指定答案（Zhang 2004）不可达，L38 的"see the references below"指针悬空 |
| Weights calculation（1.9.3） | **数学** | 114 | CNB 一节唯一的定量内容全丢，只剩定性描述 |
| References（1.9.3） | 文献 | 28 | CNB 实证主张的出处（Rennie et al. 2003）不可达 |
| References（1.9.4） | 文献 | 68 | 伯努利/MNB 对比的三条出处不可达 |
| Probability calculation（1.9.5） | **数学** | 97 | `CategoricalNB` 的平滑公式全丢，与 MNB 的对应关系看不出来 |

合计 318 词 / 官方 1,485 词 ≈ **21.4%**。

### B. 回填记录（所有者授权）

已按官方 1.9.1 页原文回填上述 5 块，插入位置与标题层级按其结构（`### Weights calculation`、`### Probability calculation`、`## References` ×3），公式改写为规范 LaTeX。**未改动任何原有句子**；回填后行号整体后移（如 `## 1.9.1` 由 L44 → L49），全库引用已按新行号重写。

### C. 残留的排版噪声（未修补）

- 伪 LaTeX `\left(\right.` 出现在高斯似然里（L54）、`\left|\right.` 在条件独立式里（L19）；
- `arg \underset{y}{max}`（L31）不是标准写法；
- L171 的 `> [!note]` 是转换产物（官方页此处确为 Note）；
- 数学块内的空格化写法（`P \left(y \mid x_{1} , \ldots\right)`）是 MathJax alttext 的痕迹，语义正确但不可直接编译。

### D. 外部核查（官方 API 页与 `sklearn/naive_bayes.py` @ tag 1.9.1，**均不在剪藏内**）

| 参数 | 默认值 | 为什么重要 |
| --- | --- | --- |
| `GaussianNB(priors=None, var_smoothing=1e-9)` | 不收缩先验；**方差下限** | `epsilon_ = var_smoothing × max(var(X, axis=0))` 被加到每个方差上（源码 L451/L491）。这是与 `shrinkage`/`reg_param` 并列的**第三个收缩旋钮**，文档一个字没提 |
| `alpha=1.0` | Laplace（=1） | 与 L84 的两种命名一致，但文档没说"默认就是 Laplace" |
| `fit_prior=True`、`class_prior=None` | 从数据学类先验 | 与 LDA/QDA 的 `priors` 同一个坑：重采样后阈值漂移 |
| `force_alpha=True` | 保留 α=0 | α=0 会真的产生零概率，文档只暗示不设平滑就有零概率 |
| `ComplementNB(norm=False)` | 权重**默认不**按 L1 归一化 | 文档 L103 的归一化步骤在默认配置下**并不执行** |
| `BernoulliNB(binarize=0.0)` | >0 即视为 1；`None` 表示输入已二值 | tf-idf 直接喂进去会被二值化，这是最常见的静默改变 |
| `CategoricalNB(min_categories=None)` | 未知 | 与"类别集合是特征级契约"直接相关 |

### E. 分歧：`partial_fit` 的覆盖范围

- **来源论述**（L165）：只有 `MultinomialNB`、`BernoulliNB`、`GaussianNB` 暴露 `partial_fit`。
- **外部核查**（1.9.1 源码与 5 个 API 页）：五个估计器都有该方法（`_BaseDiscreteNB.partial_fit` 被 MNB/CNB/BNB 继承，`CategoricalNB` 自行覆写）。
- **处理**：保留来源原话，同时记录外部核查结果；本库**不据此改写剪藏**（回填只补丢失内容，不改既有句子）。若要确定你的版本行为，直接 `hasattr(ComplementNB(), "partial_fit")`。

## 局限

1. **只有定性比较，没有数字**。整节没有一张图、一个基准表；CNB 与 MNB 的比较、Bernoulli 与 Multinomial 的比较全部是转述他文的断言（L88、L132）。
2. **"为什么错假设还能 work"被外包给参考文献**（L38 → Zhang 2004）。回填后文献条目可达，但论文本身不在本库，本页不重建该论证。
3. **完全不涉及概率校准**。「bad estimator」（L42）没有给诊断方法、没有给补救手段（Platt/isotonic 等一律未提）。
4. **不涉及特征工程与向量化**。词表、n-gram、tf-idf 的实际用法在未入库的 8.2/8.3 节。
5. **不给复杂度**。只说"extremely fast"（L40），没有 $O(\cdot)$ 表达式。
6. **`CategoricalNB` 的类别契约只写了要求，没写违约行为**（未见类别、`min_categories` 的作用）。
7. **缺失值、连续/离散混合特征、多标签**一律未覆盖。

## 待解问题

1. **NB 的概率输出"多坏"、能不能救？** 来源只给了警告。需要校准方法与可靠性图（`sklearn.calibration`）作为独立来源。
2. **Zhang 2004 的"最优性"到底证了什么条件？** 现在文献条目已在剪藏中（L44–L47），但论文未入库。
3. **CNB 的归一化是否可能破坏单调性？**（见关键主张 5 的本库综合）需要读 Rennie et al. 2003 或做小实验。
4. **`var_smoothing` 与 `shrinkage` 的关系**：两者都是"给协方差加一个正的下界/收缩"，是否可视为同一族的两个端点？文档未讨论。
5. **`BernoulliNB` 在短文档上更好的断言**（L132）有多少来自"二值化把长文档的计数信息压掉"？本库无证据。
6. **`alpha` 的量级与词表大小的耦合**（见关键主张 4 的本库综合）：$\alpha n$ 使 α 的"有效强度"随特征数变化，这一点会与 [[sklearn-linear-models]] 的"alpha 不可跨估计器比较"形成同构问题。
7. **`CategoricalNB` 遇到训练集未见类别时的行为**（`min_categories` 的作用）——需读源码或实验。

## 行号索引（回填后）

| 主题 | 行号 |
| --- | --- |
| 贝叶斯定理 / 条件独立 / MAP 判定 | L10、L19、L31、L34 |
| 「变体只差 $P(x_i\mid y)$」 | L36 |
| 小样本、快、维度灾难 / **bad estimator** | L38–L40 / **L42** |
| 主 References（Zhang 2004，回填） | L44–L47 |
| GaussianNB 似然 + MLE / iris doctest | L53–L57 / L59–L70 |
| Multinomial 平滑公式 / Laplace 与 Lidstone | L78–L82 / L84 |
| Complement 定性描述 / Weights calculation（回填）/ CNB References（回填） | L88 / L90–L112 / L114–L118 |
| Bernoulli 判定规则 / 与 MNB 的差别 / 短文档建议 / References（回填） | L124–L128 / L130 / L132 / L134–L143 |
| Categorical 类别分布 / Probability calculation（回填）/ ordinal 编码要求 | L147–L149 / L151–L159 / L161 |
| 外存拟合 / 首次传 classes / chunk note | L165 / L167 / L171–L172 |

## 关联页面

- 概念：[[naive-bayes]]（五个生成假设、平滑作为正则化面、失效模式）、[[gaussian-discriminant-analysis]]（`GaussianNB` ⊂ LDA ⊂ QDA 的嵌套与参数代价）、[[regularization-and-sparsity]]（第四类旋钮）、[[leaderboard-vs-cv]]（概率 vs 排序 vs 阈值）
- 工具：[[scikit-learn]]（`naive_bayes` 估计器矩阵与默认值表）
- 实践：[[linear-models-as-baselines]]（零成本基线族的定位）
- 同批来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]
