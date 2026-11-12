---
title: "朴素贝叶斯（Naive Bayes）：条件独立的代价、五种类条件假设与平滑作为正则化面"
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - naive-bayes
  - generative-model
  - classifier
  - smoothing
sources:
  - "[[sklearn-naive-bayes]]"
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-covariance]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.x 的 naive_bayes 模块（推断，见 [[sklearn-naive-bayes]] 的版本判定）
---

# 朴素贝叶斯

本页整理「**假设所有特征在给定类别下条件独立**」这一族生成式分类器：这一条假设换来了什么、付出了什么、五个成员各自假设了什么分布、以及在什么条件下它会安静地给出错误答案。事实来自 [[sklearn-naive-bayes]]（引用写作 `[[sklearn-naive-bayes]], L42`）；凡本库推导、跨页综合或未验证的推断都显式标注。

## 一条假设换来两样东西

朴素贝叶斯把 $p(x\mid y)$ 写成 $\prod_i p(x_i\mid y)$（[[sklearn-naive-bayes]], L19、L31）。这一刀切下去换到两样东西：

| 换到的 | 机制 | 依据 |
| --- | --- | --- |
| **参数量从 $O(d^2)$ 降到 $O(d)$** | 联合密度的耦合项全部丢掉，每个特征独立估一维分布 | L19、L40 |
| **训练成本极低、可增量** | 只需计数或均值/方差，无迭代；离散族支持 `partial_fit` | L40、L165 |

付出的则是**全部特征间相关性信息**：若两个特征高度相关，NB 会把同一份证据数两遍。

**本库综合（判断准则）**：因此「NB 能不能用」不是"特征是否真的独立"（几乎从不成立），而是"**相关性是否会改变类别的相对排序**"——排序不变时，概率会偏但 argmax 不变。来源自己也把"worked quite well in many real-world situations"归因于经验（L38），并把理论解释外包给了参考文献（Zhang 2004，L44–L47）。

## 五种类条件假设

| 变体 | $p(x_i\mid y)$ | 适用特征形态 | 平滑/参数 | 位置 |
| --- | --- | --- | --- | --- |
| `GaussianNB` | 高斯（$\mu_y,\sigma_y$ 用 MLE） | 连续 | `var_smoothing`（外部核查） | L51–L57 |
| `MultinomialNB` | 多项分布（计数） | 计数/tf-idf（非负） | `alpha`（默认 1.0 = Laplace） | L74–L84 |
| `ComplementNB` | 多项，但统计量取**类补集** | 同上，尤其不平衡数据 | `alpha`、`norm`（外部核查，默认 `False`） | L88–L112 |
| `BernoulliNB` | 伯努利（二值） | 0/1 出现向量 | `alpha`、`binarize`（默认 0.0） | L122–L132 |
| `CategoricalNB` | 逐特征独立类别分布 | 已 ordinal 编码的类别 | `alpha`、`min_categories`（外部核查） | L147–L161 |

四件容易读漏的事：

1. **`MultinomialNB` 的输入不是"概率"而是"计数"**，tf-idf 也work（L74），但负数不行——这一点来源没写，属实现约束。
2. **`BernoulliNB` 与 `MultinomialNB` 不是替换关系**：伯努利**显式惩罚"某个指示特征没出现"**，多项变体"would simply ignore a non-occurring feature"（L130）。所以短文档上二者常有明显差异，来源的建议是"evaluate both"（L132）。
3. **`ComplementNB` 的权重来自补集**（L90–L106），判定用 $\arg\min$ 而非 $\arg\max$（L109–L112）；它是**为不平衡数据**设计的（L88）。
4. **`CategoricalNB` 自带编码契约**：$X$ 必须已编码成 $0..n_i-1$（L161），平滑分母是 $N_c+\alpha n_i$（L159）。

## 参数代价：它比 LDA 还省

与 [[gaussian-discriminant-analysis]] 的推导口径一致（自由参数个数，**本库推导**）：

| 成员 | 合计自由参数 | $d=100,K=10$ |
| --- | --- | --- |
| `GaussianNB` | $2Kd+K-1$ | 2,009 |
| LDA | $\frac{d(d+1)}{2}+Kd+K-1$ | 6,059 |
| QDA | $K\frac{d(d+1)}{2}+Kd+K-1$ | 51,509 |

**本库综合**：`GaussianNB` 是「QDA + 对角协方差」的价格（[[sklearn-lda-qda]], L60–L63），比 LDA 便宜三倍，因为它连"共享协方差"都不估，只估 $2Kd$ 个标量。这解释了它为什么在 $d$ 大、样本少时很少崩——代价是**类内相关性完全不可见**。

## 平滑就是正则化：第四个面

[[regularization-and-sparsity]] 已记录两个面（系数惩罚、协方差收缩）。NB 加进来第三个对象：**概率/方差本身**。

| 旋钮 | 作用对象 | 语义 | 依据 |
| --- | --- | --- | --- |
| `alpha`（离散族） | $\hat\theta_{yi}=\frac{N_{yi}+\alpha}{N_y+\alpha n}$ | 每个特征加 $\alpha$ 个伪计数；α=1 Laplace、α<1 Lidstone、α=0 无平滑 | L78–L84 |
| `var_smoothing`（`GaussianNB`） | $\sigma_y^2$ | 加一个**下限** $\epsilon=$ `var_smoothing`×$\max_j \mathrm{var}(X_j)$（默认 1e-9） | 外部核查 |

三个要点：

1. **`alpha` 是"按特征数缩放"的**（分母含 $\alpha n$），所以同一个 `alpha` 在 1,000 词表与 50,000 词表上的有效强度不同。**本库综合**：这与 [[linear-models-as-baselines]] 的「alpha 不可跨估计器比较」是同一类陷阱的另一个实例。
2. **平滑保证的是"不为零"，不是"更准"**。来源的理由写得很明确：`alpha` 用来照顾未见特征并**防止零概率**（L84）。
3. **`var_smoothing` 与 `shrinkage` 家族的相似性只是形式上的**。`shrinkage` 把协方差拉向一个结构化目标（1.2 节文档说「对角阵」，2.6 节的文档与实现是 $\frac{\mathrm{Tr}}{p}I$，见 [[covariance-estimation]] §2.3），`var_smoothing` 只是给每个方差加一个正的下界——它**不改变非对角项**（`GaussianNB` 本来就没有非对角项）。因此把两者当"同一个超参"是错的；这是**本库判断**，文档未讨论。

## 「坏估计量」：三条工程推论

来源唯一的警告（L42）：NB 是"decent classifier"但"bad estimator"，`predict_proba` 的输出"not to be taken too seriously"。

**本库推论（综合，非来源论述）**：

| 用法 | 受影响程度 | 理由 |
| --- | --- | --- |
| `predict` 的类别标签 | 不受影响 | argmax 只看排序 |
| 仅依赖排序的指标（AUC、平均精度） | 大概率不受影响 | 单调变换不改变排序 |
| `predict_proba` 当**阈值**（风控、告警、采样决策） | **受影响** | 概率被推向 0/1，阈值语义失真 |
| 多模型概率**加权融合** / 概率乘积 | **受影响** | 若其他模型校准良好，权重会系统性偏向 NB |
| 期望收益、成本敏感决策 | **受影响** | 需要的是校准后的概率，不是"自信度" |

来源没有给“多坏”的量化，也没有给校准方法（Platt scaling、isotonic 等一律未提）。因此上表第 3–5 行是**本库综合**，可作为假设检验的起点，不能当来源结论引用。

**与 SVM 的对照（跨来源综合）**：[[sklearn-svm]] 是本库第二份说“分数 ≠ 概率”的来源，但它给了一条具体修法（Platt 校准，[[sklearn-svm]], L128），并同时列出修法与分数不一致的两种可观测情形（L135–L136）。两条合起来读数：**校准把“不够可信”换成“可信但非保序”**，不保证分数与概率的 argmax 一致。四层输出对照见 [[support-vector-machines]] 的「一个分类器的四个输出面」。

## 失效模式

1. **重复/高度相关的特征 ⇒ 过度自信**。同一份证据被 $k$ 个相关特征各数一遍，后验被推向 0/1。来源未讨论；**本库综合**。
2. **分布假设与数据不匹配**：把计数当二值（`BernoulliNB` 的默认 `binarize=0.0` 会静默二值化，外部核查）、把类别编码当连续（喂给 `GaussianNB`）、词表外类别喂给 `CategoricalNB`（L161 的契约违约行为未文档化）。
3. **先验从数据学**：`fit_prior=True`（外部核查）＋ $P(y)$ 即训练集类比例（L34），所以过采样/欠采样后阈值漂移——与 LDA/QDA 的 `priors` 是同一个陷阱（[[sklearn-lda-qda]] 关键主张 7）。
4. **不平衡数据**：默认的 `ComplementNB` **不做**权重 L1 归一化（`norm=False`，外部核查），而文档 L103 描述的归一化是公式的一部分——按文档的公式实现与按默认参数运行不是同一件事。

## 与其他基线族的关系

| 族 | 假设 | 参数代价 | 零成本？ | 本库页面 |
| --- | --- | --- | --- | --- |
| 判别式线性（Ridge/Logistic） | 决策面线性 | $O(d)$，带正则强度 | 是（内置 CV） | [[linear-models-as-baselines]] |
| 生成式判别分析（LDA/QDA） | 类条件**联合**高斯 | $O(d^2)$ / $O(Kd^2)$ | 是（闭式） | [[gaussian-discriminant-analysis]] |
| 朴素贝叶斯 | 类条件**独立**（对角/Factored） | $O(Kd)$ | 是（计数/矩估计） | 本页 |
| 树/集成 | 轴对齐分段常数 | — | 否（需调参 + 更多算力） | [[decision-trees]] / [[ensembles]] |

**本库综合**：三族零成本基线最好**一起跑**，因为它们的失效方式互不重叠——线性模型怕非线性交互，LDA/QDA 怕非高斯，NB 怕特征冗余。三者都失败时，才值得把算力投给树模型。

## 待解问题

- **NB 的概率“多坏”、能不能救？** 来源只给警告（L42）。需要 `sklearn.calibration` 作为独立来源。**部分线索**：[[sklearn-svm]] 的 Platt 校准是同一问题的另一条已知路径，但它在 1.9 已被弃用（理由是非线程安全），且不保证与分数同序——说明“校准”本身是一个有代价的独立步骤。
- **特征冗余导致的过度自信能否量化？** 本库没有来源；可作为假设做实验（构造 $k$ 份相关特征，观察 `predict_proba` 的漂移）。
- **Zhang 2004 的"最优性"是在什么条件下成立的？** 文献条目已在剪藏（L44–L47），论文未入库。
- **`alpha` 与词表大小的交互**：是否存在让 $\alpha n$ 归一化的等价写法（如按平均文档长度缩放）？
- **`var_smoothing` 是否应随特征尺度调整？** 它按 $\max_j\mathrm{var}(X_j)$ 缩放，因此对"量纲差异极大"的特征集是不对称的——这是**本库读源码得到的推断**，未验证。

## 关联页面

- 来源：[[sklearn-naive-bayes]]
- 工具：[[scikit-learn]]（`naive_bayes` 估计器矩阵与默认值）
- 概念：[[gaussian-discriminant-analysis]]（`GaussianNB` ⊂ LDA ⊂ QDA 的嵌套与参数代价推导）、[[regularization-and-sparsity]]（平滑作为第四个正则化面）、[[support-vector-machines]]（另一个“分数 ≠ 概率”的来源与四层输出）、[[leaderboard-vs-cv]]（排序指标 vs 概率指标）
- 实践：[[linear-models-as-baselines]]（零成本基线族的定位与尺度约束）
