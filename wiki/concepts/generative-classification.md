---
title: "生成式分类（Generative Classification）：用 P(x|y) 反推 P(y|x)"
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - generative-model
  - classifier
  - bayes-rule
  - llm
  - tabular
sources:
  - "[[sklearn-lda-qda]]"
  - "[[sklearn-naive-bayes]]"
  - "[[kaggle-tfms-llms-tabular]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.x 的 discriminant_analysis 与 naive_bayes 模块（推断）
  - GPT-2 124M 上的「行文本化」生成式分类器（单赛题、单种子，见 [[kaggle-tfms-llms-tabular]]）
---

# 生成式分类

判别式分类器学 $P(y\mid x)$；生成式分类器学 $P(x\mid y)$ 与 $P(y)$，再用 Bayes 公式把两者换过来。这条轴在教科书里很老（[[gaussian-discriminant-analysis]]、[[naive-bayes]] 都是它的成员），但 [[kaggle-tfms-llms-tabular]] 给了一个新的极简实例：**把表格行写成一句话，让 GPT 预测「哪个标签让这句话最可能」**——同一个 GPT-2 骨干、同一份数据，只因为**标签写在句首还是句末**，就从判别式变成生成式，并在该赛题上拿到 0.96242 对 0.961 的差距（[[kaggle-tfms-llms-tabular]], L42–L48）。

本页整理这根轴上的**代数、实例、成本与失效边界**。事实来源已在各节标注；凡本库推导均写「**综合**」或「**推导**」。

## 一根轴，两端

| | 判别式 | 生成式（本页） |
| --- | --- | --- |
| 建模对象 | $P(y\mid x)$ | $P(x\mid y)$ 与 $P(y)$ |
| 目标函数和指标的关系 | 直接最大化（对数）似然 / 间隔 | 间接：先拟合 $x$，再反推 $y$ |
| 训练信号 | 只用在「区分这些类别」上 | 用在「复原 $x$ 的全部细节」上，包括与 $y$ 无关的部分 |
| 多类成本 | 一次前向给 $K$ 个 logit | **每类一次**（$K$ 次密度评估） |
| 典型失败模式 | 假设空间不足 | 密度模型容量不足，或 $x$ 里噪声太多 |
| 诊断优势 | 系数/重要性是近似的 | 证据可以**精确**按特征相加（见「证据的可加性」） |

**综合**：这条轴与「特征之间是否独立」是**两个正交的轴**。朴素贝叶斯是「生成式 + 条件独立」，LDA/QDA 是「生成式 + 高斯（共享或独立协方差）」，Storyteller 是「生成式 + 神经自回归密度」——三者都放弃了或没有放弃独立性假设，但都仍然按生成式方式分类。因此「生成式过时了」这类说法没有意义：换掉的是密度模型，不是轴。

## 代数

二分类，令 $y\in\{0,1\}$：

$$
\text{log-odds}(1\mid x)=\log\frac{P(y=1)}{P(y=0)}+\log p(x\mid y=1)-\log p(x\mid y=0),\qquad P(y=1\mid x)=\sigma(\text{log-odds})
$$

多类就是每个类各算一次 $\log P(y=k)+\log p(x\mid k)$ 再做 softmax。[[kaggle-tfms-llms-tabular]] 的打分代码（L104–L117）正是这个式子的三项：先验 logit、$\log P(x\mid\text{True})$、$\log P(x\mid\text{False})$。

两条必须记住的性质：

1. **配分函数 $Z$ 会约掉**（推导）：$\log p(x\mid y)=\log \tilde p(x\mid y)-\log Z(y)$，只有当 $Z$ 与 $y$ 无关时才能忽略；「同一个骨干 + 同一套词表 + 同样的行序列化」是让这一条近似的工程条件。
2. **只需要同类之间可比的分数**，不需要校准。对 AUC 这类只依赖排序的指标（[[evaluation-metrics]]），任何单调变换都无害；要报概率、定阈值或优化交叉熵时，缺的那一项就必须补回来。

## 三个实例，同一根轴

| 实例 | 密度模型 $p(x\mid y)$ | 独立性 | 每类参数代价 | 来源 |
| --- | --- | --- | --- | --- |
| `GaussianNB` | 各特征一维高斯的乘积 | **有**（对角协方差） | $O(d)$ | [[gaussian-discriminant-analysis]]、[[naive-bayes]] |
| LDA / QDA | 多元高斯，协方差共享 / 每类独立 | 无（但高斯） | $O(d^2)$ | 同上 |
| 文本化自回归 LM（Storyteller） | GPT-2 在 `"目标 is v, 列1 is v, …"` 上的 token 级似然 | 无（自回归，可表达任意依赖） | 124M 级参数量 | [[kaggle-tfms-llms-tabular]], L88–L130 |

**这张表最值得记的一点**：从 `GaussianNB` 到 Storyteller，**没有换掉生成式框架，只是把「乘积形式」换成了「自回归形式」**。而自回归形式恰好能在**一个**模型里表达字段之间的任意依赖，代价从「闭式的均值 + 协方差」变成了「一次神经网络训练」。

## 证据的可加性（生成式独有的诊断）

因为 $\log p(x\mid y)$ 是逐 token 的对数和，可以按字段分组求和，得到每个字段对 log-odds 的**精确**贡献：

$$
\text{evidence}(x_j)=\log p(x_j\mid y=1)-\log p(x_j\mid y=0).
$$

[[kaggle-tfms-llms-tabular]] 的 row 0 示例如下（图 `row0_evidence.png`，单位 nats）：

| 项 | 证据 |
| --- | --- |
| 先验 $\log P(\text{True})-\log P(\text{False})$ | −0.20 |
| Type of Travel = Personal Travel | **−2.04** |
| Inflight wifi service = 4 | **+1.34** |
| Age = 36 | −0.44 |
| Customer Type = Loyal Customer | +0.26 |
| Flight Distance = 694 | +0.11 |
| 其余 16 个字段 | −0.09 |
| 合计（一阶） | **−1.05** → $P=0.26$ |

**注意加法性的来源**：它来自「按字段把 token log-prob 求和」，**不是**来自字段独立。分组内部仍然包含字段之间的依赖——这正是它与朴素贝叶斯的关键区别，也是为什么它可以和注意力重要性 / 积分梯度这类**近似**归因（[[tabular-deep-learning]] 的注意力一节）区别开：这里的每一项都是对最终分数的**恒等**分解，不是代理。

## 成本：$K$ 次前向，还要乘顺序

- **判别式**：一次前向 → 全部 $K$ 个 logit。
- **生成式（LDA/NB）**：闭式，成本在估计协方差/条件分布，不在推理。
- **生成式（text LM）**：每类一次前向。若密度模型对输入顺序敏感（因果解码器对「行」没有天然顺序），还要对 $O$ 个顺序取平均 —— [[kaggle-tfms-llms-tabular]] 用 $K=2$、$O=30$，即**每行 60 次前向**，而它的判别式版本只要 1 次（L47、L130–L132）。来源**没有把这条成本写进任何对比**。

**综合**：生成式的推理成本 = $K \times O \times$（一次密度评估），在 LLM 级骨干上是数量级差。选它之前先算这条乘积，而不是只看 AUC。

## 数值：在一个 −69 的数和另一个 −69 的数之间

`two_directions.png` 里 row 0 的两个总和是 $\sum\log p = -69.40$（True）与 $-68.55$（False）。分类信号是**两个约 −69 的数之间 0.85 nats 的差**，而 21 个字段的逐项加和又必须在正确的 token 边界上切开。这带来两条工程要求（**综合**）：

1. **累加用 fp32**：来源用 bf16 训练，但未说明打分精度；bf16 的有效位数（约 8 位二进制）在 −69 量级上会带来 $10^{-2}$ 级的误差，与 0.85 的差同量级。
2. **前缀切分必须精确**：`tok_lp[n_prefix - 1:]` 依赖 `n_prefix` 是否包含末位空格；偏移一个 token，误差会直接变成系统性 bias 而不是噪声。

## 什么时候生成式会赢

1. **数据的生成过程本身就是「先 $y$ 后 $x$」**。这是 [[kaggle-tfms-llms-tabular]] 的机制主张：该赛题数据被推断为由 GReaT（GPT-2 骨干）生成、标签最先（L195–L204），于是「生成器自己对哪个标签最可能」的答案就是最强分类器。**这条是假设，未经非合成数据验证**。
2. **类别少、每类数据足**：$K$ 小则成本可控，每类密度都估得准——LDA/NB 的经典场景。
3. **需要证据分解**：如上节，生成式给出恒等的逐字段贡献。
4. **小样本 + 强结构假设**：朴素贝叶斯的偏差-方差交换（[[naive-bayes]]）。

## 什么时候会输

- **$K$ 大**：成本线性乘 $K$，且每类都要一个可用的密度估计。
- **密度模型容量不足**：$\log p(x\mid y)$ 全都很小、差别被噪声淹没（$x$ 维度高、每类样本少）。
- **必须给 $x$ 的全部细节建模**：与 $y$ 无关的噪声也要花容量去拟合——这是判别式的天然优势。
- **序列化格式与生成时不一致**：判别式对无意义的格式差异更鲁棒，生成式会把「`0.0` 写成 `0`」当成证据（[[kaggle-tfms-llms-tabular]], L145）。
- **依赖长程结构**：朴素贝叶斯的独立假设失效（[[naive-bayes]] 的失效模式）；此时要么换密度模型，要么退回判别式。

## 与判别式不是对立，而是可切换

[[kaggle-tfms-llms-tabular]] 最值得记的观察是：**同一个骨干、同一份数据，只移动标签的位置，就换了建模方向**——标签放末尾 = 判别式（Detective），放句首 = 生成式（Storyteller）（L42–L48、L152–L158）。所以这不是「两种架构」的选择，而是**序列化/提示设计**的一个选择，与 [[tabular-deep-learning]] 里「位置信息怎么注入」属于同一类问题。

作者给出的实用建议是**两个都训再 stack**：它们从相反方向读同一行，错误不同（L156 的「they make different errors and stack nicely」）；最终 stack 里 10 个 GPT 系 pair 承担约 80% 的输出（L240）。这条与 [[ensembles]] 的「平均降方差需要误差不相关」是同一条机制。

## 待解问题

1. 生成式的 $+1\text{e-}3$ 优势是**数据生成过程**带来的，还是**建模方向**本身带来的？需要一个「生成过程不是 LM」的数据集做对照——来源没有做。
2. 对顺序做 log-sum-exp 得不到归一化似然。要输出校准概率，缺的 $Z$ 该怎么补（固定顺序？重要性采样？）？
3. 在 $K$ 很大（多分类）时，有没有「一次前向给所有类」的生成式近似（例如把 $y$ 也 token 化后取条件分布的边际）？
4. 生成式的证据分解与 LDA 的判别系数、NB 的 log-likelihood ratio 是什么关系？三者在同一数据上会不会给出不同排序？
5. bf16 与 fp32 下 $\sum\log p$ 的差有多大？会不会改变 AUC？（见「数值」一节）
6. 若密度模型的序列化与训练时不同形（分隔符、小数位、缺失写法），AUC 会掉多少？这条没有对照。

## 关联页面

- 经典成员：[[gaussian-discriminant-analysis]]（LDA/QDA/GaussianNB 的嵌套与几何）、[[naive-bayes]]（条件独立的收益与代价）
- 新实例：[[kaggle-tfms-llms-tabular]]（文本化行 + GPT-2 + Bayes 反转）
- 判别式对照：[[support-vector-machines]]、[[regularization-and-sparsity]]、[[multi-layer-perceptron]]
- 选型与集成：[[tabular-model-selection]]、[[ensembles]]、[[gbdt-vs-tabular-dl]]
- 指标口径：[[evaluation-metrics]]（秩指标 vs 概率指标）、[[leaderboard-vs-cv]]
