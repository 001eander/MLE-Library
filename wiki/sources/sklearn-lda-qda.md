---
title: "scikit-learn 用户指南 1.2「Linear and Quadratic Discriminant Analysis」：生成式分类器、Mahalanobis 几何与协方差收缩（stable 文档剪藏）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - discriminant-analysis
  - covariance
  - reference-doc
source_path: raw/articles/sklearn/1.2. Linear and Quadratic Discriminant Analysis.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.2「Linear and Quadratic Discriminant Analysis」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Linear and Quadratic Discriminant Analysis（scikit-learn 用户指南第 1.2 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；无代码片段、无 doctest，只有公式与两张示例图外链 |
| 原文 | <https://scikit-learn.org/stable/modules/lda_qda.html>（剪藏 frontmatter `source`） |
| 本地文件 | `raw/articles/sklearn/1.2. Linear and Quadratic Discriminant Analysis.md`（146 行，21,050 字节） |
| SHA-256 | `b2e50ad9a1b4406924a1eaaa7ed5e9980bdb147b8b78f85160d0df68e9dede55` |
| 完整性 | **正文完整、参考文献全缺**：官方页无内容折叠块（8 个 `<details>` 全是侧栏导航），正文逐句可对；但真正的 3 条参考文献被替换成 4 段重复正文，4 处引用标记 `[1][2][3]` 被吞（详见「转换损失与外部核查」） |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x**（外部核查所得，见「版本判定方法」）。判据：QDA 的 SVD 推导写作 $\Sigma_k=\frac{1}{n}X_k^{T}X_k$，而 1.6/1.7/1.8 均为 $\frac{1}{n-1}$，1.9 起才改为 $\frac{1}{n}$ |
| 插图 | 2 个不同文件（正文各引用一次，shrinkage 图在末尾重复出现一次）；已抓取入库 `raw/assets/sklearn-lda-qda/` 并**逐张目视核查** |
| 代码 | 无 |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-lda-qda]], L115`。凡超出剪藏内容的部分（API 参数、版本差异、图内读数）都在句子里显式标注「外部核查」或「本库」，不冒充剪藏原文。

## 概要

这是 sklearn 用户指南中很短却密度很高的一节（5 个小节，约 1,600 词正文）：它把 LDA/QDA 写成**同一个生成式模型的两个特例**，然后用三件事把"经典分类器"讲成工程可用的东西——决策面的几何（Mahalanobis）、降维的几何（sphering + K−1 维投影），以及小样本下的协方差估计问题（shrinkage）。

对本库的意义有三层：

1. **它补上了 [[sklearn-linear-models]] 缺失的"生成式一侧"**。1.1 节的模型全部是判别式（直接拟合 $p(y\mid x)$ 或决策函数），本节第一次给出 $p(x\mid y)$ 的建模路径，并把 LDA 明确写成「QDA 共享协方差矩阵」的特例（L67）。
2. **它把「正则化」从系数空间挪到了协方差空间**。`shrinkage`、`covariance_estimator`、以及 API 里的 QDA `reg_param`（外部核查）都是对 $\Sigma$ 下手，与 Ridge/Lasso 对 $w$ 下手是**两个不同对象上的两套正则**（见 [[regularization-and-sparsity]] 新增小节）。
3. **它是本库第二份可判版本的 sklearn 剪藏，而且比第一份更精确**：1.1 只能判到 ≥1.8，本节可判到 1.9.x，因为 1.9 恰好改了这里的一个公式。两页合起来给 [[scikit-learn]] 工具页提供了版本锚点。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 两个分类器的卖点：闭式解、天生多类、「无需调参」 | L10–L20 |
| 1.2.1 Dimensionality reduction using LDA | `transform` + `n_components`；输出维必小于类数 | L22–L30 |
| 1.2.2 Mathematical formulation | 类条件高斯 + 贝叶斯 → 后验 | L32–L48 |
| 1.2.2.1 QDA | 逐类协方差的 log 后验；对角协方差 ⇒ GaussianNB | L50–L63 |
| 1.2.2.2 LDA | 共享 $\Sigma$ ⇒ Mahalanobis ⇒ 线性决策面；$\omega_k$ / $\omega_{k0}$ 即 `coef_` / `intercept_` | L65–L83 |
| 1.2.3 LDA 降维的数学形式 | 类均值落在 ≤ K−1 维仿射子空间；sphering 等价；对类均值做 PCA | L85–L93 |
| 1.2.4 Shrinkage and Covariance Estimator | 小样本协方差估计、Ledoit-Wolf `'auto'`、端点 0/1、OAS、`covariance_estimator` | L95–L109 |
| 1.2.5 Estimation algorithms | `svd` / `lsqr` / `eigen` 三个求解器与 QDA 的 SVD 推导 | L111–L119 |
| （伪）References | 4 段重复正文，**真实文献缺失** | L121–L146 |

## 关键主张

### 1. 卖点是"闭式 + 多类 + 不用调参"（来源论述，且与同页后文张力）

> These classifiers are attractive because they have closed-form solutions that can be easily computed, are inherently multiclass, have proven to work well in practice, and have no hyperparameters to tune.（L12）

前三项在页内都有支撑：闭式解来自高斯 + 贝叶斯（L34–L46），多类来自「每类一个均值一个协方差」的构造（L113），"work well in practice" 是无法核验的断言。**第四项与本节自己冲突**：L97–L103 介绍了 `shrinkage`（`'auto'` 或 0–1 的连续值）与 `covariance_estimator`，L115–L119 有三个 `solver` 选项。外部核查还多出 `priors`（默认＝训练集类比例）与 QDA 的 `reg_param`（见第 4 节）。

**本库判断**：这句话应当读作「**基础模型**没有需要搜索的超参」——没有 `alpha`、`C`、`n_estimators` 那类正则强度，只有一个**是否需要收缩协方差**的结构性选择。把它当成"开箱即用"会漏掉小样本下最关键的开关。

### 2. LDA 同时是一个监督降维器，输出维被类数硬性封顶

来源：`transform` 实现监督降维，「the dimension of the output is necessarily less than the number of classes」（L24），且 `n_components`「has no influence on the `fit` and `predict` methods」（L26）。

外部核查（官方 API 页，非剪藏）：`n_components` 的默认与上限是 `min(n_classes - 1, n_features)`，且「This parameter only affects the `transform` method」。

**本库综合**：这是一个**有损且封顶极低**的降维器——10 类问题最多给 9 维。它的用途不是"通用压缩"，而是"把 K−1 维内最可分的子空间拿出来"，与 PCA（无监督、上限 `min(n, p)`）是两种不同目标，官方示例 `plot_pca_vs_lda` 就是在做这个对照（L30）。

### 3. QDA 与 LDA 的嵌套关系，以及 QDA 与高斯朴素贝叶斯的等价条件

- QDA：每类自己的协方差 $\Sigma_k$，log 后验含 $-\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)^{T}\Sigma_k^{-1}(x-\mu_k) + \log P(y=k)$（L52–L58）。
- 对角 $\Sigma_k$ ⇒ 各类特征条件独立 ⇒ **等价于 `naive_bayes.GaussianNB`**（L60–L63）。
- LDA：$\Sigma_k=\Sigma$ 对所有权 $k$ 成立（L67）；此时 $-\frac12\log|\Sigma|$ 与 $k$ 无关被吸收进常数，log 后验只剩 Mahalanobis 项加先验（L69–L71）⇒ 线性决策面（L83）。

由此得到 LDA 的两种等价读法（都由来源给出）：**Mahalanobis 最近均值 + 先验**（L73），以及**仿射形式** $\omega_k^{T}x+\omega_{k0}$，其中 $\omega_k=\Sigma^{-1}\mu_k$、$\omega_{k0}=-\frac12\mu_k^{T}\Sigma^{-1}\mu_k+\log P(y=k)$——来源明确说这两个量「correspond to the `coef_` and `intercept_` attributes」（L75–L81）。

**本库综合**：这三层是同一个模型的三个粒度——`GaussianNB` ⊂ LDA ⊂ QDA。对工程的含义是：**QDA 的额外自由度只在各类协方差形状不同时才有价值**，否则它只是在估计更多参数（见图 1 第 1、2 行）。

### 4. 降维的几何：先白化，再投影到类均值张成的子空间

来源给出的论证链条（L87–L93）：

1. $K$ 个均值 $\mu_k\in\mathbb{R}^d$ 落在**至多 K−1 维**的仿射子空间 $H$ 里（两点成线、三点成面）；
2. LDA 等价于先把数据 *sphering* 成协方差为单位阵，再取欧氏最近均值（L89）；
3. 白化空间里的欧氏距离可以只在与 $H$ 正交的补空间外比较——那些维度对所有类贡献相同——所以分类隐含了一次到 K−1 维的线性投影（L91）；
4. 想降到更低的 $L$ 维，就投影到「使投影后 $\mu_k^{*}$ 方差最大」的子空间 $H_L$——「in effect, we are doing a form of PCA for the transformed class means」（L93），这就是 `n_components`。

**本库综合**：第 4 步是本页最容易被误读的一句。它不是"对数据做 PCA"，而是**对（白化后的）类均值做 PCA**——样本量等于类数，因此这一步几乎不可能过拟合，但也完全看不见类内结构。

### 5. shrinkage：小样本下的协方差正则化，但**只能配两个求解器**

来源论述（L97–L103）：

- 动机：训练样本相对特征数太少时，经验样本协方差是坏的估计量，收缩能提升泛化（L97）。
- `shrinkage='auto'`：「automatically determines the optimal shrinkage parameter in an analytic way following the lemma introduced by Ledoit and Wolf」（L97）。
- **约束**：`shrinkage` 只在 `solver='lsqr'` 或 `'eigen'` 下有效，QDA **只有 `'eigen'`**（L97）。
- 端点语义：0 = 用经验协方差；1 = 完全收缩，用**对角方差矩阵**当作协方差估计；中间值 = 两者的收缩版（L99）。
- 反例/更优点：若数据（条件于类）确实高斯，OAS（`sklearn.covariance.OAS`）的 **MSE 更小**，因此「will yield a better classification accuracy than if Ledoit and Wolf or the empirical covariance estimator is used」（L101）。
- `covariance_estimator`：可换成任何具备 `fit` 方法与 `covariance_` 属性的协方差估计器（L103）。

外部核查（实现层，非剪藏内容）：源码在 `solver='svd'` 且 `shrinkage is not None` 时直接抛 `NotImplementedError("shrinkage not supported with 'svd' solver.")`，`covariance_estimator` 同理（LDA 抛 `ValueError`，提示换 solver）——所以这不是“默认值隐式失效”，而是显式报错（核查对象：`sklearn/discriminant_analysis.py`，1.9.X 分支）。

**本库综合（重要口径）**：L97 的「optimal」指的是**协方差矩阵估计误差的最小化**，不是分类准确率。证据是 L101 自己用 MSE 来比较 Ledoit-Wolf 与 OAS——如果优化目标是准确率，"MSE 更小"就不构成理由。所以 `shrinkage='auto'` 应理解为「一个有明确损失函数的默认值」，不是「分类任务上的最优值」；真要选，就该把 OAS / Ledoit-Wolf / 固定值当三条候选做 CV。

外部核查（官方 API 页）：`shrinkage` 与 `covariance_estimator` **互斥**（两处都写着 "This should be left to None if … is used"）。

### 6. 三个求解器的能力矩阵

| 求解器 | 分类 | `transform` | shrinkage / 自定义协方差估计器 | 是否显式算 $\Sigma$ | 来源位置 |
| --- | --- | --- | --- | --- | --- |
| `svd`（**两者默认**） | 是 | LDA 是 | **否** | 否 | L115 |
| `lsqr`（仅 LDA） | 是 | 否 | 是 | 是（解 $\Sigma\omega=\mu$，不求逆） | L117 |
| `eigen`（LDA 与 QDA） | 是 | LDA 是 | 是（QDA 唯一可用） | 是 | L119 |

两点细节值得单独记：

- **QDA 的 SVD 为什么可行**：$\Sigma_k$ 按定义等于 $\frac{1}{n}X_k^{T}X_k=\frac{1}{n}VS^2V^{T}$，其中 $X_k=USV^{T}$ 是（中心化后的）类内矩阵的 SVD，所以 log 后验可以不显式构造 $\Sigma$（L115）。LDA 则要算两次 SVD：中心化输入矩阵一次、类均值向量一次。
- **`eigen` 在两边含义不同**：LDA 里它优化「between-class scatter / within-class scatter」之比；QDA 里它是**逐类**协方差的特征分解（L119）。

外部核查补充（官方 API 页）：`svd` 求解器下 LDA 的 `covariance_` 默认**不存储**（需 `store_covariance=True`），其它求解器总是计算并存储；QDA 的 `covariance_` 需要 `store_covariance=True`，且用最大似然（有偏）估计。

### 7. 「先验」在页内只出现在公式里，没有作为参数讨论

L113 明说 log 后验依赖「class priors $P(y=k)$」，公式里也一直带着 $\log P(y=k)$（L55、L70、L81），但**全文没有出现 `priors` 参数**。外部核查：`priors` 默认 `None`，此时用训练集的类比例；也可以显式给一个长度 `n_classes` 的数组。

**本库综合**：这是本节最实用的一处文档缺口。默认取经验类比例的后果是——**重采样（过采样/欠采样）之后 fit 的 LDA/QDA，其决策阈值会跟着重采样后的比例移动**，而这一点在正文里完全看不出来。不平衡数据上必须显式设 `priors` 或改用后验概率+自定义阈值。

## 图表核查

两张外部图于 2026-11-11 从官方 **1.9** 路径（`https://scikit-learn.org/1.9/_images/`，HTTP 200）抓取，入库 `raw/assets/sklearn-lda-qda/`，**逐张目视核查**。剪藏只有外链，不含图内信息。

| 图 | 文件 | 尺寸 | 行号 | SHA-256（前 12 位） | 核查结论 |
| --- | --- | --- | --- | --- | --- |
| LDA vs QDA 决策面 | `sphx_glr_plot_lda_qda_001.png` | 800×1200 | L14、L16 | `268cd8556935` | 3 行 × 2 列，行标签为 "Data with fixed and spherical covariance" / "with fixed covariance" / "with varying covariances"；每格画类条件协方差椭圆（黑）与决策面 |
| shrinkage 对比 | `sphx_glr_plot_lda_001.png` | 640×480 | L105、L143 | `e66714782aea` | 纵轴 Classification accuracy，横轴 `n_features / n_samples`（0–3.7）；三条线：LDA（黄）、LDA+Ledoit Wolf（蓝虚线）、LDA+OAS（红点线） |

图内读数（**本库目视所得，来源未给任何数字**）：

| 观测 | 读数 |
| --- | --- |
| 图 1 第 1 行（球状、等协方差） | LDA 与 QDA 的边界几乎重合，且都近似线性 |
| 图 1 第 2 行（等协方差、非球状） | 两者边界仍几乎重合 |
| 图 1 第 3 行（协方差不等） | LDA 保持直线；QDA 变成双曲线型的两支边界，且两个类椭圆长轴互相垂直 |
| 图 2 LDA（无收缩） | 比值 ≈0 时 ≈0.97；≈0.6 时 ≈0.87；≈0.85 时 ≈0.74；≈1.0 时回弹到 ≈0.77；≈1.45 时 ≈0.82；≈1.9 触到坐标下限 0.65 后**离图** |
| 图 2 LDA + Ledoit-Wolf | 全程 ≈0.90–0.97；比值 ≈1.0 处 ≈0.955，≈2.0 处 ≈0.93（局部小凹），≈3.5 处 ≈0.89 |
| 图 2 LDA + OAS | 全程 ≈0.92–0.97；**在比值 ≳0.6 后一直高于 Ledoit-Wolf**，比值 3.5 处 ≈0.92 vs ≈0.89 |

三条由图得出的判断（**本库综合**）：

1. 图 1 的第 1、2 行直接可视化了「LDA 是 QDA 的特例」：等协方差时 QDA 学不到有用的二次项。所以"QDA 更灵活"不构成"QDA 更好"。
2. 图 2 说明 shrinkage 不是微调：**无收缩 LDA 在 `n_features/n_samples` 越过 ≈1 之后基本失效**，而两个收缩版本到 3.7 都还在 0.89 以上。这与 L97 的定性说法一致，并给出了幅度（≈0.77 → ≈0.95）。
3. OAS 与 Ledoit-Wolf 的差距（≈0.005–0.03）方向与 L101 的说法一致，但这是**单张示例图、单一数据生成过程**，不能当作"OAS 普遍更好"的证据。

## 转换损失与外部核查

### 1. 真正的参考文献整段丢失，替换成 4 段重复正文

剪藏末尾 L121 有 `References` 标题，但 L123–L135 的 `[^1]`–`[^4]` **全部是正文段落的重复**（LDA log 后验、QDA 决策面、LDA 降维、shrinkage），L137–L143 又把 shrinkage 的三段与图重复了一遍。

官方 1.9 页的 References 是 3 条真实文献：

| 编号 | 官方条目（**外部核查所得，非剪藏内容**） |
| --- | --- |
| [1] | “The Elements of Statistical Learning”, Hastie T., Tibshirani R., Friedman J., Section 4.3, p.106-119, 2008. |
| [2] | Ledoit O., Wolf M. *Honey, I Shrunk the Sample Covariance Matrix.* The Journal of Portfolio Management 30(4), 110-119, 2004. |
| [3] | R. O. Duda, P. E. Hart, D. G. Stork. *Pattern Classification* (Second Edition), section 2.6.2. |

**后果**：本库**不能引用任何文献条目**，只能引用本页正文与 [[sklearn-linear-models]] 里同一批文献的转述。想核实 Ledoit-Wolf 的 lemma 与 OAS 的推导，必须另找原文（列入待解问题）。

### 2. 4 处引用标记被吞，留下断句

| 行号 | 剪藏现状 | 官方 1.9 原文 | 标记目标 |
| --- | --- | --- | --- |
| L75 | The log-posterior of LDA can also be written as: | …can also be written **[3]** as: | Duda/Hart/Stork §2.6.2 |
| L83 | …leading to quadratic decision surfaces. See for more details. | …See **[1]** for more details. | ESL §4.3 |
| L93 | …used in the `transform` method. See for more details. | …See **[1]** for more details. | ESL §4.3 |
| L97 | …following the lemma introduced by Ledoit and Wolf. | …by Ledoit and Wolf **[2]**. | Ledoit & Wolf 2004 |

### 3. 数学噪声

公式全部被转成 `\left(\right.` 形式的伪 LaTeX（L37、L45、L55、L70），`Cst` 被拆成 `C s t`，$\Sigma^{-1}$ 被拆成 `\Sigma_{k}^{- 1}`（L55）。语义可辨，但**直接复制会出错**；本库在 wiki 页统一改写为规范 LaTeX。

`> [!note]`（L60–L63）是转换器把官方 `.. note::` 指令转成 Obsidian callout 的结果，不是抓取源的原文形态。

### 4. 用户指南没有覆盖、但 API 页有的参数（外部核查）

这一节全部**不属于剪藏内容**，取自官方 API 页（`sklearn.discriminant_analysis.*`，2026-11-11 核查）：

| 参数 | 默认 | 行为 | 对本库的意义 |
| --- | --- | --- | --- |
| `priors`（两者） | `None` | `None` 时用训练集类比例；可显式给 `(n_classes,)` | 重采样后阈值会漂移（见关键主张 7） |
| `reg_param`（仅 QDA） | `0.0` | $S^2\leftarrow(1-\rho)S^2+\rho I$，$S^2$ 是类协方差 | **与 `shrinkage` 是两个不同旋钮**：终点是单位阵，而 `shrinkage=1` 的终点是"对角方差" |
| `store_covariance`（两者） | `False` | LDA：`svd` 下不存 `covariance_`，其它求解器总是存；QDA：`covariance_` 需显式开启 | 想检查协方差形状就必须开，否则属性不存在 |
| `tol`（两者，语义不同） | `1e-4` | LDA：`svd` 下判定奇异值是否显著的秩阈值；QDA：**只控制秩亏警告，不影响预测** | 同一名字两种含义 |
| `solver`（两者） | `'svd'` | LDA：`svd`/`lsqr`/`eigen`；QDA：`svd`/`eigen` | 默认的 `svd` 不能配 `shrinkage`（见关键主张 5） |
| `n_components`（仅 LDA） | `None` | 默认 `min(n_classes-1, n_features)`，上限同值 | 见关键主张 2 |

### 5. 版本判定方法（可复核）

对官方 1.6 / 1.7 / 1.8 / 1.9 页做正文逐词比对（2026-11-11，`scikit-learn.org/<ver>/modules/lda_qda.html`）：**1.8 与 1.9 之间唯一一处实质差异**就是 QDA 的 SVD 推导。

| 特征 | 1.6 | 1.7 | 1.8 | 1.9 / stable | 剪藏 |
| --- | --- | --- | --- | --- | --- |
| QDA 协方差写作 $\frac{1}{n-1}X_k^{T}X_k$ | 是 | 是 | 是 | 否 | 否 |
| QDA 协方差写作 $\frac{1}{n}X_k^{T}X_k$ | 否 | 否 | 否 | 是 | **是**（L115） |
| 上标 `X^T`（大写 T） | 否（`X^t`） | 否 | 是 | 是 | 是 |
| 8 个 `1.2.x` 标题 | 是 | 是 | 是 | 是 | 是 |

结论：**1.9.x**（1.9.0 与 1.9.1 在本页不可区分）。旁证：QDA API 页说明 `covariance_` 用「maximum likelihood (biased)」估计，1.9 的改动是把 user guide 对齐实现（**本库综合**）。

## 局限

- **无数字、无基准**：全页没有任何准确率、耗时、维度规模。唯一量化信息在图 2 里，且未给生成该图的数据规模与随机种子。
- **文献层缺失**：3 条参考文献全丢（见转换损失第 1 条），本页所有「Ledoit-Wolf 定理」「ESL §4.3」都只能停在"文档这么写"。
- **不覆盖 API 参数**：`priors`、`reg_param`、`store_covariance`、`tol` 一个都没提。仅靠本节会漏掉不平衡数据与小样本最实用的两个开关。
- **不提假设失效怎么办**：数据非高斯、类协方差随特征高度相关、特征强共线时 LDA/QDA 的表现，本页没有任何说明；也没有与 logistic 回归（判别式）的对照。
- **不覆盖多类策略细节**：QDA 多类时是直接按 K 类后验建模，没有 one-vs-rest 的讨论；`coef_` 在二类与多类下的形状差异（API 页：`(n_features,)` 或 `(n_classes, n_features)`）也未提。

## 待解问题

1. **`shrinkage='auto'` 与分类准确率的关系**：Ledoit-Wolf 的解析解最小化的是协方差 MSE（见本页关键主张 5）。在什么条件下这个目标与分类准确率一致？需要用图 2 的生成过程做扫描验证。
2. **`reg_param`（QDA，终点 $I$）与 `shrinkage`（终点对角方差/$\mu I$）的实际差异有多大**？文档没有对照，两者终点不同、尺度依赖也不同，值得实验。
3. **LDA/QDA 对特征缩放真的不敏感吗？** 纯生成式模型理论上对仿射变换等变，但 `shrinkage`/`reg_param` 的收缩目标（对角阵、单位阵）**不是**仿射等变的 → **假设**：一旦启用收缩，结果就会依赖特征尺度。文档完全未讨论，需实验确认。
4. **`priors` 偏离经验比例时决策面如何移动**？文档只给公式，没有给"改先验 = 改阈值"的等价关系。这是不平衡数据上最实用的推论，但需要自己推。
5. **`svd` 求解器下 `tol=1e-4` 的秩截断在什么数据上会真的丢维度**？L115 只说"不显式算协方差"，外部核查说 `tol` 是奇异值阈值——两者组合的具体行为没有说明。
6. **与 logistic 回归的对照**：ESL §4.3（[1]，缺失）正是讨论这个的，本库无来源，需要另找材料。

## 行号索引

| 主题 | 位置 |
| --- | --- |
| 两个分类器 + 「无需调参」卖点 | L10–L12 |
| LDA/QDA 决策面示例图与说明 | L14–L16 |
| 监督降维、输出维 < 类数、`n_components` 不影响 fit/predict | L24–L26 |
| 贝叶斯后验公式 | L34–L40 |
| 类条件高斯密度 | L42–L48 |
| QDA 的 log 后验 | L52–L58 |
| 对角协方差 ⇒ GaussianNB | L60–L63 |
| LDA = 共享协方差；Mahalanobis 解释 | L67–L73 |
| `coef_` / `intercept_` 的来源 | L75–L81 |
| 线性 vs 二次决策面（含丢失的 `[1]`） | L83 |
| 类均值落在 K−1 维仿射子空间；sphering | L87–L89 |
| 投影到 H；再降到 L 维 = 对类均值做 PCA（含丢失的 `[1]`） | L91–L93 |
| shrinkage 动机、`'auto'`、solver 约束 | L97 |
| shrinkage 端点 0/1 的语义 | L99 |
| OAS 的 MSE 更小 / 准确率更高 | L101 |
| `covariance_estimator` 的接口要求 | L103 |
| shrinkage 示例图 | L105–L109 |
| log 后验依赖先验、均值、协方差 | L113 |
| `svd` 默认、不能用 shrinkage、QDA 的 SVD 推导、LDA 两次 SVD | L115 |
| `lsqr`：只分类、显式算 Σ、解而不求逆 | L117 |
| `eigen`：LDA 的 scatter 比 / QDA 的逐类分解 | L119 |
| 伪 References（真实文献缺失） | L121–L146 |

## 关联页面

- 工具：[[scikit-learn]]（`discriminant_analysis` 模块表面、默认值与版本边界）
- 概念：[[gaussian-discriminant-analysis]]（生成式分类器的几何与假设：Mahalanobis、嵌套关系、参数增长）
- 概念：[[regularization-and-sparsity]]（系数惩罚 vs 协方差收缩两个正则面）
- 实践：[[linear-models-as-baselines]]（生成式基线的定位与"什么时候先上 shrinkage"）
- 同批来源：[[sklearn-linear-models]]（判别式一侧；两份剪藏的版本判定可互证：线性模型页面判到 ≥1.8，本节判到 1.9.x）
- 研究：本来源暂未产生智能体层问题（[[automated-research-open-questions]] 未新增条目）
