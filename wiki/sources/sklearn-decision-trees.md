---
title: "scikit-learn 用户指南 1.10「Decision Trees」：递归划分、贪心最优性、缺失值与代价复杂度剪枝（stable 文档剪藏）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - decision-tree
  - cart
  - pruning
  - reference-doc
source_path: raw/articles/sklearn/1.10. Decision Trees.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.10「Decision Trees」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 1.10. Decision Trees（scikit-learn 用户指南第 1.10 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；12 段代码块（doctest）、6 张图（5 张 PNG + 1 张 SVG）、11 个行间公式块、1 张表格 |
| 原文 | <https://scikit-learn.org/stable/modules/tree.html>（剪藏 frontmatter `source`，未钉版本；2026-11-11 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/1.10. Decision Trees.md`（389 行、30,746 字节、正文约 3,877 词；**未做任何回填**，且**末行无换行**） |
| SHA-256 | `9893b4855443cc6ce99ab5d6e1ba99e1e14f9cbfb1aa8acc8e3caa0071e3d601` |
| 完整性 | **3 个内容折叠块全丢（146 + 224 + 160 = 530 词散文）**、2 条参考文献（`[1]` Raschka、`[BRE]` Breiman）、4 处页内锚点/引用标记；公式有转换噪声。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.9.x（可进一步定为 1.9.1）**，三条独立证据见「版本判定」 |
| 插图 | 6 张，已入库 `raw/assets/sklearn-decision-trees/`（全部目视核查，读数见表） |
| 代码 | 12 段代码块：二分类 toy（L43–L47）、`predict`（L53）、`predict_proba`（L62）、iris 拟合（L71–L76）、`plot_tree`（L82）、`export_graphviz` 两段（L91–L94、L98–L104）、`export_text`（L112–L119）、回归 toy（L143–L148）、缺失值三段（L330–L336、L340–L347、L351–L358） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-decision-trees]], L221`。凡超出剪藏内容的部分（API 默认值、版本史、运行时行为）都显式标注「外部核查」或「本机实测」。

> [!warning] 本文件未被修改
> 与 [[sklearn-svm]]、[[sklearn-naive-bayes]] 不同，本次**没有取得所有者对 `raw/` 回填的授权**，因此剪藏保持抓取时的原始字节（sha256 `9893b485…`，含末尾缺换行）。丢失内容只在「转换损失」一节里按官方 1.9.1 页逐句记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节引入本库第七个模型族：**递归划分（recursive partitioning）**。它与前六个族的关系不是"又一个分类器"，而是**换了一种假设形式**——不再假设一个全局函数（线性/二次/高斯似然/核映射/邻域），而是把特征空间切成轴对齐的矩形，在每个矩形里给一个常数（L10 的 "piecewise constant approximation"；L31 复述）。

对本库的意义有五层：

1. **第一个「模型 = 一组 if-then-else 规则」的族**。它的容量不由参数个数决定，而由**树的形状**决定（深度、叶子数、每个叶子的样本数）。因此它同时是第一个"超参比参数多"的模型：参数在拟合中产生，容量在**拟合前**由五六个结构旋钮规定（L213–L225）。
2. **第一个把「算法最优性」写成明文限制的族**。来源直接说最优树是 NP-complete，实践算法都是贪心的，因此**不能保证全局最优**（L32）。这句话解释了为什么这一族需要随机化与集成（L32、L260），也解释了为什么"调树"常常是在跟一个不稳定的优化过程打交道（L30）。
3. **第一个「解释性」有两个不同出口的族**：可视化（L18、L79–L127、L218）与后面的特征重要性（**本节完全没提**，见「局限」）。来源把它叫 white box（L23），但 white box 不等于"解释就是因果"。
4. **第一个原生支持缺失值的族**（L318–L366，另见 [[sklearn-nearest-neighbors]] 里"没有缺失值概念"的对照）。它不用插补，而是在每个候选阈值上**同时评估 NaN 走左还是走右**（L324），预测时按训练时学到的方向走。
5. **第一个带「显式结构惩罚项」的族**：代价复杂度剪枝 $R_\alpha(T)=R(T)+\alpha|\tilde T|$（L373）罚的是**叶子个数**，与 Lasso 罚系数、Ridge 罚平方和是同一类"用惩罚换简单"的写法，但作用对象是模型结构。已折入 [[regularization-and-sparsity]] 的第六个正则化面。

## 文档结构

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 非参数 + 分段常数；8 条优势、6 条劣势（含 NP-complete、不稳定、无法外推、XOR/parity、类别偏置） | L10–L34 |
| 1.10.1 Classification | `DecisionTreeClassifier` 的输入；`predict` 的平票规则；`predict_proba` 的语义；iris 三件套（`plot_tree` / `export_graphviz` / `export_text`） | L36–L132 |
| （折叠块，已丢）Alternative ways to export trees | Graphviz 安装、审美参数、`export_text` 的定位 | 官方页；剪藏只剩 L88 标题残迹 |
| 1.10.2 Regression | `DecisionTreeRegressor`；y 为浮点 | L134–L154 |
| 1.10.3 Multi-output | 叶子存 $n$ 个值 + 准则取平均下降；`predict_proba` 返回 list；两图 | L156–L186 |
| 1.10.4 Complexity | 训练/推理复杂度表；每节点 $O(d\,n\log n)$；缓存技巧；推理 $O(\text{depth})$ | L188–L211 |
| 1.10.5 Tips on practical use | 10 条工程建议（`max_depth=3` 起步、`min_samples_leaf=5` 起步、权重、float32、稀疏 csc/csr） | L213–L225 |
| 1.10.6 Tree algorithms | ID3 / C4.5 / C5.0 / CART 的差异（**折叠块已丢**，只剩 CART 一句） | L227–L231 |
| 1.10.7 Mathematical formulation | 递归划分、$G(Q_m,\theta)$、`splitter` 两档、三个停止条件、Gini/LogLoss/MSE/Poisson/MAE | L233–L316 |
| 1.10.8 Missing Values Support | 训练时 NaN 双评估；预测三条规则；ExtraTree 的随机分配 | L318–L366 |
| 1.10.9 Minimal Cost-Complexity Pruning | $R_\alpha$、$\alpha_{\text{eff}}$、weakest link、`ccp_alpha` | L368–L389 |

## 关键主张

### 1. 非参数 ≠ 无超参（来源论述 + 本库综合）

L10 的定义是"**non-parametric** supervised learning method"，同一段末句补上机制："A tree can be seen as a piecewise constant approximation"（L31 在劣势清单里复述了同一句）。

**本库综合**：在 [[nearest-neighbors]] 里，"非参数"等价于"参数个数为零，模型就是训练集"；在树里，非参数只意味着**参数个数不预先固定**（随数据长出来），而**容量在拟合前就由结构旋钮锁定了**。两者的工程后果完全相反：

| | 最近邻 | 决策树 |
| --- | --- | --- |
| 参数个数 | 0 | 拟合后产生（每节点一个特征+阈值） |
| 模型体积 | = 训练集 | = 叶子数 × 常数 |
| 训练阶段 | 只建索引 | 真的在搜索切分（贪心） |
| 容量旋钮 | $k$ / $r$ | `max_depth`、`min_samples_leaf`、`min_impurity_decrease`、`max_leaf_nodes`、`ccp_alpha` |

所以"非参数"只说明**假设空间随数据增长**，不说明"不用调参"。

### 2. 优势与代价是一一对应的（来源论述）

来源给了 8 条优势（L16–L25）与 6 条劣势（L27–L34）。它们不是两列清单，而是三组对偶：

| 优势 | 对偶的代价 |
| --- | --- |
| 简单、可解释、可可视化（L18） | 可解释的粒度是"这个矩形"，不是"因果关系"（L23 只主张布尔可读） |
| 几乎不用预处理（L19） | 分段常数 → **不能外推**（L31），且预测不平滑 |
| 推理是对数时间（L20） | 只在**平衡**树成立；见「分歧」第 1 条 |
| 能吃数值 + 类别、多输出（L21–L22） | 本实现**不支持类别变量**（L21 后半句） |
| 白盒、可用统计检验验证（L23–L24） | 不稳定：数据小扰动 → 完全不同的树（L30） |
| 假设被违反时也能跑（L25） | 类别不平衡会造出偏置树（L34） |

再加两条只能算"算法层面"的限制：最优树 NP-complete（L32）、XOR/parity/multiplexer 学不动（L33）。

**本库综合**：L32 与 L33 是同一件事的两面——贪心只能做局部最优，而奇偶类概念恰好要求"两个特征联合看"，单变量轴对齐切分无法用一次切分表达。这直接给出两条工程结论：(a) 想把 XOR 型结构学出来，要么靠多棵树（集成），要么先做特征交叉；(b) 树对**单调变换**不变（因为只比较大小），所以缩放无用，但对**旋转**敏感（轴对齐）。

### 3. 两级输出：类别、概率、平票（来源论述 + 本机实测）

- 分类接口与其它分类器相同：`X` 稠密或稀疏、`y` 整数（L40）。
- **平票取 `classes_` 里索引最小者**（L57 的 "lowest index amongst those classes"）。
- **`predict_proba` = 叶内该类样本比例**（L59）。

**本机实测（scikit-learn 1.6.1）**：同叶内两类各 2 个样本时 `predict_proba = [[0.5, 0.5]]`，`predict` 返回 `classes_` 中靠前的那个——`y=['b','a']` 时 `classes_=['a','b']`，返回 `'a'`（不是 `'b'`）。所以"最低索引"指的是**排序后的类标签位置**，不是原始编码的数值，也不是样本出现顺序。

**外部核查（1.9.1 API 页 + 本机实测）**：`predict_proba` 在给了 `sample_weight` / `class_weight` 时返回的是**加权**比例，不再是"样本个数之比"。本机用三个样本 `y=[0,1,1]`、不可分：默认 `predict_proba=[[0.333, 0.667]]`，`class_weight='balanced'` 时变成 `[[0.5, 0.5]]`。L59 的措辞在加权场景下会误导。

### 4. 回归的三个准则，以及一个被版本钉死的成本数字（来源论述 + 外部核查）

L292–L316 给出三个准则，差别不只在损失，还在**叶子输出什么**：

| 准则 | 叶子预测值 | 适用 | 成本备注 |
| --- | --- | --- | --- |
| MSE（`squared_error`，默认） | 叶内均值 $\bar y_m$ | 通用 | 基准 |
| Poisson deviance（`poisson`） | 叶内均值 $\bar y_m$ | 计数/频率；**必须 $y\ge0$**（L308） | 实现最小化**半** deviance（L308） |
| MAE（`absolute_error`） | 叶内**中位数**（L310–L314） | 离群点稳健 | "**3–6× slower** … as of version 1.8"（L316） |

**外部核查（官方 1.9.1 API 页 + 1.8/1.9 whats_new）**：这句"as of version 1.8"有明确出处——1.8 把 MAE 的拟合从 $O(n^2)$ 改成 $O(n\log n)$（`whats_new/v1.8`：*"now runs much faster: O(n log n) complexity against previous O(n^2)"*），1.8 还修了 MAE "sometimes make sub-optimal splits" 的缺陷。**本机实测（1.6.1，早于该优化）**：3,000×20 的随机回归上 MSE 约 0.044 s、MAE 约 1.03 s，即 **≈23×**，而不是 3–6×。**结论**：L316 这个倍数不能跨版本引用；1.8 之前引用它会把成本低估约 4–8 倍。

另外，1.9 弃用了 `criterion="friedman_mse"`（官方 API 页 "Changed in version 1.9"），而剪藏的回归准则一节只剩三项——两者一致。

### 5. 多输出：叶子存向量，准则取平均下降（来源论述）

L156–L170：多输出问题的"笨办法"是每个输出训一个模型（L160）；树可以用一个模型同时预测 $n$ 个输出，改动只有两条：

- 叶子存 $n$ 个值而不是 1 个（L164）；
- 切分准则取**所有输出的平均下降**（L165）。

收益给了两条：训练更快（只有一个估计器），泛化可能更好（输出之间相关的信息可以共享）（L160）。接口后果：`predict` 输出 $n$ 个值，`predict_proba` 输出 **list of arrays**（L170，本机实测 1.6.1 确认为 `list`）。

**本库判断（未验证）**：L165 的"平均"是在**原始量纲**上取的——若各输出方差差几个数量级，平均下降会被大方差输出支配。这与 [[linear-models-as-baselines]] 里"惩罚项与尺度耦合"是同类问题，但来源没有任何提示。见「待解问题」。

### 6. 复杂度：三个不同的量级，别混用（来源论述）

L190–L195 的表（平衡二叉树的最坏情况）：

| `splitter` | 总训练成本 | 总推理成本 |
| --- | --- | --- |
| `'best'`（默认） | $O(n_{\text{features}}\,n_{\text{samples}}^2\log n_{\text{samples}})$ | $O(\log n_{\text{samples}})$ |
| `'random'` | $O(n_{\text{features}}\,n_{\text{samples}}^2)$ | $O(\log n_{\text{samples}})$ |

L197–L203 再给"**每个节点**"的成本 $O(d\,n\log n)+O(d\,n)$，并解释两项分别是"排序"与"线性扫描候选切分点"，后者在 `'best'` 下是次项可以丢掉（L203）。L205 说明把每节点成本按 $n_{\text{nodes}}=n_{\text{leaves}}-1$ 求和后最坏是 $O(n_{\text{samples}})$ 个节点。L207 说实现用缓存避免重复排序，于是总时间就是 $O(d\,n\log n)$（该句末尾的 `[1]` 引用**在剪藏中丢失**）。L209 说推理成本**与 splitter 无关**，只取决于深度 $O(\text{depth})$，平衡树时 $\text{depth}=O(\log n)$。

**本库综合**：这一节里其实有三个量级并存——(a) 每个节点、(b) 整棵树、 (c) 单次推理。L20 的"推理是对数时间"只等于 (c) 在平衡树上的特例，不是一般结论（见「分歧」）。

### 7. 实用建议：10 条里有一半是结构性约束（来源论述）

L213–L225 的清单，按可控性分三类：

| 类型 | 条目 | 位置 |
| --- | --- | --- |
| 数据形状 | 特征多容易过拟合；先做降维（PCA/ICA/特征选择） | L215–L216 |
| 拟合过程 | 训练时可视化；`max_depth=3` 起步再加深；`min_samples_split`/`min_samples_leaf`；`min_samples_leaf=5` 起步（分类类别少时 `=1` 常常最好） | L218–L220 |
| 权重与类别 | 先平衡数据；有 `sample_weight` 时用 `min_weight_fraction_leaf` 这类**权重感知**的准则 | L222–L223 |
| 数值实现 | 内部一律 `np.float32`（会复制一次）；极稀疏时 `fit` 前转 `csc_matrix`、`predict` 前转 `csr_matrix` | L224–L225 |

其中 L221 的注解是本节最容易读漏的一句：**`min_samples_split` 直接数样本、与 `sample_weight` 无关**（"a node with m weighted samples is still treated as having exactly m samples"）。**本机实测（1.6.1）**：6 个样本、权重 `[1,1,1,1,1,100]`，`min_samples_split=6` 仍会切分，`=7` 才停——完全按未加权样本数判断。

### 8. 贪心切分与两种 splitter（来源论述）

L235–L253 给出形式化：对节点 $m$ 的每个候选切分 $\theta=(j,t_m)$，把数据分成 $Q_m^{\text{left}}$（$x_j\le t_m$）与其余，用不纯度 $H$ 的**加权和**打分

$$G(Q_m,\theta)=\frac{n_m^{\text{left}}}{n_m}H\!\left(Q_m^{\text{left}}\right)+\frac{n_m^{\text{right}}}{n_m}H\!\left(Q_m^{\text{right}}\right)$$

再取 $\theta^*=\arg\min_\theta G$（L249–L253）。`splitter` 两档（L255–L260）：

- `'best'`（默认）：在所有特征 × 所有候选阈值（排序后相邻不同值的中点）上**穷举**，取精确最小（L257）；
- `'random'`：每个特征只**采样一个**随机阈值，是贪心搜索的随机近似，省时间（L258；该句的 "see …" 锚点文字丢失）。

L260（1.9 新增句）：`ExtraTreeClassifier` / `ExtraTreeRegressor` 默认 `splitter='random'`，与对应决策树**共享同一实现**，只是默认参数不同。

L262–L268 给出三个停止条件：达到 `max_depth`、$n_m<$ `min_samples_split`、不纯度下降 < `min_impurity_decrease`；其余条件在各自 docstring 里（L268，外部核查：`min_samples_leaf`、`min_weight_fraction_leaf`、`max_leaf_nodes`、`ccp_alpha`）。

### 9. 缺失值：不是插补，是"每个阈值学一个方向"（来源论述 + 本机实测）

训练时（L320–L324）：对非缺失数据上的每个候选阈值，**分别评估** NaN 全走左、全走右两种情况，取更好的那个。预测时（L326–L359）三条规则：

1. 默认：按训练时该切分学到的方向走（L328；doctest `X=[0,1,6,NaN]`、`y=[0,0,1,1]` → `predict(X)` 得到 `[0,0,1,1]`）。
2. 两边的准则评估**相同**时，平票**走右**（L338；doctest 返回 `array([1])`）。
3. 训练时该特征**没见过** NaN：预测时走**样本更多**的子节点（L349；doctest 返回 `array([1])`，`predict_proba=[[0,1]]`）。

ExtraTree 一族（L361–L366）：切分时随机选阈值，非缺失值按阈值分左右，**NaN 随机分**到左右；每个特征都重复，取其中最好的切分。预测阶段规则与决策树相同（L363–L366）。

**本机实测（1.6.1）**：三段 doctest 的输出逐字复现，规则 1–3 均成立。

**外部核查（1.9 whats_new）**：1.9 起 `criterion="absolute_error"` 也支持**稠密** X 的缺失值（#32119），并支持 `monotonic_cst` + 缺失值组合；同时修了"某特征在某些 NaN 模式下被误判为无 NaN"的 bug（#32193）。**因此"哪些准则支持缺失值"是随版本变化的**，而 1.9 的剪藏把 1.8 里那句准则清单删掉了——见「转换损失 B」。

### 10. 代价复杂度剪枝：唯一带惩罚项的结构正则（来源论述 + 外部核查）

L368–L378：最小代价复杂度剪枝（minimal cost-complexity pruning）以 $\alpha\ge0$ 参数化

$$R_\alpha(T)=R(T)+\alpha\,|\tilde T|$$

其中 $|\tilde T|$ 是终端节点数（L373）。**关键差异**：经典定义里 $R(T)$ 是误分类率，而 scikit-learn 用终端节点的**样本加权不纯度**（L376）。单节点成本 $R_\alpha(t)=R(t)+\alpha$；节点的有效 $\alpha$ 定义为子树与节点成本相等处

$$\alpha_{\text{eff}}(t)=\frac{R(t)-R(T_t)}{|T|-1}$$

$\alpha_{\text{eff}}$ 最小的非终端节点是"最弱环节"，被剪掉；当剪枝后树的最小 $\alpha_{\text{eff}}$ 超过 `ccp_alpha` 时停止（L378）。

**外部核查**：`ccp_alpha` 默认 `0.0`，即**默认不剪枝**（官方 1.9.1 API 页："By default, no pruning is performed"）；配套的 `cost_complexity_pruning_path` 可一次给出整条 $\alpha$ 路径（剪藏完全没提这个 API）。

**本库综合**：这是本库第一种**罚模型结构**的正则（罚叶子数），与罚系数（Lasso/Ridge）、罚协方差（shrinkage）、罚频率（平滑）、罚松弛（SVM 的 $C$）、罚原型（`shrink_threshold`）并列。它也是本节唯一一个"先长满再剪"的**后剪枝**手段，与前剪枝的四个旋钮不是同一层（见 [[decision-trees-selection-and-tuning]]）。

## 版本判定

**结论：1.9.x，可进一步定为 1.9.1。** 三条独立证据：

| # | 证据 | 判据 |
| --- | --- | --- |
| 1 | 1.8→1.9 正文差异 | 官方 1.8 页**没有** L260 那句"ExtraTree… share the same implementation"，1.9 页**新增**了它；剪藏有这句 |
| 2 | 1.8→1.9 删除的句子 | 官方 1.8 页有 "The criterion supported when there are missing values are 'gini', 'entropy', 'log_loss' … 'squared_error', 'friedman_mse', 'poisson'"；1.9 页删除；剪藏**没有**这句 |
| 3 | `/stable/` 指向 | 2026-11-11 抓取的 `stable/modules/tree.html` 与 `1.9/modules/tree.html` **逐行相同**（文本级 diff 为空），页标题为 "scikit-learn 1.9.1 documentation" |

旁证（外部核查，不依赖剪藏）：`whats_new/v1.9` 的 tree 段落正好解释了上面两处文本改动——`friedman_mse` 被弃用（所以准则清单从文档里消失）、`absolute_error` 开始支持稠密缺失值（所以"哪些准则支持 NaN"的旧说法不再准确）。

**证据强度：强**（比 [[sklearn-svm]] 的"1.7–1.9.1 无改动"更硬：这里是**有改动**且方向唯一）。仍不能排除"1.9.1 之后的小版本又改了别处"，但本节所有功能性结论都已按 1.9.1 核对。

## 转换损失

官方 1.9.1 页本节共 **3 个内容折叠块**（其余 8 个 `<details>` 是侧栏导航），剪藏**一个都没保留内容**：

| # | 折叠块 | 官方词数 | 剪藏里的残迹 | 丢了什么 |
| --- | --- | --- | --- | --- |
| A | *Alternative ways to export trees* | 146（散文） | L88 只剩一行标题残迹 `Alternative ways to export trees [#](…)`，其后是 3 段代码 + 3 张图 | Graphviz 的安装方式（conda/pip）、"导出的是上图那棵 iris 树"、`export_graphviz` 的审美参数（`filled`/`rounded`/`special_characters`）、Jupyter 内联渲染、以及 `export_text` 的定位句（"不需要外部库、更紧凑"） |
| B | *Various decision tree algorithms* | 224 | L229–L231 只剩问题句 + "sklearn 用优化版 CART"一句 | ID3（1986, Quinlan）、C4.5（去掉类别限制、转规则集、按规则精度排序、剪规则前件）、C5.0（专有许可、更省内存）、CART（支持回归、不产生规则集、二叉）的完整对照 |
| C | *Shannon entropy* | 160 | L286–L290 只有 Gini 与 Log Loss 两个公式 | "熵准则 ≡ 最小化 log loss"的推导：$\mathrm{LL}(D,T)=-\frac1n\sum\sum_k I(y_i=k)\log T_k(x_i)$、叶内概率为常数、以及 $\mathrm{LL}(D,T)=\sum_{m\in T}\frac{n_m}{n}H(Q_m)$ |

另有四类**非折叠块**损失：

| 类型 | 具体 | 位置 |
| --- | --- | --- |
| 参考文献 | `[1]` S. Raschka, *Stat 451: Machine learning lecture notes*, UW-Madison (2020)（整条丢失，只剩 L211 的空标题）；`[BRE]` Breiman et al., *Classification and Regression Trees*, 1984（丢失，L384 的 References 从 4 条变 3 条） | L211、L384–L389 |
| 引用标记 | L207 的 `[1]`、L370 的 `[BRE]`（"described in Chapter 3 of."） | L207、L370 |
| 页内锚点文字 | L19 "支持**缺失值**"、L21 "见**算法**一节"、L258 "（见**复杂度**）"——三处都只留下断句 | L19、L21、L258 |
| 公式噪声 | `n_{f e a t u r e s}`、`\left(\right.`、`m e d i a n`、`\overset{\sim}{T}` 等 | 全篇 |

**量化**：丢掉的散文 146+224+160 = **530 词**，加两条文献约 40 词。以官方页本节（含代码与公式 token）约 4,035 词为分母计 ≈ **13%**；若只按散文计，比例更高（这一节本来就短，三个折叠块覆盖了"导出树""算法谱系""熵=log loss"三个独立主题，全部丢失）。

**结构性后果**（本库判断）：丢的正好是三类"读起来像背景知识"的内容，所以剪藏在**接口层面**仍然完整（所有 API 名、所有公式主干都在），但失去了：

1. 熵准则与 log loss 的**等价性**（这是 L286 那个"Log Loss or Entropy"标签的唯一解释）；
2. ID3/C4.5/C5.0 的区别（L227 的标题因此悬空，读者无法知道 CART 与其他算法差在哪）；
3. `export_text` 的定位（L111 的代码块前没有任何说明，读者不知道它为什么存在）。

## 图表核查

6 张图中 5 张 PNG 于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），SVG 同源；已入库 `raw/assets/sklearn-decision-trees/`，**逐张目视核查**。

| 文件 | 尺寸/字节 | SHA-256（前 16） | 用在哪 |
| --- | --- | --- | --- |
| `sphx_glr_plot_tree_regression_001.png` | 640×480 / 30,562 | `ffbef8e01320a46c` | L14（引言）与 L136（回归），同一张图用了两次 |
| `sphx_glr_plot_tree_regression_002.png` | 640×480 / 20,621 | `6e0d665146d1866a` | L174（多输出） |
| `sphx_glr_plot_iris_dtc_002.png` | 640×480 / 38,558 | `bf2292863bd4a18e` | L86（`plot_tree`） |
| `sphx_glr_plot_iris_dtc_001.png` | 640×480 / 79,203 | `d373c0b1ade1fba4` | L109（决策面） |
| `iris.svg` | 22,002 字节 | `6fd6e790c1a510c3` | L107（Graphviz 导出） |
| `sphx_glr_plot_multioutput_face_completion_001.png` | 1000×1130 / 95,431 | `33839c9d4aecf7e8` | L178（人脸补全） |

**读数（只记录看到的东西）**：

1. **图 1（回归，L14/L136）**：标题 "Decision Tree Regression"；蓝线 `max_depth=2` 只有约 4 个台阶，绿线 `max_depth=5` 有十余个台阶，全部是**水平段 + 垂直跳变**，没有任何斜线。绿线在 $x\approx3.3$ 处直接砸到 $y\approx-1.6$（对准一个离群点）再跳回，$x\approx0.3$ 处有一个上冲；蓝线整条只有约 4 个台阶，在 $x\approx1$–$2.2$ 停在 $y\approx0.7$ 附近。两条线都**没有画到数据范围之外**——图上没有外推段可看。这张图就是 L31"不平滑、不分段连续、不善外推"的直接证据，也是 L29"过深 = 过拟合"的直接证据。
2. **图 3（`plot_tree`，L86）**：整棵 iris 树，**根节点 `X[3] <= 0.8`（$X[3]$ = petal width）、gini = 0.667、samples = 150、value = [50, 50, 50]**；左子纯 setosa（50）；右侧 `X[3] <= 1.75`（gini 0.5，100 个样本）；再下一层 `X[2] <= 4.95`（54，gini 0.168，value [0,49,5]）与 `X[2] <= 4.85`（46，gini 0.043，value [0,1,45]）。最深一层出现 `gini=0.0, samples=1` 的叶子——即**单个样本被单独隔离**，正是 L205 所说"长到每样本一叶"的最坏情况。本机 1.6.1 在 iris 上拟合出的树是 9 叶、深度 5、根 `X[3] <= 0.8`，与图一致。
3. **图 4（决策面，L109）**：6 个特征两两组合的面板，全部是**轴对齐矩形**色块；出现细长条带（相邻两次切分挨得很近），不是平滑边界；setosa 在多数面板里被一条直线/直角干净隔开，versicolor 与 virginica 之间是你中有我的窄带。图上没有错误率数字。
4. **图 2（多输出回归，L174）**：数据点落在一个圆环上；`max_depth=2` 的预测点（蓝）只落在圆环上的几个位置（台阶状），`max_depth=8`（深红）几乎贴着圆环，但有明显的锯齿与个别偏离点。说明多输出树把"两个输出"当成一个二维目标同时逼近。
5. **图 6（人脸补全，L178）**：5 列（true faces / Extra trees / K-nn / Linear regression / Ridge）× 5 行。Extra trees 与 K-nn 两列明显更糊、更"块状"，在部分行的**上下半脸拼接处出现横向痕迹**；Linear regression 列在嘴部周围有明显振铃/鬼影；Ridge 列最平滑。图内没有任何数值指标。
6. **图 5（`iris.svg`）**：Graphviz 导出，带特征名与类名（root 标注 `petal length (cm) ≤ 2.45`，gini 0.6667，samples 150，value [50,50,50]，class setosa）。**这张图是陈旧的**：文件头注释为 `Generated by graphviz version 2.36.0 (20140111)`，且 1.6 / 1.7 / 1.8 / 1.9 四个版本文档提供的 `iris.svg` **字节完全相同**（sha256 前缀均为 `6fd6e790c1a510c3`），说明它是静态资产、不是每次构建重新生成的。它的根切分（petal length ≤ 2.45）与同一页 L120 的 `export_text` doctest（根 = petal width ≤ 0.80）、图 3（根 = `X[3] <= 0.8`）以及本机 1.6.1 的运行时结果**都不一致**——见「分歧」第 4 条。

## 外部核查

以下全部**不在剪藏内**，来源为官方 1.9.1 API 页、`whats_new/v1.8`/`v1.9`，或本机运行时（scikit-learn **1.6.1**，见每条的标注）。

### A. 四个估计器的完整默认值（官方 1.9.1 API 页）

| 参数 | `DecisionTreeClassifier` | `DecisionTreeRegressor` | `ExtraTreeClassifier` | `ExtraTreeRegressor` |
| --- | --- | --- | --- | --- |
| `criterion` | `'gini'` | `'squared_error'` | `'gini'` | `'squared_error'` |
| `splitter` | `'best'` | `'best'` | **`'random'`** | **`'random'`** |
| `max_depth` | `None` | `None` | `None` | `None` |
| `min_samples_split` | `2` | `2` | `2` | `2` |
| `min_samples_leaf` | `1` | `1` | `1` | `1` |
| `min_weight_fraction_leaf` | `0.0` | `0.0` | `0.0` | `0.0` |
| `max_features` | `None` | `None` | **`'sqrt'`** | **`1.0`** |
| `max_leaf_nodes` | `None` | `None` | `None` | `None` |
| `min_impurity_decrease` | `0.0` | `0.0` | `0.0` | `0.0` |
| `class_weight` | `None` | —（无此参数） | `None` | — |
| `ccp_alpha` | `0.0` | `0.0` | `0.0` | `0.0` |
| `monotonic_cst` | `None` | `None` | `None` | `None` |
| `random_state` | `None` | `None` | `None` | `None` |

剪藏的 Tips 一节**从不提默认值**，因此"用默认参数跑一次"在文档里查不到：`max_depth=None` 意味着"长到叶子纯或样本不足"（即默认会过拟合），`ccp_alpha=0.0` 意味着"默认不剪枝"。

### B. 缺失值与准则的版本史（`whats_new/v1.8`、`v1.9`）

- 1.8 页曾有"支持缺失值的准则"清单（`gini`/`entropy`/`log_loss` 与 `squared_error`/`friedman_mse`/`poisson`），1.9 页删除该句。
- 1.9 起 `absolute_error` 支持**稠密** X 的缺失值（#32119），并支持 `monotonic_cst` + 缺失值（#27630）。
- 1.9 弃用 `criterion="friedman_mse"`（API 页 "Changed in version 1.9: Criterion "friedman_mse" was deprecated."）；它在旧版里与 `squared_error` 行为相同。
- 1.9 修了 tree 的 NaN 检测边界 bug（#32193）与缺失值 + Poisson 的不纯度计算（#32119）。

**含义**：剪藏（1.9.x）既不列准则清单，也不说 MAE 现在能吃 NaN。想知道"我的准则能不能直接喂 NaN"，必须查目标版本的 API/发布说明。

### C. 本机运行时复核（scikit-learn 1.6.1）

| # | 问题 | 结果 |
| --- | --- | --- |
| 1 | 缺失值三条规则（L328/L338/L349） | 三段 doctest 输出逐字复现 |
| 2 | 平票取"最小索引"（L57） | `y=['b','a']` → `classes_=['a','b']`，平票返回 `'a'`（排序位置，不是原始编码） |
| 3 | `predict_proba` 是否为加权比例 | 是：`class_weight='balanced'` 把 `[1/3, 2/3]` 改成 `[0.5, 0.5]` |
| 4 | `min_samples_split` 是否忽略 `sample_weight`（L221） | 是：6 样本、权重 `[1,1,1,1,1,100]`，`=6` 仍切、`=7` 才停 |
| 5 | `max_features` 默认差异的实际后果 | 同一数据（200×20）：`DecisionTreeClassifier` 默认 13 叶，`ExtraTreeClassifier` 默认 79 叶（`'sqrt'`） |
| 6 | `np.float32` 内部转换（L224） | 相邻 float64 值 0.30000001 / 0.30000002 在 float32 下**塌成同一个 0.3**：4 个样本只切出 3 叶，`predict(0.300000015)` 返回 1.5（两个目标值的平均）；阈值本身是 float32 舍入后的 0.25000001 / 0.65000001 |
| 7 | MAE 相对 MSE 的成本（L316） | 3,000×20 随机数据：MSE 0.044 s vs MAE 1.03 s ≈ **23×**（1.6.1 尚未含 1.8 的 $O(n\log n)$ 优化） |
| 8 | Poisson 要求 $y\ge0$（L308） | `y=[-1,2]` → `ValueError: Some value(s) of y are negative…` |
| 9 | 多输出 `predict_proba` 返回类型（L170） | `list`，每个元素形状 `(n_samples, n_classes_k)` |
| 10 | iris 全树的根切分 | `X[3] <= 0.8`，9 叶、深度 5；`random_state ∈ {None,0,1,2,3}` 五个取值根切分相同（**因此本库不把 iris 根切分的差异归因于 `random_state`**） |

**版本警告**：本机是 1.6.1，剪藏是 1.9.1。上面第 1、2、4、5、6、8、9 条是**机制层**的，可以认为跨版本成立；第 7 条明确**不跨版本**；第 3 条依赖 `predict_proba` 的加权实现，1.9.1 未实测。

## 分歧

1. **推理复杂度：L20 与 L209 冲突**。L20（优势清单）说"预测成本是对数级"，L209 说"只取决于深度 $O(\text{depth})$"，只在**近似平衡**树时才是 $O(\log n)$。一棵退化到单链的树（每层只切一个样本）推理是 $O(n)$。L20 把平衡树的特例写成了通用优势。
2. **"能处理类别变量"与本实现无关**。L21 前半句是树族的一般性质，后半句立刻说 scikit-learn **不支持**（"does not support categorical variables for now"）。这句 "for now" 至少在 1.6 页与 1.9 页**字面相同**（外部核查：两版该句未变），因此它不是"即将支持"的时间戳，而是一句多年未更新的免责声明。
3. **`predict_proba` 的定义在加权下不准确**。L59 说它是"叶内该类训练样本的**比例**"；本机实测加权的 `class_weight`/`sample_weight` 会改变返回值（见上表第 3 条）。文档没有第二句话解释这件事。
4. **`iris.svg` 是陈旧资产，与同页自己的 doctest 矛盾**。图 5 的根是 `petal length ≤ 2.45`，L120 的 `export_text` 与图 3 的根是 `petal width ≤ 0.80`；该 SVG 在 1.6–1.9 四版文档中字节相同、生成于 2014 年（graphviz 2.36.0）。**本库没有把它读作"1.9 的树长这样"**。补充一个技术事实（本库计算）：这两个候选根切分在 iris 上的加权 Gini **完全相同**（都是 $1/3$），所以差异来自**平票如何处理**，不是"图算错了"——但两张图确实不是同一棵树。
5. **"哪些准则支持缺失值"在三处口径不同**：1.8 页有清单（不含 `absolute_error`）→ 1.9 页删掉清单 → 1.9 的 `whats_new` 说 `absolute_error` 也支持。这是文档演进而不是互相矛盾，但意味着**任何"树天然支持缺失值"的说法都必须带版本与准则**。
6. **"non-parametric" 与"超参很多"的张力**（本库综合，不是来源矛盾）。来源用 non-parametric 描述假设空间（L10），但本节的实用建议几乎全在讲超参（L213–L225）。这与 [[nearest-neighbors]] 的"零参数"叙事相反，两者都自称 non-parametric——所以这个词对工程决策几乎没有信息量。

## 局限

- **本节不覆盖特征重要性**。`feature_importances_`、`permutation_importance`、impurity-based 重要性的偏置问题，在剪藏里**一次都没出现**（grep 为 0）。所以本页只能说"树可可视化"，**不能**说"本节支持重要性解释"。相关讨论在未入库的 1.11 Ensembles 与 5.2 Permutation importance。
- **本节不覆盖 `tree_` 结构与遍历 API**：`apply`、`decision_path`、`get_depth`、`get_n_leaves`、`cost_complexity_pruning_path`、`export_text`/`export_graphviz` 的参数细节，剪藏都没讲（`export_text` 只剩一段代码）。
- **本节不覆盖 `monotonic_cst`（单调约束）**，尽管 1.4 页的签名里已含该参数（外部核查）。剪藏 grep 为 0。
- **本节不覆盖 `max_features`**（grep 为 0），而它正是 `ExtraTree*` 与 `DecisionTree*` 默认行为的关键差异（外部核查：`'sqrt'` vs `None`，`1.0` vs `None`）。
- **`class_weight` 参数在剪藏中完全没有出现**（grep 为 0），只有"先平衡数据"的散文（L222）。把 L222 直接读成"用 `class_weight='balanced'`"是推断，不是来源论述。
- **选型结论极少**。本节不说"什么时候用树而不是线性/最近邻/SVM"，也不给任何基准读数；唯一的量化数字是 MAE 的 3–6×（L316）与复杂度表的量级。
- **未读原始文献**：ID3/C4.5/C5.0 的原始描述（Quinlan 1993 等）在剪藏中被替换成折叠块且已丢失；`[BRE]` Breiman 1984、`[1]` Raschka 2020 两条文献整条丢失，无法核对。
- **未做 1.9.1 运行时验证**：PyPI 上当前可安装的最高版本是 1.6.1（本机环境即是），因此所有"实测"都来自 1.6.1。

## 待解问题

1. **多输出准则的尺度依赖**：L165 的"平均下降"是否等价于逐输出标准化后平均？若输出量纲差异大，是否会退化为主导输出的单目标树？（本库假设，需实验）
2. **`predict_proba` 的校准质量**：叶内频率在叶子样本很少时方差极大；本节没有任何校准论述（对照 [[sklearn-svm]] 的 Platt scaling 与 [[sklearn-naive-bayes]] 的 "bad estimator"）。需要 `sklearn.calibration`（未入库）。
3. **`ccp_alpha` 与 CV 的联合选择**：`cost_complexity_pruning_path` 给出的 $\alpha$ 序列是否可直接用于 `GridSearchCV`？路径是否对 `random_state` 稳定？
4. **前剪枝四旋钮的等价类**：`max_depth`、`min_samples_leaf`、`min_impurity_decrease`、`max_leaf_nodes` 之间存在哪些近似等价关系？文档完全没有讨论它们的交互。
5. **缺失值规则在分布偏移下的后果**：训练时未见 NaN 的特征，预测时被送到"样本更多"的子节点（L349）——这是一个**不基于特征值**的默认方向，它在部署数据漂移时会不会系统性偏向多数类？（本库假设）
6. **`monotonic_cst` 的语义与失效条件**：它与 `splitter='random'`、缺失值、多输出的组合限制是什么？（1.9 whats_new 提到组合支持，但本节不提）
7. **树在高维稀疏数据上的行为**：L215 只说"特征多容易过拟合"，L225 说稀疏时用 csc/csr 更快——没说稀疏时切分搜索的复杂度是否仍是 $O(d\,n\log n)$。
8. **图 5 的陈旧资产是否在别处也出现**：本库已确认 `iris.svg` 在 1.6–1.9 四版完全一致。其他章节的静态图是否同样陈旧，需要逐图核对（本库尚未系统检查）。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 定义（非参数、分段常数） | L10 |
| sine 曲线示例 | L12–L14 |
| 8 条优势（含"缺值支持""见算法"两处断句） | L16–L25 |
| 6 条劣势（过拟合、不稳定、不可外推、NP-complete、XOR/parity、类别偏置） | L27–L34 |
| 分类：toy 例、平票规则、`predict_proba` 语义 | L38–L66 |
| iris + `plot_tree` | L68–L86 |
| 折叠块 A 残迹（导出树） | L88–L127 |
| `export_text` 的 depth-2 树输出 | L120–L126 |
| 回归 toy | L134–L150 |
| 多输出（叶子存 n 值、平均下降、list 输出） | L156–L170 |
| Dumont 参考文献 | L184–L186 |
| 复杂度表 | L192–L195 |
| 每节点成本与两项解释 | L197–L203 |
| 求和到 $n_{\text{nodes}}$；缓存技巧（`[1]` 丢失） | L205–L207 |
| 推理 $O(\text{depth})$ | L209 |
| 空 References（Raschka 丢失） | L211 |
| Tips 10 条 | L213–L225 |
| `max_depth=3` 起步 / `min_samples_leaf=5` 起步 | L218–L220 |
| `min_samples_split` 忽略 `sample_weight` | L221 |
| 权重感知的 `min_weight_fraction_leaf` | L223 |
| float32 / 稀疏 csc-csr | L224–L225 |
| 折叠块 B 残迹（算法谱系） | L227–L231 |
| 数学形式：$G(Q_m,\theta)$ | L233–L253 |
| `splitter='best'` / `'random'` | L255–L260 |
| 三个停止条件 | L262–L268 |
| Gini / Log Loss | L280–L290 |
| MSE / Poisson / MAE | L296–L316 |
| MAE 的 3–6× 备注 | L316 |
| 缺失值：训练机制 | L320–L324 |
| 缺失值：预测三条规则 + doctest | L326–L359 |
| ExtraTree 的缺失值处理 | L361–L366 |
| 代价复杂度剪枝 | L368–L378 |
| References（缺 `[BRE]`） | L384–L389 |

## 关联页面

- 概念：[[decision-trees]]（本节的机制页）、[[regularization-and-sparsity]]（第六个正则化面）、[[nearest-neighbors]]（另一种"非参数"）、[[support-vector-machines]]、[[gaussian-discriminant-analysis]]
- 实践：[[decision-trees-selection-and-tuning]]（前/后剪枝、缺失值、float32 的工程写法）、[[linear-models-as-baselines]]、[[nearest-neighbors-selection-and-tuning]]
- 工具：[[scikit-learn]]（`tree` 模块的估计器矩阵、默认值第 45–55 条、版本边界）
- 同批剪藏：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]、[[sklearn-svm]]、[[sklearn-nearest-neighbors]]
- 未入库的相邻章节：1.11 Ensembles（树集成）、1.13 Feature selection、8.3 Preprocessing（缺失值插补的替代路线）、5.2 Permutation importance
