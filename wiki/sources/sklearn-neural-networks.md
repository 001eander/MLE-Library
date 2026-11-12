---
title: "scikit-learn 用户指南 1.17「Neural network models (supervised)」：MLP 分类/回归、三个求解器与正则化（stable 文档剪藏，未回填）"
type: source
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - neural-network
  - mlp
  - reference-doc
source_path: raw/articles/sklearn/1.17. Neural network models (supervised).md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.17「Neural network models (supervised)」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 1.17. Neural network models (supervised)（scikit-learn 用户指南第 1.17 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；8 个小节、6 个围栏代码块 + 1 个缩进代码块、2 张图、1 个行间公式、1 个 `[!warning]` 提示框、**0 张表** |
| 原文 | <https://scikit-learn.org/stable/modules/neural_networks_supervised.html>（剪藏 frontmatter `source`，未钉版本；2026-11-11 抓取时 `/stable/` 页标题为 1.9.1） |
| 本地文件 | `raw/articles/sklearn/1.17. Neural network models (supervised).md`（163 逻辑行、12,713 字节、1,388 词；`wc -l` 报 162 是因为**末行无换行**；未做任何回填） |
| SHA-256 | `1677e5a963e50cf1f49d373cc904b77cf81330903a8210b35ed741aa4c0babec` |
| 完整性 | **3 个内容 `<details>` 折叠块全丢（622 词）**：优缺点（93 词）、数学表述（460 词）、References（69 词）；正文其余段落逐句对齐。另丢 5 条文献。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.8.0 – 1.9.1**；抓取日 `/stable/` 为 1.9.1，故取 **1.9.1** 作为工作版本（见「版本判定」） |
| 插图 | 2 张，已入库 `raw/assets/sklearn-neural-networks/`，均从 `https://scikit-learn.org/1.9/_images/` 抓取（HTTP 200），见「图表核查」 |
| 代码 | 6 段围栏 doctest：`MLPClassifier` 拟合（L31–L41）、`predict`（L45–L48）、`coefs_` 形状（L52–L55）、`predict_proba`（L61–L65）、多标签（L71–L84）、`warm_start` 循环（L155–L162）；另有 1 段缩进的 `StandardScaler` 示例（L138–L146），**不是 doctest**（直接使用未定义的 `X_train`） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-neural-networks]], L137`。凡超出剪藏内容的部分（API 默认值、实现细节、版本史、丢失折叠块的原文）都显式标注「外部核查」或「官方 1.9.1 页」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `1677e5a9…`）。丢失的 3 个内容块只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库第九个 sklearn 文档来源，也是第一个**「参数化、非凸、迭代训练」的模型族**（[[multi-layer-perceptron]]）。此前八个模块里，线性模型与 SVM 是凸优化，树是贪心但不是梯度迭代，KNN 根本没有训练阶段；MLP 第一次把「随机初始化 + 梯度下降 + 多个局部极小」带进本库。对本库的意义有五层。

1. **它不是「逻辑回归加一点非线性」**。来源的判据是：输入层与输出层之间可以有一个或多个**隐藏层**，每层先做加权线性求和、再过非线性激活（L15、L21）。因此假设空间随 `hidden_layer_sizes` 改变，而不是随特征工程改变。
2. **模型体积第一次由「架构」而不是「训练集」决定**。KNN 的模型就是训练集（[[nearest-neighbors]]），线性模型的体积由特征数决定；MLP 的体积是若干权重矩阵，`coefs_` 的每一项对应一层边界（L23）。这使「模型大小」成为可直接读出的量。
3. **三个求解器对应三种训练契约**。SGD 与 Adam 支持在线/小批量（`partial_fit`），L-BFGS 不支持（L125–L129）；来源还给了「小数据 L-BFGS、较大数据 Adam、调好学习率后 SGD + momentum」的经验顺序（L149），但**默认求解器是 Adam**（外部核查），两者并不总是一致。
4. **缩放是硬前提，不是优化项**。来源用 "highly recommended" 要求缩放，并要求对测试集使用**同一个**缩放（L137）。
5. **剪藏保留了接口，却丢掉了全部「为什么」**。3 个内容折叠块正好是：优缺点（非凸、局部极小、随机初始化导致验证准确率波动）、数学表述（三个输出面的公式、两种损失、反向传播与停止条件）、5 条文献。因此本页必须把「MLP 为什么会因种子而变」归因于**官方丢失折叠块**，而不能只引剪藏正文。

## 文档结构

| 章节 | 解决什么问题 | 行号 |
| --- | --- | --- |
| 警告框 | 不适合大规模应用；无 GPU；更快/更灵活的实现见 Related Projects | L10–L11 |
| 1.17.1 Multi-layer Perceptron | 定义 $f:R^m\to R^o$、隐藏层、`coefs_`/`intercepts_` 的索引契约 | L13–L23 |
| （丢失的 `<details>`）Advantages and disadvantages | 优点（非线性、`partial_fit` 在线学习）与缺点（非凸/局部极小/随机初始化、超参多、对缩放敏感） | 官方页；剪藏无 |
| 1.17.2 Classification | `MLPClassifier`、交叉熵、多类 softmax、多标签 logistic + 0.5 阈值 | L25–L91 |
| 1.17.3 Regression | `MLPRegressor`、输出层恒等、平方误差、多输出回归 | L93–L97 |
| 1.17.4 Regularization | `alpha` 的 L2 正则；正则强度对决策面的影响 | L99–L109 |
| 1.17.5 Algorithms | SGD、Adam、L-BFGS 的更新方式；SGD/Adam 支持在线与小批量，L-BFGS 不支持 | L111–L129 |
| 1.17.6 Complexity | 反向传播的时间复杂度；建议从小网络开始 | L131–L133 |
| （丢失的 `<details>`）Mathematical formulation | 单隐藏层单神经元的 $f(x)$、tanh/logistic/softmax、交叉熵与 MSE 的完整损失、反向传播与梯度下降、停止条件 | 官方页；剪藏无 |
| 1.17.7 Tips on Practical Use | 缩放、`alpha` 网格、三个求解器的经验选择 | L135–L149 |
| 1.17.8 More control with warm_start | `warm_start=True` + `max_iter=1` 的手动迭代写法 | L151–L162 |
| （丢失的 `<details>`）References | Rumelhart 1986、Bottou 2010、Ng et al. 2011、LeCun et al. 1998、Kingma & Ba 2014 | 官方页；剪藏无 |

## 关键主张

### 1. MLP 的定义是「隐藏层 + 非线性激活」，不是「逻辑回归的别名」（来源论述）

L15 给出函数形式 $f:R^m\to R^o$，并明确它与 logistic regression 的差别：输入与输出之间可以有**一个或多个非线性隐藏层**。L21 进一步给出单层机制：每个隐藏神经元先算 $w_1x_1+\cdots+w_mx_m$，再过一个非线性 $g(\cdot)$，例如 tanh。

**本库综合**：这句话决定了两件事。第一，MLP 的假设空间不是「特征的线性组合」，而是「特征的 learned representation 的线性组合」；第二，特征工程的作用被削弱，但**缩放的作用被放大**——因为第一层的线性求和直接吃特征的量级。

### 2. `coefs_` / `intercepts_` 是「按层边界索引」的列表（来源论述）

L23 的契约是：`coefs_[i]` 是第 $i$ 层到第 $i+1$ 层的权重矩阵；`intercepts_[i]` 是加到第 $i+1$ 层的偏置向量。L52–L55 的 doctest 则给出 `hidden_layer_sizes=(5, 2)` 时的三个矩阵形状 `[(2, 5), (5, 2), (2, 1)]`。

**本库综合**：列表长度是 `n_layers_ - 1`，不是「隐藏层数」。把 `coefs_[0]` 读成「第一隐藏层」会错一位；`coefs_[-1]` 才是输出层。

### 3. 三个输出面：logistic、softmax、identity（来源论述 + 官方丢失数学节）

| 任务 | 输出层 | 损失 | `predict_proba` | 位置 |
| --- | --- | --- | --- | --- |
| 二分类 | logistic | 交叉熵 | 两列 | L57–L65 |
| 多分类 | softmax | 交叉熵 | $K$ 列 | L67 |
| 多标签 | 每个类独立 logistic | 逐类二元交叉熵 | $K$ 列 | L69–L84 |
| 回归 | identity | 平方误差 | 无 | L95–L97 |

来源只说二分类/多分类/多标签的行为，**没有给出 softmax 与 logistic 的公式**——那些在丢失的 Mathematical formulation 折叠块里（官方 1.9.1 页）：二分类过 $g(z)=1/(1+e^{-z})$ 并以 0.5 为阈值；多分类过 $\operatorname{softmax}(z)_i=e^{z_i}/\sum_l e^{z_l}$；回归直接输出 $f(x)$。

**本库综合**：多标签不是「多分类的另一种写法」。多分类的各类概率被 softmax 绑成和为 1；多标签的每个类各自过一个 logistic，因此可以同时为 1。L69–L84 的多标签 doctest 也采用逐类 0.5 阈值。

### 4. 回归默认是平方误差，但 1.7 起 API 已有 `loss='poisson'`（来源论述 + 外部核查）

L95–L97 说 `MLPRegressor` 输出层没有激活函数（等价于 identity），因此用**平方误差**，并支持多输出。剪藏正文只到这里。

外部核查（官方 1.9.1 API 页）：`MLPRegressor` 的签名是 `loss='squared_error'`，可选 `{'squared_error', 'poisson'}`；`poisson` 内部使用 log-link（指数输出）并要求 $y\ge0$。API 页明确写着 **Changed in version 1.7: Added parameter `loss` and option `'poisson'`**。也就是说：剪藏（≥1.8）对应的版本已经有这个参数，但**用户指南 1.17.3 完全没提**。

这是一个「文档只描述了默认路径」的典型：读 1.17.3 会以为 `MLPRegressor` 只有平方误差；实际上 1.7 起可以用 Poisson 损失做计数回归。

### 5. `alpha` 是 L2，但只罚权重矩阵，不罚偏置（来源论述 + 外部核查）

L101 说两个估计器都用 `alpha` 做 L2 正则，"penalizing weights with large magnitudes"。官方丢失的数学节把目标写成

$$\text{Loss}=\text{数据损失}+\frac{\alpha}{2n}\lVert W\rVert_2^2$$

其中 $W$ 是权重矩阵。外部核查 1.6.1 实现（`sklearn/neural_network/_multilayer_perceptron.py` 的 `_backprop`）：L2 项只对 `self.coefs_` 求和，**不含 `intercepts_`**。因此偏置不会被收缩。

**本库综合**：`alpha` 与 Ridge 的 $ℓ_2$ 是同一个机制，但作用对象是「一组矩阵」，且损失被 $1/n$ 归一化。所以同一个 `alpha` 在 Ridge、MLP、Lasso 之间**不可直接比较**；跨架构（宽度/深度）也不可比，因为 $\lVert W\rVert^2$ 的量级随参数个数变化（[[regularization-and-sparsity]]）。

### 6. 三个求解器：SGD、Adam、L-BFGS，训练契约不同（来源论述 + 外部核查）

L113–L129 的分配是：

- **SGD**：按梯度更新，学习率 $\eta$ 控制步长；支持在线与 mini-batch。
- **Adam**：随机优化器，但按低阶矩的自适应估计自动调整更新量；支持在线与 mini-batch。
- **L-BFGS**：用 Scipy 的 `fmin_l_bfgs_b` 近似 Hessian 与逆 Hessian；**不支持在线，也不支持 mini-batch**。

外部核查实现层：`partial_fit` 在 `solver='sgd'` 与 `'adam'` 上可用；在 `solver='lbfgs'` 上对象**没有 `partial_fit` 属性**。这与 L129 的「does not support online nor mini-batch」一致，但比文字更强：不是「支持得不好」，而是**没有这个入口**。

### 7. 复杂度公式：宽度是平方项，迭代次数是线性乘子（来源论述）

L133 给出反向传播的时间复杂度：

$$O\left(i\cdot n\cdot\left(m\cdot h+(k-1)\cdot h\cdot h+h\cdot o\right)\right)$$

其中 $n$ 样本、$m$ 特征、$k$ 隐藏层、每层 $h$ 个神经元、$o$ 输出、$i$ 迭代次数。来源的建议是「从小数量的隐藏神经元和少量隐藏层开始」。

**本库综合**：这个式子解释了三个工程现象：① 加宽一层是 $h^2$ 成本，加深一层是线性叠加；② `max_iter` 是乘在整体上的，所以「多跑 10 倍轮数」与「把宽度加 10 倍」不是同一种代价；③ 它完全不含 GPU/并行项——scikit-learn 的实现就是 CPU 上的矩阵乘法（L11 明确说没有 GPU 支持）。

### 8. 缩放是硬前提，且必须对测试集复用同一个缩放（来源论述）

L137 的措辞是 "sensitive to feature scaling, so it is highly recommended to scale your data"：可以缩到 $[0,1]$、$[-1,+1]$，或标准化到均值 0、方差 1；并且必须对测试集应用**同一个**缩放。L147 推荐把 `StandardScaler` 放进 `Pipeline`。

**本库综合**：第一层的线性求和与后续优化都直接受特征量级影响，因此缩放属于模型假设的一部分，而不是可有可无的清洗步骤。具体收益不能脱离数据与目标环境给出固定数值。

### 9. `alpha` 的经验网格是 $10^{-1}$ 到 $10^{-6}$，默认值就在网格内（来源论述 + 外部核查）

L148 说 `alpha` 最好用 `GridSearchCV` 找，范围通常是 `10.0 ** -np.arange(1, 7)`，即 $0.1, 0.01, 0.001, 10^{-4}, 10^{-5}, 10^{-6}$。外部核查 1.9.1 签名的默认 `alpha=0.0001`（$10^{-4}$）正好在网格内。

**本库综合**：`alpha` 控制权重矩阵的 L2 惩罚，但它与优化难度不是同一个轴；停止轮数、泛化与权重范数都应分别观察，不能由单一训练准确率推断正则效果。

### 10. 求解器建议与默认值不一致：小数据推荐 L-BFGS，默认却是 Adam（来源论述 + 外部核查）

L149 的经验是：L-BFGS 在小数据集上收敛更快、解更好；较大数据集上 Adam 很稳健、通常很快；SGD + momentum/nesterov 在**学习率调对**时可以更好。

但外部核查 1.9.1 签名：`solver='adam'` 是默认。也就是说，按文档「小数据优先 L-BFGS」的建议，必须**显式**写 `solver='lbfgs'`；默认参数不会替你做这个选择。

**本库综合**：这不是文档矛盾，而是「建议」与「默认值」的分工。默认值要覆盖多数场景（Adam 对大中小数据都不至于太差），而建议是在你愿意调参时换更合适的解。工程上应把 `solver` 当成**必调项**，不是默认项。

### 11. `warm_start=True` + `max_iter=1` 是手动训练循环，不是「增量学习新数据」（来源论述 + 外部核查）

L153–L162 给出的写法是：`warm_start=True`、`max_iter=1`，然后在外层循环里反复 `fit`，并在每轮之间做监控/检查。实现层的 `n_iter_` 是「本次 fit 的迭代数」，不是跨 fit 累计值；权重则在上一轮解的基础上继续更新。

**本库综合**：`warm_start` 复用解与优化器状态，适合「我想在每轮之间看 loss/验证分数」的场景；如果只是想要更多轮，直接调大 `max_iter` 更简单。两者在数值上不一定等价（优化器状态、early stopping 计数、学习率调度都会受影响）。

### 12. 剪藏丢掉的三块，正好是「为什么」而不是「怎么调」（结构性判断）

保留的正文是完整的 API 叙述：类名、doctest、三个求解器、缩放建议、`alpha` 网格、`warm_start` 写法。丢掉的三块是：

- 优缺点：非凸损失、多个局部极小、不同随机初始化 → 不同验证准确率；超参多；对缩放敏感。
- 数学表述：单隐藏层公式、tanh、logistic/softmax、交叉熵/MSE 的完整损失、反向传播、停止条件。
- 5 条文献。

**本库判断**：这不是「随机丢了几段」，而是**把「为什么 MLP 需要调参/需要固定种子/需要缩放」整块拿走了**。只读剪藏会得到「一个有很多参数的分类器」，读不到「它是一个非凸优化问题」。因此本页把非凸性与种子敏感性单列为关键主张，并明确标注来源是官方丢失折叠块，而不是剪藏正文。

## 版本判定

结论：**1.8.0 – 1.9.1**；抓取日 `/stable/` 页标题为 1.9.1，故工作版本取 **1.9.1**。四条独立证据（把剪藏逐句与官方 1.6/1.7/1.8/1.9 四页比对）：

| # | 剪藏 | 官方页差异 | 排除的版本 |
| --- | --- | --- | --- |
| 1 | L15 `X = \left\{x_1, x_2, ..., x_m\right\}` | 1.6 为 `X = {x_1, x_2, ..., x_m}`（无 `\left\{`），1.7 起加花括号 | 1.6 |
| 2 | L63 `array([[1.967e-04, 9.998e-01],` | 1.6 为 `array([[1.967...e-04, 9.998...-01],`（旧 repr 截断） | 1.6 |
| 3 | L67 `indices where the value is 1 represent` | 1.6 为 `represents` | 1.6 |
| 4 | L116 `w \leftarrow w - \eta \left[\alpha ... \right]` | 1.7 用 `(...)`，1.8 改为 `\left[...\right]` | 1.7 |

补充：1.7→1.8 的差异只有这一处公式括号（以及少量换行），1.8→1.9 的**正文逐词 diff 为空**。因此无法用措辞把 1.8 与 1.9 再分开；抓取日 `/stable/` 页标题为 "scikit-learn 1.9.1 documentation"，故取 1.9.1。本节所有功能性结论都不依赖这条判定。

外部旁证（不属于剪藏）：`MLPRegressor` 的 `loss='poisson'` 是 **1.7 新增**（官方 API 页 "Changed in version 1.7"）。剪藏对应的版本 ≥1.8，按理应包含该功能，但用户指南正文没有更新——这是「版本判定」与「文档内容」之间的一条独立证据：**剪藏不是旧到 1.6，而是新到已经包含 1.7 的 API、却仍只写平方误差**。

## 转换损失

官方 1.9.1 页本节正文（不含代码块）约 1,740 词，剪藏约 1,151 词。丢失的是 **3 个内容 `<details>` 折叠块，合计 622 词 ≈ 36%**：

| 丢失项 | 形态 | 词数 | 内容要点（据官方 1.9.1 页） |
| --- | --- | --- | --- |
| *Advantages and disadvantages of Multi-layer Perceptron* | `<details>` | 93 | 优点：能学非线性；能用 `partial_fit` 做在线学习。缺点：有隐藏层时损失非凸、存在多个局部极小，**不同随机权重初始化会导致不同验证准确率**；需要调隐藏神经元数、层数、迭代数；对特征缩放敏感 |
| *Mathematical formulation* | `<details>` | 460 | 单隐藏层单神经元的 $f(x)=W_2g(W_1^Tx+b_1)+b_2$；默认 $g$ 为 tanh；二分类过 logistic、多分类过 softmax、回归用 identity；分类的平均交叉熵与回归的 MSE（均带 $\alpha\lVert W\rVert_2^2/(2n)$）；反向传播与 $W^{i+1}=W^i-\epsilon\nabla Loss_W^i$；停止条件为达到最大迭代数或损失改进低于阈值 |
| *References* | `<details>` | 69 | Rumelhart, Hinton & Williams（反向传播）；Bottou（SGD, 2010）；Ng et al.（Backpropagation, 2011）；LeCun et al.（Efficient BackProp, 1998）；Kingma & Ba（Adam, 2014） |

结构性后果（本库判断）：

- 丢的三块恰好是**三类「读起来像背景」的内容**：非凸优化、完整损失、文献。因此剪藏在**接口层**完整（类名、doctest、`coefs_` 契约、求解器名称、缩放建议都在），但失去了：为什么不同种子会给出不同分数、为什么 `alpha` 的尺度与 $n$ 有关、以及三个输出面的公式。
- 5 条文献**整块丢失**，剪藏里没有对应的引用标记；本页对文献的引用只能指向官方 1.9.1 页，不能引剪藏行号。
- 公式与文本的转换噪声（不影响语义，但影响搜索）：L116 的 `Loss` 被拆成 `L o s s`；L15 的集合写成 `\left\{...\right\}`；L19 的图注末尾多一个 `#`；L21 的神经元集合写成 `\left\{x_i \left|\right. x_1,...\right\}`（原文是 `{x_i | x_1, ..., x_m}`）。

## 图表核查

2 张图均于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（HTTP 200），入库 `raw/assets/sklearn-neural-networks/`，**逐张目视核查**并记录尺寸与 SHA-256 前 16 位：

| # | 文件 | 尺寸 | 字节 | SHA-256 前 16 位 | 目视读数 |
| --- | --- | --- | --- | --- | --- |
| 1 | `multilayerperceptron_network.png` | 783×851 | 69,327 | `101e15495ae3bcaf` | 输入层有 `Bias +1` 与 $x_1,\dots,x_n$；隐藏层有 `Bias +1` 与 $a_1,a_2,\dots,a_k$；输出是单个 $f(X)$。所有连线都是全连接，**图上没有标注激活函数或权重值**；图用 $x_n$ 而正文用 $m$ 个特征，是符号不一致，不是模型差异 |
| 2 | `sphx_glr_plot_mlp_alpha_001.png` | 1700×900 | 329,945 | `80e4e325070c35fa` | 3 行数据集 × 5 列 `alpha`（0.10、0.32、1.00、3.16、10.00，即 $10^{-1}$ 到 $10^{1}$ 的等比五档）。左上第一列是原始散点，其余格右下角是测试准确率读数：第 1 行 **.875 / .950 / .950 / .875 / .475**；第 2 行 **.775 / .875 / .900 / .400 / .400**；第 3 行 **.875 / .950 / .950 / .925 / .825**。低 `alpha` 的边界有明显锯齿/尖角，高 `alpha` 变成平滑的斜带；`alpha=10` 在三行都明显欠拟合（第 1、2 行接近机会水平） |

**注意**：图 2 的读数与正文的 `alpha` 网格（$10^{-1}$ 到 $10^{-6}$）不是同一组值——图只扫到 $10^{1}$，且只展示五档。不要把图上的 `.950` 当成「`alpha=1.0` 的通用最优值」；它只对应图中三个玩具数据集。

## 外部核查（不在剪藏内）

### A. 两个估计器的默认签名（官方 1.9.1 API 页）

`MLPClassifier` 与 `MLPRegressor` 共享同一组默认值（除 `MLPRegressor` 多一个 `loss`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hidden_layer_sizes` | `(100,)` | 一个隐藏层、100 个神经元；**不是「无隐藏层」** |
| `activation` | `'relu'` | 可选 `'identity'`、`'logistic'`、`'tanh'`、`'relu'` |
| `solver` | `'adam'` | 可选 `'lbfgs'`、`'sgd'`、`'adam'` |
| `alpha` | `0.0001` | L2 强度；在来源建议的网格 $10^{-1}$–$10^{-6}$ 内 |
| `batch_size` | `'auto'` | 等价于 `min(200, n_samples)`（1.6.1 源码同） |
| `learning_rate` | `'constant'` | 仅 SGD 使用；另有 `'invscaling'`、`'adaptive'` |
| `learning_rate_init` | `0.001` | SGD/Adam 的初始学习率 |
| `power_t` | `0.5` | 仅 `learning_rate='invscaling'` 使用 |
| `max_iter` | `200` | 随机优化器的最大 epoch 数；L-BFGS 的最大迭代数 |
| `shuffle` | `True` | 仅 SGD/Adam；每 epoch 打乱样本索引 |
| `random_state` | `None` | 控制初始化、`shuffle` 与早停的验证划分（三者共享同一个 RNG；改一项会移动后续抽样） |
| `tol` | `0.0001` | 停止阈值；同时用于损失/验证分数的改进判断 |
| `warm_start` | `False` | 复用上次解与优化器状态 |
| `momentum` | `0.9` | 仅 SGD |
| `nesterovs_momentum` | `True` | 仅 SGD |
| `early_stopping` | `False` | 默认**不早停**，所以 `max_iter=200` 是硬上限 |
| `validation_fraction` | `0.1` | 仅 `early_stopping=True` 时使用；从训练集内部再切 10% |
| `beta_1` / `beta_2` | `0.9` / `0.999` | 仅 Adam |
| `epsilon` | `1e-08` | 仅 Adam，数值稳定项 |
| `n_iter_no_change` | `10` | SGD/Adam 连续多少轮没有超过 `tol` 的改进就停；L-BFGS 不用 |
| `max_fun` | `15000` | 仅 L-BFGS，最大函数调用次数 |
| `loss`（仅 `MLPRegressor`） | `'squared_error'` | 可选 `'poisson'`，1.7 新增；`poisson` 用 log-link，要求 $y\ge0$ |

### B. 实现细节（源码 `sklearn/neural_network/_multilayer_perceptron.py` @ 1.6.1；1.9.1 同构）

- **L2 只罚 `coefs_`，不罚 `intercepts_`**：`_backprop` 里 `for s in self.coefs_: values += s@s`，损失加 `0.5 * alpha * values / n_samples`。
- **`batch_size='auto'`**：`min(200, n_samples)`；`batch_size` 大于样本数时会被 `np.clip` 截断并给 warning。
- **早停的验证划分**：`train_test_split(..., test_size=validation_fraction, stratify=y if 二/多分类 else None)`；多标签**不 stratify**。`partial_fit` + `early_stopping=True` 直接 `ValueError`。划分用的 `random_state` 与初始化、shuffle 共用同一个 RNG。
- **`n_iter_no_change` 两种停止路径**：`early_stopping=True` 看验证分数；否则看训练损失。两者都用 `tol`。
- **`max_fun` 只在 L-BFGS 路径使用**；`momentum`/`nesterovs_momentum` 只在 SGD；`beta_1`/`beta_2`/`epsilon` 只在 Adam。
- **`n_iter_` 是本次 fit 的迭代数**：`warm_start` 循环里每次 fit 都从 1 重新计，不累加。

### C. 与来源正文的接口错位

- 来源正文只说 `MLPRegressor` 用平方误差（L95–L97），API 1.9.1 已有 `loss='poisson'`（1.7 新增）。这是**文档落后于 API**，不是版本判断错误。
- 来源正文说「L-BFGS 不支持在线/小批量」（L129），实现层是**根本没有 `partial_fit` 属性**；外部核查比正文更强。
- 来源正文完全不提 `early_stopping`、`batch_size`、`validation_fraction`、`n_iter_no_change`、`max_fun`、`beta_1/2`、`epsilon`、`momentum` 等参数；本页上述默认值全部来自 API 页/源码，不是剪藏论述。

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 分歧

1. **回归损失：正文只写平方误差，API 1.7 起已支持 `loss='poisson'`**（L95–L97 vs 官方 1.9.1 API 页 "Changed in version 1.7"）。这不是数值矛盾，而是**用户指南没有跟上 API**：按 1.17.3 读，读者不会知道 Poisson 回归的存在。
2. **求解器建议与默认值不一致**：L149 说小数据上 L-BFGS 更快更好，但 1.9.1 默认 `solver='adam'`（外部核查）。建议是「愿意调参时的选择」，默认值是「不调参时的保底」；两者可以同时成立，但工程上必须显式选。
3. **非凸性只存在于丢失的折叠块**：剪藏正文没有任何「多个局部极小 / 不同初始化导致不同验证准确率」的句子。只读剪藏会把 MLP 误读成「给定数据就有确定结果（除 `random_state` 外）」。
4. **图 1 的符号与正文不一致**：正文用 $m$ 个输入特征（$x_1,\dots,x_m$），图用 $x_1,\dots,x_n$。这是绘图符号问题，不影响模型定义；引用图时不要把它当成「输入维度是 $n$」的证据。
5. **图 2 的 `alpha` 范围与正文建议不同**：正文建议 $10^{-1}$–$10^{-6}$（L148），图只扫 $10^{-1}$–$10^{1}$；图上的 `.950` 不能外推成「$alpha=1$ 最优」。
6. **`alpha` 的作用对象**：正文说 "penalizing weights with large magnitudes"（L101），容易被读成「所有权重与偏置」；实现只罚 `coefs_`，不罚 `intercepts_`（外部核查）。这不是矛盾，但「权重」一词在代码里是狭义。
7. **`MLPRegressor` 的 API 首句仍写 "optimizes the squared error"**，而同页参数表已有 `loss='poisson'`（官方 1.9.1 页）。这是 API 文档内部的口径不一致，不是用户指南独有的问题；引用时以参数表/签名为准。

## 局限

- 本节**不覆盖**深度学习的工程栈：没有卷积、循环、注意力、归一化层、GPU、混合精度、分布式训练；L11 明确建议大规模/更灵活架构去看 Related Projects。因此本页不能回答「MLP 与 Transformer/CNN 怎么选」。
- 本节**不覆盖** `sklearn.calibration`（概率校准）、`sklearn.inspection`（特征重要性/部分依赖）、`sklearn.model_selection`（CV 切分）与 `sklearn.preprocessing`（缩放的完整选项）。`predict_proba` 的数值不能当校准概率用，本页只能提示，不能给校准方案。
- 本节**没有**任何关于缺失值、类别特征、稀疏输入的说明；MLP 的实现是否接受 NaN/稀疏矩阵、 categorical 变量要不要 one-hot，本页未验证，**不作断言**。
- 丢失的 3 个折叠块（622 词 ≈ 36%）包含**唯一**的非凸性说明、完整损失公式与 5 条文献；本页对这些内容的引用都指向官方 1.9.1 页，**不能引剪藏行号**。
- 本页不包含历史本机运行时实验；`loss='poisson'`、`early_stopping` 细节与 `max_fun` 行为需在 `uv` 目标版本环境中另行验证。
- 本页不提供「某数据集上最佳 `hidden_layer_sizes`/`alpha`」的数值；网格只代表来源建议的搜索范围，不是调参结果。

## 待解问题

1. `MLPRegressor(loss='poisson')` 在 1.9.1 上的实际行为：与 `PoissonRegressor` 的 log-link 是否数值等价？`alpha` 的尺度是否仍按 $1/n$？待目标环境验证。
2. 早停的验证划分是**从训练集内部再切 10%**；在时间序列/分组数据上，这个随机切分会不会泄漏？实现显示二/多分类会 stratify、多标签不 stratify，时间结构仍未验证。
3. `n_iter_no_change` 在没有 `early_stopping` 时看训练损失，在 `early_stopping` 时看验证分数；两者的停止点差异需在固定环境中系统对照。
4. `alpha` 只罚 `coefs_` 不罚 `intercepts_`：这对小样本、大偏置的模型有什么实际影响？待消融。
5. `hidden_layer_sizes=()` 是否**精确**等价于 `LogisticRegression`？结构上是一层线性输出 + logistic/softmax，但损失归一化、正则尺度、求解器都不同，不能断言数值等价。
6. `batch_size='auto'` 在 $n<200$ 时等于全批量；此时 SGD/Adam 与 L-BFGS 的差异是否只剩优化器？待验证。
7. `warm_start` 与 Adam 的优化器状态：连续 `fit` 时 `beta_1/beta_2` 的矩估计是否跨 fit 保留？需检查优化器内部状态。
8. 多标签的 0.5 阈值是硬编码；有没有 API 入口改逐类阈值？本页未找到，未断言「不能」。
9. `predict_proba` 的校准质量：MLP 的输出概率是否比 logistic/NB 更准？需用可靠性图评估。
10. 与 [[ensembles]] 的组合：MLP 作为 Bagging/Stacking 的基学习器是否值得（训练成本 × 种子方差）？待目标环境对照。
11. 与 [[nearest-neighbors]] 的对比：在中等样本、低维、强非线性边界上，MLP 与 KNN/树集成的实际差距？未做基准。
12. 1.9.1 是否修改了 `max_fun`/`n_iter_no_change` 的默认值或早停实现？签名已核相同，行为未核。

## 行号索引

| 主题 | 行号 |
| --- | --- |
| 大规模/GPU 警告 | L10–L11 |
| MLP 定义与隐藏层 | L15 |
| 单隐藏层机制、激活函数 | L21 |
| `coefs_`/`intercepts_` 索引契约 | L23 |
| `MLPClassifier` 与反向传播 | L27 |
| 输入数组形状 | L29 |
| 分类 doctest（拟合/预测/权重形状/概率） | L31–L65 |
| 仅交叉熵损失 | L57 |
| 多类 softmax | L67 |
| 多标签 logistic + 0.5 阈值 | L69–L84 |
| `MLPRegressor`、identity 输出、平方误差、多输出 | L95–L97 |
| `alpha` 的 L2 正则 | L101–L103 |
| 三个求解器 | L113 |
| SGD 更新公式 | L115–L119 |
| Adam 的自适应矩估计 | L123 |
| SGD/Adam 支持在线/小批量 | L125 |
| L-BFGS 与 Scipy 实现 | L127 |
| L-BFGS 不支持在线/小批量 | L129 |
| 反向传播复杂度 | L133 |
| 缩放建议与 `StandardScaler` | L137–L147 |
| `alpha` 的 GridSearchCV 网格 | L148 |
| 三个求解器的经验选择 | L149 |
| `warm_start` + `max_iter=1` | L153–L162 |
