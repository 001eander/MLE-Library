---
title: 门控特征学习单元（GFLU）：把 GRU 的门控搬到无时序的表格上
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - deep-learning
  - gating
  - feature-selection
  - interpretability
sources:
  - "[[gandalf]]"
status: draft
---

# 门控特征学习单元（GFLU）

GFLU 是 [[gandalf]] 提出的表格处理单元，用作「可替换 MLP 的特征表示层」（p. 2）。它的做法是把 GRU 的门控从**时间步**搬到**阶段**，并在每个阶段加一个可学的稀疏特征掩码。与 [[tabular-deep-learning]] 里的 token 化路线相比，它不把列变成 token、也不用注意力：结构来源是**逐阶段的加权更新**。

## 为什么需要一个专门的单元

原文给的三个出发点（p. 3、p. 4）：

1. 树模型用信息增益做特征选择，因此对噪声更不敏感；全连接网络「considers all the features and their interactions」；
2. TabNet 的注意力做特征选择，但「it operates at the instance level」；DANet 用 α-entmax 掩码加 BatchNorm，本文「avoid the need for batch normalization and introduce learnable sparsity」；
3. 表格「lacks any inherent temporal or sequential nature, devoid of a predefined order among its rows and columns」，所以不能直接把 GRU 当序列模型用。

## 四处改造

| | GRU | GFLU |
| --- | --- | --- |
| 递推轴 | 时间步 $t$ | 阶段 $n$ |
| 权重 | 跨时间步共享 | 逐阶段不同（$W_n^z,W_n^r,W_n^o$） |
| 输入 | 每步新的一段序列 | 同一份特征矩阵 $X$，每阶段换一个掩码 |
| 记忆 | 隐藏状态 | 学到的特征表示（$\tilde d$ 可与 $d$ 不同，实验取相同） |

原文对第 2 点的理由：「identical transformations at every stage may not be optimal and could even hinder performance」（p. 5）；对堆叠而非拼接的理由：「the stacked GFLUs refining the feature representation by choosing the right subset of features stage by stage」（p. 5）。

## 前向代数

记 $X\in\mathbb R^d$，第 $n$ 阶段的掩码为 $M_n\in\mathbb R^d$，$X_n=M_n\odot X$。则（式 7–12，p. 6–7）

$$
M_n=\text{t-softmax}(F_n,t),\qquad X_n=M_n\odot X,
$$
$$
z_n=\sigma\big(W_n^z\cdot[H_{n-1};X_n]\big),\qquad
r_n=\sigma\big(W_n^r\cdot[H_{n-1};X_n]\big),
$$
$$
\tilde H_n=\tanh\big(W_n^o\cdot[r_n\odot H_{n-1};X]\big),\qquad
H_n=(1-z_n)\odot H_{n-1}+z_n\odot\tilde H_n .
$$

三个容易读漏的点：

- **掩码只进两个门，候选用全特征 $X$**（式 11 写的是 $X$ 不是 $X_n$）。原文理由是「prevent the induced sparsity in the feature masks from impeding gradient flow」（p. 6），并称经验上优于把掩码作用到整个输入。
- 更新是**插值**而非覆盖：$z_n\to0$ 时该阶段几乎不改变表示，$z_n\to1$ 时几乎整体替换。
- 两个门可以合并成单个 $W_n^i\in\mathbb R^{2d\times 2d}$ 一次算出（p. 7）。

**参数账本（综合，按式 10–12 的维度推算）**：取 $\tilde d=d$，每阶段为 $W_n^i$ 的 $4d^2$、$W_n^o$ 的 $2d^2$、$F_n$ 的 $d$，合计 $6d^2+d$，即**随阶段数线性增长**。$N$ 在调参中可达 30（表 6，p. 19），这是「参数效率」一说需要注意的前提。

## t-softmax：可学的稀疏掩码

普通 softmax 的每一项恒正，做不了硬选择。t-softmax 先给每项一个权重再归一化（式 5–6，p. 5）：

$$
\text{softmax}(a_i,w_i)=\frac{w_i e^{a_i}}{\sum_j w_j e^{a_j}},\qquad
w_i=\mathrm{ReLU}\big(a_i+t-\max(a)\big),\ t>0 .
$$

由此得到三条可操作的读数：

1. **稀疏是精确置零**，不是「压小」：$a_i\le\max(a)-t$ 的项权重为 0，分子分母同时消失；
2. **至少有 1 项非零**：最大化项拿到 $w=t>0$，其余拿到 $\mathrm{ReLU}(a_i-\max a+t)$；
3. **代价接近普通 softmax**：原文称它「almost as fast as the regular softmax」，而 α-entmax 与 sparsemax「require a sorting operation」（p. 5）。

原文紧接着写「all the weights, $w_i$, are positive」，与式 6 的 ReLU 不一致（只有 $a_i>\max(a)-t$ 的项为正），见 [[gandalf]] 分歧 10。

**`Init Sparsity` 的语义（综合，含推导）**：原文用 r-softmax 只为了给 $t$ 定初值（式 16，p. 8）：

$$
\text{r-softmax}(x,r)=\text{t-softmax}(x,t_r),\qquad t=-\text{quantile}(x,r)+\max(x).
$$

代回第 1 条：被保留的项满足 $a_i>\max(a)-t=\text{quantile}(a,r)$，即**约 $r$ 比例的项被置零**。这与搜索空间 `Init Sparsity ~ Uniform(0.0,0.9)`（p. 12）以及「sparsity between 0.0 and 0.5 is a good range」的表述一致。原文只说「use this as a way to initialize the t parameter」，未写出这层等价关系。

另外两个边界：$t$ 可以设为可学参数，此时「proper initialization is not essential, but helpful」（p. 6）；若 $t$ 固定为不可学，$t=0$ 会让所有项权重为 0，因此原文限定 $t>0$（式 6，p. 5；这一步是**综合**）。

## 初始化：用分布造阶段多样性

$$F_n\sim\text{Beta}(\alpha_n,\beta_n),\qquad \alpha_n,\beta_n\sim U(0.5,10) \quad\text{（式 15，p. 8）}$$

原文的理由是两点：**多样性**（每个阶段对特征有不同视角）与**稀疏**（Beta 采样偏向小值），并明确「Departing from a Gaussian distribution」。这解释了一个设计后果：不同阶段抽到不同的 $(\alpha_n,\beta_n)$，掩码形状因此天然不同，不需要额外正则去防止阶段塌缩。

## 解释性：把掩码求和

$$I=\sum_{n=1}^N M_n,\qquad I^{norm}_i=\frac{I_i}{\sum_{i=1}^D I_i}\quad\text{（式 17，p. 8）}$$

掩码是**逐列、逐阶段**的全局量（不随样本变化），因此可以直接读成一列特征在整条网络里的平均权重；这与 GBDT 的重要性同性质。**综合**：每个 $M_n$ 是 t-softmax 的输出，在被保留的项上加和为 1，因此 $I$ 的总质量恰为 $N$，$I^{norm}$ 的含义是「第 $i$ 列占全部阶段权重的份额」。

这条解释性有一个已声明的代价：重要性只反映**门控件**有多依赖该列，而候选支路始终看全特征，所以「掩码为 0 的列」仍会通过 $\tilde H_n$ 影响输出。

## 与其它表格结构的对照

| | GFLU（本页） | FT-Transformer（[[ft-transformer]]） | 行文本化（[[kaggle-tfms-llms-tabular]]） |
| --- | --- | --- | --- |
| 结构来源 | 逐阶段门控更新 | 特征 = token，注意力混合 | 语言模型读序列 |
| 特征选择 | 每阶段一个全局稀疏掩码 | 无显式选择（注意力权重可事后读） | 无显式选择 |
| 列序 | 掩码逐列，与顺序无关 | 不注入位置信息 | 训练时重排、推理时平均 |
| 并行性 | 阶段间顺序依赖，不可并行 | token 维全并行 | 序列内因果 / 双向取决于骨干 |
| 规模旋钮 | 阶段数 $N$（2–30） | 层数 / 维度 / 头数 | 骨干与序列长度 |

## 代价与失败模式

- **顺序计算**：原文自述「its inherent sequential processing across stages, which precludes parallel processing」（p. 14）。这是结构性的，不是实现问题。
- **掩码稀疏与梯度**：作者只把掩码放在门控支路，理由是避免阻断梯度，但没有给出这组对照的数字（见 [[gandalf]] 待解问题 3）。
- **阶段数敏感的调参**：FANOVA 显示 `GFLU Stages` 影响最大且「doesn't show any particular pattern」（p. 12–13），表 6 中既有 2 阶段也有 30 阶段的获胜配置，说明这个旋钮没有稳定默认值。
- **dropout 复合**：`GFLU Dropout` 逐阶段施加，「hence compounding the effect」，推荐区间 0–0.05（p. 12–13）。
- **可解释性的边界**：$I$ 只在掩码层面成立；把它当作「删掉这列会掉多少分」的因果量会失真，原文自己也用替换为高斯噪声的 MoRF/LeRF 去单独测保真度（p. 14）。

## 待解问题

1. 掩码只进两个门、候选看全特征，这组设计相对「掩码作用于整个输入」的增益有多大？原文称有经验支持，未给数字。
2. $t$ 设为可学时，训练中它实际落在什么范围？是否与 `Init Sparsity` 强相关？
3. 阶段间是否会出现掩码塌缩（多阶段选同一批列）？只要 $\alpha_n,\beta_n$ 各自采样能否避免？
4. 阶段间顺序依赖在 $d$ 很大、$N$ 很大时的 wall-clock 代价？原文只给参数量与 MACs。
5. 类别特征进 GFLU 的方式：原文只说「we will just need to average the $I$s associated with the corresponding embeddings」（p. 13），没有给编码方案与实验。
6. 这套门控与 GRU 之外的序列单元（LSTM、Highway）搬过来是否同样可行？原文只做了 GRU 的适配。

## 关联页面

- 来源：[[gandalf]]
- 概念：[[tabular-deep-learning]]（token 化与注意力路线）、[[feature-selection]]（三家族与选择稳定性）、[[multi-layer-perceptron]]、[[regularization-and-sparsity]]
- 对照与实践：[[gbdt-vs-tabular-dl]]、[[tabular-model-selection]]
