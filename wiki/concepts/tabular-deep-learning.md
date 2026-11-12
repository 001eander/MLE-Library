---
title: 表格数据的深度学习（Tabular Deep Learning）：把每一列变成一个 token
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - deep-learning
  - transformer
  - feature-embedding
  - attention
  - resnet
sources:
  - "[[ft-transformer]]"
  - "[[sklearn-preprocessing]]"
status: draft
verified: 2026-11-12
applies_to:
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13）
  - PyTorch >=1.8,<3
---

# 表格数据的深度学习（Tabular Deep Learning）

表格 DL 与图像/文本 DL 的根本差别不在“用什么层”，而在**输入是什么**：一张表有 $k$ 列，列与列之间没有天然顺序，数值列与类别列的语义不同，而且 $k$ 通常远小于序列长度。FT-Transformer 的做法是把每一列变成一个 $d$ 维 token，再在“特征序列”上做注意力（[[ft-transformer]], p. 4）。本页整理这条路线的最小代数、默认配置、预处理契约、归纳偏置与失败模式；具体选型见 [[tabular-model-selection]]，与 GBDT 的对照见 [[gbdt-vs-tabular-dl]]。

> **证据边界**：架构与实验数字来自 [[ft-transformer]]（NeurIPS 2021，arXiv v5）；代码行为来自官方仓库的本地 checkout（`package/rtdl_revisiting_models.py`，commit `e3ed46c`）。本页没有重跑训练；`stats.json` 的结果复核见来源页的“官方实现核查”。

## 一张表就是 $k$ 个 token

记号：数据集 $D=\{(x_i,y_i)\}_{i=1}^n$，$x_i=(x_i^{(num)},x_i^{(cat)})$，特征总数 $k$（[[ft-transformer]], p. 3）。任务可以是二分类、多分类或回归。

FT-Transformer 的前向过程可以写成四步：

1. **Feature Tokenizer**：每个特征 $x_j$ 变成一个 $d$ 维向量 $T_j$；
2. **拼接**：$T=\mathrm{stack}[T_1,\ldots,T_k]\in\mathbb{R}^{k\times d}$；
3. **加 `[CLS]`**：$T_0=\mathrm{stack}[[CLS],T]$，再过 $L$ 层 Transformer；
4. **预测**：取 $T_L[CLS]$，过 `LayerNorm → ReLU → Linear` 得到 $\hat y$（[[ft-transformer]], p. 4–5）。

关键点：**没有位置编码**。代码里只有特征 embedding、`[CLS]` embedding 和 Transformer 层，没有任何 positional embedding（`package/rtdl_revisiting_models.py` L692–L749）。因此模型对特征的排列是**置换等变**的（在同时置换对应特征参数的前提下；加 `[CLS]` 后，`[CLS]` 的输出对特征排列不变）；特征之间的交互完全由注意力学习。这是“把表当集合”的归纳偏置，也是它与文本 Transformer 的最大结构差异（**综合**，来源正文没有显式讨论）。

## Feature Tokenizer 的代数

对第 $j$ 个特征：

$$
T_j=b_j+f_j(x_j)\in\mathbb{R}^d,\qquad f_j:\mathcal X_j\to\mathbb{R}^d.
$$

- 数值特征：$T_j^{(num)}=b_j^{(num)}+x_j^{(num)}\cdot W_j^{(num)}$，其中 $W_j^{(num)}\in\mathbb{R}^d$，是**逐元素**乘一个可学习的 $d$ 维向量；
- 类别特征：$T_j^{(cat)}=b_j^{(cat)}+e_j^{\top}W_j^{(cat)}$，其中 $W_j^{(cat)}\in\mathbb{R}^{S_j\times d}$ 是查找表，$e_j$ 是 one-hot；
- $b_j$ 是**每个特征独立**的 $d$ 维偏置（[[ft-transformer]], p. 4，图 2a）。

代码与公式一一对应（`package/rtdl_revisiting_models.py` L201–L209、L270–L288，节选）：

```python
# 数值特征：逐特征线性映射 + 逐特征偏置
x = x[..., None] * self.weight      # (batch, n_num, d)
x = x + self.bias[None]             # (batch, n_num, d)

# 类别特征：逐列 embedding + 逐特征偏置
x = torch.stack(
    [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
)
if self.bias is not None:
    x = x + self.bias
```

**为什么特征偏置重要**：表 5/23 的消融里，去掉特征偏置后在 8 个数据集中的 7 个变差（CA 0.470 vs 0.459、HE 0.381 vs 0.391、JA 0.724 vs 0.732、HI 0.727 vs 0.729、AL 0.958 vs 0.960、CO 0.964 vs 0.970、MI 0.751 vs 0.746）；YE 是例外（8.843 vs 8.855，无偏置更好）（[[ft-transformer]], p. 9、p. 23）。来源仍称特征偏置“必要”；按列均值看是 7/8，不是 8/8。

## Transformer 主干：PreNorm、去掉第一层 normalization、最后一层只算 CLS

来源用 **PreNorm**（normalization 放在每个残差分支的入口），理由是更容易优化；在 PreNorm 下还必须**去掉第一层 Transformer 的第一个 normalization**，否则效果不好（[[ft-transformer]], p. 5）。代码把这一点写成了“**CRUCIAL**”（`package/rtdl_revisiting_models.py` L482–L492），并在第 0 层不注册 `attention_normalization`（L621–L656）。

```python
for i_block, block in enumerate(self.blocks):
    x_identity = x
    if 'attention_normalization' in block:   # 第 0 层没有这个 key
        x = block['attention_normalization'](x)
    x = block['attention'](x[:, :1] if i_block + 1 == n_blocks else x, x)
    x = x_identity + block['attention_residual_dropout'](x)
    # ... FFN 分支同理 ...
```

这段代码还有两个主文没写的细节：

1. **最后一层只计算 `[CLS]` 的 query**（`x[:, :1]`），key/value 仍是全部 token；预测语义不变，但最后一层的 per-token 注意力/表示不会被计算；
2. **PostNorm 时要省略预测头里的 LayerNorm**（[[ft-transformer]], p. 18）。

**PreNorm vs PostNorm 的取舍**：PostNorm 可能给出更好的目标指标，但通常需要学习率 warmup 或特殊初始化；来源为了“所有模型用同一套简单优化”选了 PreNorm（[[ft-transformer]], p. 17–18）。这不是“PreNorm 永远更好”，而是**优化简单性**与**峰值指标**的交换。

## 默认配置与成本

表 12 的默认 FT-T（[[ft-transformer]], p. 18）：

| 项目 | 值 |
| --- | --- |
| 层数 | 3 |
| 特征 embedding 维度 | 192 |
| 头数 | 8 |
| 激活 / FFN 宽度因子 | ReGLU / $4/3$ |
| attention dropout | 0.2 |
| FFN dropout | 0.1 |
| residual dropout | 0.0 |
| 初始化 | Kaiming |
| 参数量 | 929K（按 100 个数值特征计） |
| 优化器 | AdamW，lr $10^{-4}$，wd $10^{-5}$ |

**1–6 层的缩放规则**不在论文正文里；官方包把它固定成（`package/rtdl_revisiting_models.py` L752–L779）：

| 层数 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| `d_block` | 96 | 128 | 192 | 256 | 320 | 384 |
| attention dropout | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
| FFN dropout | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |

来源自己说默认配置是“educated guess”、没有投入很多调参，缩放规则也可能次优（[[ft-transformer]], p. 18）。因此**默认配置是起点，不是最优**；表 4 显示默认 FT-T 集成已经很强，但调参后的 GBDT 会在部分数据集反超。

**成本**：$k$ 个 token 的 MHSA 是 $O(k^2d)$；特征数很大时这是主要瓶颈（[[ft-transformer]], p. 5）。表 10 的实测倍数从 AD 的 0.9× 到 Yahoo 的 13.8×（[[ft-transformer]], p. 16）；但这是调参后配置 + 早停的时间，不是复杂度缩放律。特征数很大时可考虑 Linformer 式 KV 压缩（Epsilon 用了 `kv_compression=0.064`、`headwise`，$1+2000$ 个 token → 约 128 维；`output/epsilon/ft_transformer/tuned/0/stats.json`）；包 README 同时警告：**特征数不够大时 Linformer 可能更慢，且它对精度的影响研究不足**。

## 预处理是模型的一部分

表格 DL 的默认预处理与 GBDT 不同：

- **数值特征**：默认 `QuantileTransformer(output_distribution='normal')`；代码取 `n_quantiles=max(min(n_train//30, 1000), 10)`、`subsample=1e9`，并在 fit 前对 **train** 加 `N(0, 1e-3)` 的噪声（按列 std 缩放），以处理“取值种类很少”的特征（[[ft-transformer]], p. 6、p. 14；`lib/data.py` L21–L44）。
- **例外**：Helena/ALOI 用标准化（ALOI 是图像数据）；Epsilon 用原始特征，因为预处理反而伤害深度模型（[[ft-transformer]], p. 6）。
- **目标**：回归目标按 train 的均值/标准差标准化（式 3，p. 14）。
- **GBDT**：不做特征预处理，因为树对特征平移/缩放不敏感（[[ft-transformer]], p. 14）。
- **类别特征**：神经网络所有类别特征用同维 embedding；XGBoost 用 one-hot，CatBoost 用内置类别支持（[[ft-transformer]], p. 6）。

```python
normalizer = sklearn.preprocessing.QuantileTransformer(
    output_distribution='normal',
    n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
    subsample=1e9,
    random_state=seed,
)
if noise:
    stds = np.std(X_train, axis=0, keepdims=True)
    noise_std = noise / np.maximum(stds, noise)
    X_train += noise_std * np.random.default_rng(seed).standard_normal(X_train.shape)
normalizer.fit(X_train)
```

注意：分位数变换是**保秩**映射，不是保相关/保距离；它把每列压到近似正态，但会改变列间相关结构。表格 DL 用它主要是为了让优化器看到尺度相近、尾部可控的输入，而不是为了“更好的统计性质”（[[sklearn-preprocessing]] 的“分位数与幂变换”一节）。

## MLP 与 ResNet 是必要的对照

**MLP**（式 1，p. 3）：`Linear(MLPBlock(...))`，`MLPBlock = Dropout(ReLU(Linear(x)))`。它是表格 DL 的 sanity check：调参后 MLP 的平均排名 4.8，已经优于 TabNet/SNN/AutoInt/GrowNet，接近 DCN2（4.7）（[[ft-transformer]], p. 7）。

**ResNet**（式 2，p. 3）：`x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))`，预测头 `Linear(ReLU(BatchNorm(x)))`。调参后平均排名 3.3，是“没有任何 DL 模型能稳定超过”的基线（[[ft-transformer]], p. 7）。

一个实现差异：官方包的 `MLP` 对所有层用同一宽度，并在注释里标明这是与论文的 `# NOTE[DIFF]`——论文对首尾层单独调宽，但包作者发现同宽“至少在这些数据集上不更差”（`package/rtdl_revisiting_models.py` L35–L39）。这说明**论文的超参搜索空间与推荐实现不必逐字一致**。

## 归纳偏置：为什么它有时赢、有时输

FT-T 的归纳偏置可以概括为三条：

1. **特征是一个集合**：没有位置编码，列顺序不影响输出；
2. **每个特征有独立的 token 空间**：数值列学逐列线性映射，类别列学查找表；交互由注意力在 token 间学习；
3. **`[CLS]` 聚合**：所有特征通过注意力汇总到一个向量，再做预测。

这使它在“目标函数更像树/分段函数”的数据集上比 ResNet 更稳：合成插值实验里，ResNet 随目标从 DL-friendly 变 GBDT-friendly 而显著退化，CatBoost 相反，FT-T 在整个 α 区间保持竞争力（[[ft-transformer]], p. 9，图 3）。来源据此推测 FT-T 更适合逼近一部分“树状”函数（p. 9）。这是**机制假设**：论文没有做表征层面的验证，也没有证明注意力真的在学树结构。

反过来，FT-T 的失败模式也很清楚：

- **特征数很大**：注意力成本 $O(k^2)$，训练时间也会上升（Yahoo 699 个数值特征、正文取整为 700，实测 13.8×；Epsilon 2000 特征需要 Linformer）；
- **类别特征覆盖不足**：11 个数据集只有 Adult 有 8 个类别特征，因此“高基数类别特征上 FT-T 更好”没有证据；
- **小样本/缺失值**：来源没有缺失值实验；小样本下 929K 参数（按 100 个数值特征计）可能过重；
- **Epsilon 单模型最差**：0.8982，低于（更差于）TabNet 0.8896 等（表 2）；该数据集用了 Linformer 缩放与启发式默认配置，未做完整调参；
- **注意力不是因果解释**：AM 与 PT/IG 的秩相关只是排序一致性；AM 与 IG 的排序相关不代表两者的重要性含义相同（[[ft-transformer]], p. 10）。

## 与其它页面的关系

- 选型与落地：[[tabular-model-selection]]
- 与 GBDT 的对照：[[gbdt-vs-tabular-dl]]
- 基线概念：[[multi-layer-perceptron]]、[[ensembles]]、[[regularization-and-sparsity]]
- 预处理契约：[[feature-preprocessing]]、[[preprocessing-pipeline-design]]
- 评估协议：[[cross-validation]]、[[hyperparameter-search]]、[[evaluation-metrics]]
- 来源：[[ft-transformer]]

## 待解问题

1. 无位置编码的“特征集合”假设，在特征有天然分组/层次结构时是否成为限制？
2. 高基数类别特征上，FT-T 的类别 embedding 与 GBDT 的目标编码谁更稳？
3. 缺失值应该编码成额外 token、插补，还是直接用 GBDT？
4. 最后一层只算 `[CLS]` query 的优化，对注意力重要性图与训练动态的影响？
5. 默认 1–6 层缩放规则在论文数据之外是否仍合理？
6. PreNorm + 去掉第一层 norm 的“CRUCIAL”是否随层数/数据规模变化？
7. 把 FT-T 蒸馏到 ResNet/MLP 的精度-延迟曲线？
8. 无位置编码是否意味着 FT-T 对特征重排完全不变？数值/类别列的内部顺序是否也无关？（代码上是，但未做实验验证）
