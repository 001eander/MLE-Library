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
  - "[[kaggle-q-ft]]"
  - "[[kaggle-q-ft-v2]]"
status: draft
verified: 2026-11-12
applies_to:
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13）
  - PyTorch >=1.8,<3
  - Qwen/Qwen3-0.6B-Base（config.json 于 2026-11-12 核对）
  - answerdotai/ModernBERT-base（safetensors.total = 149,655,232，2026-11-12 核对）
---

# 表格数据的深度学习（Tabular Deep Learning）

表格 DL 与图像/文本 DL 的根本差别不在“用什么层”，而在**输入是什么**：一张表有 $k$ 列，列与列之间没有天然顺序，数值列与类别列的语义不同，而且 $k$ 通常远小于序列长度。FT-Transformer 的做法是把每一列变成一个 $d$ 维 token，再在“特征序列”上做注意力（[[ft-transformer]], p. 4）。本页整理这条路线的最小代数、默认配置、预处理契约、归纳偏置与失败模式；具体选型见 [[tabular-model-selection]]，与 GBDT 的对照见 [[gbdt-vs-tabular-dl]]。

> **证据边界**：架构与实验数字来自 [[ft-transformer]]（NeurIPS 2021，arXiv v5）；代码行为来自官方仓库的本地 checkout（`package/rtdl_revisiting_models.py`，commit `e3ed46c`）。Q-FT 两节来自 Kaggle 竞赛帖 [[kaggle-q-ft]] 与 [[kaggle-q-ft-v2]]（单次 5 折、单种子、无代码），其参数账本由本库按 Hugging Face 配置手算复核。本页没有重跑任何训练。

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

## LLM 当骨干：把表当成一句话

FT-T 把每列变成一个 token，但 token 只在一个小 Transformer 里（929K 参数）。[[kaggle-q-ft]] 把同一件事推到极端：**token 直接进一个语言模型**，维度与词向量对齐。

**接口**：Hugging Face 模型可以接收 `inputs_embeds` 而不是 token id，因此可以完全跳过词嵌入表——模型看到的是 22 个 1,024 维向量，而不是文本（[[kaggle-q-ft]], L84）。这是“不把行写成文本”的技术前提。

**三段式嵌入**（每列一个 embedder，[[kaggle-q-ft]], L41–L46）：

$$t_j=\mathrm{RMSNorm}\Big(e^{\text{col}}_j+e^{\text{val}}_j(x_j)+f^{\text{num}}_j(x_j)\Big)\cdot g,\qquad \mathrm{RMS}(t_j)\approx 0.030$$

- $e^{\text{col}}$：列身份，用列名的语言模型词向量（如 “Flight Distance” 各 token 的均值）**初始化**后训练；
- $e^{\text{val}}$：按该列的 code 查表；
- $f^{\text{num}}$：只给数值列，标准化后的 $x$ 过 $\sin(2\pi f x)/\cos(2\pi f x)$（32 个可学习频率）再进一个小 MLP（64 → 1,024，图上是单层 + ReLU）；
- $g$：把 RMS 压到 ≈0.030（“Qwen 典型词向量的大小”），让表 token 与文本 token 同尺度。

**逐列编码是这个方案里最可复用的部分**（[[kaggle-q-ft]], L54–L60）：类别列用词表，评分列直接用等级，Age 每个不同值一个 code，Flight Distance 用“出现 ≥3 次的距离各给一个 route id + 其余归 rare”的词表（3,301），延迟分 8 个桶；同时**只有这 4 个数值列**额外接数值嵌入。注意 Age 的 code 是 29、数值是 $-0.217$——code 与数值是两条并行通道，不是二选一。

**代价**：解码器只有因果注意力，因此只有最后一个位置（CLS）能看到全部特征。作者的对策是**训练时每行每步重排特征顺序**，推理时对 5 种顺序的 logit 取均值（[[kaggle-q-ft]], L93、L120）。这条增强是全篇唯一大于折间波动的消融：固定顺序 0.953546 vs 重排 0.960444（**−0.0069**）。

## 冻结骨干 + 低秩适配（LoRA）

[[kaggle-q-ft]] 用 LoRA 把可训练参数压到骨干的约 2%（[[kaggle-q-ft]], L99–L104）：

$$\text{output}=Wx+\frac{\alpha}{r}B(Ax),\qquad r=16,\ \alpha=32,\ \alpha/r=2$$

- **B 初始化为零** → 第 0 步精确等价于原始权重；训练只动 $A,B$；
- 挂在**每层全部 7 个投影**上：attention 的 q/k/v/o + MLP 的 gate/up/down；
- 参数账本（本库按 Qwen3-0.6B-Base 的 `config.json` 手算，与来源一致）：每层 360,448 × 28 层 = **10,092,544**；另有 feature tokenizer + CLS/头约 **3.87M**；冻结骨干 440,466,432（不含被 tie 的词嵌入表）；
- 冻结**不等于免费**：0.6B 权重仍要跑前向与反向（[[kaggle-q-ft]], L104）。

一个反直觉的读数：**冻结骨干（只训 tokenizer + 头）只比 LoRA 低 0.0007**（0.959788 vs 0.960444，[[kaggle-q-ft]], L131）。也就是说在这个任务上，LoRA 的容量不是瓶颈；同一个作者的“随机初始化骨干与预训练骨干只差 8e-6”进一步说明**语言预训练本身没有提供收益**——但那条证据只有单折、且两组可训练参数集不同（见来源页分歧 3）。

## order-free：把「表是集合」从训练技巧变成结构约束

FT-T 表达“特征是一个集合”的方式是**不注入位置信息**；Q-FT 的方式是**重排 + 因果掩码**（训练技巧）；[[kaggle-q-ft-v2]] 的方式是**编码器 + 所有 token 的 `position_ids` 都设为 0**（结构约束）。

- ModernBERT 用 RoPE，位置由传入的 `position_ids` 驱动；全部传 0 就没有位置可言（[[kaggle-q-ft-v2]], L66）；
- 作者自查：打乱特征顺序后输出变化约 **3e-6**（浮点误差量级），因此训练不需要重排、推理只需一次前向（[[kaggle-q-ft-v2]], L68）；
- 池化从“读末尾 CLS”改为 **23 个输出向量的均值**（21 特征 + 2 summary）；CLS head、attention-pooling、mean+max 在噪声内相同（[[kaggle-q-ft-v2]], L72）。

**边界**：ModernBERT-base 是局部窗口 128 与全局注意力每 3 层交替（`local_attention=128`、`global_attn_every_n_layers=3`），所以“所有 token 互相可见”只在**行长 < 128** 时成立——这个任务一行 24 个 token，局部≈全局。若把行拉长到数百 token，这条等价性就不再成立（[[kaggle-q-ft-v2]] 分歧 6）。

**两条加在 token 上的外部信息**（同一来源）：每个特征 token 额外加 **9 个**关于该取值的数（计数、路由画像、无标签典型性残差、原始数据的平滑 log-odds），另加 **2 个 summary token**（28 个行统计数、17 个原始数据块）——折 0 约 +4e-4。**同类特征给 GBDT 约 +1.6e-3**，是 transformer 的近 4 倍（[[kaggle-q-ft-v2]], L100）；作者的解读是逐值嵌入已经学到了 counts 与路由画像能说的东西。这是“人工特征 vs 模型内建表征”可替代性的一个直接读数，见 [[gbdt-vs-tabular-dl]]。

**表格自监督预训练**（同一来源，L105–L113）：ELECTRA 式的替换 token 检测（RTD）——随机把 15% 的值换成同列的另一行值，逐 token 判断真/假。不使用标签，因此语料可以是 train + test + 外部数据集（1.13M 行），且**与 CV 折无关**：只跑一次，5 折共享同一初始化。折 0 约 +3e-4；预训练 10 epochs 反而更差，用外部数据集做**有监督**预训练则完全没用（L120）。

## 三条路线的对照

| | FT-Transformer（[[ft-transformer]]） | Q-FT（[[kaggle-q-ft]]） | Q-FT v2（[[kaggle-q-ft-v2]]） |
| --- | --- | --- | --- |
| 位置信息 | 无（不注入） | RoPE 0…21 + 每行重排 | RoPE，全部位置编号 0 |
| 注意力 | 双向 | 因果 | 双向（局部 128 / 每 3 层全局） |
| 池化 | `[CLS]` | 末尾 CLS | 23 个输出均值 |
| 数值特征 | 逐特征标量乘 | 周期特征 + 小 MLP | 同左 + 9 个 side features |
| 类别特征 | 查找表 | 查找表 + 列嵌入 | 同左 |
| 预训练 | 无 | 语言预训练（消融称无用） | 语言 + 表格 RTD |
| 可训练参数 | 929K（100 个数值特征） | 10.1M LoRA + 3.9M tokenizer | 全部 149M |
| 数据规模 | 11 个数据集，多为千—万行 | ~70 万行，单赛题 | ~70 万行 + 1.13M 预训练语料 |

共同点比差异更值得记：**三条路线都把每一列变成一个向量，都主张列的顺序没有语义**。差别在于用什么机制表达这个假设（不注入 / 训练时重排 / 位置恒零），以及用多大的骨干去混合这些 token。

## 与其它页面的关系

- 选型与落地：[[tabular-model-selection]]
- 与 GBDT 的对照：[[gbdt-vs-tabular-dl]]
- 基线概念：[[multi-layer-perceptron]]、[[ensembles]]、[[regularization-and-sparsity]]
- 预处理契约：[[feature-preprocessing]]、[[preprocessing-pipeline-design]]
- 评估协议：[[cross-validation]]、[[hyperparameter-search]]、[[evaluation-metrics]]
- 来源：[[ft-transformer]]、[[kaggle-q-ft]]、[[kaggle-q-ft-v2]]

## 待解问题

1. 无位置编码的“特征集合”假设，在特征有天然分组/层次结构时是否成为限制？
2. 高基数类别特征上，FT-T 的类别 embedding 与 GBDT 的目标编码谁更稳？
3. 缺失值应该编码成额外 token、插补，还是直接用 GBDT？
4. 最后一层只算 `[CLS]` query 的优化，对注意力重要性图与训练动态的影响？
5. 默认 1–6 层缩放规则在论文数据之外是否仍合理？
6. PreNorm + 去掉第一层 norm 的“CRUCIAL”是否随层数/数据规模变化？
7. 把 FT-T 蒸馏到 ResNet/MLP 的精度-延迟曲线？
8. 无位置编码是否意味着 FT-T 对特征重排完全不变？数值/类别列的内部顺序是否也无关？（代码上是，但未做实验验证）
9. 「不注入位置」与「位置恒为零」在 RoPE 下是否严格等价？两者对注意力分布的影响是否相同？（Q-FT v2 用 3e-6 的自查间接支持等价，但没有对照实验）
10. 表格 RTD 的收益能外推到多大数据规模？1.13M 行语料里有多大比例是测试集特征，这部分“传导式”信息值多少分？
11. token 级嵌入（每列一个向量）与 FT-T 的线性投影，在**类别列少、数值列多**的数据上谁更省？Q-FT 的 tokenizer 需要 3,479 个 code 的词表，这在基数更高的赛题里会膨胀到多少？
