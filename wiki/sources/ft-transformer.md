---
title: "Revisiting Deep Learning Models for Tabular Data（FT-Transformer，NeurIPS 2021）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - deep-learning
  - transformer
  - resnet
  - gbdt
  - benchmark
  - feature-embedding
  - attention
source_path: raw/papers/FTTransformer.pdf
extracted_path: raw/extracted/FTTransformer.md
source_kind: pdf
published: 2021
status: draft
---

# Revisiting Deep Learning Models for Tabular Data（FT-Transformer）

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | *Revisiting Deep Learning Models for Tabular Data* |
| 作者 | Yury Gorishniy、Ivan Rubachev、Valentin Khrulkov、Artem Babenko |
| 机构 | Yandex；Moscow Institute of Physics and Technology；Higher School of Economics |
| 发表 | NeurIPS 2021（p. 1 页脚）；本地 PDF 为 **arXiv:2106.11959v5**，边注日期 **2023-10-26**（p. 1） |
| 本地文件 | `raw/papers/FTTransformer.pdf`（25 页，1,676,287 B） |
| SHA-256 | `f2faa32ffd8c15ed8534677f978dc02730fdbe1f94e9c9557ada6aa19163478d` |
| 抽取 | `raw/extracted/FTTransformer.md`（1,678 行，`pdftotext -layout`，含 `## Page N`） |
| 代码 | <https://github.com/yandex-research/tabular-dl-revisiting-models>；本地 checkout `~/Thirdparties/rtdl-revisiting-models`，commit `e3ed46c`（2024-11-13），`rtdl_revisiting_models` **0.0.2** |
| 许可 | 代码仓库为 Apache 2.0；论文版权归原作者。本页只引用短代码片段作解释，不转售、不整包复制 |
| 主题 | 表格深度学习的同协议比较；ResNet 基线；FT-Transformer；GBDT 对照；合成目标插值；注意力特征重要性 |

**来源定位**。这不是一篇提出单一新模型的论文，而是一份**基准与方法学报告**：先指出表格 DL 文献缺少可比协议与强基线，再用同一套训练/调参协议比较 9 个 DL 模型与 2 个 GBDT，最后给出两个可复用基线（ResNet、FT-Transformer）。它的主要价值在协议、对照与失败模式，而不在单个 SOTA 数字。

## 结构

| 页 | 内容 |
| --- | --- |
| 1 | 摘要、贡献列表、问题陈述 |
| 2 | 相关工作：可微树、注意力模型、显式乘法交互 |
| 3 | 记号；MLP（式 1）；ResNet（式 2） |
| 4 | FT-Transformer；图 1、图 2；Feature Tokenizer；`[CLS]`；Transformer 堆叠 |
| 5 | PreNorm、预测头、局限；其它模型（SNN/NODE/TabNet/GrowNet/DCN V2/AutoInt/XGBoost/CatBoost） |
| 6 | 11 个数据集（表 1）；预处理、调参、评估、集成、优化器；类别特征处理 |
| 7 | DL 单模型结果（表 2）；DL 集成结果（表 3）；4.4 结论 |
| 8 | 与 GBDT 对比（表 4）；4.6 “FT-T 的优势出现在 GBDT 胜过 ResNet 的数据集上” |
| 9 | 合成目标插值实验（图 3）；特征偏置/AutoInt 消融（表 5） |
| 10 | 注意力图特征重要性（表 6）；结论；参考文献开始 |
| 11–13 | 参考文献 |
| 14–25 | 补充材料 A–H：软硬件、数据与预处理、全量单模型/集成结果（表 8/9）、训练时间（表 10）、调参时间预算（表 11）、FT-T 细节（表 12/13）、各模型实现与搜索空间（表 14–22）、合成实验细节（表 23、算法 1）、附加数据集（表 24/25） |

## 关键主张

### 1. 问题：缺少可比协议与强基线

来源的出发点是：现有表格 DL 论文各自使用不同基准与实验协议，导致“哪个模型更好”不清楚；同时领域缺少易于使用、跨任务稳定的基线（p. 1–2）。作者把 MLP 视为当时的主要简单基线，但指出它并不总能对其他竞争者构成挑战（p. 2）。

**来源论述（Source claims）**：两项贡献是 ResNet 基线（此前被忽视）与 FT-Transformer；两者都在相同训练与调参协议下与大量模型比较（p. 1–2）。

### 2. 两个模型：ResNet 与 FT-Transformer

**ResNet**（式 2，p. 3）：

$$
\begin{aligned}
\mathrm{ResNet}(x) &= \mathrm{Prediction}(\mathrm{ResNetBlock}(\ldots(\mathrm{ResNetBlock}(\mathrm{Linear}(x))))) \\
\mathrm{ResNetBlock}(x) &= x + \mathrm{Dropout}(\mathrm{Linear}(\mathrm{Dropout}(\mathrm{ReLU}(\mathrm{Linear}(\mathrm{BatchNorm}(x)))))) \\
\mathrm{Prediction}(x) &= \mathrm{Linear}(\mathrm{ReLU}(\mathrm{BatchNorm}(x)))
\end{aligned}
$$

块的输入/输出同宽，残差分支内是 `BatchNorm → Linear → ReLU → Dropout → Linear → Dropout`；“几乎清晰的主路径”被作者认为有利于优化（p. 3）。

**FT-Transformer**（p. 4）：先把每一列变成一个 $d$ 维 token，再在 token 序列上堆叠 Transformer 层，最后用 `[CLS]` 的表示做预测。Feature Tokenizer 的代数：

$$
T_j = b_j + f_j(x_j)\in\mathbb{R}^d,\qquad
T_j^{(num)}=b_j^{(num)}+x_j^{(num)}\cdot W_j^{(num)},\qquad
T_j^{(cat)}=b_j^{(cat)}+e_j^{\top}W_j^{(cat)}
$$

其中 $b_j$ 是第 $j$ 个特征的特征偏置，$e_j$ 是类别特征的 one-hot 向量；$T=\mathrm{stack}[T_1^{(num)},\ldots,T_{k^{(num)}}^{(num)},T_1^{(cat)},\ldots,T_{k^{(cat)}}^{(cat)}]\in\mathbb{R}^{k\times d}$（p. 4，图 2a）。

**Transformer 与预测头**：把 `[CLS]` 的 embedding 追加到 $T$，得到 $T_0=\mathrm{stack}[[CLS],T]$，再应用 $L$ 层 $T_i=F_i(T_{i-1})$；预测为

$$
\hat y=\mathrm{Linear}(\mathrm{ReLU}(\mathrm{LayerNorm}(T_L[CLS]))).
$$

来源使用 **PreNorm**，并称在 PreNorm 下必须去掉第一层 Transformer 的第一个 normalization 才能取得好结果；PostNorm 可能给出更好的目标指标，但通常需要 warmup 或特殊初始化（p. 5、p. 17–18）。

**局限（来源自述）**：FT-T 比 ResNet 等简单模型更耗硬件与时间；特征数“太大”时不易扩展；原因是 vanilla MHSA 对特征数的二次复杂度；可用高效注意力近似缓解，也可蒸馏到更简单架构（p. 5）。

### 3. 协议：同一划分、同一调参流程、15 seeds、3 个集成

- **数据**：11 个公开数据集；每个数据集只有一个固定的 train/validation/test 划分，所有算法共用（p. 5–6、表 1）。
- **任务**：二分类、多分类、回归；排序数据集（Yahoo、Microsoft）按 pointwise 回归处理（p. 6）。
- **预处理**：所有深度模型共用同一预处理；默认 `QuantileTransformer`；Helena/ALOI 用标准化；Epsilon 用原始特征（预处理反而有害）；回归目标标准化（p. 6、p. 14）。
- **调参**：Optuna + TPE（多数算法）；最佳超参按验证集选，测试集不参与调参；其余算法遍历论文推荐配置（p. 6）。
- **评估**：每个配置跑 **15 个随机种子**，报告测试集均值；部分算法也报告默认配置（p. 6）。
- **集成**：把 15 个单模型分成 3 组、每组 5 个，组内平均预测，得到 3 个集成（p. 6）。
- **统计**：主文与补充表格使用单侧 Wilcoxon 检验，`p = 0.01`；“top”定义为“与最佳均值的差距不显著”（p. 7、p. 14）。
- **训练**：分类交叉熵、回归 MSE；除 TabNet/GrowNet 用 Adam 外，其它用 AdamW；**不使用学习率调度**；`patience = 16`（p. 6）；FT-T 默认 `lr=1e-4`、`weight_decay=1e-5`（表 12、p. 18）。
- **类别特征**：XGBoost 用 one-hot；CatBoost 用内置类别支持；神经网络所有类别特征用同维 embedding（p. 6）。

### 4. 主要结果：FT-T 平均排名最好，但不是每个数据集都最好

表 2（p. 7）的 15-seed 均值与平均排名：

| 模型 | 平均排名（std） |
| --- | --- |
| FT-Transformer | **1.8（1.2）** |
| ResNet | 3.3（1.8） |
| NODE | 3.9（2.8） |
| DCN2 | 4.7（2.0） |
| MLP | 4.8（1.9） |
| AutoInt | 5.7（2.3） |
| GrowNet | 5.7（2.2） |
| SNN | 6.4（1.4） |
| TabNet | 7.5（2.0） |

按列均值逐数据集看（表 2，p. 7）：

- FT-T 最好：CA 0.459、AD 0.859、JA 0.732、HI 0.729、CO 0.970；
- ResNet 最好：HE 0.396、AL 0.963；
- NODE 最好：YE 8.784、YA 0.753、MI 0.745；
- **Epsilon 单模型 FT-T 反而是全表最差**：0.8982，优于它的有 TabNet 0.8896、AutoInt 0.8949、NODE 0.8958、ResNet 0.8969 等；该数据集用 Linformer 缩放与启发式默认配置，未做完整调参。另一个异常是：表 8 的 FT-T 默认配置在 EP 上是 0.8959，而“tuned”是 0.8982——按验证集选出的配置在测试集上回退，来源没有解释。
- 因此“performs best on most tasks”应读作**平均排名 + 5/11 列均值最佳**，不是“每个数据集都最佳”。

**集成后**（表 3，p. 7）：FT-T 在 CA 0.448、AD 0.860、HE 0.398、JA 0.739、HI 0.731、AL 0.967、EP 0.8984、YE 8.751、CO 0.973、YA 0.747、MI 0.743。按列均值，FT-T 在 **9/11** 个数据集上优于 NODE，AD 持平（0.860 vs 0.860），唯一落后是 YE（NODE 8.716 vs FT-T 8.751，RMSE 越低越好）。因此 4.4 的“FT-Transformer outperforms NODE”在集成口径上成立；单模型口径则不是每个数据集都成立（见“分歧”）。

### 5. 与 GBDT 对比：没有普适赢家

表 4（p. 8）比较默认与调参后的集成：

| 设置 | 关键读数 |
| --- | --- |
| 默认超参 | FT-T 集成在 **9/11** 个数据集上同时优于默认 XGBoost/CatBoost；AD 两者都更好，CA 只有 CatBoost 更好 |
| 调参后 | CatBoost/XGBoost 在 **CA、AD、Yahoo** 明显反超，MI 也略好；ALOI 的 GBDT 为空（不可调/极慢） |
| 结论 | “there is still no universally superior solution”；DL 在多数任务上更好只说明基准略偏 “DL-friendly”（p. 8） |

来源还指出：GBDT 不适合**类别数很多**的多分类——Helena（100 类）表现不理想，ALOI（1000 类）甚至慢到无法调参（p. 8）。

### 6. 4.6 与 5.1：FT-T 的优势集中在“GBDT 胜过 ResNet”的数据集

来源观察：FT-T 相对 ResNet 的优势，主要出现在 GBDT 优于 ResNet 的数据集（CA、AD、Covertype、Yahoo、Microsoft），而在其余数据集与 ResNet 持平（p. 8）。

为验证这一点，作者构造合成任务：固定 $x\sim\mathcal N(0,I_k)$，在两种目标之间插值

$$
y=\alpha\cdot f_{GBDT}(x)+(1-\alpha)\cdot f_{DL}(x),
$$

其中 $f_{GBDT}$ 是 30 棵随机决策树的平均预测，$f_{DL}$ 是 3 层随机初始化 MLP；目标在训练前标准化（p. 9、p. 23）。图 3（p. 9）的目视读数：

- ResNet：α=0 时 RMSE≈0.36，α=0.5 时≈0.56，α=1 时≈0.645；
- FT-Transformer：≈0.38 → 0.40 → 0.39 → 0.29 → 0.23；
- CatBoost：≈0.47 → 0.46 → 0.39 → 0.28 → 0.19。

即 ResNet 随目标变“树友好”而显著退化，CatBoost 相反，FT-T 在整个区间保持竞争力（p. 9）。来源据此推测：FT-T 更适合逼近一部分“树状”函数，这与表 4 的观察相关（p. 9）。

**注意**：这是**合成数据上的机制证据**，不是对真实数据集的因果解释；“注意力在学树结构”属于假设，来源没有做表征层面的验证。

### 7. 消融：特征偏置通常必要，Transformer 主干优于 AutoInt

表 5/23（p. 9、p. 23）在 8 个数据集上比较：

| 模型 | CA | HE | JA | HI | AL | YE | CO | MI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AutoInt | 0.474 | 0.372 | 0.721 | 0.725 | 0.945 | 8.882 | 0.934 | 0.750 |
| FT-T（无特征偏置） | 0.470 | 0.381 | 0.724 | 0.727 | 0.958 | 8.843 | 0.964 | 0.751 |
| FT-T | **0.459** | **0.391** | **0.732** | **0.729** | **0.960** | **8.855** | **0.970** | **0.746** |

**7/8 个数据集上带特征偏置更好；YE 是唯一例外**（无偏置 8.843 < 有偏置 8.855，RMSE 越低越好）。来源仍把特征偏置称为“必要”，并在表 5 的 7 列加粗 FT-T；本页按数值记录这一处例外。FT-T 在 8/8 个数据集上都优于 AutoInt。来源把差异归因于：AutoInt 的 embedding 层没有特征偏置，主干不是 vanilla Transformer，且推理不使用 `[CLS]`（p. 9）。

### 8. 注意力图是廉价的特征重要性

对第 $i$ 个样本，取 `[CLS]` 的平均注意力图 $p_i$；再对样本平均：

$$
p=\frac{1}{n_{samples}}\sum_i p_i,\qquad
p_i=\frac{1}{n_{heads}\times L}\sum_{h,l}p_{ihl}.
$$

来源用 permutation test（PT）的排序作为参照，比较注意力图（AM）与 Integrated Gradients（IG）的秩相关（表 6，p. 10）：

| 数据集 | AM | IG |
| --- | --- | --- |
| CA | 0.81 (0.05) | 0.84 (0.08) |
| HE | 0.77 (0.03) | 0.74 (0.03) |
| JA | 0.78 (0.05) | 0.75 (0.04) |
| HI | 0.91 (0.03) | 0.72 (0.03) |
| AL | 0.84 (0.01) | 0.89 (0.01) |
| YE | 0.92 (0.01) | 0.50 (0.03) |
| CO | 0.84 (0.04) | 0.90 (0.02) |
| MI | 0.86 (0.02) | 0.56 (0.02) |

AM 在 5/8 个数据集上高于 IG；成本是 **1 次前向**，而 PT 需要 `n_features + 1` 次前向，IG 通常慢几个数量级（p. 10）。来源结论是“简单平均注意力图在成本-效果上是好选择”；这**不**意味着 AM 与 IG 的重要性含义相同，也不意味着注意力是因果解释。

### 9. 调参与成本：默认配置已经很强，延长调参收益有限

- 表 11（p. 17）：FT-T 在 CA 上 0.25h 已达 0.466，6h 为 0.459；XGBoost 0.437→0.432。来源称 FT-T 在 Optuna 前 10 次随机采样后就能得到不错的指标，且其它算法的延长调参没有显著收益（p. 16–17）。
- 表 10（p. 16）：ResNet 与 FT-T 的训练时间（秒，15 run 均值）与倍数：CA 72/187（2.6×）、AD 144/128（**0.9×**）、HE 363/536（1.5×）、JA 163/576（3.5×）、HI 91/257（2.8×）、AL 933/2864（3.1×）、EP 704/934（1.3×）、YE 777/1776（2.3×）、CO 4026/5050（1.3×）、YA 923/12712（**13.8×**）、MI 1243/2857（2.3×）。
- 来源把 Yahoo 的大倍数归因于 700 个特征（表 1 为 699 个数值特征，正文取整；p. 16）；AD 的 0.9× 说明“FT-T 总是更慢”不成立于单个数据集，训练时间取决于调参后的层数/宽度/早停，而不是只由架构决定。

## 官方实现核查（本地 checkout，非 `raw/`）

仓库不在 `raw/` 内，也不是论文的确定性抽取；它是所有者 clone 到 `~/Thirdparties/rtdl-revisiting-models` 的外部证据。以下路径均相对仓库根目录，commit 为 `e3ed46c`（2024-11-13）。仓库 README 明确推荐 `package/` 而非 `bin/` + `lib/` 的原始实现，并称两者只有标记为 `# NOTE[DIFF]` 的小差异。

### 论文 → 代码对照

| 论文/图 | 代码位置 | 核查结论 |
| --- | --- | --- |
| Feature Tokenizer（p. 4，图 2a） | `package/rtdl_revisiting_models.py` L161–L210、L212–L293 | 数值特征 `x[..., None] * W + b`；类别特征逐列 `nn.Embedding` 再相加偏置；与公式一致 |
| 默认配置（表 12） | `package/rtdl_revisiting_models.py` L752–L779；`output/california_housing/ft_transformer/default/0/stats.json` | 3 层/192 维/8 头/attn 0.2/FFN 0.1/residual 0.0；AdamW 1e-4、wd 1e-5、batch 256、patience 16；与表 12 一致 |
| PreNorm 且去掉第一层 normalization（p. 5） | `package/rtdl_revisiting_models.py` L482–L492（docstring）、L621–L656（构造/forward） | docstring 称这一步 **CRUCIAL**；第 0 层没有 `attention_normalization` |
| 预测头（p. 5） | `package/rtdl_revisiting_models.py` L664–L670 | `LayerNorm → ReLU → Linear`；PostNorm 时省略该 LayerNorm 的说明在 p. 18 |
| 权重衰减豁免（p. 18） | `package/rtdl_revisiting_models.py` L780–L838；`bin/ft_transformer.py` L375–L381 | 包版：CLS/数值/类别 embedding、所有 `*_normalization`、所有 `.bias` 的 wd=0；原始版按名字含 `tokenizer`/`.norm`/`.bias` 过滤 |
| 分位数预处理与噪声（p. 6、p. 14） | `lib/data.py` L21–L44 | `QuantileTransformer(output_distribution='normal', n_quantiles=max(min(n//30,1000),10), subsample=1e9)`；噪声按列 std 缩放后只加到 train，再 fit |
| 类别嵌入（p. 6） | `package/rtdl_revisiting_models.py` L212–L293 | `bias=True` 时每个特征一个独立偏置；与论文“所有类别特征同维 embedding”一致 |
| Epsilon 的 Linformer 缩放（p. 18） | `output/epsilon/ft_transformer/tuned/0/stats.json` | `kv_compression=0.064`、`kv_compression_sharing='headwise'`、1 层/96 维；$1+2000$ 个 token × 0.064 ≈ 128 维投影，与正文“projection dimension 128”吻合 |
| 合成插值实验（p. 9） | `bin/analysis_ft_transformer_vs_resnet.py` L205–L230；`output/analysis_ft_transformer_vs_resnet/*/0.toml` | `coefs=[0,0.25,0.5,0.75,1]`；`y = a*y_tree + (1-a)*y_mlp`；每个 α 重新标准化目标；树集成 30 棵×100 节点×深度 10；MLP 3×256；CatBoost 20000 iterations/早停 10/CPU；ResNet 4 blocks/256/hidden 1.5/dropout 0.5；FT-T 默认 |
| 集成（p. 6） | `bin/ensemble.py` L34–L65 | 种子组 `[0–4]`、`[5–9]`、`[10–14]`；分类器先 `expit`/`softmax` 再平均概率；NODE 的集成只用 val/test（不含 train） |

### 代码片段

**Feature Tokenizer**（`package/rtdl_revisiting_models.py` L201–L209、L270–L288，节选）：

```python
class LinearEmbeddings(nn.Module):
    def __init__(self, n_features, d_embedding):
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))

    def forward(self, x):
        x = x[..., None] * self.weight      # 每个数值特征一个 d 维向量
        return x + self.bias[None]          # 再逐特征加偏置


class CategoricalEmbeddings(nn.Module):
    def forward(self, x):
        x = torch.stack(
            [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        )
        if self.bias is not None:
            x = x + self.bias
        return x
```

**默认配置与 1–6 层缩放规则**（L752–L779）：

```python
@classmethod
def get_default_kwargs(cls, n_blocks: int = 3):
    return {
        'n_blocks': n_blocks,
        'd_block': [96, 128, 192, 256, 320, 384][n_blocks - 1],
        'attention_n_heads': 8,
        'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1],
        'ffn_d_hidden': None,
        'ffn_d_hidden_multiplier': 4 / 3,
        'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
        'residual_dropout': 0.0,
        '_is_default': True,
    }
```

这张表**不在论文正文里**：论文只说用了启发式缩放规则、可能次优（p. 18）；代码把 1–6 层的具体取值固定下来。

**去掉第一层 normalization + 最后一层只算 `[CLS]` 的 query**（L642–L670，节选）：

```python
for i_block, block in enumerate(self.blocks):
    x_identity = x
    if 'attention_normalization' in block:   # 第 0 层没有这个 key
        x = block['attention_normalization'](x)
    # 最后一层只计算 [CLS] 的注意力输出
    x = block['attention'](x[:, :1] if i_block + 1 == n_blocks else x, x)
    x = x_identity + block['attention_residual_dropout'](x)

    x_identity = x
    x = block['ffn_normalization'](x)
    x = x + block['ffn_residual_dropout'](block['ffn'](x))

x = x[:, 0]  # [CLS]
```

“最后一层只算 `[CLS]`”是主文没有写的实现优化；它不改变预测语义，但会改变你在最后一层读到的 per-token 注意力/表示。

**分位数预处理 + 噪声 workaround**（`lib/data.py` L21–L44，节选）：

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
return {k: normalizer.transform(v) for k, v in X.items()}
```

**权重衰减豁免**（`bin/ft_transformer.py` L375–L381）：

```python
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
```

包版 `make_parameter_groups()` 用对象而不是名字做同样的事，并额外覆盖 CLS/数值/类别 embedding；如果你自己实现，**不要**只对 `model.parameters()` 传一个统一 `weight_decay`。

### `stats.json` 独立复核（2026-11-12，本地只读）

方法：对 `output/<dataset>/<algorithm>/tuned/*/stats.json`（以及必要的 `default/`、`*_ensemble/`）取 `metrics.test.score` 的 15-seed 均值；回归列的比较用绝对值（仓库把 RMSE 存为负分）。与论文表 2/3/4 逐格比对：

| 论文表 | 可复核格数 | 最大绝对偏差 | 缺失格 |
| --- | --- | --- | --- |
| 表 2（DL 单模型） | 95/99 | **0.0005** | GrowNet 在 HE/JA/AL/CO 为“–”；NODE HE/AL 与 FT-T Yahoo 用 `default`，与论文一致 |
| 表 3（DL 集成） | 30/33 | **0.0005** | NODE HE/AL、FT-T Yahoo 用 `default_ensemble`，与论文一致 |
| 表 4（GBDT 对照，默认） | 33/33 | **0.0005** | 无 |
| 表 4（GBDT 对照，调参） | 41/41 | **0.0005** | XGBoost/CatBoost 在 ALOI 为“–”；FT-T Yahoo 用 `default`，与论文一致 |

结论：仓库保存的结果文件能在 **0.0005** 内复现论文主表；缺口恰好对应论文声明的“默认配置/不可调”例外。这验证的是**结果文件的存储与聚合**，不是训练管线；本次没有重跑训练，也没有安装 `rtdl_revisiting_models`。

## 转换损失

抽取整体完整（正文、表 2–25、补充材料 A–H、参考文献都在），但有五类损失：

1. **p. 4 的图被压平**：图 1/图 2 的矢量文字变成巨量空白与 `<latexit …>` base64 垃圾；公式与图注仍可恢复，但**不能**只凭抽取文本重建图。
2. **公式排版**：式 (1)(2) 的换行/对齐丢失，`Feature Tokenizer` 的 `where b_j is ...` 说明句被排到公式之前；需按 p. 4 目视核查恢复。
3. **表格样式**：表 2 的粗体在文本层不可读；v5 的正文表全部为黑色文本，附录表 8/9 才有红/蓝字，且为旋转排版。**不能**把表 2 的粗体当作“列内最优”的证据（见“分歧”）。
4. **算法 1**：缩进与伪代码结构被压平，但步骤文字完整（p. 24）。
5. **图 3 的坐标数字**：抽取把 y 轴刻度与正文交错；趋势需目视读图（本页已做）。

## 分歧与待解

1. **4.6 的“exactly”与 Adult 的集成结果不完全一致**。来源说 FT-T 的优势“恰好”出现在 GBDT 优于 ResNet 的数据集（CA、AD、Covertype、Yahoo、Microsoft；p. 8）。但表 4/9 的集成均值里，Adult 是 FT-T **0.860** vs ResNet **0.857**（FT-T 略差）；单模型表 2 则是 0.859 vs 0.854（FT-T 略好）。差距很小、可能落在“top”统计噪声内，但“exactly”在集成口径上不成立。
2. **“FT-T outperforms NODE”在集成口径成立，但不是每个数据集都成立**。表 3 中 FT-T 在 9/11 个数据集上优于 NODE，AD 持平（0.860 vs 0.860），唯一落后是 YE（NODE 8.716 vs FT-T 8.751）；单模型表 2 里 NODE 还在 YA（0.753 vs 0.756）与 MI（0.745 vs 0.746）略好。FT-T 的优势主要来自它在 NODE 明显更差的数据集（HE/JA/HI/AL/EP/CO）上更稳，而不是“每个数据集都赢”。
3. **“同一调参协议”有多个例外**：Yahoo 完全未调参、Epsilon 只用启发式默认配置、NODE 在 HE/AL 未调参、GrowNet 不支持多类、CatBoost 在 ALOI 不可调（p. 6、p. 18、p. 21）。因此表 4 的“调参后”不是所有模型都同等调参。
4. **表 2 的粗体不可解释为“列内最优”**。Epsilon 列 FT-T `0.8982` 被加粗，而 TabNet `0.8896` 更好；Adult 行有三个不同值同时加粗。v5 正文表没有颜色，只有粗体；附录表 8/9 的图注才定义 bold/blue/red 三档。**本库只使用数值与图注定义，不据粗体下结论。**
5. **默认 XGBoost 的单模型与集成数值完全相同**（表 8/9 均为 `±0.0`）。一个合理但未被来源明确陈述的解释是：默认 XGBoost 在固定种子下确定性训练，5 个“不同种子”给出相同模型，集成没有收益（**假设**，需在具体版本上验证）。
6. **Wilcoxon `p=0.01` 未提多重比较校正**。9 个模型 × 11 个数据集的成对检验会增加假阳性；来源的“top”判定应理解为**未校正的显著性描述**，不是强推断（本库方法学备注，非来源主张）。
7. **“FT-T 总是更贵”与 AD 的 0.9× 冲突**。表 10 是**调参后配置 + 早停**的实测时间，不是架构复杂度缩放律；AD 上 FT-T 反而比 ResNet 快。不能把表 10 外推为“给定 k 的时间公式”。
8. **CO2 论述没有测量**。来源用“表格问题无处不在 + MHSA 二次复杂度”论证 CO2 风险（p. 5），但没有能耗读数；这是定性风险陈述。

## 局限

- **类别特征覆盖极窄**：11 个数据集中只有 Adult 有 8 个类别特征；其余 10 个全是数值特征（表 1，p. 6）。因此本页不能支撑“FT-T 在高基数类别特征上更强”的结论。
- **无缺失值实验**：来源没有讨论缺失值处理；FT-T 的 tokenizer 也没有缺失值分支。
- **单一划分**：每个数据集只有一个固定 train/val/test 划分；没有多种划分的方差，无法区分“数据集难度”与“划分运气”。
- **无模型无关技巧**：不预训练、不额外损失、不做数据增强/蒸馏/学习率 warmup/decay；这是为了比较架构归纳偏置，但也意味着结果不是“工程上限”（p. 5）。
- **调参预算不完全可比**：不同算法的搜索空间、迭代数与不可调例外不同（p. 6、p. 18、表 13–22）。
- **统计口径**：单侧 Wilcoxon、未校正；表 2 的“top”依赖最佳均值，且粗体/颜色语义在 v5 中不完整。
- **解释性**：注意力图只是相关性排序，不是因果；没有校准、不确定性或分布偏移实验。
- **成本**：表 10 是特定硬件（V100 32GB）与调参后配置的时间，不可直接换算到其它硬件/实现。

## 待解问题

1. 在**高基数类别特征**为主的数据集上，FT-T 相对 CatBoost/目标编码 + GBDT 的胜负如何？
2. 有缺失值时，Feature Tokenizer 应该学缺失 token、用均值插补，还是交给 GBDT？
3. 多种 train/val/test 划分下，表 2 的排名是否稳定？“5/11 列均值最佳”有多少来自划分选择？
4. 4.6 的“exactly”在单模型上成立、在集成上对 Adult 不成立；这是噪声还是集成改变了归纳偏置？
5. 去掉第一层 normalization 的“CRUCIAL”在不同数据规模/层数下是否仍成立？PostNorm + warmup 能否反超？
6. 最后一层只算 `[CLS]` query 的优化，对注意力重要性图与训练动态有什么影响？
7. 默认配置的 1–6 层缩放规则是否在论文数据之外仍合理？它与 Wies et al. (2021) 的“理论缩放规则”差多少？
8. 表 2 的粗体/表 8 的颜色在原始 NeurIPS 版本中到底如何定义？v5 是否在生成时丢了颜色？
9. 默认 XGBoost 的 `±0.0` 是否确实由确定性训练导致？在 GPU/不同 `nthread` 下是否仍为零方差？
10. 把 FT-T 蒸馏到 ResNet/MLP 后，精度与延迟的权衡如何？（来源只提了“可以蒸馏”，没有实验，p. 5）

## 行号索引（`raw/extracted/FTTransformer.md`）

| 页 | 起始行 | 主要内容 |
| --- | --- | --- |
| 1 | 8 | 标题、摘要、贡献 |
| 2 | 69 | 相关工作 |
| 3 | 129 | 记号、MLP、ResNet |
| 4 | 191 | FT-T、图 1/2、Feature Tokenizer、`[CLS]` |
| 5 | 510 | PreNorm、预测头、局限、其它模型 |
| 6 | 572 | 数据集、表 1、预处理/调参/评估 |
| 7 | 634 | 表 2/3、4.4 |
| 8 | 697 | 表 4、4.5、4.6 |
| 9 | 758 | 图 3、5.1、5.2、表 5 |
| 10 | 830 | 5.3、表 6、结论、参考文献 |
| 11–13 | 892–1045 | 参考文献 |
| 14 | 1046 | 补充 A/B、表 7 |
| 15 | 1100 | 表 8/9 |
| 16 | 1147 | 表 10、D.2 |
| 17 | 1196 | 表 11、E.1 |
| 18 | 1254 | 表 12/13、E.3 |
| 19–23 | 1312–1593 | 表 14–22、F 各模型 |
| 24 | 1594 | 算法 1、表 24 |
| 25 | 1659 | 表 25 |

## 复核状态

本页与关联页面于 2026-11-12 由所有者复核并提交为 `ingest: Revisiting Deep Learning Models for Tabular Data（FT-Transformer）`。代码片段只做了源码阅读与 `stats.json` 复核，未安装包、未重跑训练。
