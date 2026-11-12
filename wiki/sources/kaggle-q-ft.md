---
title: "Q-FT：把表格行当嵌入喂给 Qwen 的 Tabular Transformer（Kaggle S6E10 讨论帖）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - kaggle
  - tabular
  - deep-learning
  - transformer
  - feature-embedding
  - lora
  - fine-tuning
  - llm
source_path: raw/articles/kaggle/How to build a Tabular Transformer!.md
source_kind: article
status: draft
---

# Q-FT：把表格行当嵌入喂给 Qwen 的 Tabular Transformer

## 元数据

| 项目 | 值 |
| --- | --- |
| 原标题 | *How to build a Tabular Transformer!*（副标题 *Q-FT: a Qwen transformer that reads tabular rows as embeddings*） |
| 作者 | Chris Deotte（Kaggle `cdeotte`），页面自述「1st in this Competition」 |
| 载体 | Kaggle 讨论区技术帖：<https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009> |
| 赛事 | Kaggle Playground Series **Season 6 Episode 10**（Predicting Airline Passenger Satisfaction），指标 **ROC AUC** |
| 本地文件 | `raw/articles/kaggle/How to build a Tabular Transformer!.md`（**198 行**、11,287 B、`wc -w` 1,863 词） |
| SHA-256 | `e0344940ccf62dd91bfead8bbd9e38f8d1113bf415db29946e1283f07e922eca` |
| 剪藏日期 | `created: 2026-11-11`（frontmatter）；正文只有相对时间「Posted a month ago」（L13），**剪藏未保存原帖发布日期** |
| 语言预训练权重 | `Qwen/Qwen3-0.6B-Base`（L113） |
| 代码 | **无**。剪藏内 fenced code block 计数为 0，只有一个行内 LoRA 公式（L99）；讨论区亦无仓库链接 |
| 插图 | 4 张，托管在 `raw.githubusercontent.com/cdeotte/Kaggle_Images`；已入库 `raw/assets/kaggle-q-ft/`，**逐张目视核查**（见「图表核查」） |
| 系列位置 | Q-FT 系列**第一篇**；正文 UPDATE 指向第二篇（L153–L155），本库另有 [[kaggle-q-ft-v2]] |

**来源定位**。这不是论文，也不是基准报告，而是一篇**工程配方帖**：它回答的是「怎么把一张表塞进一个语言模型」，并给出一次比赛里的单次 5 折结果。它的价值在于三步都很具体——**行→token 的编码方案、LoRA 的挂载方式、训练超参**；它的证据只有一名选手的一次划分、一个种子、无代码，因此本库把它当**可复现性未知的方法描述**，而不是「LLM 打败 GBDT」的证据（见「局限」与「分歧」）。

**页码说明**：本页 `L##` 指剪藏文件的行号，不对应原帖的任何分节编号。

## 结构

| 行 | 内容 |
| --- | --- |
| L1–9 | 剪藏 frontmatter（`author`、`published` 为空） |
| L11–14 | 赛事面包屑、作者、相对时间、H3 标题 |
| L16–25 | Q-FT 简介与「At a glance」四个数字 |
| L27–37 | §1 一张图讲清思路（每个 feature 变成一个 1,024 维向量） |
| L39–48 | §2 一个特征如何变成一个 token（三段式嵌入 + RMSNorm） |
| L50–62 | 「How each column is coded」表：21 列的编码方式与词表大小 |
| L64–80 | 「Worked example」：训练集第 0 行的 21 个 code 与 4 个数值 |
| L82–93 | §3 把向量送进 Qwen：`inputs_embeds`、随机序、CLS、因果掩码 |
| L95–106 | §4 LoRA 的代数、挂载位置与参数账本 |
| L108–121 | §5 训练配方表（数据、优化器、日程、精度、成本） |
| L123–146 | §6 结果：fold-0 消融表 + 5 折 OOF 对比表 + 逐折分数 |
| L148–155 | §7 结论与指向第二篇的 UPDATE |
| L157–198 | 评论区 8 条（含作者对「去掉因果掩码」的回复） |

## 关键主张

### 1. 任务与规模：raw features only，每折 55.97 万行

- 数据每折 **559,708** 行 `outer-train`，`StratifiedKFold`、5 折、`seed=42`，留出折只预测不训练（L112）。
- 「raw features only」= 不构造任何特征工程列，直接用 21 个原始列（L22、L142）。
- 反推：`559,708 + 139,927 = 699,635`，且 `699,635 / 5 = 139,927` 整除 → **训练集 699,635 行、每折 139,927 行**（本库推算，剪藏未直接给出）。

### 2. 一行 = 22 个位置：21 个特征 token + 1 个 CLS

- 每个特征变成 **1,024** 维向量（与 Qwen 词向量同宽），拼成 `21 × 1,024` 的矩阵，「stands in for a sentence」（L44–L46、L80）。
- 训练时**每行每步重排** 21 个 token，再把 CLS 追加在末尾（L84）。
- 只有最后的位置（CLS）能看到全部 21 个特征，因为 Qwen 用的是因果注意力（L85–L86）；隐藏状态过 `Linear(1024 → 1)` 得到 logit（L86）。

### 3. 三段式特征嵌入 + RMSNorm 对齐

每个特征的向量是三部分之和（L41–L46）：

1. **列嵌入**：表示「这是哪一列」，用列名的 Qwen 词向量（如 "Flight Distance" 各 token 向量的均值）**初始化**后继续训练；
2. **值嵌入**：按该特征的 code 查表；
3. **数值嵌入**（只给 Age、Flight Distance、两个 Delay）：标准化后的 $x$ 过 $\sin(2\pi f x)$/$\cos(2\pi f x)$（**32 个可学习频率**），再进一个小 MLP（64 → 1,024）。

最后 **RMSNorm × gain** 把向量的 RMS 压到 **≈ 0.030**——作者称这是「Qwen 典型词向量的大小」，让嵌入向量的尺度与普通 token 一致（L46）。周期特征这一步作者点名与 RealMLP 同源（L46）。

### 4. 21 列的编码方案是这篇最可复用的部分

| 列 | 编码 | 词表 | 数值嵌入 |
| --- | --- | --- | --- |
| Gender / Customer Type / Type of Travel / Class | 类别 code | 2 / 2 / 2 / 3 | 否 |
| 13 个服务评分（wifi … cleanliness） | 评分等级 0–5 | 各 6 | 否 |
| Age | 每个不同年龄一个 code | 75 | 是（标准化年龄） |
| Flight Distance | 出现 **≥3 次**的距离各给一个 route id，其余归「rare」 | 3,301 | 是（标准化 log 距离） |
| Departure / Arrival Delay | 分桶 `{0, 1–5, 6–15, 16–30, 31–60, 61–120, >120, missing}` | 各 8 | 是（标准化 $\log(1+\text{minutes})$） |

来源对 Flight Distance 的解释是**双编码的理由**（L61）：它在这个数据里表现得像 *route ID*（每个距离有自己的乘客构成），所以 ID 让模型学「逐路由效应」，而数值让模型在相邻距离之间共享信息（L61）。

**泄漏边界**：词表「只用 train + test 的特征构造（label-free）」（L52）——即**传导式（transductive）**词表，用到了测试集特征，但不含标签。

### 5. 走一遍训练集第 0 行

L64–L80 给出的具体映射（本库原样记录，用于核对实现）：Gender=Female→0、Customer Type=Loyal→0、Age=36→29（$x=-0.217$）、Type of Travel=Personal→1、Class=Eco→1、Flight Distance=694→436（$x=-0.594$）、13 个评分原值即 code、两个 Delay 都→0（$x=-0.289$ / $-0.281$）、标签 `satisfaction = False` 从不作为输入。

注意 Age 的 code 是 29 而数值是 $-0.217$：**code 是词表下标，数值是另一条通道**，两者同时进入同一个 token。

### 6. `inputs_embeds` 是关键接口

Hugging Face 模型可以直接接收向量而非 token id，从而**跳过词嵌入表**（L84）。这正是「不需要把行写成文本」的技术前提：模型看到的是 22 个 1,024 维向量，而不是字符串。

### 7. 随机重排是这篇里唯一幅度足够大的消融

固定顺序 vs 每步重排：**0.953546 vs 0.960444**，差 **0.0069 AUC**（L93、L134）。作者的解释是：因果 Transformer 逐个读特征，固定顺序会让模型绑定到某一个序列；每行重排相当于告诉模型「顺序无意义」，同时充当强数据增强（L93）。

这是全文唯一一个**大于折间波动**的差值（五折跨度 0.00164），因此也是唯一可以放心接受的效果量（见「分歧」第 3 条）。

### 8. LoRA：只训练 1,010 万参数

- 冻结全部权重 $W$，学习低秩修正 $W x + \frac{\alpha}{r} B(Ax)$，$r=16$、$\alpha=32$、$\alpha/r=2$（L99–L101）。
- **$B$ 初始化为零**，因此第 0 步等价于原始 Qwen（L102）。
- 适配器挂在**每层全部 7 个投影**上：attention 的 q/k/v/o 与 MLP 的 gate/up/down（L103）。
- 可训练参数 **10.1M**（约占骨干 2%），另有 **3.9M** 在 feature tokenizer 与 CLS/头（L104）；冻结的 0.6B 仍要跑前向与反向。

### 9. 训练配方（可直接抄的部分）

| 项目 | 值（L112–L121） |
| --- | --- |
| 损失 | CLS logit 上的二元交叉熵 |
| 优化器 | AdamW(fused)；LoRA 适配器 lr **2e-4**，tokenizer/CLS/头 lr **1e-3**；**无 weight decay** |
| 日程 | **3 epochs = 3,279 steps**（batch 512），3% 线性 warmup 后线性衰减到 0 |
| 精度 | Transformer bf16 autocast，**fp32 master weights**，梯度裁剪 1.0 |
| 增强 | 每步每行重新随机特征顺序 |
| 推理 | 对 **5 种特征顺序**（表原顺序 + 4 个随机）的 logit 取均值 |
| 成本 | ~920 rows/s、26 GB 峰值显存、单 A100 每折 ~52 分钟（5 折在 5 张卡上并行） |

### 10. 结果：与调参 XGBoost 打平，低于两个 TFM

**fold-0 消融**（L127–L134，全部 3 epochs）：

| 变体 | Fold-0 AUC | 来源给的解读 |
| --- | --- | --- |
| LoRA r16 + 预训练 Qwen | **0.960444** | best |
| 随机初始化骨干（同架构）+ 全量训练 | 0.960436 | pretraining adds nothing |
| 冻结骨干（只训 tokenizer + 头） | 0.959788 | 冻结的 LLM 已经是像样的特征混合器 |
| 随机列嵌入（不用列名） | 0.958479 | names don't matter |
| 全量微调，lr 2e-5 | 0.958004 | 学习率对全量微调太低 |
| 固定特征顺序 | 0.953546 | 重排必不可少 |

**5 折 OOF**（L138–L146）：

| 模型 | OOF AUC |
| --- | --- |
| TabPFN-3.5（raw features，微调） | 0.961248 |
| Kumo-Tabular Large（raw features，微调） | 0.960378 |
| **Q-FT（LoRA，raw features only）** | **0.959795** |
| 调参 XGBoost（raw features，8 seeds） | 0.959852 |
| 调参 RealMLP（raw features，8 seeds） | 0.959311 |

逐折：0.959602 / 0.959070 / 0.960712 / 0.960388 / 0.959283（L146）。

## 外部核查（剪藏之外，2026-11-12）

### Qwen3-0.6B-Base 的配置逐项比对

取 <https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json>（HTTP 200）：

| 项 | 来源声称（L113） | `config.json` 实测 |
| --- | --- | --- |
| 层数 | 28 | `num_hidden_layers` = 28 ✓ |
| 宽度 | 1,024 | `hidden_size` = 1024 ✓ |
| attention 头 | 16 | `num_attention_heads` = 16 ✓ |
| KV 头 | 8 | `num_key_value_heads` = 8 ✓ |
| MLP 宽度 | 3,072 | `intermediate_size` = 3072 ✓ |
| 未提 | — | `head_dim` = **128**、`tie_word_embeddings` = **true**、`attention_bias` = false、`vocab_size` = 151,936 |

结论：来源对骨干的描述**逐项正确**，缺的 `head_dim` 与权重绑定可以从下面的参数账本反推出来。

### 参数账本手算复核（这是本次最有说服力的核查）

按 `config.json` 的尺寸重算，三个数字全部对上：

- **LoRA**：每层 $r(1024+2048)$(q) + $r(1024+1024)$(k) + $r(1024+1024)$(v) + $r(2048+1024)$(o) + 3 × $r(1024+3072)$(gate/up/down) = 360,448；× 28 层 = **10,092,544** ≈ 文中「10.1M」（L104）✓
- **骨干**：每层 ≈ 15.73M（含 q/k 的 head-wise RMSNorm），× 28 = **440,466,432** ≈ 文中「about 440M weights in its 28 layers」（L97）✓——注意这是**不含**（被 tie 的）词嵌入表的口径；连表合计 **596,048,896** ≈ 0.6B ✓
- **feature tokenizer**：值嵌入词表 = 9 + 13×6 + 75 + 3,301 + 16 = **3,479** 个 code × 1,024 = 3,562,496；加 21 个列嵌入、21 个 RMSNorm gain、4 个数值 MLP（64 → 1,024，见 `feature_to_token.png` 的 `64 → ReLU → 1,024`）、CLS 与头 ≈ **3.87M** ≈ 文中「3.9M」（L104）✓

即：来源的参数量**不是估的，是可以从词表大小与模型配置精确重算的**。四条独立路径互相印证，说明这篇的作者至少在这个层面上算得很实。

### 图 4 反证 head 维度

`lora.png` 里 W 标注为「e.g. 1,024 × 2,048 for q_proj」。q 的输入维度 1,024、输出 2,048 = 16 heads × 128 `head_dim`，与 `config.json` 一致；8 个 KV 头 × 128 = 1,024，正好对应 k/v 的 LoRA 参数量。**图形与配置、与手算三方一致。**

### 被引用模型的存在性

- `Prior-Labs/tabpfn_3_5`（HF，created 2026-09-09）存在；
- `nvidia/Kumo-Tabular`（HF，created 2026-09-01）存在。

因此对比表里的模型名不是虚构。但「Large」变体、以及「fine-tuned」的具体配方**无法核实**；两行的分数同样无法核实。

### 无法核实的事项

Kaggle 原页用 `curl` 取得 **HTTP 200 但仅 5,728 B 的 JS 外壳**，正文由前端渲染 → **发布日期、票数、名次、评论区完整名单均不可独立核实**。`posted a month ago` 只能换算到「约 2026-10」，故本页不写 `published`。

## 图表核查（4 张，全部入库并逐张目视）

1. **`pipeline.png`**（1,800 × 600）：`One table row`（含 `label: satisfied?`）→ `Feature tokenizer`（「1 vector per feature / 21 × 1,024 numbers / trainable (3.9M)」）→ `Shuffle + add CLS`（「random order per row; CLS last」）→ `Qwen3-0.6B`（「28 decoder layers」，layer 1/2/28 标 `(+LoRA)`，注「dashed = frozen weights」）→ `h at CLS`（`Linear → logit → sigmoid → P(satisfied)`）。**图中把 3.9M 明确挂在 feature tokenizer 上**，正文 L104 则把 3.9M 说成「feature tokenizer 与 CLS/头」——这是两处口径的轻微差异（见分歧 9）。
2. **`feature_to_token.png`**（1,800 × 660）：`Flight Distance = 694` 同时送进三个分支——列嵌入（`e_col["Flight Distance"]`，`init: Qwen vectors of the name`）、值嵌入（`route id 436 → table row`，`table: 3,301 × 1,024`）、数值嵌入（`x = standardized log(694) = −0.594`，`[sin, cos](2π f x), 32 freqs → 64 → ReLU → 1,024`）——三路求和后 `RMSNorm × gain (≈ 0.030)` → `t ∈ ℝ^1024`。**数值 MLP 是单层 `64 → 1,024`**（与我的 3.87M 手算一致，不是两层）；`3,301 × 1,024` 与 `route id 436`、`−0.594` 三处数字与正文表格、worked example 全部吻合。
3. **`sequence_attention.png`**（1,800 × 800）：上半是「Sequence for one row (random order) → 22 positions」，token 依序为 Class / Seat comf. / Age / Flight D. / Wifi / … / Gender / Arr. delay / **CLS**（绿色，注「head reads this position」）；下半是 **22 × 22** 的下三角因果掩码，末行整行橙色，图例写「CLS row: attends to all 21 features」。图内文字补充「At prediction time the logit is averaged over 5 orders: the table's order + 4 random ones」。**图证实掩码确实是 22 个位置，且 CLS 位于最后一位**。
4. **`lora.png`**（1,800 × 459）：`x`（input 1,024）分两路——上 `W (frozen, pretrained)`（`e.g. 1,024 × 2,048 for q_proj`）、下 `A (1,024 → 16)` → `B (init 0) (16 → 2,048)`，B 路标 `× α/r = 2`，两路相加 → `output: W·x + 2·B·A·x`。**图把「B 为零」与「α/r = 2」都画在路径上**，与正文 L101–L102 一致。

四张图**没有任何一处与正文的定量描述冲突**；唯一的信息增量是「3.9M 归属」（图→tokenizer，正文→tokenizer + CLS/头，但两者都约等于 3.9M，因为 CLS 与头只有约 2K）与「数值 MLP 只有一层」。

## 转换损失

- **零代码**：全篇 fenced code block = 0，只有 L99 一个行内公式。因此「Q-FT」在剪藏里是**配方文字**而非实现，任何复现都要自行写 tokenizer、`inputs_embeds` 拼装、逐行重排与 5 序平均。
- **公式只剩一处**：LoRA 的 $Wx + \frac{\alpha}{r}B(Ax)$ 是行内代码；周期特征、RMSNorm、交叉熵都只有文字描述，没有公式。
- **图片原为外链**：4 张图在剪藏里都是 `raw.githubusercontent.com` 外链，本次已按所有者授权入库到 `raw/assets/kaggle-q-ft/`（详见上节 hash 与尺寸）。
- **元数据缺失**：`author`/`published` 为空；正文的「Posted a month ago」是相对时间。
- **评论区被压平**：8 条评论的作者链接与相对时间保留，但嵌套/点赞数与「Topic Author」标记混进正文流（L157–L198），且末条评论断在「Posted a month ago」之后。
- **原帖分节编号**：正文里 §1–§7 的编号在剪藏中保留（`## 1\.` 形式），说明这是原文编号而非剪藏产生。

## 分歧

1. **「ties a tuned XGBoost」严格来说是「略低于」**：Q-FT 0.959795 vs XGB 0.959852，差 **5.7e-5**（L142–L143）。同表里 TabPFN-3.5 高出 **0.00145**。结论段（L149）用的是「works great / ties」，但同页数据不支持「优于」。
2. **0.959795 不是五折均值**：五折分数均值是 **0.959811**（L146），比报告的 OOF 高 1.6e-5。这说明报告的 OOF 是**合并预测的 AUC**，而不是「各折 AUC 的均值」——两种口径在文中从未区分，容易被读者当成同一个数。
3. **「pretraining adds nothing」不可分辨**：0.960444 vs 0.960436，差 **8e-6**；而同一次运行的五折跨度是 **0.00164**，相差约 200 倍。而且两行**可训练参数集不同**（LoRA 10.1M vs 随机初始化后的全量训练），不是单一变量对照。评论区（L183）也正是从这里推出「收益来自架构而非语言预训练」——同样的证据，两种解释都无法被单折数据排除。
4. **「names don't matter」这一行配置缺失**：L132 写「Random column embeddings (no column names) 0.958479 — names don't matter (**compare full FT 0.958004**)」。它对比的是**全量微调**那一行，而不是同 LoRA 配置下的「有列名 vs 无列名」。因此这一行同时改了两个变量（列嵌入初始化、训练方式未知），**不能支持「列名不重要」**。真正可比的对照应是「LoRA + 随机列嵌入」，表中没有。
5. **「learning rate too low」是归因假设**：全量微调 lr 2e-5 的 0.958004 被解释成「学习率太低」，但没有 lr 扫描；也可能是 3 epochs 对全量微调不够，或没有 warmup 适配。
6. **成本不自洽**：920 rows/s × 3 epochs × 559,708 行 ≈ **30.4 分钟**，而报告是 **~52 分钟/折**（L25、L121）。差 1.7 倍，来源未解释（吞吐可能只算了训练循环，未含验证、嵌入构建与 dataloader）。**第二篇把同一口径改成了「5 折共 ~50 GPU-分钟」**（见 [[kaggle-q-ft-v2]]），说明作者自己意识到了这个比较方式的问题。
7. **步数隐含 `drop_last=True`**：`3 × floor(559,708 / 512) = 3,279`，而 `3 × 559,708 = 1,679,124` 行 → 实跑 `3,279 × 512 = 1,678,848` 行，每 epoch 丢 276 行（0.05%）。公式可逆，但正文没写 `drop_last`。
8. **位置措辞差一**：L84「Append a learned CLS vector as position 22」与 L85「rotary positions (0…21)」。总共 22 个位置，CLS 是**第 22 个**（0-based 下标 21）。`sequence_attention.png` 画的正是 22 个位置、CLS 在最后一位，所以是措辞问题而不是设计问题。
9. **3.9M 的归属两处口径不同**：如图表核查所述，`pipeline.png` 把 3.9M 标在 feature tokenizer 上，正文说是「feature tokenizer 与 CLS/头」。差额约 2K，不影响结论，但两处说法不一致。
10. **推理用「表原顺序 + 4 随机」与「顺序无意义」有张力**：如果模型真的学到了置换不变，取 5 种顺序的均值应当只是降噪；文中既说「shuffling teaches it that order is meaningless」（L93），又把**原顺序**放进平均（L120），说明作者并不完全相信模型学到了不变性。第二篇用编码器 + `position_ids=0` 把这件事从「训练技巧」变成「结构约束」，可以看作对这条张力的回答。

## 局限

- **单次划分、单种子**：一个 `StratifiedKFold(5, seed=42)`，所有对比共享同一个划分；没有重复划分方差，也没有多种子。
- **无代码、无 OOF/test 预测文件**：方法描述到「配方」为止，复现需自行补齐全部实现细节；评论区的索取（第二篇）与作者的回复都没有提供代码。
- **无置信区间**：所有差值都在 1e-4 量级，而可估的折间波动是 1e-3 量级。
- **传导式词表**：词表用 train + test 特征构造（L52）。label-free，但 CV 分数不再是「只用 80% 数据」的模拟。
- **对比对象口径不同**：TFM 两行是「raw features, fine-tuned」，Q-FT 是「raw features only」自训 LoRA；XGBoost/RealMLP 是「8 seeds」平均。三种「raw features」含义不同。
- **成本口径**：5 卡并行 → 墙上 52 分钟不等于单卡 52 分钟（见分歧 6）。
- **是竞赛帖，不是可评审的基准**：没有协议声明、没有代码、没有第三方复现。

## 待解问题

1. 去掉因果掩码（双向注意力 + 平均池化）会怎样？评论区（L183–L187）问了，作者答「I have not tried that yet」——**第二篇正是从这里出发**，见 [[kaggle-q-ft-v2]]。
2. RMS 目标 0.030 这个常数有多敏感？它是按 Qwen 词向量的经验尺度选的，换骨干就要重选。
3. `route id ≥ 3 次` 的阈值、delay 的 8 个分桶边界，是调过的还是拍的？文中无消融。
4. 5 种顺序平均 vs 单一顺序推理，差多少？（若接近 0，则「顺序不变性」已被学到；若不是，则是集成效应。）
5. LoRA rank 只试了 16；r=8/32/64 与 α 的交互未知。
6. 列嵌入用列名词向量初始化，与随机初始化在同一 LoRA 配置下的对照缺失（见分歧 4）。
7. 若把 21 列的编码换成「数值列也只用 code、类别列也加数值嵌入」，结果如何？当前的不对称（Age 有 code+数值、评分只有 code）是数据驱动的还是随手定的？
8. 这套 tokenizer 换到别的赛题（多分类、回归、缺失值多、类别基数高）是否仍然可用？本文只有一道二分类题。
9. 「冻结骨干已经是像样的特征混合器」（0.959788）意味着什么？如果冻结骨干就有 0.9598，那 10.1M LoRA 只贡献了 0.0007——这与「预训练无用」的结论方向相反，值得单独验证。

## 行号索引（`raw/articles/kaggle/How to build a Tabular Transformer!.md`）

| 行 | 内容 |
| --- | --- |
| L13 | 作者、名次、相对时间 |
| L22 / L24 / L25 | 0.959795 / 10.1M / ~52 min |
| L41–L46 | 三段式嵌入；RMSNorm ≈ 0.030；RealMLP 同源 |
| L52 | 词表来自 train + test（label-free） |
| L54–L60 | 21 列编码表 |
| L66–L78 | worked example（第 0 行） |
| L61 | Flight Distance 为什么同时要 ID 与数值 |
| L84–L86 | `inputs_embeds`、随机序、CLS、因果注意力 |
| L93 | 固定顺序损失 0.0069 |
| L99–L104 | LoRA 代数、7 个投影、10.1M + 3.9M |
| L112–L121 | 训练配方表 |
| L127–L134 | fold-0 消融（6 行） |
| L138–L146 | 5 折 OOF 对比（5 行）+ 逐折分数 |
| L148–L155 | 结论与 Part 2 链接 |
| L157–L198 | 评论区（8 条） |

## 与其它页面的关系

- 系列第二篇：[[kaggle-q-ft-v2]]（编码器 + `position_ids=0` + side features + RTD）
- 概念：[[tabular-deep-learning]]（本页的编码方案被整理进「LLM 当骨干」与「冻结骨干 + 低秩适配」两节）
- 实践：[[tabular-model-selection]]（作为第五个候选基线的成本与准入条件）
- 赛事流程：[[competition-ml-workflow]]、[[leaderboard-vs-cv]]
- 对照：[[gbdt-vs-tabular-dl]]、[[ft-transformer]]（同一条「每列一个 token」路线，但无位置编码 + 全量训练）
