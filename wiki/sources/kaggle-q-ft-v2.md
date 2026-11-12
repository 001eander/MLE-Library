---
title: "Q-FT v2：把表格行读成集合的 order-free ModernBERT（Kaggle S6E10 讨论帖 Part 2）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - kaggle
  - tabular
  - deep-learning
  - transformer
  - pre-training
  - encoder
  - feature-embedding
  - llm
source_path: raw/articles/kaggle/Part 2 - How to build a Tabular Transformer!.md
source_kind: article
status: draft
---

# Q-FT v2：把表格行读成集合的 order-free ModernBERT

## 元数据

| 项目 | 值 |
| --- | --- |
| 原标题 | *Part 2 - How to build a Tabular Transformer!*（副标题 *Q-FT v2: an order-free ModernBERT that reads a table row as a set*） |
| 作者 | Chris Deotte（Kaggle `cdeotte`）；剪藏 L13 自述「1st in this Competition」 |
| 载体 | Kaggle 讨论区技术帖：<https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334> |
| 赛事 | Kaggle Playground Series **Season 6 Episode 10**，指标 **ROC AUC**；与本系列第一篇同一赛题 |
| 本地文件 | `raw/articles/kaggle/Part 2 - How to build a Tabular Transformer!.md`（**188 行**、12,509 B） |
| SHA-256 | `e56969e22604bc47a5991b7e450aa9fca6ebd98f80aea93f23d904fa217a8e1c` |
| 剪藏日期 | frontmatter `created: 2026-11-12`；正文写「Posted 22 days ago」（L13），**未保存原帖日期** |
| 骨干 | `answerdotai/ModernBERT-base`（L127）；另用「原始 airline satisfaction 数据集」作为外部数据 |
| 代码 | **无**。fenced code block = 0；评论区有人明确索要代码与 OOF/test 预测（L182 附近），未获提供 |
| 插图 | 4 张，已入库 `raw/assets/kaggle-q-ft-v2/`，**逐张目视核查**（见「图表核查」） |
| 系列位置 | Q-FT 系列**第二篇**；第一篇为 [[kaggle-q-ft]]，第三篇（1 名总写，已剪藏未入库）见 `raw/articles/kaggle/TFMs and LLMs with tabular data!.md` |

**来源定位**。这是同一作者对第一篇的**架构替换**：把「因果解码器 + 重排特征」换成「编码器 + 所有 token 位置编号为 0」，再补上两类信息（人工统计量、无标签预训练）。它对本库的价值有三层：**把「表是集合」从训练技巧变成结构约束**、**给出一个可抄的表格自监督预训练任务**、以及**又一次展示单折消融与全折 OOF 的差距**。证据强度与前篇相同：单次 5 折、单种子、无代码。

**页码说明**：本页 `L##` 指剪藏文件行号。原帖有两个小节都编号为 `## 7`（Results 与 Conclusions，L135、L147），本页按原样记录。

## 结构

| 行 | 内容 |
| --- | --- |
| L1–9 | 剪藏 frontmatter |
| L11–13 | 面包屑、「more_horiz」UI 残片、作者与相对时间 |
| L18–33 | 标题、「三处改动 +1e-3 / 便宜 5 倍」、At a glance 四个数字 |
| L35–46 | §1 与 Q-FT 的七项对照表 |
| L48–52 | §2 管线图 |
| L54–78 | §3 order-free：`position_ids = 0`、3e-6 检查、mean head、「试了没用」的三项 |
| L80–100 | §4 side features（9 个数）+ 2 个 summary token；+4e-4 |
| L102–120 | §5 RTD 预训练；+3e-4；两条细节 |
| L122–133 | §6 训练配方表 |
| L135–146 | §7 Results：fold-0 五步消融链 |
| L147–151 | §7 Conclusions（编号重复） |
| L153–188 | 评论区 3 条（含一条独立复现 + 一条索要代码） |

## 关键主张

### 1. 三个改动：编码器、side features、自监督预训练

来源把相对 Q-FT 的变化列为「三处改动 +1e-3 AUC、训练便宜约 5×」（L22）：

1. **order-free 编码器**：用双向 ModernBERT 替换因果 Qwen，并把**每个 token 的 `position_ids` 都设为 0**（L65–L66）；
2. **side features**：每个特征 token 额外接收 9 个关于该取值的数值，另加 2 个「summary token」概括整行（L80–L96）；
3. **在表上自监督预训练**：替换 token 检测（RTD），不使用任何标签（L102–L113）。

### 2. 七项对照（这是全篇最紧凑的一页）

| | Q-FT | v2（L37–L46） |
| --- | --- | --- |
| 骨干 | Qwen3-0.6B 解码器（因果） | ModernBERT-base 编码器（双向） |
| 特征顺序 | positions 0…21 + 每行重排 | 全部 `position_id = 0`，顺序在结构上无意义 |
| 读答案的位置 | 末尾 CLS | **23 个输出向量的均值** |
| 训练什么 | LoRA rank 16 | **全部 149M** |
| 输入 | 21 个特征 token | 21 特征 token + side features + **2 个 summary token** |
| 预训练 | 无（只有 Qwen 语言预训练） | **RTD** |
| 速度 / OOF | 920 rows/s、52 min/折；0.959795 | 2,930 rows/s、6.5 min/折；**0.960798** |

### 3. order-free 的实现细节：位置编号而不是换模型

- 因果解码器下顺序会泄漏：token 只能看更早的位置，答案必须在最后，位置编号让模型能学「第 3 个 token 是……」（L58–L59）。
- ModernBERT 用 RoPE，而 RoPE 由传入的 `position_ids` 驱动；**全部传 0**，则所有 token 看起来都没有位置（L66）。
- **验证**：打乱特征顺序后输出变化约 **3e-6**，「即浮点误差」（L68）。`v2_order_free.png` 里把这条写在图上。
- 池化：`mean(23 个非 CLS 输出) → Linear(768 → 1)`；CLS head、attention-pooling、mean+max **在噪声内相同**，故选最简（L72）。

**「试过但没用」的三项**（L76–L78）：ModernBERT-large 不比 base 好；DeBERTa-v3（base 与 large）更差；**随机初始化的 ModernBERT 与预训练的相同**——作者由此再申一次「语言预训练不是这套方法起作用的原因」。

### 4. side features：9 个数字，4 类来源

每个特征 token 额外乘一个可学习的 **9 × 768** 矩阵（每列一个），加到 token 上（L82–L90）：

1. **取值有多常见**：该值在 train + test 的计数、在原始数据集中的计数、以及两者比值；
2. **路由画像**：Flight Distance 在这个数据里像 route ID——同一距离上该列的均值与离散度，以及原始数据集的均值；
3. **在这个行里有多典型**：一个**无标签**模型用其余 20 列预测该列，取「期望值」与「残差（实际 − 期望）」；
4. **原始数据的意见**：「satisfied」在该确切取值上的**平滑 log-odds**（用原始数据集的标签）。

2 个 summary token（L93–L94）：

- **行统计（28 个数）**：13 个评分的和/均值/离散度/计数、延迟差、若干阶跃点 → `Linear` → 一个 token；
- **原始数据块（17 个数）**：只在原始数据集上训练的一个模型的预测，加上「与该行在 Flight Distance + Age + Class 等键上匹配的原始行」的统计 → `Linear` → 一个 token。

**效果：fold-0 约 +4e-4，两个种子一致**（L100）。作者同时给出一个**关键的负面对照**：同类特征给 **GBDT 约 +1.6e-3**，是 transformer 收益的近 4 倍；他的解释是「transformer 的逐值嵌入已经学到了 counts 与路由画像能说的东西」（L100）。

### 5. RTD：用 ELECTRA 的思路给表做自监督

1. 取一行，对 21 个值中约 **15%** 用**同一列、随机另一行**的值替换（L105–L107）；
2. 对每个 token 判断「真 / 被替换」，一个小 `Linear(768→1)` 头 + BCE（L107–L108）；
3. **不用标签**，所以可以在 **train + test + 原始数据集 = 1.13M 行**上预训练（L111）；
4. 因为不看标签，预训练**与折无关**：只跑一次（3 epochs，18 分钟），5 折都从同一份权重开始（L112）；
5. 预训练结束后丢掉检测头，再在 satisfaction 上微调（L113）。

**效果约 +3e-4**（L117），两个种子一致。两条附带的发现：RTD 之后微调**更早到峰值**，所以 epoch 从 3 降到 2（第 3 个 epoch 过拟合，L119）；**10 epochs 反而更差**，而「用原始数据集做有监督预训练完全没用」（L120）。

### 6. 训练配方

| 项目 | 值（L126–L133） |
| --- | --- |
| 每折数据 | 559,708 行 outer-train（`StratifiedKFold` 5、seed 42） |
| 骨干 | 22 层、宽 768、12 头；双向；**所有 token `position_ids = 0`** |
| 每行 token | CLS + 21 特征 + 2 summary = **24** |
| 头 | `mean(23 个非 CLS 输出) → Linear(768 → 1)` |
| 预训练（一次） | RTD，15% 替换、逐 token BCE、**1.13M 行 × 3 epochs**、batch 512 |
| 微调 | BCE；**全部权重**；AdamW，ModernBERT lr **5e-5**、tokenizer 与头 lr **1e-3**；batch 512；**2 epochs**；3% warmup 后线性衰减；bf16 autocast；梯度裁剪 1.0 |
| 推理 | **一次前向**（order-free，无需顺序平均） |
| 成本 | ~2,930 rows/s、11.4 GB 峰值显存、**每折 ~6.5 分钟 + 共享预训练 18 分钟**，单 A100 |

### 7. 结果：fold-0 五步链

| 变体 | Fold-0 AUC（L141–L145） | 相对上一步 |
| --- | --- | --- |
| Q-FT：Qwen3-0.6B、因果、LoRA、重排 | 0.960444 | — |
| Qwen3-0.6B + 双向注意力 + mean head | 0.960602 | **+1.58e-4** |
| ModernBERT-base、order-free、mean head | 0.960368 | **−2.34e-4** |
| + side features | 0.960655 | +2.87e-4 |
| **+ RTD 预训练** | **0.960977** | **+3.22e-4** |

来源的文字数字是「side features +4e-4」「RTD +3e-4」（L100、L117，各自「两个种子一致」），而上表实测是 +2.87e-4 与 +3.22e-4。两者不完全一致，见「分歧」。

### 8. 成本：三种「快了几倍」并存

来源同时给出四个速度量（本库换算）：

| 口径 | 变化 | 倍数 |
| --- | --- | --- |
| 吞吐（rows/s） | 920 → 2,930 | **3.18×** |
| 图上标的速度 | — | **3.6×**（`v2_order_free.png`） |
| 每折墙上时间 | 52 min → 6.5 min | **8.0×**（但 epoch 数从 3 降到 2） |
| 5 折 GPU-分钟 | ~260 → ~50 | **5.2×** |

八个数字里有三个不同来源的加速比，文中未对齐（见「分歧」第 5 条）。但**这一次的成本内部自洽**：`559,708 / 2,930 = 191 s/epoch`，× 2 epochs = **6.4 分钟** ≈ 报告的 6.5 分钟 ✓（第一篇的「52 分钟」与「920 rows/s」之间差 1.7 倍，见 [[kaggle-q-ft]] 分歧 6）。

## 外部核查（剪藏之外，2026-11-12）

### ModernBERT-base 配置与参数量

取 <https://huggingface.co/answerdotai/ModernBERT-base/raw/main/config.json>（HTTP 200）与模型 API 的 `safetensors` 字段：

| 项 | 来源声称（L127） | 实测 |
| --- | --- | --- |
| 层数 | 22 | `num_hidden_layers` = 22 ✓ |
| 宽度 | 768 | `hidden_size` = 768 ✓ |
| attention 头 | 12 | `num_attention_heads` = 12 ✓ |
| 参数量 | 149M（L32） | `safetensors.total` = **149,655,232** ✓ |
| 未提 | — | `intermediate_size` = 1,152、`local_attention` = **128**、`global_attn_every_n_layers` = **3**、`global_rope_theta` = 160,000、`local_rope_theta` = 10,000、`vocab_size` = 50,368 |

**参数量还原**：按非 GeGLU 的直觉算（`Wi: 768→1152`）会得到 **129,583,104**，比实测少 **20.07M**；若按 **GeGLU**（`Wi: 768→2×1152`）算，得 **149,047,296**，与实测 149,655,232 只差 0.4%（剩余部分在 embedding 之后的 norm/pooler/头）。因此「149M」是对的，同时它也**暴露了一个剪藏里没写的事实：ModernBERT 的 FFN 是 GeGLU 门控结构**。本库第一次手算时用的是非门控公式，因此低估了 20M——这条记在这里，供后来者不要重复。

### 与「every token attends to every other token」的冲突

L65 写「ModernBERT is an encoder: every token attends to every other token（bidirectional attention）」，`v2_pipeline.png` 也标「bidirectional attention」。但配置里 `local_attention = 128`、`global_attn_every_n_layers = 3` → ModernBERT 实际是**局部滑窗（128）与全局注意力交替**。

在本任务里这不改变结论：一行只有 **24** 个 token < 128，局部窗口覆盖整行，**局部≈全局**。所以正确说法是「24 个 token 的行内，双向注意力实际上退化为全连接」，而不是「ModernBERT 天然全连接」。这条值得写进分歧，因为它影响「换更长的行会不会失效」的判断。

### 与第一篇的成本口径对照

第一篇报「每折 52 分钟」而 5 折并行运行；第二篇把比较改成 **GPU-分钟（260 vs 50）**，这才是同口径。本库据此认为第二篇的成本对比**可用**，第一篇的成本对比**不可用**。

## 图表核查（4 张，全部入库并逐张目视）

1. **`v2_pipeline.png`**（1,800 × 660）：`One table row` → `Feature tokenizer + side features`（「21 feature tokens + 2 summary tokens / each 768 numbers / trainable」）→ `Add CLS`（「every token at **position 0**」「no shuffling: the row is a set」）→ `ModernBERT-base`（「22 layers, width 768」「bidirectional attention」「all 149M weights trained」「pretrained with RTD」）→ `mean of the 23 outputs`（`Linear → logit → sigmoid → P(satisfied)`）。底部图例「green = new since the Q-FT post」，绿色标在 tokenizer 与 RTD 上。**图里的 149M 与 HF 实测一致**。
2. **`v2_order_free.png`**（1,800 × 780）：左「Before (Q-FT, decoder): causal mask + positions 0, 1, 2, …」，8×8 下三角（蓝）；右「Now (ModernBERT, encoder): full attention + EVERY token at position 0」，8×8 全绿。右侧文字：「no shuffling, no order averaging, **3.6× faster**」「output is identical for any feature order (checked: changes ~3e-6 under shuffling)」。**注意这是 8×8 的示意图，不是 22/24 个位置的真实尺寸**（对比第一篇的 `sequence_attention.png` 画了完整的 22 个位置）。
3. **`v2_token.png`**（1,800 × 860）：`Flight Distance = 694` 分四路——列嵌入（`which column this is` / `init: name word vectors`）、值嵌入（`route id 436 → table row` / `one vector per value`）、数值嵌入（`sin/cos(2π f x) → MLP` / `x = standardized log(694)`）、**Side features (NEW)**（`9 numbers about this value`：`counts · route profile · typicality · original-data log-odds → × W`）——四路求和 → `RMSNorm × gain → t ∈ ℝ^768`。底部虚线框「Plus 2 SUMMARY tokens for the whole row (NEW)」：`row statistics (28 numbers …) → Linear → one token`、`original-dataset block (17 numbers …) → Linear → one token`。**注意这一版没画 `3,301 × 1,024` 的表尺寸**（第一篇画了），也没画 RMS 的具体数值（第一篇画了 ≈0.030）。
4. **`v2_rtd.png`**（1,800 × 720）：Step 1「corrupt: swap ~15% of the values with the same column of a random OTHER row」——示例行 Gender=Female、Age=36、**Class=Business**（红，原为 Eco）、Flight D.=694、Wifi=4、**Seat comf.=5**（红，原为 1）、Gate=3、Arr. delay=0，红字注「red = replaced」；→ `feature tokenizer → ModernBERT (same model as in the pipeline)`；Step 2「predict, for EVERY token: "was this value replaced?"」，逐 token 标 `real`/`replaced`；底部两条：「Uses only features: train + test + original rows (1.13M rows), never a label → done ONCE, shared by all folds」「3 epochs, 18 min on one A100. Then the detection head is dropped and the model is fine-tuned on satisfaction」。

四图与正文一致，且**图文互补**：正文说「for every token」，图只画了 8 个特征 token（含 `…`），**没有画 CLS 与 summary token 是否参与检测**（见待解问题 3）。

## 转换损失

- **零代码**：fenced block = 0。RTD 的「随机换值 → 逐 token 预测」听上去简单，但换值时是否保持同列分布、是否对同一行多次替换、summary/CLS token 是否也参与，这些决定实现细节的部分全靠文字。
- **无表格数字化**：fold-0 消融链以新表格给出（L139–L145），但 side features 的「9 个数」没有逐个定义到可算的地步（如「平滑 log-odds」的平滑强度、route profile 的「spread」是 std 还是 IQR），只能从 `v2_token.png` 的关键词推测。
- **无原始数据集的说明**：全篇未给「original airline satisfaction dataset」的来源、行数、与赛题数据的重叠情况，只在图上写「a model trained on the original data」。
- **图片原为外链**：4 张已入库 `raw/assets/kaggle-q-ft-v2/`。
- **评论区被压平**：3 条评论，其中一条是完整的第三方复现报告（含硬件、耗时、OOF/LB/BCE 数字），格式被压成单段。
- **分节编号重复**：两个 `## 7`（L135、L147）在剪藏中保留，判断为原帖笔误而非剪藏损失。

## 分歧

1. **「三处改动 +1e-3」与 fold-0 消融链不自洽**：OOF 差是 **+1.003e-3**（0.959795 → 0.960798），但同一页 fold-0 链的总差只有 **+5.33e-4**（0.960444 → 0.960977）。「三处改动」在 fold-0 上的合计效果约为导语宣称的一半。
2. **消融链不单调**：`Qwen 双向 + mean head`（0.960602）→ `ModernBERT order-free`（0.960368）是**退步 2.34e-4**，即「换编码器」这一步在 fold-0 上是负收益；正收益全部来自 side features（+2.87e-4）与 RTD（+3.22e-4）。导语把三者打包成 +1e-3，掩盖了这一步的符号。
3. **效果量与自报值不一致**：side features 自报「+4e-4」，表上是 **+2.87e-4**；RTD 自报「+3e-4」，表上 **+3.22e-4**。两者自报之和（7e-4）大于实测合计（5.33e-4）。
4. **「三处改动」与消融链的项数不对应**：链上有 4 步（双向化、换骨干、side features、RTD），导语说 3 处改动。其中「Qwen 双向 + mean head」这一步既不是「编码器」也不是「side features/RTD」。
5. **三种加速比并存**：吞吐 3.18×、图注 3.6×、每折墙上 8.0×、GPU-分钟 5.2×（见「关键主张 8」）。它们各自都能自圆其说（epoch 数变了、并行方式变了），但文中只写「about 5× cheaper」，读者容易误读到别的口径上。
6. **「every token attends to every other token」与 ModernBERT 的实际结构不符**：`local_attention=128` + 每 3 层一个全局层（见外部核查）。24 token 的行内不影响结论，但这是**结构描述错误**，且它恰好决定了「行变长时会不会失效」这个外推边界。
7. **「for EVERY token」的检测范围含糊**：正文说对每个 token 判断真/假，但 CLS 与 2 个 summary token 并不对应单个取值，图里也没画它们。若检测头只作用于 21 个特征 token，则「every」应改为「每个特征 token」。
8. **v2 已不是「raw features only」**：side features 第 4 类用原始数据集的**标签**（平滑 log-odds），summary token 用**在原始数据上训练的模型的预测**。因此 v1→v2 的 +1e-3 混入了「外部监督信息」这一变量，与 v1 那句「raw features only 就能打平调参 XGBoost」**不可直接比较**。
9. **RTD 用了测试集特征，且 5 折共享同一初始化**：label-free，所以不构成标签泄漏；但 CV 不再是「只用 80% 数据训练」的模拟。这一点正文只写「与折无关、只跑一次」的好处（L112），没讨论代价。
10. **fold-0 与 OOF 的乐观偏差**：fold-0 0.960977 vs OOF 0.960798，差 **+1.79e-4**（第一篇是 +6.5e-4）。方向一致，量级不同，说明单折试点不能替代全折报告。

## 局限

- **单次 5 折、单种子**；所有消融都在 **fold-0** 上，且「两个种子一致」只是文字断言，没有方差或逐种子数字。
- **无代码、无 OOF/test 预测**：评论区（L182 附近）明确索要「training code, fold assignments, OOF predictions and test predictions」，正文未回应。
- **外部数据未描述**：原始数据集的行数、来源、与赛题数据的重叠比例都没给；side features 的「平滑 log-odds」的平滑强度也未给。
- **第三方复现是评论级证据**：评论者 Naji 自述在 T4×2 上用 `nn.DataParallel`、以 `sdpa` 替换 FlashAttention-2，RTD 63 min + 5 折 137 min（总计约 3.5 小时），OOF **0.96075**、LB 0.96082、Train/Val BCE 0.2223/0.2233（L169–L175）。数字与作者接近，但**这是评论、无代码、未含外部原始数据集**（评论自述「Without even adding the external Original Dataset yet」）——只能当**独立复现的线索**，不能当核实。
- **与第一篇的可比性**：骨干、训练方式、输入、数据来源四项同时改变，不是受控对照。
- **成本仍是单硬件**：单 A100、bf16、batch 512；吞吐与显存不可直接迁移。

## 待解问题

1. 如果把 `position_ids=0` 换成「可学习的逐列位置嵌入」，结果会怎样？也就是说，「顺序无意义」是数据性质还是这个数据集的巧合？
2. 24 个 token 时局部窗口 ≥ 行长，所以「双向 = 全连接」。若行变得更长（例如 200 个特征 token），ModernBERT 的局部层会不会反而成为限制？这正是分歧 6 的外推边界。
3. RTD 的检测头作用在哪些 token 上？CLS 与 summary token 参与吗？（见分歧 7）
4. side features 的 9 个数具体怎么算？「平滑 log-odds」的平滑系数、route profile 的 spread 定义都缺失。
5. 「同类特征给 GBDT +1.6e-3、给 transformer 只有 +4e-4」这条对照是在什么配置下得到的？如果 GBDT 用了完全相同的 9 个 side features，那这是一条很强的机制证据；如果只是「相似的」特征，则只能当方向性暗示。
6. 外部数据集的标签用于 side features，是否在赛规上被允许？剪藏没有引用任何规则文本。
7. RTD 语料是「train + test + 原始」，那么把 test 换成「只用 train 特征」会掉多少分？这是判断「传导式预训练带来了多少不公平优势」的关键对照。
8. 为什么 CLS 保留但不参与池化？它只作为注意力中的「一个额外 token」存在，这与 v1 的「CLS 是答案槽」完全不同，是否有意为之？
9. 从 149M 全量微调换回 LoRA（参数少两个数量级）会掉多少？v1 的证据是「冻结骨干」只比 LoRA 低 0.0007，暗示 LoRA 的容量已足够——在编码器上是否仍成立？
10. 前篇与本篇都指向「tabular pretraining + in-context learning（类似 TFM）」（L151）。这条路线与 TabPFN/Kumo 的差距（本篇 0.960798 vs 0.961248/0.960378）是否主要由预训练语料规模决定？

## 行号索引（`raw/articles/kaggle/Part 2 - How to build a Tabular Transformer!.md`）

| 行 | 内容 |
| --- | --- |
| L13 | 作者、名次、相对时间 |
| L20–L22 | 三处改动 +1e-3、5× 便宜 |
| L30–L33 | 0.960798 / 24 tokens / 149M / ~6.5 min |
| L37–L46 | 七项对照表 |
| L54–L68 | order-free 推理、`position_ids = 0`、3e-6 检查 |
| L72 | mean head；三种池化在噪声内相同 |
| L76–L78 | large/DeBERTa-v3/随机初始化均无用 |
| L82–L90 | 9 个 side features 的四类 |
| L93–L94 | 2 个 summary token（28 / 17 个数） |
| L100 | +4e-4；GBDT 同类特征 +1.6e-3 |
| L105–L113 | RTD 五步 |
| L117–L120 | +3e-4；2 epochs；10 epochs 更差；有监督预训练无用 |
| L126–L133 | 训练配方表 |
| L139–L145 | fold-0 五步消融链 |
| L149–L151 | 结论与下一步 |
| L169–L175 | 第三方复现（评论） |

## 与其它页面的关系

- 系列第一篇：[[kaggle-q-ft]]（Qwen + LoRA + 重排）
- 概念：[[tabular-deep-learning]]（order-free 与「冻结骨干 / 低秩适配」两节都引本页）
- 实践：[[tabular-model-selection]]、[[competition-ml-workflow]]
- 评估：[[leaderboard-vs-cv]]（单折 vs 全折）
- 对照：[[gbdt-vs-tabular-dl]]（side features 的 4 倍差）、[[ft-transformer]]（无位置编码的另一条实现路径）
