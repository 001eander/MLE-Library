---
title: "TFMs and LLMs with tabular data：S6E10 第一名方案（Detective 与 Storyteller 两种文本化 GPT 分类器）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - kaggle
  - tabular
  - deep-learning
  - transformer
  - llm
  - generative-model
  - stacking
  - competition
source_path: raw/articles/kaggle/TFMs and LLMs with tabular data!.md
source_kind: article
status: draft
---

# TFM 与 LLM 读表格：target last 与 target first 两种「文本化 GPT 分类器」

## 元数据

| 项目 | 值 |
| --- | --- |
| 原标题 | *1st Place - TFMs and LLMs with tabular data!* |
| 作者 | Chris Deotte（Kaggle `cdeotte`），自述 1 名（L10、L14） |
| 载体 | Kaggle Playground S6E10 方案帖：<https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data> |
| 赛事 | Kaggle Playground Series **Season 6 Episode 10**，*Predicting Airline Passenger Satisfaction*，指标 **ROC AUC** |
| 本地文件 | `raw/articles/kaggle/TFMs and LLMs with tabular data!.md`（**261 行**、18,235 B、2,658 词、末行无换行） |
| SHA-256 | `fa18b2e03552e177147bafe93314f128ed133546067bf2a6bfad4b2248cafc17` |
| 剪藏日期 | frontmatter `created: 2026-11-11`；正文无绝对日期。原页 `curl` 只返回 **6,165 B 的 JS 外壳**，**名次与发布日不可独立核实** |
| 代码 | **无仓库、无脚本**。5 个 fenced block（3 个 `dart` 文本示例、2 个 Python 片段）+ 1 个公式块（L97–L100） |
| 插图 | **11 张**，已入库 `raw/assets/kaggle-tfms-llms-tabular/`，**逐张目视核查**（见「图表核查」） |
| 系列位置 | Q-FT 系列**第三篇**（总写）。前两篇：[[kaggle-q-ft]]（L33 引用）、[[kaggle-q-ft-v2]]（L34 引用） |

**来源定位**。前两篇讲的是「每列变成一个嵌入向量」，这一篇把方向彻底换掉：**不训练 feature tokenizer，直接把整行写成文本交给 GPT**，唯一的设计选择变成 **target 放在哪里**。它对本库的价值集中在三处：

1. 给出**生成式分类器**（$P(x\mid y)$ 经 Bayes 反转）在表格任务上的一个完整可抄配方，并把「判别式 vs 生成式」从教科书概念变成一张可对照的分数表；
2. 给出**公共 OOF/test pair 池**的 stacking 工程细节（logit 化、clip、折内 refit），以及「单一模型 >> 245 个模型的 stack」这种反直觉读数；
3. 把「**数据本身是被语言模型写出来的**」当作一条可检验的建模前提，并据此解释为什么某个模型会赢。

**页码说明**：本页 `L##` 指剪藏文件行号；图注中的数字来自目视核查，均已标明出处文件名。

## 结构

| 行 | 内容 |
| --- | --- |
| L1–9 | 剪藏 frontmatter |
| L10–L17 | 标题、赛事、导语与两个 headline 数字 |
| L19–L27 | §1 致谢：245 对 pair 的来源、作者公开的三个数据集/笔记本 |
| L29–L55 | §2 引言、两种文本化分类器的对照表、行的文本格式 |
| L57–L82 | §2.1 target last（Detective）：训练、推理代码、row 0 读数、定位为判别式 |
| L84–L150 | §2.2 target first（Storyteller）：Bayes 公式、打分代码、四步配方、worked example、三条细节 |
| L152–L158 | §2.3 二者怎么选 |
| L160–L168 | §3 公共 OOF/test pair 与 stacking 协议 |
| L170–L191 | §4 9 个 TFM + 两个 GBDT/DL 基线、26 对 TFM pair 的边际收益 |
| L193–L206 | §5 「这份数据是语言模型写出来的」：GReaT 假设与三条证据 |
| L208–L222 | §6 单模型分数表 |
| L224–L243 | §7 加进 stack 的历史、最终选择 |
| L245–L256 | §8 四类「foundation 风格」模型 |
| L258–L262 | 引用信息 + 评论区外壳 |

## 关键主张

### 1. 两种「把行写成文本」的分类器

对照表（L42–L48）与图 `two_directions.png` 是同一件事的两种表达：

| | Tabular LLM #1：target LAST | Tabular LLM #2：target FIRST |
| --- | --- | --- |
| 昵称 | the **Detective** | the **Storyteller** |
| 它在问什么 | 「线索读完了，答案是什么？」 | 「逐个试每种结局，哪种让这个故事最可能？」 |
| 建模对象 | $P(\text{target}\mid\text{features})$ | $P(\text{features}\mid\text{target})$，用 Bayes 反转 |
| 每行前向次数 | 1 | 2（每个 target 值）× 若干特征顺序 |
| 本赛题 | ~0.961 AUC | **0.9624 AUC**，作者称「最好的单模型」 |

两种都用「任意小 GPT」，作者用的是 **GPT-2（124M）**（L50）。行的写法统一为 `"列 is 值"` 用 `", "` 连接（L52–L55）。

### 2. Detective：把 target 放在末尾，读下一个 token 的概率

- **训练**：按「特征在前、标签在后」的文本微调因果语言模型；loss 可以只放在标签 token 上，也可以放在全部 token 上（L67，图 `target_last.png` 同句）。
- **增强**：**每一步都打乱特征顺序**——「表行没有天然顺序，这教会模型把行读成集合，也是很好的数据增强」（L67）。
- **推理**：把文本截到 `satisfaction is`，取下一个 token 的概率，再对 `True` / `False` 归一（L69–L76）。图 `target_last.png` 补了两个正文没写的细节：合法的下一个 token 是**带前导空格的** `" True"` / `" False"`，其余 token 概率 0.000。
- **定位**：这是**判别式**分类器，「像 XGBoost 一样直接学 $P(\text{target}\mid\text{features})$」（L82），每行一次前向，本赛题约 0.961。

### 3. Storyteller：把 target 放在最前，用 Bayes 反转

公式（L95–L100）：

$$\text{log-odds}(y=1)=\underbrace{\big[\log P(y=1)+\log P(x\mid y=1)\big]}_{\text{story 1}}-\underbrace{\big[\log P(y=0)+\log P(x\mid y=0)\big]}_{\text{story 0}},\qquad P=\sigma(\text{log-odds})$$

打分代码（L104–L117）的要点有三：**只对特征 token 求和**（`tok_lp[n_prefix - 1:]`，跳过 target 前缀）；两个 target 值各跑一次整段前向；先验以 logit 形式相加（`math.log(prior / (1 - prior))`，`prior = 0.4436`）。

### 4. 四步配方（这是全篇最可复用的一段）

1. **Replica**：在**原始数据集**（129,880 行的 Airline Passenger Satisfaction）上微调 GPT-2，行写成 `"列 is 值"` + 每行随机列序；作者给的最省事写法是 GReaT 库 `GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)`（L123）。**起点决定上限**：在完成第 2 步之后，60-epoch 的 replica 比 8-epoch 的高 **+1.2e-3 AUC**，而直接用原始 GPT-2 差得多（L124–L125）。
2. **Classifier**：每折继续训练，**target 在前**、21 个特征每个 step 重新随机排序、loss 放在全部 token 上；8 epochs、AdamW、lr 5e-5 线性衰减到 0、batch 32、bf16；GPT-2 small 约 600 rows/s，单张 A100 **约 2 小时/折**（L126–L128）。
3. **打分两次**：每行算 $\log P(x\mid y=1)$ 与 $\log P(x\mid y=0)$（L129）。
4. **对特征顺序平均**：模型见过随机顺序，所以问它若干个顺序并用 log-sum-exp 合并；**30 阶**就拿到大部分收益（L130–L132，图 `orders_curve.png`）。

### 5. Worked example（本库第一个「可逐项复算」的生成式分类读数）

图 `row0_evidence.png` 把 row 0（真标签 `False`）的证据拆成逐字段的 $+\log p(\text{field}\mid\text{True})-\log p(\text{field}\mid\text{False})$（单位 nats）：

| 项 | 证据（nats） |
| --- | --- |
| prior：$\log P(\text{True})-\log P(\text{False})$ | −0.20 |
| Type of Travel = Personal Travel | **−2.04** |
| Inflight wifi service = 4 | **+1.34** |
| Age = 36 | −0.44 |
| Customer Type = Loyal Customer | +0.26 |
| Flight Distance = 694 | +0.11 |
| 其余 16 个字段合计 | −0.09 |
| **log-odds（一个顺序）** | **−1.05** → $P(\text{satisfied})=0.26$ |

正文的读数：单顺序 −1.05 → 0.26；30 阶 log-sum-exp −1.07 → **0.256**；「两个模型一致：Detective 对同一行给 0.23」（L134–L141）。

### 6. 三条「会决定成败」的细节

- **格式必须与原始数据一致**：整数写 `36`、浮点写 `0.0`、缺失写 `None`、标签写 `True`/`False`（L145）。
- **分词器要用 GPT-2 自己的**：本赛题里「逐位数字分词器（如 Qwen）的模型更差」（L146）。
- **变体用来做多样性**：GPT-2 medium/large 分数相近但错误不同；**target anywhere**（训练时标签在 22 个字段中的随机位置）是可叠加的第三种变体，按随机顺序平均 log-odds 打分（L147–L150）。

### 7. 公共 OOF/test pair 池的 stacking 协议

每个 pair = `train.csv` 的折外预测 + 测试集预测，全部用同一套 `StratifiedKFold(5, shuffle=True, random_state=42)`（L162）。合并协议（L164–L166）：

1. 预测值一律转 logit（先 clip 到 `1e-15`，再 clip 到 $\pm30$；已经是 logit 的直接用）；
2. 在 4 折的 OOF logit 上拟合 **L2 逻辑回归**（cuML，`C=1`），预测第 5 折；
3. 测试侧对 5 个折模型取平均——顺带抹掉 cuML 的 run-to-run 噪声。

结果：最好的单对约 **0.9617**；245 对 stack 得 **CV 0.96218 / public LB 0.96186**。作者自陈一条口径警告：**部分公开成员在被打分的那一折上早停**，所以公共 CV 略偏乐观（L168）。

### 8. 9 个 TFM 与两个基线（L172–L191）

| 模型 | OOF AUC | | 模型 | OOF AUC |
| --- | --- | --- | --- | --- |
| TabPFN-3.5 | **0.96115** | | EXAONE | 0.95856 |
| Kumo-Tabular large | 0.96004 | | Kumo-Tabular small | 0.95848 |
| XGBoost（baseline） | 0.95985 | | LimiX-2 | 0.95847 |
| RealMLP（baseline） | 0.95931 | | Causilo | 0.95828 |
| TabFM | 0.95887 | | TabICL-2 | 0.95809 |
| | | | Mitra-2 | 0.95707 |
| | | | **9 个 TFM 一起 stack** | **0.96120** |

作者的机制说明：TFM 靠「把训练行当上下文读进来」（in-context learning）预测，归纳偏置与树不同，因此 stack 得好（L189）。社区把这条推得更远，例如把 Flight Distance 声明为类别型「route ID」的 TabPFN-3.5。在公共池里，**26 对 TFM pair** 给另外 219 对再加 **+1.4e-4 CV**（0.96204 → 0.96218）；某对公开 TabPFN-3.5 还是最终 stack 里**权重最大的非 GPT 成员**（L191）。

### 9. 「这份数据是语言模型写出来的」（全篇的机制核心）

Kaggle Playground 数据是合成的。作者认为它由 **GReaT**（Borisov et al. 2023, arXiv:2210.06280）生成，机制两步（L195–L198）：先在原数据集上把每行写成 `"列 is 值"`（随机列序）微调 GPT-2，再逐 token 采样新行。

三条证据（L200–L204）：

| 证据 | 内容 |
| --- | --- |
| **GPT-2 分词器指纹** | 3 位数距离是单个 token，4 位数是共享前缀的 2 个 token |
| **生成设置** | 以 temperature 0.7、top-k 50 采样，能复现该数据集相对原始数据的扭曲 |
| **标签最先** | 标签是**第一个**被生成的；label-first 的提示能精确复现训练集的标签率 |

推论：既然每行是「先标签、再 21 个特征逐 token」，那么**最优分类器就是生成器自己对『哪个标签让这一行最可能』的回答**；Storyteller 训的正是这个问题，所以它比任何「把特征当数字读」的模型都更接近生成器（L206）。

### 10. 单模型分数表与「加进 stack」的历史

| 单模型（L210–L220） | OOF AUC |
| --- | --- |
| GPT-2 small，target first（单独 public LB **0.96189**） | **0.96242** |
| GPT-2 small，target anywhere，24 epochs | **0.96245** |
| GPT-2 medium，target first | 0.96237 |
| GPT-2 large，target first | 0.96236 |
| GPT-Neo 125M，target first | 0.96216 |
| target last（同族模型） | ~0.961 |
| 最好的公开 pair | ~0.9617 |
| 调参 XGBoost（作者自己的） | 0.96149 |
| TabPFN-3.5 | 0.96115 |

stack 历史（L228–L236）：245 对 → 248 对（+GPT-2 small、+自家 XGB/RealMLP）CV 0.96266 / LB 0.96223 → 247 对（去掉 XGB/RealMLP、+GPT-2 large）0.96277 / 0.96242 → 250 对（+target-anywhere、+medium）0.96282 / 0.96249 → 259 对 0.96290 / **0.96262**。作者的三条解读：加**一个** GPT 分类器给 245 个强 pair 带来 **+4.9e-4 CV、+3.7e-4 LB**；原因是它**准确且与其它一切都很不像**（它建模 $P(x\mid y)$ 而不是 $P(y\mid x)$）；最终 stack 里 **10 个 GPT 系 pair 承担约 80% 的输出**。最终提交两条：259 对（public LB 最好）与 332 对（CV 最高 0.96299，多 73 对来自合作者）。

### 11. 四类「foundation 风格」表格模型

§8 与图 `tabular_model_family.png` 给的分类法是本库目前最简洁的一张路线图：

| 家族 | 代表 | 建模对象 |
| --- | --- | --- |
| 表格基础模型（TFM） | TabPFN、TabICL、Kumo、LimiX、Mitra | 把训练行当上下文 |
| LLM + 表格嵌入 | Q-FT、Q-FT v2（[[kaggle-q-ft]]、[[kaggle-q-ft-v2]]） | 每列一个向量 → transformer，把行读成集合 |
| LLM + 文本，target last | Detective | $P(\text{target}\mid\text{features})$ |
| LLM + 文本，target first | Storyteller | $P(\text{features}\mid\text{target})$ |

作者的主张是「这四类现在都能超过调参 GBDT」（L249）——**这一条与本篇自己的表冲突**，见「分歧」第 3 条。

## 本库核算（数值是否闭合）

| 检查 | 计算 | 结论 |
| --- | --- | --- |
| 每折训练行数 | 前两篇给出 `699,635` 行、5 折 outer-train `559,708`（= 0.8 × 699,635） | 与 L126–L128 的「8 epochs / 600 rows/s / ~2 h」自洽：`8 × 559,708 / 600 = 7,463 s ≈ 2.07 h` ✓ |
| Replica 成本 | `60 × 129,880 = 7,792,800` 行；按 600 rows/s 应为 **3.6 h**，图 `target_first_recipe.png` 写 **≈10 h** | **差 2.8×**（batch 16 而非 32 可解释一部分），见分歧 8 |
| pair 池闭合 | `219 + 26 = 245` ✓；`0.96204 → 0.96218 = +1.4e-4` ✓；245 → 248 = +3（GPT small + 自家 XGB + RealMLP）✓；248 → 247 = −2 +1 ✓ | 最后两行不闭合，见分歧 6 |
| 单模型 vs 245 对 stack | 单模型 OOF `0.96242` vs stack CV `0.96218` → **+2.4e-4**；public LB `0.96189` vs `0.96186` → **+3e-5** | 「打平」只在 LB 上成立，见分歧 5 |
| 图内先验一致性 | `−69.40 − (−68.55) = −0.85`；`−0.85 + (−0.20) = −1.05` ✓；`σ(−1.05) = 0.259` ✓；`σ(−1.07) = 0.255` ✓ | 图之间自洽；**代码的 `prior = 0.4436` 对应 −0.227**，与图上的 −0.20 不符，见分歧 7 |
| waterfall 求和 | `−0.20 −2.04 +1.34 −0.44 +0.26 +0.11 −0.09 = −1.06` vs 图上 **−1.05** | 两项小数舍入误差，不影响结论 |
| Detective 归一 | `0.225 / (0.225+0.775) = 0.225` → 图与正文写 **0.23** | 舍入口径（half-up），非错误 |
| fold-0 vs OOF | 顺序曲线在 fold-0 上 30 阶 **0.96260**、96 阶 **0.96265**；同模型 OOF `0.96242` | fold-0 高 **+1.8e-4 ~ +2.3e-4**，与前两篇同向（+6.5e-4、+1.8e-4） |
| TFM 层级 | 9 个 TFM stack `0.96120` 仅比最好的单个 TFM `0.96115` 高 **+5e-5**；两者都低于调参 XGBoost `0.96149` | TFM 之间错误高度相关；「TFM 超过调参 GBDT」不成立 |

## 图表核查（11 张，全部入库并逐张目视）

| 文件 | 尺寸 / 字节 | 目视读数 |
| --- | --- | --- |
| `banner.png` | 1914×822 / 1,662,620 | 装饰图：「Read the row the way it was written」+「1st place · Playground S6E10」，机器人拿「Passenger Survey」读「Age is 36 / Class is Eco / wifi is 4」。无定量内容 |
| `two_directions.png` | 1960×800 / 189,820 | 两栏总图。左：`p(True)=0.225`、`p(False)=0.775`、$P=p(T)/(p(T)+p(F))$、「discriminative, like XGBoost」。右：**`Σ log p` True = −69.40、False = −68.55**，log-odds 展开式、「generative, then flipped with Bayes' rule」。**这两个 log-prob 绝对值只出现在图里** |
| `detective.png` | 1672×941 / 1,845,745 | 装饰图：侦探机器人 + 「satisfied? … **False! (77%)**」= $1-0.225=0.775$ ✓ |
| `target_last.png` | 1960×600 / 125,740 | 流程：行文本（特征在前）→ GPT-2 微调（「last position →」）→ 下一个 token 概率（`" True"` 0.225 / `" False"` 0.775 / **anything else 0.000**）→ 归一得 0.23。图注「fine-tune: loss on the label token (or on all tokens)」、「one forward pass per row」 |
| `storyteller.png` | 1672×941 / 1,854,748 | 装饰图：两个「believability」仪表盘，Ending True / Ending False 两本书，同一组字段 |
| `target_first_recipe.png` | 1960×660 / 152,423 | 四步：Replica（**once，≈10 h，1 GPU**）、Classifier（**per fold，≈2 h，GPT-2 small**）、Score twice（2 texts per row per order）、Many orders（**6 → 30 orders：+6e-4 AUC**）。**「≈10 h」与「+6e-4」只出现在图里** |
| `orders_curve.png` | 1440×720 / 76,303 | 横轴 log 刻度 6/10/16/30/48/64/96/128，纵轴 **fold-0 AUC**：0.96198 / 0.96217 / 0.96236 / **0.96260** / 0.96262 / 0.96264 / **0.96265** / 0.96263；箭头标「we use 30」。**30 阶之后基本平坦，128 阶略降** |
| `row0_evidence.png` | 1960×919 / 107,461 | waterfall：prior −0.20、Personal Travel −2.04、wifi=4 +1.34、Age −0.44、Loyal +0.26、Flight 694 +0.11、其余 16 字段 −0.09 → −1.05 → 0.26。横轴定义「evidence = log p(field\|True) − log p(field\|False)（nats）」 |
| `two_roads.png` | 1672×941 / 1,831,179 | 装饰图：`features → target` 与 `target → features` 两块路牌 + 「Same model, same row, opposite directions」 |
| `stack_history.png` | 1800×760 / 122,010 | 两条线（CV、public LB）六个点：`245 public pairs` 0.96218/0.96186 → `one GPT classifier (alone)` **0.96242**/0.96189 → `+GPT-2 small` 0.96266/0.96223 → `+GPT-2 large` 0.96277/0.96242 → `+medium & label-anywhere` 0.96282/0.96249 → `final stack (259 pairs)` 0.96290/0.96262 |
| `tabular_model_family.png` | 1960×500 / 111,331 | 四栏：TFM（「learn from the training rows in context」）、LLM+embeddings（Q-FT, v2）、LLM+text target last、LLM+text target first。标题写的是「four kinds of **"foundation-style"** tabular models」 |

**图与正文的互补**：正文把「30 阶顺序」和「+1.2e-3 replica」当成一句话带过，而 **+6e-4（6→30 阶）** 与 **≈10 h（replica）** 只在图里；`Σ log p = −69.40 / −68.55` 也只在图里，它同时说明**分类信号是两个 −69 量级数字之间 0.85 nats 的差**（见待解问题 6）。

## 转换损失

- **无代码**。5 个 block 全是行内片段：`log_p` 的 `n_prefix - 1` 依赖「target 前缀的 token 数是否含末位空格」，而 `row` 字符串本身以空格开头（L113），这类字符级细节没有任何说明。
- **优先级歧义**：`row` 以空格开头（L113），而第 2 节的行定义是 `"列 is 值"` 用 `", "` 连接（L50）；两种拼法在 GPT-2 分词下是否同形，来源未交代。
- **无逐折数字、无种子、无方差**。全篇最细的粒度是 OOF 单值。
- **「245 对 pair」无清单**：只点名感谢了 6 个贡献者，没有逐对来源、模型名或分数。
- **TFM 缺规格**：9 个模型的版本、规模（Kumo 的 large/small 到底是多大）、是否在本赛题上微调、推理配置全部未写。
- **评论区只剩一行外壳**（L262：「HotnessThis comment will be made public once posted.」）——评论内容全部丢失，这与前两篇能被压平成正文流不同。
- **图片原为外链**：11 张已入库 `raw/assets/kaggle-tfms-llms-tabular/`（本轮经所有者授权）。
- **数据规模未写**：正文从未给 `train.csv` 行数；本库用前两篇的 `699,635` 反推（见「本库核算」）。

## 跨来源冲突（与已入库两篇同赛题来源的对照）

三篇是同一赛题、同一作者，可以直接对表。**下表是本页最需要注意的一节**：

| 模型 | 本篇 | [[kaggle-q-ft]] / [[kaggle-q-ft-v2]] |
| --- | --- | --- |
| XGBoost **baseline**（本文 L178） | **0.95985** | — |
| Q-FT 所称「**调参** XGBoost」 | — | **0.959852** |
| 调参 XGBoost（**作者自己的**，本文 L219） | **0.96149** | — |
| RealMLP **baseline**（本文 L179） | **0.95931** | — |
| Q-FT 所称「**调参** RealMLP」 | — | **0.959311** |
| TabPFN-3.5 | 0.96115（L176、L220） | 0.961248 |
| Kumo-Tabular large | 0.96004（L177） | 0.960378 |

**两条基线的五位有效数字完全相同**（0.95985/0.959852、0.95931/0.959311）。这支持一个**假设**（非来源陈述）：Q-FT 两篇里被称为「调参 XGBoost / 调参 RealMLP」的对照，就是本篇这两行**未充分调参的 baseline**；若是这样，[[kaggle-q-ft]] 那句「ties a tuned XGBoost」应当改读为「比本篇的调参 XGBoost（0.96149）低约 1.7e-3」，而 Q-FT v1 与 v2 的定位也要相应下调。三篇都未说明基线的同一性，本库不覆盖旧结论，只把这条冲突留在这里与 [[tabular-model-selection]]、[[gbdt-vs-tabular-dl]]。

## 分歧

1. **「target first 是最好的单模型」与自己的表冲突**。L48 与 L212 称 target-first 0.96242 为最佳单模型，但 L213 的 **target-anywhere 24 epochs = 0.96245** 更高（+3e-5）。而 §5 的机制论证预测的恰恰是「label 最先 = 生成器顺序」，与「标签位置随机更好」方向相反；来源没有讨论这 3e-5。
2. **target-last 的读数不是纯 target-last 模型**。L80 自注该 worked example 的模型「trained with the target at random positions and read with the target last」，所以 L48/L217 的「~0.961」混入了 target-anywhere 训练；§2.3 却把它当作 target-last 的独立对照。
3. **「四类模型都能超过调参 GBDT」被本篇自己的表否证**（L249 vs L174–L187、L219）。调参 XGBoost 0.96149 > 最好的 TFM 0.96115 > 9 个 TFM 的 stack 0.96120；**只有文本化的 GPT 分类器（0.96242）真正超过了调参 XGB**。
4. **顺序平均曲线的纵轴是 fold-0 AUC，与 §6 的 OOF 表不同口径**。曲线最高点 **0.96265**（96 阶）高于全篇所有 OOF 数字（最高 0.96245），两者不可直接比较；而「用 30 阶」这个选择也是在同一折上做的（轻度的选择偏差）。
5. **「一个模型打平 245 对 stack」只在 public LB 上成立**（0.96189 vs 0.96186，+3e-5）。在 CV 上单模型 **高 +2.4e-4**（0.96242 vs 0.96218）——按作者自己的口径（CV 才是稳健的、且公共 CV 还偏乐观），这里应该说的是「单个模型超过 245 对 stack」。
6. **pair 数不闭合**。247 → 250 只列了两项新增（target-anywhere、GPT-2 medium）却 +3 对；「10 个 GPT 系 pair 承担约 80% 的输出」也没给口径（$|\text{权重}|$ 之和？非零权重个数？）。
7. **先验有两个取值**。代码写 `prior = 0.4436`（→ logit **−0.227**，L105），而 `row0_evidence.png` 与 `two_directions.png` 的 prior 项是 **−0.20**（→ 先验 ≈ 0.450）。差 0.027 nats，不影响 0.26 的结论，但两处并非同一次计算。
8. **replica 成本对不上**。图 `target_first_recipe.png` 写「once（≈10 h，1 GPU）」，而 60 epochs × 129,880 行按 L128 的 600 rows/s 只要 **3.6 h**；若要 10 h，等效吞吐得降到 ~217 rows/s。
9. **「逐位数字分词器（Qwen）更差」是归因假设**。Qwen 与 GPT-2 small 在参数量、预训练语料、词表规模、分词粒度上同时不同，本篇只改了分词器这一个叙事变量。
10. **GReaT 的认定强度有限**。「3 位数是一个 token、4 位数是两个 token」这条指纹只能说明**采样器用了 GPT-2 词表**，任何以 GPT-2 为骨干的生成器都会留下它；temperature 0.7 / top-k 50 与「label 最先」是行为证据，不是生成器身份证据。
11. **「最优分类器就是生成器的答案」忽略了非归一化**。对顺序做 log-sum-exp 得不到归一化似然（缺 $Z$），所以 Storyteller 只是生成器的一个**单调分数**近似，而不是 Bayes 最优分类器本身（见 [[generative-classification]]）。

## 局限

- **单赛题、单次 5 折、无种子数组**：所有被当作结论的差值都在 1e-4～1e-3，而前两篇给出同一赛题的五折跨度是 1.6e-3 量级。
- **数据是合成的 Playground 数据**。作者自己在 §5 说明这一点，但没有给出任何**非合成**数据的对照；因此「target first 在真实数据上是否仍占优」在本篇内无法回答。
- **公共 pair 池的偏差只能定性**：「部分成员在被打分的那一折早停」意味着 CV 偏乐观，但偏多少没量化。
- **无代码**。可复现的入口只有三个公开数据集/笔记本（124 对 pair、11 对 TFM pair、stacking notebook，L27），本库未下载。
- **名次不可核实**：原页返回 JS 外壳（6,165 B），「1st place」只能记为作者自述。
- **单个硬件、单个精度**：单 A100、bf16、batch 32；成本与显存不可迁移，且**推理成本完全没报**（target first 每行要 $2\times30=60$ 次前向）。

## 待解问题

1. 换成**真实（非合成）**表格数据，Storyteller 相对 target-last 的增益还在吗？这是「§5 机制解释」唯一的判决性实验，而来源没有做。
2. 30 阶 log-sum-exp 是对顺序的什么近似？若要输出**校准概率**或优化 log loss，缺的归一化常数该怎么补（重要性采样？固定一组顺序？）？
3. 「60-epoch replica 比 8-epoch 高 +1.2e-3」是同折同种子的对照吗？这个效应**大于本篇其余所有增益之和**，却只有一条子项。
4. 245 对公共 pair 里有多少在被打分的那一折早停？把 pair 池的 CV 偏差量化出来，会不会改变「加一个 GPT 分类器 +4.9e-4」的解读？
5. 「10 个 GPT 系 pair 承担约 80% 的输出」用的是哪个口径？换成权重绝对值之和、非零个数、或 Shapley，结论是否一致？
6. bf16 下把 −69.40 与 −68.55 相减（0.85 nats）稳定吗？换 fp32 或用 log-sum-exp 的数值稳定写法，分数会变吗？
7. 把行文本改成**与 GReaT 训练时逐字符一致**（同样的分隔符、小数位、缺失写法）还能再涨吗？这是「格式」变量的直接检验。
8. §4 的 9 个 TFM 是纯 in-context 推理，还是在本赛题上微调过？（前两篇把 TabPFN-3.5 记为「raw features 微调」，本篇 L189 说 TFM「learn from the training rows in context」）
9. 「6 → 30 阶 = +6e-4」只在 fold-0 上测过；换到第 2–5 折是否复现？
10. 若把 245 对公共 pair 按**模型族**分组先组内 stack、再组间 stack，是否比一次性 245 维逻辑回归更好？作者的「pair 越多越稳」（L243）与「245 对大部分互相抵消」（L240）并不完全相容。

## 行号索引（`raw/articles/kaggle/TFMs and LLMs with tabular data!.md`）

| 行 | 内容 |
| --- | --- |
| L14–L17 | 赛事、两个 headline（单模型 LB 0.96189、加进 stack 后 0.96223/0.96262） |
| L21–L27 | 245 对 pair 的致谢名单与作者公开的三个数据集 |
| L31–L38 | 引用前两篇 Q-FT、提出「不需要 feature tokenizer」 |
| L42–L48 | 两种分类器的对照表 |
| L50–L55 | GPT-2 124M、行的文本格式 |
| L61–L82 | target last：训练、推理代码、row 0 读数、判别式定位 |
| L88–L100 | target first：行格式、Bayes 公式 |
| L104–L117 | 打分代码（prior 0.4436、`log_p`） |
| L121–L132 | 四步配方（replica / classifier / score twice / orders） |
| L134–L141 | worked example：−1.05 → 0.26、30 阶 −1.07 → 0.256、Detective 0.23 |
| L143–L150 | 格式、分词器、变体三条细节 |
| L152–L158 | 二者怎么选；「target first 在数据由 LM 写出时赢」 |
| L162–L168 | stacking 协议与 245 对结果、早停警告 |
| L172–L191 | 9 个 TFM + 2 个基线、26 对 TFM pair 的边际收益 |
| L195–L206 | GReaT 假设与三条证据 |
| L210–L222 | 单模型分数表与差值叙事 |
| L228–L243 | stack 历史表、+4.9e-4、80% 权重、两种最终选择 |
| L245–L256 | 四类 foundation 风格模型 |
| L258–L262 | 引用信息与评论区外壳 |

## 与其它页面的关系

- 系列前两篇：[[kaggle-q-ft]]（Qwen + LoRA + 重排）、[[kaggle-q-ft-v2]]（order-free ModernBERT + side features + RTD）
- 概念：[[generative-classification]]（本页的 $P(x\mid y)$ 路线是它的新实例）、[[tabular-deep-learning]]、[[ensembles]]、[[leaderboard-vs-cv]]
- 实践：[[tabular-model-selection]]、[[competition-ml-workflow]]
- 对照：[[gbdt-vs-tabular-dl]]（同一赛题的四类模型阶梯）、[[gaussian-discriminant-analysis]]、[[naive-bayes]]（生成式分类的经典实例）
