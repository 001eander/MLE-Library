---
title: 表格任务选型：先立 GBDT 与 ResNet 两个基线，再决定要不要上 FT-Transformer
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - model-selection
  - gbdt
  - transformer
  - resnet
  - hyperparameter-tuning
sources:
  - "[[ft-transformer]]"
  - "[[kaggle-q-ft]]"
  - "[[kaggle-q-ft-v2]]"
  - "[[kaggle-tfms-llms-tabular]]"
  - "[[gandalf]]"
status: draft
verified: 2026-11-12
applies_to:
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13）
  - PyTorch >=1.8,<3
  - Qwen/Qwen3-0.6B-Base（config.json 于 2026-11-12 核对）
  - answerdotai/ModernBERT-base（safetensors.total = 149,655,232，2026-11-12 核对）
---

# 表格任务选型：先立 GBDT 与 ResNet 两个基线

本页把 [[ft-transformer]] 的实验协议转成可执行的选型流程。核心判据不是“哪个模型 SOTA”，而是**在你的数据形态、类别特征、特征数与预算下，哪个基线值得先跑、哪个模型值得加**。与 GBDT 的逐数据集对照见 [[gbdt-vs-tabular-dl]]；架构细节见 [[tabular-deep-learning]]。

> **适用**：中小规模表格数据、有固定评估指标、单机可跑、允许做 15 seeds × 若干配置的实验。
> **不适用**：文本/图像/序列为主的任务；在线延迟极紧、无法承受 FT-T 训练成本的场景；没有固定验证集、只能靠在线反馈的场景。

## 五个准入问题

在选模型前先回答：

1. **任务与指标是什么？** 二分类/多分类/回归；指标是 RMSE、accuracy、logloss 还是业务指标？指标决定预处理与输出头，也决定“top”怎么判定（[[evaluation-metrics]]）。
2. **$n$ 与 $k$ 各是多少？** 样本少、特征多时，929K 参数的 FT-T 可能过重；$k$ 很大时注意力成本是 $O(k^2)$。
3. **类别特征有多少、基数多大？** 本文的 11 个数据集里只有 Adult 有 8 个类别特征（[[ft-transformer]], p. 6）；类别特征为主时，GBDT/目标编码的证据更强。
4. **预算是多少？** 训练时间、调参时间、推理延迟。FT-T 相对 ResNet 的实测倍数是 0.9×–13.8×（表 10，p. 16），Yahoo（699 个数值特征，正文取整 700）达 13.8×。
5. **是否需要可微管线？** 需要端到端训练、embedding 复用、多任务头时，FT-T 的结构优势才明显；否则 GBDT 往往更省。

## 基线顺序：GBDT → MLP → ResNet → FT-T

来源的结论是“没有普适赢家”（[[ft-transformer]], p. 8）。因此推荐顺序是：

| 顺序 | 模型 | 作用 | 何时可以停 |
| --- | --- | --- | --- |
| 1 | **GBDT**（XGBoost/CatBoost） | 表格默认强基线；调参后常在 CA/AD/Yahoo 类数据上赢 | 如果 GBDT 已明显优于其它基线，且没有“必须可微”的需求 |
| 2 | **MLP** | sanity check；调参后平均排名 4.8，能击败多数复杂 DL 模型 | 如果 MLP 已接近 GBDT，说明数据更适合 MLP-like 模型 |
| 3 | **ResNet** | 比 MLP 更强的 MLP-like 基线；平均排名 3.3 | 如果 ResNet 已稳定优于 MLP，且没有“特征交互”需求 |
| 4 | **FT-Transformer** | 特征 token 化 + 注意力；平均排名 1.8 | 如果 FT-T 相对 ResNet 的增益足以抵消训练成本 |

**不要跳过 MLP/ResNet 直接上 FT-T**：来源明确说调参让 MLP/ResNet 变得有竞争力，且 FT-T 的优势主要集中在 GBDT 胜过 ResNet 的数据集上（[[ft-transformer]], p. 7–8）。跳过基线会让你把“调参不足”误判为“架构更好”。

## 第五个候选：LLM / 嵌入空间的表格 transformer

[[kaggle-q-ft]] 与 [[kaggle-q-ft-v2]] 代表另一条路线：不训小 Transformer，而是把每列变成一个与语言模型词向量**同宽**的向量，直接送进预训练好的骨干（Qwen3-0.6B，LoRA；或 ModernBERT-base，全量）。这条路线在同一个赛题上给出的数字是：

| 模型 | OOF AUC | 备注 |
| --- | --- | --- |
| TabPFN-3.5 | 0.961248 | 表格基础模型，raw features 微调 |
| Q-FT v2（ModernBERT + RTD） | 0.960798 | 24 token、149M 全量；**用了外部数据集** |
| Kumo-Tabular Large | 0.960378 | 表格基础模型 |
| 调参 XGBoost（8 seeds） | 0.959852 | |
| Q-FT（Qwen + LoRA） | 0.959795 | 21 token、10.1M 可训练；raw features only |
| 调参 RealMLP（8 seeds） | 0.959311 | |

（数字来自 [[kaggle-q-ft]], L138–L146 与 [[kaggle-q-ft-v2]], L30；**单次 5 折、单种子、无代码**。）

> **基线同一性存疑**：同一赛题的第三篇（[[kaggle-tfms-llms-tabular]]）把 **XGBoost 0.95985、RealMLP 0.95931** 列为本条目的 **baseline**（L178–L179），而自己的**调参** XGBoost 是 **0.96149**（L219）。两行数字与上表中「调参 XGBoost 0.959852」「调参 RealMLP 0.959311」**五位有效数字完全相同**。合理假设：上表的两个“调参”对照实际上是未充分调参的基线；如果是，Q-FT 两篇的「ties a tuned XGBoost」应改读为「比调参 XGBoost 低约 1.7e-3」。在这一点被澄清前，不要把这张表的 XGBoost/RealMLP 两行当成“调参上限”。TabPFN-3.5 与 Kumo 也有小幅差异（0.961248 vs 0.96115、0.960378 vs 0.96004），可能是不同运行的同一模型。

**先看结论再看成本**：这条路线在同一道题上**没有赢过调参 XGBoost**（v1 低 5.7e-5，v2 靠外部数据 + 自监督预训练才高出 0.00094），也低于两个表格基础模型。所以它不是“更强的基线”，而是一个**在特定数据形态下值得一试的候选**。

**四个准入问题**（四个都答“是”才继续）：

1. **样本量够大吗？** 两个来源都是 ~70 万行、2 万量级的特征基数。样本少时，149M 骨干 + 逐列词表会直接过拟合。
2. **列是“码”还是“数”？** 这套编码最占便宜的是**评分类有序离散列**（0–5 共 6 个码）、低基数类别列、以及“看起来像 ID”的连续列（Flight Distance 被当成 route ID）。全是高精度连续数值、列数很多的表不会因此受益。
3. **有 GPU 且能接受一次预训练开销吗？** v2：单 A100 每折 ~6.5 分钟 + 共享预训练 18 分钟（5 折共 ~50 GPU-分钟）；v1：每折 ~52 分钟（[[kaggle-q-ft-v2]], L22、L133）。**v2 的成本已经低于很多 FT-T 实验**。
4. **允许用外部数据/测试集特征吗？** v2 的 side features 用了原始数据集的**标签**，RTD 预训练用了**测试集特征**，且 5 折共享同一初始化（[[kaggle-q-ft-v2]], L111–L112）。这在竞赛里可能合规，在真实项目里需要单独判断。

**必须先写下来的四个决定**（不写就等于默认值替你决定）：

| 决定 | v1 的做法 | v2 的做法 | 不写会怎样 |
| --- | --- | --- | --- |
| 位置信息 | RoPE 0…21 + 每步重排 | 全部 `position_ids = 0` | 顺序变成模型可学的信号，消融看起来变成“重排很重要” |
| 池化 | 末尾 CLS | 23 个输出均值 | 因果掩码下只有末尾位置能看到全行 |
| 每列编码 | 类别词表 / 评分等级 / route id / 延迟分桶 | 同左 | 高基数连续列当普通数值会丢掉“路由效应” |
| 词表语料 | train + test（label-free） | 同左 | 只用 train 会让低频 code 在测试集上退化为 unknown |

**诊断信号**：

- **验证集好、测试集差**：先查词表与 side features 是否吃了测试集特征（传导式），再看是否用了外部标签。
- **换骨干没用**（large 不比 base 好、DeBERTa 更差、随机初始化与预训练相同）：说明收益来自 **tokenizer + 数据表示**，不是语言预训练（[[kaggle-q-ft-v2]], L76–L78）。此时应该把精力投在编码方案，而不是换更大的模型。
- **单折提升小于 1e-3**：先看折间跨度。两个来源的五折跨度都是 1.6e-3 量级，1e-4 的“改进”不可分辨（[[leaderboard-vs-cv]]）。
- **同样的 side features 给 GBDT 增益更大**：这是正常现象（transformer 的逐值嵌入已经学到一部分），不要据此判定“transformer 不行”（[[kaggle-q-ft-v2]], L100）。

**与 FT-T 的分工**：FT-T 是“小模型 + 统一协议 + 15 seeds”的**基准级**工具；Q-FT 系列是“大骨干 + 单赛题 + 单种子”的**探索级**工具。选型时前者用来定基线，后者用来回答“如果给模型更大的表征容量、更多预训练，还剩多少空间”。

## 第六个候选：行文本化的 GPT 分类器

[[kaggle-tfms-llms-tabular]] 把上一条路线再简化一步：**不训 tokenizer**，把整行写成 `"列 is 值"` 的句子，直接交给 GPT-2。唯一的设计选择是 target 放哪：末尾 = 判别式，句中 = 生成式（见 [[generative-classification]]）。同一赛题的读数：

| 模型 | OOF AUC | 成本 |
| --- | --- | --- |
| target first（GPT-2 small，30 个列序） | **0.96242**（L212） | 复刻 ~10 h + 每折 ~2 h；**每行 60 次前向** |
| target anywhere（24 epochs） | 0.96245（L213） | 同上 |
| 调参 XGBoost（作者自己的） | 0.96149（L219） | — |
| target last（同族） | ~0.961（L217） | 每行 1 次前向 |

**五个准入问题**（前三个必须答“是”）：

1. **数据是否足够大且行内字段不太多？** 约 13 万行原始数据 + 22 个字段是这次成功的条件；每行前向成本随字段数线性增长。
2. **推理预算能接受 $2K\times O$ 次前向吗？** $K$ 类 × $O$ 个列序，本例是 60 次；判别式版本只要 1 次。
3. **能接受「先把原数据集复刻一遭」的前置成本吗？** 60-epoch replica 比 8-epoch 的高 **+1.2e-3 AUC**（L124）——是这个配方里最大的单效应。
4. **数据是真实的，还是合成的？** 若真实，没有任何对照支持生成式（target first）优越；作者的机制解释只适用于「数据由 LM 写成」的情形（L158、L195–L204）。
5. **指标是排序型吗？** AUC 下非归一化的 log-sum-exp 无害；要报概率、定阈值、优化交叉熵时，缺的归一化必须补。

**四条必须先写下来的决定**：

| 决定 | 本来源的做法 | 不写会怎样 |
| --- | --- | --- |
| target 位置 | 末尾（判别）/ 句首（生成）/ 随机（第三种变体） | 变成不可解释的「提示魔法」 |
| 文本格式 | 整数 `36`、浮点 `0.0`、缺失 `None`、标签 `True`/`False`（L145） | 格式差异被当成证据 |
| 分词器 | GPT-2 自己的（L146） | 长数字被逐位切分，信号变噪声 |
| 列序 | 训练每步重排，推理对 30 个顺序 log-sum-exp | 因果模型的顺序偏差无法估计 |

**诊断信号**：

- **target-first 只比 target-last 高不到 2e-3，且数据非合成**：先回到 target-last（便宜 60 倍）。
- **把行文本写成另一格式（分隔符/小数位）后大跌**：序列化敏感度过高，进入「序列化选择的不确定性远大于模型差异」的状态。
- **一折上的提升小于 3e-4**：不可分辨（单折比 OOF 高 1.8e-4 ~ 2.3e-4，见 [[leaderboard-vs-cv]]）。
- **该模型与调参 GBDT 差不到 1e-3**：不要为它重建整个管线，只当作 stack 里的一个成员。

## 可选候选：GFLU 式逐阶段门控（GANDALF）

[[gandalf]] 代表第三种结构来源：不 token 化、不用注意力，用 GRU 式门控逐阶段重写特征表示，并在每阶段用 t-softmax 掩码做全局特征选择（机制见 [[gated-feature-learning-unit]]）。它不是本文档的基线之一，只在下面四个问题都答「是」时值得当候选：

1. **列多、且不少列是噪声？** 作者自述优势「might be more pronounced in datasets with a higher feature count or where the relationship between features and target outcomes isn't straightforward」（[[gandalf]], p. 14），并只在特征数 > 20 的子集上做实验（p. 10）。
2. **能接受阶段间顺序计算？** 「its inherent sequential processing across stages, which precludes parallel processing」（p. 14）。阶段数在调参中可达 30（表 6，p. 19）。
3. **需要的是列级全局重要性，而不是实例级解释？** 重要性是各阶段掩码之和 $I=\sum_n M_n$（式 17，p. 8），不随样本变化；但它只反映门控支路的依赖，候选支路始终看全特征。
4. **能腾出一次 100 trials 的贝叶斯搜索预算？** 论文没有给可开箱的默认配置，只给搜索空间（五个维度）与逐数据集最优值（表 6/7，p. 19–20），且实验中 MLP 头固定为 32/16（p. 11）。

**代价与证据强度**：论文报的是 ADTM，Tabular Benchmark 18 个数据集子集上 0.829 第一，TabSurvey 4 个数据集上 0.945 第三（[[gandalf]], p. 9–10）；两者都是自己不重跑基线、无显著性检验、多数数据集单折的结果。硬件是单卡 RTX 3060（p. 8），所以成本门槛低，但**本库没有独立复现，也未核查 `pytorch_tabular` 的实现与论文是否一致**。因此它的定位是「在小表上值得跑一次对照」，而不是「新的默认基线」。

**一条坑**：不要引用「only three crucial hyperparameters」当作调参成本优势。同一篇的搜索空间含 5 个维度（含 learning rate 与 weight decay），而 FANOVA 显示阶段数影响最大且「doesn't show any particular pattern」，最优值在 2 到 30 之间都有（[[gandalf]], p. 11–13、p. 19）。

## 默认还是调参：四格判断

表 4（[[ft-transformer]], p. 8）给出四种组合：

| 组合 | 观察 | 行动 |
| --- | --- | --- |
| 默认 GBDT vs 默认 FT-T | 默认 FT-T 集成在 9/11 个数据集上**同时**优于两个默认 GBDT；AD 落后，CA 只输给 CatBoost | 先跑默认 FT-T 作为“开箱”基线 |
| 调参 GBDT vs 调参 FT-T | GBDT 在 CA/AD/YA/MI 反超；FT-T 在 HE/JA/HI/EP/YE/CO 更好 | 如果 GBDT 是候选，必须调参后再比 |
| 默认 FT-T vs 调参 FT-T | 差距通常不大（如 CA 0.454 vs 0.448） | 默认配置可用于快速筛选，不必一开始就大搜 |
| 调参 GBDT vs 调参 FT-T | 没有普适赢家 | 以验证集为准，按数据形态决定 |

**调参预算**：表 11（p. 17）显示 FT-T 在 0.25h 已接近 6h 的结果，延长调参对其它算法收益有限；但这只是 CA/AD/HI 三个数据集、5 个种子的结论，不能外推到所有数据。实践中先用小预算筛模型，再对前 2 名加大预算。

## 预处理：DL 与 GBDT 走两条路

| 步骤 | 深度模型 | GBDT |
| --- | --- | --- |
| 数值特征 | 默认 `QuantileTransformer(output_distribution='normal')`，fit 前对 train 加 `N(0,1e-3)` 噪声 | 不预处理（树对平移/缩放不敏感） |
| 标准化例外 | Helena/ALOI 用 `StandardScaler`；Epsilon 用原始特征 | — |
| 回归目标 | 按 train 均值/标准差标准化 | 通常不需要 |
| 类别特征 | 神经网络用同维 embedding；XGBoost one-hot；CatBoost 内置 | 目标编码/内置类别支持 |

代码（`lib/data.py` L21–L44，节选）：

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
normalizer.fit(X_train)   # 只 fit train；再 transform val/test
```

**纪律**：`fit` 只在 train 上；噪声只加到 train（用于估计分位数，不改变原始特征）；val/test 用同一个 normalizer。分位数变换保秩但会改变相关结构，不要把它当“无损清洗”。

## 训练配置模板（FT-Transformer）

来源的默认训练配置（表 12、`output/california_housing/ft_transformer/default/0/stats.json`）：

```json
{
  "model": {
    "n_layers": 3, "d_token": 192, "n_heads": 8,
    "activation": "reglu", "d_ffn_factor": 1.333333333333333,
    "attention_dropout": 0.2, "ffn_dropout": 0.1,
    "residual_dropout": 0.0, "prenormalization": true
  },
  "training": {
    "optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-5,
    "batch_size": 256, "patience": 16, "n_epochs": 1000000000
  }
}
```

要点：

- **不加学习率调度**（来源为了统一协议；p. 6）。如果你加 warmup/decay，需要重新验证，不能直接套用表 2。
- **早停 patience=16**，看验证集；测试集不参与调参（p. 6）。
- **weight decay 不是全场统一**：Feature Tokenizer、LayerNorm、所有 bias 的 wd=0（p. 18）。原始实现用名字过滤（`bin/ft_transformer.py` L375–L381）：

```python
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])
```

包版 `model.make_parameter_groups()` 做同样的事；不要直接 `AdamW(model.parameters(), weight_decay=1e-5)`。

- **batch size** 按数据集预设（256/512/1024；表 7，p. 14）；大特征数据集用更大 batch 与 Linformer 压缩。

## 类别特征：先确认你是否有足够证据

- 本文只有 Adult 一个数据集有类别特征（8 个），因此**不能**从这篇论文推出“FT-T 在高基数类别特征上优于 GBDT”。
- 神经网络侧：所有类别特征用同维 embedding；数值/类别特征分开传入（`FTTransformer(n_cont_features=..., cat_cardinalities=[...])`）。
- MLP/ResNet 侧：类别特征需要先编码成连续特征；one-hot 适合低基数，高基数可考虑 embedding（包 README 的建议）。
- 高基数类别 + 小数据：先跑 CatBoost/目标编码 + GBDT，再考虑 FT-T。

## 评估协议：别用单 seed 的最佳值

来源的协议可以直接复用（[[ft-transformer]], p. 6–7）：

1. **固定一个 train/val/test 划分**，所有模型共用；测试集只在最后用一次。
2. **在验证集上调参**；报告测试集均值。
3. **至少 5 个种子，最好 15 个**（来源主实验 15；消融/分析 5）；报告均值与标准差。
4. **集成**：把种子分成 3 组、每组 5 个，组内平均预测；报告 3 个集成的均值。分类器平均**概率**（先 `expit`/`softmax`），不是平均 logits（`bin/ensemble.py` L34–L65）。
5. **统计**：成对比较用单侧 Wilcoxon，`p=0.01`；但注意来源没有做多重比较校正，“top”是未校正的显著性描述（本库方法学备注）。
6. **同时报告默认与调参**：默认配置代表“开箱成本”，调参后代表“预算充足”；两者混在一起比较会误导。

## 诊断信号

- **FT-T 不比 ResNet 好**：先检查预处理、类别编码、调参预算；不要直接归因于“数据不适合 Transformer”。
- **GBDT 大幅领先**：数据可能是 GBDT-friendly（分段/轴对齐/低阶交互）；FT-T 的增益可能只在部分数据集上出现（4.6）。
- **$k$ 很大且训练慢**：先算 $O(k^2)$ 成本；考虑 Linformer，但包 README 警告小 $k$ 时可能更慢、精度影响研究不足。
- **类别特征多**：先确认本文证据覆盖不足；用 GBDT/目标编码作对照。
- **默认 FT-T 已经很好**：不必大搜；表 11 显示小预算已接近大预算。
- **种子方差大**：增加种子数/集成；不要用单个最佳 seed 下结论。
- **调参后 GBDT 反超**：这是来源预期的结果，不是 bug；按验证集选型。

## 常见坑

1. 直接上 FT-T，跳过 GBDT/MLP/ResNet 基线；
2. 用默认 GBDT 对比调参 FT-T，或反过来；
3. 把测试集用于调参/早停；
4. 对 `model.parameters()` 统一加 weight decay；
5. 不做分位数/标准化，或对全量数据 fit 预处理器；
6. 在回归目标未标准化时直接训练；
7. 类别特征只做 one-hot 就断言“embedding 没用”；
8. 用单 seed 或最佳 seed 报告；
9. 分类集成平均 logits（与概率平均不等价）；
10. 把表 10 的时间当架构缩放律（AD 上 FT-T 反而更快）；
11. 把注意力图当因果解释；
12. 把“FT-T 平均排名第一”误读为“每个数据集都第一”（Epsilon 单模型最差）；
13. 忽略 GBDT 在多分类大类别数上的弱点（Helena/ALOI）；
14. 不记录版本/硬件/种子，导致结果不可复现。

## 可复制骨架

```python
from rtdl_revisiting_models import FTTransformer

model = FTTransformer(
    n_cont_features=x_cont.shape[1],
    cat_cardinalities=cat_cardinalities,
    d_out=1,
    **FTTransformer.get_default_kwargs(),   # 3 层/192 维/8 头
)
optimizer = model.make_default_optimizer()  # AdamW 1e-4, wd 1e-5 + 豁免组
```

完整训练循环需要自己写：MSE/交叉熵、`patience=16`、验证集早停、15 seeds、3×5 集成。不要假设包内自带训练器；`rtdl_revisiting_models` 只提供模型与参数组。

## 可自动化程度

| 步骤 | 可自动化 | 备注 |
| --- | --- | --- |
| 预处理 | 高 | 固定 normalizer + 折内 fit |
| 基线训练 | 高 | GBDT/MLP/ResNet/FT-T 都可脚本化 |
| 调参 | 高 | Optuna TPE；注意搜索空间与预算 |
| 模型选择 | 中 | 需要按验证集与成本权衡；不能只看均值 |
| 类别特征策略 | 低 | 依赖基数/语义；本文证据不足 |
| 失败诊断 | 低 | 需要看学习曲线、方差、数据形态 |

## 关联页面

- 来源：[[ft-transformer]]、[[kaggle-q-ft]]、[[kaggle-q-ft-v2]]、[[kaggle-tfms-llms-tabular]]、[[gandalf]]
- 概念：[[tabular-deep-learning]]（LLM 当骨干、LoRA、order-free、RTD、行文本化）、[[gated-feature-learning-unit]]、[[generative-classification]]、[[multi-layer-perceptron]]、[[ensembles]]、[[feature-preprocessing]]、[[hyperparameter-search]]、[[cross-validation]]、[[evaluation-metrics]]
- 对照：[[gbdt-vs-tabular-dl]]
- 相邻实践：[[competition-ml-workflow]]、[[preprocessing-pipeline-design]]、[[hyperparameter-search-and-tuning]]、[[cross-validation-design-and-tuning]]、[[ensembles-selection-and-tuning]]
