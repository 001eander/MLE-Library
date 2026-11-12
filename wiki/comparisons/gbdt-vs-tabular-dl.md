---
title: GBDT 与表格深度模型：没有普适赢家，只有数据形态与预算的匹配
type: comparison
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - gbdt
  - deep-learning
  - model-selection
  - benchmark
sources:
  - "[[ft-transformer]]"
  - "[[kaggle-q-ft]]"
  - "[[kaggle-q-ft-v2]]"
  - "[[kaggle-tfms-llms-tabular]]"
  - "[[gandalf]]"
status: draft
verified: 2026-11-12
applies_to:
  - 论文 arXiv:2106.11959v5（2023-10-26）
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13）
---

# GBDT 与表格深度模型：没有普适赢家

[[ft-transformer]] 的中心结论不是“FT-Transformer 打败了 GBDT”，而是：**默认配置下深度模型更容易开箱即用；充分调参后 GBDT 会在部分数据集反超；两者没有普适赢家**（[[ft-transformer]], p. 8）。本页把这条结论拆成可读的表：先给读表规则，再给默认/调参两种口径的胜负，最后给选型判据。

## 先定指标方向，再读胜负

表格指标方向不一致，极易读反：

| 指标 | 方向 | 数据集 |
| --- | --- | --- |
| RMSE | 越低越好 | CA、YE、YA、MI |
| Accuracy | 越高越好 | AD、HE、JA、HI、AL、EP、CO |

例如 HE 是 accuracy：NODE 0.359 < ResNet 0.396，含义是 **ResNet 更好**，不是 NODE 更好。后文所有“更好”都按这张表的方向解释。

## 单模型：FT-T 在 5/11 列均值最佳

表 2（[[ft-transformer]], p. 7）逐列均值的最佳模型：

| 数据集 | 指标 | 最佳单模型 | 均值 |
| --- | --- | --- | --- |
| CA | RMSE | FT-Transformer | 0.459 |
| AD | Acc | FT-Transformer / AutoInt（并列） | 0.859 |
| HE | Acc | ResNet | 0.396 |
| JA | Acc | FT-Transformer | 0.732 |
| HI | Acc | FT-Transformer | 0.729 |
| AL | Acc | ResNet | 0.963 |
| EP | Acc | TabNet | 0.8896 |
| YE | RMSE | NODE | 8.784 |
| CO | Acc | FT-Transformer | 0.970 |
| YA | RMSE | NODE | 0.753 |
| MI | RMSE | NODE | 0.745 |

**FT-T 最佳 5 列（CA/AD/JA/HI/CO），ResNet 2 列（HE/AL），NODE 3 列（YE/YA/MI），TabNet 1 列（EP）**。FT-T 的平均排名仍是 1.8（表 2），因为它在多数数据集上处于前列；但它在 Epsilon 单模型上是全表最差（0.8982，低于 TabNet 0.8896、AutoInt 0.8949、NODE 0.8958、ResNet 0.8969）。这就是“平均排名第一”与“每列都第一”的差别。

## 默认 vs 调参：结论会翻转

表 4（[[ft-transformer]], p. 8）的集成结果，按“FT-T 是否同时优于两个 GBDT”统计：

| 口径 | FT-T 同时优于 XGBoost/CatBoost | GBDT 同时优于 FT-T | 混合/不可比 |
| --- | --- | --- | --- |
| **默认超参** | **9/11**：HE、JA、HI、AL、EP、YE、CO、YA、MI | 1/11：AD | CA（CatBoost 更好，FT-T 好于 XGBoost） |
| **调参后** | **6/10 可比**：HE、JA、HI、EP、YE、CO | 4/10：CA、AD、YA、MI | AL：GBDT 为空（不可调/极慢） |

读法：

- **默认配置**：FT-T 集成在 9 个数据集上同时赢过两个默认 GBDT；只有 Adult 两个 GBDT 都更好，CA 上 CatBoost 更好。来源因此说“FT-Transformer allows building powerful ensembles out of the box”（p. 8）。
- **调参后**：GBDT 在 CA、AD、Yahoo 明显反超；MI 也略好（GBDT 0.741–0.742 vs FT-T 0.743）。FT-T 在 HE/JA/HI/EP/YE/CO 仍更好；ALOI 的 GBDT 不可调，FT-T 0.967 与 ResNet 0.966 仍可用。
- **“DL 在多数任务上更好”只对调参后的 6/10 成立**，且来源自己提醒：这说明基准略偏 “DL-friendly”，不说明 DL 在任何意义上更“好”（p. 8）。

## 成本：GBDT 更轻，但倍数不是常数

表 10（[[ft-transformer]], p. 16）的 ResNet/FT-T 训练时间（秒，15 run 均值）：

| 数据集 | 特征数 | ResNet | FT-T | FT-T/ResNet |
| --- | --- | --- | --- | --- |
| CA | 8 | 72 | 187 | 2.6× |
| AD | 14 | 144 | 128 | **0.9×** |
| HE | 27 | 363 | 536 | 1.5× |
| JA | 54 | 163 | 576 | 3.5× |
| HI | 28 | 91 | 257 | 2.8× |
| AL | 128 | 933 | 2864 | 3.1× |
| EP | 2000 | 704 | 934 | 1.3× |
| YE | 90 | 777 | 1776 | 2.3× |
| CO | 54 | 4026 | 5050 | 1.3× |
| YA | 699 | 923 | 12712 | **13.8×** |
| MI | 136 | 1243 | 2857 | 2.3× |

要点：

- GBDT 通常比两者都轻；FT-T 的训练成本**不是**特征数的单调函数（AD 0.9×，EP 只有 1.3×，Yahoo 13.8×）。时间取决于调参后的层数/宽度/早停与实现，不能把表 10 当复杂度公式。
- 特征数很大时，$O(k^2)$ 的 MHSA 是主要风险；Epsilon 用 Linformer 式 KV 压缩（`kv_compression=0.064`、`headwise`，$1+2000$ token → 约 128 维），但包 README 警告：特征数不够大时 Linformer 可能更慢，且精度影响研究不足。
- 来源没有测能耗；“CO2 风险”是定性论述（p. 5）。

## 机制：为什么 GBDT 在某些数据上赢

来源用合成目标插值把“架构差异”与“数据集差异”分开（[[ft-transformer]], p. 9）：

$$
y=\alpha\cdot f_{GBDT}(x)+(1-\alpha)\cdot f_{DL}(x),
$$

$f_{GBDT}$ 是 30 棵随机树的平均，$f_{DL}$ 是 3 层随机 MLP；α=0 是 DL-friendly，α=1 是 GBDT-friendly。图 3 的读数：

| α | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| --- | --- | --- | --- | --- | --- |
| ResNet | 0.36 | 0.415 | 0.56 | 0.635 | 0.645 |
| FT-Transformer | 0.38 | 0.40 | 0.39 | 0.29 | 0.23 |
| CatBoost | 0.47 | 0.46 | 0.39 | 0.28 | 0.19 |

ResNet 随目标变“树友好”而显著退化；CatBoost 相反；FT-T 在整个区间保持竞争力。来源推测 FT-T 更适合逼近一部分树状函数（p. 9）。这是**机制假设**：合成数据上的现象不能直接外推到真实表；来源也没有做表征层面的验证。

一个可操作的推论：**如果 GBDT 在你的数据上大幅领先，先把它当“目标函数更像树”的信号**，不要只用“DL 调参不够”解释；反之，如果 FT-T 大幅领先，检查 GBDT 是否受类别数/类别特征/调参预算限制。

## 另一套基准上的读数：GANDALF 的第一名发生在它自己选的子集上

上两节全部来自 [[ft-transformer]] 的 11 数据集同协议。[[gandalf]] 给的是同一时期另一份读数，但**协议不同**：作者只跑自己的模型，其余模型沿用 Tabular Benchmark 与 TabSurvey 已发布的分数，不重跑（[[gandalf]], p. 9）。

| 基准 | GANDALF | 最强的 GBDT | 最强的 DL 对手 | 逐数据集最优 |
| --- | --- | --- | --- | --- |
| Tabular Benchmark（18/45 数据集，ADTM↑） | **0.829** | XGBoost 0.577 | SAINT 0.384、FT-T 0.370 | 13/18 |
| TabSurvey（4 数据集，ADTM↑） | 0.945 | XGBoost 0.957、CatBoost 0.947 | SAINT 0.915 | 2/4 |

（图 3 p. 9、图 4 p. 10、表 3 p. 17、表 5 p. 18。TabSurvey 的 ADTM 对回归做了方向改写，MSE 越低越好。）

三个读法：

1. **「第一」只对 Tabular Benchmark 的 18 个数据集子集成立**。该子集按「仅中等规模、无 target/data 变换、特征数 > 20」筛出（[[gandalf]], p. 10），而 TabSurvey 上同一模型落后 XGBoost 约 0.012、落后 CatBoost 0.002。
2. **不能把 0.829 与 FT-T 的「平均排名 1.8」直接对比**：前者是 min-max 缩放后的 ADTM 均值，后者是跨 11 个数据集的平均名次，且数据集集合、指标与调参预算都不同。两篇只在 covertype、Higgs、heloc 等数据集名称上重叠。
3. **TabSurvey 上的两胜两负值得单独看**：GANDALF 在 Covertype（97.7）与 Cal. Housing 的 MSE（0.16）上最优，在 Adult（86.7 vs LightGBM 87.4）与 HIGGS（76.9 vs SAINT 79.8）上不是。

**证据边界**：无显著性检验；表 1 中多数数据集只有 1 折，只有 9 个数据集有标准差（[[gandalf]], p. 16–17）；表 3 与表 4 的量级不一致，回归指标的尺度无法从原文确定（见 [[gandalf]] 分歧 3）；图 4 只画 15 个模型而表 5 有 21 个，筛选规则未说明。因此这张表只能读成「在别人的基准上，GANDALF 在最强的 GBDT 之列，但没有稳定超过它们」，不能读成「DL 赢了 GBDT」。

## 单一竞赛证据：人工特征给 GBDT 的收益约为 DL 的 4 倍

上面是 11 个数据集的同协议基准；下面这条只有 **n=1**，但方向明确，且恰好能检验“机制”一节的说法。

[[kaggle-q-ft-v2]] 在同一赛题上给 transformer 与 GBDT 喂了**同类人工特征**（counts、路由画像、无标签典型性残差、外部数据集的平滑 log-odds，加上两个行级 summary token），读数：

| 模型 | 加入这类特征后的增益 |
| --- | --- |
| token 化的 transformer（ModernBERT） | 约 **+4e-4** AUC（fold 0，两个种子一致） |
| GBDT | 约 **+1.6e-3** AUC |

（[[kaggle-q-ft-v2]], L100。）作者的解读是：**transformer 的逐值嵌入已经学到了 counts 与路由画像能说的东西**，所以同一批信息的边际价值被压到 1/4。

**为什么写进本页**：它给出了“GBDT 赢”的一个与目标函数形态无关的解释——**人工特征与模型内建表征是部分可替代的**。这与 FT-T 论文的 α 实验互补：α 实验说明“目标函数形态”会拉开差距（[[ft-transformer]], p. 9，见上一节），这条说明“**特征供给**”也会拉开差距，而且方向相反（DL 把特征吸收进嵌入，GBDT 把特征当输入）。

**证据强度声明**：(a) 单一赛题、单次 5 折、单种子；(b) 剪藏未给 GBDT 的配置、也未确认两侧用的是**完全相同**的 9 个特征（只说 “the same kind of features”）；(c) transformer 侧的 +4e-4 与消融表的 +2.87e-4 不一致（见来源页分歧 3）。因此这条只能当**方向性机制证据**，不能当量化的选型依据。

**仍可行动的推论**：如果你已经在给模型喂大量统计类特征，先问一句“模型自己能不能从逐列嵌入里学出来”；若模型是 GBDT/线性/扁平 MLP，这类特征仍然值得做。

## 同一赛题上的五类模型阶梯（n=1，但方向清楚）

[[kaggle-tfms-llms-tabular]] 把同一场竞赛的各类模型放在了一张可比较的尺子上（均 OOF AUC，除注明外）：

| 类别 | 代表 | 分数 |
| --- | --- | --- |
| LLM + 文本，target first（生成式） | GPT-2 small + 30 个列序 | **0.96242**（L212） |
| LLM + 文本，target last（判别式） | 同族模型 | ~0.961（L217） |
| **调参 GBDT** | 作者自己的 XGBoost | **0.96149**（L219） |
| 表格基础模型（TFM） | TabPFN-3.5 / 9 个 TFM 的 stack | 0.96115 / 0.96120（L176、L187） |
| LLM + 表格嵌入（token 化） | Q-FT v2 / Q-FT | 0.960798 / 0.959795（前两篇） |
| 调参 DL | RealMLP | 0.95931（L179，作为 baseline） |

三个读法：

1. **在这道题上，赢家是「建模方向」而不是「模型大小」**：同一个 124M GPT-2，只把 target 从前排到后，就从 0.961 变成 0.96242；而 0.6B 的 Qwen（Q-FT）是 0.959795。
2. **TFM 并没有超过调参 GBDT**（0.96115/0.96120 < 0.96149），与该来源 §8 的「四类都能超过调参 GBDT」（L249）相反。
3. **stack 的边际收益与族内相关性成反比**：9 个 TFM 堆起来只值 +5e-5（[[ensembles]]），而一个文本化 GPT 模型加进 245 对公共 pair 值 +4.9e-4（L238）。

**证据强度与一处必须保留的争议**：全部数字来自同一赛题、同一作者、单次 5 折；且**基线的同一性有争议**——该来源的 XGBoost baseline 0.95985 / RealMLP 0.95931 与前两篇所称的「调参」XGBoost 0.959852 / RealMLP 0.959311 五位有效数字相同（见 [[kaggle-tfms-llms-tabular]] 的「跨来源冲突」）。若那两个其实是未充分调参的基线，则 Q-FT 两节“与调参 GBDT 打平”的定位还要往下调。

## 基准偏差：这篇论文的结论有边界

[[ft-transformer]] 的 11 个数据集有明确的形态偏差（[[ft-transformer]], p. 6、p. 14）：

- **10/11 个数据集只有数值特征**；只有 Adult 有 8 个类别特征；
- 没有缺失值实验；
- 每个数据集只有一个固定划分；
- 不使用预训练、额外损失、数据增强、蒸馏、学习率 warmup/decay（p. 5）；
- 调参预算不完全可比（Yahoo 未调参、Epsilon 启发式默认、NODE 在 HE/AL 未调参、GrowNet 不支持多类、CatBoost 在 ALOI 不可调）；
- Wilcoxon `p=0.01` 未做多重比较校正。

因此“FT-T 在 6/10 个调参后数据集上更好”应读作：**在这套以数值特征为主、无缺失、单一划分、统一（但非完全等同）调参协议的基准上**。它不能回答高基数类别特征、缺失值、分布偏移或在线延迟约束下的选型。

## 选型判据

| 场景 | 首选 | 次选 | 理由 |
| --- | --- | --- | --- |
| 快速拿到强基线 | 调参 GBDT | 默认 FT-T | GBDT 轻、稳；默认 FT-T 开箱强 |
| 类别特征多/基数高 | CatBoost/GBDT + 目标编码 | FT-T + embedding | 本文类别证据只有 Adult（8 个类别特征） |
| 特征数很大（数百到数千） | GBDT / ResNet / MLP | FT-T + Linformer | $O(k^2)$ 注意力；Yahoo 13.8× |
| 样本少、参数预算紧 | GBDT / 线性 / 小 MLP | 小 FT-T | 默认 FT-T 929K 参数（100 数值特征） |
| 需要端到端/可微/多任务 | FT-T / ResNet | MLP | 可共享 embedding、接多任务头 |
| 追求最低 RMSE/最高 Acc | 两者都调参 | — | 没有普适赢家；按验证集定 |
| 多分类、类别数很多 | FT-T / ResNet | GBDT（可能慢到不可调） | Helena 100 类、ALOI 1000 类；GBDT 在 ALOI 为空 |
| 推理预算极紧（单次前向） | GBDT / ResNet | 文本化 target-last | 生成式要 $K\times O$ 次前向；本例 60:1 |
| 数据是某个生成器的输出 | 生成式似然模型（target first） | 调参 GBDT | 只在「能认定生成器、且其序列化可复原」时成立；赛题特异 |

**默认配置的定位**：默认 FT-T 是一个非常好的“开箱基线”，在 9/11 个数据集上同时赢过默认 GBDT（p. 8）。但它不是“调参后仍最强”的保证；如果预算允许，GBDT 与 FT-T 都要调参后再比。

## 待解问题

1. 在**类别特征为主、高基数**的数据集上，CatBoost/目标编码与 FT-T embedding 的胜负如何？
2. 有缺失值时，GBDT 的原生缺失处理 vs FT-T 的插补/缺失 token，哪个更稳？
3. 多种 train/val/test 划分下，表 4 的“调参后 6/10”是否稳定？
4. 表 4 的粗体/颜色在原始版本中如何定义？v5 只有粗体，且与列内最优不完全一致（见来源页“分歧”）。
5. GBDT 的“不可调”（ALOI）是实现问题还是本质问题？换 LightGBM/GPU 后是否仍不可调？
6. FT-T 的增益是否主要来自“更像树”的目标，还是来自 embedding/注意力对类别特征的处理？需要真实数据的消融。
7. 在在线延迟约束下，蒸馏后的 FT-T 与 GBDT 的精度-延迟曲线如何？
8. 如果把调参预算统一为“相同 wall-clock 时间”，结论是否改变？表 11 只覆盖 3 个数据集。
9. “同类人工特征给 GBDT 的收益是 DL 的 4 倍”在多大数据集上成立？需要同一批特征、同一验证划分的受控对照。
10. 同一赛题上「文本化生成式 > 调参 GBDT > TFM > token 化 LLM」这个顺序在第二个数据集上还成立吗？
11. 四类模型（GBDT / TFM / token 化 LLM / 文本化 LLM）的相对位置，在**非合成数据**上是什么样？目前全部证据来自一个 Playground 赛题。

## 关联页面

- 来源：[[ft-transformer]]、[[kaggle-q-ft]]、[[kaggle-q-ft-v2]]、[[kaggle-tfms-llms-tabular]]、[[gandalf]]
- 概念：[[tabular-deep-learning]]、[[gated-feature-learning-unit]]、[[generative-classification]]、[[ensembles]]、[[decision-trees]]、[[multi-layer-perceptron]]、[[evaluation-metrics]]
- 实践：[[tabular-model-selection]]、[[competition-ml-workflow]]、[[hyperparameter-search-and-tuning]]、[[cross-validation-design-and-tuning]]
