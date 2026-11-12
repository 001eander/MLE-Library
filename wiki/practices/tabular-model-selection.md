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
status: draft
verified: 2026-11-12
applies_to:
  - rtdl_revisiting_models 0.0.2（本地 checkout e3ed46c，2024-11-13）
  - PyTorch >=1.8,<3
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

- 来源：[[ft-transformer]]
- 概念：[[tabular-deep-learning]]、[[multi-layer-perceptron]]、[[ensembles]]、[[feature-preprocessing]]、[[hyperparameter-search]]、[[cross-validation]]、[[evaluation-metrics]]
- 对照：[[gbdt-vs-tabular-dl]]
- 相邻实践：[[competition-ml-workflow]]、[[preprocessing-pipeline-design]]、[[hyperparameter-search-and-tuning]]、[[cross-validation-design-and-tuning]]、[[ensembles-selection-and-tuning]]
