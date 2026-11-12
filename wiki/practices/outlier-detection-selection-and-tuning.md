---
title: 异常检测的选型与调参：先判污染与是否要上线，再选形状假设，最后定阈值
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - anomaly-detection
  - outlier-detection
  - novelty-detection
  - scikit-learn
  - tuning
  - one-class-svm
  - isolation-forest
  - local-outlier-factor
sources:
  - "[[sklearn-outlier-detection]]"
  - "[[sklearn-covariance]]"
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-ensembles]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.6.1（本机实测版本）
  - scikit-learn 1.9.1（文档版本；`contamination` 与 `gamma` 的默认值在 1.6–1.9 之间未变，0.22 起才是 `'auto'`/`'scale'`）
---

# 异常检测的选型与调参

> 本页是**落地流程**，机制与「为什么」在 [[anomaly-detection]]；全部数字要么引 [[sklearn-outlier-detection]] 的行号，要么标「本机实测」（scikit-learn 1.6.1）。

## 一、五个准入问题

顺序不能换——前两问答错，后面调得再细也没用。

1. **训练集被污染了吗？** 被污染 → `outlier detection`（异常不能成簇）；干净 → `novelty detection`（异常可以成簇）。分不清时**假设被污染**，用 outlier 模式，并接受「异常比例」是一个需要承诺的超参。
2. **要给新样本打分吗？** 要 → 必须选有 `predict`/`decision_function` 的估计器（IF/EE/OCSVM/SGD-OCSVM，或 `LocalOutlierFactor(novelty=True)`）。不要 → 可以用 `fit_predict`，但那就**不能上线**。
3. **异常可能成簇吗？** 可能 → **排除 LOF**（实测召回 0.03）；不可能（孤立点）→ LOF 最敏感。
4. **数据是单峰、近似高斯吗？** 是 → `EllipticEnvelope` 最便宜且最可解释（Mahalanobis）；否（多峰/流形）→ 排除 EE 或至少换 `MinCovDet` 的假设再看。
5. **$n$ 与 $p$ 的量级？** $n\gg p$ 且 $n\le10^4$ → 全都可以；$n>10^5$ → IF 或 SGD-OCSVM + 核近似（OCSVM 在 $10^5$ 上实测 29s，比 SGD 版慢约 130×）；$p$ 很大 → IF（`max_features<1.0`）或先降维。

**不满足准入就该停手**：本节的所有估计器都假设「正常样本占多数」。若正常样本本身是少数（例如欺诈检测里 3% 是正常），这些方法的方向就反了。

## 二、按「形状假设」选族（不是按名字）

| 若你相信…… | 选 | 关键旋钮 | 本机实测的依据 |
| --- | --- | --- | --- |
| 正常数据近似高斯、单峰 | `EllipticEnvelope` | `contamination`（默认 **0.1**，无 `'auto'`）、`support_fraction` | 污染训练集上 AUC **0.9872**（660 测试样本，实测 12） |
| 异常「少而不同」、维度高、$n$ 大 | `IsolationForest` | `max_samples`（默认 → 256）、`contamination`、`n_estimators` | AUC 0.9866；$n$ 从 $10^3$ 到 $10^5$ fit 时间 0.045→0.140s（实测 21） |
| 异常是**局部**稀疏点、密度不均 | `LocalOutlierFactor` | `n_neighbors`（默认 20）、`novelty`、`metric` | 局部离群点排名 2/802（EE 排 802/802，实测 16） |
| 需要一条显式边界、可核化、$n$ 中等 | `OneClassSVM` | `nu`（默认 0.5）、`gamma`（默认 `'scale'`）、`kernel` | AUC 0.9610（$\nu$=0.1）→ 0.9809（$\nu$=0.5）（实测 12） |
| 同上的大规模版本 | `SGDOneClassSVM` + `Nystroem` | `nu`、`n_components`、`max_iter` | 与核化版一致率 98.7%（100 成分）/ 100%（500）（实测 13） |

两条**跨估计器的默认值事实**（选型时必须知道）：

- **默认阈值口径不同**：IF/LOF 是 `'auto'`（常数 −0.5 / −1.5），EE 是 0.1，OCSVM 没有 `contamination`。→ 不要比较「谁报了多少个」。
- **从 1.6.1 起**：`IsolationForest(max_features='auto')` 已经非法（只接受 int 或 `(0,1]`）；`EllipticEnvelope(contamination='auto')` 也非法。写跨版本代码要显式给数值。

## 三、定阈值：五步

1. **先只看排序**：用 `score_samples`（或 `-decision_function`）算 AUC / PR-AUC / top-$k$ 命中率。这一步**与阈值无关**，是唯一能公平比较估计器的口径。
2. **再定污染率先验**：写下来源（历史标注比例 / 业务容量 / 人工抽检）。没有先验时，用 `contamination='auto'` 只作为**探索**，因为它的实际报出率是不可控的（高斯数据上 IF 13.9–15.4%、LOF 0.2–1.4%）。
3. **设 `contamination` 并检查数量**：IF/EE 的报出数**精确**等于 $c\cdot n$（1,100 样本上 55/110/220/550，实测 3）；LOF 的 `predict` 会比 `negative_outlier_factor_ < offset_` 少几个（32 vs 33、61 vs 66，实测 9）——**用 `predict` 的数，别用 `nof` 的数**。
4. **不要在训练集上验证**：IF/EE/OCSVM 在训练集上的报出数是被 `contamination` 直接决定的（同义反复）；LOF 的 `novelty=True` 更糟，它的训练集分数与库里的 `negative_outlier_factor_` **不是同一个量**（最大差 0.586）。
5. **上线后重算阈值而不是重训**：`offset_` 是拟合期属性，`contamination` 是它唯一的入口。要按业务容量调整报出量时，优先考虑**用手算分位数覆盖 `offset_`**，并把这个数写进配置（不要留在模型对象里）。

```python
# 阈值与分数分离的写法（sklearn ≥1.6 均可用）
import numpy as np
est = IsolationForest(contamination="auto", random_state=0).fit(X_train)
scores = est.score_samples(X_val)                  # 越大越正常
thr = np.quantile(scores, 0.02)                    # 业务上可接受的 2% 报出率
# 等价于把 offset_ 换成 thr：predict 用的是 scores < offset_
is_outlier = scores < thr
assert np.all(is_outlier == (est.decision_function(X_val) < thr - est.offset_))
```

## 四、三个容量旋钮

| 旋钮 | 对象 | 太小 | 太大 | 本库读数 |
| --- | --- | --- | --- | --- |
| `n_neighbors` | LOF | 只看见极小邻域 → 异常比例高时失效 | 趋近全局密度 → 退化成「距离」 | 17% 污染时 5/20/35/60 → AUC 0.5183/0.9326/0.9766/0.9752 |
| `nu` | OCSVM / SGD-OCSVM | 边界太松（报得少） | 边界太紧（报得多） | 0.01/0.1/0.3/0.5 → 训练集异常率 6.8/11.4/30.2/50.2% |
| `max_samples` | IF | 单棵树看不见结构 | 成本上升且不再「隔离」 | `'auto'` = `min(256, n)`；$n$=300 与 5000 都只取 256 |
| `n_estimators` | IF | 分数方差大 | 收益饱和（路径长度是均值） | 剪藏示例用 10→20 棵演示 `warm_start` |

**调参顺序**：先用 `contamination='auto'` + 默认容量看分数分布 → 用 AUC/top-$k$ 定容量（$k$ 或 $\nu$）→ 最后定 `contamination`。反过来做会得到「阈值在替容量背锅」的假象。

**`n_neighbors` 的硬上界**：必须 $< n$。超过不报错，**静默夹到 $n-1$**（330 点上 400 → 329），结果是**一个异常都报不出来**（实测 18）。同理 $k$ 必须大于「一个簇至少要有的点数」（[[sklearn-outlier-detection]] L166）。

## 五、预处理：先缩放，再谈模型

- **必须缩放**：`OneClassSVM`、`SGDOneClassSVM`、`LocalOutlierFactor`。实测：固定 `gamma=0.5` 时单列 ×100 让 **51.5%** 的标签翻转；`gamma='scale'` 也翻 11%（实测 11）；LOF 的 `metric` 一换，报出数从 24 变 8（实测 23）。
- **不需要缩放的**：`IsolationForest`（轴对齐切分，整体 ×1000 与单列 ×100 的分数 maxdiff 0.0，实测 5）与 `EllipticEnvelope`（自估协方差 → 仿射等变，单列 ×100 后秩相关 1.0、标签一致率 1.0）。但它们仍有别的硬前提：IF 的 `max_samples`、EE 的 $n>p$。
- **度量是模型的一部分**：`LocalOutlierFactor(metric='cosine')` 在同一数据上报 8 个异常，`euclidean`/`manhattan` 都报 24 个（实测 23）。文本/嵌入向量默认该用 `cosine`，但必须把「换了度量」当成**换了模型**，重新定阈值。
- **`Pipeline` 里不要放阈值**：`offset_` 在 `fit` 时定，放进 `Pipeline` 后 `set_params` 改 `contamination` 会重训整个管线（包括缩放的均值方差），这在推理端通常不是你想要的行为。

## 六、评估：三条路，按可靠性排序

1. **有真值**：用 `decision_function` 算 ROC-AUC（阈值无关）。同一污染训练集上的实测排序：EE 0.9872 ≈ IF 0.9866 > LOF(35) 0.9822 > OCSVM(0.5) 0.9809 > SGD-OCSVM 0.9795 > LOF(20) 0.9789 > OCSVM(0.1) 0.9610（实测 12）。**不要用 `predict` 的准确率**——它被 `contamination` 直接决定。
2. **无真值**：稳定性（换种子/重抽子样本，top-$k$ 异常的重合率）+ 人工抽检 top-$k$ 的「能否解释」。注意 IF 与 LOF 的分数都随 `random_state` 变（LOF 在 `novelty=True` 下没有随机性，但邻域随 $k$ 变）。
3. **阈值回测**：用历史标注固定阈值算精度/召回。这一步不可省——`contamination` 入库就是一句「污染率是 $c$」的承诺。

**本库的警告**：官方示例（`plot_anomaly_comparison`、`plot_outlier_detection_bench`）都用**已知真值**画 ROC，这在真实场景里通常没有。把它当「上限」而不是「预期」。

## 七、成本表（$d=5$，单机 CPU，1.6.1 实测）

| $n$ | `IsolationForest` | `LOF(20)` | `OneClassSVM` | `EllipticEnvelope` | `SGDOneClassSVM`+`Nystroem(100)` |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 0.045s | 0.005s | 0.001s | 0.141s | 0.015s |
| 10,000 | 0.072s | 0.154s | 0.092s | 0.514s | 0.015s |
| 100,000 | 0.140s | 1.254s | **29.078s** | 5.211s | 0.222s |

要点：**IF 的成本几乎与 $n$ 无关**（`max_samples=256` 的功劳）；**OCSVM 在 $10^5$ 上开始不可用**；EE 的 MCD 是 $O(n\cdot\text{迭代})$ 且常数不小。官方图 1 的面板标题（`.00s`–`.16s`，$n$=100 量级）只能当量级参考（[[sklearn-outlier-detection]]「图表核查」）。

## 八、18 条坑（每条都对应一次真实的沉默失败）

1. **`contamination` 不是统一旋钮**：IF/LOF 的 `'auto'` 是常数（−0.5/−1.5），EE 的默认是 0.1，OCSVM 没有这个参数。
2. **`contamination=0.0` 不合法**（区间 `(0.0, 0.5]`），`'auto'` 也不是 0。
3. **`EllipticEnvelope(contamination='auto')` 抛 `InvalidParameterError`**（它只接受 float）。
4. **`nu=0` 与 `nu>1` 都非法**；`nu` 的实际语义是训练集异常/支持向量比例（$\nu$=0.01 → 6.8%，不是 ≤1%）。
5. **LOF 默认没有 `predict`/`decision_function`/`score_samples`**（`AttributeError`，不是 `NotFittedError`）。
6. **`novelty=True` 时 `fit_predict` 抛 `AttributeError`**。
7. **在训练集上调用 `novelty=True` 的 `predict` 会得到与 `fit_predict` 不同的标签**（1.2%），且分数与 `negative_outlier_factor_` 最大差 0.586。
8. **LOF 的 `predict` 数量 ≠ `nof < offset_` 的数量**（32 vs 33、61 vs 66）：前者重算分数。
9. **`n_neighbors ≥ n` 静默夹到 $n-1$**，结果退化成「全正常」。
10. **`n_neighbors` 太小 + 污染率高 = 随机**（$k$=5、17% 污染 → AUC 0.5183）。
11. **LOF 看不见密集异常簇**（召回 0.03）。
12. **全局型估计器看不见局部异常**（EE 把 8σ 的点排到 802/802）。
13. **重复点不会被判异常**：完全相同的 20 个点 `negative_outlier_factor_` 恰好 −1.0，`novelty=True` 的 `decision_function` 是 +0.5。
14. **`IsolationForest(warm_start=True)` 改 `max_samples` 后重训无警告**：旧树仍按旧值训练。
15. **`n_estimators` 不变时的重复 `fit` 不训练新树**，只给一条 `UserWarning`。
16. **`EllipticEnvelope` 在 $n\le p$ 时报内部下标错误**（`ValueError: kth(=7) out of bounds (4)`），不是友好提示；继承 [[covariance-estimation]] 的全部 MCD 警告（`support_` 不是 h）。
17. **`IsolationForest(max_features='auto')` 在 1.6.1 已非法**（1.9.1 同）；写老代码的迁移点。
18. **`predict` 的 `±1` 不是 0/1**：接 `sklearn.metrics` 前必须重编码，否则 `accuracy_score` 直接算错。

## 九、可自动化程度

| 步骤 | 可自动化 | 说明 |
| --- | --- | --- |
| 数据画像（是否多峰、$n/p$、尺度差） | 高 | 直方图/相关系数/KMeans 粗分即可判「是否该排除 EE」 |
| 选形状假设（本页第二节的表） | 中 | 可从画像推出 2–3 个候选，最终仍要人工定「异常能否成簇」 |
| 容量扫描（$k$ / $\nu$ / `max_samples`） | 高 | 有真值时网格 + AUC；无真值时看 top-$k$ 稳定性 |
| `contamination` 定值 | **低** | 需要业务先验（可接受报出量），不能从数据推 |
| 缩放与度量选择 | 中 | 可从特征尺度与相似度语义推断，但度量切换等于换模型 |
| 阈值回测 | 低 | 依赖历史标注 |
| 上线后的漂移监控 | 中 | 报出率突增是最便宜的信号 |

## 关联页面

- 机制：[[anomaly-detection]]
- 来源：[[sklearn-outlier-detection]]
- 相邻实践：[[covariance-estimation-selection-and-tuning]]（`EllipticEnvelope` 的 MCD 基础）、[[nearest-neighbors-selection-and-tuning]]（`n_neighbors`/度量）、[[ensembles-selection-and-tuning]]（IF 与 RF 的差别）、[[svm-selection-and-tuning]]（$\nu$ 与 `gamma`）
- 工具：[[scikit-learn]]
