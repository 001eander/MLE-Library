---
title: "协方差估计的选型与调参：先判 n/p、再定 α 的来源、最后选损失口径"
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - covariance
  - shrinkage
  - precision-matrix
  - robust-statistics
  - workflow
sources:
  - "[[sklearn-covariance]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.6.1–1.9.1（API 签名两版一致；行为在 1.6.1 复核）
---

# 协方差估计的选型与调参

本页是把 [[covariance-estimation]] 的机制落到工程决策上。所有数字都来自 [[sklearn-covariance]] 的「本机实测」节（scikit-learn 1.6.1），标注为「实测」；标注为「外部核查」的来自 1.9.1 文档/源码。**没有任何一条来自官方建议**——2.6 节本身不给调参流程。

## 一、先判该不该用：五个问题

1. **你要的是 $\Sigma$ 还是 $K$？**
   - 只要 $\Sigma$（白化、采样、PCA 的特征值、GMM 的协方差）：收缩类足够。
   - 要 $K$（Mahalanobis、LDA/QDA 的决策面、偏相关、条件独立图）：**必须**考虑稀疏或稳健，因为求逆会放大误差。
2. **下游是「用它算似然」还是「用它解释结构」？**
   - 似然/预测：CV 或 LW/OAS 的 α 都可以，目标是 held-out NLL。
   - 结构/边：**不能用 CV 选的 α**（实测多 2.8–6.8 倍边），要按 F1/边数手调 `alpha`（§3.3）。
3. **$n/p$ 是多少？** 这是本节唯一有量级判据的变量（§2）。
4. **有离群点吗？** 只要「测量/记录错误、罕见但规律出现的观测」存在（L118 的原话），经验与收缩估计器都会被带偏；此时先 MCD 或先做离群点诊断（§4）。
5. **你要不要把 $\Sigma$ 的估计交给 LDA/QDA？** 如果模型本身就是判别分析，优先用 `covariance_estimator=` 接本节的对象，而不是自己估完再传（§5）。

## 二、按 $n/p$ 选：一张实测表

同一份合成数据（$p=20$、真值 12 条非对角边）、同一组种子，只看**协方差的平均绝对误差**（越小越好）：

| $n/p$ | 经验 | LW | OAS | 手调 GLasso | GLassoCV | 建议 |
| --- | --- | --- | --- | --- | --- | --- |
| $\gg1$（100） | 0.0205 | 0.0196 | 0.0196 | **0.0096**（α=0.02） | 0.0055 | 差距很小；要边就用 GLasso，否则经验/收缩都行 |
| ~10 | 0.0625 | 0.0470 | 0.0469 | 0.0186（α=0.4） | **0.0134** | 收缩已明显有用；要结构再加 GLasso |
| ~3 | 0.1096 | 0.0559 | 0.0559 | **0.0217**（α=0.4） | 0.0214 | 必须离开经验协方差 |
| ~1 | 0.1975 | 0.0459 | **0.0357** | 0.0384（α=0.4） | 0.0437 | 经验协方差已不可用；收缩/稀疏二选一 |
| <1 | — | — | — | 见下 | — | 先保证 $n>p$，否则连 MCD 都不能用 |

配套的 **held-out 负对数似然**（另一个数据集，$p=5/10/20$，8 种子均值）：

| 设定 | Empirical | LW | OAS | 手设 α=0.1 | oracle |
| --- | --- | --- | --- | --- | --- |
| $p=5,n=40$ | 7.344 | 7.295 | 7.304 | **7.267** | 6.999 |
| $p=10,n=30$ | 16.018 | 14.740 | 14.749 | 14.999 | 13.900 |
| $p=20,n=20$ | `inf` | **30.608** | **30.608** | 39.776 | 27.801 |

**读法**：

- **$n/p\ge10$**：任何合理估计器都差不多（差 0.02–0.03），不必纠结；把精力放在下游。
- **$n/p\approx3$**：收缩已值 1.3 个 nats，**必须**离开经验协方差。
- **$n/p\le1$**：经验协方差**不可用**（NLL = `inf`）；此时 `ShrunkCovariance(0.1)` 也不够（39.8），要么用 LW/OAS，要么把 α 调到 0.3 以上。
- **数据驱动的 α 不等于更好的 NLL**：第一行 α=0.1 赢了 LW/OAS。**不要因为「用的是 Ledoit-Wolf」就跳过 α 的验证。**

## 三、收缩类：四条路 + 三个检查

### 3.1 四条路

```python
from sklearn.covariance import (EmpiricalCovariance, ShrunkCovariance,
                                LedoitWolf, OAS)
# 1) 不收缩（只在 n >> p 时安全）
S0 = EmpiricalCovariance().fit(X).covariance_
# 2) 固定 α（默认 0.1 是任意值，不是最优）
S1 = ShrunkCovariance(shrinkage=0.1).fit(X).covariance_
# 3) 数据定 α（最小化 Σ 的 MSE）
lw = LedoitWolf().fit(X);  S2 = lw.covariance_;  alpha_lw = lw.shrinkage_
# 4) 高斯假设下的 α（MSE 更小）
S3 = OAS().fit(X).covariance_
```

**实测确认 2/3/4 是同一个算子**：`ShrunkCovariance(shrinkage=lw.shrinkage_)` 与 `lw.covariance_` 逐位相同。所以第 2 条不是「另一个模型」，而是「手设 α 的那个模型」。

### 3.2 必查三项

1. **`assume_centered` 是否符合数据状态**。`True` 表示「不减均值」，不是「已近似零均值」。在未居中的数据上误设，实测对角 1.348→2.550，`location_` 静默为零。**检查**：`np.abs(X.mean(0)).max()`；若 >1e-8，就别设 `True`。
2. **`shrinkage_` 的值**。它可能落在意料之外的位置：$n=2$ 时恒为 0（收缩关闭），$p=1$ 时为 0，各向同性大样本时接近 1。**检查**：打印 `lw.shrinkage_`；若为 0 且 $n$ 很小，说明退化而不是「数据很好」。
3. **尺度是否可比**。收缩目标 $\frac{\mathrm{Tr}}{p}I$ 随尺度变，实测把一列×100 后 LW 的 α̂ 从 0.0814 掉到 0.0303。**检查**：若各列量纲不同，先 `StandardScaler` 再估（等价于在相关阵上收缩），或者改用 LDA 式的 `shrinkage='auto'` 路径。

### 3.3 α 怎么定（如果不用 LW/OAS）

- **网格**：`shrinkage` ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0}，用**下游指标**（held-out NLL、CV 准确率、Mahalanobis 分离度）选，不要用 $\Sigma$ 的 MSE 选——两者不等价（§2）。
- **端点要跑**：`shrinkage=1` 是 $\frac{\mathrm{Tr}}{p}I$（对角全等），在很多数据上会明显过罚；不跑端点就不知道区间在哪。
- **报告 α**：跨数据集比较「收缩好不好」时，必须同时报 α̂，否则无法判断是方法好还是 α 恰好合适。

## 四、稀疏精度矩阵（GraphicalLasso）：五步落地

### 4.1 标准做法

```python
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.pipeline import make_pipeline

# 关键：先标准化，再估 K（L94 + 实测 17）
Xs = StandardScaler().fit_transform(X)

gl = GraphicalLasso(alpha=0.05, assume_centered=False).fit(Xs)
K  = gl.precision_          # 稀疏精度矩阵
S  = gl.covariance_         # 对应的协方差
```

**为什么必须标准化**：实测把一列乘 1000 后，原始尺度 F1 从 0.308 掉到 **0.207**（104 条边 vs 66 条）；标准化后回到 0.308。$\alpha$ 是绝对阈值，量纲不同 → 每个变量受的惩罚不同。

### 4.2 α 的选择：先看路径，再看目标

- **手调优先**：在 0.01 → 0.8 的对数网格上数非对角非零数，找到「边数落在业务可解释范围」的 α。实测（种子 1、$p=20$、真值 12 条边）：α 0.2 → 72 条，α 0.4 → **12 条（F1 0.833）**，α 0.8 → 0 条。
- **`GraphicalLassoCV` 只能当「下界」**：它按 held-out 似然选 α，系统性偏小（实测 34–82 条边 vs 真值 12），但协方差误差最小。**要边就别用 CV 的 α；要 $\Sigma$ 就用**。
- **`alphas=4` 不是候选列表**：它是初始路径上的点数，网格由 $\alpha_{\max}=$ 经验协方差非对角元素最大绝对值生成，再 refine 4 轮（外部核查源码）。想控制网格就显式传 `alphas=[...]`。

### 4.3 三个必查

1. **`ConvergenceWarning`**。实测近共线列 + α=0.001 → **125 条**「Objective did not converge」；α=0.01 → 3 条；α=0.1 → 0 条。**警告是「α 太小」的诊断信号**，不是可忽略的噪声。
2. **边数是否等于真值量级**（如果有真值）。没有真值时用**稳定性**检查：不同子样本的 `precision_` 支持集是否重合（本库未提供现成工具，需自己 bootstrap）。
3. **`alpha=0` 是否被误用**：它合法，且等价于 `EmpiricalCovariance`（实测 maxdiff 0.0）——不是「无惩罚但更稳」，就是 MLE。

### 4.4 什么时候根本不该用 GLasso

- $n$ 远小于 $p$ 且没有强结构假设：实测 $n=10,p=20$ 时 α=0.05 不收敛、仍给 238 条非零边；此时应换更结构化的模型（FA、低秩 + 对角），而不是硬调 α。
- 变量高度共线且 $\alpha$ 必须很小才能保留边：$(p,n)$ 与 $\alpha$ 的选择互锁，先做共线性诊断。

## 五、接进 LDA / QDA / GaussianNB

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import OAS, MinCovDet

LDA(solver='lsqr', covariance_estimator=OAS())          # 换估计器
LDA(solver='lsqr', shrinkage='auto')                    # 标准化 + LW + 还原
LDA(solver='lsqr', shrinkage=0.2)                       # 手设 α
LDA(solver='lsqr', covariance_estimator=MinCovDet())    # 稳健版
```

**实测的三条接线事实**：

1. **`shrinkage='auto'` 与手设 float 不是同一个东西**：前者内部先 `StandardScaler` → `ledoit_wolf` → 按 `scale_` 还原，因此**对特征缩放等变**（decision function 差 8.9e-15）；后者用 `shrunk_covariance`，**不等变**（同一数据缩放后 decision function 差 7.22）。
2. **`shrinkage=1` 的终点是标量乘单位阵，不是方差对角阵**：实测 `covariance_ = 133.3276\,I`，而经验对角是 `[1.06, 24.25, 383.12]`。1.2 节文档说「对角阵」，与之冲突（见 [[covariance-estimation]] §2.3）。
3. **`covariance_estimator` 与 `shrinkage` 互斥**：两者都设会抛 `ValueError`；`solver='svd'` 下两者都不支持（`NotImplementedError`）。

**`GaussianNB` 不走这条路**：它的 `var_smoothing` 是方差下限，不是收缩（[[naive-bayes]]）。对角假设是**结构选择**，不是本节四个估计器之一。

## 六、稳健估计（MinCovDet）：四步 + 三个陷阱

### 6.1 四步

```python
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet, EmpiricalCovariance

mcd = MinCovDet(random_state=0).fit(X)      # 先固定种子
print(mcd.raw_support_.sum())               # h = ceil((n+p+1)/2)，不是 support_
d2  = mcd.mahalanobis(X)                    # 用稳健 K 算距离
bad = d2 > chi2.ppf(0.975, df=X.shape[1])   # 与内部重加权阈值一致
Xc  = X[~bad]                               # 丢掉后再跑下游模型
```

### 6.2 三个陷阱

1. **`support_` ≠ h**：它是重加权后的掩码（实测 178/200），h 在 `raw_support_`（102）。想检查「算法选了多少点」必须读后者。
2. **校正改变量级**：`raw_covariance_` 的 $\det$ 是 0.0762，`covariance_` 是 0.7365（≈9.7×）。**报告时必须说明用哪一组**；跨估计器比较只用 `covariance_`。
3. **`n > p` 是硬前提**，但错误信息不友好：$n=4,p=10$ 抛 `ValueError: kth(=7) out of bounds (4)`。先自己校验。

### 6.3 什么时候不用 MCD

- 数据本身就是重尾分布（不是「少量离群点」）：MCD 会把重尾的尾部当离群点剪掉，估出的是**截断分布**的协方差，与下游假设不一致。
- $n$ 很小（$n<2p$）：h 太小，协方差本身估不稳。
- 只需要 $\Sigma$ 的**平均**行为、不需要抗污染：收缩类更简单。

## 七、用哪个损失口径评估？

四个口径，回答四个不同问题，**不要混用**：

| 口径 | 怎么算 | 回答什么 | 实测示例 |
| --- | --- | --- | --- |
| **held-out NLL** | `-est.score(X_test)` | 对下游似然/预测有没有用 | LW 7.295 vs 经验 7.344（$p=5,n=40$） |
| **$\Sigma$ 的误差** | `error_norm(true_cov)` 或平均绝对误差 | 估计得准不准 | GLasso 0.0186 vs LW 0.0470（$n/p=10$） |
| **Mahalanobis 分离度** | 污染点均值 / 内点均值 | 能不能区分异常 | MCD 124.3/3.1 vs 经验 10.3/2.2 |
| **支持集 F1** | 与真值边的重合度 | 结构恢复对不对 | α=0.4 的 0.833 vs CV 的 0.255 |

**本库综合**：这四个口径在本节里**系统性冲突**——CV 的 α 给最小 $\Sigma$ 误差但最多边；MCD 给最好的分离度但改变 $\det$ 的量级；NLL 最优的 α 在 $n/p=8$ 时不是 LW 的 α。所以「哪个协方差估计器更好」这个问题**必须先说清用哪个口径**。

## 八、15 条坑（按踩到的频率排序）

1. **`assume_centered=True` 用在未居中的数据上**：对角偏大、`location_` 静默为零（实测 1.348→2.550）。
2. **把 `support_` 当 h**：读到的是重加权掩码（178 vs 102）。
3. **以为奇异协方差会报错**：`pinvh` 静默给伪逆，`score` 返回 `-inf`。
4. **在未标准化的数据上跑 GLasso**：量纲差 1000 倍时 F1 掉 1/3。
5. **忽略 GLasso 的 `ConvergenceWarning`**：近共线 + 小 α 刷出上百条，结果不可信。
6. **把 `GraphicalLassoCV` 的边数当结构**：多 2.8–6.8 倍。
7. **把 LW/OAS 的 α 当「下游最优」**：它优化 $\Sigma$ 的 MSE。
8. **用 `ShrunkCovariance()` 的默认 0.1 当「合理默认」**：它是任意值，$p=20,n=20$ 时 NLL 39.8 vs LW 30.6。
9. **在 $n=2$ 上指望 LW 收缩**：α 恒为 0（代数恒等式）。
10. **对 LDA 的 `shrinkage=1` 期待「只保留方差」**：实测是标量乘单位阵。
11. **以为 `shrinkage='auto'` 和 `shrinkage=0.2` 等变**：前者等变、后者不等变。
12. **`store_precision=False` 后读 `precision_`**：属性存在但值为 `None`。
13. **`block_size` 被当成精度参数**：实测 100/1000/17 结果逐位相同，它只影响内存。
14. **`MinCovDet` 不固定 `random_state` 就做 A/B**：本例虽可重复，但 FastMCD 是随机搜索，跨数据不保证。
15. **跨版本引用 α̂**：1.6.1 的 `MDS.n_init` 那类默认值变更在协方差模块也存在风险（`eps`、`covariance=` 等参数），引用时写版本。

## 九、可自动化程度

| 步骤 | 可自动化 | 说明 |
| --- | --- | --- |
| 判断 $n/p$ 档位 | **完全** | 纯算术，可按 §2 的表直接选族 |
| `assume_centered` 自检 | **完全** | 检查 `abs(mean)` |
| α 网格 + held-out NLL 选择 | **完全** | 有验证集即可，指标确定 |
| 标准化 → GLasso 路径扫描 | **完全** | 边数-α 曲线可脚本化 |
| 支持集 F1 | **半自动** | 有真值才可算；无真值时需 bootstrap 稳定性 |
| MCD 的污染率设定 | **半自动** | `support_fraction` 可由假设的污染率反推，但需要业务输入 |
| 「该不该用协方差模型」 | **不可** | 取决于下游任务与假设（[[gaussian-discriminant-analysis]] 的边界） |
| 结构恢复结果是否可信 | **不可** | 需要领域知识判断边是否有意义 |

## 关联页面

- 机制：[[covariance-estimation]]
- 来源：[[sklearn-covariance]]、[[sklearn-outlier-detection]]（`EllipticEnvelope` = MCD + 阈值）
- 实践：[[outlier-detection-selection-and-tuning]]（把同一组估计器当诊断工具）、[[linear-models-as-baselines]]（LDA/QDA 的收缩配置）
- 工具：[[scikit-learn]]
