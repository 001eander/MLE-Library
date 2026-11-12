---
title: 成分分解的选型与调参：先问「要方差、要零件、还是要主题」
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - decomposition
  - dimensionality-reduction
  - pca
  - nmf
  - topic-model
  - workflow
sources:
  - "[[sklearn-decomposition]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档与签名）；行为在 1.6.1 复核
---

# 成分分解的选型与调参

## 先做四问

在选 `PCA` / `NMF` / `FA` / `FastICA` / `LatentDirichletAllocation` 之前，先回答四个问题：

1. **你要的是「压缩」还是「解释」？**
   - 压缩：PCA、TruncatedSVD、IncrementalPCA。
   - 解释：SparsePCA、DictionaryLearning、NMF、FA、LDA（成分本身要能读）。
2. **$X$ 有负值吗？**
   - 有：不要直接上 NMF（要求非负）。先检查是否能平移/取绝对值，或改用 PCA/FA。
   - 没有：NMF、SparsePCA、LDA（计数）都可考虑。
3. **你要不要给新样本 `transform`？**
   - 要：PCA/TruncatedSVD/IncrementalPCA/SparsePCA/DictionaryLearning/NMF/FA/FastICA/LDA 都有 `transform`（但语义不同）。
   - 不要：可以只看 `fit_transform`；但仍建议保留 `components_` 以便解释。
4. **你的评估标准是什么？**
   - 重建/方差：PCA、TruncatedSVD、NMF（`reconstruction_err_`）。
   - 下游任务：把 `transform` 结果接分类/回归，用 CV。
   - 解释：成分稀疏度、主题词、因子载荷——**这些都不是 sklearn 的标量指标**，需要人工或额外度量。

> [!warning] 不要用「哪个图更好看」选模型
> 来源的 faces 图（[[sklearn-decomposition]], L186）里，NMF 面板并不比 PCA 更「零件化」；真正的局部斑块出现在 SparsePCA 与正字典。**图是示例，不是选型依据。**

## 预处理契约

| 方法 | 中心化 | 缩放 | 非负 | 其它 |
| --- | --- | --- | --- | --- |
| `PCA` | **自动** | **不自动** | 不要求 | `whiten=True` 改输出协方差 |
| `IncrementalPCA` | 自动 | 不自动 | 不要求 | 分块；内存随 batch |
| `TruncatedSVD` | **不中心化** | 不自动 | 不要求 | 文本 LSA 常用；中心化后等价 PCA |
| `SparsePCA` | 自动（内部） | 不自动 | 不要求 | `alpha` 控制稀疏 |
| `DictionaryLearning` | 不自动 | 不自动 | 可选 `positive_code`/`positive_dict` | 字典原子单位球 |
| `NMF` | 不自动 | 不自动 | **要求** | 负值会报错；平移会改零的含义 |
| `FactorAnalysis` | 自动（内部） | 不自动 | 不要求 | 噪声逐特征 |
| `FastICA` | 内部白化 | 不自动 | 不要求 | **`whiten` 必须开**（或外部白化） |
| `LatentDirichletAllocation` | 不自动 | 不自动 | **要求计数/非负** | 输入文档-词矩阵 |

**三条硬前提**（来源论述）：

- **PCA 只中心化、不缩放**（[[sklearn-decomposition]], L16）。要不要缩放由你决定；本机 iris 上 2 维 PCA + logistic 的 CV：原始 0.96、标准化 0.9133——**缩放不是必定更好**。
- **TruncatedSVD 不中心化**（L134–L136）。文本 tf-idf 矩阵通常不中心化；若你对它做 PCA，等于先减去均值，会破坏稀疏性。
- **ICA 必须白化**（L268）。`whiten='unit-variance'` 是 1.3 起的默认；`whiten=False` 在 1.6.1 会忽略 `n_components`（输出回到原特征数）。

## 按目标选族

| 目标 | 首选 | 次选 | 不要用 | 理由 |
| --- | --- | --- | --- | --- |
| 线性降维/可视化 | `PCA` | `TruncatedSVD`（稀疏文本） | `NMF`（若只为压缩） | PCA 有解析解、可解释方差 |
| 非线性降维 | `KernelPCA` | [[manifold-learning]] 的 Isomap/LLE/t-SNE | `PCA`（线性） | 核方法能展开非线性；但反投影是近似 |
| 稀疏可解释成分 | `SparsePCA` | `DictionaryLearning` | `PCA`（稠密） | $ℓ_1$ 把系数压到零 |
| 图像/文本「零件」 | `NMF` | `DictionaryLearning` | `PCA`（有负系数） | 非负加法表示 |
| 潜变量/噪声建模 | `FactorAnalysis` | PPCA（`PCA` 的概率解释） | `PCA`（无噪声模型） | FA 允许异方差噪声 |
| 盲源分离 | `FastICA` | — | `PCA`（只去相关） | ICA 最大化独立，不只看方差 |
| 主题发现 | `LatentDirichletAllocation` | `NMF` | 聚类（硬标签） | 主题是分布，不是簇 |
| 大数据外存 | `IncrementalPCA` / `MiniBatchNMF` / `MiniBatchDictionaryLearning` | `TruncatedSVD` | 全量 `PCA`/`NMF` | 内存/时间 |

**综合（本库）**：如果只想要一个「安全的线性基线」，先用 `PCA` + 下游 CV；如果成分需要给人读，再考虑 SparsePCA/NMF；如果数据是文本计数，先试 TruncatedSVD + 下游模型，再试 LDA/NMF 主题。

## $k$、`alpha` 与主题数怎么定

### PCA/TruncatedSVD 的 $k$

来源只给「保留大部分方差」的定性说法（L55），没有给阈值。可操作的三条路（**本库综合**）：

1. **累计解释方差**：`explained_variance_ratio_.cumsum()` 到 0.90/0.95；简单，但会受特征相关性影响。
2. **下游 CV**：把 `transform` 结果接分类/回归，扫 $k$；最贴近用途，但成本 × 折数。
3. **重构误差/平行分析**：用重建误差或与随机矩阵的方差对比；本节没有给公式，属于外部方法。

> [!note] `n_components=None`
> `PCA` 的默认 `n_components=None`（1.9.1 签名）会取 `min(n_samples, n_features)`；`PCA(2)` 才是指定 2 维。不要以为默认就是 2。

### SparsePCA/DictionaryLearning 的 `alpha`

- `alpha` 越大，零越多、重构越差（L89）。本机 iris（3 成分）：0.01/0.1/1/10 → 零 3/3/8/10，重构误差 1.903/1.911/2.4/10.721。
- **先扫 `alpha` 再看下游 CV**：只看零的比例会选到过罚的点。
- `DictionaryLearning` 的 `alpha` 作用在**码**上（L176），与 SparsePCA 作用在成分上不同；不要直接复用同一个网格。

### NMF 的 $k$、`init`、`beta_loss`

- $k$：没有内置选择；用 `reconstruction_err_` 扫，或下游 CV。
- `init`：稠密数据优先 `nndsvda`/`nndsvdar`，稀疏数据可试 `nndsvd`；`solver='mu'` 时不要用会引入大量零的 `nndsvd`（L304）。本机 4 成分误差：`nndsvd` 4.4395、`nndsvda` 2.3878、`nndsvdar` 1.4714、`random` 2.0252。
- `beta_loss`：计数数据常用 KL（$\beta=1$），频谱常用 IS（$\beta=0$），欧氏重构用 Frobenius（$\beta=2$）；**不同 $\beta$ 的误差不可比**（本机 3.9579/2.475/1.6254）。

### LDA 的主题数

- `n_components` 是主题数；没有内置「最佳主题数」。用 `perplexity` 扫、下游任务评估，或人工看主题词。
- `learning_method='batch'` 更稳，`'online'` 更快但依赖 `batch_size` 与学习率（L396）。
- **不要用聚类指标（silhouette/ARI）评估 LDA**：文档-主题是分布，不是硬标签。

## 求解器与规模

| 场景 | 选择 | 注意 |
| --- | --- | --- |
| 数据能进内存、要精确 | `PCA(svd_solver='full')` | $O(n_{max}^2 n_{min})$ |
| 只要前 $k$ 个成分 | `PCA(svd_solver='randomized')` | 反投影不精确（L67） |
| 数据不能进内存 | `IncrementalPCA(partial_fit)` | 内存随 batch，不随 $N$ |
| 核方法、$n$ 大 | `KernelPCA(eigen_solver='randomized'/'arpack')` | 成分数受 $n$ 限制；反投影近似 |
| 字典学习、大数据 | `MiniBatchDictionaryLearning(partial_fit)` | 无默认停止条件；靠 `max_iter`/`max_no_improvement` |
| NMF、大数据 | `MiniBatchNMF(partial_fit)` | `forget_factor` 让旧批次衰减 |
| LDA、大数据 | `LatentDirichletAllocation(learning_method='online', partial_fit)` | 质量依赖 batch 与学习率 |

**成本口径**：本节没有给完整复杂度表；上表的复杂度只在 randomized PCA/KernelPCA 处有来源，其余是**本库综合**。报告成本时应写「估计器拟合次数 × 每次迭代成本」，不要只写「很快/很慢」。

## 初始化、随机性与可复现

1. **固定 `random_state`**：PCA（randomized/arpack）、SparsePCA、DictionaryLearning、NMF、FastICA、LDA 都有随机性。
2. **固定 `init`**：NMF 的 `init` 比 `random_state` 更能改变结果；`nndsvd`/`nndsvda`/`nndsvdar`/`random` 是四个不同起点。
3. **固定 `solver`**：NMF 的 `cd` 与 `mu` 支持不同的 `beta_loss`；`mu` 不能更新零（L304）。
4. **报告符号/排列**：PCA/ICA 的成分可翻符号；NMF 的 W/H 可互相吸收尺度；比较两次结果前先对齐。
5. **多 seed 报告分布**：NMF/LDA 的非凸性使单次结果不可靠；至少跑 3–5 个种子，报告重构误差/困惑度的均值与极差。

## `transform` 语义与 Pipeline

| 估计器 | `fit_transform` 返回 | `components_` | `transform(X_new)` |
| --- | --- | --- | --- |
| `PCA` | scores（$n\times k$） | 主成分（$k\times p$） | 投影到主成分 |
| `TruncatedSVD` | $U\Sigma$（$n\times k$） | $V^\top$（$k\times p$） | $XV$ |
| `SparsePCA` | 编码（$n\times k$） | 稀疏成分（$k\times p$） | 用固定成分求编码 |
| `DictionaryLearning` | 码（$n\times k$） | 字典（$k\times p$） | 用固定字典稀疏编码 |
| `NMF` | **W**（$n\times k$） | **H**（$k\times p$） | 用 H 求新 W |
| `FactorAnalysis` | 因子得分（$n\times k$） | 载荷（$k\times p$） | 用载荷求得分 |
| `FastICA` | 独立成分（$n\times k$） | 解混矩阵（$k\times p$） | 用解混矩阵变换 |
| `LatentDirichletAllocation` | 文档-主题（$n\times k$） | 主题-词（$k\times p$） | 用主题-词求文档-主题 |

**Pipeline 写法**：

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    StandardScaler(),          # PCA 不缩放，显式决定
    PCA(n_components=2, random_state=0),
    LogisticRegression(max_iter=1000),
)
```

- **只在训练折 `fit`**：`PCA`/`TruncatedSVD` 虽然无监督，仍应只在训练集拟合，测试集只 `transform`。
- **`IncrementalPCA.partial_fit` 的泄漏**：如果你在全量数据上分块 `partial_fit`，测试集信息已经进入成分；正确做法是只在训练集上分块。
- **NMF 的 `W` 要保存**：`fit_transform` 返回的 W 不会自动存进模型；`transform` 会重新求 W，不保证逐位相同（本机差 3.72e-4 / 7.68e-4）。

## 16 条坑

1. **PCA 不缩放**（L16）：量纲大的特征独占成分；但缩放也不是必定更好（本机 CV 0.96→0.9133）。
2. **`whiten=True` 不是「更好」**：它把低方差方向放大；下游对噪声敏感时会变差。
3. **TruncatedSVD 不中心化**（L136）：在非文本数据上会保留均值方向；中心化后才等价 PCA。
4. **`n_components=None` 不是 2**：PCA 默认取 `min(n_samples, n_features)`。
5. **randomized 的 `inverse_transform` 不精确**（L67）：重构任务要比较，不能假设。
6. **SparsePCA 的 `alpha` 过罚**：本机 `alpha=10` 误差 10.721；零多不等于模型好。
7. **`split_sign` 不是稀疏化**：长度翻倍，结果非负；正文写作 `split_code`，API 参数是 `split_sign`。
8. **`SparseCoder` 有空的 `fit`**：它不学习字典，但方法存在；不要用它判断「能不能进 Pipeline」。
9. **NMF 喂负值**：直接报错；平移会改变零的含义。
10. **NMF `mu` + `nndsvd`**：零锁死，误差更差（L304；本机 4.4395 vs 1.4714）。
11. **比较不同 `beta_loss` 的误差**：量纲不同（本机 3.9579 vs 1.6254）。
12. **`transform(X)` ≠ `fit_transform(X)`**：NMF 差 3.72e-4/7.68e-4；训练集编码要保存。
13. **FastICA `whiten=False` 忽略 `n_components`**：输出回到原特征数（本机 4 维）。
14. **FA 成分不正交**（L254）：旋转后解释会变；不要把载荷当 PCA 主轴。
15. **LDA 主题不是簇**：文档-主题是分布；用 silhouette/ARI 评估会误导。
16. **在测试集上重新 `fit`**：无监督也有泄漏；成分只能来自训练集。

## 可自动化程度

| 步骤 | 可自动化 | 需要人判断 |
| --- | --- | --- |
| 预处理（中心化/缩放/非负检查） | 高（检查 dtype/min） | 缩放是否改变语义 |
| 选族 | 中（按上表规则） | 「解释」是否重要 |
| 选 $k$ | 中（扫 + CV） | 可解释性与压缩的权衡 |
| 选 `alpha`/`beta_loss`/主题数 | 低 | 业务含义与人工读主题 |
| 初始化/种子扫描 | 高（脚本） | 选哪个解 |
| 结果解释 | 低 | 成分/主题命名 |
| 报告 | 高（模板） | 符号/排列对齐说明 |

**综合（本库）**：最值得自动化的是「预处理检查 + 候选模型扫描 + 种子扫描 + 下游 CV」；最不该自动化的是「成分/主题命名」与「可解释性判断」。AutoResearch 可以把前者做成循环，把后者留给人。

## 检查清单

- [ ] $X$ 是否满足非负/计数要求（NMF/LDA）？
- [ ] 是否显式决定缩放（PCA 不缩放）？
- [ ] 是否只在训练集 `fit`，测试集只 `transform`？
- [ ] `n_components` 是显式选择还是默认 `None`？
- [ ] `random_state`/`init`/`solver` 是否固定并记录？
- [ ] `transform(X)` 与 `fit_transform(X)` 是否已区分（NMF）？
- [ ] 是否检查了成分的符号/排列/旋转？
- [ ] 是否用下游 CV 而不是只看重构误差？
- [ ] 是否报告了 `beta_loss`/`alpha`/主题数等会改变量纲的参数？
- [ ] 解释性结论是否标注了「来源论述 / 综合 / 假设」？

## 关联页面

- 概念：[[matrix-factorization]]（$X\approx UV$ 骨架、六族约束、可辨识性）、[[manifold-learning]]（图方法换坐标）、[[feature-selection]]（删列 vs 投影）、[[clustering]]（硬标签 vs 软编码）、[[gaussian-discriminant-analysis]]（有监督 LDA）、[[regularization-and-sparsity]]（$ℓ_1$/$ℓ_2$）
- 来源：[[sklearn-decomposition]]（逐节摘要、版本判定、转换损失、本机实测）
- 工具：[[scikit-learn]]（`decomposition` 模块的估计器矩阵与默认值）
