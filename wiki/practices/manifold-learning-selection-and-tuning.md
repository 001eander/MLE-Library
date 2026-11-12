---
title: "流形学习的选型与调参：先缩放、再定 $k$、最后挑被保量；15 条坑与可自动化程度"
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - dimensionality-reduction
  - manifold-learning
  - scikit-learn
  - workflow
sources:
  - "[[sklearn-manifold-learning]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.9.1（接口与文档论述；默认值取自 1.9.1 API 页签名）
  - scikit-learn 1.6.1（本页标「实测」的行为；`MDS` 的默认值在 1.9 已变，见第 6 条）
---

# 流形学习的选型与调参

本页回答四个问题：**该不该用、用哪个方法、$k$ 与维数怎么定、结果怎么验**。机制与"被保量"对照表在 [[manifold-learning]]，来源、图读数与 15 条实测在 [[sklearn-manifold-learning]]。引用写作 `[[sklearn-manifold-learning]], L176`。

## 一、先判「该不该用」

**合适的三种场景**：

1. **可视化与探索**（$d_\text{out}=2$ 或 3）。这是来源给的首要用途（L32、L162），也是唯一不要求 `transform` 的场景。
2. **怀疑数据落在低维流形上，且下游模型对冗余维敏感**（KNN、核方法、小样本线性模型）。
3. **需要一个去噪后的几何先验**（例如看簇是否真的分开、看顺序结构是否单调）。

**不该用的四种场景**：

- **需要给新样本定位**。`TSNE`、`MDS`、`ClassicalMDS`、**`SpectralEmbedding`** 都**没有 `transform`**（1.6.1 实测 `hasattr`）；`Isomap` 与 `LocallyLinearEmbedding` 虽有，但只是外推近似（训练集上 `transform` 与 `embedding_` 差 3.4e-13 / 2.3e-4），两者的坐标含义并不相同（本库判断）。生产管线里要做特征变换，用 PCA 或直接训一个 encoder（[[multi-layer-perceptron]]）。
- **$N$ 很大**。这些方法对 $N$ 基本是二次的（$O[dN^2]$，官方折叠块；见 [[manifold-learning]] 的成本表），t-SNE 的 Barnes-Hut 也只到 $10^5$ 量级（折叠块）。
- **需要跨运行稳定复现**。LLE 的 arpack 求解器两次 `fit` 的坐标最大差 3.39e-06，arpack 与 dense 之间差 0.2755（实测 8）；t-SNE 不同种子的坐标最大差 3.54（实测 12）。
- **需要概率或可解释的每个维度**。降维后的第 1 维没有语义；要语义就得回头看原特征（[[feature-selection]]）或做投影后的载荷分析（PCA）。

## 二、按「保什么」选方法

| 你的目标 | 首选 | 次选 | 不要用 | 依据 |
| --- | --- | --- | --- | --- |
| 展开单个连续流形（Swiss roll、S-curve） | **Isomap** | LLE（$k$ 小时） | Spectral（会把二维压成一维弧，图 16） | L56、图 14/16 |
| 找簇、看局部结构、类别分离 | **t-SNE** | Spectral Embedding | Isomap/LLE（它们的目标是展开，不是分簇） | L155 |
| 保住全局距离的近似（有理论目标） | **classical MDS / PCA** | metric MDS | t-SNE（全局保真度低一档） | L128、实测表 |
| 只需要序（不相似度只有排序意义） | **non-metric MDS** | — | metric MDS（会强行拟合数值） | L122 |
| 必须精确、要解析解、要可重复 | **`ClassicalMDS`** | PCA | 任何迭代法 | L138 |
| 有预计算的距离/邻接矩阵 | Isomap / `SpectralEmbedding(affinity='precomputed')` | t-SNE(`metric='precomputed'`, 需 `init='random'`) | LLE（无 precomputed 入口） | 外部核查签名、实测 9 |

**来源唯一给过的选型句**（L155）："While Isomap, LLE and variants are best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples."——把这句话当第一道分叉就够用：**要展开 → Isomap/LLE；要分簇 → t-SNE**。

## 三、`n_neighbors`（$k$）：最敏感的参数，且有硬下界

这一节**完全不给 $k$ 的选法**（[[sklearn-manifold-learning]] 局限 3），但 $k$ 同时决定三件事：图的连通性、LLE 的秩亏程度、MLLE/HLLE 是否能跑。

**硬约束（实测 4–5，1.6.1）**：

| `method` | 约束 | 违反时 |
| --- | --- | --- |
| `standard` | 无 | — |
| `modified` | `n_neighbors >= n_components` | `ValueError: modified LLE requires n_neighbors >= n_components` |
| `hessian` | `n_neighbors > n_components*(n_components+3)/2` | `ValueError: … must be greater than [n_components * (n_components + 3) / 2]`（$n_c=3$ 时 $k\ge7$） |
| `ltsa` | 无校验 | 内部形状错误（`could not broadcast … (2,2) … (2,3)`） |

**经验做法（本库综合，非来源）**：

1. 从 $k=10$ 起；$n_c=2$ 时把 $k\in\{5,10,20,50\}$ 扫一遍，用 `trustworthiness` + 全局 Spearman 选（见第五节）。
2. **前提是先缩放**（第四节）。不缩放时扫 $k$ 是在优化一个错误的图。
3. Isomap 额外看**连通性**：$k$ 太小时会 `UserWarning: The number of connected components … Completing the graph to fit Isomap might be slow`（实测 11），这是"你的 $k$ 太小"的官方信号——而补边后的测地距离是人造的。
4. **SpectralEmbedding 的 `n_neighbors=None` 等于 $n/10$**（外部核查 1.6.1 源码与实测：$n=200\to20$、$500\to50$、$1000\to100$）。这是个惊人的默认：数据越多、邻域越大，局部结构被抹平得越厉害——**用 Spectral 时必须显式给 `n_neighbors`**。

## 四、缩放是硬前提，不是优化项

来源的原话：「Make sure the same scale is used over all features. Because manifold learning methods are based on a nearest-neighbor search, the algorithm may perform poorly otherwise.」[[sklearn-manifold-learning]], L176

**实测代价**（S-curve 400 点 + 4 个噪声特征，其中一个放大 1,000 倍；`n_neighbors=10`）：

| 输入 | Isomap 的 trustworthiness（对真实流形坐标） | LLE 的 trustworthiness |
| --- | --- | --- |
| 原始尺度 | **0.5029** | 0.5028 |
| `StandardScaler` 后 | **0.8952** | 0.5671 |

读法：**不缩放时 Isomap 几乎瞎了**（0.50 接近随机嵌入的 0.51）；LLE 恢复得少（0.57），因为它连 `reconstruction_error_` 都是 0.0（局部权重被完美拟合，几何信息全丢）——**"误差为 0" 不是好兆头**，是这个退化。

## 五、输出维数与评估：三个指标，别只看一个

**（1）来源的建议只对 Isomap 成立。** L177 说「用重构误差定维数，误差会随 `n_components` 下降直到等于本征维数」。实测：

| 方法 | 指标 | 结果 | 与 L177 一致？ |
| --- | --- | --- | --- |
| Isomap | `reconstruction_error()`（**方法**） | S-curve：0.3986 → 0.0918 → 0.0802（$d=1,2,3$）；digits：3141 → 2360 → 1327 → 955 → 898 → 884（$d=1,2,5,10,20,30$） | 方向一致，但**没有 $d$ 处的拐点**（30 维仍在降） |
| LLE | `reconstruction_error_`（**属性**） | digits：0.000000 → 0.000007 → … → **0.007899**（$d=1\to12$） | **相反**（单调上升） |
| Spectral / t-SNE | 无此指标 | t-SNE 有 `kl_divergence_` | 不适用 |

**（2）用 `trustworthiness` 定 $k$ 与 $d$**：`trustworthiness(X, X_embedded, n_neighbors=5)`。它只惩罚"假邻居"（不该近的变近了），所以**对 t-SNE 天然有利**，跨方法比较时要意识到这一点（[[manifold-learning]] 评估节）。

**（3）用全局 Spearman 定"要不要保住全局结构"**（本库补充）：原始与嵌入的成对欧氏距离的相关（200 样本子集单独拟合，不能与上一段的 500 样本 `trustworthiness` 直接并列比较）：PCA +0.6456、t-SNE(`init='pca'`) +0.6201、Isomap +0.5518、t-SNE(`init='random'`) +0.5141、LLE +0.4616、Spectral +0.4199。

**（4）$d$ 的选择在 API 层的两个陷阱**：

- `LocallyLinearEmbedding` 要求 `n_components <= n_features`，否则 `ValueError: output dimension must be less than or equal to input dimension`（实测 3）。
- `Isomap` 没有这个限制（64 维数据上 `n_components=70` 也能跑），而 `n_components == n_samples` 时报的是 `ValueError: There are significant negative eigenvalues …`——**错误信息完全不提你的维数请求**（实测 15）。

## 六、t-SNE 的五个参数（来源在折叠块里丢了，本页按官方 1.9.1 页补）

剪藏只保留了 t-SNE 的机制与四条缺点；五个优化参数的说明在官方 `Optimizing t-SNE` 折叠块里（[[sklearn-manifold-learning]] 转换损失）。要点：

1. **`perplexity=30.0`**：有效邻居数，$=2^{S}$（$S$ 是条件分布的 Shannon 熵）。**大而噪声多的数据需要更大的 perplexity**；小数据上要给小值（且必须 $<$ `n_samples`，实测 `perplexity=600`、$n=500$ → `ValueError: perplexity must be less than n_samples`）。
2. **`early_exaggeration=12.0`**：早期把原空间联合概率乘一个因子以拉开簇间空隙；过大时 KL 会在该阶段上升。一般不用调。
3. **`learning_rate='auto'`**：官方文字写的是"样本量除以 early exaggeration"，**实现是 `max(N / early_exaggeration / 4, 50)`**（1.6.1 源码与实测：$N=500$ → 50、$N=8000$ → 166.67）。**小样本上 `auto` 会一直被 50 的下限卡住**，因此"auto"在小数据上等于常数。
4. **`max_iter=1000`、`n_iter_without_progress=300`**：官方说迭代数通常不需要调。实测 5 个种子都在 999 轮结束（`n_iter_`），说明默认预算够用。
5. **`method='barnes_hut'`（默认）与 `angle=0.5`**：Barnes-Hut 是近似法，`angle` 只在这里有意义；精确法在 500 样本上 KL 更低（0.33716 vs 0.39931）。**Barnes-Hut 硬限 $n_c\le3$**（`n_c=4` → `ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm`）。

**种子策略（来源明确允许，L157/L163）**：跑多个 `random_state`，取 `kl_divergence_` 最小者。实测跨度很小（0.39580–0.39934，0.0035）而**坐标差很大（3.54）**——所以要报"最优 KL"，不要报"平均坐标"或跨种子比较坐标。

## 七、MDS 的三种配置与一个尺度陷阱

- **`MDS`（metric / non-metric）是迭代法，`ClassicalMDS` 是特征分解**：同一份 digits 的图上读数是 2.393s（metric）、16.625s（non-metric）、**0.105s**（classical）。若只需要"距离的近似"，先跑 classical 作为基线。
- **`stress_` 不能跨配置比较**（实测 13）：metric + `normalized_stress='auto'` → 780.03；non-metric + `'auto'` → 0.29784（Stress-1）；non-metric + `False` → 140.16。**报告 MDS 结果必须同时写 `metric_mds` 与 `normalized_stress`。**
- **1.9 把 `n_init` 从 4 改成 1**（外部核查）：实测 `n_init=1` 的 stress 431.06，`n_init=4` 是 419.09（同一数据、同一 `random_state`）。**同一份代码换版本会变差**，要稳定就显式写 `n_init=4`（或更多）。
- **1.8 改名与弃用**：`metric` → `metric_mds`；`dissimilarity` 1.8 弃用、1.10 移除；`init` 新增（默认值将在 1.10 变成 `classical_mds`）。跨版本代码要按目标版本写。

## 八、成本口径（来自图标题的读数）

同一份 digits、同样 2 维输出（来源页图表核查表）：

| 方法 | 图标题时间 | 相对标准 LLE |
| --- | --- | --- |
| 随机投影 | 0.001s | 0.005× |
| TruncatedSVD | 0.003s | 0.02× |
| LDA（用标签） | 0.006s | 0.03× |
| ClassicalMDS | 0.105s | 0.6× |
| 标准 LLE | 0.187s | 1× |
| Isomap | 0.346s | 1.9× |
| HLLE | 1.811s | 9.7× |
| LTSA | 2.090s | 11× |
| metric MDS | 2.393s | 13× |
| t-SNE（Barnes-Hut） | 2.413s | 13× |
| MLLE | 2.434s | 13× |
| non-metric MDS | 16.625s | 89× |

**口径警告**：这些数字是官方示例脚本在作者机器上打印的，`n=1797`、`k` 与 `n_c` 各不相同、版本未知。只能当量级，不能当基准；要在自己的数据上复测。

## 九、坑清单（按「踩到才发现」排序）

1. **`n_neighbors=5` 是所有估计器的默认**（外部核查），而来源全篇不给默认值。$k=5$ 正是官方 digits 示例用的值——**MLLE/HLLE/LTSA 在 $k=5$ 下把数据塌成细丝**（图 7–9），不要把这个默认当成"合理起点"。
2. **`SpectralEmbedding.n_neighbors=None` = $n/10$**（实测），不是固定 10。数据量一涨，局部性就没了。
3. **`MDS.n_init` 在 1.9 从 4 变 1**（外部核查 + 实测 431 → 419）：跨版本对比 MDS 的 `stress_` 会得出"变差了"的错误结论。
4. **`stress_` 三种量纲**（780 / 0.298 / 140），且 1.6.1 里 `metric=True` + `normalized_stress=True` 直接 `ValueError`（1.7 起才支持）。
5. **LLE 的 `reconstruction_error_` 随 `n_components` 上升**（实测 1），而来源说要"下降"。用 `0.0` 的重构误差当"拟合完美"更是危险信号（不缩放时实测就是 0.0，几何全丢）。
6. **`Isomap` 的 `reconstruction_error` 是方法、不是属性**（`reconstruction_error()`）；写成下划线会 `AttributeError`。
7. **`LocallyLinearEmbedding` 拒绝稀疏输入**（`TypeError: Sparse data was passed for X, but dense data is required`），而 `Isomap`/`SpectralEmbedding` 接受。
8. **`SpectralEmbedding(affinity='nearest_neighbors')` 遇到稀疏输入会静默回退到 `rbf`**（实测 9，带 warning）——图的定义被换掉而模型名没变。另外它**根本没有 `transform`**（1.6.1），只能 `fit_transform`。
9. **t-SNE 的稀疏失败点在 `init='pca'`**，不在 `method`：要喂稀疏矩阵必须 `init='random'`（实测 10）。
10. **`eigen_solver='arpack'` 的奇异失败只发生在窄条件下**：实测 `method='modified', k=3, n_c=3` 复现 `Factor is exactly singular`，改 `'dense'` 即好；但"重复点""不连通"这两个来源说的条件我没复现（实测 6）。所以错误信息里那句 "`eigen_solver='dense'` is recommended" 是**唯一可靠的逃生口**。
11. **LLE 的坐标不可复现**（arpack 两次差 3.39e-06；arpack vs dense 差 0.2755），但**结构可复现**（trust 0.9196 vs 0.9197）。要位级复现就用 `eigen_solver='dense'`；要位置无关的结论就用 `trustworthiness`。
12. **t-SNE 的 `learning_rate='auto'` 不是 $N/\text{ee}$，而是 $\max(N/\text{ee}/4, 50)$**（实测 $N=8000$ → 166.67）。小样本上它恒为 50。
13. **Barnes-Hut 硬限 `n_components<=3`**（`n_c=4` 报错），而 `TSNE` 的 `n_components` 默认是 2——所以「用 t-SNE 做 10 维特征」这件事默认就会报错。
14. **`n_components <= n_features` 只对 LLE 家族成立**；Isomap 允许 $n_c > D$（64 维上跑 70 维），`n_c == n_samples` 的报错信息是「负特征值」，与真实原因无关。
15. **不连通图有两种不同待遇**：Isomap **自动补边**（测地距离变成人造的，只有 warning），Spectral **只 warning 不修图**（结果不可信）。两者都不报错。

## 十、对智能体的可自动化程度

| 步骤 | 可自动化 | 条件 |
| --- | --- | --- |
| 缩放 | **完全** | 强制 `StandardScaler`；不允许关闭（实测 0.5029 → 0.8952） |
| $k$ 扫描 | **完全** | 需要显式跳过违反硬下界的组合（`modified`/`hessian`） |
| 维数选择 | **部分** | Isomap 可用 `reconstruction_error()`；其他方法必须用 `trustworthiness` + 全局 Spearman，而这两个指标的"好"阈值依赖数据 |
| 方法选择 | **部分** | 「展开 vs 分簇」可由下游用途判定（L155）；「哪个方法在这份数据上更好」只能靠扫描 |
| t-SNE 种子扫描 | **完全** | 按 `kl_divergence_` 取最小；必须报告种子数与跨度 |
| 图的连通性检查 | **完全** | 捕获 `UserWarning`（`n_connected_components`）与 `trustworthiness` 联判 |
| 稀疏/稠密兼容 | **完全** | 先试 `csr_matrix`，按异常回退 `toarray()` 或 `init='random'` |
| 结果解释 | **不可** | 嵌入空间的轴没有语义；簇的分离度不等于可分性（官方折叠块原文：t-SNE 上分不开**不**意味着模型不能分类） |

## 待解问题

1. `n_neighbors` 的最优值在缩放后是否与 `n_components` 有关？本页只做了 $k$ 的一维观察，没有二维扫描。
2. `SpectralEmbedding` 在 $n=10^5$ 以上、默认 $n/10$ 邻居时的内存行为未测。
3. 降维放进 `Pipeline` 的正确写法（能否用 `fit_transform` + 下游 `fit`）、以及每折重建图的成本，未实测。
4. `trustworthiness` 的 `n_neighbors` 默认 5 与建模用的 $k$ 应该一致吗？没有来源讨论。
5. 1.9.1 上 `LocallyLinearEmbedding.reconstruction_error_` 是否仍是"随 $d$ 上升"（本机只有 1.6.1）。

## 关联页面

- 概念：[[manifold-learning]]（被保量对照表、成本模型、评估指标）
- 来源：[[sklearn-manifold-learning]]（18 张图的读数、15 条实测、8 条分歧）
- 工具：[[scikit-learn]]（`manifold` 模块的默认值清单，第 101–115 条）
- 相邻实践：[[nearest-neighbors-selection-and-tuning]]（kNN 图的代价与 `algorithm` 选择）、[[clustering-selection-and-tuning]]（同一份无监督数据的另一条路线：分组而非换坐标）、[[feature-selection-workflow]]（另一条降维路线：删列）、[[mlp-selection-and-tuning]]（有监督地"学表示"）
