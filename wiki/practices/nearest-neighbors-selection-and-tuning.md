---
title: "最近邻的选型与调参：缩放前提、k 与搜索结构、预计算图与 NCA 的 A/B"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - nearest-neighbors
  - scikit-learn
  - hyperparameter-tuning
  - workflow
sources:
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-linear-models]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.7.x – 1.9.x 的 neighbors 模块（用户指南 1.6 节；该节 1.7/1.8/1.9 正文无实质差异，见 [[sklearn-nearest-neighbors]] 的版本判定）
  - 本页的「实测」条目在本机 scikit-learn 1.6.1 上执行，与该节所属版本不同，只用于确认定性行为（默认值与 `auto` 规则两版一致）
  - 用户指南 1.6 节的选型内容约 20% 在剪藏丢失的折叠块里，本页引用官方 1.9 页原文时单独标注
---

# 最近邻的选型与调参

本页回答四个具体问题：**这一族什么时候该用**、**必做哪些前置步骤**、**三个真正能调的东西怎么调**、**哪些默认值会沉默地骗你**。事实来自 [[sklearn-nearest-neighbors]]，引用写作 `[[sklearn-nearest-neighbors]], L103`；标「官方 1.9 页」的取自剪藏丢失的折叠块原文，标「实测」的是本次 ingest 在本机 scikit-learn 1.6.1 上跑出来的，标「本库判断」的是本库的工程推断。

概念层（为什么这样、失效机制是什么）见 [[nearest-neighbors]]；本页只管怎么做。

## 一、先判「该不该用」，再谈参数

KNN 没有需要拟合的参数，所以「调参」在这里的含义与前面几族完全不同：不是调正则强度，而是**调邻域定义**（$k$ 或 $r$、权重、度量）。因此第一件事不是选 $k$，而是确认数据落在它的优势区间里。

| 判据 | KNN 有利 | KNN 不利 | 依据 |
| --- | --- | --- | --- |
| 样本量 $N$ | 小样本；暴力搜索「can be very competitive」 | $N$ 大时暴力「quickly becomes infeasible」 | L138 |
| 维度 $D$ | 低维（KD 树在 $D<20$ 高效） | 高维：半径邻域受「curse of dimensionality」打击 | L105、L144 |
| 决策边界 | 高度不规则（非参数方法的长处） | 边界本就近似线性时，多余自由度只是噪声 | L14 |
| 推理成本 | 可以离线预计算邻居图 | 线上延迟必须随 $N$ 增长 | L194–L236 |
| 模型体积 | 可接受「模型＝训练集」 | 需要下发小模型 | L12、L99 |
| 可解释性 | 可以给出「哪 $k$ 个样本」 | 没有系数、没有类条件分布 | L99 |
| 概率输出 | 只需相对排序 | 需要校准概率（`predict_proba` 是投票占比） | 外部核查，见下方说明 |

**本库判断**：最后两行是最常被忽略的。`predict_proba` 的归一化口径是「加权投票占比」，不是似然，因此它**可以直接当排序用、不能直接当概率用**（外部核查：`sklearn/neighbors/_classification.py` 的注释即写 `normalize 'votes' into real [0,1] probabilities`）。要校准就走 `CalibratedClassifierCV`，与 [[svm-selection-and-tuning]] 里 SVM 的处理相同。

## 二、缩放是前提，不是优化项

来源对度量的表述是「in general, any metric measure: standard Euclidean distance is the most common choice」（L12），并说稀疏输入只支持 Minkowski 家族（L16）。**整节没有一句提到特征缩放**——这是本节最值得记录的文档缺口。

但官方示例自己给 KNN 套了缩放（外部核查：`plot_classification.py` 与 `plot_nca_classification.py` 都写成 `Pipeline(StandardScaler(), KNeighborsClassifier(...))`，图 1 与图 8/9 即出自这两条流水线）。

**本库综合**：这里的尺度问题比线性模型更靠前——线性模型里尺度进的是**优化**（惩罚项、一阶求解器），KNN 里尺度进的是**定义**。距离怎么算，模型就是什么：

- 不缩放时，方差最大的特征支配整个距离，其余特征近似被忽略；
- 因此**换缩放 = 换模型**，不是「让训练更稳」；
- 「不缩放」比「缩放」更需要理由（例如所有特征本来就是同一物理量纲）。

与 [[svm-selection-and-tuning]] 对照：那里官方用 "highly recommended to scale" 明说了，这里只能靠示例脚本反推。**结论相同，证据强度不同**。

## 三、$k$：唯一的容量旋钮，但它是离散的且与实现耦合

$k$ 在来源里只有一句论述（L103）：

> The optimal choice of the value $k$ is highly data-dependent: in general a larger $k$ suppresses the effects of noise, but makes the classification boundaries less distinct.

没有默认值、没有选取方法。默认值是 **5**（外部核查：1.9.1 签名；实测：本机 1.6.1 的 `KNeighborsClassifier()` / `KNeighborsRegressor()` / `NearestNeighbors()` / `KNeighborsTransformer()` 均为 `n_neighbors=5`）。

$k$ 的特殊之处在于它同时决定三件事：

1. **平滑度**（L103）。$k=1$ 时零训练误差、边界贴合噪声；$k$ 大时接近全局多数类。
2. **查询成本**。$k$ 增大要搜索更大范围的参数空间，且 $k>1$ 时树遍历需要内部结果队列（官方 1.9 页，剪藏丢失的折叠块）；$k$ 相对 $N$ 很大时剪枝失效，暴力反而更快。
3. **走哪条实现路径**。`algorithm='auto'` 在 $k \ge N/2$ 时直接选暴力（官方 1.9 页；实测：$N=40$、`n_neighbors=20` → `_fit_method == 'brute'`）。

**本库判断（工程建议，非来源）**：$k$ 是**离散**参数，不能用梯度式或连续网格的直觉。实践上

- 先在 $\{1, 3, 5, \dots\}$ 这类小数集上扫，不要一上来扫 1–100；
- 判断「信号在不在」用大跨度（1 / 5 / 15 / 50），判断「最优点在哪」再加密；
- $k$ 的扫描与 `weights` 必须**联合**扫：`weights='distance'` 在大 $k$ 下等价于「只看最近的几个邻居」，两个参数不是正交的（本库判断）。

**不平衡数据必须单独说**：多数票天然偏向多数类，而 $k$ 固定时少数类的邻域可能被多数类完全淹没。来源未提这一点。可选做法是按类别分层采样后再 fit，或用 `weights='distance'` 让近邻压过数量——但两者都改变的是「谁参与投票」，不是「票怎么算」，效果需实测。

## 四、`weights`：三档取值，两个隐藏行为

来源给出同一套三档（分类 L107、回归 L121）：`'uniform'`（默认）、`'distance'`（权重 $\propto 1/d$）、可调用对象（接收距离数组、返回权重数组）。

图上能看到差别（本库目视核查，见 [[sklearn-nearest-neighbors]] 的图表核查）：图 1 里 `uniform` 的边界是少量长直段，`distance` 的边界出现明显锯齿与小块凹陷；图 2 里 `uniform` 是分段常数阶梯，`distance` 是穿点曲线并在个别位置出现尖峰。**含义**：`distance` 更贴合局部、更不平滑。

两个不在剪藏里的行为，本次 ingest 已在本机实测确认：

1. **零距离会让 `distance` 退化成 1-NN。** 构造 $X=[0,1,2,3]$、$y=[0,1,1,0]$、$k=3$，在训练点 $x=0.0$ 处查询：
   - `weights='uniform'` 的邻居标签是 $\{0,1,1\}$，多数票应为 **1**；
   - `weights='distance'` 实际返回 `[0]`、`predict_proba=[[1., 0.]]`，即**只有距离 0 的那个点拿了全部权重**。

   回归路径同样如此：$X=[0,1,2]$、$y=[0,10,20]$、$k=3$ 时在 $x=0.0$ 处 `predict` 返回 `0.0`，而等权均值应为 `10.0`。**这回答了两页此前挂着的待解问题**：分类与回归确实共用同一个 `_get_weights`，零距离规则在回归上同样生效（实测，1.6.1）。
2. **投票平票取最小标签。** 两个等距邻居、标签分别为 $\{1,0\}$ 时 `predict` 返回 `0`；把标签顺序换成 $\{0,1\}$ 仍返回 `0`（实测，1.6.1）。所以结果**不依赖训练集行顺序**，但**依赖标签编码**——重编码会沉默地改变平票结果。

第 2 条与来源 L25 的警告是两件事，不要混：L25 说的是**距离**并列（第 $k$ 与第 $k+1$ 个邻居距离相同、标签不同）时结果依赖训练数据顺序；平票规则说的是**计数**并列。两条都不会报错。

## 五、`algorithm` 与 `leaf_size`：只改速度，但必须选对

三种结构算的是同一个数学对象（L136–L156），所以**选错只慢不错**——这是本库目前唯一能把「实现选择」与「模型选择」干净分开的地方。代价口径（官方 1.9 页，剪藏丢失的折叠块；剪藏只剩 L142 的「总共 $O[DN\log N]$」）：

| 结构 | 查询代价 | 受什么影响 |
| --- | --- | --- |
| brute | $O[DN]$ | 基本不受 $D$、$k$、数据结构影响 |
| ball tree | 约 $O[D\log N]$ | 稀疏、内在维度低的数据更快；对任意结构比 KD 稳 |
| KD tree | $D<20$ 时约 $O[D\log N]$；$D$ 大时近 $O[DN]$ | 轴对齐划分，对任意结构改善不如 ball tree；**开销可能让它比暴力更慢** |

三条选型事实（都来自这块丢失的折叠块，因此**剪藏里查不到**）：

- **小样本时暴力更快**：$N \lesssim 30$ 时 $\log N$ 与 $N$ 同级。
- **`k` 很大时暴力更快**：剪枝能力下降。
- **查询点很少时暴力更快**：树的构造成本摊不掉。

`algorithm='auto'` 的规则（官方 1.9 页原文；实测 1.6.1 行为一致）：

> selects 'brute' if any of: 输入稀疏；`metric='precomputed'`；$D > 15$；$k \ge N/2$；`effective_metric_` 不在 `kd_tree`/`ball_tree` 的 `VALID_METRICS` 里。否则在 `kd_tree` 与 `ball_tree` 中取第一个包含该度量的。

实测复核（1.6.1）：`n_features=15` → `kd_tree`，`n_features=16` → `brute`。这与 L144 的「$D<20$ 高效」**不是同一个数**（15 vs 20），两者都是经验阈值，来源没有解释差异。

**本库判断（实操）**：

- 默认 `'auto'` 不要动。它已经按 $D$、$k$、稀疏性、度量四条规则选了路径，手动指定反而更容易踩到「树在 $D$ 大时更慢」。
- 想知道实际走的哪条，读 `_fit_method`（下划线私有属性，仅用于诊断）。
- `leaf_size` 只在树路线上有意义（官方 1.9 页明确：对暴力查询无引用）。默认 30 是「太大变暴力、太小走路开销大」的折中；BallTree 的存储约为训练集的 $1/\text{leaf\_size}$ 倍——这是它唯一真正影响内存的参数。
- 度量支持面不同：`KDTree.valid_metrics` 9 个，`BallTree.valid_metrics` 24 个，含 `mahalanobis`、`hamming`、`haversine`、`pyfunc`（L91、L93）。**要用 `mahalanobis` 就只能走 ball tree**，`auto` 会自动选它。

## 六、预计算邻居图：把最贵的一步变成可复用中间产物

L194–L236 是本节工程价值最高的一段：把「邻居」从估计器里拿出来，做成可缓存的稀疏图（`kneighbors_graph` / `radius_neighbors_graph` / `KNeighborsTransformer` / `RadiusNeighborsTransformer`）。

三个用法：

1. **换其他超参时复用同一张图**。配 `Pipeline(..., memory=缓存目录)`，改下游参数不会重算邻居（doctest 见 L201–L214）。
2. **用 `n_jobs` 细分并行**——不是所有下游估计器都暴露这个参数（L216）。
3. **换成近似最近邻或特殊数据类型的实现**（L218），只要满足格式契约。

格式契约六条（L220–L230）：CSR（COO/CSC/LIL 也接受）；只存相对训练集的近邻、**必须含距离 0 的点与对角线**；每行 `data` 升序（可选，乱序会被稳定排序并多一次开销）；值非负；行内 `indices` 不重复；下游用 $k$ 近邻时每行至少存 $k$（或 $k+1$）个。

**$k$ 含不含自身**（L232 note）：两种定义都不完美——含自身时训练与测试的非自身邻居数不同；不含自身时 `fit(X).transform(X)` 与 `fit_transform(X)` 不一致（违反 sklearn API）。`KNeighborsTransformer` 选择含自身，但 `mode='distance'` 时**多算 1 个**以兼容其他估计器。官方建议自定义实现一律多给 1 个。

实测确认这条「多算 1 个」：$n\_samples=3$、`KNeighborsTransformer(mode='distance', n_neighbors=3)` 在 `transform` 时抛出 `n_neighbors = 4, n_samples_fit = 3`（实测，1.6.1）。所以**自定义 ANN 变换器必须多给 1 个**，否则会在边界样本上失败。

自定义实现的检查单（本库整理）：

- [ ] 输出 CSR；每行 `indices` 无重复、`data` 非负；
- [ ] 明确决定「含不含自身」，并记录在类文档里；
- [ ] `mode='distance'` 时多算 1 个；`mode='connectivity'` 时给二值（`SpectralClustering` 要这个）；
- [ ] 训练集与自身比时**保留对角线**（距离 0）；
- [ ] 下游是 $k$ 近邻时，确认每行 $\ge k$ 个（建议 $k+1$）；
- [ ] 用 `check_estimator` 之外再手测 `fit(X).transform(X) == fit_transform(X)`。

## 七、零参数基线：`NearestCentroid`

来源称它「has no parameters to choose, making it a good baseline classifier」（L168），并在非凸类、类方差差异极大时失效，因为假设所有维度方差相同。

**用法（本库判断）**：

- **先缩放**。它本质是「比到类质心的欧氏距离」，与 KNN 同一类尺度依赖。这一条来源未提。
- **默认构造就是零超参**，适合当第一行基线。
- 一旦设 `shrink_threshold` 就不再是零超参：它是「除以类内方差 → 减阈值 → 过零置零」的软阈值（L184），需要 CV 选阈值，官方示例的收益只有 0.81 → 0.82（L186）。

它也是 [[gaussian-discriminant-analysis]] 谱系的最强假设端：LDA 令 $\Sigma=\sigma^2 I$ 就退化成它，因此可以当作「LDA 失效是因为协方差估错了，还是因为模型类选错了」的对照实验（详见该页）。

## 八、NCA：当作需要 A/B 的增强，不是默认项

NCA 学的是线性变换 $L$（L240、L288–L304），与 KNN 组合使用；`transform` 后可直接做监督降维（L276–L284）。

先看成本再决定要不要试（L316、L320）：

- 训练要存 $n_{samples}^2$ 的成对距离 → **$N=10^5$ 时不可行**；
- 每次迭代 $O(n_{components}\cdot n_{samples}\cdot\min(n_{samples},n_{features}))$，乘 `max_iter`（默认 50，外部核查）；
- `transform` 是 $L X^{T}$，无额外空间。

再看证据。来源的措辞是「aims to improve the accuracy」（L240）与「has been shown to work well in practice」（L250）——都是**目标**，不是保证。官方示例图的实测读数是一次反例（本库目视核查图 8/9）：

| 配置 | 准确率 |
| --- | --- |
| `KNN (k = 1)` | **0.90** |
| `NCA, KNN (k = 1)` | **0.89** |

同一脚本、同一 `test_size=0.7, random_state=42`、同两个特征（外部核查示例脚本：`X[:, [0, 2]]`）。而剪藏里的 doctest 是 `0.9619`（L269），四特征、$k=3$，**没有给同设置的 KNN 对照**——所以**不能**把 0.9619 读成「NCA 的增益」。

A/B 写法（本库判断）：

1. 同一个 CV 折、同一个预处理流水线、同一个 $k$；
2. 两臂只差 NCA 这一步：`KNeighborsClassifier` vs `Pipeline(NCA(), KNeighborsClassifier)`；
3. 若用了 `n_components`，两臂的比较就不再是「要不要 NCA」，而是「要不要这个降维」——分开报；
4. 报均值 ± 折间标准差，单次切分的 1 个点差异不构成证据（见 [[leaderboard-vs-cv]]）。

值得试的场合（本库判断）：样本量中等（$N \lesssim 10^4$）、特征数不小、且欧氏距离明显不合适（同类样本在原始空间里不近）。不值得试的场合：$N$ 很大（内存 $O(N^2)$）、需要极简流水线、或已经用了其他度量学习。

## 九、可复现性：三个沉默的依赖

| 依赖 | 现象 | 检查成本 |
| --- | --- | --- |
| 训练集**行顺序**（L25） | 第 $k$/$k+1$ 邻居距离相同、标签不同时，结果取决于行顺序 | 打乱后重跑 |
| **标签编码** | 投票平票取最小标签（实测，1.6.1），重编码会改变预测 | 标签随机重映射后重跑 |
| **零距离** | `weights='distance'` 在某点退化到 1-NN（实测，1.6.1），且不报警 | 检查查询点是否落在训练集上 |

三者都不会报错。**本库判断**：在拿到一条 KNN 分数之前，先跑一遍「打乱行顺序 + 随机重编码标签」的稳定性检查，成本很低。

## 十、坑清单（按「踩到才发现」排序）

1. **不缩放就跑**。度量即模型，方差最大的特征吃掉距离。
2. **`weights='distance'` 撞上训练点**。静默退化成 1-NN，`predict_proba` 变成 one-hot。
3. **把 `predict_proba` 当概率**。它是投票占比，未校准。
4. **`k` 设得比 `N/2` 大**。`auto` 直接切暴力，你以为在用树。
5. **在高维上用 `RadiusNeighbors*`**。半径内可能一个点都没有，默认 `outlier_label=None` 直接抛 `ValueError`（外部核查）。
6. **在 `D>15` 上手动指定 `kd_tree`**。这是 `auto` 明确会避开的情形（官方 1.9 页）。
7. **`leaf_size` 调到极大**。查询退化成暴力，且没有任何警告（官方 1.9 页）。
8. **预计算图漏掉对角线或零距离点**。下游静默算错，不报错（L224）。
9. **自定义 ANN 变换器不多给 1 个邻居**。边界样本上 `n_neighbors > n_samples_fit`（实测）。
10. **`NearestCentroid(metric=...)` 传入第三个度量**。1.5 起只接受 `'euclidean'`/`'manhattan'`，其余直接报 `InvalidParameterError`（实测，1.6.1）。
11. **用 NCA 而不做 A/B**。官方示例里它并不总是赢（0.89 vs 0.90）。
12. **在大数据上开 NCA**。$O(N^2)$ 内存，先看 `n_samples`。

## 十一、对智能体的可自动化程度

| 环节 | 可否自动 | 说明 |
| --- | --- | --- |
| 缩放的判断 | 半自动 | 可枚举「不缩放 / StandardScaler / RobustScaler」当三臂，但必须报告，不能默认 |
| $k$ 与 `weights` 联合扫描 | **可全自动** | 空间小（两个离散参数），CV 成本低 |
| `algorithm` / `leaf_size` | 不该自动 | 不影响预测，只在超大规模上影响时间；自动搜索会浪费预算 |
| 预计算图缓存 | 可全自动 | `Pipeline(memory=...)`，纯工程设置 |
| NCA 是否启用 | 半自动 | 可当第四臂自动跑，但必须报对照 |
| 概率输出 | 需人工确认 | 是否需要校准取决于下游用法，代码看不出来 |
| 不平衡处理 | 需人工确认 | 分层、加权、`weights='distance'` 三者改变的东西不同 |

**本库判断**：这一族是自动化搜索里**最便宜**的族（两个离散超参、没有收敛问题、没有随机性），因此适合当任何任务的**第一个基线**——它给出的分数可以当「这个任务的信号有多强」的标尺，而成本只有几秒。

## 待解问题

1. **$k$ 有没有可迁移的起点？** 来源只说 "highly data-dependent"（L103）。本库不收录无来源的经验法则（如 $k\approx\sqrt{N}$），需要另找文献。
2. **`weights='distance'` 撞上训练点的实际发生率有多高？** 实测确认了机制，但没有量化「留一交叉验证时必然发生」这一后果——LOO 下每个查询点都是训练点，是否意味着 `weights='distance'` + LOO 恒等价于 1-NN？（本库推断，未实测）
3. **`auto` 的 $D>15$ 与 L144 的 $D<20$ 两个阈值从哪来？** 两者都是经验值，来源未解释。
4. **`leaf_size` 在 $N=10^5$ 以上是否仍以 30 为佳？** 官方只说 30 是折中，没有数据。
5. **不平衡数据下 $k$ 与 `weights` 的联合最优是否稳定？** 本库只有机制推断，没有实验。
6. **预计算图 + 近似最近邻的精度损失有多大？** L218 只说「可以用」，没有召回/精度口径。

## 关联页面

- 来源：[[sklearn-nearest-neighbors]]
- 概念：[[nearest-neighbors]]（为什么这样、失效机制）、[[anomaly-detection]]（kNN 图的第三种消费方式：LOF 的局部密度比）、[[gaussian-discriminant-analysis]]（`NearestCentroid` 是 LDA 的等方差极限）、[[regularization-and-sparsity]]（原型收缩＝软阈值的第五个面）
- 实践：[[linear-models-as-baselines]]（零参数基线的位置）、[[outlier-detection-selection-and-tuning]]（`n_neighbors` 与度量的异常检测用法）、[[svm-selection-and-tuning]]（同一类「缩放是硬前提」）、[[competition-ml-workflow]]（基线在竞赛流程里的位置）、[[leaderboard-vs-cv]]（单次切分的差异不构成证据）
- 工具：[[scikit-learn]]
