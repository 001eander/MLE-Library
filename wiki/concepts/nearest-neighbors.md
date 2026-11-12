---
title: 最近邻方法（Nearest Neighbors）：非泛化学习、邻域定义与学到的度量
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - nearest-neighbors
  - instance-based
  - metric-learning
  - nonparametric
sources:
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-outlier-detection]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.7.x – 1.9.x 的 `sklearn.neighbors`（用户指南 1.6 节无实质改动，见 [[sklearn-nearest-neighbors]] 的版本判定）
  - 默认值与两条掩码规则已在 scikit-learn 1.6.1 上实测（见 [[sklearn-nearest-neighbors]] 转换损失 E 节）；跨 1.6→1.9 的稳定性仅在这几项上得到确认
---

# 最近邻方法

本页整理「不做泛化、只保存样本」这一族方法：邻域的两种定义、三个真正的旋钮、三种搜索结构、两个退化/扩展形态（最近质心与 NCA），以及它们的失效条件。事实来自 [[sklearn-nearest-neighbors]]，引用写作 `[[sklearn-nearest-neighbors]], L103`；标「外部核查」的部分取自 1.9.1 源码或官方页的折叠块，不在剪藏内。

## 一句话定义与它的三个推论

来源的措辞是「不构造一般化的内部模型，只是保存训练实例」（L99）；引言里补上机制："they simply 'remember' all of its training data"（L12）。

**本库综合**：这条定义把「模型」与「训练集」等同，于是三件事自动成立：

1. **没有训练阶段**，只有建索引；「训练成本」是数据结构的函数，不是参数个数的函数；
2. **推理成本 ∝ 训练集规模**，所以它的优势区间是小样本、低维、边界高度不规则（L14）；
3. **没有参数 = 没有参数共享**，多输出回归按列独立做，人脸补全例子（L125）就是 768 个输出各自取邻域均值。

代价同样直接：**模型体积等于训练集体积**，且「解释模型」=「解释数据」，没有系数、没有支持向量、没有类条件分布可读。

## 两种邻域定义：固定数量 vs 固定体积

| | $k$ 近邻（`KNeighbors*`） | 半径邻域（`RadiusNeighbors*`） |
| --- | --- | --- |
| 邻域 | 最近的 $k$ 个点（整数，L101） | 半径 $r$ 内所有点（浮点数，L101） |
| 邻居数 | 恒为 $k$ | 随局部密度变化（L105） |
| 适合 | 均匀采样 | **非均匀采样**：稀疏处自动少用邻居（L105） |
| 空邻域 | 不会发生 | 会发生；默认直接报错（外部核查） |
| 高维 | 退化（所有点都「远」） | 退化更彻底（半径内可能一个点都没有，L105） |
| 失效后果 | 稀疏区的 $k$ 个邻居其实很远 | 密集区邻居过多，计算成本不受控 |

**本库综合**：$r$ 可以读作「局部自适应的 $k$」——密集处等价于大 $k$、稀疏处等价于小 $k$。两者是**两种正则化方向**：$k$ 控制「平均多少个样本」，$r$ 控制「平均多大范围」。前者保住了「每点计算量恒定」，后者保住了「邻域的空间尺度一致」。

## 三个旋钮（以及一个不明显的第四个）

| 旋钮 | 作用 | 默认值（外部核查） | 剪藏是否提默认值 |
| --- | --- | --- | --- |
| `n_neighbors`（$k$） | 容量：越大越平滑、边界越模糊（L103） | **5** | 未提 |
| `weights` | `'uniform'` 等权 / `'distance'` 权重 $\propto 1/d$ / callable（L107） | `'uniform'` | 提了 |
| `metric`（+ `p`、`metric_params`） | 定义「近」是什么（L12、L16） | `'minkowski'`、`p=2` | 只泛泛提 |
| `algorithm` / `leaf_size` | **只影响速度**，不影响预测（原则上） | `'auto'` / `30` | 前者提取值、后者只在代码里出现 |

四点必须一起看：

1. **$k$ 与训练集规模耦合**。$k$ 越大，每次查询要访问的邻居越多；外部核查显示 `algorithm='auto'` 在 `k >= n_samples // 2` 时直接退回暴力搜索——所以 $k$ 同时是**平滑度**与**实现路径**的开关。
2. **$k$ 没有闭式最优、也没有跨模型换算**。L103 只说 "highly data-dependent"。这与 SVM 的 $C=1/\alpha$（[[support-vector-machines]]）形成对照：这里没有任何「把别人的正则强度搬过来」的公式。
3. **`weights='distance'` 的零距离规则是隐藏的**（外部核查，`_get_weights`）：$1/d$ 出现 $\infty$ 时，该行整体替换成「是否等于 0」的掩码——**命中训练点就等价于 1-NN**。分类与回归共用这个函数（**已实测**，1.6.1：回归 $X=[0,1,2]$、$y=[0,10,20]$、$k=3$ 在 $x=0$ 处返回 `0.0` 而非均值 `10.0`）。这不是理论边角：留一交叉验证下每个查询点都是训练点，因此 `weights='distance'` 配 LOO 时该规则**恒定生效**（本库推断）。
4. **`p` 只在 `metric='minkowski'` 时生效**，且 $p=1/2/\infty$ 会被改写成 `manhattan`/`euclidean`/`chebyshev`（外部核查）。给别的度量传 `p` 不会报错，也不会有任何作用。

## 距离是超参：尺度契约

前几个模型族的尺度问题都出现在**优化**里（惩罚项吃尺度、一阶求解器吃尺度，见 [[linear-models-as-baselines]]）；这里尺度出现在**定义**里：距离怎么算，模型就是什么。

- 来源说度量「in general, any metric measure」，并说欧氏是「the most common choice」（L12）；稀疏输入只支持 Minkowski 家族（L16）。
- **剪藏一句没提特征缩放**。但官方示例给 KNN 套的是 `Pipeline(StandardScaler(), KNeighborsClassifier(...))`（外部核查：`plot_classification.py`、`plot_nca_classification.py`），图 1 与图 8/9 都出自这类流水线。
- **本库综合**：KNN 的判定在「所有特征同单位、同量级」时才有意义——否则度量被方差最大的特征支配。这不是数值稳定性问题（不像 `sag`），而是**模型定义问题**：换缩放 = 换模型。因此「不缩放」比「缩放」更需要理由。

## 三种搜索结构：同一数学对象，三种代价

| 结构 | 构造 | 查询 | 维度的作用 | 度量支持 |
| --- | --- | --- | --- | --- |
| brute force | 无 | $O[D N^{2}]$ 全对（单次 $O[DN]$，L138） | 无 | 最广（含 callable、`precomputed`） |
| KD tree | 很快（只沿坐标轴切，不算 $D$ 维距离，L144） | 约 $O[D\log N]$；每查询 $O[\log N]$ 次距离计算 | $D<20$ 高效，之后接近 $O[DN]$（L144） | 9 个（L91） |
| ball tree | 较贵（质心 + 半径，L148） | 三角不等式给节点内所有点的上下界，一次距离计算即可剪枝（L150–L156） | 高维、结构化数据可胜过 KD（L156） | 24 个，含 `mahalanobis`、`hamming`、`haversine`、`pyfunc`（L93） |

三点结构性说明：

1. **它们算的是同一个东西**，选错只慢不错——这与 `SVC(kernel='linear')` vs `LinearSVC`（换实现即换模型，见 [[support-vector-machines]]）正好相反。本库目前只有这里能干净地把「实现选择」与「模型选择」分开。
2. **三角不等式是 ball tree 的唯一机制**：$\lVert x+y\rVert\le\lVert x\rVert+\lVert y\rVert$（L153，剪藏公式已损坏）。节点内所有点都落在以质心 $C$、半径 $r$ 的超球里，因此「到质心的距离」给出到所有点的上下界，不必逐点算。
3. **维度让树失效的机制不同**：KD 树失效是因为「轴对齐划分」在高维退化成几乎每次都要访问多数节点；球树失效是因为「半径相对数据尺度不再有区分度」（本库综合；来源只给了「curse of dimensionality」与「highly structured data」两句，L105、L156）。

**外部核查（剪藏丢失的折叠块）**：官方原文把查询代价写成暴力 $O[DN]$、ball $O[D\log N]$、KD 在 $D$ 小时约 $O[D\log N]$ 并在 $D$ 大时退化到近 $O[DN]$（「树的开销甚至可能让它比暴力慢」）；$N\lesssim30$ 时 $\log N$ 与 $N$ 同级，暴力更快；`leaf_size` 就是「切回暴力」的阈值。**这段在剪藏里完全不存在**，所以只看剪藏会把 L142 的 $O[DN\log N]$ 误读成「树总是更快」。

## 从「邻居」到「图」：预计算契约

来源用一整节（L194–L236）把「邻居」从估计器里拿出来，做成可缓存的稀疏图。三个好处与六条格式要求：

- **两种图**：`connectivity`（二值邻接，给 `SpectralClustering`）、`distance`（距离图，给 `DBSCAN`）（L198）。
- **三个好处**：改别的超参时复用同一张图（配 `Pipeline(memory=…)`）；用 `n_jobs` 细分并行；换成近似最近邻或特殊数据类型的自定义实现（L200–L218）。
- **六条格式**：CSR（COO/CSC/LIL 也接受）；只存相对训练集的近邻且**必须含距离 0 的点与对角线**；行内 `data` 升序（可选，乱序会多一次稳定排序）；非负；`indices` 不重复；下游用 $k$ 近邻时每行至少 $k$（或 $k+1$）个（L220–L230）。
- **$k$ 的歧义**：`n_neighbors` 含不含自身，两种定义都有问题（训练/测试的非自身邻居数不同，或破坏 `fit(X).transform(X) == fit_transform(X)`）；`KNeighborsTransformer` 选「含自身」，但 `mode='distance'` 时多算 1 个；官方建议自定义实现一律多给 1 个（L232）。

**本库综合**：这是本库第一条「把最贵的一步变成可复用中间产物」的具体路径。它的意义不在 KNN 本身，而在于**近似最近邻可以外挂**——只要满足六条格式，就可以把任意 ANN 库接进 sklearn 的聚类/流形/分类器里（L218）。

## 形态一：最近质心（NearestCentroid）

来源把它当作「每类一个质心」的算法，并说它类似 `KMeans` 的标签更新阶段（L168）。

- **边界是线性的**（本库推导）：到两类质心等距的集合是一个超平面，所以它属于线性分类器，只是没有「权重」这个概念——质心之差就是权重方向。
- **假设**：所有维度方差相同（L168）。因此它在**非凸类**与**类方差差异极大**时失效；来源直接把 LDA/QDA 列为「不做此假设的更复杂方法」。
- **与 LDA 的关系**（本库综合）：LDA 假设类内协方差相同且任意，判别函数里出现 $\Sigma^{-1}$；把 $\Sigma$ 取成 $\sigma^2I$ 时 $\sigma^2$ 约掉，就退化成「比到质心的欧氏距离」。所以 `NearestCentroid` 是 LDA 谱系里假设最强的一端（[[gaussian-discriminant-analysis]] 的嵌套关系再往下加一层）。
- **最近收缩质心**（L184）：把每个类的每个特征**除以该类内方差**，减去 `shrink_threshold`，过零置零 → 该特征不再参与比较。官方唯一读数：0.81 → 0.82（L186）。

**本库综合**：`shrink_threshold` 与 Lasso 的软阈值是**同一个算子** $S(z,\alpha)=\operatorname{sign}(z)\max(0,|z|-\alpha)$（对比 [[regularization-and-sparsity]] 的软阈值一节），但作用对象是「标准化后的质心偏移」，不是回归系数。差别有两个后果：

1. 它产生**特征维的精确 0**，因此可以直接读「哪些特征被删了」（`deviations_`，外部核查：1.9.1 起还有 `centroids_` 与 `within_class_std_dev_` 可读）；
2. 因为先除类内方差，它对**特征单位**的敏感度与 Lasso 不同——但这不等于「尺度无关」（本库判断，未验证）：距离最终仍在原始空间里算。

## 形态二：NCA（学一个度量）

NCA 不学分类器，学的是**线性变换** $L$，使得在 $L$ 投影后的空间里做随机最近邻分类的正确率最大（L240、L288–L304）：

$$\max_{L}\sum_{i}p_i,\qquad p_i=\sum_{j\in C_i}p_{ij},\qquad p_{ij}=\frac{\exp\left(-\lVert Lx_i-Lx_j\rVert^{2}\right)}{\sum_{k\ne i}\exp\left(-\lVert Lx_i-Lx_k\rVert^{2}\right)}$$

**外部核查（剪藏丢失的折叠块）**：这等价于学一个**平方 Mahalanobis 度量** $M=L^{T}L\succeq0$，即

$$\lVert L(x_i-x_j)\rVert^{2}=(x_i-x_j)^{T}M(x_i-x_j)$$

**本库综合**：这条等式把 NCA 放进本库已有的两套几何里——$M$ 与 [[gaussian-discriminant-analysis]] 的 Mahalanobis 是同一个对象（那里由协方差给出，这里由优化给出），$L$ 与 [[support-vector-machines]] 的核映射 $\phi$ 则是「改几何」的两种不同手段（一个显式低秩、一个隐式高维）。区别在监督信号：NCA 用标签直接优化「邻居是不是同类」。

四条使用事实：

| 事实 | 来源 | 含义 |
| --- | --- | --- |
| 输出一个 $(n_{components}, n_{features})$ 的 $L$，可 `transform` 做**监督降维** | L240、L278 | 它同时是降维器；`n_components` 默认 `None` = 不降维（外部核查） |
| 训练存 $n_{samples}^{2}$ 成对距离 | L316 | 内存 $O(N^2)$，$N=10^5$ 不可行 |
| 每次迭代 $O(n_{components}\cdot n_{samples}\cdot\min(n_{samples},n_{features}))$ | L316 | 与 `max_iter`（默认 50）相乘，外加 L-BFGS-B 的全梯度 |
| 无类分布假设，边界可不规则 | L250 | 与 LDA 的分界点就在这句 |

**本库综合（重要边界）**：NCA 的「更好」是**目标**，不是保证。官方示例图在同一脚本、同一 $k=1$、同一两特征切分下给出 **KNN 0.90 / NCA+KNN 0.89**（本库目视核查图 8/9），而剪藏的 doctest（0.9619，L269）没有给对照。因此 NCA 应被视为「需要 A/B 验证的增强」（见 [[nearest-neighbors-selection-and-tuning]]）。

## 形态三：LOF（把 kNN 图变成密度比）

`LocalOutlierFactor` 与本页前两个形态不同：它不输出标签，而输出「**局部密度比**」——某点的分数 = 其 $k$ 近邻的平均局部密度 / 该点自身局部密度（[[sklearn-outlier-detection]], L164）。正常点 ≈ 1，异常点 ≫ 1。

三条与本页直接相关的后果：

1. **$k$ 在这里是唯一的容量旋钮，且有上下两个硬边界**：下界是「一个簇至少要有的点数」（否则簇内点互相成为局部异常），上界是样本数（[[sklearn-outlier-detection]], L166）。上界**不被校验**：实测 $n=330$ 时 `n_neighbors=400` 被静默夹到 329，结果退化成「全正常」（0 个异常）。
2. **度量就是模型**（与本页「距离是超参」同一条）：实测同一数据 `metric='cosine'` 报 8 个异常、`euclidean`/`manhattan` 各报 24 个（[[sklearn-outlier-detection]]「本机实测」23）。
3. **它是传导式（transductive）估计器**：默认只有 `fit_predict`，训练集的分数在 `negative_outlier_factor_`；切成 `novelty=True` 之后，`score_samples` 在**训练点上**与 `negative_outlier_factor_` 最大差 0.586（自身是否算进自己邻域），所以两个数不能互相校验。

**本库综合**：LOF 补上了本页缺失的一层——kNN 图不仅能「分类」和「降维」（Isomap/Spectral/t-SNE），还能**当密度估计器**。反过来，这也是 kNN 图最弱的一种用法：图只保留 $k$ 个邻居，「密集簇」与「稀疏背景」在图上看起来同样连通，因此 LOF **看不见成簇的异常**（实测召回 0.03，而 `IsolationForest` 1.00）。详见 [[anomaly-detection]]。

## 与其他模型族的对照

| | 最近邻 | 线性/惩罚模型 | 判别分析 | 朴素贝叶斯 | SVM |
| --- | --- | --- | --- | --- | --- |
| 学什么 | **什么都不学**（存数据） | 系数 $w$ | 类均值与协方差 | 类条件分布 | 支持向量的对偶系数 |
| 参数个数 | 0 | $d$ | $O(Kd)$ 到 $O(Kd^2)$ | $O(Kd)$ | 由 $C$/ν 与支持向量数决定 |
| 决策面 | 任意（$k$ 越大越平滑） | 线性 | 线性/二次 | 二次（轴向受限） | 由核决定 |
| 尺度依赖 | **强**（度量即模型） | 强 | 弱（不收缩时） | 弱（按特征独立建模） | 强 |
| 能否给概率 | `predict_proba` = 投票占比（外部核查） | logistic 有；其余无 | 有（生成式） | 有但常不可信 | 需校准（1.9 起弃用内建） |
| 推理成本 | ∝ 训练集 | $O(d)$ | $O(Kd^2)$ | $O(Kd)$ | ∝ 支持向量数 |

## 失效模式

| 失效 | 机制 | 迹象 |
| --- | --- | --- |
| 维度诅咒 | 高维下「近」与「远」的对比消失；半径邻域可能为空（L105） | 增加特征后分数下降；`RadiusNeighbors*` 报错 |
| 尺度支配 | 大方差特征吃掉整个距离（本库综合） | 标准化前后结果差异巨大 |
| $k$ 太小 | $k=1$ 零训练误差、边界贴合噪声（L103 的反面） | 训练分数 1.0、测试分数低 |
| $k$ 太大 | 边界过度平滑，接近「全局多数类」（L103） | 少数类被吞 |
| 距离并列 | 结果依赖训练数据行顺序（L25）；投票平票取最小标签（外部核查） | 打乱行顺序后预测变化 |
| 类不平衡 | 多数票天然偏向多数类；$k$ 固定时少数类邻域被淹没（本库综合，来源未提） | 少数类召回极低 |
| 成本失控 | 推理 $\propto N$；NCA 内存 $O(N^2)$（L316） | 线上延迟随数据增长 |
| 预计算图不合规 | 缺零距离/对角线、`indices` 重复、每行不足 $k$ 个（L220–L230） | 下游结果偏或报错 |

## 待解问题

1. **$k$ 的选取有没有可迁移的起点？** 来源只说 "highly data-dependent"（L103），本库目前只有「用 CV 扫」。是否有关于 $k\approx\sqrt{n}$ 之类的经验法则、以及它在多类/不平衡数据上的失效条件，需要另找来源（本库**不收录**无来源的经验法则）。
2. ~~**`weights='distance'` 的零距离规则在回归里是否同样退化为 1-NN？**~~ **已解答（实测，1.6.1）**：回归同样退化。分类与回归共用 `_get_weights`，在训练点处回归返回该点标签而非邻域均值。剩余问题改为：LOO 场景下这条规则的系统偏差有多大（未测）。
3. **`NearestCentroid` 与 LDA 的等价性有多强？** 本库推导是「$\Sigma=\sigma^2I$ 的极限」；`priors` 的 $2\log\pi_k$ 修正与 LDA 的 `priors` 是否同一口径未核（1.9.1 新增参数，用户指南未提）。
4. **$\lVert Lx\rVert$ 与 Mahalanobis 的等价是否意味着 NCA 可以解释为「学一个白化矩阵」？** 若是，NCA 与 LDA 的白化（[[gaussian-discriminant-analysis]]）就是同一操作的两个监督目标。本库未验证。
5. **近似最近邻接入预计算图后，结果偏差有多大？** L218 只说「可以用」，没有精度/召回口径。

## 关联页面

- 来源：[[sklearn-nearest-neighbors]]、[[sklearn-outlier-detection]]（LOF 的全部细节）
- 实践：[[nearest-neighbors-selection-and-tuning]]（缩放、$k$、`algorithm`、NCA 的 A/B 写法）、[[linear-models-as-baselines]]（零参数基线的位置）
- 概念：[[regularization-and-sparsity]]（原型收缩 = 第五个正则化面）、[[gaussian-discriminant-analysis]]（Mahalanobis 与等方差假设）、[[support-vector-machines]]（改几何的另一条路）、[[naive-bayes]]（另一种「弱假设 + 高偏差」的族）、[[manifold-learning]]（kNN 图的另一群消费者：Isomap/Spectral/t-SNE）、[[anomaly-detection]]（kNN 图的密度比消费方式：LOF）
- 工具：[[scikit-learn]]
