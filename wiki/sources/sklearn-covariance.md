---
title: "scikit-learn 用户指南 2.6「Covariance estimation」：经验协方差、收缩（LW/OAS）、稀疏精度矩阵与 MCD 稳健估计（stable 文档剪藏，未回填）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - scikit-learn
  - covariance
  - shrinkage
  - precision-matrix
  - robust-statistics
  - reference-doc
source_path: raw/articles/sklearn/2.6. Covariance estimation.md
source_kind: article
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.9.1（文档正文；行为在本机 1.6.1 复核）
---

# scikit-learn 用户指南 2.6「Covariance estimation」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | 2.6. Covariance estimation（scikit-learn 用户指南第 2.6 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页为项目文档——**本库核查所得**） |
| 形态 | 官方用户指南 HTML→Markdown 剪藏；**0 个代码块、0 条 doctest、1 个行内公式（L32）+ 1 个行间公式块（L101–L103）、2 个 `[!note]` 提示、5 张唯一图、1 个 2×2 表、6 处 Examples 列表** |
| 原文 | <https://scikit-learn.org/stable/modules/covariance.html>（剪藏 frontmatter `source`，未钉版本；2026-11-12 抓取时 `/stable/` = 1.9.1） |
| 本地文件 | `raw/articles/sklearn/2.6. Covariance estimation.md`（**182 逻辑行**、25,054 字节、`wc -w` 2,488 词；末行无换行；**未做任何回填**） |
| SHA-256 | `0d6198017272a70301c04384e342fb0bc5238e64c1eb3f15b3e2bbe830120db6` |
| 完整性 | **散文几乎完整**（官方正文 1,848 词，剪藏正文 1,804 词）：丢 **4 条参考文献条目**与 **4 处页内引用标记**；另有 **5 个脚注把 Ledoit-Wolf / OAS / MCD 小节重复了 631 词**（Web Clipper 伪影），末尾第 2 张表被截断在一行表头。详见「转换损失」 |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.7.0 – 1.9.1，工作版本 1.9.1**（1.6→1.7 的四道措辞判据全部命中；1.7→1.8→1.9 正文逐词 diff 为空；见「版本判定」） |
| 插图 | **5 张唯一图全部下载入库** `raw/assets/sklearn-covariance/`，**5 张均已目视核查**（见「图表核查」） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-covariance]], L28`。凡超出剪藏内容的部分（API 默认值、官方参考文献原文、运行时行为、源码细节）都显式标注「外部核查」「官方 1.9.1 页」或「本机实测」。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `0d619801…`）。丢失的 4 条参考文献与 4 个引用标记只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库一个此前**没有名字的类目**：前十二个 sklearn 文档都在估计「一个模型」（$p(y\mid x)$、簇标签、坐标、成分），2.6 节估计的是**数据本身的二阶结构** $\Sigma$ 与它的逆 $K=\Sigma^{-1}$——它不预测任何东西，但几乎所有模型都在暗地里消费它。四个小节因此是同一个问题的四种回答：

1. **第一次把「估计量」当作研究对象，而不是模型的零件。** 1.2 节的 LDA/QDA 只是**使用**协方差（`covariance_estimator` 指向的就是本节的对象），本节反过来问：$\Sigma$ 本身怎么估、估错会怎样、用什么损失衡量（L16、L28、L42、L81）。这让「协方差估计误差」第一次成为可测量的量（本页实测用 Frobenius / 平均绝对误差 / held-out 负对数似然三种口径）。
2. **四个估计器可以被压成「一个算子 + 一个 α 的来源 + 一个结构假设」。** 经验协方差是 α=0、收缩是 α∈(0,1] 时向 $\frac{\mathrm{Tr}\hat\Sigma}{p}I$ 的凸组合、Ledoit-Wolf 与 OAS 只是两种**解析地**定 α 的规则（L32、L42、L61）。本机实测确认：`LedoitWolf().covariance_` 与 `ShrunkCovariance(shrinkage=LedoitWolf().shrinkage_)` **逐位相同**，OAS 同理。因此「三个估计器」在实现上是一个算子，差别全在 α 的取法。
3. **第一次把「精度矩阵」当成一等输出。** $K$ 正比于偏相关矩阵，$K_{ij}=0$ ⇔ 条件独立（L79），于是「估一个稀疏的 $K$」等价于「从数据里学一张条件独立图」，即 *covariance selection*。这是本库第一次出现「模型参数就是一张图」的对象——与 2.3 节聚类**输出标签**、5.2 节（未入库）输出重要性都不同：这里输出的是**变量之间的边**。
4. **第一次把「估计量的目标函数」与「下游指标」明确分开。** L42 说 LW 的 α 最小化的是「估计的与真实的协方差之间的 MSE」，L61 说 OAS 在高斯假设下 MSE 更小——**都不是**分类准确率或 held-out 似然。本机实测把这条口径量化了：在 $p=5,n=40$ 上，α 的 MSE 最优值（LW 0.207）在 held-out 负对数似然上**输给**手设的 α=0.1（7.516 vs 7.450，8 个种子均值）；反过来在 $p=20,n=20$ 与 $p=20,n=10$ 上，任何 α>0 都远好于 α=0（经验协方差的 NLL 为 `inf`）。
5. **它把前面几页接了起来**：[[gaussian-discriminant-analysis]] 的 `shrinkage`/`covariance_estimator` 有了实体（`LedoitWolf`/`OAS`/`MinCovDet`）、[[naive-bayes]] 的 `var_smoothing` 与收缩有了精确对照、[[clustering]] 里 KMeans 的「等协方差」假设与 GMM 的 $\Sigma_k$ 有了估计层、[[matrix-factorization]] 里 FactorAnalysis 的「低秩 + 对角噪声」是**结构化协方差**的另一条路、[[nearest-neighbors]] 的 Mahalanobis 度量与 NCA 学到的 $M$ 是同一个几何对象。

## 文档结构

| 章节 | 解决什么问题 | 估计对象 | 行号 |
| --- | --- | --- | --- |
| 引言 | i.i.d. 假设；样本量/结构/同质性决定估计质量 | — | L10–L12 |
| 2.6.1 Empirical covariance | MLE 渐近无偏；`assume_centered` 契约 | $\hat\Sigma_{\text{MLE}}$ | L14–L22 |
| 2.6.2 Shrunk Covariance | 病态/不可逆时的凸组合；α 的三种取法 | $(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$ | L24–L75 |
| 2.6.2.1 Basic shrinkage | 用户给定 α；收缩＝特征值平移＝$ℓ_2$ 惩罚的 MLE | 同上 | L26–L38 |
| 2.6.2.2 Ledoit-Wolf | 使 **MSE** 最小的解析 α；各向同性时的反直觉行为 | 同上，α 由数据定 | L40–L57 |
| 2.6.2.3 OAS | 高斯假设下比 LW **MSE 更小**的 α | 同上，α 由数据定 | L59–L75 |
| 2.6.3 Sparse inverse covariance | 学稀疏精度矩阵＝学条件独立图 | $K$（L1 惩罚） | L77–L114 |
| 结构恢复 4 条注意 | 标准化、hub 节点、$n$ vs 边数、CV 选太多边 | — | L89–L97 |
| 2.6.4 Robust Covariance | 离群点会毁掉经验/收缩估计 | 修剪后的 $\Sigma$ | L116–L120 |
| 2.6.4.1 Minimum Covariance Determinant | h 个「好」样本 + 一致性校正 + 重加权 | $\Sigma_{\text{MCD}}$、$\mu_{\text{MCD}}$ | L122–L135 |
| （5 个脚注） | **重复** Ledoit-Wolf / OAS / MCD 正文（631 词） | — | L143–L182 |

## 关键主张（按剪藏顺序）

### A. 经验协方差：MLE 的承诺与两个前提（L16–L18）

- **「样本量足够大」是唯一前提**（L16）：经验协方差是对总体协方差**渐近无偏**的 MLE。剪藏没有给「足够大」的判据，也没有说 $n/p$ 到多少算够——本页的实测表（见「本机实测」1–5）把它补成可读的量级：$n/p\approx8$ 时经验协方差的 held-out NLL 只比 oracle 差 0.35（7.344 vs 6.999），$n/p=1$ 时直接 `inf`。
- **归一化是 $1/n$ 而不是 $1/(n-1)$**（外部核查，源码 `empirical_covariance`）。本机实测：对角与 `np.var(ddof=0)` 逐位相同，与 `np.cov` 的比是 0.98 = 49/50（$n=50$）。这与 1.2 节 QDA 的 $\frac{1}{n}X_k^TX_k$ 是同一个约定（[[sklearn-lda-qda]] 的版本判据正是这条），引用公式时要一致。
- **`assume_centered` 是「不再减均值」而非「数据已近似零均值」**（L18）。官方 1.7 起把这句话改写成了正面表述：「若 `assume_centered=True`，则训练集与测试集的所有特征都应零均值；否则两者都应由用户中心化，或使用 `assume_centered=False`」。**本机实测的代价**：在不居中的数据上误设 `True`，对角从 1.348 涨到 2.550（多算的是 $\bar x^2$），且 `location_` 被静默置为全零——这比「结果略偏」严重得多，是本节最容易踩的一脚。
- **`assume_centered` 同时决定训练与推理两侧**（L18）：这是本库第一处明确写出「估计器对**测试集**也有预处理契约」的地方。把 `EmpiricalCovariance` 放进 Pipeline 时，中心化只能由用户负责。

### B. 收缩：一个公式、三个 α 来源（L28–L75）

- **MLE 的失败不是「有偏」，而是「特征值估不准」**（L28）：所以**由它求逆得到的精度矩阵不准**，甚至数值上不可逆。这句话把「协方差估计」与「下游用它做什么」接上了：LDA/QDA 只需要 $K$；PCA 只需要 $\Sigma$ 的特征值；Mahalanobis 需要 $K$。三者对 $\Sigma$ 的误差敏感方向不同（本库综合）。
- **收缩是一个凸组合**（L32）：$\Sigma_{\text{shrunk}}=(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$。来源同时给了三种等价读法——压小「最小/最大特征值之比」、给每个特征值加同一偏移、求 $ℓ_2$ 惩罚的 MLE。**本机实测逐位复现该式**（α=0/0.1/0.5/1 的 maxdiff 均为 0.0），且 α=1 时对角等于 $\mathrm{Tr}\hat\Sigma/p$、非对角全为 0（4 特征合成数据：0.88836 = 0.88836）。
- **LW 的 α 最小化的是 $\Sigma$ 的 MSE**（L42）：不是似然、不是分类/回归误差。这是本节最重要的一句口径，本页在关键判断 4 与「分歧」8 里两次用到。
- **各向同性总体的反直觉行为**（L46–L51）：当总体协方差是单位阵的倍数时，样本量增大**不会**让 LW 的 α 趋于 0，反而趋于 1。来源给的理由是「LW 意义下的最优估计本来就是单位阵的倍数」——这在逻辑上自洽，但读者容易把它读成「α 会一直变大」。本机实测（$p=5$、真值 $I$）：$n=1000$ 时 α 均值 0.789（6 个种子跨 0.381–1.000）、$n=3000$ 0.789、$n=10000$ 0.908、$n=30000$ 0.960——**趋势向上但不单调，且种子间极差可达 0.62**。作为对照，各向异性（diag 1..5）时同样的 $n$ 给 α = 0.0113 / 0.0011。**结论：这条 note 说的是「目标」而不是「路径」**（本库判断）。
- **OAS 的前提是高斯**（L61）：在这一假设下 α 的取法给出比 LW 更小的 MSE。剪藏只说结论，不给公式（公式在丢失的参考文献 Chen et al. 2010 里）。
- **三者是同一个算子**（本机实测）：`np.allclose(LedoitWolf().covariance_, ShrunkCovariance(shrinkage=lw.shrinkage_).covariance_)` 为 `True`，OAS 同样为 `True`。因此选型问题不是「用哪个估计器」，而是「**谁来定 α、定多少**」。这条在官方文档里没有明说，是本节最实用的综合。
- **α 的两个退化点**（本机实测，官方文档未提）：
  - $p=1$ 时 LW 直接返回 `shrinkage_ = 0.0`（源码分支：单特征「结果与收缩无关」）；
  - $n=2$ 时 LW 的 α **恒等于 0**，且这是代数恒等式而非数值噪声（居中后 $x_2=-x_1$，可推出 $\hat\beta/n\equiv\hat\delta$）——即**样本最少的情形下收缩恰好关闭**。$n=3$ 起 α 才回到 0.32–0.62。
- **精度矩阵是伪逆，不会报错**（本机实测 + 源码）：`EmpiricalCovariance` 用 `linalg.pinvh` 计算 `precision_`，docstring 明写「Estimated pseudo inverse matrix」。$n=10,p=20$（秩 9）时 fit 不抛异常，`precision_` 是 $20\times20$ 的伪逆，`score` 返回 `-inf`。这与 L28「有时无法求逆」并不冲突——文档说的是数学事实，实现选择的是**静默伪逆**。

### C. 稀疏精度矩阵：从「估 Σ」到「学一张图」（L79–L105）

- **$K$ 正比于偏相关矩阵**（L79）：$K_{ij}=0$ ⇔ 在给定其余变量时 $x_i\perp x_j$。因此「让 $K$ 稀疏」＝「学条件独立关系」＝*covariance selection*。这是本节与 2.2/2.3/2.5 三节最大的不同：输出不是坐标、标签或成分，而是**变量间的边**。
- **稀疏与收缩不是同一件事**（L81）：小样本（$n$ 与 $p$ 同量级或更小）时稀疏估计器**往往更好**；反过来（$n\gg p$）或数据高度相关时**数值不稳定**；但只有稀疏估计器**能恢复非对角结构**，收缩不能。本机实测把这三句拆开验证（见「本机实测」10–13）：
  - 「稀疏好于收缩」在 $n/p\ge3$ 时成立且幅度很大（$n=60,p=20$：$\alpha=0.4$ 的平均绝对误差 0.0217 vs LW 0.0559），在 $n=p$ 时只是持平（0.0384 vs 0.0357）；
  - 「$n\gg p$ 时不稳定」在 $n=2000$（$n/p=100$）**没有复现**——只要 α 不过小（0.02 以上），没有收敛警告、F1 反而最高（α=0.4 时 F1=1.000）；
  - 「高度相关时不稳」**复现为 `ConvergenceWarning`**：两列几乎共线时 $\alpha=0.001$ 触发 125 条「Objective did not converge」，$\alpha=0.01$ 降到 3 条，$\alpha=0.1$ 为 0 条。**警告本身就是 α 太小的信号**，而不是可以忽略的噪声。
- **目标函数与惩罚范围**（L101–L103）：$\hat K=\arg\min_K \mathrm{tr}SK-\log\det K+\alpha\|K\|_1$；L105 明确 $\|K\|_1$ 是**非对角**元素绝对值之和。外部核查源码 `_objective`：`alpha * (abs(precision).sum() - abs(diag(precision)).sum())`，与文档一致。**这是本库第一处「L1 惩罚作用在矩阵上、但排除对角」的例子**——与 Lasso 的系数惩罚、SparsePCA 的成分惩罚都不同（后者连对角也没有意义）。
- **`GraphicalLassoCV` 的 α 不是「恢复边」的最优值**（L97 最后一条 + 本机实测）：CV 选出的 α 系统性偏小，导致边数远多于真值。**（种子 2）**$n=60$ 时 CV 给 α=0.274、34 条边（真值 12 条，F1 0.522）；$n=200$ 时 α=0.126、52 条边（F1 0.375）；**种子 1** 在同设定 $n=60$ 上给 α=0.188、82 条边（F1 0.255）——**CV 的 α 自身对种子敏感**，但「偏小」的方向一致。但**同一组的协方差误差却是最小的**（0.0134 vs 手设 α=0.4 的 0.0186）——**似然目标与结构恢复目标在这一节里明确冲突**，这是本页最重要的综合之一。
- **`alpha=0` 是允许的，且等价于经验协方差**（本机实测）：`GraphicalLasso(alpha=0.0)` 与 `EmpiricalCovariance` 的 `covariance_` 逐位相同（maxdiff 0.0）。因此「稀疏」这个旋钮可以从「关」连续调到「全零」，不需要换类——与 `ShrunkCovariance(shrinkage=0)` 对称。

### D. 稳健估计：MCD 的三步与两个「h」（L118–L128）

- **先说结论再说方法**（L118）：经验与收缩估计器**对离群点非常敏感**，真实数据应改用稳健估计器；或者反过来，用稳健估计器**做离群点检测**（按 Mahalanobis 距离丢/降权）。这是本库第一次出现「同一个估计器既是建模工具又是诊断工具」。
- **MCD 的三步**（L124）：① 找 $h$ 个「好」观测、算它们的经验协方差；② 重标定以补偿这次选择（*consistency step*）；③ 按 Mahalanobis 距离给样本加权，得到重加权估计（*reweighting step*）。L126 补上 FastMCD，并说明它**同时**给出稳健的位置估计。
- **`raw_location_` / `raw_covariance_` 是校正前的量**（L128）；校正后的是 `location_` / `covariance_`。本机实测：两者在本例中差 0.211（L2），`det` 从 0.0762 升到 0.7365（约 9.7 倍）——**校正不是微调，它改变矩阵的量级**，做跨估计器比较时必须说明用的是哪一组。
- **`support_` 不是 h**（本机实测 + 源码，官方文档未提）：默认 `support_fraction=None` 时 $h=\lceil (n+p+1)/2\rceil=102$（$n=200,p=3$），但 `support_.sum()` 是 **178**——`support_` 在 `reweight_covariance` 里被**重加权掩码**覆盖了（阈值 `dist_ < chi2(p).isf(0.025)`），h 只保留在 `raw_support_` 里。想检查「算法选了多少个点」必须读 `raw_support_`。这是本节最容易误读的一个属性。

## 版本判定

**1.7.0 – 1.9.1，工作版本 1.9.1，证据强。**

1. **1.6→1.7 的四道措辞判据全部命中**（93 条句子探针在 1.6/1.7/1.8/1.9 四版页的命中数是 **50/54/54/54**，差的 4 条正好是 1.6 的旧措辞）：
   - L18「if `assume_centered=True`, then all features in the train and test sets should have a mean of zero…」：1.6 是「if `assume_centered=False`, then the test set is supposed to have the same mean vector as the training set…」；
   - L32「a simple convex transformation」：1.6 是「a simple **a** convex transformation」；
   - L49「when the number of **samples** grows」：1.6 是「number of **sample** grows」；
   - L51「is **a** multiple of the identity」：1.6 是「is multiple of the identity」。
2. **1.7→1.8、1.8→1.9 的正文逐词 diff 为**空（按句子切分比对，无 replace/insert/delete）。
3. **抓取日 `/stable/modules/covariance.html` 与 `/1.9/modules/covariance.html` 字节相同**（65,494 B），页标题 `scikit-learn 1.9.1 documentation`，页脚 1.9.1。
4. **反向核对**：官方 1.9.1 页的 82 条长句里，29 条「未在剪藏中找到」经逐条检查全部是 Markdown 渲染差异（标题粘连、链接语法、公式 `\(...\)` vs `$...$`、引用标记 [1]–[4]、参考文献段），**没有一条是正文缺失**。

## 转换损失

**这是本库目前「正文最完整、损失最集中在文献层」的一份**：官方页正文 1,848 词，剪藏正文 1,804 词，差额几乎全部来自 4 条参考文献条目（约 60 词）。

| 丢失项 | 官方 1.9.1 页原文 | 对应剪藏位置 |
| --- | --- | --- |
| 参考文献 **[1]** | O. Ledoit and M. Wolf, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices”, *Journal of Multivariate Analysis*, Volume 88, Issue 2, February 2004, pages 365-411. | L57 的 `References` 标题下为空 |
| 参考文献 **[2]** | “Shrinkage algorithms for MMSE covariance estimation.”, Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O., *IEEE Transactions on Signal Processing*, 58(10), 5016-5029, 2010. | L69 的 `References` 标题下为空 |
| 参考文献 **[3]** / **[4]** | [3] P. J. Rousseeuw. Least median of squares regression. *J. Am Stat Ass*, 79:871, 1984. [4] *A Fast Algorithm for the Minimum Covariance Determinant Estimator*, 1999, American Statistical Association and the American Society for Quality, TECHNOMETRICS. | L130 的 `References` 标题下为空 |
| 页内引用标记 **[1]** | 「In their 2004 paper **[1]**, O. Ledoit and M. Wolf propose…」 | L42 只剩「In their 2004 paper,」 |
| 页内引用标记 **[2]** | 「Chen et al. **[2]** derived a formula…」 | L61 只剩「Chen et al. derived」 |
| 页内引用标记 **[3]** | 「…introduced by P.J. Rousseeuw in **[3]**.」 | L124 只剩「…introduced by P.J. Rousseeuw **in.**」——**句子被截断** |
| 页内引用标记 **[4]** | 「Rousseeuw and Van Driessen **[4]** developed the FastMCD algorithm…」 | L126 只剩「Rousseeuw and Van Driessen developed」 |
| 末尾第 2 张表 | 2 行（表头 + 两张图） | L181–L182 只到表头分隔行，**没有表体** |

**脚注伪影（不是丢失，是重复）**：`[^1]`–`[^5]`（L143–L182）把 Ledoit-Wolf、OAS、MCD 三段正文**整段重复了一遍**，共 **631 词**（占 `wc -w` 的 25%）。因此：

- `wc -w` 的 2,488 词**不能**当作官方口径，有效正文是 1,804 词（+ 631 词重复）；
- 直接按行号引用时，L44/L46–L51/L55 与 L145/L147–L152/L156 是同一段文字的两次出现，**引用前一处即可**；
- 末尾第 2 次出现的表（L179–L182）被截断，同样应引用 L139–L141 的第一份。

**公式渲染噪声**（不影响语义，但抄写会错）：

- L32：$\Sigma_{s h r u n k}$（字母间多空格）、`T r \hat{\Sigma}`、`I d` —— 应为 $\Sigma_{\mathrm{shrunk}}$、$\mathrm{Tr}\hat\Sigma$、$\mathrm{Id}$；
- L101–L102：`argmin_{K}` 与 `\left(\right.` / `\left.\right)` 残片；`tr S K` 缺 `\mathrm`；
- L32 的 $p$ **在正文里从未定义**（按上下文是特征数）；官方页同样没定义，属两边共有的缺口。

## 外部核查（不在剪藏内）

**1.9.1 签名（官方 API 页；1.6.1 实测签名一致）**：

| 对象 | 签名 |
| --- | --- |
| `EmpiricalCovariance` | `(*, store_precision=True, assume_centered=False)` |
| `ShrunkCovariance` | `(*, store_precision=True, assume_centered=False, shrinkage=0.1)` |
| `LedoitWolf` | `(*, store_precision=True, assume_centered=False, block_size=1000)` |
| `OAS` | `(*, store_precision=True, assume_centered=False)` |
| `GraphicalLasso` | `(alpha=0.01, *, mode='cd', covariance=None, tol=0.0001, enet_tol=0.0001, max_iter=100, verbose=False, eps=2.22e-16, assume_centered=False)` |
| `GraphicalLassoCV` | `(*, alphas=4, n_refinements=4, cv=None, tol=0.0001, enet_tol=0.0001, max_iter=100, mode='cd', n_jobs=None, verbose=False, eps=2.22e-16, assume_centered=False)` |
| `MinCovDet` | `(*, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)` |
| `empirical_covariance` | `(X, *, assume_centered=False)` |
| `shrunk_covariance` | `(emp_cov, shrinkage=0.1)`（注意：**输入是已经算好的矩阵**，不是数据） |
| `ledoit_wolf` | `(X, *, assume_centered=False, block_size=1000)` |
| `oas` | `(X, *, assume_centered=False)` |

**其余外部核查（源码 1.6.1，官方 1.9.1 页文字一致）**：

- **`ShrunkCovariance.shrinkage=0.1` 是任意默认值**，不是「最优」；官方示例用 CV 与 LW/OAS 对照，说明 0.1 只是一个起点。
- **`GraphicalLassoCV.alphas=4` 是「路径上的个数」，不是 α 的候选列表**：源码先取 $\alpha_{\max}=$ 经验协方差非对角元素的最大绝对值，再在 $[10^{-2}\alpha_{\max},\alpha_{\max}]$ 上取 4 个对数等距点，然后做 4 轮 refine（`n_refinements=4`）。`cv=None` → 5 折。
- **`MinCovDet` 没有 `support_fraction_` 属性**（1.9.1 签名与 1.6.1 一致都没有）；要复算 $h$ 只能自己写 $\lceil (n+p+1)/2\rceil$。
- **`MinCovDet` 的 docstring 把默认 h 写错**：写作「`(n_samples + n_features + 1) / 2 * n_samples`」，而源码是 `int(np.ceil(0.5 * (n_samples + n_features + 1)))`。前者按字面算是 $h\approx n^2$，显然不成立——**引用默认值时以源码为准**。
- **校正与重加权的公式**（源码）：校正因子 $=\mathrm{median}(\text{dist}_i)/\chi^2_p(0.5)$；重加权掩码 $=\text{dist}_i<\chi^2_p(0.975)$。
- **`GraphicalLasso` 的 `covariance=` 参数**用于「已知协方差、只想求稀疏逆」的场景，剪藏完全没提。
- **`covariance` 模块还有 `EllipticEnvelope`、`log_likelihood`、`fast_mcd`、`ledoit_wolf_shrinkage`**；前者的用法已在 2.7 节入库，见 [[sklearn-outlier-detection]] 与 [[anomaly-detection]]；本节本身不展开。

## 图表核查

5 张图于 2026-11-12 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-covariance/`，逐张目视并记录尺寸与 SHA-256 前 16 位。

| 图 | 文件 | 尺寸 | sha256 前 16 位 | 目视读数 |
| --- | --- | --- | --- | --- |
| 1 | `sphx_glr_plot_covariance_estimation_001.png` | 640×480 | `482aebe3a8a0d6e2` | 标题 “Regularized covariance: likelihood and shrinkage coefficient”；横轴 shrinkage（对数，$10^{-2}$–$10^{0}$），纵轴 “Error: negative log-likelihood on test data”（对数刻度）。蓝实线从 ≈6.5×10² 单调降到 ≈1.45×10²（α≈0.7–0.8 处最低），然后到 α=1 微微回升；红虚线 “Real covariance likelihood” 在 ≈1.05×10²。三条竖线集中在 0.70–0.80：深紫（OAS）最左、洋红（Ledoit-Wolf）居中、青色（cross-validation best）最右——**三者几乎重合，CV 的最小值略偏大** |
| 2 | `sphx_glr_plot_lw_vs_oas_001.png` | 640×480 | `cf362d9cbe0de400` | 上下两栏，横轴 $n_{\text{samples}}$ 5–30。上栏 “Squared error”：LW（深蓝 + 误差棒）从 ≈55 单调降到 ≈2，OAS（橙）在整个区间几乎水平（≈2–3）；下栏 “Shrinkage”：LW 从 ≈0.83 升到 ≈0.97，OAS 稳定在 0.985–0.995。**这张图是 note L46–L51 的实证：OAS 的 α 一开始就接近 1，LW 需要更多样本才追上** |
| 3 | `sphx_glr_plot_sparse_cov_001.png` | 1000×600 | `59142724e42555e6` | 2×4 面板。上排协方差：Empirical（满屏噪声色块）、Ledoit-Wolf（**几乎只剩对角，非对角被洗白**）、GraphicalLassoCV（少量粉色非对角块，位置与真值接近）、True。下排精度：Empirical（密集噪点）、Ledoit-Wolf（几乎纯对角）、GraphicalLasso（稀疏的蓝/白非对角点）、True。**图 3 直接支持 L81 的最后一句：收缩恢复不了非对角结构，稀疏可以** |
| 4 | `sphx_glr_plot_robust_vs_empirical_covariance_001.png` | 640×480 | `90e08b4fd512a63f` | 上栏位置误差 $\lvert\mu-\hat\mu\rvert^2$：全数据均值（绿）随污染率从 ≈0 升到 ≈0.75（污染 32.5%），稳健位置（洋红）与纯数据均值（黑）几乎持平在 0.05 以下。下栏 RMSE：全数据经验协方差（绿）在污染 ≈6% 时已越过 4.5，稳健协方差（洋红）与纯数据（黑）都 ≈0.02–0.05。**经验协方差在个位数百分比的污染下就崩溃，MCD 不动** |
| 5 | `sphx_glr_plot_mahalanobis_distances_001.png` | 1000×500 | `5e85e45480d15441` | 标题 “Mahalanobis distances of a contaminated data set”；黑点 = inliers、红点 = outliers；蓝虚线 = MLE 距离等高线、橙点线 = MCD 距离等高线。MLE 的椭圆被红点**拉长并向右倾斜**（长轴伸到 $x\approx7.5$），MCD 的椭圆紧凑、贴合黑色团块。后果：在最外圈，红点落在 MLE 椭圆**之内或附近**、却在 MCD 椭圆**之外**——同一批离群点在两种度量下「极端程度」完全不同 |

> 未查看/未描述的图：无（官方页共 5 张，全部入库并目视）。

## 本机实测（scikit-learn **1.6.1**、NumPy 2.0.2、SciPy 1.13.1）

> **版本警告**：文档是 1.9.1，本机最高可安装 1.6.1。以下只证明**机制存在**，数值不应外推到 1.9.1。所有脚本均以固定 `random_state` 复现。

> **合成数据的两个种子**：稀疏实验用同一构造（$p=20$、真值 12 条非对角边）跑了两个种子——**种子 1** 用于第 14 条的 α 细表与 CV 边数（82 条），**种子 2** 用于第 15–16 条的跨 $n$ 比较（$n=60/200/2000$）。两组的绝对值不同（例如 $n=60$、α=0.4 的平均绝对误差 0.0217 vs 0.0384），但方向一致；跨种子差异本身也是一条证据（CV 的 α 对种子敏感）。

**A. 归一化与中心化（L16–L18）**

1. `empirical_covariance` 的对角 = `np.var(X, axis=0)`（ddof=0），与 `np.cov`（ddof=1）的比 = 49/50 = 0.98（$n=50$）→ **MLE 的 $1/n$ 约定被确认**。
2. `assume_centered=True` 用在未居中的数据上（均值 1–4）：对角 1.348→**2.550**（第 1 列）、1.212→5.655（第 2 列），`location_` 全零 → 静默出错，不报错。
3. `EmpiricalCovariance(store_precision=False)`：`precision_` 属性存在但**值为 `None`**（property 返回 None，不是 AttributeError）；`get_precision()` 仍能算出来。想检查形状时容易踩。

**B. 收缩算子（L32–L34）**

4. 公式逐位复现：`ShrunkCovariance(shrinkage=s).covariance_` 与 $(1-s)\hat\Sigma+s\frac{\mathrm{Tr}\hat\Sigma}{p}I$ 在 $s\in\{0,0.1,0.5,1\}$ 上 maxdiff 均为 **0.0**；$s=1$ 的图像是 $0.88836\,I$（$\mathrm{Tr}\hat\Sigma/p=0.888361$），非对角 0.0。
5. `LedoitWolf` 与 `OAS` 都**精确等于**某个 `ShrunkCovariance(shrinkage=α̂)`：本例 α̂ = 0.08141 / 0.10390，`allclose` 均为 `True`；两者协方差的最大差 0.0234。→ **「三个估计器」是一个算子 + 三种 α 来源**。
6. `ledoit_wolf(block_size=1000/100/17)` 三种分块给出**逐位相同**的结果（maxdiff 0.0）→ `block_size` 只是内存参数，不改结果。
7. 尺度敏感性：把第 1 列乘 100，LW 的 α̂ 从 0.0814 **降到 0.0303**，OAS 从 0.1039 降到 0.0523 → 收缩**不是**仿射等变的（目标 $\frac{\mathrm{Tr}}{p}I$ 随尺度变）。这与 [[gaussian-discriminant-analysis]] 待解问题 3 直接相关。

**C. 各向同性 note（L46–L51）**

8. 真值 $\Sigma=I$、$p=5$、每格 5 个种子：$n$ = 50 / 100 / 300 / 1000 / 3000 / 10000 / 30000 → LW 的 α 均值 = 0.940 / 0.999 / 0.928 / 0.789 / 0.789 / 0.908 / 0.960（**非单调，$n=1000$ 与 3000 最低，单种子跨 0.381–1.000**）。**「趋于 1」的方向成立，但不是单调收敛**。
9. 各向异性对照（diag 1..5）：$n=1000$ → α=0.0113，$n=10000$ → 0.0011。**同一个估计器、同一个 $n$，只因总体协方差不是单位阵的倍数，α 就从 0.79 掉到 0.001**。这条对照是 note 的真正含义。

**D. 似然、可行性、退化点**

10. held-out 负对数似然（$p$ 维、20000 个测试样本、8 个种子均值）：

| 设定 | Empirical | LW | OAS | Shrunk(0.1) | Shrunk(0.3) | oracle |
| --- | --- | --- | --- | --- | --- | --- |
| $p=5,n=40$（$n/p=8$） | 7.344 | 7.295 | 7.304 | **7.267** | 7.308 | 6.999 |
| $p=10,n=30$（$n/p=3$） | 16.018 | 14.740 | 14.749 | 14.999 | **14.687** | 13.900 |
| $p=20,n=20$（$n/p=1$） | **`inf`** | 30.608 | **30.608** | 39.776 | 31.958 | 27.801 |

→ 收缩的收益随 $n/p$ 下降而放大；$n=p$ 时不收缩直接不可用；但**数据驱动的 α 并不比一个好手设的 α 更好**（第 1 行 0.1 优于 LW）。
11. $n=10,p=20$（秩 9）：`score` = `-inf`、`precision_` 仍是 $20\times20$ 伪逆、fit **无异常**。
12. $p=1$ → `LedoitWolf().shrinkage_ = 0.0`；$n=1$ → 两条 `UserWarning`（“Only one sample available”）、协方差全零、`score` = `-inf`；$n=2$ → α **恒为 0.0**（6/6 种子），$\hat\beta$ 的数值也确实为 0。
13. 各向同性数据上 α 取到 1 时，`precision_` 的计算会抛 **3 条 scipy `RuntimeWarning`**（`divide by zero` / `overflow` / `invalid value`，位置 `scipy/linalg/_basic.py:1542`，即 `pinvh`），但结果**正确**（$\Sigma K=I$ 的误差 1.1e-16）。这是一个「警告吓人、结果无碍」的 scipy 伪影，1.6.1 可稳定复现。

**E. 稀疏精度矩阵（L79–L105）**

14. **（种子 1）**合成 $p=20$、真值 12 条非对角边、$n=60$：α 0.01 / 0.05 / 0.1 / 0.2 / 0.4 / 0.8 → 非对角非零数 **360 / 268 / 172 / 72 / 12 / 0**，F1 0.065 / 0.086 / 0.130 / 0.286 / **0.833** / 0.000。α 过大时退化成对角（F1 归零）。
15. **（种子 2）**`GraphicalLassoCV(cv=5)` 在 $n=60$ 选 α=0.2743 → **34 条边**（真值 12），F1 0.522；$n=200$ 选 α=0.1258、52 条边（F1 0.375）；$n=2000$ 选 α=0.0361、64 条边（F1 0.316）。**「CV 选的 α 会选太多边」被量化：多 2.8–6.8 倍**（种子 1 的同设定 $n=60$ 更差：α=0.1879 → **82 条边**，= 6.8×）。
16. 但 CV 的**协方差误差最小**：$n=200$ 时 CV 0.0134、α=0.4 为 0.0186、LW 0.0470、经验 0.0625。→ 「边数正确」与「$\Sigma$ 误差小」不是同一个目标。
17. 标准化（L94）：$n=60$ 时 α=0.2 的 F1 从 0.286（协方差）升到 0.308（相关阵）；把第 1 列乘 1000 后，**原始尺度 F1 0.207 → 标准化 0.308**（边数 104 → 66）。→ 「先从相关阵估计」在有量纲差异时是**决定性的**，在尺度相近时只是略好。
18. `GraphicalLasso(alpha=0.0)` 与 `EmpiricalCovariance` 的 `covariance_` 逐位相同（maxdiff 0.0）→ α=0 合法且等于 MLE；α<0 触发 `InvalidParameterError`（报错文字写 `[0.0, inf]`，但 1.6.1 实际接受 0.0）。
19. $n=10<p=20$：α=0.05 出现 “did not converge” 且 238 条非对角非零；α 0.1/0.3/0.6 → 190/88/14 条。**小样本并不自动给出稀疏解，α 仍需手调**。

**F. MCD（L118–L128）**

20. $n=200,p=3$、前 20 个样本整体平移 +6（10% 污染）：`raw_support_.sum()` = **102** = $\lceil(200+3+1)/2\rceil$；`support_.sum()` = **178**（重加权后扩展）；`raw_location_` 与 `location_` 最大差 0.211；`det` 0.0762 → 0.7365。
21. 同一数据：MCD 协方差的 Frobenius 误差（对 $\Sigma=I$）**0.313**，经验协方差 **9.394**；Mahalanobis 均值——MCD：污染 124.34 / 内点 3.07（分离 40×），经验：污染 10.33 / 内点 2.19（分离 4.7×）。→ 图 5 的数值版本。
22. `random_state=None` 两次 fit 在本例上**结果相同**（maxdiff 0.0），在更难的一例（10 个样本 +2.5）上也相同（185 vs 183 个 support 点在本例一致）。→ 「随机」影响搜索路径，但只要全局最优明显，结果可重复；**不能据此断言 MCD 确定**。
23. $n=4<p=10$ 的报错是 `ValueError: kth(=7) out of bounds (4)`——**不是**一句「需要 $n>p$」的校验信息。文档未提这个前提。

## 分歧

1. **4 处引用标记被吞，其中 L124 变成病句**（「introduced by P.J. Rousseeuw in.」）。读剪藏会误以为原文如此；官方是「in [3].」。
2. **4 个 `References` 段落为空块**（L57、L69、L130、L172），但正文明确引用 [1]–[4]。因此「Ledoit-Wolf 2004 / Chen et al. 2010 / Rousseeuw 1984 / Rousseeuw & Van Driessen 1999」四条文献**不能引剪藏行号**，只能引官方 1.9.1 页或本页的「转换损失」表。
3. **「$\alpha$ 的 MSE 最优 = 下游最优」不成立**：L42 只说 α 最小化 $\Sigma$ 的 MSE，本机实测（第 10 条）显示 MSE 最优的 α 在 held-out NLL 上输给手设小 α。文档没有提示这两个目标不同。
4. **L34「bias/variance trade-off」没有给出分解**，也没有说 α 增大时哪一项升、哪一项降；剪藏与官方页都缺。本页只能引用结论，不能引用机制。
5. **note L46–L51 的表述像「α 随 $n$ 单调增大」**，实测（第 8 条）非单调且种子方差极大。文档说的是「optimal estimate 是单位阵的倍数」，这句话成立；但把它读成「路径单调」是过度解读。
6. **L81「$n\gg p$ 时稀疏估计器数值不稳定」在本机未复现**（$n=2000,p=20$，α≥0.02 无警告、F1 最高）。真正复现的不稳定条件是**近共线列 + 极小 α**（125 条收敛警告），文档没有说清是哪一种。
7. **L97「CV 选的 α 会选太多边」与 CV 的似然目标其实一致**：CV 最小化的是 held-out 负对数似然，而似然偏爱更密的 $K$（第 15–16 条）。文档把这条写成「即使条件有利也会选太多」，读起来像缺陷；本页的读法是**目标函数不同**，不是 bug。
8. **`support_` 的语义与 L124 的 h 不对应**：读者按文档找「h 个点」会读到重加权后的 178 个；h 在 `raw_support_`。两处（默认值公式、属性语义）文档都没写清，且 docstring 的默认 h 公式本身写错。
9. **`MinCovDet` 的 `n>p` 前提没写**：$n<p$ 时抛的是内部下标错误，不是友好校验。
10. **`precision_` 是伪逆**：L28 说「有时无法求逆」，读者会以为会报错；实现用 `pinvh` 静默给伪逆（`score` 才暴露为 `-inf`）。文档与实现的「失败模式」不同。

## 局限

- **没有一行代码**：本节 0 个 doctest、0 个示例脚本内联，所有 API 用法都要去 Examples 链接或 API 页。这是本库 sklearn 来源里唯一「纯散文 + 图」的一节。
- **四个 α 的解析式全部缺席**：LW 的 $\pi/\rho/\gamma$、OAS 的闭式解都不在正文，只在丢失的文献里。因此「α 是怎么算出来的」在本节**无法回答**。
- **复杂度/成本完全没有**：`GraphicalLasso` 的迭代成本、MCD 的 FastMCD 抽样次数、`block_size` 的作用都没写。
- **`n` 与 `p` 的相对规模只有定性判据**（L16「large enough」、L81「on the order of or smaller」），没有阈值、没有曲线。
- **收缩与稳健/稀疏的组合未讨论**：能否先 MCD 再 GLasso、能否对相关阵做收缩，文档都不回答。
- **`EllipticEnvelope`（2.7 节）只在 L118 的「或者用来做离群点检测」里隐含**，本节不给用法。

## 待解问题

1. **LW 的 α 估计式**（$\hat\pi$、$\hat\rho$、$\hat\gamma$ 三项）到底是什么？需要读 Ledoit & Wolf 2004（文献条目已丢，需另找）。
2. **OAS 的闭式 α** 与「高斯下 MSE 更小」的证明是什么？需要 Chen et al. 2010。
3. **MCD 的 breakdown point 与 h 的定量关系**？默认 $h=\lceil(n+p+1)/2\rceil$ 对应多大的污染容忍度？
4. **收缩 + 稀疏能否组合**（对 $\Sigma$ 收缩后再对 $K$ 加 L1）？是否有已知的估计器？
5. **是否应该在相关阵而非协方差阵上做收缩？** L94 与图 3 都暗示标准化更好，但本节只对 GraphicalLasso 明说；LW/OAS 是否也该先标准化（LDA 的 `'auto'` 就是这样做的，见 [[gaussian-discriminant-analysis]]）？
6. **`GraphicalLassoCV` 的 refit 语义**：`alpha_` 是在 refine 后的网格上选的，但 `covariance_` 是在哪个 α 上重拟合的？源码有 `refit` 逻辑，本次未读透。
7. **p≫n 时 GLasso 的可用性**：本次只测到 $n=10,p=20$；更高维（$p=1000$）是否仍能给出有意义的稀疏解，未验证。
8. **`MinCovDet` 在「没有明显分离」的数据上随机性有多大**？本次两例都收敛到同一解，需要构造难例（污染点贴近主体、多个局部极小）才能测。
9. **各向同性时 α→1 触发的 `pinvh` 警告**是否在 1.9.1 修复？若未修复，日志里会出现大量假警报。
10. **`EmpiricalCovariance.score` 的常数项**：源码用 $-\frac{1}{2}(\log\det\Sigma+\mathrm{tr}(S\Sigma^{-1}))$ 形式（不含 $p\log 2\pi$？），因此不同估计器的 NLL **绝对量级**不可跨实现比较，只能同实现内比较。本次未逐行核对。

## 行号索引

| 行号 | 内容 |
| --- | --- |
| L10–L12 | 引言；`sklearn.covariance`；i.i.d. 假设 |
| L14–L22 | 2.6.1 经验协方差；`assume_centered` 契约；Examples |
| L24–L38 | 2.6.2 / 2.6.2.1 基本收缩；公式（L32）；bias/variance（L34） |
| L40–L57 | 2.6.2.2 Ledoit-Wolf；MSE 口径（L42）；各向同性 note（L46–L51）；空 References（L57） |
| L59–L75 | 2.6.2.3 OAS；高斯假设（L61）；图 1（L65–L67）；空 References（L69）；图 2（L75） |
| L77–L105 | 2.6.3 稀疏逆协方差；偏相关（L79）；稀疏 vs 收缩（L81）；`alpha`（L83）；图 3（L85–L87）；结构恢复 4 条（L89–L97）；目标函数（L101–L103）；GLasso 来源（L105） |
| L107–L114 | Sparse 的 Examples；**唯一幸存的文献**（Friedman et al. 2008, L113–L114） |
| L116–L128 | 2.6.4 稳健估计；三步 MCD（L124）；FastMCD（L126）；`raw_*` 属性（L128） |
| L130–L141 | 空 References（L130）；Examples；2×2 表（L139–L141） |
| L143–L182 | `[^1]`–`[^5]` 脚注：**重复** Ledoit-Wolf / OAS / MCD 正文 631 词；L182 表被截断 |
