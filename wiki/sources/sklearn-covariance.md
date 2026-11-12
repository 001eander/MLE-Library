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
  - scikit-learn 1.9.1（文档正文与 API 签名）
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

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-covariance]], L28`。凡超出剪藏内容的部分（API 默认值、官方参考文献原文、源码细节）都显式标注「外部核查」或「官方 1.9.1 页」；不包含历史本机运行时实验。

> [!warning] 本文件未被修改
> 本次**没有取得所有者对 `raw/` 回填的授权**，剪藏保持抓取时的原始字节（sha256 `0d619801…`）。丢失的 4 条参考文献与 4 个引用标记只在「转换损失」一节里按官方 1.9.1 页记录，**未写回原文件**。所有行号指向原始剪藏。

## 概要

这一节补上本库一个此前**没有名字的类目**：前十二个 sklearn 文档都在估计「一个模型」（$p(y\mid x)$、簇标签、坐标、成分），2.6 节估计的是**数据本身的二阶结构** $\Sigma$ 与它的逆 $K=\Sigma^{-1}$——它不预测任何东西，但几乎所有模型都在暗地里消费它。四个小节因此是同一个问题的四种回答：

1. **第一次把「估计量」当作研究对象，而不是模型的零件。** 1.2 节的 LDA/QDA 只是**使用**协方差（`covariance_estimator` 指向的就是本节的对象），本节反过来问：$\Sigma$ 本身怎么估、估错会怎样、用什么损失衡量（L16、L28、L42、L81）。评估时应区分协方差误差、结构恢复误差与 held-out 似然，不能共用一个「更好」。
2. **四个估计器可以被压成「一个算子 + 一个 α 的来源 + 一个结构假设」。** 经验协方差是 α=0、收缩是 α∈(0,1] 时向 $\frac{\mathrm{Tr}\hat\Sigma}{p}I$ 的凸组合、Ledoit-Wolf 与 OAS 只是两种**解析地**定 α 的规则（L32、L42、L61）。因此「三个估计器」可理解为一个算子，差别全在 α 的取法。
3. **第一次把「精度矩阵」当成一等输出。** $K$ 正比于偏相关矩阵，$K_{ij}=0$ ⇔ 条件独立（L79），于是「估一个稀疏的 $K$」等价于「从数据里学一张条件独立图」，即 *covariance selection*。这是本库第一次出现「模型参数就是一张图」的对象——与 2.3 节聚类**输出标签**、5.2 节（未入库）输出重要性都不同：这里输出的是**变量之间的边**。
4. **第一次把「估计量的目标函数」与「下游指标」明确分开。** L42 说 LW 的 α 最小化的是「估计的与真实的协方差之间的 MSE」，L61 说 OAS 在高斯假设下 MSE 更小——**都不是**分类准确率或 held-out 似然。选择 α 时必须声明要优化哪一种损失，不能把协方差 MSE 最优直接读成下游最优。
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

- **「样本量足够大」是唯一前提**（L16）：经验协方差是对总体协方差**渐近无偏**的 MLE。剪藏没有给「足够大」的判据，也没有说 $n/p$ 到多少算够；相对规模与奇异风险需在具体数据和目标版本中评估。
- **归一化是 $1/n$ 而不是 $1/(n-1)$**（外部核查，源码 `empirical_covariance`）。这与 1.2 节 QDA 的 $\frac{1}{n}X_k^TX_k$ 是同一个约定（[[sklearn-lda-qda]] 的版本判据正是这条），引用公式时要一致。
- **`assume_centered` 是「不再减均值」而非「数据已近似零均值」**（L18）。官方 1.7 起把这句话改写成了正面表述：「若 `assume_centered=True`，则训练集与测试集的所有特征都应零均值；否则两者都应由用户中心化，或使用 `assume_centered=False`」。误设后不会自动纠正均值，且 `location_` 会按零均值假设记录。
- **`assume_centered` 同时决定训练与推理两侧**（L18）：这是本库第一处明确写出「估计器对**测试集**也有预处理契约」的地方。把 `EmpiricalCovariance` 放进 Pipeline 时，中心化只能由用户负责。

### B. 收缩：一个公式、三个 α 来源（L28–L75）

- **MLE 的失败不是「有偏」，而是「特征值估不准」**（L28）：所以**由它求逆得到的精度矩阵不准**，甚至数值上不可逆。这句话把「协方差估计」与「下游用它做什么」接上了：LDA/QDA 只需要 $K$；PCA 只需要 $\Sigma$ 的特征值；Mahalanobis 需要 $K$。三者对 $\Sigma$ 的误差敏感方向不同（本库综合）。
- **收缩是一个凸组合**（L32）：$\Sigma_{\text{shrunk}}=(1-\alpha)\hat\Sigma+\alpha\frac{\mathrm{Tr}\hat\Sigma}{p}I$。来源同时给了三种等价读法——压小「最小/最大特征值之比」、给每个特征值加同一偏移、求 $ℓ_2$ 惩罚的 MLE。α=1 时对角等于 $\mathrm{Tr}\hat\Sigma/p$、非对角为 0。
- **LW 的 α 最小化的是 $\Sigma$ 的 MSE**（L42）：不是似然、不是分类/回归误差。这是本节最重要的一句口径，本页在关键判断 4 与「分歧」8 里两次用到。
- **各向同性总体的反直觉行为**（L46–L51）：当总体协方差是单位阵的倍数时，样本量增大**不会**让 LW 的 α 趋于 0，反而趋于 1。来源给的理由是「LW 意义下的最优估计本来就是单位阵的倍数」——这在逻辑上自洽，但读者容易把它读成「α 会一直变大」。**这条 note 说的是「目标」而不是「路径」**（本库判断）。
- **OAS 的前提是高斯**（L61）：在这一假设下 α 的取法给出比 LW 更小的 MSE。剪藏只说结论，不给公式（公式在丢失的参考文献 Chen et al. 2010 里）。
- **三者是同一个算子**（本库综合）：经验协方差、LW 与 OAS 可以统一写成收缩形式，只是 α 的来源不同。因此选型问题不是「用哪个估计器」，而是「**谁来定 α、定多少**」。这条在官方文档里没有明说，是本节最实用的综合。
- **精度矩阵是伪逆**（源码与 docstring 外部核查）：`EmpiricalCovariance` 用 `linalg.pinvh` 计算 `precision_`。这与 L28「有时无法求逆」并不冲突——文档说的是数学事实，实现选择的是**伪逆**；奇异时的分数与下游行为需另行检查。

### C. 稀疏精度矩阵：从「估 Σ」到「学一张图」（L79–L105）

- **$K$ 正比于偏相关矩阵**（L79）：$K_{ij}=0$ ⇔ 在给定其余变量时 $x_i\perp x_j$。因此「让 $K$ 稀疏」＝「学条件独立关系」＝*covariance selection*。这是本节与 2.2/2.3/2.5 三节最大的不同：输出不是坐标、标签或成分，而是**变量间的边**。
- **稀疏与收缩不是同一件事**（L81）：小样本（$n$ 与 $p$ 同量级或更小）时稀疏估计器**往往更好**；反过来（$n\gg p$）或数据高度相关时**数值不稳定**；但只有稀疏估计器**能恢复非对角结构**，收缩不能。收敛警告应作为诊断信号，具体稳定条件需在目标环境验证。
- **目标函数与惩罚范围**（L101–L103）：$\hat K=\arg\min_K \mathrm{tr}SK-\log\det K+\alpha\|K\|_1$；L105 明确 $\|K\|_1$ 是**非对角**元素绝对值之和。外部核查源码 `_objective`：`alpha * (abs(precision).sum() - abs(diag(precision)).sum())`，与文档一致。**这是本库第一处「L1 惩罚作用在矩阵上、但排除对角」的例子**——与 Lasso 的系数惩罚、SparsePCA 的成分惩罚都不同（后者连对角也没有意义）。
- **`GraphicalLassoCV` 的 α 不是「恢复边」的最优值**（L97 最后一条）：来源明确提醒 CV 可能选出过多边。CV 优化的是 held-out 似然而不是支持集 F1，因此**似然目标与结构恢复目标可能冲突**，这是本页最重要的综合之一。

### D. 稳健估计：MCD 的三步与两个「h」（L118–L128）

- **先说结论再说方法**（L118）：经验与收缩估计器**对离群点非常敏感**，真实数据应改用稳健估计器；或者反过来，用稳健估计器**做离群点检测**（按 Mahalanobis 距离丢/降权）。这是本库第一次出现「同一个估计器既是建模工具又是诊断工具」。
- **MCD 的三步**（L124）：① 找 $h$ 个「好」观测、算它们的经验协方差；② 重标定以补偿这次选择（*consistency step*）；③ 按 Mahalanobis 距离给样本加权，得到重加权估计（*reweighting step*）。L126 补上 FastMCD，并说明它**同时**给出稳健的位置估计。
- **`raw_location_` / `raw_covariance_` 是校正前的量**（L128）；校正后的是 `location_` / `covariance_`。做跨估计器比较时必须说明用的是哪一组。
- **`support_` 不是 h**（源码外部核查）：`support_` 在 `reweight_covariance` 里被**重加权掩码**覆盖，h 保留在 `raw_support_` 里。想检查「算法最初选了多少个点」必须读 `raw_support_`，不能只看 `support_`。

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

**1.9.1 签名（官方 API 页）**：

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

**其余外部核查（源码与官方 1.9.1 页文字）**：

- **`ShrunkCovariance.shrinkage=0.1` 是任意默认值**，不是「最优」；官方示例用 CV 与 LW/OAS 对照，说明 0.1 只是一个起点。
- **`GraphicalLassoCV.alphas=4` 是「路径上的个数」，不是 α 的候选列表**：源码先取 $\alpha_{\max}=$ 经验协方差非对角元素的最大绝对值，再在 $[10^{-2}\alpha_{\max},\alpha_{\max}]$ 上取 4 个对数等距点，然后做 4 轮 refine（`n_refinements=4`）。`cv=None` → 5 折。
- **`MinCovDet` 没有 `support_fraction_` 属性**（1.9.1 签名没有）；要复算 $h$ 只能自己写 $\lceil (n+p+1)/2\rceil$。
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

## 验证边界

本页不包含历史本机运行时实验或性能读数。运行时行为只在 `uv` 创建的目标版本环境中重新验证后收录。

## 分歧

1. **4 处引用标记被吞，其中 L124 变成病句**（「introduced by P.J. Rousseeuw in.」）。读剪藏会误以为原文如此；官方是「in [3].」。
2. **4 个 `References` 段落为空块**（L57、L69、L130、L172），但正文明确引用 [1]–[4]。因此「Ledoit-Wolf 2004 / Chen et al. 2010 / Rousseeuw 1984 / Rousseeuw & Van Driessen 1999」四条文献**不能引剪藏行号**，只能引官方 1.9.1 页或本页的「转换损失」表。
3. **「$\alpha$ 的 MSE 最优 = 下游最优」不成立**：L42 只说 α 最小化 $\Sigma$ 的 MSE，而 held-out 似然、结构恢复与下游模型指标是不同目标。文档没有提示这些目标可能给出不同选择。
4. **L34「bias/variance trade-off」没有给出分解**，也没有说 α 增大时哪一项升、哪一项降；剪藏与官方页都缺。本页只能引用结论，不能引用机制。
5. **note L46–L51 的表述像「α 随 $n$ 单调增大」**：文档说的是「optimal estimate 是单位阵的倍数」，这句话成立；但把它读成「路径单调」是过度解读。
6. **L81「$n\gg p$ 时稀疏估计器数值不稳定」的条件不充分**：文档没有区分样本规模、共线程度与 α 大小，具体稳定条件需在目标环境验证。
7. **L97「CV 选的 α 会选太多边」与 CV 的似然目标其实一致**：CV 最小化的是 held-out 负对数似然而不是支持集 F1。文档把这条写成「即使条件有利也会选太多」，读起来像缺陷；本页的读法是**目标函数不同**，不是 bug。
8. **`support_` 的语义与 L124 的 h 不对应**：读者按文档找「h 个点」会读到重加权后的 178 个；h 在 `raw_support_`。两处（默认值公式、属性语义）文档都没写清，且 docstring 的默认 h 公式本身写错。
9. **`MinCovDet` 的 `n>p` 前提没写**：具体报错与可用性边界需在目标版本环境中核对。
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
7. **p≫n 时 GLasso 的可用性**：高维下是否仍能给出有意义的稀疏解，需在目标版本环境中验证。
8. **`MinCovDet` 在「没有明显分离」的数据上随机性有多大**？需要构造污染点贴近主体、多个局部极小的难例才能测。
9. **各向同性时 α→1 的 `pinvh` 行为**是否会产生警告？需在目标版本源码与环境中核对。
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
