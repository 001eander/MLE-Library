---
title: "scikit-learn 用户指南 1.4「Support Vector Machines」：最大间隔、核函数与 libsvm/liblinear 双实现（stable 文档剪藏，含所有者授权的 raw 回填）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - svm
  - kernel-method
  - reference-doc
source_path: raw/articles/sklearn/1.4. Support Vector Machines.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.4「Support Vector Machines」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Support Vector Machines（scikit-learn 用户指南第 1.4 节） |
| 作者 | scikit-learn developers（剪藏 `author` 为空；官方页脚署名为 scikit-learn developers，BSD License——**本库核查所得**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；9 段 doctest、4 张外链插图、8 个行间公式块 |
| 原文 | <https://scikit-learn.org/stable/modules/svm.html>（剪藏 frontmatter `source`，未钉版本） |
| 本地文件 | `raw/articles/sklearn/1.4. Support Vector Machines.md`（**回填后** 460 行、5,582 词、51,493 字节；**抓取时** 397 行、4,694 词） |
| SHA-256（回填后） | `af5ba9e823539c7e448613f965669965d5231b79ef6eb69806af3c4d73d38cac` |
| SHA-256（抓取时原始字节） | `021bd03bdeeac7bd224b21847c795a0faa954bec3d1dd135c76c92eac92e5ad0` |
| 完整性 | **抓取时缺 6 个内容折叠块（898 词 ≈ 官方正文 19.0%）、10 处页内锚点文字、8 条参考文献与 6 个引用标记**，经所有者授权已回填；仅剩公式排版噪声与重复脚注块（详见「转换损失、回填与外部核查」） |
| 剪藏时间 | frontmatter `created: 2026-11-11`；`published` 为空，故本页不写发布日期 |
| 文档版本 | **1.7.x – 1.9.1（该节无改动，已验证）**；与官方 1.9.1 逐句一致，见「版本判定」 |
| 插图 | 4 张，已入库 `raw/assets/sklearn-svm/`（全部目视核查，读数见表） |
| 代码 | 9 段 doctest：`SVC` 二类（L36–L50）、支持向量属性（L54–L65）、多类 ovo/ovr（L77–L101）、`SVR`（L169–L178）、`kernel=` 字符串（L229–L236）、Python 函数核（L268–L275）、Gram 矩阵（L281–L297） |

入库说明：本页依据本地剪藏撰写，引用写作 `[[sklearn-svm]], L42`。凡超出剪藏内容的部分（API 默认值、1.9 的弃用）都显式标注「外部核查」或「本库」。

> [!warning] 本文件被修改过（本库第三例）
> `raw/` 的规则是不可变，但本次由所有者明确授权回填。抓取时的原始字节 sha256 为 `021bd03b…`（397 行）；**当前文件不再是抓取时的原始字节**（`af5ba9e8…`，460 行）。回填内容全部取自官方 1.9.1 同页，逐句可对；唯一的结构性改动是把两行折叠块标题残迹 `Using Python functions as kernels [#](…)` / `Using the Gram matrix [#](…)` 改写为 `####` 标题，并删除了文末重复的空 `References` 标题（0 行内容）。**本页及所有引用本来源的页面，行号一律按回填后的新文件计。**

## 概要

这是用户指南里「一个模型族、两套实现、三种任务」的典型一节：几何直觉（最大间隔）→ 三条使用提示（缩放、缓存、C）→ 两种参数化（C 与 ν）→ 原问题与对偶的完整推导。

对本库的意义有四层：

1. **补上「间隔」这一支**。此前入库的线性模型（[[sklearn-linear-models]]）、判别分析（[[sklearn-lda-qda]]）、朴素贝叶斯（[[sklearn-naive-bayes]]）都是**从损失或分布出发**推决策面；SVM 是唯一从**几何间隔 + 约束优化**出发的，因此它的正则化读法（`C`）与前四页的 `alpha`、`shrinkage`、平滑都不同构（见 [[regularization-and-sparsity]] 新节）。
2. **第一次给出「规模判据」而非「准确率判据」**：同一问题的两个实现（libsvm 的 `SVC` 与 liblinear 的 `LinearSVC`）复杂度差一个量级（L192、L194），这是本库第一条把**算力边界**写成选型依据的论述（见 [[svm-selection-and-tuning]]）。
3. **「分数 ≠ 概率」的第二个来源，且这次给了修法**。[[sklearn-naive-bayes]] 只说 NB 的概率不可信；本节给出 Platt 校准这一条具体路径（L128）**同时**列出它与分数不一致的两种可观测情形（L135–L136）。讽刺的是，这条路径本身在 1.9 已被弃用（见「分歧」）。
4. **多类的**内部/接口分离**在 sklearn 里最绕的一处**：`SVC` 内部恒用 ovo，接口默认暴露 ovr（L75、L92、L140），而 `dual_coef_` 的排布被官方自己形容为 "somewhat hard to grasp"（L113）。

## 文档结构

| 章节 | 解决的问题 | 行号（回填后） |
| --- | --- | --- |
| 引言 | 定义、四条优点、两条缺点、dense/sparse 输入要求 | L10–L24 |
| 1.4.1 Classification | 三个分类器；与 `LinearSVC` 的差异（`squared_hinge`、正则化截距）；doctest；支持向量属性 | L26–L71 |
| 1.4.1.1 Multi-class | ovo 内部、ovr 接口、`decision_function_shape` 切换、`LinearSVC` 的 ovr | L73–L103 |
| **Details on multi-class strategies**（回填） | `crammer_singer`；`coef_`/`intercept_`/`dual_coef_` 三个属性的形状与排布 | L105–L120 |
| 1.4.1.2 Scores and probabilities | `decision_function`；Platt 校准；两条不一致情形；`break_ties` | L126–L140 |
| 1.4.1.3 Unbalanced problems | `class_weight`（仅 SVC）与 `sample_weight`（七个估计器）都是改 `C` | L142–L157 |
| 1.4.2 Regression | ε-SVR；三个实现；`LinearSVR` 同样正则化截距 | L159–L182 |
| 1.4.3 Density estimation | `OneClassSVM`，正文只留一句 + 指针 | L184–L188 |
| 1.4.4 Complexity | QP 求解器 $O(n_f n_s^2)$–$O(n_f n_s^3)$；liblinear 近线性 | L190–L194 |
| 1.4.5 Tips on Practical Use | 数据拷贝、kernel cache、`C`、缩放、`shrinking`、`nu`、不平衡、随机性、L1 | L196–L216 |
| 1.4.6 Kernel functions | 四个内置核的公式；`kernel=` 参数 | L218–L238 |
| 1.4.6.1 RBF parameters | `C` 与 `gamma` 的语义；指数间隔网格搜索 | L240–L249 |
| 1.4.6.2 Custom Kernels | 自定义核的两个副作用；Python 函数核与 Gram 矩阵（均回填） | L251–L301 |
| 1.4.7 Mathematical formulation | 间隔直觉；`SVC` 原问题/对偶/决策函数；`C = 1/α` 的对照 | L303–L344 |
| — LinearSVC / NuSVC（回填） | hinge loss 原问题（为何不能核技巧）；ν 的重参数化语义 | L346–L358 |
| 1.4.7.2 SVR | ε-SVR 原问题/对偶/预测式；`dual_coef_` 存的是差值 | L360–L384 |
| — LinearSVR（回填） | ε-insensitive loss 原问题 | L386–L392 |
| 1.4.8 Implementation details | libsvm + liblinear 的 C/Cython 封装 | L394–L396 |
| References（回填） | `[9]`–`[16]` | L398–L414 |

## 关键主张

### 1. 四条优点、两条缺点，以及它们各自的可检验含义（来源论述）

优点（L14–L17）：高维有效；维度数大于样本数时仍有效；决策函数只用**训练集的一个子集**（支持向量），因此内存高效；可换核函数。

缺点（L21–L22）：特征数远大于样本数时，**选核与正则强度的过拟合风险**成为关键；**不直接给概率估计**，"these are calculated using an expensive five-fold cross-validation"。

**本库综合**：这两条缺点的实际后果完全不同。第一条是「调参空间随维度放大」——它解释了为什么 [[svm-selection-and-tuning]] 把缩放与 C/γ 网格列为必做项；第二条是**成本**问题而非精度问题，因为校准用的是内部五折（L128、L131），开销与 `probability=False` 不在一个量级。

输入要求（L24）：dense 与 sparse 都支持，但**预测时的稀疏性必须与拟合时一致**（"to use an SVM to make predictions for sparse data, it must have been fit on such data"）；最优性能用 C-order `float64` 或 `csr_matrix`。**本库综合**：这是一条容易被 pipeline 破坏的契约——上游变换器若在训练时给稠密、在推理时给稀疏，不会报错但会走慢路径。

### 2. 决策函数只依赖支持向量：稀疏性的几何来源（来源论述）

"SVMs decision function … depends on some subset of the training data, called the support vectors"（L52），属性 `support_vectors_` / `support_` / `n_support_` 分别给向量本身、训练集索引与每类个数（L52–L65）。

来源在数学节把机制说清楚了：对偶解给出决策函数

$$\sum_{i \in SV} y_i \alpha_i K(x_i, x) + b$$

而"我们只需要对支持向量求和（即落在间隔内的样本），因为其余样本的对偶系数 $\alpha_i$ 为零"（L339）。三个相关属性：`dual_coef_` 存 $y_i\alpha_i$、`support_vectors_` 存支持向量、`intercept_` 存 $b$（L341）。

**本库综合（重要边界）**：这里的稀疏是**样本维的稀疏**，不是**特征维的稀疏**。`SVC` 的判决向量在特征空间里通常稠密；只有 `LinearSVC(penalty='l1', dual=False)` 才产生特征维稀疏（L216）。两者在工程上的用途完全不同：前者决定预测成本随训练集规模的增速，后者决定"能不能读出一组特征"。这条区分在 [[support-vector-machines]] 与 [[regularization-and-sparsity]] 里各写了一遍。

图片核查见下表图 1：图上的三个圈正是"落在间隔边界上"的样本。

### 3. 两套实现：libsvm 与 liblinear 不是替代关系（来源论述）

`SVC` / `NuSVC` 走 libsvm，`LinearSVC` / `LinearSVR` 走 liblinear（L396）。两者的差异被来源写成了三条具体后果：

| 差异 | 内容 | 位置 |
| --- | --- | --- |
| 损失 | `LinearSVC` 用 `squared_hinge` | L32 |
| 截距 | liblinear **会正则化截距**；可用 `intercept_scaling` 削弱，"which allows the intercept term to have a different regularization behavior" | L32、L165 |
| 属性 | `LinearSVC` 缺 `support_` 一类属性 | L32 |

来源为此专门警告："The classification results and score can therefore differ from the other two classifiers"（L32）。**但注意同一句话在回归节被照抄**（L165 说的是 `LinearSVR`，却写 "differ from the other two classifiers"）——这是文档的措辞漏改，不影响语义。

**本库综合**：`squared_hinge` 而非 `hinge`、截距进正则，这两条叠加意味着 **`LinearSVC(C=c)` 与 `SVC(kernel='linear', C=c)` 不是同一个模型的两个实现**，不能拿 CV 分数直接互换。这是本页最重要的一条实操结论（[[svm-selection-and-tuning]] 据此写成"换实现要重扫 C"）。

`LinearSVC` 的原问题（回填，L346–L352）把这点写明了：它直接优化 hinge 形式的

$$\min_{w,b}\frac{1}{2}w^{T}w + C\sum_i \max\left(0, 1-y_i\left(w^{T}\phi(x_i)+b\right)\right),$$

"不像对偶形式，它不含样本之间的内积，所以著名的核技巧无法应用。这就是 `LinearSVC` 只支持线性核的原因（$\phi$ 是恒等映射）"（L352）。**这是本库见到的最简洁的"为什么线性实现更快"的解释**。

### 4. 多类：内部恒 ovo，接口默认 ovr（来源论述）

- `SVC`/`NuSVC` 内部**永远**用 one-vs-one 训练 $\frac{K(K-1)}{2}$ 个二类器；`decision_function_shape='ovr'`（默认）只是把 ovo 的输出**单调变换**成 ovr 形状，以"与其他分类器保持一致的接口"（L75）。
- `decision_function_shape` 可以在 fit 后改（doctest 里直接 `clf.decision_function_shape = "ovr"`，L86），输出列数从 6 变 4（L84、L89）。
- `LinearSVC` 则是真的 one-vs-rest，训练 $K$ 个模型（L92、L94–L101）。
- 另有 `multi_class='crammer_singer'`（回填，L107）：官方口径是"实践中通常偏好 one-vs-rest，因为结果大多相似而运行时间显著更短"。

属性排布（回填，L109–L120）——这是本节唯一讲清 `dual_coef_` 的地方：

| 属性 | ovr `LinearSVC` | ovo `SVC`/`NuSVC` |
| --- | --- | --- |
| `coef_` | `(n_classes, n_features)` | `(K(K-1)/2, n_features)` |
| `intercept_` | `(n_classes,)` | `(K(K-1)/2,)` |
| `dual_coef_` | — | `(n_classes-1, n_SV)`，"somewhat hard to grasp" |
| 行序 | 按"one"类顺序 | "0 vs 1", "0 vs 2", … "n-1 vs n" |

`dual_coef_` 的列是**支持向量**，每列有 $K-1$ 个对偶系数（每个系数属于该支持向量参与的一个二类器），且"some, but not all, of these dual coefficients, may be zero"（L113）。三类的例子给出具体排布（L115–L120）。

**本库综合**：这个排布的直接后果是——**不能把 `dual_coef_` 读成"每个支持向量的重要性"**。同一个支持向量在 2 个不同的二类器里有 2 个不同符号/量级的系数，其中可能为 0。要做样本重要性分析，`SVC` 走 `dual_coef_` 是错的入口；要么用 ovr 的 `coef_`（线性核），要么换 `LinearSVC`。

### 5. 分数 ≠ 概率：Platt 校准与三条不一致（来源论述）

`decision_function` 给"每类分数（二类时每样本一个分数）"（L128）。`probability=True` 时启用 `predict_proba` / `predict_log_proba`；二类用 **Platt scaling**（对 SVM 分数做逻辑回归，且**用一次额外的交叉验证来拟合**），多类按 [10] 的方式扩展（L128）。

三条限定：

1. 这套校准**对所有估计器都可用**（`CalibratedClassifierCV`），但 `SVC`/`NuSVC` 的是 libsvm 内建，**不依赖** `CalibratedClassifierCV`（L131）。
2. 交叉验证在大数据上昂贵（L133）；概率可能与分数不一致（L135–L136）：**分数的 argmax 未必是概率的 argmax**；二类时 `predict` 的标签可能与 `predict_proba < 0.5` 同时发生。
3. Platt 方法"已知有理论问题"；**只要置信分数、不需要概率，就该 `probability=False` 并用 `decision_function`**（L138）。

另有一条 tie 规则：`decision_function_shape='ovr'` 且类数 > 2 时，`predict` **默认不打破平局**（返回第一个并列类）；要 `break_ties=True` 才等于 `np.argmax(decision_function(...), axis=1)`，"but have in mind that it comes with a computational cost"（L140）。

**本库综合**：这四段合起来给出一个清晰的分层——`decision_function`（原始分数，单调可比）/ `predict`（带 tie 规则的判决）/ `predict_proba`（昂贵的、可能与前者不一致的校准产物）。三者不能互相替代；[[leaderboard-vs-cv]] 的指标选择必须指定在哪一层上算。

### 6. 不平衡：两个参数，同一个机制（来源论述）

- `class_weight`：来源说**只有 `SVC` 有**（`NuSVC` 没有），形式 `{class_label: value}`，作用是"把类 `class_label` 的 `C` 设成 `C * value`"（L146）。**前置警告**：外部核查发现 1.9.1 的 `NuSVC` 签名也有 `class_weight` 且会在 `_validate_targets` 里计入 `class_weight_`——这句可能已过时，见「分歧」第 2 条。
- `sample_weight`：`SVC`、`NuSVC`、`SVR`、`NuSVR`、`LinearSVC`、`LinearSVR`、`OneClassSVM` 七个都在 `fit` 里支持，把第 $i$ 个样本的 `C` 设成 `C * sample_weight[i]`（L150）。
- 实践建议：不平衡时"set `class_weight='balanced'` and/or try different penalty parameters `C`"（L213）。

**本库综合**：这与 logistic 回归里 `class_weight`×`C` 的耦合是**同一个陷阱的不同表述**（[[scikit-learn]] 默认值第 2 条）。含义是：类别权重不是"独立的旋钮"，它乘进正则强度里，所以**加权后必须重扫 `C`**，否则你以为在调类别平衡，其实在调正则。

### 7. 复杂度：这一节的真正用途是选实现（来源论述）

- libsvm 的 QP 求解器在 $O(n_{\text{features}} \times n_{\text{samples}}^2)$ 与 $O(n_{\text{features}} \times n_{\text{samples}}^3)$ 之间，"取决于 libsvm 缓存用得多有效（依数据集而定）"；数据很稀疏时把 $n_{\text{features}}$ 换成**每样本平均非零特征数**（L192）。
- 线性情形下 liblinear 的 `LinearSVC` "much more efficient … can scale almost linearly to millions of samples and/or features"（L194）。

**本库综合**：这两句构成一条**硬判据**：训练样本到 $10^5$ 量级、核方法才开始吃力；到 $10^6$ 只有线性实现可行。它与 L200 的缓存建议（`cache_size` 从默认 200MB 提到 500–1000MB）是同一件事的两面——复杂度上界能否达到，取决于缓存命中。

### 8. 实践提示里有三条会沉默出错（来源论述 + 本库判断）

| 提示 | 来源原话要点 | 为什么算"沉默" |
| --- | --- | --- |
| 缩放 | "not scale invariant, so **it is highly recommended to scale your data**"，并给 `make_pipeline(StandardScaler(), SVC())`（L203–L210） | 不缩放**不报错**，只是分数变差；且测试集必须用同一缩放 |
| `C` | 默认 1，"a reasonable default choice"；噪声多时**调小**（= 更多正则）；`LinearSVC`/`LinearSVR` 对大 `C` 不敏感，而大 `C` 训练时间可长 10 倍（L201–L202） | 默认值恰好可用，于是没人检查它 |
| `shrinking` | 引 [12]：迭代次数大时收缩能缩短训练；但**松散求解**（大 `tol`）时不开收缩反而更快（L211） | 默认 `True`，在粗网格搜索里可能反而更慢 |

另外三条：`nu` 近似训练误差与支持向量的比例（L212）；`SVC`/`NuSVC` 只在 `probability=True` 时用随机数，`LinearSVC` 只在 `dual=True` 时用（此时"同一输入有轻微不同结果是常见的，可试更小的 `tol`"）（L214–L215）；`LinearSVC(penalty='l1', dual=False)` 给特征维稀疏解，`C` 越大选的特征越多，"全零模型"的临界 `C` 可用 `l1_min_c` 算（L216）。

### 9. 自定义核的两个副作用（来源论述，回填补齐）

L255–L258：分类器行为"与任何其他分类器相同"，除了两点——

- `support_vectors_` **为空**，只保留 `support_` 里的索引；
- `fit()` 的**第一个参数存的是引用而非副本**，"if that array changes between the use of `fit()` and `predict()` you will have unexpected results"。

接口契约（回填，L264）：核函数吃两个矩阵 `(n_samples_1, n_features)`、`(n_samples_2, n_features)`，返回 `(n_samples_1, n_samples_2)`；或用 `kernel='precomputed'` 传 Gram 矩阵，此时"**所有**训练向量与测试向量之间的核值都必须提供"（L279）。

**本库综合**：第二条是本页最危险的静默错误——它意味着**不能对传入的数组做原地修改**（包括某些 `inplace=True` 的预处理、`numpy` 视图上的操作）。与第一条合起来：自定义核下你既拿不到支持向量、也拿不到训练数据快照，模型实际上不可解释、不可审计。

### 10. 数学形式：C 是逆正则，ν 是 C 的重参数化（来源论述）

`SVC` 原问题（L317–L321）：最小化 $\frac12 w^Tw + C\sum_i \xi_i$，约束 $y_i(w^T\phi(x_i)+b)\ge 1-\xi_i$、$\xi_i\ge 0$。来源的读法："通过最小化 $\|w\|^2$ 来最大化间隔，同时对误分类或落在间隔边界内的样本施加惩罚"；"惩罚项 `C` 控制惩罚强度，因此充当**逆正则化参数**"（L323）。

对偶（L325–L331）：$\min_\alpha \frac12\alpha^TQ\alpha - e^T\alpha$，s.t. $y^T\alpha=0$、$0\le\alpha_i\le C$，其中 $Q_{ij}\equiv y_iy_jK(x_i,x_j)$。$\alpha_i$ 上界为 $C$，这正是"支持向量"的代数定义。

一条跨模型换算（L344，note）：libsvm/liblinear 用 `C`，多数其他估计器用 `alpha`，"两者正则强度的精确等价关系取决于各自优化的目标函数"；**对 `Ridge` 回归有 $C = 1/\alpha$**。

`NuSVC`（回填，L356–L358）：ν-SVC 是 C-SVC 的**重参数化**，因此数学等价；ν ∈ (0,1] 是**margin errors 比例的上界**与**支持向量比例的下界**；margin error = 落在自己间隔边界错侧的样本（误分类，或分类正确但没越过间隔）。

`SVR`（L362–L384）：ε-SVR 只惩罚偏差 ≥ ε 的样本（$\xi_i$ 与 $\xi_i^*$ 分上下两侧），预测式为 $\sum_{i\in SV}(\alpha_i-\alpha_i^*)K(x_i,x)+b$，`dual_coef_` 存的是**差值**（L384）——与分类的 $y_i\alpha_i$ 不同。

**本库综合**：`C = 1/α`（对 Ridge）给了跨模型迁移超参的唯一一条**可算**的换算；对 Lasso、logistic 则没有（目标不同构）。这条与 [[regularization-and-sparsity]] 里"跨估计器 `alpha` 不可比"是同一结论的正反两面：**要么有闭式换算，要么重扫网格**。

## 版本判定

判定方法同前两次 ingest：把官方 1.6 / 1.7 / 1.8 / 1.9.1 页的 article 正文清洗成词序列后逐段比对。

| 版本 | 该节的差异 |
| --- | --- |
| 1.6 | 1.4.1.1 用旧措辞："In total, `n_classes * (n_classes - 1) / 2` classifiers are constructed and each one trains data from two classes…（`decision_function_shape` 默认 `'ovr'`）" |
| **1.7** | 改写为现在的 ovo 措辞："Internally, the solver always uses this 'ovo' strategy to train the models…"（L75）；**这是本节最后一次实质改动** |
| 1.8 | 与 1.7 逐句相同（10 词窗口比对：双向零差异） |
| **1.9.1** | 与 1.8 逐句相同（同上） |

剪藏与 1.9.1 **逐句一致**（清洗后按句子长度 ≥40 字符比对，仅 12 处不匹配，全部为标题粘连、公式渲染差异与示例列表粘连，已逐条核对）。

结论：`applies_to: scikit-learn 1.7.x – 1.9.1（用户指南 1.4 节无改动）`。**这比 1.1（≥1.8）与 1.9 NB（1.9.x，证据弱）都更硬**，因为 1.7/1.8/1.9.1 三版在本节的字面完全相同。

> [!warning] 但「文档版本」≠「API 版本」
> 1.9 **弃用了 `probability` 参数**（外部核查，见「分歧」），而本节的正文完全没改。因此"剪藏 = 1.9.1 文档"不等于"这里写的用法在 1.9.1 里仍是推荐用法"。这是本库第一次遇到这类错位。

## 转换损失、回填与外部核查

### A. 抓取时丢失的 6 个内容折叠块（898 词，官方正文的 19.0%）

根因同 [[sklearn-linear-models]] 与 [[sklearn-naive-bayes]]：官方页把内容放在 `<details>` 折叠块里，转换器整块丢弃。**本页的 6 个折叠块全部是内容块**（该页另外 8 个 `<details>` 是侧栏导航，不在 article 内）。

| 折叠块 | 词数 | 丢失的是什么 | 回填位置 |
| --- | --- | --- | --- |
| Details on multi-class strategies | 390 | **唯一**讲清 `coef_`/`intercept_`/`dual_coef_` 形状与排布的地方；`crammer_singer` | L105–L120 |
| Using Python functions as kernels | 105 | 核函数的接口契约（两个矩阵 → 核矩阵）；无此段时 L242 的标题只是一个空壳 | L260–L275 |
| Using the Gram matrix | 166 | `kernel='precomputed'` 的用法与"必须提供所有训练×测试核值"（代码块本身幸存） | L277–L297 |
| LinearSVC | 85 | hinge-loss 原问题，以及"为何不能核技巧" | L346–L352 |
| NuSVC | 98 | ν 的语义（margin error 比例上界 / 支持向量比例下界） | L354–L358 |
| LinearSVR | 54 | ε-insensitive loss 原问题 | L386–L392 |

自证：抓取版 L103「See for a complete description of the decision function.」中的指针是空锚点，而它指向的 "Details on multi-class strategies" 根本不存在；L242/L253 只剩标题残迹、下面直接是代码块。

### B. 被吞掉的 10 处页内锚点文字

根因**已定位**：转换器在 `<a class="reference internal" href="#…">` 时丢弃 `<span class="std std-ref">` 内的文字（`href` 以 `#` 开头的才丢，跨页链接全部幸存）。下表按**行**归并（L10 一行内有 3 处），共 10 处：

| 位置（回填后） | 丢失文字 | 回填后句子 |
| --- | --- | --- |
| L10 | `classification` / `regression` / `outliers detection` | "used for classification, regression and outliers detection" |
| L17 | `Kernel functions` | "different Kernel functions can be specified" |
| L21 | `Kernel functions` | "avoid over-fitting in choosing Kernel functions" |
| L22 | `Scores and probabilities` | "(see Scores and probabilities, below)" |
| L32 | `Mathematical formulation` | "(see section Mathematical formulation )" |
| L52 | `Mathematical formulation` | "(detailed in the Mathematical formulation )" |
| L103 | `Mathematical formulation` | "See Mathematical formulation for a complete description" |
| L165 | `Implementation details` | "See Implementation details for further details." |

注意：回填后官方原有的"空格 + 右括号"排布（`(see section Mathematical formulation )`）被保留未改，属原文的转换残留。

### C. 整节丢失的参考文献与 6 个引用标记

官方的 References 有 8 条（`[9]`–`[16]`），抓取版**一条都没有**，只剩 L285「We recommend and as good references」与文末空标题。已回填到 L398–L414：

| 编号 | 条目 | 作用 |
| --- | --- | --- |
| [9] | Platt, *Probabilistic outputs for SVMs…* | Platt 校准的出处 |
| [10] | Wu, Lin & Weng, *Probability estimates for multi-class…*, JMLR 5:975-1005, 2004 | 多类概率 |
| [11] | Fan et al., *LIBLINEAR…*, JMLR 9:1871-1874, 2008 | liblinear |
| [12] | Chang & Lin, *LIBSVM* | libsvm |
| [13] | Bishop, *PRML* ch.7 Sparse Kernel Machines | 理论（来源推荐） |
| [14] | Smola & Schölkopf, *A Tutorial on Support Vector Regression* | 实践（来源推荐） |
| [15] | Schölkopf et al., *New Support Vector Algorithms*, 2000 | ν-SVC |
| [16] | Crammer & Singer, *Multiclass Kernel-based Vector Machines*, JMLR 2001 | `crammer_singer` |

同一根因还吞掉了正文里的 **6 个引用标记**：L128 的 `[9]`（Platt scaling）与 `[10]`（多类扩展）、L202 的 `[11]`（C 大时训练更慢）、L211 的 `[12]`（shrinking 引文）、L396 的 `[12]` 与 `[11]`（libsvm/liblinear）。**全部已回填**（重复脚注块里的对应四处也一并补齐）；另两个标记 `[15]`（ν-SVC）与 `[16]`（Crammer-Singer）随回填折叠块一起到位（L356、L107）。`[9]`–`[16]` 的编号起点是 9 而非 1，因为 `[1]`–`[8]` 属于同批其他章节，本节未引用。

### D. 未修的部分：~877 词的重复脚注块

抓取版把正文的 1.4.1.2、1.4.5（`C`、缩放、`shrinking`、`nu`、不平衡、随机性、L1）、1.4.7 note 与 1.4.8 整段**复制成脚注**（`[^2]`、`[^3]`、`[^4]`、5 个 `[^undefined]`、`[^5]`、`[^7]`），位于 L416–L460。这是转换器的 footnote 抽取错乱，**不是内容损失**。

**保留未删**（视为抓取痕迹，且删除会改动更多原始字节）。后果：引用时应优先引正文的对应行（如 L128 而非 L416），本页所有引用均已如此。唯一例外是回填时对重复段落做了同样的引用标记补齐（L446 的 `[12]`），以保持两处一致。

### E. 未修的公式排版噪声

正文公式被转成 `\left(\right.` / `\left.\right)` 形式（L223 多项式核、L315 的 $\{1,-1\}^n$、L331、L376），渲染结果正确但源码已坏；复杂度一节把 $n_{features}$ 拼成 $n_{f e a t u r e s}$（L192）。**回填的新内容一律写规范 LaTeX**，原有公式未改（与 [[sklearn-linear-models]] 的处理一致）。

### F. 外部核查（不在剪藏内，逐条标注）

核查对象：`sklearn/svm/_classes.py` @ tag `1.9.1`（GitHub raw）、`doc/whats_new/v1.9.rst`、官方 1.9.1 页。

| 参数 | 1.9.1 默认值 | 剪藏是否提及 |
| --- | --- | --- |
| `C` | `1.0` | 提了（L201） |
| `kernel` | `'rbf'` | 未提默认值 |
| `gamma` | **`'scale'`** | **完全未提**（正文只描述 γ 的语义，L242、L224） |
| `degree` / `coef0` | `3` / `0.0` | 只提"由参数指定"（L223） |
| `shrinking` | `True` | 只讨论取舍（L211） |
| `probability` | **`"deprecated"`（1.9 起）** | 未提弃用 |
| `tol` | `1e-3` | 未提 |
| `cache_size` | `200` | 提了（L200） |
| `class_weight` | `None` | 提了语义（L146），但"仅 SVC 有"存疑，见「分歧」2 |
| `max_iter` | `-1` | 未提 |
| `decision_function_shape` | `'ovr'` | 提了（L75） |
| `break_ties` | `False` | 提了行为（L140） |
| `NuSVC.nu` / `SVR.epsilon` / `OneClassSVM.nu` | `0.5` / `0.1` / `0.5` | 只提语义（L212、L362） |
| `LinearSVC` 的 `penalty`/`loss`/`dual`/`multi_class` | `'l2'` / `'squared_hinge'` / `'auto'` / `'ovr'` | 只提 `squared_hinge`（L32）与 L1 组合（L216） |

## 分歧

### `probability=True` 在 1.9 已被弃用，而本文档仍在教它

- **来源论述**（L128、L131、L138）：`probability=True` 启用 `predict_proba`；校准内建于 libsvm，不依赖 `CalibratedClassifierCV`；若只需分数就用 `decision_function`。
- **外部核查**（`whats_new/v1.9.rst`，`sklearn.svm` 段）：

  > The `probability` parameter of `sklearn.svm.SVC` and `sklearn.svm.NuSVC` is deprecated **due to not being thread-safe** and will be removed in 1.11. Use `CalibratedClassifierCV` with the respective estimator and `ensemble=False` instead.

  同版还弃用了 `probA_` / `probB_` 两个属性（同一 PR 系列的后续）。源码 `_classes.py` @ 1.9.1 的签名已是 `probability="deprecated"`，docstring 标注 `.. deprecated:: 1.9`。
- **性质**：文档落后于 API，且**理由不是精度而是线程安全**。这一点对智能体场景尤其重要——并行/多进程里跑 `SVC(probability=True)` 会踩到 libsvm 的全局状态。
- **处理**：两处都保留。来源页按"剪藏这么写"记录，工具页与实践页按"1.9 起改用 `CalibratedClassifierCV(SVC(), ensemble=False)`"记录并标注版本边界。**不静默改写剪藏正文**。

### `NuSVC` 到底有没有 `class_weight`？

- **来源论述**（L146）：「`SVC`（但**不是** `NuSVC`）在 `fit` 方法里实现了 `class_weight` 参数」。
- **外部核查**（`sklearn/svm/_classes.py` @ 1.9.1）：`NuSVC.__init__` 的签名包含 `class_weight=None`，直接传给 `BaseSVC.__init__`；`_base.py` 的 `BaseSVC._validate_targets` 无条件调用 `compute_class_weight(self.class_weight, classes=cls, y=y_)` 写入 `class_weight_`，再传给 libsvm。1.6.1 / 1.7.2 / 1.8.0 的签名里也都已存在。
- **性质**：**文档疑似过时**（至少在 1.6.1 起就如此）；`NuSVC` 自己的 docstring 写的是"…for `SVC`"，属复制残留。
- **本库处理**：**不改写来源句子**；来源页、概念页、实践页与工具页统一写成"来源说仅 `SVC` 有，但 1.9.1 的实现接受该参数"。**未做运行时验证**（不跑本机环境），所以只断言"签名接受且代码路径会用它"，不断言"行为与 SVC 等价"。

## 图表核查

4 张官方示例图于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-svm/`，**逐张目视核查**。

| 图 | 文件 | 尺寸 | SHA-256（前 16） | 对应正文 |
| --- | --- | --- | --- | --- |
| 图 1 | `sphx_glr_plot_separating_hyperplane_001.png` | 640×480 | `1a7a6fa2cdf7647d` | L307 |
| 图 2 | `sphx_glr_plot_iris_svc_001.png` | 640×480 | `ba62ed3edae812a7` | L30 |
| 图 3 | `sphx_glr_plot_separating_hyperplane_unbalanced_001.png` | 640×480 | `187d77a8f6ff118e` | L148 |
| 图 4 | `sphx_glr_plot_weighted_samples_001.png` | 1400×600 | `ac75593295f03918` | L152 |

**图 1（最大间隔）**：两类散点（浅蓝在上、深橙在下），一条实线判决边界 + 两条平行虚线间隔；**恰好 3 个样本被圈出**——1 个浅蓝（约 x≈7.4, y≈−4.9）与 2 个深橙（约 x≈5.9, y≈−6.9；x≈8.0, y≈−7.5）。三点的位置印证 L339「只需对支持向量求和」与 L309「线性不可分时支持向量落在间隔**内**」：本例线性可分，所以支持向量正好压在虚线上。

**图 2（iris 四宫格）**：`SVC(linear)` 与 `LinearSVC` 两格的判决边界**几乎重合**（都是直线），但支持向量圈的分布不同——**这是本库对 L32「结果与分数可能不同」的唯一图形旁证**：损失（`hinge` vs `squared_hinge`）与截距是否正则化改变了支持向量的选择，而边界在小数据上差别很小。RBF 与 polynomial(degree=3) 两格边界明显弯曲，poly 格左下角还多出一小块深橙区域。

**图 3（不平衡）**：蓝类为密集大样本，橙类只有百点量级且聚在 (1.8, 1.8) 附近。黑线 = 未加权，红线 = 加权。红线整体向**蓝（多数）类一侧平移**，从而扩大判给橙类的区域——与 L146「`value` 把该类的 `C` 乘上 `value`」的方向一致。图为示意数据，且无图例坐标刻度，读数只作方向性参考。

**图 4（加权样本）**：左右两格为 `decision_function` 的等高线填充，圆面积 ∝ `sample_weight`。左格权重全等，边界沿两类分布的中线；右格少数大权重样本把边界拉过去，其中一个位于深色区的大圆（约 (0.7, −1.9)）周围出现明显的凸起。**该图无坐标轴刻度**，只能做定性比较。

## 局限

- 回填后的文本仍不等于官方 HTML：公式保留转换噪声（E）、L416–L460 有 ~877 词重复脚注（D）、页内锚点文字已恢复但**链接本身不存在**（回填的是纯文字）。引用具体论断时以正文行为准。
- 本页**未读** [9]–[16] 任何一篇文献。因此"Platt 校准有理论问题"（L138）、"ν 是 margin error 比例上界"（L358）在本库中都是**来源转述**，不是已核实结论。
- 本节不覆盖 `sklearn.calibration`（`CalibratedClassifierCV` 的 `ensemble` 语义、校准曲线）、`sklearn.kernel_approximation`（Nyström / RBFSampler）、`sklearn.multiclass`（`OneVsRestClassifier` 的通用做法）、`sklearn.svm.l1_min_c` 的细节。这四块都在未入库章节里。
- `gamma` 的默认值（`'scale'`）与 `dual='auto'` 的选取规则**文档完全没有**，本页只能标为外部核查；不要据此认为"文档没写就可以忽略"。
- 本节对 `OneClassSVM` 只有两句话（L186–L188），任何关于 novelty detection 的结论都应转向 2.7 节（未入库）。

## 待解问题

1. **`CalibratedClassifierCV(SVC(), ensemble=False)` 与旧的 `SVC(probability=True)` 是否数值等价？** changelog 只说弃用理由是线程安全，没说两条路径的算法是否同一套（是否同样是内部五折 + Platt）。需要读 `sklearn/calibration.py` 或做数值对照。
2. **"五折交叉验证"（L22）与 `ensemble=False` 的折数是否一致？** 若替代方案用不同的折数，那么"昂贵"的定量含义也随之改变。
3. **`NuSVC` 到底有没有 `class_weight`？** L146 说"but not NuSVC"，但 1.9.1 的签名接受该参数（见「分歧」2）。本库因不跑本机 Python 环境，**未做运行时验证**；需一个最小复现（`NuSVC(class_weight='balanced')` 是否报错、`class_weight_` 是否被填充）才能定性。
4. **$O(n_f n_s^2)$–$O(n_f n_s^3)$ 的实际拐点在哪？** 来源说"dataset dependent"，本库没有任何实数。这是"何时该转 `LinearSVC`"的唯一量化依据，目前是空的。
5. **`dual='auto'` 的选择规则是什么？** 文档不提，源码签名是 `"auto"`。它决定了 `LinearSVC` 在 `n_samples > n_features` 时走哪条路，从而决定随机性是否生效（L215）。
6. **`break_ties=True` 的"computational cost"具体多少？** 它要算完整的 ovr `decision_function` 再取 argmax，量级应是 $O(K)$ 倍；来源未给。
7. **自定义核下 `support_` 的索引在多类 ovo 下如何对齐到 `decision_function` 的列？** 回填的多类节只讲 `dual_coef_` 排布，未与 `support_` 联立说明。
8. **`l1_min_c` 的数值稳定性**：L216 只给"可用它算全零模型的临界 C"。在本库关心的 $p \gg n$ 场景下是否可靠，未验证。

## 行号索引（回填后）

- L10 定义；L14–L17 优点；L21–L22 缺点；L24 输入要求
- L28 三个分类器；L32 `squared_hinge`/截距/属性差异；L36–L65 支持向量 doctest；L52 决策函数只依赖支持向量
- L75 ovo 内部/ovr 接口；L92 `LinearSVC` 的 ovr；L103 指针；**L105–L120 多类属性排布（回填）**
- L126–L140 分数与概率；L128 Platt；L131 与 `CalibratedClassifierCV` 的关系；L133–L138 不一致与建议；L140 tie breaking
- L144–L152 `class_weight` / `sample_weight`；L146 仅 SVC 有 `class_weight`
- L161–L167 回归；L165 `LinearSVR` 的差异
- L186–L188 `OneClassSVM`
- L192 复杂度；L194 liblinear 近线性
- L198–L216 实践提示；L203 缩放；L211 shrinking；L216 L1
- L220–L225 四个核；L242–L244 C 与 gamma；L255–L258 自定义核副作用；**L260–L297 自定义核细节（回填）**
- L305–L311 间隔直觉；L315–L344 SVC；L323 C 是逆正则；L339 只对支持向量求和；L344 `C = 1/α`
- **L346–L352 LinearSVC 原问题（回填）**；**L354–L358 NuSVC 的 ν（回填）**
- L362–L384 SVR；**L386–L392 LinearSVR 原问题（回填）**
- L396 双实现封装；L398–L414 参考文献（回填）
- L416–L460 重复脚注块（转换痕迹，勿引用）

## 关联页面

- 概念：[[support-vector-machines]]（本来源的主概念页）、[[regularization-and-sparsity]]（`C` 作为正则化面）、[[naive-bayes]]（同为"分数 ≠ 概率"的来源）、[[leaderboard-vs-cv]]（指标该在哪一层算）
- 实践：[[svm-selection-and-tuning]]（本来源的实践页）、[[linear-models-as-baselines]]（线性核 SVC 的位置）
- 工具：[[scikit-learn]]（`svm` 估计器矩阵与默认值）
- 同批来源：[[sklearn-linear-models]]、[[sklearn-lda-qda]]、[[sklearn-naive-bayes]]
