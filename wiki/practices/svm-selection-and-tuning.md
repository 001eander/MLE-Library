---
title: "SVM 的选型与调参：规模判据、缩放前提、C/γ 网格与六个坑"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - svm
  - scikit-learn
  - hyperparameter-tuning
  - workflow
sources:
  - "[[sklearn-svm]]"
  - "[[sklearn-linear-models]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.7.x – 1.9.1 的 svm 模块（用户指南 1.4 节，该节三版无改动）
  - scikit-learn 1.9.x 起 `SVC(probability=True)` 已弃用，1.11 移除（外部核查，见 [[sklearn-svm]] 的分歧节）
---

# SVM 的选型与调参

本页回答三个具体问题：**该选哪个 SVM 估计器**、**必做哪些前置步骤**、**哪些默认值会沉默地骗你**。事实来自 [[sklearn-svm]]，引用写作 `[[sklearn-svm]], L192`；本库自己的判断一律标注。

## 一、先选实现，再调超参

SVM 在 sklearn 里是两个不相干的实现（libsvm 与 liblinear，[[sklearn-svm]], L396），**它们不是同一个模型的两个后端**。选错实现的代价比调错 `C` 更大。

| 情形 | 选择 | 理由 | 依据 |
| --- | --- | --- | --- |
| $n_s \lesssim 10^5$，需要非线性 | `SVC(kernel='rbf')` | 唯一能上核的实现 | L192 |
| $n_s \gtrsim 10^6$，只需线性 | `LinearSVC` | "almost linearly to millions of samples and/or features" | L194 |
| 同上，且不想被拷贝数据 | `SGDClassifier(loss='hinge')` | 官方明确建议（输入是稠密 C-order 数组时） | L199 |
| 要特征维稀疏解 | `LinearSVC(penalty='l1', dual=False)` | 唯一提供特征选择的 SVM | L216 |
| 要"我只能容忍 x% 的 margin error"这种语义 | `NuSVC` / `NuSVR` | ν 直接是这个比例 | L212、L358 |
| 只要线性核、但想保留核接口 | `SVC(kernel='linear')` | 注意：与 `LinearSVC` **不是**同一模型 | L32 |
| 单类异常检测 | `OneClassSVM` | 正文只有两句话，细节在 2.7 节（未入库） | L186–L188 |

**本库综合**：第 6 行是最常见的误判。`SVC(kernel='linear')` 用 hinge（或 ν 版本）、截距**不进正则**；`LinearSVC` 用 `squared_hinge`、截距**进正则**且可用 `intercept_scaling` 削弱（L32、L350–L352）。两条差异叠加，使**同一 `C` 下两个模型的结果不同**——官方自己也警告"The classification results and score can therefore differ"（L32）。所以：换实现必须重扫 `C`，不能只换类名。

## 二、缩放是硬前提，不是优化项

来源的原话是 "Support Vector Machine algorithms are not scale invariant, so **it is highly recommended to scale your data**"（L203）。这是本库入库的五类模型里唯一被官方用"highly recommended"标注的预处理要求。

必须做对的两件事：

1. **缩放器只能在训练折上 `fit`**。把 `StandardScaler` 放进 `Pipeline` 是最省事的保证（L207–L210 就是官方给的写法）。
2. **推理时必须用同一套缩放参数**。来源特别强调 "the *same* scaling must be applied to the test vector"（L203）。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(probability=False))
```

**本库综合（为什么不能只做"标准化一下"）**：间隔是基于欧氏距离的几何量，而 `gamma`（RBF）与 `C` 都在同一个距离尺度上起作用。未缩放时，量纲大的特征会主导距离，`C` 与 `gamma` 的最优值随之漂移——也就是说**预处理与超参是耦合的，换预处理必须重扫网格**。这条与 [[linear-models-as-baselines]] 的"尺度硬约束"同源，但后果更重：线性模型的系数尺度会变，SVM 是**决策面本身**会变。

## 三、C 与 gamma：指数间隔网格，先缩放再搜

来源给的建议很短：`C` 与 `gamma` 两个都要考虑；"one is advised to use `GridSearchCV` with `C` and `gamma` spaced exponentially far apart"（L242–L244）。

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

gs = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False)),
    {"svc__C": np.logspace(-2, 2, 5), "svc__gamma": np.logspace(-3, 1, 5)},
    cv=5,
    n_jobs=-1,
)
```

三个语义（L242–L244）：

- `C`：折中"训练样本被误分类"与"决策面简单"。**小 `C` → 面更平滑；大 `C` → 力求全部训练样本正确**。
- `gamma`：单个训练样本的影响力半径。**越大 → 影响越局部**（其他样本必须更近才受影响）；必须 > 0（L224）。
- 两者"对性能都至关重要"（L244）——不是"一个主一个次"。

**本库综合（成本侧）**：`C` 增大会让训练时间上升，"sometimes up to 10 times longer"（L202）。因此网格应按对数尺度**从小到大**扫，先看分数是否已饱和，再决定是否继续放大——而不是一开始就铺满 $10^{-3}$–$10^{3}$。

`gamma` 的默认值不在剪藏里（外部核查）：1.9.1 的签名是 `gamma="scale"`（即 $1/(n_{\text{features}}\cdot \mathrm{Var}(X))$），`degree=3`、`coef0=0.0`。**文档正文只描述 γ 的语义，不说默认值**——这意味着"用默认 gamma 跑一次"这件事在文档里根本查不到。

## 四、概率：默认关，而且 1.9 起旧写法已弃用

三层输出的用途（[[support-vector-machines]] 有完整表）：

| 需求 | 用法 | 成本 |
| --- | --- | --- |
| 排序、阈值、AUC | `decision_function` | 无额外成本 |
| 类标签 | `predict`（多类 ovr 下默认不破平局，L140） | 无 |
| 校准概率 | `probability=True` + `predict_proba` | **内部交叉验证**，大数据上昂贵（L22、L133） |

来源的建议是明确的：如果只要置信分数、不需要概率，就**关掉概率**用 `decision_function`（L138）。理由不只是成本——概率与分数可能不一致："分数的 argmax 未必是概率的 argmax"，二类时 `predict` 可能与 `predict_proba < 0.5` 同时发生（L135–L136）。

> [!warning] 1.9 的 API 变化（外部核查）
> `SVC(probability=True)` 与 `NuSVC(probability=True)` 在 1.9 **已弃用**，原因是**非线程安全**，1.11 移除；替代写法是 `CalibratedClassifierCV(SVC(), ensemble=False)`，`probA_` / `probB_` 同时弃用。剪藏的正文（1.9.1 的页面）**没有反映这一点**。
> 工程含义：`n_jobs=-1` 或多进程里跑 `SVC(probability=True)` 会踩到 libsvm 的全局状态；这也是为什么上面的网格示例显式写了 `probability=False`。

## 五、多类：内部恒 ovo，接口默认 ovr

- `SVC` / `NuSVC` **永远**训练 $\frac{K(K-1)}{2}$ 个二类器（ovo）；`decision_function_shape='ovr'` 只是把输出**单调变换**成 $K$ 列（L75）。`LinearSVC` 是真的 ovr，训练 $K$ 个（L92）。
- `decision_function_shape` 可以在 `fit` 之后改（L86）——改的是输出形状，不重训。
- `decision_function_shape='ovr'` 且 $K>2$ 时，`predict` **默认不破平局**，返回第一个并列类；要 `break_ties=True` 才等于 `np.argmax(decision_function(...), axis=1)`（L140）。
- 属性排布（回填，L109–L120）：ovo 的 `coef_` / `intercept_` 行数是 $\frac{K(K-1)}{2}$，`dual_coef_` 形状 `(n_classes-1, n_SV)`，**每列有 $K-1$ 个系数、其中可能为 0**。

**本库综合（三个实操后果）**：
1. **`K` 大时 ovo 的训练量按 $K^2$ 增长**，而 `LinearSVC` 是 $K$。类数上百时这不是小差别。
2. **平局规则会改变报告的数字**。同一模型 `break_ties=False/True` 的准确率不同；任何"准确率"结论都必须声明用的是哪个。
3. **不要用 `dual_coef_` 做样本重要性**。多类 ovo 下它按列（支持向量）× 行（其余类）排列，符号与大小都随二类器变化。

## 六、不平衡：三个参数，同一个机制

- `class_weight`：来源说**只有 `SVC` 有**（`NuSVC` 没有），形式 `{class_label: value}`，作用是**把该类的 `C` 乘上 `value`**（L146）。注意这个“只有 SVC”的说法有争议，见下方注释。
- `sample_weight` 七个估计器都有，**把第 $i$ 个样本的 `C` 乘上 `sample_weight[i]`**（L150）。
- 来源建议：不平衡时 `class_weight='balanced'` 和/或"试试不同的 `C`"（L213）。

```python
SVC(class_weight="balanced", C=...)   # 然后重扫 C
```

**本库综合**：这里的关键不是"要加权"，而是**加权改变的是正则强度**。`C` 与类别权重相乘进同一个松弛项，所以：

- 加权后必须**重扫 `C`**，否则你以为在调类别平衡，其实在调正则；
- `class_weight` 与 `sample_weight` **同时**用会叠乘，容易把有效 `C` 推到极端；
- **`NuSVC` 的 `class_weight`**：文档说它没有，但 1.9.1 的源码里 `NuSVC` 签名接受该参数并会计算 `class_weight_`（外部核查，见 [[sklearn-svm]] 的分歧节）。暂存两说，需要一次运行时复现才能定性——**在确认前，对 `NuSVC` 的不平衡问题至少可以用 `sample_weight`**。

这与 [[scikit-learn]] 默认值清单里 logistic 的 `class_weight`×`C` 耦合是同一个陷阱。

## 七、内存与速度：四个可调项

| 项 | 默认 | 调整依据 |
| --- | --- | --- |
| `cache_size` | 200（MB，外部核查） | 大问题上内核缓存"对运行时间有强影响"；有 RAM 就提到 500–1000（L200） |
| 数据布局 | — | C-order 连续 `float64` 才不被拷贝（L198）；`LinearSVC` 会**总是**转成 liblinear 内部稀疏格式（L199） |
| `max_iter` | -1（`SVC`；`LinearSVC` 为 1000，均为外部核查） | 无上界；大数据上应设上限以便失败可见 |
| 实现 | `SVC` | 线性核 + 大数据时换 `LinearSVC` 或 `SGDClassifier`（L194、L199） |

**本库综合**：`cache_size` 是唯一"改了立刻见效但不改也不报错"的参数——它不改结果，只改时间。在网格搜索里它被反复重置，因此**在大问题上应把 `cache_size` 写进估计器构造参数**，而不是留在默认。

## 八、可复现性：只有两种情况与随机数有关

| 估计器 | 何时随机 | 依据 |
| --- | --- | --- |
| `SVC` / `NuSVC` | 仅当 `probability=True`（用于概率估计的样本洗牌） | L214 |
| `LinearSVC` | 仅当 `dual=True`（选特征用的坐标下降顺序） | L215 |
| `OneClassSVM` | 无概率估计 → 不随机 | L214 |
| `SVR` / `NuSVR` / `LinearSVR` | 无 `class_weight`、无概率 → 确定性 | L214–L215 |

`LinearSVC(dual=True)` 下"同一输入轻微不同结果是常见的"；来源的建议是**试更小的 `tol`**（L215）。

**本库综合**：这条对评估器设计很重要——`SVC(probability=False)` 是**确定性**的，同一数据同一参数必然复现，这在 [[evaluator-design-and-reward-hacking]] 的"同一配置换种子重复"检查里意味着**没有种子可换**。要估计噪声只能换 CV 折或换子采样。

## 九、特征维稀疏（L1）的具体用法

`LinearSVC(penalty='l1', dual=False)` 给稀疏解："只有一部分特征权重非零"；`C` 越大模型越复杂（选的特征越多）；**全零模型**的临界 `C` 可用 `l1_min_c` 算（L216）。

**本库综合**：这条给了"用 SVM 做特征选择"的唯一正规入口。注意三个前提：`penalty='l1'` 必须配 `dual=False`（来源的写法，L216）；截距仍被正则化（L32），所以 `intercept_scaling` 会影响"稀疏到什么程度"；以及 `C` 在这里同时是**正则强度**和**特征数控制**——两个目标共用一个旋钮，网格必须比 RBF 情形更细。

## 十、自定义核的检查单

| 检查 | 原因 |
| --- | --- |
| 核函数签名是「两个 `(n, d)` 矩阵 → `(n1, n2)` 矩阵」 | 否则形状报错或被静默广播（L264） |
| `kernel='precomputed'` 时**训练×测试**的核值都提供 | 只给训练 Gram 会在 `predict` 时才失败（L279） |
| 传给 `fit` 的数组**之后不再改动** | `fit` 存引用；改动会让 `predict` 结果异常（L258） |
| 不要指望 `support_vectors_` | 自定义核下它为空，只有索引（L257） |
| 大 `n` 时评估内存 | Gram 是 $O(n^2)$（L279） |
| `n_jobs>1` 时核函数必须可并行 | 每次求值都走 Python（L264） |

## 十一、常见坑清单（按"踩到才发现"排序）

1. **未缩放就调参**。分数差且网格结论不可迁移（L203）。
2. **把 `SVC(kernel='linear')` 与 `LinearSVC` 当作等价**。损失与截距正则都不同（L32）。
3. **`probability=True` 拖慢网格搜索**，且 1.9 起已弃用；并行下还有线程安全问题（L22、L133 + 外部核查）。
4. **只记 `predict` 的准确率而没声明平局规则**。多类 ovr 默认返回第一个并列类（L140）。
5. **加权后没重扫 `C`**。`class_weight` / `sample_weight` 就是乘 `C`（L146、L150）。
6. **推理时稀疏性/稠密性与训练不一致**。能跑但走慢路径（L24）。
7. **`LinearSVC` 总是拷贝输入**。大数据上这是主要开销，需换 `SGDClassifier`（L199）。
8. **`cache_size` 留在默认 200MB**。大问题上时间显著变长，且无任何提示（L200）。
9. **给 `LinearSVC` 传 `penalty='l1'` 却忘了 `dual=False`**（L216 的写法）。
10. **自定义核下原地修改了训练数组**（L258）。
11. **在多类模型上用 `dual_coef_` 解释样本重要性**（L113）。
12. **期望 `SVC` 的 `random_state` 保证复现**：`probability=False` 时它不起作用（L214）。

## 十二、对智能体的可自动化程度

**本库综合**，与 [[competition-ml-workflow]] 的分阶段表同一视角：

| 环节 | 可自动化程度 | 依据与风险 |
| --- | --- | --- |
| 选实现（按 $n_s$） | **高** | 只有两条判据（L192、L194），可写死 |
| 缩放 + Pipeline | **高** | 官方给了标准写法（L207–L210） |
| `C`/`gamma` 网格 | **高** | `GridSearchCV` 一步；但**成本随 `C` 上升**，需设预算 |
| 概率校准 | 中 | 1.9 起写法变了（外部核查）；且与分数不一致（L135） |
| 不平衡处理 | **中** | 加权与 `C` 耦合，需要联合搜索而非单参数搜索 |
| 判断"要不要上核方法" | 低 | 文档只给复杂度，不给与线性模型的对照实验 |
| 解释模型（支持向量/系数） | 低 | 多类 ovo 的 `dual_coef_` 不可直接读（L113） |

一条可操作建议：**把 SVM 放在"线性基线打过之后"**。若 `LinearSVC` 或 `LogisticRegression` 的 CV 已接近目标，核 SVM 的收益通常抵不过 $O(n_s^2)$ 的成本；反过来，若线性基线的训练误差都下不去，RBF 是下一个该试的族（[[linear-models-as-baselines]] 的失效方式互补表）。

## 待解问题

- **规模判据的拐点具体是多少？** 本页只能写"$10^5$ 可行 / $10^6$ 只有线性"，这是从渐近复杂度推的，**没有实测**。需要一组 $n_s$ × `cache_size` 的耗时数据。
- **`cache_size` 的最优设置与 `n_s` 是什么关系？** 来源只说"有 RAM 就调大"。
- **`probability=True` → `CalibratedClassifierCV(ensemble=False)` 迁移后，分数会不会变？** 外部核查只给了替代写法，没有等价性说明。
- **多类 ovr 的平局发生率有多高？** 若在某些数据上频繁发生，`break_ties` 就不是可选项。本库无数据。
- **`LinearSVC(penalty='l1')` 下 `C` 与特征数的关系曲线**能否用来做特征选择？`l1_min_c` 只给全零的临界点。
- **RBF 与 `Nystroem` / `RBFSampler` 近似的精度差距**：来源指向 Kernel Approximation（L238）但该章未入库。

## 关联页面

- 来源：[[sklearn-svm]]（本页的事实基础）、[[sklearn-linear-models]]（`C = 1/α` 与线性基线）
- 概念：[[support-vector-machines]]（机制）、[[regularization-and-sparsity]]（`C` 作为正则化面）、[[leaderboard-vs-cv]]（指标层级）
- 实践：[[linear-models-as-baselines]]（基线顺序与尺度约束）、[[outlier-detection-selection-and-tuning]]（`OneClassSVM`：同一个 `nu` 换成「训练集异常率」）、[[competition-ml-workflow]]（模型选择在流程中的位置）、[[evaluator-design-and-reward-hacking]]（确定性模型的噪声估计）
- 工具：[[scikit-learn]]（`svm` 估计器矩阵与默认值清单）
