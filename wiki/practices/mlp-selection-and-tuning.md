---
title: "MLP 的选型与调参：先缩放、再定容量、最后选求解器"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - neural-network
  - mlp
  - hyperparameter-tuning
  - preprocessing
sources:
  - "[[sklearn-neural-networks]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.0 – 1.9.1 的 `MLPClassifier` / `MLPRegressor`（默认值与行为在 1.6.1 实测，签名在 1.9.1 核对）
---

# MLP 的选型与调参

本页是 [[multi-layer-perceptron]] 的工程侧：什么时候值得用 scikit-learn 的 MLP、先改哪个旋钮、哪些默认值会沉默地改变结果、以及怎么报告一个可信的分数。事实来自 [[sklearn-neural-networks]]（官方用户指南 1.17 节，1.8.0–1.9.1），引用写作 `[[sklearn-neural-networks]], L137`；默认值标「外部核查」，运行时读数标「本机实测 1.6.1」。

## 先判该不该用

scikit-learn 的 MLP 有明确的定位（L10–L11）：**不是大规模应用，也没有 GPU 支持**；更快/更灵活的实现应看 Related Projects。因此第一步不是调参，而是判断工具是否匹配：

| 场景 | 建议 | 理由 |
| --- | --- | --- |
| 中小规模表格数据、非线性交互明显、CPU 可接受 | **可以试** | MLP 能学稠密表示；`partial_fit` 可做在线/小批量 |
| 大规模数据、图像、序列、需要 GPU | **不要用这个实现** | L11 明确无 GPU；没有卷积/循环/注意力层 |
| 需要强可解释性（系数/规则/重要性） | **优先线性模型或树** | MLP 没有 `feature_importances_`；解释要靠外部方法 |
| 需要原生缺失值/类别特征 | **优先树集成** | 本节没有缺失值/类别支持说明；树/GBDT 有原生支持 |
| 已有强 GBDT/线性基线且差距很小 | **先不引入 MLP** | MLP 的训练方差与调参成本更高；收益不确定（本库判断） |

**本库综合**：MLP 不是「默认更强」的模型。它值得引入的场景通常是：线性模型明显欠拟合、树集成的轴对齐边界不够、且数据规模仍在 CPU 可接受范围内。把它当「最后的非线性尝试」比当「第一基线」更合适。

## 硬前提：缩放

来源的措辞是 "sensitive to feature scaling, so it is highly recommended to scale your data"（L137）。三种可选：缩到 $[0,1]$、缩到 $[-1,+1]$、或标准化到均值 0/方差 1；并且必须对测试集应用**同一个**缩放。L147 推荐把 `StandardScaler` 放进 `Pipeline`。

本机实测 1.6.1（600 样本 × 10 特征，其中一列 ×1000、一列 ×0.001，5 个种子）：

| 预处理 | 5 个种子的验证准确率 | 平均 |
| --- | --- | --- |
| 不缩放 | 0.822、0.883、0.739、0.767、0.728 | **0.788** |
| 训练集拟合 `StandardScaler` 后变换 | 0.883、0.889、0.889、0.894、0.889 | **0.889** |

不缩放组的极差约 0.155，缩放组约 0.011。**缩放不只是提分，它降低结果对种子的敏感度**（本库判断）。

正确写法：

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(50,), solver="lbfgs", max_iter=2000,
                  random_state=0),
)
clf.fit(X_train, y_train)   # scaler 只在这份训练数据上 fit
score = clf.score(X_test, y_test)
```

**不要**先在全部数据上 `scaler.fit(X)` 再 CV；那会把测试折的均值/方差泄漏进训练。剪藏里的 `StandardScaler` 示例（L138–L146）直接使用未定义的 `X_train`，它是**片段**，不是可运行的 doctest。

## 容量：先定架构，再定宽度

来源给出的复杂度是（L133）：

$$O\left(i\cdot n\cdot\left(m\cdot h+(k-1)\cdot h\cdot h+h\cdot o\right)\right)$$

并建议「从小数量的隐藏神经元和少量隐藏层开始」。外部核查 1.9.1 的默认是 `hidden_layer_sizes=(100,)`——**一个隐藏层、100 个神经元**，不是「无隐藏层」。

### 参数量的手算

`coefs_` 的形状是「每层边界的矩阵」，列表长度 = `n_layers_ - 1`（L23）。对 $m$ 个输入、一个宽度 $h$ 的隐藏层、$o$ 个输出：

$$\text{参数量}=(m\cdot h+h)+(h\cdot o+o)$$

本机实测：

| `hidden_layer_sizes` | `n_layers_` | 二分类 `coefs_` | 含义 |
| --- | --- | --- | --- |
| `()` | 2 | `[(m,1)]` | 没有隐藏层，只有线性输出 |
| `(h,)` | 3 | `[(m,h),(h,1)]` | 一个隐藏层 |
| `(h1,h2)` | 4 | `[(m,h1),(h1,h2),(h2,1)]` | 两个隐藏层 |

### 起点建议

1. **先试一个隐藏层**，宽度从**小于默认 100** 的量级开始（例如 10–50；这是工程判断，不是来源给出的数值）。来源只要求「从小开始」，没有给具体数。
2. 如果欠拟合（训练/验证都低），再**加宽**或**加层**；加宽是 $h^2$ 成本，先加宽通常更贵。
3. 如果过拟合（训练高、验证低），先加 `alpha` 或早停，而不是继续加宽。
4. `hidden_layer_sizes=()` 在结构上是一个线性输出层；它可以作为「MLP 是否比线性模型更好」的快速对照，但**不要把它当成 `LogisticRegression` 的等价替换**（损失归一化、正则尺度、求解器都不同）。

## 正则：`alpha` 的网格与读数

来源建议用 `GridSearchCV` 找 `alpha`，范围通常是 `10.0 ** -np.arange(1, 7)`（L148），即：

$$0.1,\ 0.01,\ 0.001,\ 10^{-4},\ 10^{-5},\ 10^{-6}$$

外部核查 1.9.1 的默认 `alpha=1e-4` 正好在这个网格内。实现层只罚 `coefs_`、不罚 `intercepts_`（外部核查 1.6.1 源码），所以「`alpha` 很大时所有参数都变小」是错的——偏置不受影响。

本机实测（`make_moons`，400 样本，`(50,)`，L-BFGS，`max_iter=2000`）：

| `alpha` | 训练准确率 | 权重 L2 范数 | `n_iter_` |
| --- | --- | --- | --- |
| $10^{-5}$ | 0.980 | 100.66 | 1742 |
| $10^{-3}$ | 0.9725 | 82.49 | 2000（撞上限） |
| $0.1$ | 0.920 | 12.60 | 1295 |
| $1.0$ | 0.910 | 5.96 | 337 |
| $10.0$ | 0.8575 | 1.97 | 98 |

**怎么读这张表**：

- `alpha` 增大 → 权重范数下降 → 训练准确率下降，这是容量被压小的直接证据。
- 但 `n_iter_` 不是单调的：$10^{-3}$ 撞上限、$10^{-5}$ 反而在 1742 轮收敛。**正则强度与优化难度是两个轴**，不要用「训练轮数」判断正则是否合适。
- 图 2 的官方示例给出更直观的欠拟合读数：`alpha=10` 时三行数据集的测试准确率分别掉到 .475/.400/.825（见 [[sklearn-neural-networks]] 图表核查）。图只扫到 $10^{1}$，不能外推成「$alpha=1$ 最优」。

**选择方法**：`GridSearchCV` / `RandomizedSearchCV` 在 `Pipeline` 里扫 `alpha`；同时记录训练分数与验证分数，避免只看验证分数而不知道模型是欠拟合还是过拟合。

## 求解器：先按数据规模与在线需求选

| 求解器 | 适用 | `partial_fit` | 批 | 关键默认（外部核查） | 备注 |
| --- | --- | --- | --- | --- | --- |
| `lbfgs` | 小数据、全批量 | **无** | 全批量 | `max_iter=200`、`max_fun=15000` | 来源说小数据上更快/更好（L149） |
| `adam` | 较大数据、默认 | **有** | mini-batch | `learning_rate_init=1e-3`、`beta_1=0.9`、`beta_2=0.999`、`epsilon=1e-8` | 来源说大范围稳健（L149） |
| `sgd` | 愿意调学习率、需要 momentum | **有** | mini-batch | `learning_rate='constant'`、`momentum=0.9`、`nesterovs_momentum=True` | 来源说调好学习率可优于另两者（L149） |

三条硬约束：

1. **默认是 Adam**（外部核查 1.9.1 签名），但来源建议小数据用 L-BFGS。小数据上要**显式**写 `solver='lbfgs'`。
2. **L-BFGS 没有 `partial_fit`**：本机实测 `hasattr(model, 'partial_fit') == False`。需要在线/小批量就只能选 SGD/Adam（L125–L129）。
3. **L-BFGS 的迭代上限是 `max_iter`，函数调用上限是 `max_fun`**；`n_iter_no_change` 只对 SGD/Adam 生效（外部核查 API 页/源码）。不要拿 L-BFGS 去调 `n_iter_no_change`。

**本库综合**：求解器不是「精度旋钮」，而是**训练契约旋钮**。它决定数据怎么进入、能不能增量、以及优化问题的数值路径。先按「数据是否分块到达、规模是否小到能全批量」选求解器，再调学习率/轮数。

## 停止条件与早停

外部核查 1.9.1 的默认值：

- `max_iter=200`；
- `tol=1e-4`；
- `n_iter_no_change=10`；
- `early_stopping=False`；
- `validation_fraction=0.1`（仅早停开启时使用）。

因为 `early_stopping=False` 是默认，**`max_iter=200` 是硬上限**。本机实测 1.6.1：

- 1000×20 二分类、默认参数 → `n_iter_=200` 并抛 `ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached...`；
- 同一数据、`(50,)`、`max_iter=500`、`n_iter_no_change=10`：`early_stopping=False` → `n_iter_=395`；`True` → `n_iter_=34`，`validation_scores_` 长度 34，`best_validation_score_=0.94`；
- `tol` 控制停止：`tol=1e-7` → 2000 轮（撞上限），`tol=1e-4` → 395，`tol=1e-1` → 12。

**怎么用**：

1. 先设一个足够大的 `max_iter`（例如 500–2000，视数据而定），观察 `ConvergenceWarning` 是否消失；不要用默认 200 就下结论。
2. 如果训练损失早就走平，`n_iter_no_change` 会在 SGD/Adam 上提前停；没有早停时它看**训练损失**，开启早停时看**验证分数**（外部核查 1.6.1 源码）。
3. 如果开启 `early_stopping=True`，`validation_fraction=0.1` 是从**训练集内部**再切 10%，不是测试集。小数据上这 10% 可能只有几十个样本，早停分数会很吵；此时要么增大比例，要么用外部 CV 选择轮数（本库判断）。
4. `partial_fit` + `early_stopping=True` 会直接 `ValueError`（外部核查源码）；在线学习与早停不能同时用。

## 可复现性与种子

官方丢失的优缺点折叠块明确写：不同随机权重初始化会导致不同验证准确率（见 [[sklearn-neural-networks]] 转换损失）。本机实测 1.6.1：

- 同一 `random_state=7` 两次拟合 → `coefs_` 逐元素完全相等；
- 同一数据、8 个种子（0–7）→ 验证准确率 0.9367–0.9533，极差 0.0167；
- 因此**单次分数不能代表配置的稳定水平**。

`random_state` 通过同一个 RNG 控制初始化、`shuffle` 与早停验证划分；三者共享序列，改求解器或开关早停会改变 RNG 的消费顺序。所以「固定 `random_state`」只能保证同一条代码路径可复现，不能把不同配置的差异归因于模型本身。

**报告建议**（本库判断）：

1. 固定 CV 折（`StratifiedKFold(random_state=...)` 或固定 `random_state` 的 `cross_val_score`）；
2. 每个配置跑 3–5 个 `random_state`，报告均值与极差/标准差；
3. 比较两个配置时，用**同一批折、同一批种子**；
4. 如果种子间极差大于配置间差异，先增加种子数，而不是宣布胜者。

## 多输出与多标签

| 任务 | 写法 | 输出形状 | 注意 |
| --- | --- | --- | --- |
| 多分类 | `y` 为类别标签 | `predict_proba` $(n,K)$ | softmax，各类和为 1 |
| 多标签 | `y` 为 0/1 指示矩阵 | `predict` $(n,K)$、`predict_proba` $(n,K)$ | 每个类独立 logistic；0.5 硬阈值 |
| 多输出回归 | `y` 为 $(n,t)$ | `predict` $(n,t)$ | 输出层 identity；默认平方误差 |

本机实测：

- 多标签 `y=[[0,1],[1,1]]`、`(15,)`、L-BFGS → `predict([[1.,2.]])` 返回 `[[1,1]]`，`predict([[0.,0.]])` 返回 `[[0,1]]`；`classes_=[0,1]`、`n_outputs_=2`、`out_activation_='logistic'`。
- 多输出回归 3 维输入、2 维输出 → `coefs_` 形状 `[(3,20),(20,2)]`、`predict` 形状 `(n,2)`、`out_activation_='identity'`。

**注意**：多标签的 `predict` 固定按 0.5 二值化，没有逐类阈值参数；需要别的阈值就自己拿 `predict_proba` 切。`predict_proba` 也不是校准概率——MLP 没有内置校准，可靠性图/校准器在未入库的 `sklearn.calibration`。

## 推荐顺序（从零到一）

1. **跑基线**：线性/逻辑回归 + 树/GBDT 各一个。MLP 只有在基线明显欠拟合时才值得引入（本库判断）。
2. **建管线**：`Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(...))])`；缩放只对训练折 fit。
3. **定架构**：先一个隐藏层，宽度从小于默认 100 的量级试起；用 `hidden_layer_sizes=()` 做「线性输出」对照。
4. **定求解器**：小数据先 `lbfgs`；大数据/在线先 `adam`；需要 momentum 再试 `sgd`。不要用默认 Adam 代替这个判断。
5. **定 `alpha`**：在 `10**-np.arange(1,7)` 上扫，记录训练/验证分数与权重范数。
6. **定停止**：把 `max_iter` 放大到不再出现 `ConvergenceWarning`；再决定是否开早停、`tol`/`n_iter_no_change` 怎么设。
7. **种子扫描**：每个候选配置 3–5 个种子，报告均值与极差。
8. **最终评估**：在固定折上做 CV；如果要做概率输出，另做校准；如果要做解释，另做 permutation importance/部分依赖。

## 18 条坑

1. **不缩放**：第一层直接吃量级；实测平均 0.788 vs 0.889（缩放后）。
2. **缩放泄漏**：在全部数据上 `fit` scaler 再 CV；必须放进 `Pipeline`，只对训练折 fit。
3. **默认 Adam 跑小数据**：来源建议小数据 L-BFGS；默认不会替你换。
4. **把默认 `(100,)` 当最优架构**：它只是默认；来源建议从小网络开始（L133）。
5. **`max_iter=200` 撞上限**：默认不早停，200 是硬上限；`ConvergenceWarning` 不是装饰。
6. **`alpha` 过大**：权重范数被压得很小，训练/验证都欠拟合（图 2 的 `alpha=10` 行）。
7. **`alpha` 过小 + 大网络**：训练准确率高、验证差距大；先加正则/早停/数据，不要只加轮数。
8. **以为 L-BFGS 能 `partial_fit`**：对象没有该方法；在线学习只能用 SGD/Adam。
9. **把 `n_iter_` 当累计轮数**：`warm_start` 循环里每次 `fit` 都归 1；要看权重是否变化。
10. **早停用了测试集**：`validation_fraction` 是从训练集内部再切，不是外部测试集。
11. **小数据上 `validation_fraction=0.1` 太小**：验证集可能只有几十个样本，早停分数噪声大；这是工程判断，来源未给最小值。
12. **多标签当多分类**：softmax 强制互斥；多标签必须让每个类独立过 logistic。
13. **把 0.5 当可调阈值**：`predict` 固定 0.5；需要别的阈值就自己切 `predict_proba`。
14. **把 `predict_proba` 当校准概率**：MLP 没有内置校准；需要可靠性图/校准器。
15. **找不到 `feature_importances_`**：MLP 不提供；解释要走 permutation importance/部分依赖（未入库）。
16. **以为 `alpha` 罚了偏置**：实现只罚 `coefs_`，不罚 `intercepts_`。
17. **以为 `MLPRegressor` 只有平方误差**：1.7 起有 `loss='poisson'`，但用户指南 1.17.3 只写平方误差。
18. **忽略种子方差**：同一配置换 `random_state` 可能差 1–2 个百分点（实测极差 0.0167）；单次分数会高估结论强度。

## 可自动化程度

| 步骤 | 可自动化 | 说明 |
| --- | --- | --- |
| 缩放 | **高** | `Pipeline` + `StandardScaler`，无判断 |
| 架构搜索 | **中** | 可网格/随机搜索，但每次拟合成本高；需要早停/预算控制 |
| `alpha` 搜索 | **高** | 一维网格，`GridSearchCV` 直接扫 |
| 求解器选择 | **中** | 可按数据规模/是否在线规则化，但「哪个更快更好」需实测 |
| 早停 | **高** | 一个布尔 + `validation_fraction`；但小数据上噪声大 |
| `tol`/`n_iter_no_change` | **中** | 可扫，但收益通常小于架构/`alpha` |
| 种子扫描 | **高** | 固定折 + 多 `random_state`，完全可脚本化 |
| 概率校准 | **低** | 需要额外数据/校准器；不在本节范围 |
| 特征重要性/解释 | **低** | 需要 permutation importance/部分依赖；不在本节范围 |
| 与基线的取舍 | **低** | 需要业务/成本判断；不是纯分数问题 |

## 关联页面

- 来源：[[sklearn-neural-networks]]
- 概念：[[multi-layer-perceptron]]（机制、非凸性、三个求解器）、[[regularization-and-sparsity]]（`alpha` 作为 $ℓ_2$ 的矩阵版）、[[support-vector-machines]]（缩放与核方法对照）、[[nearest-neighbors]]（非参数 vs 参数化）、[[decision-trees]]（轴对齐 vs 学到的表示）、[[ensembles]]（是否把 MLP 放进集成）
- 工具：[[scikit-learn]]（`neural_network` 模块的默认值与版本边界）
- 实践：[[linear-models-as-baselines]]（先跑线性基线）、[[svm-selection-and-tuning]]（缩放是硬前提）、[[nearest-neighbors-selection-and-tuning]]（缩放即模型定义）、[[ensembles-selection-and-tuning]]（基学习器的成本与方差）、[[competition-ml-workflow]]（MLP 在竞赛流程里的位置）
