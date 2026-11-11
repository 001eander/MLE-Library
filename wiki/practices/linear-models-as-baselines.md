---
title: "把线性模型当基线：预处理约束、正则强度选择与常见坑"
type: practice
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - linear-model
  - regularization
  - baseline
  - workflow
sources:
  - "[[sklearn-linear-models]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.x – 1.9.x（推断，见 [[sklearn-linear-models]] 的版本判定）
  - 表格型中小规模数据上的线性/广义线性基线
---

# 把线性模型当基线

本页回答一个具体问题：**在一条 MLE 流水线里，线性模型应该放在什么位置、要满足什么前提、哪些默认值会骗你**。事实来自 [[sklearn-linear-models]]，引用写作 `[[sklearn-linear-models]], L146`；凡是本库自己的推断都显式标注。

## 为什么值得先跑线性模型

- **成本有闭式下界**：OLS 走 SVD，$O(n_{\text{samples}} n_{\text{features}}^2)$（L58）。这是绝大多数特征工程实验的"零成本对照"。
- **正则强度几乎免费**：`RidgeCV` / `LassoCV` / `ElasticNetCV` / `LogisticRegressionCV` 把交叉验证做进了估计器（L122、L200、L517）。一次 `fit` 就得到"该模型在这个特征集上的最好成绩"，不用手写搜索循环。
- **可解释且有确定性**（除 RANSAC 与 SGD 家族外）：系数直接可读，同数据同参数结果可复现。这在智能体场景里很重要——**基线不稳定时，后续所有比较都失去意义**（**本库综合**）。
- **它们是唯一有"内置验证"的估计器族**。这一点决定了它们在流水线里的用途：不是"最后可能赢的模型"，而是**给特征集和切分方式定标尺的模型**。

来源对线性模型的定位也一致：[[kaggle-getting-started]] 把 Linear/Logistic Regression 列为「性能稍逊但适合作 Base Model」（`raw/articles/Kaggle 入门指南.md:103`），本来源则给出这些 Base Model 的当代用法。

## 三条硬约束

### 1. 尺度：$ℓ_1$/$ℓ_2$ 惩罚与一阶求解器都吃尺度

文档里有两条直接证据：`sag`/`saga` 在表格中标记为「Robust to unscaled datasets: no」（L499）；`lbfgs` 的「performance suffers on poorly scaled datasets and on datasets with one-hot encoded categorical features with rare categories」（L938）。

**本库综合**：惩罚项直接作用在系数上，因此特征尺度变化会改变"哪个特征更贵"。未标准化时，$ℓ_1$ 会优先砍掉数值量级小的特征，而不是信息量小的特征。这不是文档写的结论，而是从目标函数（L146）直接推得的；文档只对求解器的数值行为作了说明。

### 2. 共线性：四种模型四种反应

| 模型 | 对近似共线的反应 | 依据 |
| --- | --- | --- |
| OLS | 设计矩阵接近奇异 → 估计对目标噪声极敏感、方差大 | L40 |
| Ridge | 加大 `alpha` 收缩，系数「更稳健于共线性」 | L70 |
| Lasso | 相关特征中「大概率领随机选一个」 | L247 |
| ElasticNet | 「倾向于都选」，并继承 Ridge 的部分旋转稳定性 | L247、L249 |
| MultiTask 系列 | 按任务共享支撑，非零是整列 | L231、L235 |

**直接后果（本库综合）**：用 Lasso 的系数做"哪个特征重要"的结论时，相关特征组内的选择是**不稳定**的——换种子或换折就可能换特征。要判断"这组特征有没有用"，看 EN 或看整组的 CV 边际收益，而不是看 Lasso 选中了哪一个。

同理，哑变量全保留（dummy variable trap）会让 OLS 奇异；Ridge 能容忍（收缩救回），Lasso 会在组内随机选一个。这一点来源未讨论，属**本库综合**。

### 3. `fit_intercept`

多项式扩展示例里显式用 `LinearRegression(fit_intercept=False)`（L834）——因为 `PolynomialFeatures` 已经生成了常数项列。**本库综合**：两者同时开启会造成常数列与截距完全共线，系数不可解释（模型仍能拟合）。同理，`liblinear` 会惩罚截距（L499），特征未中心化时截距被拉小，表现会与其它求解器不同。

## 正则强度怎么选：三条路，三种代价

同一件事（选 `alpha` / `C`）有三条路径，代价与失效条件都不同：

| 路径 | 估计器 | 代价 | 失效条件（来源明说） |
| --- | --- | --- | --- |
| 内置 K 折 CV | `LassoCV`、`ElasticNetCV`、`LogisticRegressionCV` | 路径算 $k+1$ 次 | 依赖 CV 切分正确（本库补充：与 [[leaderboard-vs-cv]] 同一前提） |
| 内置 LOO | `RidgeCV`、`RidgeClassifierCV` | 便宜，「efficient Leave-One-Out」 | `alpha` 不能为 0（L122）；LOO 的方差特性见下 |
| 信息准则 | `LassoLarsIC`（AIC/BIC） | 路径只算一次 | 需要正确的自由度估计、大样本渐近、真实模型在候选集内、**病态问题（特征多于样本）上往往失效**（L212） |

关于第二条路的**本库综合**：LOO 不是"更好的 K 折"。它每次只用 1 个样本做验证，单次估计方差大；文档只说了它"efficient"（能解析计算），没有说它更准。要用 K 折就显式传 `cv`（L137）。

关于第三条路的**本库综合**：AIC/BIC 在**训练集内**计算（L210），选的是"离散路径上的第几个模型"。官方示例图（`plot_lasso_lars_ic`）的横轴确实是 model sequence 0–12 而不是 alpha 数值——也就是说 alpha 的分辨率由路径网格决定，不是连续的。

### 一个容易忽略的顺序问题

`RidgeCV` 的 alpha 网格示例是 `np.logspace(-6, 6, 13)`（L129）。**本库综合**：正则强度的量级与样本量、特征数、是否标准化强相关，所以 (a) 先标准化再扫 alpha；(b) 换预处理就重扫 alpha，不要沿用上一次的最优值；(c) 跨估计器比较时不要复用同一个 alpha 值（见下节第 1 条）。

## 五个坑

1. **`alpha` 不可跨估计器比较**。Ridge 的目标是 $\lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_2^2$（L67），Lasso/ElasticNet 是 $\frac{1}{2n}\lVert Xw-y\rVert_2^2 + \alpha\lVert w\rVert_1 + \dots$（L146、L256）。同样数值的 `alpha` 在两者中对应的"惩罚/损失"比例差了 $n$ 的量级。跨模型调参必须各自重扫（**本库综合**；文档只说 `alpha` 与 SVM 的 `C` 的换算是 `1/C` 或 `1/(n_samples*C)`，"取决于估计器与目标函数"，L227）。
2. **`class_weight` 会改有效正则**。目标除以 $S=\sum_i s_i$、正则除以 $SC$，所以「样本权重乘以 $b$ 等价于把 `C` 乘以 $b$」（L484）。调类别权重和调 `C` 是同一个旋钮的两个方向；只调其中一个会以为自己是"两个独立的超参"。
3. **`liblinear` 的两个偏差**：惩罚截距（L499 标 "bad"）；`fit_intercept=False` 且 `decision_function` 恰为 0 时预测的类与 sklearn 封装相反（L505）。当样本大量落在决策边界上时，这类模型本身已经欠拟合——文档的建议是把 `fit_intercept=True` 并调 `intercept_scaling`（L505）。
4. **`Lasso` 的 `tol` 是 duality gap**（L190），与"系数变化量"不是一回事。特征尺度差异大时，同一个 `tol` 对应的实际精度不同（**本库综合**）。
5. **`QuantileRegressor` 走线性规划**（`scipy.optimize.linprog`，L794），成本随样本数增长远快于 OLS（**本库综合**，文档未给复杂度）。想要预测区间又在乎速度时，文档给的另一条路是 `GradientBoostingRegressor(loss="quantile", alpha=τ)`（L792）。

## 稳健回归：一句话规则与它自己的反例

来源的取舍建议（L649–L662）：`HuberRegressor` 除样本极多外最快；默认参数下 Theil-Sen 与 RANSAC 都不如 Huber 稳健；RANSAC 擅长 y 方向大离群点、Theil-Sen 擅长 X 方向中等离群点；**「When in doubt, use RANSAC」**（L662）。

但官方自己的三张示例图给出了条件性反例（**本库综合**，读数取自 `raw/assets/sklearn-linear-models/` 中 `sphx_glr_plot_robust_fit_00{2,3,5}.png` 图例，均已目视核查）：

| 场景 | OLS | Theil-Sen | RANSAC | Huber |
| --- | --- | --- | --- | --- |
| X 方向小偏差 | 0.003 | 0.002 | 0.003 | 0.002 |
| y 方向小偏差 | 1.009 | **0.034** | 0.328 | **0.011** |
| y 方向大偏差 | 11.055 | 0.234 | **0.002** | 0.011 |

读法：(a) y 方向的离群点对 OLS 是灾难（误差高两个量级）；(b) **离群点不大时 RANSAC 明显差于 Theil-Sen 与 Huber**（0.328 vs 0.034/0.011）；(c) 只有偏差变大时 RANSAC 才反超（0.002）。所以「拿不准用 RANSAC」在"离群点幅度小"的情形与官方示例相左——**先看离群点幅度，再选估计器**。

另一条成本读数：同一张 Theil-Sen 示例图里标注 `OLS 0.00s / Theil-Sen 0.15s / RANSAC 0.00s`（`sphx_glr_plot_theilsen_001.png`）。而 Theil-Sen 的复杂度按 $\binom{n_{\text{samples}}}{n_{\text{subsamples}}}$ 增长（L720–L726），大数据上不可行。

## 成本概览

| 估计器 | 成本来源 | 数量级 |
| --- | --- | --- |
| OLS / Ridge（`cholesky`） | SVD 或 Cholesky | $O(n p^2)$，$n \ge p$（L58） |
| Lasso / ElasticNet（坐标下降） | 迭代 × 特征数；可用 gap safe screening 剪枝 | 未给复杂度（L176–L192） |
| LARS / LassoLars | 路径一次算完 | 与 OLS 同阶（L118，经外部核查补全） |
| OMP | 每步正交投影 | 未给 |
| Theil-Sen | 子样本组合数 | $\binom{n}{n_{\text{subsamples}}}$（L720） |
| RANSAC | `max_trials` × 每次拟合 | 未给 |
| QuantileRegressor | 线性规划 | 未给（L794） |
| `*CV` 系列 | 上述成本 × (k+1) | 见「三条路」表 |

## 对智能体的可自动化程度

**本库综合**，与 [[competition-ml-workflow]] 的分阶段表同一视角：

| 环节 | 可自动化程度 | 依据与风险 |
| --- | --- | --- |
| 跑线性基线 | **高** | 一次 `fit`，无随机性（除 SGD/RANSAC） |
| 预处理 | 中 | 标准化必须**在 CV 内 fit**，否则泄漏；这是 agent 最常犯的错（本库判断，非来源论述） |
| alpha / C 选择 | **高** | 内置 CV 一步到位；但网格量级依赖预处理，换特征就得重扫 |
| 稳健回归选型 | 低—中 | 官方一句话规则与自家示例有冲突（见上），需要看离群点幅度 |
| 判断"线性够不够" | 低 | 文档不提；需要与非线性模型的 CV 对照 |

由此得到一条可操作建议（**本库综合**）：**把线性模型当作"特征集评估器"而不是"最终模型"**。它廉价、确定、自带 CV，适合回答"这批特征有没有信号"；一旦特征集确定，再谈树模型/集成是否值得多花算力。

## 待解问题

- **`alpha` 与 $n$ 的不一致是否被实现层补偿？** Ridge 目标无 $1/n$、Lasso 有（L67 vs L146）。若实现层没有额外缩放，则"跨估计器 alpha 不可比"成立；若实现层做了归一，则该结论要弱化。需要读源码或做实验（见 [[sklearn-linear-models]] 待解问题 1）。
- **标准化前后最优 `alpha` 的换算关系**有闭式吗？若有，agent 可以省掉重扫；文档未提。
- **`RidgeCV` 的 LOO vs K 折在真实数据上的差异有多大**？文档只说 LOO "efficient"，没给方差比较。
- **Theil-Sen 在小样本上的成本**是否可以接受？官方示例 0.15s 是单一数据规模下的读数。
- 线性模型与 GBDT 的 CV 差值在什么数据规模上才值得转向非线性模型？本库无来源，需要外部证据。

## 关联页面

- 来源：[[sklearn-linear-models]]、[[kaggle-getting-started]]
- 工具：[[scikit-learn]]
- 概念：[[regularization-and-sparsity]]、[[leaderboard-vs-cv]]
- 实践：[[competition-ml-workflow]]（七段流程与阶段自动化程度）、[[evaluator-design-and-reward-hacking]]（验证切分与代理指标）
