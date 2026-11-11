---
title: "scikit-learn 用户指南 1.1「Linear Models」：估计器全家谱与求解器约定（stable 文档剪藏）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - scikit-learn
  - linear-model
  - regularization
  - reference-doc
source_path: raw/articles/sklearn/1.1. Linear Models.md
source_kind: article
status: draft
verified: 2026-11-11
---

# scikit-learn 用户指南 1.1「Linear Models」

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | Linear Models（scikit-learn 用户指南第 1.1 节） |
| 作者 | scikit-learn developers（剪藏未保留署名；官方页页脚为 `© Copyright 2007 - 2026, scikit-learn developers (BSD License)`，**本库核查所得，非剪藏内容**） |
| 形态 | 官方用户指南的 HTML→Markdown 剪藏；**不是论文、不是基准报告**；正文含可执行的 doctest 片段 |
| 原文 | <https://scikit-learn.org/stable/modules/linear_model.html>（剪藏的 frontmatter `source` 字段） |
| 本地文件 | `raw/articles/sklearn/1.1. Linear Models.md`（980 行，约 86.1 KB；含所有者手工补入的 1.1.14 与图书馆员回填的三处公式） |
| SHA-256 | `03324a207556b360212a8de47b0bbbe74c746bc65be19144602c145c71f2fcd1`（**修正版**，非原始剪藏；修正明细见「转换损失与外部核查」第 3 条） |
| 完整性 | **残缺**：官方正文的 23 个折叠块（29.3%）全部缺失。本页把每一处缺口都标注出来，不把补写内容冒充剪藏原文 |
| 剪藏时间 | frontmatter `created: 2026-11-11` |
| 文档版本 | **剪藏未记录**。外部核查（2026-11-11）：官方 `stable` 现为 **1.9.1**；本页内容同时匹配 `1.8.0` 与 `1.9.1` 的结构（41 个 `1.1.x` 标题逐一相同，含 1.1.3.1 gap safe screening），而 `1.7.2` / `1.6.1` 没有该小节、也没有 `np.float64(...)` 形式的 doctest 输出。因此判定为 **≥ 1.8**，补丁版本无法确定（**本库推断**，核查方法见下节） |
| 插图 | 18 个不同文件（正文中多处重复引用），全部为官方示例图外链；已抓取入库 `raw/assets/sklearn-linear-models/` 并**逐张目视核查** |
| 代码 | doctest 片段随文档版本变化（例如 `reg.intercept_` 返回 `np.float64(0.13636)`，这是 NumPy ≥ 2 的 repr） |

入库说明：本页依据本地剪藏撰写，引用给到章节号与行号（`[[sklearn-linear-models]], L146`）。剪藏是**残缺的转换产物**（23 个折叠块全缺、内部引用文字丢失、公式噪声化），凡涉及缺口处的论断均按「外部核查」单独标注来源，不冒充剪藏内容。

## 概要

这是 scikit-learn 用户指南中最"重"的一节：它把 `sklearn.linear_model` 的全部估计器按**目标函数**排开，并给出每个估计器的复杂度、求解器、正则化参数选择方式与已知失效条件。文档不是教程，写法一律是「目标函数 → 实现 → 超参 → 适用条件」。

对本库的意义有三层：

1. **它是第一个"现行 API 参考"类来源**。此前六个来源（[[aide]]、[[alphaevolve]]、[[autoresearch]]、[[hyra-1-0]]、[[hyra-2-0]]、[[kaggle-getting-started]]）都在谈**怎么组织研究、怎么判断分数可信**；本来源补的是"具体工具此刻的行为"。[[kaggle-getting-started]] 的 sklearn 片段停留在 2016 年的 `sklearn.grid_search` / `cross_validation`，本来源给出的是这些 API 的当代形态（[[kaggle-getting-started]], L120、L143、L208）。
2. **它给出的是"默认值"而不是"最佳实践"**。文档反复说明默认行为成立的条件（如 logistic 默认带正则、`Ridge` 默认 `solver="auto"`、`RidgeCV` 默认 LOO），这些默认值正是智能体最容易沉默继承的东西。
3. **它自己承认有一大类问题是线性模型不解决的**：高维稳健拟合「very hard，本节模型大概率不适用」（L647）。

## 文档结构

按剪藏的实际章节（行号为本文件行号）：

| 章节 | 解决的问题 | 行号 |
| --- | --- | --- |
| 引言 | 全模块约定：`coef_` = $w$，`intercept_` = $w_0$ | L10–L18 |
| 1.1.1 Ordinary Least Squares | 无惩罚最小二乘、非负约束、SVD 复杂度 | L20–L58 |
| 1.1.2 Ridge regression and classification | $ℓ_2$ 收缩、`solver="auto"` 选择顺序、`RidgeClassifier`、`RidgeCV` 的 LOO | L60–L137 |
| 1.1.3 Lasso | $ℓ_1$ 稀疏、坐标下降 + 软阈值、gap safe screening、三种 alpha 选择法 | L139–L227 |
| 1.1.4–1.1.6 Multi-task Lasso / Elastic-Net / Multi-task Elastic-Net | 共享支撑（列稀疏）、$ℓ_1$+$ℓ_2$ 混合 | L229–L278 |
| 1.1.7–1.1.9 LARS / LARS Lasso / OMP | 分段的精确路径解、$ℓ_0$ 约束的贪心 | L280–L339 |
| 1.1.10 Bayesian Regression | 把正则参数当随机变量、ARD 的逐系数精度 | L341–L429 |
| 1.1.11 Logistic regression | 二值/多类损失、`C` 与 `l1_ratio`、六个求解器的能力矩阵 | L431–L517 |
| 1.1.12 Generalized Linear Models | 连接函数 + 单位偏差、Tweedie `power` 与分布的对应 | L519–L592 |
| 1.1.13 SGD | 在线/out-of-core 学习、`Perceptron` 是 SGD 的包装、PA-I/PA-II | L594–L618 |
| 1.1.14 Robustness regression | 离群点与模型误差：RANSAC / Theil-Sen / Huber 的取舍 | L620–L798 |
| 1.1.15 Quantile Regression | 条件分位数、pinball loss、基于线性规划 | L800–L814 |
| 1.1.16 Polynomial regression | 基函数扩展、`PolynomialFeatures`、`interaction_only` 解 XOR | L800–L875 |
| 脚注 | AIC/BIC 公式、LassoLarsIC 的噪声方差估计、ARD 原文重复、solver 说明 | L877–L981 |

## 关键主张

以下每条都注明是**来源论述**还是**本库综合**。

### 1. 全模块只有一套约定：目标函数定义估计器

来源开宗明义：整个模块用 $w$ = `coef_`、$w_0$ = `intercept_`（L16），之后每个估计器都先写目标函数再写用法。这不是修辞：`Ridge` 的目标是 $\min_w \|Xw-y\|_2^2 + \alpha\|w\|_2^2$（L67），**没有** $1/n$ 因子；`Lasso` 是 $\min_w \frac{1}{2n_{\text{samples}}}\|Xw-y\|_2^2 + \alpha\|w\|_1$（L146），**有**。两者对同一个 $\alpha$ 的含义因此不同（见 [[regularization-and-sparsity]]）。

### 2. OLS 的方差来自共线性，复杂度来自 SVD

> The coefficient estimates for Ordinary Least Squares rely on the independence of the features. … the least-squares estimate becomes highly sensitive to random errors in the observed target, producing a large variance.（L40）

求解走 $X$ 的 SVD，$X$ 形状 `(n_samples, n_features)` 时为 $O(n_{\text{samples}} n_{\text{features}}^2)$，前提 $n_{\text{samples}} \ge n_{\text{features}}$（L58）。非负最小二乘用 `positive=True` 切换（L50）。

### 3. Ridge 一族的三个实用约定

- `alpha` 越大收缩越强、对共线性越稳（L70）。
- `solver="auto"` 按固定顺序挑求解器：`positive=True` → `lbfgs`；$X$ 非稀疏 → `cholesky`；否则 → `sparse_cg`（L78–L84 的表格）。
- `RidgeClassifier` 把二值标签转成 $\{-1,1\}$ 后回归，预测取符号；多类时视为多输出回归，取最大输出（L104）。来源给出的理由很具体：它「可以只算一次投影矩阵 $(X^TX)^{-1}X^T$」，因此在类别很多时可能显著快于 `LogisticRegression`，而实践中两者的 CV 分数（accuracy / precision-recall）可以接近（L106、L108）。
- `RidgeCV` / `RidgeClassifierCV` 的默认交叉验证是 **Leave-One-Out**，且「alpha cannot be 0 due to the formulation used to calculate Leave-One-Out error」；显式给 `cv` 时才退化为 `GridSearchCV` + K 折（L122、L137）。

### 4. Lasso 一族的稀疏机制与 alpha 选择

坐标下降把问题降成一维，解是软阈值函数

$$w_j = \frac{S\left(x_j^{T}\left(y - X_{-j}w_{-j}\right), \alpha\right)}{\|x_j\|_2^2}, \qquad S(z,\alpha) = \operatorname{sign}(z)\max\left(0, |z| - \alpha\right)$$

（L181–L190；公式已按官方页恢复为规范 LaTeX）。两个直接推论来源都写了：**软阈值在 $\alpha \ge |z|$ 时恰好为 0**（L190），**停止条件是 duality gap 小于 `tol`**（L190）；特征扫描顺序是 `selection="cyclic"` 或 `"random"`（L190），gap safe screening 则在优化过程中安全地把必然为零的特征剔出（L192）。

alpha 的三条选法（L198–L227）：

| 方法 | 估计器 | 出处 | 代价与前提 |
| --- | --- | --- | --- |
| K 折交叉验证 | `LassoCV` | L200–L202 | 高维且特征强共线时「most often preferable」 |
| K 折交叉验证（LARS 路径） | `LassoLarsCV` | L200–L202 | 能探索更相关的 alpha 取值；**样本极少、特征很多时常常更快** |
| AIC / BIC | `LassoLarsIC` | L206–L219 | 路径只算一次（vs K 折的 k+1 次），但「需要正确的自由度估计、来自大样本渐近结果、假设真实模型在候选集中，且在病态问题（如特征多于样本）上往往会失效」 |

另有两个换算关系：AIC/BIC 的默认噪声方差估计只在 `n_samples > n_features` 时有效（脚注 [^2]，L899）；`alpha` 与 SVM 的 `C` 的关系是 `alpha = 1 / C` 或 `alpha = 1 / (n_samples * C)`，**取决于估计器与目标函数**（L227）。

### 5. 结构化稀疏与"精确路径"是两种不同卖点

- `MultiTaskLasso`：`y` 是 `(n_samples, n_tasks)`，约束「所选特征对所有任务相同」→ 非零是**整列**而不是散点（L231、L235）；典型用法是时序任务，要求某特征在所有时刻都活跃（L241）。
- `ElasticNet`：$ℓ_1$+$ℓ_2$ 凸组合由 `l1_ratio` 控制；当多个特征彼此相关时「Lasso 可能随机选其中一个，elastic-net 倾向于都选」（L247）；额外好处是继承 Ridge 的部分旋转稳定性（L249）。
- `LARS`：每步找与残差最相关的特征，相关相等时沿等角方向前进；优点是特征数远多于样本时数值高效、给出**分段线性**的完整路径；缺点是基于残差迭代重拟合，对噪声特别敏感（L282–L296，后者的依据是 Weisberg 在 Efron et al. 2004 讨论部分的意见，本库未读原文）。
- `LassoLars` 与坐标下降版的差别：前者给**精确解**、且解是系数范数的分段线性函数（L300）。
- `OMP`：在 $ℓ_0$ 约束（非零个数上限）或误差约束下做前向贪心，每步用正交投影重算残差（L321–L337）。

### 6. 贝叶斯回归与 GLM：把"调参"改成"估计"

- `BayesianRidge`：把 $\alpha$（噪声精度）与 $\lambda$（系数精度）当随机变量，通过在 fit 中最大化**对数边缘似然**联合估计；超先验是 gamma 分布，默认四个超参全为 $10^{-6}$（L364–L378）。来源给的定性好处是「对 ill-posed 问题更稳健」（L403）。
- `ARDRegression`：把球形高斯先验换成**中心椭圆**高斯，每个系数有自己的精度 $\lambda_i$ → 系数更稀疏（L413–L421）；文献别名 Sparse Bayesian Learning / Relevance Vector Machine（L423）。
- GLM：预测值经逆连接函数 $h$ 与线性组合相连，平方损失换成指数族的单位偏差 $d$（L527、L537–L550）；分布选择给的是**目标域**规则：计数/相对频率 → Poisson + log；正值且偏斜 → Gamma + log；比 Gamma 更重尾 → Inverse Gaussian；概率 → Bernoulli（logit）/ Categorical（softmax）（L558–L561）。
- `TweedieRegressor` 的 `power` 映射：0 = Normal，1 = Poisson（`PoissonRegressor` 与之**严格等价**，`link='log'`），2 = Gamma（`GammaRegressor` 同理），3 = Inverse Gaussian（L569–L572）。

### 7. logistic 回归的求解器矩阵与两个"沉默改变正则强度"的开关

来源明确：logistic 默认带正则（L440）——「No regularization amounts to setting C to a very high value」。四个惩罚项由 `C` 与 `l1_ratio` 表达（L473–L479），二值目标是

$$\min_w \frac{1}{S}\sum_{i=1}^{n} s_i\left(-y_i\log\hat p(X_i) - (1-y_i)\log(1-\hat p(X_i))\right) + \frac{r(w)}{SC}$$

其中 $s$ 是 `sample_weight` 与 `class_weight` 的逐元素乘积，$S=\sum_i s_i$（L468–L471）。由此来源直接给出一个反直觉结论：

> multiplying the sample weights by a constant $b > 0$ is equivalent to multiplying the (inverse) regularization strength `C` by $b$.（L484）

求解器能力矩阵（L499 的 HTML 表，已整理）：

| | lbfgs | liblinear | newton-cg | newton-cholesky | sag | saga |
| --- | --- | --- | --- | --- | --- | --- |
| L2 | 是 | 是 | 是 | 是 | 是 | 是 |
| L1 | 否 | 是 | 否 | 否 | 否 | 是 |
| Elastic-Net | 否 | 否 | 否 | 否 | 否 | 是 |
| 无惩罚 | 是 | 否 | 是 | 是 | 是 | 是 |
| multinomial 多类 | 是 | 否 | 是 | 是 | 是 | 是 |
| **惩罚截距**（"bad"） | 否 | **是** | 否 | 否 | 否 | 否 |
| 大数据集更快 | 否 | 否 | 否 | 否 | 是 | 是 |
| 对未缩放数据鲁棒 | 是 | 是 | 是 | 是 | **否** | **否** |

选择建议（L501）：默认 `lbfgs`（稳健）；`n_samples >> n_features` 用 `newton-cholesky`（可到很高精度）；大数据用 `saga`（尤其高 `tol`）；更大数据可考虑 `SGDClassifier(loss="log_loss")`，更快但需更多调参。脚注补充：`lbfgs` 在未良好缩放、以及含稀有类 one-hot 特征时性能下降（L938）。

两个边角但真实的差异：`solver="liblinear"` 与外部 liblinear 库在 `fit_intercept=False`、`decision_function` 恰为 0 时预测的类相反（L505）；系数 p 值与置信区间需 statsmodels 或 bootstrap，sklearn 内没有（L515）。

### 8. 稳健回归（1.1.14）：三种估计器的取舍与 breakdown point

本节由**所有者手工补入**（官方页对应 1.1.14，剪藏原缺，见下节）。

- 概念：breakdown point = 「拟合开始丢失内点之前可以容忍的离群点比例」（L645）；高维下稳健拟合「very hard，这里的模型大概不适用」（L647）。
- 三方取舍：`HuberRegressor` 除样本数极多外应快于 RANSAC / Theil-Sen（后两者在小样本子集上拟合）；但默认参数下 Theil-Sen 与 RANSAC 的稳健性都不如 `HuberRegressor`；RANSAC 比 Theil-Sen 快且随样本数扩展更好；RANSAC 更擅长 **y 方向的大离群点**，Theil-Sen 更擅长 **X 方向的中等离群点**（高维下失效）；「When in doubt, use RANSAC」（L649–L662）。
- Theil-Sen：单变量简单线性回归的 breakdown point 约 **29.3%**（L714）；复杂度按 $\binom{n_{\text{samples}}}{n_{\text{subsamples}}}$ 增长，因此大数据上不可行（L720–L732）。
- Huber：对 `epsilon` 之外的样本施加**线性**损失而非丢弃，与 RANSAC/Theil-Sen 的区别是「不忽略离群点，而是给更小的权重」（L743）；建议 `epsilon=1.35` 以获得 95% 统计效率（L768）；与 `SGDRegressor(loss="huber")` 的两个差别是**尺度不变**（缩放 X/y 后不必重设 `epsilon`）与少样本时更省（L777–L779）。

### 9. 分位数回归与基函数扩展

分位数回归估计条件中位数/其他分位数，而 OLS 估计条件均值（L786）；它给的是**区间**，且对异方差或非正态误差也成立（L788）；实现基于 pinball loss 的线性规划，走 `scipy.optimize.linprog`（L794）。旁证：`GradientBoostingRegressor(loss="quantile", alpha=τ)` 也能给条件分位数（L792）。

多项式回归不引入新模型，只是换特征：`PolynomialFeatures(degree=d)` 把 $[x_1,x_2]$ 变成 $[1,x_1,x_2,x_1^2,x_1x_2,x_2^2]$（L825），配 `Pipeline` + `LinearRegression(fit_intercept=False)` 可以精确恢复三阶多项式系数（L850–L860）；`interaction_only=True` 只保留交互项，因此布尔特征可以线性分类 XOR（L862–L886）。

## 图表核查

18 张图于 2026-11-11 从官方 **1.8** 版本路径（`https://scikit-learn.org/1.8/_images/`）抓取，存入 `raw/assets/sklearn-linear-models/`，**逐张目视核查**。剪藏中全部为外链，未含图内信息。

| 图 | 文件 | 尺寸 | 行号 | 核查结论 |
| --- | --- | --- | --- | --- |
| OLS 拟合 | `sphx_glr_plot_ols_ridge_001.png` | 1000×500 | L27 | 双面板 Train/Test 散点 + 回归线，标题 "Linear Regression"，无残差信息 |
| Ridge 路径 | `sphx_glr_plot_ridge_path_001.png` | 640×480 | L72 | 10 个特征的系数随 alpha 变化；**x 轴 alpha 从左到右递减（$10^{-2}\to10^{-10}$）**，即左侧强收缩、右侧无收缩 |
| LassoCV 折曲线 | `sphx_glr_plot_lasso_model_selection_002.png` | 640×480 | L204 | 每折 MSE + 黑粗线（折平均）+ 竖虚线（CV 选中的 alpha ≈0.8）；标题标注 **train time 0.17s** |
| LassoLarsCV 折曲线 | `sphx_glr_plot_lasso_model_selection_003.png` | 640×480 | L204 | 同左，标题标注 **Lars train time 0.06s**，CV 选中 alpha ≈0.6 |
| 多任务支撑 | `sphx_glr_plot_multi_task_lasso_support_001.png` | 800×500 | L235 | 左 Lasso 面板非零散点、右 MultiTaskLasso 面板为贯穿所有 task 的整列 → 直接印证 L235 |
| 多任务拟合 | `sphx_glr_plot_multi_task_lasso_support_002.png` | 640×480 | L235 | Ground truth（绿）vs Lasso（蓝，两段被压成 0 的平台）vs MultiTaskLasso（黄，更贴合） |
| Lasso vs LARS 路径 | `sphx_glr_plot_lasso_lasso_lars_elasticnet_path_001.png` | 640×480 | L302 | 实线 Lasso / 虚线 LARS，小 alpha 段两者分叉，大 alpha 段重合 |
| Lasso vs ElasticNet 路径 | `sphx_glr_plot_lasso_lasso_lars_elasticnet_path_002.png` | 640×480 | L256 | 虚线 ElasticNet 路径更平滑、在更大 alpha 处成组收缩到 0 |
| AIC/BIC 选择 | `sphx_glr_plot_lasso_lars_ic_001.png` | 640×480 | L214 | 横轴是 **model sequence 0–12**（不是 alpha 数值）；AIC 与 BIC 在前 5 个模型几乎重合，之后分叉；竖虚线标出选中序号 7 |
| GLM 分布 PDF | `poisson_gamma_tweedie_distributions.png` | 1200×350 | L550 | 三联图 Poisson / Tweedie(p=1.5) / Gamma；**Poisson 与 Tweedie 在 Y=0 有点质量，Gamma 没有** → 与图注一致（L550） |
| Theil-Sen | `sphx_glr_plot_theilsen_001.png` | 640×480 | L624、L716、L973 | "Corrupt y"：OLS（青）几乎水平、Theil-Sen（橙）与 RANSAC（绿）沿主体趋势；图例含拟合耗时 **OLS 0.00s / Theil-Sen 0.15s / RANSAC 0.00s** |
| X 方向小偏差 | `sphx_glr_plot_robust_fit_002.png` | 500×400 | L634 | "Corrupt X, Small Deviants"：误差（平均绝对偏差）OLS 0.003 / Theil-Sen 0.002 / RANSAC 0.003 / Huber 0.002 |
| y 方向小偏差 | `sphx_glr_plot_robust_fit_003.png` | 500×400 | L642 | "Corrupt y, Small Deviants"：**OLS 1.009 / Theil-Sen 0.034 / RANSAC 0.328 / Huber 0.011** |
| y 方向大偏差 | `sphx_glr_plot_robust_fit_005.png` | 500×400 | L642 | "Corrupt y, Large Deviants"：**OLS 11.055 / Theil-Sen 0.234 / RANSAC 0.002 / Huber 0.011** |
| RANSAC | `sphx_glr_plot_ransac_001.png` | 640×480 | L672 | 黄色离群带把线性回归斜率拉平，RANSAC（浅蓝）沿内点趋势 → 印证 L672 的算法说明 |
| Huber vs Ridge | `sphx_glr_plot_huber_vs_ridge_001.png` | 640×480 | L745 | `epsilon` = 1 / 1.5 / 1.75 / 1.9 四条线 + ridge：**epsilon 越大越接近 Ridge**；两个极端离群点把 ridge 拟成负斜率 |
| 分位数回归 | `sphx_glr_plot_quantile_regression_002.png` | 640×480 | L790 | "Quantiles of heteroscedastic Normal distributed target"：0.05 分位数线**负斜率**、0.5 与真实均值重合、0.95 更陡 → 异方差下区间不平行 |
| 多项式基 | `sphx_glr_plot_polynomial_interpolation_001.png` | 640×480 | L806 | degree 3 / 4 / 5 与 B-spline 对比：degree 4 在两端发散、degree 5 穿过训练点但振荡，B-spline 边缘外为常数 |

抓取文件与校验（SHA-256 前 12 位，抓取源 `https://scikit-learn.org/1.8/_images/`，均返回 HTTP 200）：`poisson_gamma_tweedie_distributions` `c3267850fbfa`、`..._plot_huber_vs_ridge_001` `bba924d70af7`、`..._plot_lasso_lars_ic_001` `1120bed41109`、`..._path_001` `6e240dfcc68f`、`..._path_002` `16ea1b7f5ea6`、`..._model_selection_002` `3c33c597323a`、`..._model_selection_003` `61f3b83147cb`、`..._multi_task_lasso_support_001` `52050602f8e8`、`..._multi_task_lasso_support_002` `cb8a4274e4fc`、`..._plot_ols_ridge_001` `68164b86fe59`、`..._plot_polynomial_interpolation_001` `63bb0fe2ca86`、`..._plot_quantile_regression_002` `1ee9850dc42b`、`..._plot_ransac_001` `4ef234d3234f`、`..._plot_ridge_path_001` `94062092e4ac`、`..._plot_robust_fit_002` `4e4ba1c9141c`、`..._plot_robust_fit_003` `f1c4e243cdfb`、`..._plot_robust_fit_005` `f9cfdae4444b`、`..._plot_theilsen_001` `328cba59b7d1`。

由此得到三条**来源未写、由图读数得出**的观察（**本库综合**）：

1. **`LassoLarsCV` 比 `LassoCV` 快**在这个示例里是 0.06s vs 0.17s（图标题实测），与 L202「样本极少时常常更快」的定性说法方向一致，但示例的样本量、特征数未标注，不能外推。
2. **「When in doubt, use RANSAC」（L662）是条件性的**：官方自己的三张稳健拟合图里，y 方向**小**偏差时 RANSAC 的误差（0.328）比 Theil-Sen（0.034）和 Huber（0.011）差一个量级；只有偏差变大（`..._005`）时 RANSAC 才反超（0.002）。所以来源的建议在"离群点不大"时与自家示例相左。
3. **AIC/BIC 的横轴是模型序号**，说明信息准则选的是**离散路径上的第几个模型**；路径网格决定了 alpha 的分辨率——这是把 AIC/BIC 当"免费调参"时容易忽略的分辨率上限（**本库综合**）。

## 转换损失与外部核查

这一节是本页最重要的部分：**剪藏不是完整文档**。所有者已手工补回 1.1.14 的正文，但仍有系统性缺口。核查方法：以官方 `1.8.0` 页面为基准（与剪藏同结构），逐块比对。

### 1. 折叠块整块缺失（占官方正文 29.3%）

官方页把大量内容放在 `<details>` 折叠块里：**8 个 `Mathematical details`、7 个 `References`、`Solvers’ details`、`Practical considerations`、`Examples of use cases`、`Details of the algorithm`、`Theoretical considerations`**，共 23 个内容块、19,266 字符，占官方正文（65,844 字符）的 29.3%。**剪藏一个都没有保留**。

可验证的自证：剪藏 L210 写着「cf. to “Mathematical details” section below」，而同名的折叠块并不存在（全文中 "Mathematical details" 只出现这一次）。

缺失内容包括（举例，均为官方 1.8 内容，**不是剪藏内容**）：

| 丢失内容 | 官方所在块 | 对本库的影响 |
| --- | --- | --- |
| 多类 logistic 的完整损失与 $W_k$ 定义、$r(W)$ 的 $ℓ_1$/$ℓ_2$/EN 形式 | 1.1.11.2 `Mathematical details` | 剪藏只留「过参数化用 K 个权重向量」的定性说明（L491） |
| `l1_min_c`、`OneVsRestClassifier`、`liblinear` 不能学真多类、newton-cholesky 的内存二次依赖 | 1.1.11.3 `Solvers’ details` | 剪藏只有 L501 一段选择建议；求解器的失效细节全缺 |
| GLM 的 `Practical considerations`：$Xw$ 可能为负故需逆连接；以 counts/exposure 为目标的 Poisson 用法；Tweedie `power` 交叉验证需显式 scoring | 1.1.12.1 | 这类"上手即踩"的细则在剪藏中完全缺失 |
| GLM 的 `Examples of use cases`（降雨、保险理赔、生产中断、新闻分类等） | 1.1.12.1 | 选分布的经验映射缺失 |
| LARS 的 `Mathematical formulation`（`coef_path_` 形状 `(n_features, max_features + 1)`） | 1.1.8 | 剪藏只有性质描述 |
| 分位数回归的 `Mathematical details`（pinball loss 的显式形式、"对离群点比平方误差稳健"） | 1.1.15 | 剪藏只有一句"基于 pinball loss 的线性规划" |
| 多项式回归的 `Mathematical details`（$z=[x_1,x_2,x_1x_2,x_1^2,x_2^2]$ 的重新参数化论证） | 1.1.16 | 结论在正文，论证缺失 |
| 全部 7 处参考文献列表 | 各节 `References` | 剪藏留下 9 个空的 `References` 标题（L429、L563、L770、L897、L907、L927、L942、L979 等），文献本身没有 |

### 2. 内部交叉引用文字被吞（70 处 `std-ref`）

官方页有 70 处 `class="std std-ref"` 链接，剪藏丢掉了链接**文字**，只留下语法空洞。已定位的断句与官方原文对照：

| 行号 | 剪藏现状 | 官方 1.8 原文 |
| --- | --- | --- |
| L18 | …generalized linear models, see**.** | …see **Logistic regression**. |
| L118 | This method has the same order of complexity as**.** | …as **Ordinary Least Squares**. |
| L421、L919 | In contrast to the**,** each coordinate… | In contrast to the **Bayesian Ridge Regression**, each coordinate… |
| L491、L930 | …preserve the symmetrical inductive bias regarding ordering of classes, see**.** … as shown in**.** | …see **[16]**. … as shown in **[16]**. |

L491 的 `[16]` 指向官方页脚注 16（Logistic 优化器比较的文献），而**剪藏的脚注编号体系本身是断的**（只有 1、2、3、4、5、6、7、8、9、10、11、12、13、20），所以这类引用在剪藏内无法还原——只能标注"目标不可解析"。

### 3. 1.1.14 的补入与回填（raw 已非原始剪藏）

1.1.14（Robustness regression）在原始剪藏中整节缺失。2026-11-11 的处理分两步：

1. **所有者手工补入**正文结构、三张 robust_fit 图与 RANSAC/Huber 段（L620–L798），但留下 3 处公式空缺、2 处引用语法错误。
2. **图书馆员按所有者授权回填**（同一天），全部内容取自官方 1.8 页、按该文件既有的 `$$` 显示公式风格插入：

| 回填位置 | 补入内容 |
| --- | --- |
| L720 之后 | $\binom{n_{\text{samples}}}{n_{\text{subsamples}}}$（Theil-Sen 时间/空间复杂度） |
| L752 之后（"minimizes"） | $\min_{w, \sigma} \sum_{i=1}^{n}\left(\sigma + H_{\epsilon}\left(\frac{X_i w - y_i}{\sigma}\right)\sigma\right) + \alpha \|w\|_2^2$ |
| L758 之后（"loss function is given by"） | $H_{\epsilon}(z) = z^2$（$|z| < \epsilon$），否则 $2\epsilon|z| - \epsilon^2$ |
| L730、L734 | 把 `[[14](url)]` / `[[15](url)]` 改为 `[[14]](url)` / `[[15]](url)`，即渲染为带链接的 `[14]`/`[15]` |

**代价与后果**：raw 文件的 SHA-256 因此变为 `03324a20…`，**不再是抓取时的原始字节**；行数从 965 增到 980，L720 之后的行号整体后移（1.1.14.3 结束于 L739、1.1.14.4 结束于 L782、1.1.15 为 L784–L798、1.1.16 为 L800–L875、脚注自 L877 起）。本库所有页面已按新行号重写过引用。

**仍未修复**（属剪藏转换损失，不属于 1.1.14）：
- 官方页的 `Details of the algorithm` / `Theoretical considerations` 折叠标题在剪藏中不存在，RANSAC 四步（L681–L694）与 Theil-Sen 的 `is_data_valid`/breakdown point 段落是所有者以正文形式并入的。
- L649 的 "Trade-offs: which estimator ?" 是官方 `<p>` 文本而非标题。
- `[[14]]` / `[[15]]` 仍与本文档自己的脚注体系（`[^13]`、`[^20]`）不一致：官方用脚注编号 14/15 指向同一批文献，剪藏的脚注编号是断的，回填只修了链接语法，没有重建编号体系。

### 4. 数学与表格噪声

- 所有公式被转成 `\left(\right.` 形式的伪 LaTeX（如 L67、L146、L181）。语义可辨，但直接复制到代码或论文里会出错。本库在 wiki 页面统一改写为规范 LaTeX。
- logistic solver 矩阵是原始 HTML `<table>`（L499），渲染正常但 `rg` 不可检索。
- 剪藏 L543 与脚注 L966 的 Categorical 单位偏差写作 $I(y=i)\,y_i\log\frac{I(y=i)}{\hat{I(y=i)}}$，与官方 1.8 的 $\hat y_i$ 形式不符，疑为转换错误（**本库判断**）。

### 5. 版本判定方法（可复核）

对官方各版本页做特征字符串比对（2026-11-11）：

| 特征 | 1.6.1 | 1.7.2 | 1.8.0 | 1.9.1 | 剪藏 |
| --- | --- | --- | --- | --- | --- |
| `1.1.3.1 gap safe screening` 小节存在 | 否 | 否 | 是 | 是 | 是 |
| `It is also the only solver that supports Elastic-Net` | 否 | 否 | 是 | 是 | 是 |
| `newton-cholesky is a good choice` | 否 | 是 | 是 | 是 | 是 |
| doctest 输出 `np.float64(0.13636)` | 否 | 是 | 是 | 是 | 是 |
| `1.1.x` 标题集合（41 项） | — | — | 41 | 41（与 1.8 相同） | 41 |

结论：**≥ 1.8**；1.8.0 与 1.9.1 在这些特征上不可区分。1.9 重写了 1.1.3.1 的正文（新增 duality gap 的 `Mathematical details`、删去"A clever method to speedup…"一段），因此**引用该小节的措辞应对应 1.8 版**。

## 局限

- **残缺文档**：折叠块全缺（29.3% 正文）、70 处内部引用文字丢失、3 处公式在 1.1.14 内空缺、参考文献全无。任何"文档没这么说"的推论都必须先排除"转换时丢了"这一解释。
- **版本不确定**：只能判到 ≥ 1.8。官方 `stable` 会继续漂移，本页对行为的所有描述都是**对该版本的快照**。
- **无对照、无基准**：这是 API 文档，不是实验报告。"`RidgeClassifier` 可能显著更快""LassoLarsCV 常常更快"都是定性判断，除剪藏自带示例图的耗时（0.17s vs 0.06s）外无量化证据。
- **示例图的耗时不可当基准**：图标题里的 `train time` 来自示例脚本的一次运行，未给硬件、数据规模与重复次数。
- **引用的外部文献未读**：Weisberg 对 LARS 的批评（L294–L296）、Theil-Sen 的 Dang et al.（L730）、Huber 的 Robust Statistics p. 172（L772）均为来源转述。
- **不覆盖工程管线**：缺失值、类别编码、Pipeline、特征缩放与 CV 切分分别在 8.3、1.13、3.1 等其它章节（本库同批剪藏中尚有 17 篇未入库），本页不能替代。

## 待解问题

1. **`alpha` 与样本量的关系在哪里被"补偿"？** Ridge 目标无 $1/n$、Lasso 有（L67 vs L146），文档未说明这个不一致是否由内部实现（如 `alpha` 的缩放）消除。需要在源码或实验上验证后才能断言"跨估计器 alpha 不可比"（见 [[regularization-and-sparsity]] 的表述强度）。
2. **`saga` 的"对未缩放数据不鲁棒"具体表现为收敛失败还是精度损失？** 表格只给 yes/no（L499），无量化。
3. **`RidgeCV` 的 LOO 为什么不能取 alpha=0？** 文档只说"due to the formulation"（L122），没有给公式。
4. **AIC/BIC 在 `n_samples > n_features` 之外的表现**：L899 的脚注限定条件之外，`LassoLarsIC` 的失效是渐变还是突变？
5. **`HuberRegressor` 的 `epsilon=1.35`** 对应 95% 效率（L768），该常数与 `SGDRegressor(loss="huber")` 的 `epsilon` 语义是否一致？文档只说前者尺度不变。
6. **缺失的 `Solvers’ details` 与 `Practical considerations` 是否有本库需要的关键约束**（如 `l1_min_c`、GLM 的 exposure 用法）？若需要，应单独入库官方 1.8 的这两个折叠块（**不建议**用当前剪藏作唯一依据）。

## 行号索引

| 主题 | 位置 |
| --- | --- |
| 全模块约定 `coef_` / `intercept_` | L16 |
| OLS 目标与共线性警告 | L24–L42 |
| NNLS（`positive=True`） | L48–L52 |
| OLS 复杂度（SVD，$O(n p^2)$） | L58 |
| Ridge 目标与 alpha 含义 | L66–L70 |
| Ridge 用法示例与 `solver="auto"` 顺序表 | L74–L84 |
| `RidgeClassifier` 与投影矩阵只算一次 | L104–L110 |
| `RidgeCV` 的 LOO 与 alpha≠0 | L122–L137 |
| Lasso 目标、坐标下降、`lasso_path` | L141–L162 |
| 坐标下降推导、软阈值、`tol`=duality gap、screening | L176–L192 |
| `LassoCV` vs `LassoLarsCV` | L200–L204 |
| AIC/BIC 与 `LassoLarsIC` 的失效条件 | L206–L219 |
| `alpha` 与 SVM 的 `C` | L225–L227 |
| MultiTaskLasso（列稀疏） | L229–L241 |
| ElasticNet 与相关特征 | L243–L264 |
| MultiTaskElasticNet | L266–L278 |
| LARS 优缺点 | L282–L296 |
| LARS Lasso（精确解、分段线性） | L298–L317 |
| OMP（$ℓ_0$ 约束） | L319–L339 |
| 贝叶斯回归动机 | L343–L362 |
| BayesianRidge（对数边缘似然、gamma 先验） | L364–L407 |
| ARD（逐系数精度） | L409–L429 |
| logistic 默认正则、GLM 特例 | L433–L454 |
| 二值损失 (1) 与四种惩罚 | L464–L484 |
| 多类过参数化说明 | L486–L491 |
| solver 能力矩阵与选择建议 | L493–L501 |
| liblinear 边角差异、p 值 | L505–L515 |
| GLM 目标与单位偏差表 | L527–L546 |
| 分布选择规则 | L558–L561 |
| `TweedieRegressor` 的 `power` 映射 | L565–L572 |
| SGD / Perceptron / PA | L594–L618 |
| 稳健回归概念与 breakdown point | L620–L647 |
| 三估计器取舍（含 "When in doubt"） | L649–L662 |
| RANSAC 四步与 `is_data_valid` | L664–L694 |
| Theil-Sen（29.3%、复杂度、文献 14/15） | L703–L739 |
| Huber（epsilon、与 SGDRegressor 的差别） | L741–L782 |
| 分位数回归与 pinball loss | L784–L798 |
| `PolynomialFeatures`、Pipeline、XOR | L800–L875 |
| 脚注：AIC/BIC 公式、噪声方差、ARD 重复、solver 说明 | L877–L981 |

## 关联页面

- 工具：[[scikit-learn]]（本剪藏对应的模块表面、默认值与版本风险）
- 实践：[[linear-models-as-baselines]]（作为基线的用法、五个坑、成本与自动化程度）
- 概念：[[regularization-and-sparsity]]（$ℓ_1$/$ℓ_2$/EN 的几何、先验与稀疏性来源）
- 概念：[[leaderboard-vs-cv]]（内置 CV / 信息准则作为代理指标）
- 对照来源：[[kaggle-getting-started]]（2016 年的 API 与模型清单；本来源是其现代对应物）
- 研究：本来源暂未产生智能体层问题（[[automated-research-open-questions]] 未新增条目）
