---
title: 超参数搜索的落地流程：先冻结协议、再选搜索器、最后按拟合次数算预算
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - hyperparameter-tuning
  - model-selection
  - workflow
  - grid-search
  - randomized-search
  - successive-halving
sources:
  - "[[sklearn-hyperparameter-tuning]]"
  - "[[sklearn-cross-validation]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-ensembles]]"
  - "[[kaggle-getting-started]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（用户指南 3.2 节与 `model_selection` 搜索器签名）
  - scikit-learn 1.6.1（本页全部运行时读数）
---

# 超参数搜索的落地流程

本页把 [[hyperparameter-search]] 的机制翻译成可执行的顺序：**协议 → 空间 → 搜索器 → 预算 → 报告**。前一步没定就做后一步，是本节所有坑的共同来源。

两条前置：[[cross-validation-design-and-tuning]] 的「五个准入问题」必须先答完（否则你搜的是另一个问题的解）；本页的读数是 scikit-learn **1.6.1**，目标文档是 **1.9.1**，签名差异见 [[sklearn-hyperparameter-tuning]] 的「外部核查」。

## 一、五个准入问题

| 问题 | 答案为「是」时的动作 | 代价 |
| --- | --- | --- |
| ① 这个参数真的会影响结果吗？ | 才放进空间；否则留在默认值并记录 | 每加一维，网格按乘积膨胀 |
| ② 评估协议（splitter、指标、折数）定死了吗？ | 定死后**冻结**，写成常量复用 | 协议在搜索中改动 = 整轮作废 |
| ③ 你要的是「报分数」还是「选配置」？ | 只选配置 → 搜索的 CV 可以复用；两者都要 → 嵌套或 dev/eval | 候选 × 折 的乘数 |
| ④ 拟合时间是否大致正比于样本量？ | 是 → 可考虑逐次减半；否 → 优先随机搜索 | halving 的收益全在这一条上 |
| ⑤ 有多少预算（拟合次数 / 墙钟）？ | 反推候选数与折数 | 见第四节 |

第 ③ 问是最容易跳过的一问。来源用一句很克制的话承认了搜索的性质（[[sklearn-hyperparameter-tuning]], L286）：「按各种参数设置评估来做模型选择，可以看作**用带标签的数据去『训练』网格的参数**」。因此「调参之后的 CV 分数」与「调参之前的 CV 分数」不是同一个量。

## 二、先定空间：形状比算法重要

来源只有一句方法论（L32）：**少数参数影响巨大，其余留在默认值**，并建议读 docstring 与它引的文献。本库把它展开成三条规则：

1. **按「尺度」而不是「取值」建空间**。正则强度、核宽、树深度这类参数的最优值通常跨数量级，正确写法是 `loguniform(1e-4, 1e2)` 或 `[1e-4, 1e-3, …, 1e2]`，而不是 `uniform` 或等距列表。实测（[[sklearn-hyperparameter-tuning]] 实测 B-5）：`uniform(0.001, 1000)` 采 100 次得到的 `best alpha` 是 4.696，而最优在 1e-3 附近——**采样方式直接决定能不能找到**。
2. **一次只加一维，并记录边际收益**。加一维在网格上是「乘一档」，在随机搜索上是「每条候选多一个坐标」；两者的成本性质不同（第五节）。
3. **把「不该搜」的参数明确写进报告**：`n_jobs`、`random_state`、`verbose`、`pre_dispatch` 之外，还有一类**结构性开关**（`kernel`、`solver`、`criterion`）——它们切换的是**模型**而不是连续强度，混在同一张网格里会让 `cv_results_` 的排序失去可比性。

## 三、选搜索器：一张判定表

| 条件 | 用哪个 | 理由 | 1.6.1 实测 |
| --- | --- | --- | --- |
| 候选数 $\lesssim 30$、维数 $\le 3$、单次拟合便宜 | `GridSearchCV` | 结果可枚举、可复现、无需种子 | 180 网格 24.58s（541 fits） |
| 维数多、连续参数、预算固定 | `RandomizedSearchCV` | 预算与维数解耦；分辨率随 `n_iter` 线性提升 | 60 候选 8.03s，`best_score_` 0.9885（grid 0.9890） |
| 候选数 $\gg$ 折数、拟合成本正比于样本量、能接受实验性 API | `HalvingGridSearchCV` / `HalvingRandomSearchCV` | 先小资源淘汰，再大资源决赛 | 180 网格 8.62s（811 fits）但参数与 grid 只 0/5 一致 |
| 单一正则强度、线性模型 | `*CV` 估计器（`RidgeCV`/`LassoCV`/`LogisticRegressionCV`…） | **沿正则化路径一次拟合多个值**，比外层搜索便宜得多 | 本机未测；10 个类见官方 3.2.5.1（**剪藏丢失**） |
| 只要一个数、且愿意假设 | `LassoLarsIC`（AIC/BIC） | 闭式，不用 CV | 本机未测；见 [[regularization-and-sparsity]] |
| bagging 家族、想省一份验证集 | `oob_score=True` | 袋外样本是「免费的 CV 近似」 | 见 [[ensembles]]；官方 3.2.5.3（**剪藏丢失**） |

第三行必须加两条警告：**两个 halving 估计器在 1.6.1 仍是实验性 API**（`from sklearn.experimental import enable_halving_search_cv`，不 import 直接 `ImportError`，[[sklearn-hyperparameter-tuning]] L110–L116），且它们的 `best_params_` **不是网格最优**（实测 5 个种子 0/5 一致）。

第四到六行来自官方 3.2.5 节，**该节在剪藏中整节丢失**，因此本库目前只能记录它们的名字与适用面，不能引用其正文论证（见来源页「转换损失」与「局限」）。

## 四、算预算：按拟合次数，不按「跑多久」

$$\text{fits} = \underbrace{|\Theta|\ \text{或}\ n_{\text{iter}}}_{\text{候选数}}\times \underbrace{k}_{\text{折数}} + \underbrace{1}_{\text{refit}}$$

| 动作 | 拟合次数 | 实测（1.6.1） |
| --- | --- | --- |
| 3 候选 × 5 折 + refit | 16 | 计数器估计器逐次核对 |
| 同上但 `refit=False` | 15 | — |
| 8 候选 × 5 折 | 41 | — |
| 180 候选 × 3 折 | 541 | 24.58s（RF 20 棵树、2000×20） |
| 180 候选 × 3 折、`n_jobs=-1` | 541 | 5.80s（**4.27×**） |
| `HalvingGridSearchCV` 180 候选 × 3 折、`factor=3` | **811** | 8.62s（样本量少约 31×） |
| `HalvingGridSearchCV` 8 候选 × 5 折 | **76** | 比网格贵 |
| `permutation_test_score(n_permutations=50, cv=5)` | 255 | 见 [[cross-validation-design-and-tuning]] |

三条纪律：

1. **先算拟合次数，再决定折数**。把「5 折 × 200 候选」换成「3 折 × 60 候选」通常比换搜索器划算，而且不改变答案的含义。
2. **halving 的预算要按「候选数 × 折数 × 资源比例」算**：它的拟合次数**一定不少于网格**（第一轮评估全部候选）。若你的模型有强固定开销（建索引、数据拷贝、编译），收益会消失。
3. **把 refit 算进去**：搜索通过 ≠ 最终模型能训练。refit 阶段**永远抛错**（`error_score` 不影响它，外部核查）。

## 五、评估协议：三种写法，选一种并写清

| 写法 | 何时用 | 代价 | 报什么 |
| --- | --- | --- | --- |
| **dev / eval 切分**（`train_test_split` 后只在 dev 上搜） | 数据够、只要一个可信的最终数 | 少 20–30% 训练数据 | `best_score_`（dev）+ 一次性 eval 分 |
| **嵌套 CV**（外层只用来报分） | 数据紧、候选多、两者都要 | 外层折数倍 | 外层折分数列（不是单个数） |
| **同一套 CV 既选又报** | 只用来**选**，不报分 | 0 | 明确标注「内部选择用，不作性能声明」 |

来源推荐第一种（L288–L290：把数据分成 **development set** 喂给搜索、**evaluation set** 算指标）。本库实测的乐观偏差幅度是条件性的：dev/eval 写法下 6 个种子平均 **+0.0067**（[[sklearn-hyperparameter-tuning]] 实测 C 附近）；嵌套 CV 在 [[sklearn-cross-validation]] 里测到 **+0.0000**（breast cancer）到 **+0.0500**（80 样本纯噪声）两端。所以判据是「候选数 × 折数 与 $n$ 的相对量级」，不是「一定要嵌套」。

**硬约束（本库综合）**：搜索用的 `cv` 与 Stacking 用的 `cross_val_predict` **不能共用同一套划分**，否则折外预测与调参互相污染（[[ensembles]]、[[hyperparameter-search]] 第八节）。

## 六、代码骨架

```python
import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# 换成 GridSearchCV(param_grid=...) 或 HalvingGridSearchCV(factor=3, ...)
# 时只需改搜索器这一行与空间字典的键名；cv 对象保持同一个
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1) 协议先冻结：同一个 cv 对象复用到所有搜索器
#    X_dev/y_dev 与 X_eval/y_eval 来自一次 train_test_split，后者只看一次
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC())])

# 2) 空间按尺度建：连续参数用 loguniform，离散开关用列表
space = {"svc__C": loguniform(1e-2, 1e3),
         "svc__gamma": loguniform(1e-4, 1e1),
         "svc__kernel": ["rbf"]}

# 3) 搜索器：refit=True 才有 best_estimator_ / predict
search = RandomizedSearchCV(pipe, space, n_iter=60, cv=CV,
                            scoring="f1", refit=True, n_jobs=-1,
                            random_state=0, error_score="raise")
search.fit(X_dev, y_dev)

# 4) 报告：六项 + 搜索的元信息
print(search.best_params_, search.best_score_,
      len(search.cv_results_["params"]),          # ⚠ 全列表时会被夹住
      float(np.isnan(search.cv_results_["mean_test_score"]).sum()))

# 5) 只用一次 eval 集（不要在 eval 上再做选择）
print(search.best_estimator_.score(X_eval, y_eval))
```

三处必须显式写出的默认值（否则沉默行为会吃掉结论）：

- `error_score="raise"`：默认 `np.nan`。部分失败会静默留 `nan`，**全部失败会抛 `ValueError`**（无论 `error_score` 为何，见 [[sklearn-hyperparameter-tuning]] 关键主张 5）。
- `refit=True` / `refit="<指标名>"`：多重指标时留默认直接 `ValueError`；`refit=False` 时没有 `best_estimator_`，`predict` 抛 `AttributeError`。
- `random_state=0`（仅随机搜索与 halving 有）：`GridSearchCV` **没有** `random_state`，传了即 `TypeError`。

## 七、诊断信号：七条

| 信号 | 读数怎么取 | 含义 |
| --- | --- | --- |
| ① `best_score_` 与 eval 分的差 | 两数相减 | >0.01 就要怀疑选择偏差或泄漏 |
| ② `cv_results_` 里 `nan` 的个数 | `np.isnan(...).sum()` | >0：有配置/折失败，先查原因再看排名 |
| ③ 最优落在空间边界 | 比较 `best_params_` 与网格端点 | 在边界上 = 真实最优可能在空间外（实测：`alpha` 被推到 `loguniform` 下界） |
| ④ 前 5 名的分数差 | `np.sort(mean_test_score)[-5:]` | 差 < 折间 std：排名不可信，别用参数解释性能 |
| ⑤ 平票 | `rank_test_score` 的最小值出现次数 | 全为 1 时 `best_index_` = **枚举顺序第一**（实测） |
| ⑥ halving 与 grid 是否给同一组参数 | 两个搜索器各跑一次 | 实测 0/5 一致——不一致是常态，不是 bug |
| ⑦ halving 的 `n_resources_` 是否用满 `max_resources` | 看最后一个元素与样本数 | 差得远 = 候选数/`min_resources` 没配好（官方「耗尽资源」折叠块丢失，见来源页） |

## 八、18 条坑

1. **连续参数写成列表**：`RandomizedSearchCV` 对全列表参数是**不放回**采样，`n_iter=20` 在 5 个候选上只跑 **5 行**，且不报错。
2. **连续参数写成 `uniform`**：数量级上错过最优（实测最佳落在 4.696，真最优在 1e-3 附近）。
3. **照抄剪藏 L82 的 `from sklearn.utils.fixes import loguniform`**：1.6.1 上 `ImportError`；改用 `scipy.stats.loguniform`。
4. **多重指标 + 默认 `refit`**：`ValueError`，且报错信息里的默认值是 `True` 而文档写 `None`。
5. **多重指标 + `refit=False` 后去读 `best_params_`**：不存在（`AttributeError`），只能自己从 `cv_results_` 里挑。
6. **`refit=callable` 之后读 `best_score_`**：不存在。
7. **`refit=False` 之后调 `predict`**：`AttributeError: This 'GridSearchCV' has no attribute 'predict'`（不是 `NotFittedError`）。
8. **以为 `best_estimator_` 是折内模型**：它是全量重训的（实测折内 400 样本、refit 500 样本）。
9. **把 `best_score_` 当泛化误差**：它是 $m$ 个估计量的最大值。
10. **全部候选都失败**：`ValueError: All the N fits failed.`，在 `fit` 阶段直接抛出，与 `error_score=np.nan` 的预期相反。
11. **refit 阶段失败**：永远抛错，`error_score` 管不到。
12. **`GridSearchCV(random_state=0)`**：`TypeError`，没有这个参数。
13. **在候选很少时改用 halving**：8 候选 41 → 76 次拟合；3 候选 16 → 26 次。
14. **假设 halving 给出网格最优**：实测 5 个种子 0/5 参数一致、平均分数低 0.0025。
15. **把 halving 的 `cv_results_` 当短表**：它是「候选 × 轮次」长表（含 `iter`/`n_resources` 列），行数远多于候选数。
16. **读 `resource_`**：halving 没有这个属性（只有 `min_resources_`/`max_resources_`）。
17. **把 `resource` 设成 `param_grid` 里的参数**：官方明确「不可能」（该约束在丢失的折叠块里，本页按官方 1.9.1 页记录，未实测）。
18. **`n_jobs=-1` 当免费加速**：大网格上 4.27×，小网格上进程开销可能倒挂。

## 九、成本参考（1.6.1，本机，单机多核）

| 动作 | 规模 | 墙钟 | 拟合次数 |
| --- | --- | --- | --- |
| `GridSearchCV` | 180 候选 × 3 折，RF 20 树，2000×20 | 24.58s（`n_jobs=1`）/ 5.80s（`-1`） | 541 |
| `RandomizedSearchCV` | `n_iter=60` × 3 折，同上 | 8.03s | 181 |
| `HalvingGridSearchCV` | 180 候选 × 3 折，`factor=3`，同上 | 8.62s | 811 |
| 逐次减半的资源曲线 | 同上 | — | `n_resources_ [24, 72, 216, 648, 1944]`、`n_candidates_ [180, 60, 20, 7, 3]` |
| 单次搜索的调度开销 | 3 候选 × 5 折 | 毫秒级 | 16 |

## 十、可自动化程度

| 环节 | 可自动化 | 理由 |
| --- | --- | --- |
| 冻结评估协议 | **高（必须做）** | 把 `cv` 与 `scoring` 提成常量，机械可查 |
| 空间设计 | 低 | 「哪些参数重要」需要领域知识与 docstring |
| 选搜索器 | 中 | 判定表可机械套用，但「拟合时间是否正比于样本量」要实测 |
| 预算核算 | 高 | 拟合次数是纯算术 |
| 换 `scipy.stats` 分布 / 尺度 | 高 | 有固定模式（正则强度、核宽、深度） |
| 失败与 `nan` 处理 | **高（必须做）** | `error_score='raise'` + `nan` 计数是最便宜的护栏 |
| 读 `best_score_` 作结论 | **低** | 必须配 dev/eval 或嵌套才能报分 |
| halving 的调参与解读 | 低 | 文档的「为什么」整块丢失，只能靠 `n_resources_` 反推 |
| 报告 | 高 | 字段固定（参数、分数、折分数列、`nan` 数、拟合次数、版本） |

## 关联页面

- 概念：[[hyperparameter-search]]（预算曲线、`best_score_` 的语义、选择偏差）、[[evaluation-metrics]]（「分数」本身是哪个泛函的损失；主指标不能与护栏混排）、[[cross-validation]]（分数的估计量性质）、[[leaderboard-vs-cv]]（本地分数 vs 真实目标）、[[regularization-and-sparsity]]（`*CV` 估计器与 AIC/BIC 属于「不搜索的选参」）
- 实践：[[cross-validation-design-and-tuning]]（前置：五个准入问题与报告六项）、[[svm-selection-and-tuning]]（`C`/`gamma` 的指数网格实例）、[[ensembles-selection-and-tuning]]（预算型参数的搜索）、[[mlp-selection-and-tuning]]（架构 + `alpha` 的搜索与种子方差）、[[feature-selection-workflow]]（另一个「选一个子集」的流程）
- 来源：[[sklearn-hyperparameter-tuning]]（本页全部数字与 API 行为的出处）、[[sklearn-cross-validation]]（协议层）、[[kaggle-getting-started]]（人类选手的调参五步）
- 实践补充：[[metric-selection-and-thresholding]]（搜索之前先声明泛函与符号）
- 工具：[[scikit-learn]]（搜索器矩阵与默认值第 201–230 条）
