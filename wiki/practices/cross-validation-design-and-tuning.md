---
title: 交叉验证的落地设计：先定依赖结构，再定折数，最后定报告格式
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - cross-validation
  - model-selection
  - evaluation
  - workflow
sources:
  - "[[sklearn-cross-validation]]"
  - "[[feature-selection-workflow]]"
  - "[[leaderboard-vs-cv]]"
  - "[[kaggle-getting-started]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-svm]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（API 默认值与文档，见 [[sklearn-cross-validation]]）
  - scikit-learn 1.6.1（本页数字的实测版本）
---

# 交叉验证的落地设计

本页把 [[cross-validation]] 的机制折成流程：**先用五个问题定依赖结构，再用一张表选 splitter，再定 $k$ 与重复数，最后按固定格式报告**。来源是 [[sklearn-cross-validation]]（用户指南 3.1 节）；相关但不同的两页是 [[feature-selection-workflow]]（选择器怎么放）与 [[leaderboard-vs-cv]]（本地分数与排行榜的错位）。

一句话判据：**CV 写错不会报错，只会给你一个错的数**。所以下面每一条都以「错了会看到什么」结尾。

## 一、五个准入问题

先回答，再写代码。每条决定后面的一行配置。

| # | 问题 | 若答「是」 | 若答「否」 |
| --- | --- | --- | --- |
| 1 | 样本之间有**组**（同一患者/用户/设备/实验/文档）吗？ | `GroupKFold` / `GroupShuffleSplit`（类别不平衡时 `StratifiedGroupKFold`） | 进入 2 |
| 2 | 数据有**时间顺序**，且部署时用过去预测未来吗？ | `TimeSeriesSplit`（必要时 `gap>0`） | 进入 3 |
| 3 | 目标有**稀有类**（某类样本 ≪ 折数）吗？ | `StratifiedKFold`（并检查每折的类计数） | `KFold` |
| 4 | 有**外部给定的划分**（官方 split、历史验证集）吗？ | `PredefinedSplit` | 进入 5 |
| 5 | 这次只做**评估**，还是要**选超参**？ | 选超参 → 嵌套 CV（第四节） | 单层 CV |

顺序不能调换：**分组优先于分层，分层优先于 i.i.d.**。实测证据（[[sklearn-cross-validation]] 第 22 条）：30 个单类组、5 折，`GroupKFold` 的测试折类比例极差 0.590，`StratifiedGroupKFold` 无洗牌 0.064，而**开了 `shuffle=True` 反而退到 0.375**——因为单类组无法被洗牌（这段机制在官方页丢失的 `Implementation notes` 折叠块里）。

## 二、选 splitter：一张判定表

| 场景 | 写法 | 会踩的坑 |
| --- | --- | --- |
| 干净 i.i.d.、平衡类 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` | 不洗牌 + 有序数据 → 分数崩塌 |
| 回归 / i.i.d. | `KFold(n_splits=5, shuffle=True, random_state=0)` | `shuffle=False` 时传 `random_state` 会**抛 `ValueError`** |
| 稀有类 | `StratifiedKFold(..., shuffle=True)` | 类样本数 < 折数 → 某折训练集单类 → `nan` |
| 分组 | `GroupKFold(n_splits=5)` 或 `GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=0)` | 折大小不齐；`shuffle=True` 平衡的是**组数**不是样本数 |
| 分组 + 不平衡 | `StratifiedGroupKFold(n_splits=5)` | 洗牌只发生在同类分布组之间；可能产生不平衡划分 |
| 时序 | `TimeSeriesSplit(n_splits=5, gap=g)` | 测试折大小 = $\lfloor n/(k+1)\rfloor$，剩余全进第一折；`gap` 要 ≥ 自相关尺度 |
| 留一组做「外部测试」 | `LeaveOneGroupOut()` | 组少时等价于极端的 $k$；组多时拟合次数爆炸 |
| 已有划分 | `PredefinedSplit(test_fold=[0,0,-1,…])` | 只给 0/−1 得到**唯一**一个划分（实测 `n_splits=1`） |
| 纯重采样、非划分 | `ShuffleSplit(n_splits=20, test_size=0.2)` | **不能**用于 `cross_val_predict`（`ValueError: only works for partitions`） |

三条**不是**可选项的细节：

1. **`cv=int` 在分类器上等于分层、不洗牌的 `StratifiedKFold`**。要洗牌必须显式传 splitter 对象（`cross_val_score(..., cv=5)` 与 `cv=StratifiedKFold(5)` 分数逐位相同；`cv=KFold(5)` *不同*——实测 0.98 vs 0.94）。
2. **`groups` 只在 `cv` 是分组 splitter 时生效**。`cv=5, groups=…` 只产生 `UserWarning: The groups parameter is ignored by StratifiedKFold`，分数照常返回——**这是最容易在自动化里静默出错的一处**。
3. **`train_test_split` 没有 `groups` 参数**（传 `TypeError`），所以「先 `train_test_split` 再按组 CV」在 API 层就被挡了一半；分组场景应直接用 `GroupShuffleSplit(...).split(...)` 取索引。

## 三、定 $k$、重复数与预算

| 配置 | 每折训练量 | 拟合次数 | 折间 std（iris，实测） | 适用 |
| --- | --- | --- | --- | --- |
| `KFold(5)` | $4n/5$ | 5 | 0.0163 | 默认起点 |
| `KFold(10)` | $9n/10$ | 10 | 未测（应更小偏差、更大方差） | 数据够、结论要紧 |
| `LeaveOneOut` | $n-1$ | $n$ | 0.1400（**量纲不同，不可直接比**） | 极小样本、模型极便宜 |
| `RepeatedKFold(5, 3)` | $4n/5$ | 15 | 0.0239（**比单次 5 折更大**） | 结论要抗「坏划分」 |

要点：

- **折数增加买的是偏差，不是精度**。$k$ 越大，训练集越接近全集，CV 越不保守，但折间相关性也越高。
- **重复买的是「抗划分运气」，不是「降低 std」**。15 折的 std 反而更大，因为不同重复之间不独立。
- **LOO 的 std 与 $k$ 折的 std 不是一个量**（前者是单样本正确率的离散度），不要写成「LOO 方差是 5 折的 8.6 倍」。
- **成本以拟合次数计**：$k$ 折 = $k$ 次；`RepeatedKFold(k, r)` = $kr$ 次；嵌套 CV = $k_{\text{outer}} \cdot m \cdot k_{\text{inner}}$（$m$ = 候选数）。置换检验 = $(n_{\text{perm}}+1)\cdot k$（来源写明，L638；实测 50×5 折 = 255 次拟合）。
- **先算预算再选 $k$**：AutoResearch 场景下「一夜 100 次实验」（[[autoresearch]]）通常应换更便宜的模型而不是更少的折——因为折数直接决定分数的可信度。

## 四、嵌套 CV：什么时候必需、什么时候是浪费

**必需**的条件（同时成立）：① 你要用 CV 分数**选**超参/模型；② 你还想用 CV 分数**报**这个选择的性能；③ 候选数量 × 有效样本量不宽松。

实测两个端点（[[sklearn-cross-validation]] 第 19–20 条）：

| 数据 | 候选 | 非嵌套（挑最好后报分） | 嵌套（外层 5 折） | 乐观偏差 |
| --- | --- | --- | --- | --- |
| breast cancer（569×30） | 6 个 `C` | 0.9754 | 0.9754 | **+0.0000** |
| 80 样本 × 1500 纯噪声 | 6 个 `C` | 0.6125 | 0.5625 | **+0.0500** |

结论：**「非嵌套一定虚高」是错的**，虚高幅度由「候选数 / 有效样本量 / 折数」决定。实践判据：候选数 $\times$ 折数 $\gtrsim n$ 时（本库经验阈值，非来源结论）就上嵌套；否则嵌套只是把计算量乘 5。

两层都写对的关键：**外层与内层必须用不同的划分**（同种子同 splitter 会让内层选择看到外层测试折的信息），且内层选出的配置要**在外层训练折上重训**再预测外层测试折。

内层就是一个搜索器（`GridSearchCV` / `RandomizedSearchCV` / `Halving*SearchCV`）。搜索器自身的预算、`refit` 语义与失败行为见 [[hyperparameter-search-and-tuning]]；只选不报时可以省掉外层，但要在报告里写明「此分数仅用于选择」。

## 五、报告的固定格式（六项）

只报一个数会让下游无法判断可信度（[[leaderboard-vs-cv]] 的「给提升定义噪声底」）。最少报六项：

1. **均值 ± 折间 std**，以及**逐折分数数组**（`cross_validate` 的 `test_score`）。
2. **splitter 的完整构造**（类名 + `n_splits` + `shuffle` + `random_state`）。同一份数据上 `cv=5` 与 `cv=KFold(5)` 是两件事。
3. **折大小**（`return_indices=True` 可取；分组/时序下折大小不齐，这是正常现象而非 bug）。
4. **失败折的数量与原因**。`error_score` 默认 `np.nan`，所以「分数数组里有 `nan`」是唯一痕迹；汇总时必须显式检查 `np.isnan(scores).sum()`，**不要用 `np.nanmean` 抹平**。
5. **分数算在哪个对象上**：`predict` / `decision_function` / `predict_proba`。三者可以互相矛盾（[[support-vector-machines]] 的四个输出面），同一模型换一种输出就是一个新的可评估对象。
6. **指标定义**（含多类平均方式）。`precision_macro` 与 `precision_micro` 在同一份数据上不同。

## 六、五条必须显式改掉的默认值

| 默认值 | 后果 | 改法 |
| --- | --- | --- |
| `cv=None` → 5 折（0.22 起） | 分数随库版本变化（更早是 3 折） | 显式传 splitter 对象 |
| `shuffle=False` | 有序数据上分数崩塌（实测四折全 0.0） | `shuffle=True, random_state=0` |
| `error_score=np.nan` | 失败折静默变成 `nan` | 调试期设 `error_score='raise'` |
| `groups` 在 `cv=int` 时被忽略 | 分组泄漏，只有一条 warning | `cv=GroupKFold(...)` + `groups=` |
| `return_train_score=False` | 看不到「训练分 vs 测试分」的 gap，无法判断过拟合 | 调参阶段设 `True`（成本翻倍） |

> [!note] `return_indices` 的键名
> 文档说「retain the training and testing indices」，实际返回的是 **`indices` 字典**（`{'train': [...], 'test': [...]}`），不是 `train_indices`/`test_indices`。照字面写会 `KeyError`（实测，1.6.1）。

## 七、推荐的代码骨架

```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv = GroupKFold(n_splits=5)                      # 第一节的五个问题决定这一行
est = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
res = cross_validate(
    est, X, y, groups=groups, cv=cv,
    scoring=["accuracy", "roc_auc"],
    return_train_score=True, return_indices=True,
    error_score="raise",                          # 调试期：让失败折报错
    n_jobs=-1,
)
n_fail = int(np.isnan(res["test_accuracy"]).sum())
print("folds:", len(res["test_accuracy"]), "failed:", n_fail)
print("test  acc:", res["test_accuracy"].mean().round(4), "+-", res["test_accuracy"].std().round(4))
print("train acc:", res["train_accuracy"].mean().round(4))
print("fold sizes:", [len(t) for t in res["indices"]["test"]])
print("cost:", res["fit_time"].sum().round(2), "s fit,", res["score_time"].sum().round(2), "s score")
```

**所有会 `fit` 的步骤都放在 `make_pipeline` 里**——这是本库唯一有量化证据的防泄漏手段：`SelectKBest(k=5)` 放进 CV 时 0.8100、先在全量数据上选再 CV 时 0.8500；在 120 样本 × 20,000 纯噪声特征上差距拉到 0.508 → 0.900（[[feature-selection-workflow]]）。

## 八、诊断信号：什么时候该怀疑切分本身

| 看到什么 | 先怀疑 | 检查方法 |
| --- | --- | --- |
| 分数远高于同行/直觉（如 0.99） | 分组泄漏或预处理泄漏 | 换 `GroupKFold` 看是否掉下来；把预处理塞进 `Pipeline` 再跑 |
| 分数接近机会水平 | 有序数据 + 不洗牌；或某折训练集缺类 | 打印每折的类计数与折大小 |
| 逐折分数方差极大 | 折太小；或有 1–2 个「坏折」（离群/缺类） | 看逐折数组而不是均值 |
| 训练分 ≫ 测试分 | 过拟合或高维泄漏 | `return_train_score=True`（实测 iris 上 gap 只有 ±0.025，**没有信号就是没有信号**） |
| 分数数组里有 `nan` | `error_score` 吞掉了失败折 | 改 `error_score='raise'` 重跑一次 |
| 换种子分数变化大于调参收益 | 噪声底 > 信号 | 重复 CV 或多种子，把差值记为「未判定」 |
| 离线很好、上线很差 | 切分结构错位（组/时间/分布漂移） | 按部署结构重建 splitter（[[leaderboard-vs-cv]]） |

## 九、18 条坑

1. **`cv=5` 在分类器上不是 `KFold`**——是分层、不洗牌的 `StratifiedKFold`。同一份 iris 上两种写法给 0.98 与 0.94。
2. **`shuffle=False` + `random_state=0` 直接抛 `ValueError`**（`KFold`/`StratifiedKFold`/`GroupKFold` 都一样），所以「设了种子」这件事会被误以为是生效的。
3. **有序数据 + `KFold` 会崩塌**：标签按类连续排列时四折准确率全 **0.0**，且没有任何警告。
4. **`groups` 与 `cv=int` 同时传**：`groups` 被静默忽略，只有一条 `UserWarning`。
5. **`cv=GroupKFold(...)` 不传 `groups`** 才报错（`ValueError: The 'groups' parameter should not be None.`）——所以「忘了传」在 `cv=int` 时不会暴露。
6. **某折训练集只有一个类 → 分数 `nan`**，只有 `FitFailedWarning`；`error_score='raise'` 才给出 `The number of classes has to be greater than one; got 1 class`。
7. **`np.nanmean` 会把失败折洗掉**，等于静默缩小评测集；必须显式计数。
8. **`cross_val_predict` 不是分数**：噪声回归上折均值 R² 0.2273、池化 0.8407（差 0.61）；iris 准确率上两者都是 0.98（**看不出差别**）。
9. **`cross_val_predict` 不能配 `ShuffleSplit`/`GroupShuffleSplit`**（划分要求）：`ValueError: only works for partitions`。
10. **`cross_val_predict(method='predict_proba')` 需要模型真的有该方法**：默认 `SVC` 没有 → `AttributeError: This 'SVC' has no attribute 'predict_proba'`。
11. **`return_indices=True` 的键是 `indices`**（dict），不是 `train_indices`。
12. **`RepeatedKFold(n_splits=1)` 构造时不报错**，到 `split` 才失败；`StratifiedKFold(n_splits=1)` 构造即抛错。校验不能只放在构造处。
13. **`TimeSeriesSplit` 的测试折大小是 $\lfloor n/(k+1)\rfloor$**，剩余样本**全部**塞进第一个训练折——$n=10,k=4$ 时第一折只有 2 个训练样本。
14. **`TimeSeriesSplit(n_splits=5)` 在 $n=5$ 上报「folds=6」**（内部 +1），错误信息会误导到错误的参数上。
15. **`PredefinedSplit` 只给 0/−1 时只有 1 个划分**（实测 `get_n_splits()` = 1），不是「留一法」。
16. **`train_test_split` 默认 `test_size=0.25` 且不分层**：150 样本 3 类上测试集比例 0.342/0.421/0.237（全集 0.333）；要保比例必须显式 `stratify=y`。
17. **`n_permutations` 的分辨率**：$p$ 最小可达 $1/(n_{\text{perm}}+1)$，50 次置换永远报不出 $p<0.0196$；`n_permutations=0` 直接 `InvalidParameterError`。
18. **同一套折既选超参又报分数**：选择偏差，实测在「80 样本 × 1500 纯噪声 × 6 候选」上 **+0.0500**，在 breast cancer 上 **+0.0000**。

## 十、成本参考（1.6.1，本机）

| 动作 | 拟合次数 | 实测耗时（`SVC(kernel='linear')`，150 样本） |
| --- | --- | --- |
| 5 折 `cross_val_score` | 5 | 0.0035s |
| `LeaveOneOut` | 150 | 0.078s |
| `RepeatedKFold(5,3)` | 15 | 0.0083s |
| `permutation_test_score(n_permutations=50, cv=5)` | 255 | 0.165s |
| 6 候选 × 5 折（非嵌套） | 30 | 0.07s（`LogisticRegression`，breast cancer） |
| 嵌套 CV（5 外层 × 6 候选 × 5 内层 + 5 重训） | 155 | 毫秒级；模型一换就按比例放大 |

成本的主项永远是**模型拟合次数**，而拟合次数由 $k$、重复数、候选数、置换数四个整数相乘得到——这四项都应在实验日志里记录。

## 十一、可自动化程度

| 环节 | 可自动化 | 理由 |
| --- | --- | --- |
| 选 splitter | **低** | 五个准入问题需要领域知识（谁是「组」？部署时看到未来吗？） |
| `k` / 重复数 | 中 | 有默认值（5），但应由预算与噪声底反推 |
| 嵌套 CV | 高 | 纯机械，关键是外层内层划分不同 |
| 泄漏检查 | 高 | 把 CLI 里 `fit` 的步骤全部塞进 `Pipeline` 可机械检查 |
| 失败折处理 | **高（必须做）** | `error_score='raise'` + `nan` 计数是最便宜的护栏 |
| 报告六项 | 高 | 用一个函数固定输出 |
| 判断「提升是否显著」 | 低 | 折间 std 与标准误的换算来源未给出，只能记「未判定」 |
| 置换检验 | 高 | 有现成函数；只需记住 $p$ 的分辨率 |

## 关联页面

- 概念：[[cross-validation]]（机制与失效模式）、[[leaderboard-vs-cv]]（本地分数 vs 排行榜）、[[regularization-and-sparsity]]（内置 CV 与信息准则）、[[ensembles]]（Stacking 的折外预测）、[[hyperparameter-search]]（取最大值之后的偏差）
- 实践：[[feature-selection-workflow]]（泄漏实测与选择器写法）、[[competition-ml-workflow]]（竞赛流程里的 CV 位置）、[[evaluator-design-and-reward-hacking]]（代理指标被高频优化时的防线）、[[hyperparameter-search-and-tuning]]（内层搜索器的落地：空间、预算、诊断信号）
- 来源：[[sklearn-cross-validation]]（本页全部数字的出处）、[[kaggle-getting-started]]（人类选手如何用 CV 判断改进）
- 实践补充：[[metric-selection-and-thresholding]]（切分冻结之后：声明泛函、选评分、还原符号、单独定阈值）
- 工具：[[scikit-learn]]（`model_selection` 与 `metrics` 的默认值矩阵）
