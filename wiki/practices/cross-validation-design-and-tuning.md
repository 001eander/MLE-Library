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
  - "[[kaggle-getting-started]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-svm]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（API 默认值与文档，见 [[sklearn-cross-validation]]）
---

# 交叉验证的落地设计

一句话判据：**CV 写错不一定报错，只会给你一个回答错问题的分数。** 先冻结依赖结构，再选折数与重复，最后固定报告格式。

## 一、五个准入问题

| # | 问题 | 选择 |
| --- | --- | --- |
| 1 | 样本有组吗？ | `GroupKFold` / `GroupShuffleSplit`；需要类比例时考虑 `StratifiedGroupKFold` |
| 2 | 部署时用过去预测未来吗？ | `TimeSeriesSplit`，必要时 `gap>0` |
| 3 | 有稀有类吗？ | `StratifiedKFold`，并检查每折类计数 |
| 4 | 有外部给定划分吗？ | `PredefinedSplit` |
| 5 | 只评估还是要选参？ | 选参时使用 dev/eval 或嵌套 CV，不能只报搜索最优分 |

顺序是**分组优先于分层，分层优先于 i.i.d.**。`cv=int` 在分类器上使用 `StratifiedKFold`，与显式 `KFold` 不同（[[sklearn-cross-validation]], L89）。

## 二、选 splitter

| 场景 | 写法 | 要检查什么 |
| --- | --- | --- |
| 独立同分布、分类 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=…)` | 类比例、种子、是否有时序 |
| 普通回归 | `KFold` | 排序、组、时间结构 |
| 组独立 | `GroupKFold` | 测试组交集为零、折大小 |
| 组 + 稀有类 | `StratifiedGroupKFold` | 单类组与洗牌限制，见官方 Implementation notes |
| 时序 | `TimeSeriesSplit` | 扩张窗口、折大小、`gap` |
| 重复划分 | `RepeatedKFold` / `RepeatedStratifiedKFold` | 重复降低划分选择方差，不保证折间 std 更小 |
| 随机留出 | `ShuffleSplit` / `StratifiedShuffleSplit` | 测试重叠，不能用于 `cross_val_predict` |
| 官方协议 | `PredefinedSplit` | `test_fold` 映射与 −1 语义 |
| 逐样本 | `LeaveOneOut` | 拟合成本与单样本分数的量纲 |

## 三、折数与重复

- 5 或 10 折只是来源引用的经验起点（[[sklearn-cross-validation]], L304），不是通用最优。
- 折数影响训练集规模、折内分数离散与计算成本。
- 重复用于降低划分选择方差，不要把折间 std 当标准误。
- LOO 的 $n$ 个单样本分数与 K 折的折级分数不是同一统计量，不能直接比较 std。
- 预算按「候选数 × 折数 × 重复数 × 单次拟合成本」计算。

## 四、选参与嵌套

`best_score_` 是多个 CV 估计量的最大值，不是无偏估计。两种处理：

1. **dev/eval**：在开发集搜索，在独立评估集只评一次；
2. **嵌套 CV**：外层评估、内层选参。

不要把「不嵌套必然虚高」写成固定数值结论；偏差依赖数据、候选数与搜索强度。Stacking 的折外预测也依赖同一原则，见 [[ensembles]]。

## 五、所有会 `fit` 的步骤放进管线

缩放、目标编码、supervised 特征选择、分解器与估计器都必须在训练折内拟合。先在全量数据上执行再 CV 会让测试折参与训练（[[sklearn-cross-validation]], L117；[[preprocessing-pipeline-design]]）。

如果同一 CV 同时用于选参与构造 OOF 特征，需要额外审视复用偏差，不要把 `cross_val_predict` 当泛化误差度量（L197–L202）。

## 六、固定报告六项

1. 逐折分数数组；
2. splitter 完整构造与随机种子；
3. 每折大小与必要的类/组计数；
4. 失败折计数与原因；
5. 指标算在逐折、池化预测还是最终模型上；
6. 指标定义、响应方法、阈值与版本。

另报训练成本、选择偏差处理、数据/代码版本。`np.nanmean` 不应被用来悄悄隐藏失败折。

## 七、诊断信号

| 信号 | 优先检查 |
| --- | --- |
| 分数异常好 | 泄漏、组/时序错位、先选后 CV |
| 某折 `nan` | 缺类、失败拟合、`error_score` |
| 折间差异很大 | 数据顺序、稀有类、组大小、指标量纲 |
| 离线好上线差 | 部署结构、阈值与分布偏移 |
| 搜索最优远高于 eval | 选择偏差、候选过多、复用同折 |
| stacking 异常好 | OOF 泄漏或同折选参/造特征 |

## 常见默认值

- `cv=None` 默认 5 折；
- `shuffle=False` 时不应传 `random_state`；
- `error_score=np.nan` 可能隐藏失败；
- `return_train_score=False` 默认不报训练分；
- `return_indices=True` 返回 `indices` 字典；
- `train_test_split` 默认 `test_size=0.25`、`stratify=None`，不支持 `groups`。

## 可自动化程度

| 工作 | 自动化 | 说明 |
| --- | --- | --- |
| 依赖结构检查 | 中 | 可由组、时间、类别与外部协议驱动 |
| splitter 与报告生成 | 高 | 可模板化 |
| 泄漏检查 | 中 | 可检查 API，但语义仍需审查 |
| 折数与嵌套决策 | 中 | 依赖预算与选择过程 |
| 最终显著性与上线判断 | 低 | 需要独立目标与业务成本 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]、[[kaggle-getting-started]]
- 概念：[[cross-validation]]、[[hyperparameter-search]]、[[leaderboard-vs-cv]]
- 实践：[[feature-selection-workflow]]、[[preprocessing-pipeline-design]]、[[hyperparameter-search-and-tuning]]
