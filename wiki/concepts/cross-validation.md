---
title: 交叉验证（Cross-Validation）：把分数当成估计量
type: concept
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - cross-validation
  - model-selection
  - evaluation
sources:
  - "[[sklearn-cross-validation]]"
  - "[[kaggle-getting-started]]"
  - "[[sklearn-linear-models]]"
  - "[[sklearn-svm]]"
  - "[[sklearn-ensembles]]"
  - "[[sklearn-feature-selection]]"
  - "[[aide]]"
  - "[[autoresearch]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.x – 1.9.1（用户指南 3.1 节与 API 签名，见 [[sklearn-cross-validation]]）
---

# 交叉验证（Cross-Validation）

## CV 报出的是一组折内结果

$k$ 折 CV 把每个样本轮流放入测试折，拟合 $k$ 个模型并计算 $k$ 个分数，最后报告均值（[[sklearn-cross-validation]], L47–L52）。因此结果不是「真实泛化误差」本身，而是依赖数据、切分器、模型与指标的估计量。

折间标准差描述折级结果的离散程度，**不是自动等于 CV 估计量的标准误**。折间训练集重叠、分层会人为压小离散度，因此不能把「超过一个 std」当作通用显著性判据（[[sklearn-cross-validation]], L361–L365）。

## 三种调用式对应三种产品

| 调用式 | 返回 | 用途 | 不能误解为 |
| --- | --- | --- | --- |
| `cross_val_score` | 每折一个分数 | 快速得到折内分数 | 已审计全部训练过程 |
| `cross_validate` | 分数、时间，可选训练分、折内模型与索引 | 多指标与过程审计 | 所有返回键固定不变 |
| `cross_val_predict` | 每个样本的折外预测 | 可视化、构造 stacking/blending 特征 | 泛化误差度量（[[sklearn-cross-validation]], L197–L202） |

池化折外预测后计算指标，与逐折计算再平均，在非线性指标上可能不同。`return_indices=True` 返回的是 `indices` 字典，不是两个裸 `train_indices`/`test_indices` 键。

## Splitter 是依赖结构声明

| 依赖结构 | 代表 | 主要风险 |
| --- | --- | --- |
| i.i.d. | `KFold`、`RepeatedKFold`、`LeaveOneOut`、`LeavePOut`、`ShuffleSplit` | 样本相关时回答错问题；不洗牌且有序时可能崩塌 |
| 类比例 | `StratifiedKFold`、`StratifiedShuffleSplit` | 保持比例但可能压小折间方差 |
| 组独立 | `GroupKFold`、`StratifiedGroupKFold`、`LeaveOneGroupOut`、`LeavePGroupsOut`、`GroupShuffleSplit` | 同组泄漏；折大小可能不齐 |
| 时间顺序 | `TimeSeriesSplit` | 未来信息泄漏；`gap` 与扩张窗口需要显式设计 |
| 外部给定 | `PredefinedSplit` | 错误映射外部协议；`test_fold` 的含义需核对 |

`cv=int` 在分类器上使用 `StratifiedKFold`，与显式 `KFold` 不同（[[sklearn-cross-validation]], L89）。`KFold(n_splits=n)` 等价于 LeaveOneOut（L230）。

## 分层不是统计修复

来源明确说 stratification 是为工程问题而生的补丁，可能人为压小折间变异（L361–L365）。它保证类比例近似存在，却不保证稀有类可分，也不使折间方差自动成为不确定性的可靠估计。

分组优先于分层：如果同一患者、用户、设备或文档横跨折，模型会先学到组身份。`GroupKFold(shuffle=True)` 平衡组数而不是样本数（L454）。

## 泄漏与选择偏差

### 泄漏

所有会 `fit` 的步骤都必须在训练折内完成，包括缩放、目标编码和 supervised 特征选择。先在全量数据上拟合再 CV，会让测试折参与训练过程。L117 的折叠块只剩代码，官方说明需引 [[sklearn-cross-validation]] 的转换损失与 [[preprocessing-pipeline-design]]。

### 选择偏差

用同一批折选超参再报告最优分数，得到的是多个估计量的最大值，而不是无偏泛化估计。开发集/评估集切分与嵌套 CV 是两种处理路径（[[sklearn-hyperparameter-tuning]], L284–L290）。偏差幅度依赖数据与搜索过程，不能假设固定数值。

## 失败、洗牌与默认行为

- `error_score=np.nan` 是 API 默认：失败折可能只留下 `nan`；`np.nanmean` 会把失败折隐藏。需要显式失败时用 `error_score='raise'`。
- `cv=None` 默认 5 折（0.22 起），不是「没有切分」。
- `KFold(shuffle=False, random_state=...)` 是无效组合，应显式检查。
- 有序非时序数据可能需要洗牌；时序数据洗牌会把未来泄入训练。
- 搜索器在一次 `fit` 内对所有候选复用同一套切分（L614），但跨调用不是同一协议。
- `train_test_split` 的 `stratify` 默认 `None`，并不自动分层，也不支持 `groups`。

## 置换检验能回答什么

`permutation_test_score` 的零假设是估计器没有利用特征与目标的统计依赖（L625–L631）。高 p 值既可能表示无依赖，也可能表示欠拟合；低 p 值只说明稳定优于随机猜测，不说明模型足够好。

成本为 $(n_{\text{perm}}+1)\cdot n_{\text{cv}}$ 次拟合（L638），最小 p 值为 $1/(n_{\text{perm}}+1)$。它不是效应量，也不替代独立测试集。

## 常见失效模式

- 只报均值，不报逐折数组与 splitter 构造；
- 把折间 std 当标准误；
- 分组、时序或稀有类结构未声明；
- 在全量数据上先做预处理或选择；
- 把 `cross_val_predict` 当泛化分数；
- 忽略失败折和 `nan`；
- 用同一 CV 选参、报分并构造 stacking 特征而不审视复用偏差；
- 把置换 p 值当效果大小。

## 待解问题

- 折间 std 与估计量标准误的可用换算；
- 重复次数与划分选择方差；
- 分层与分组的优先序；
- `gap` 与自相关长度；
- 同一 CV 选参与 OOF 特征构造的双重使用；
- `PredefinedSplit` 多折表示与报告规范。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-cross-validation]]、[[sklearn-hyperparameter-tuning]]、[[kaggle-getting-started]]
- 概念：[[hyperparameter-search]]、[[evaluation-metrics]]、[[leaderboard-vs-cv]]
- 实践：[[cross-validation-design-and-tuning]]、[[feature-selection-workflow]]、[[preprocessing-pipeline-design]]
