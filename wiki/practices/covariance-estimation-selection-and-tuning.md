---
title: 协方差估计的选型与调参：先判 n/p、再定 α 的来源、最后选损失口径
type: practice
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - covariance
  - shrinkage
  - precision-matrix
  - robust-statistics
  - workflow
sources:
  - "[[sklearn-covariance]]"
status: draft
verified: 2026-11-12
applies_to:
  - scikit-learn 1.7.0 – 1.9.1（文档论述与 API 签名，见 [[sklearn-covariance]]）
---

# 协方差估计的选型与调参

## 一、五个准入问题

1. 要 $\Sigma$ 还是 $K=\Sigma^{-1}$？
2. 要下游似然、矩阵误差、条件独立图，还是离群点诊断？
3. $n$ 与 $p$ 的比例如何，是否接近奇异？
4. 是否有离群点或污染？
5. 结果是直接使用，还是接 LDA/QDA 或其他模型？

只有条件独立图需要 $K$ 上的 L1；只做异常检测则还要设计阈值，而不是估计完协方差就结束。

## 二、按目标选估计器

| 目标 | 首选路径 | 必须检查 |
| --- | --- | --- |
| 大样本、低污染的协方差 | `EmpiricalCovariance` | $1/n$ 归一化、奇异性、离群点 |
| 一般收缩 | `ShrunkCovariance` | α 只是手设起点，默认 0.1 不具最优含义 |
| 解析收缩 | `LedoitWolf` / `OAS` | 目标是 $\Sigma$ 的 MSE；OAS 还高斯假设 |
| 条件独立图 | `GraphicalLasso` | 标准化、α 路径、收敛与结构误差 |
| CV 驱动的精度矩阵 | `GraphicalLassoCV` | 似然可能选择过多边；检查支持集与 NLL |
| 离群点污染 | `MinCovDet` | $n>p$、`raw_support_` 与重加权 `support_` |
| 异常检测 | `EllipticEnvelope` | 继承 MCD，再单独设计 `contamination` 阈值 |

## 三、收缩的三条路径

1. **手设 α**：用作基线或领域先验，报告选择理由；
2. **LW/OAS 解析 α**：用于协方差 MSE，不能直接当 held-out NLL 或分类准确率最优；
3. **CV/下游目标选 α**：明确优化的是似然、矩阵误差、支持集还是最终模型指标。

各向同性 note 只说最优目标可能接近单位阵倍数，不保证 α 随 $n$ 单调变化（[[sklearn-covariance]], L46–L51）。

## 四、GraphicalLasso 的五步

1. 检查并必要时就地标准化，避免量纲主导；
2. 设置 α 路径并记录网格，不把默认 `alphas=4` 当 α 的取值列表；
3. 将 `ConvergenceWarning` 当诊断信号，检查共线与过小 α；
4. 同时报告边数、支持集指标、$\Sigma$ 误差与 held-out NLL；
5. 若使用 CV，把它视为目标之一，不把 CV 当选结构恢复保证。

$K$ 稀疏对应条件独立，$\Sigma$ 稀疏不是同一件事；收缩不能替代 L1 的非对角零结构。

## 五、MCD 与异常检测

- 使用 `raw_location_`/`raw_covariance_` 还是校正后的 `location_`/`covariance_`，必须写明；
- `raw_support_` 才对应最初的 $h$ 个支持样本；
- `support_` 是重加权掩码，不能直接当 $h$；
- 检查 $n>p$ 与目标版本的报错；
- 下游 `EllipticEnvelope` 还需单独确定异常阈值，不能把 Mahalanobis 距离直接当概率。

## 六、损失口径与报告

至少区分：

- $\Sigma$ 的 MSE/Frobenius 误差；
- held-out NLL；
- 支持集 precision/recall/F1；
- Mahalanobis 分离或下游任务表现。

报告估计器构造、$n/p$、预处理、α 来源、随机种子、是否奇异、收敛警告、支持集定义与最终损失。不要用一次实验的某个口径宣布「最优」。

## 常见坑

- 把 `assume_centered=True` 当自动居中；
- 把 $1/n$ 与 $1/(n-1)$ 混用；
- 把 LW/OAS 的 MSE 最优当下游最优；
- 把 $\Sigma$ 稀疏与 $K$ 稀疏混用；
- 忽略 CV 与结构恢复的目标冲突；
- 把 `support_` 当 $h$；
- 忽略伪逆与奇异分数；
- 用历史本机实验的数字代替目标环境验证。

## 关联页面

- 来源：[[sklearn-covariance]]、[[sklearn-outlier-detection]]
- 概念：[[covariance-estimation]]、[[regularization-and-sparsity]]、[[anomaly-detection]]
- 实践：[[linear-models-as-baselines]]、[[outlier-detection-selection-and-tuning]]、[[preprocessing-pipeline-design]]
