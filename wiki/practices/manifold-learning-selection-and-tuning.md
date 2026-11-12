---
title: "流形学习的选型与调参：先缩放、再定 $k$、最后挑被保量"
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - dimensionality-reduction
  - manifold-learning
  - scikit-learn
  - workflow
sources:
  - "[[sklearn-manifold-learning]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.8.0 – 1.9.1（文档论述与 API 签名，见 [[sklearn-manifold-learning]]）
---

# 流形学习的选型与调参

## 一、先判该不该用

适合：可视化、探索低维几何、需要非线性坐标或去噪先验。

不适合：

- 需要给新样本稳定定位：`SpectralEmbedding`、`TSNE`、`MDS`、`ClassicalMDS` 没有统一的 `transform`；
- 大样本且无法承受邻居图与分解成本；
- 需要每个维度有明确业务语义或概率输出；
- 需要跨运行同一坐标系，而不只是相似结构。

如果只是要监督预测，先比较线性模型、树或 MLP，不要把降维当必经步骤。

## 二、先缩放，再选被保量

邻域方法依赖 kNN 图，尺度决定邻居（[[sklearn-manifold-learning]], L176）。先统一尺度并把它放在训练折内。

| 目标 | 首选 | 不宜误用 |
| --- | --- | --- |
| 展开单个连续流形 | Isomap / LLE | 只用于局部簇时不一定合适 |
| 保留局部重构权重 | LLE 四变体 | 不是距离或全局结构的保证 |
| 图上的局部邻近 | SpectralEmbedding | 不保证远点被推开，无 `transform` |
| 全局距离/内积 | MDS / ClassicalMDS | raw stress、Stress-1、strain 不可混比 |
| 局部簇与多尺度可视化 | t-SNE | 全局结构弱、非凸、成本与近似限制 |

t-SNE 官方判据是偏局部结构，Isomap/LLE 更偏展开连续流形（[[sklearn-manifold-learning]], L155）。

## 三、定 $k$ 与输出维数

- $k$ 太小：图可能不连通；
- $k$ 太大：局部性被抹平；
- LLE/HLLE 有来源给出的下界，具体报错需目标版本核对；
- `SpectralEmbedding.n_neighbors=None` 默认约 $n/10$（API 外部核查），大样本要评估图成本。

L177 的「重构误差选维数」不是统一判据：各方法没有统一接口与方向。应分别检查方法支持的指标，并结合下游任务、稳定性和图，而不是套同一条曲线。

## 四、MDS 与 t-SNE 的专门检查

MDS：

- metric 保距离，non-metric 保序，classical 保内积；
- `stress_` 的量纲随配置变化；
- `n_init`、`eps`、`normalized_stress` 有版本差异，必须写版本。

t-SNE：

- KL 非凸，可按官方建议用不同种子并选最低 KL；
- 不用坐标差直接比较种子；
- `init='pca'` 可缓解全局结构问题；
- Barnes-Hut 的维数与输入限制要查官方折叠块；
- `learning_rate='auto'` 有 ÷4 与下限 50，不能简写为 $n/\text{early exaggeration}$（源码外部核查）。

## 五、评估与报告

- 局部：`trustworthiness` 等，但要知道它主要惩罚假邻居；
- 全局：另行评估距离/序结构，不能由局部指标推出全局保留；
- 图：检查连通性、噪声短路与异常面板；
- 接口：记录是否有 `transform`，测试集是否被重新布局；
- 参数：$k$、缩放、初始化、求解器、随机种子与版本。

## 常见坑

- 未缩放就建图；
- 把没有 `transform` 的方法放进普通推理管线；
- 用重构误差通用选维；
- 跨配置比较 `stress_`；
- 把 ARPACK 失败都归因于重复点；
- 用坐标差判断 t-SNE 好坏；
- 只看局部指标宣称全局结构；
- 用图标题时间当跨机基准；
- 用旧实验数字代替目标环境验证。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 尺度与接口检查 | 高 |
| $k$/维数/种子扫描 | 中–高 |
| 图与连通性诊断 | 中 |
| 全局/局部指标解释 | 中 |
| 新样本部署与语义判断 | 低 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-manifold-learning]]
- 概念：[[manifold-learning]]、[[matrix-factorization]]、[[nearest-neighbors]]
- 实践：[[preprocessing-pipeline-design]]、[[decomposition-selection-and-tuning]]、[[nearest-neighbors-selection-and-tuning]]
