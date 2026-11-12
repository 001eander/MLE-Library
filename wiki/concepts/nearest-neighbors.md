---
title: 最近邻方法（Nearest Neighbors）：非泛化学习、邻域定义与学到的度量
type: concept
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - nearest-neighbors
  - instance-based
  - metric-learning
  - scikit-learn
sources:
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-outlier-detection]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.7.x – 1.9.x 的 `sklearn.neighbors`（文档与 API 签名，见 [[sklearn-nearest-neighbors]]）
---

# 最近邻方法（Nearest Neighbors）

## 模型就是训练集

最近邻属于非泛化/基于实例的方法：不构造一般化内部参数，只保存训练实例并建立索引（[[sklearn-nearest-neighbors]], L12、L99）。

三个推论：

1. 训练主要是建索引，模型体积约等于训练数据；
2. 推理需要寻找邻居，成本随训练集与维度增长；
3. 可解释性是指出近邻样本，而不是读取全局系数。

`NearestCentroid` 是例外式的轻量基线：每类一个质心，等价于各类同方差假设的极限情形，可与 LDA/QDA 对照（L168；[[gaussian-discriminant-analysis]]）。

## 固定数量与固定半径

| | `KNeighbors*` | `RadiusNeighbors*` |
| --- | --- | --- |
| 邻域 | 固定 $k$ 个样本 | 固定半径 $r$ |
| 样本数 | 恒定 | 随局部密度变化 |
| 适用 | 均匀采样 | 非均匀采样 |
| 风险 | 稀疏区的邻居很远 | 高维或稀疏区可能空邻域 |

大 $k$ 平滑噪声但模糊边界；半径是局部自适应的 $k$，但失去固定计算量（L101–L105）。

## 尺度与权重是模型定义

度量不是无关实现细节：尺度决定邻居结构。官方正文没有完整缩放指导，但示例使用 `StandardScaler`，因此应把缩放纳入模型设计与折内管线（[[sklearn-preprocessing]]）。

`weights` 三档：uniform、distance、callable（L107、L121）。源码外部核查显示，`weights='distance'` 遇到零距离时使用「命中点权重 1、其余权重 0」的掩码，因此查询点与训练点重合时会退化为 1-NN。

并列也有两种：

- 距离并列：结果依赖训练数据顺序（L25）；
- 投票并列：`weighted_mode` 按升序标签处理，平票取最小标签（源码外部核查）。

## 三种搜索结构

| 结构 | 优势 | 边界 |
| --- | --- | --- |
| brute | 无索引成本，小样本有竞争力 | 全对距离，成本高 |
| KD tree | 低维轴向划分 | 高维退化，正文经验为 $D<20$ |
| ball tree | 三角不等式剪枝，结构化高维数据可受益 | 构造更贵，收益依赖结构 |

三种结构算同一个数学对象，原则上只改成本不改预测。正文的总成本与丢失折叠块的单次查询成本是不同口径（[[sklearn-nearest-neighbors]] 转换损失）。

`algorithm='auto'` 的官方折叠块给出经验规则：$D>15` 或 $k\ge N/2$ 时倾向 brute；它与正文的 $D<20$ 不是同一个阈值。

## 预计算邻居图

`KNeighborsTransformer` 可输出 connectivity 或 distance 图并缓存复用（L194–L236）。关键契约：

- 图可以作为其他估计器的输入；
- `distance` 模式在查询训练集时可能多请求一个邻居以处理自身；
- 预计算图有格式、邻居数与含自身语义的约束；
- 自定义 ANN 实现必须显式核对这些规则。

`NearestCentroid` 的 `shrink_threshold` 对质心做软阈值，可去除噪声特征；`NearestCentroid.priors` 的 docstring 默认说明存在内部矛盾，当前只保留分歧，不推断实现意图。

## NCA：学一个度量

NCA 学线性变换 $L$ 来优化随机 LOO-KNN 概率，对应平方 Mahalanobis 度量 $M=L^TL$（L240–L320，丢失折叠块外部核查）。它把「距离」本身变成学习对象。

来源只说目标是提升表现，官方示例也可能不如普通 KNN，因此 NCA 不是必然改进；应与未变换的 KNN 用同一切分对照。

## 常见失效模式

- 未缩放就使用距离模型；
- 把 $k$ 或 $r$ 当全局通用默认；
- 忽略零距离权重掩码；
- 把搜索结构当模型假设，或反过来以为它免费；
- 忽略训练顺序与标签编码的并列规则；
- 误读预计算图的含自身语义；
- 把 `NearestCentroid` 的「无参数」理解为没有任何结构选择；
- 把 NCA 当必然提升；
- 把 LOF 的局部密度分数当普通 KNN 分数（[[sklearn-outlier-detection]]）。

## 待解问题

- NCA 的多折对照与增益条件；
- 零距离掩码的目标版本行为；
- KD 与 `auto` 阈值来源；
- `leaf_size` 的成本拐点；
- ball tree 的收益条件；
- `NearestCentroid.priors` 的实际语义；
- 预计算图需要 $k+1$ 的下游估计器；
- callable 权重下的平票规则。

运行时问题只在 `uv` 目标版本环境中复核。

## 关联页面

- 来源：[[sklearn-nearest-neighbors]]、[[sklearn-outlier-detection]]
- 概念：[[gaussian-discriminant-analysis]]、[[manifold-learning]]、[[anomaly-detection]]
- 实践：[[nearest-neighbors-selection-and-tuning]]、[[preprocessing-pipeline-design]]
