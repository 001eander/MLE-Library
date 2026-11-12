---
title: 最近邻的选型与调参：缩放前提、k 与搜索结构、预计算图与 NCA 的对照
type: practice
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - nearest-neighbors
  - instance-based
  - metric-learning
  - workflow
sources:
  - "[[sklearn-nearest-neighbors]]"
  - "[[sklearn-linear-models]]"
status: draft
verified: 2026-11-11
applies_to:
  - scikit-learn 1.7.x – 1.9.x 的 neighbors 模块（文档与 API 签名，见 [[sklearn-nearest-neighbors]]）
---

# 最近邻的选型与调参

## 一、先判该不该用

适合：小中样本、低维、非规则边界、需要实例解释或在线近邻查询。

不适合：训练集很大、维数很高、需要紧凑全局模型、稳定概率或低成本外推。先比较 `NearestCentroid`、LDA/QDA 与线性基线。

KNN 的模型就是训练集，训练建索引、推理找邻居；模型体积与推理成本必须一起评估（[[sklearn-nearest-neighbors]], L12、L99）。

## 二、缩放是模型定义

距离决定邻居，量纲改变模型。将 `StandardScaler` 等放在训练折内，并检查度量、稀疏性和缺失值；不要把「非参数」理解为不用预处理（[[preprocessing-pipeline-design]]）。

## 三、定 $k$、$r$ 与权重

- $k$：固定邻居数，越大越平滑但边界模糊；
- $r$：固定半径，邻居数随密度变，可能空邻域；
- `weights='uniform'`：多数票/均值；
- `weights='distance'`：近者权重大；源码显示零距离时退化为 1-NN；
- callable：自定义权重，需单独验证平票与零距离。

$k$ 同时影响容量、查询成本与 `auto` 后端选择。不要只按默认 $k=5$ 评估一次就结束。

## 四、搜索结构只改成本

| 结构 | 用法 |
| --- | --- |
| brute | 小样本或高维时可直接用 |
| KD tree | 低维轴向结构，正文经验 $D<20$ |
| ball tree | 结构化高维数据可受益于三角不等式 |

`algorithm='auto'` 的官方折叠块还给出 $D>15$ 或 $k\ge N/2$ 时倾向 brute；与正文阈值不是同一口径。`leaf_size` 影响索引成本与内存，不改变数学对象，需按目标环境扫描。

## 五、预计算图与 NearestCentroid

`KNeighborsTransformer` 可缓存 connectivity/distance 图。自定义时要检查：

- 是否含自身；
- `distance` 模式是否多请求一个邻居；
- 输出形状与格式；
- 下游接受 connectivity 还是 distance；
- 训练/推理是否使用同一图定义。

`NearestCentroid` 是轻基线，`shrink_threshold` 可做原型软阈值。`priors` 的 docstring 默认说明存在矛盾，不要未核对就依赖它。

## 六、NCA 必须做同切分对照

NCA 学 $L$ 与 $M=L^TL$，目标是改善随机 LOO-KNN（L240–L320）。正确比较：

1. 同一预处理与同一 splitter；
2. KNN vs NCA + KNN；
3. 报多折结果与成本；
4. 不用「 aims to improve」当保证，官方示例也可能更低。

## 常见坑

- 未缩放；
- 忽略零距离权重掩码；
- 忽略距离并列的训练顺序与投票并列的标签顺序；
- 把搜索结构当模型选择；
- 误读预计算图含自身语义；
- 把 `NearestCentroid` 的「无参数」当无结构选择；
- 把 NCA 当必然改进；
- 把 LOF 的局部密度分数当普通 KNN 输出（[[sklearn-outlier-detection]]）。

## 可自动化程度

| 工作 | 自动化 |
| --- | --- |
| 缩放与接口检查 | 高 |
| $k$/r/权重/后端扫描 | 高 |
| 图与并列规则检查 | 中 |
| NCA 对照与成本评估 | 中 |
| 最终实例解释与上线 | 低–中 |

运行时对照只在 `uv` 目标版本环境中执行，不沿用历史本机实验。

## 关联页面

- 来源：[[sklearn-nearest-neighbors]]、[[sklearn-outlier-detection]]、[[sklearn-linear-models]]
- 概念：[[nearest-neighbors]]、[[gaussian-discriminant-analysis]]、[[manifold-learning]]
- 实践：[[preprocessing-pipeline-design]]、[[manifold-learning-selection-and-tuning]]、[[linear-models-as-baselines]]
