---
title: 知识索引
type: index
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - index
---

# 知识索引

每当新增、重命名或实质修改页面时，图书馆员都会更新此目录。每个条目包含一个链接和一句话摘要。

## 开始

- [[overview|MLE Library 概览]] — 用途、分层结构与阅读路径。
- [[catalog|动态目录]] — 使用 Dataview 展示 wiki 页面及其时效性元数据的表格。

## 来源

- [[aide|AIDE：AI-Driven Exploration in the Space of Code]] — 把 MLE 改写为代码空间的解树搜索，含基准结果、实现边界与内部数字分歧。
- [[alphaevolve|AlphaEvolve：由 Gemini 驱动的算法设计编码智能体]] — 进化式程序数据库与自动评估器；结果均为厂商自述，缺少实现和版本。
- [[autoresearch|autoresearch：让智能体夜间自主跑单 GPU 训练实验]] — 三文件契约、固定 5 分钟墙钟与 `val_bpb`，含进度图目视核查与噪声选择问题。
- [[hyra-1-0|Hyra-1.0：腾讯混元的递归自我改进科研智能体]] — 轻量 harness、经验库与双层评估循环，覆盖多类科研场景。
- [[hyra-2-0|Hyra 2.0：从解题到造工具]] — 多项目长周期 harness、分层记忆、知识分级与工具化科研成果。
- [[kaggle-getting-started|Kaggle 入门指南：竞赛型 MLE 的通用流程与一次完整赛题复盘]] — 七段流程、XGB 调参、Stacking 与 CV/LB 张力，代码绑定旧 API。
- [[kaggle-q-ft|Q-FT：把表格行当嵌入喂给 Qwen 的 Tabular Transformer]] — 21 列→22 个位置、三段式嵌入与 RMSNorm、LoRA 10.1M、重排增强；含 Qwen 配置与参数账本手算复核。
- [[kaggle-q-ft-v2|Q-FT v2：把表格行读成集合的 order-free ModernBERT]] — `position_ids=0`、side features、RTD 自监督预训练；含 149M 参数复核与单折/OOF 偏差。
- [[kaggle-tfms-llms-tabular|TFMs and LLMs with tabular data：S6E10 第一名方案]] — 把整行写成文本，target 放末尾（判别式）或句首（生成式）；含四步配方、11 张目视图表、245 对 pair 的 stacking 协议与跨来源基线冲突。
- [[ft-transformer|Revisiting Deep Learning Models for Tabular Data（FT-Transformer）]] — 同协议比较 9 个表格 DL 模型与 2 个 GBDT；ResNet/FT-T 基线、注意力重要性、合成 α 实验与官方实现核查。
- [[sklearn-linear-models|scikit-learn 用户指南 1.1「Linear Models」]] — 线性估计器与求解器约定；记录严重折叠块丢失、公式空缺与外部核查。
- [[sklearn-lda-qda|scikit-learn 用户指南 1.2「Linear and Quadratic Discriminant Analysis」]] — 生成式分类、Mahalanobis 几何、降维与协方差收缩；保留参考文献丢失等分歧。
- [[sklearn-naive-bayes|scikit-learn 用户指南 1.9「Naive Bayes」]] — 条件独立、五类变体、平滑与增量拟合；含授权回填记录及 `partial_fit` 分歧。
- [[sklearn-svm|scikit-learn 用户指南 1.4「Support Vector Machines」]] — 最大间隔、核技巧、C/ν 与 libsvm/liblinear；含回填及 `probability` 弃用分歧。
- [[sklearn-nearest-neighbors|scikit-learn 用户指南 1.6「Nearest Neighbors」]] — 非泛化学习、邻域、搜索结构、预计算图、NearestCentroid 与 NCA。
- [[sklearn-decision-trees|scikit-learn 用户指南 1.10「Decision Trees」]] — 递归划分、贪心切分、缺失值与代价复杂度剪枝；含静态资产与版本分歧。
- [[sklearn-ensembles|scikit-learn 用户指南 1.11「Ensembles」]] — HGBT/GBDT、RF/ET、Bagging、Voting、Stacking 与 AdaBoost；记录大量数学和文献丢失。
- [[sklearn-feature-selection|scikit-learn 用户指南 1.13「Feature selection」]] — 过滤式、包裹式与嵌入式选择，阈值契约与 Pipeline 防泄漏。
- [[sklearn-neural-networks|scikit-learn 用户指南 1.17「Neural network models (supervised)」]] — MLP 的输出面、求解器、正则与非凸训练；保留折叠块缺失和 Poisson API 分歧。
- [[sklearn-manifold-learning|scikit-learn 用户指南 2.2「Manifold learning」]] — Isomap、LLE 四变体、Spectral、MDS 与 t-SNE；结构完整但数学折叠块丢失。
- [[sklearn-clustering|scikit-learn 用户指南 2.3「Clustering」]] — 聚类方法族与评估指标；记录结构性剪藏损失、doctest 口径和版本分歧。
- [[sklearn-decomposition|scikit-learn 用户指南 2.5「Decomposing signals in components」]] — PCA、字典学习、FA、ICA、NMF 与 LDA；约束、成分和接口差异。
- [[sklearn-covariance|scikit-learn 用户指南 2.6「Covariance estimation」]] — 经验协方差、LW/OAS、GraphicalLasso 与 MCD；文献丢失、目标口径与属性语义。
- [[sklearn-outlier-detection|scikit-learn 用户指南 2.7「Novelty and Outlier Detection」]] — outlier/novelty 两类任务、四种形状假设与 `offset_` 阈值契约。
- [[sklearn-cross-validation|scikit-learn 用户指南 3.1「Cross-validation」]] — 三种调用式、splitter 依赖结构、泄漏与置换检验；折叠块和实现说明缺失。
- [[sklearn-hyperparameter-tuning|scikit-learn 用户指南 3.2「Tuning the hyper-parameters of an estimator」]] — 网格、随机与逐次减半的预算曲线、`refit` 与失败语义。
- [[sklearn-metrics|scikit-learn 用户指南 3.4「Metrics and scoring」]] — 目标泛函与严格一致评分；仅保留 4/54 小节，指标本体严重缺失。
- [[sklearn-preprocessing|scikit-learn 用户指南 8.3「Preprocessing data」]] — 有状态预处理契约、缩放、编码、展开与目标编码交叉拟合；公式折叠块丢失。

## 概念

- [[cross-validation|交叉验证（Cross-Validation）：把分数当成估计量]] — 折内结果、splitter 依赖假设、三种调用式、泄漏、选择偏差与置换检验。
- [[hyperparameter-search|超参数搜索（Hyperparameter Search）：把预算分配给候选]] — 三条预算曲线、参数空间、拟合次数、`best_score_` 与 `refit`。
- [[evaluation-metrics|评估指标与评分函数（Metrics and Scoring Functions）]] — 目标泛函、严格一致性、三层 API、符号与决策阈值。
- [[feature-preprocessing|特征预处理（Feature Preprocessing）]] — 有状态变换、作用轴、尺度/编码/展开的取舍与折内管线纪律。
- [[clustering|聚类（Clustering）]] — 几何假设、簇数来源、归纳/传导断层与指标的机会校正和对称性。
- [[matrix-factorization|矩阵分解与成分分解（Matrix Factorization / Component Decomposition）]] — $X\approx UV$ 的约束、成分方向、预处理与非唯一性。
- [[covariance-estimation|协方差估计（Covariance Estimation）]] — $\Sigma$ 与 $K$、收缩、稀疏精度矩阵、MCD 与不同损失口径。
- [[anomaly-detection|异常检测（Anomaly / Novelty / Outlier Detection）]] — 两类任务、形状假设、分数阈值与 LOF 双模式。
- [[code-space-search|代码空间搜索与解树（Code-Space Search）]] — 解即程序、边即改进尝试；原子改动、摘要与五种组织方式。
- [[tool-centric-research|工具中心的研究（Tool-Centric Research）]] — 把产出单位从答案换成工具，讨论符号化载体与复用经济性。
- [[recursive-self-improvement|递归自我改进（RSI）]] — scaffold–data–model 闭环、四层循环与瓶颈分类。
- [[research-memory-tiers|研究记忆分层与知识成熟度分级]] — 项目状态、agent 记忆、跨任务 Books，以及观察/已验证/约定三级。
- [[experience-bank|经验库（Experience Bank）]] — 结构化 solution/日志/评分与多样化灵感的合成机制。
- [[evaluator-solution-co-evolution|评估器与解的共进化]] — 无现成评估器时的双层循环，以及独立复核后的新瓶颈。
- [[leaderboard-vs-cv|排行榜与本地验证（Leaderboard vs CV）]] — 本地 CV、Public LB、Private LB 的分工与切分复刻前提；单折试点与全折 OOF 的口径差。
- [[regularization-and-sparsity|正则化与稀疏性（Regularization and Sparsity）]] — 惩罚、收缩、平滑、结构约束与随机化预算；区分稀疏与重要性。
- [[gaussian-discriminant-analysis|高斯判别分析（Gaussian Discriminant Analysis）]] — LDA/QDA/GaussianNB 的假设嵌套、Mahalanobis 几何与参数代价。
- [[naive-bayes|朴素贝叶斯（Naive Bayes）]] — 条件独立的收益与代价、五类变体、平滑与概率估计边界。
- [[generative-classification|生成式分类（Generative Classification）]] — 用 $P(x\mid y)$ 反推 $P(y\mid x)$：代数、三个实例（NB/LDA/文本化 GPT）、证据可加性、$K\times O$ 次前向的成本与失效边界。
- [[support-vector-machines|支持向量机（Support Vector Machines）]] — 最大间隔、hinge、对偶核、样本维稀疏、C/ν 与输出面。
- [[nearest-neighbors|最近邻方法（Nearest Neighbors）]] — 非泛化学习、邻域、搜索结构、预计算图、质心与 NCA。
- [[decision-trees|决策树（Decision Trees）]] — 分段常数、轴对齐划分、贪心不稳定性、缺失值与结构剪枝。
- [[ensembles|集成方法（Ensembles）]] — 平均降方差、顺序降偏差、GBDT/RF/Stacking 与随机化预算。
- [[feature-selection|特征选择（Feature Selection）]] — 三家族、代理指标假设、阈值/个数、成本与选择稳定性。
- [[multi-layer-perceptron|多层感知机（Multi-layer Perceptron, MLP）]] — 参数化非线性、输出面、求解器、非凸优化与 L2。
- [[manifold-learning|流形学习（Manifold Learning）]] — 被保量、kNN 图、坐标输出、MDS/t-SNE 与接口边界。
- [[tabular-deep-learning|表格数据的深度学习（Tabular Deep Learning）]] — 把每一列变成一个 token：Feature Tokenizer、PreNorm、默认配置、预处理契约与失败模式；含 LLM 骨干、LoRA 与 order-free 三条路线的对照。

## 实践

- [[cross-validation-design-and-tuning|交叉验证的落地设计：先定依赖结构，再定折数，最后定报告格式]] — splitter 判定、嵌套、泄漏检查、失败折与固定报告格式。
- [[hyperparameter-search-and-tuning|超参数搜索的落地流程：先冻结协议、再选搜索器、最后按拟合次数算预算]] — 空间、搜索器、预算、refit 与独立评估。
- [[metric-selection-and-thresholding|指标选择与阈值落地的流程：先声明目标泛函，再选严格一致的评分，最后单独定阈值]] — 评分映射、响应方法、主指标护栏与阈值四步法。
- [[research-agent-harness|科研智能体 Harness 设计]] — 树搜索、单循环、多项目、进化种群与单文件贪心五种形态。
- [[evaluator-design-and-reward-hacking|评估器设计与 reward hacking 防护]] — 作弊案例、固定验证、独立复核、噪声选择与证据纪律。
- [[competition-ml-workflow|竞赛型 MLE 工作流（Kaggle 类任务）]] — 七段流程、时效性边界、CV/LB 张力、`raw features only` 与特征工程的替代关系。
- [[linear-models-as-baselines|把线性模型当基线：预处理约束、正则强度选择与常见坑]] — 尺度、共线性、内置 CV/IC、LDA/QDA、线性 SVM 与 NearestCentroid。
- [[svm-selection-and-tuning|SVM 的选型与调参]] — 规模选实现、缩放、C/γ、概率迁移、多类与成本。
- [[nearest-neighbors-selection-and-tuning|最近邻的选型与调参]] — 缩放、k/r、权重、搜索结构、预计算图与 NCA 对照。
- [[decision-trees-selection-and-tuning|决策树的选型与调参]] — 前剪枝、`ccp_alpha`、缺失值、权重与 float32。
- [[ensembles-selection-and-tuning|集成模型的选型与调参]] — HGBT/GBDT/RF/ET/Bagging/Voting/Stacking/AdaBoost 的实现与预算。
- [[feature-selection-workflow|特征选择的落地流程：三类方法怎么选、阈值与个数怎么定、常见坑与可自动化程度]] — 选择器、阈值、折内管线、成本与稳定性。
- [[mlp-selection-and-tuning|MLP 的选型与调参：先缩放、再定容量、最后选求解器]] — 架构、α、求解器、早停、种子与输出阈值。
- [[manifold-learning-selection-and-tuning|流形学习的选型与调参：先缩放、再定 $k$、最后挑被保量]] — 接口、邻域、输出维数、评估与新样本边界。
- [[clustering-selection-and-tuning|聚类的选型与调参：先缩放与度量、再选族、最后定 $k$/$eps$]] — 几何假设、参数扫描、内外部评估与稳定性。
- [[decomposition-selection-and-tuning|成分分解的选型与调参：先问「要方差、要零件、还是要主题」]] — 预处理契约、$k$、初始化、输出方向与保存。
- [[covariance-estimation-selection-and-tuning|协方差估计的选型与调参：先判 n/p、再定 α 的来源、最后选损失口径]] — 收缩、GraphicalLasso、MCD、结构与似然目标。
- [[outlier-detection-selection-and-tuning|异常检测的选型与调参：先判污染与是否要上线，再选形状假设，最后定阈值]] — 形状、容量旋钮、阈值五步与评估路径。
- [[preprocessing-pipeline-design|预处理管线的落地设计：顺序、默认值、诊断信号与常见坑]] — 七阶段顺序、默认值、折内拟合与诊断。
- [[tabular-model-selection|表格任务选型：先立 GBDT 与 ResNet 两个基线，再决定要不要上 FT-Transformer]] — 五个准入问题、基线顺序、默认/调参四格、评估协议、LLM 骨干作为第五候选与 14 条坑。

## 工具

- [[aide-scaffold|AIDE 脚手架（aideml）]] — 解树、三算子、摘要算子、基线用法与版本风险。
- [[autoresearch-repo|autoresearch 仓库]] — 三文件契约、快速开始、小机器降规模与使用边界。
- [[hyra|Hyra（Hunyuan Research Agent）]] — 1.0 单循环与 2.0 多项目架构、结果形态和证据边界。
- [[scikit-learn|scikit-learn（工具页）]] — 十八篇剪藏的模块矩阵、官方默认值、版本边界与 `uv` 验证纪律。

## 比较

- [[hyra-1-0-vs-2-0|Hyra 1.0 与 2.0 对照：解题范式 vs 工具范式]] — 目标、组织、记忆、验证、结果形态与证据强度对照。
- [[gbdt-vs-tabular-dl|GBDT 与表格深度模型：没有普适赢家]] — 默认 vs 调参、逐数据集胜负、成本、机制、基准偏差，以及人工特征收益的 4 倍差（n=1）。

## 研究

- [[automated-research-open-questions|自动化研究待验证问题]] — AIDE、AlphaEvolve、autoresearch 与 Hyra 的可信度、机制假设、工程风险和待补资料。
