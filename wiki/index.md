---
title: 知识索引
type: index
created: 2026-11-11
updated: 2026-11-11
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

- [[aide|AIDE：AI-Driven Exploration in the Space of Code（arXiv 2502.13138v1）]] — 把 MLE 改写为代码空间的解树搜索；自建 Weco-Kaggle 与 MLE-Bench / RE-Bench 汇总，含「4 倍」口径与 63 vs 80 竞赛列表的内部不一致。
- [[alphaevolve|AlphaEvolve：由 Gemini 驱动的算法设计编码智能体（Google DeepMind 官方博客）]] — 进化式程序数据库 + 自动评估器：Borg 调度、TPU Verilog、Gemini 内核与 FlashAttention、48 次乘法与 593 个外层球；全部为厂商自述，无实现、无版本。
- [[autoresearch|autoresearch：让智能体夜间自主跑单 GPU 训练实验（karpathy 仓库 README 剪藏）]] — 三文件契约（人改 `program.md`、agent 改 `train.py`、`prepare.py` 禁改）、固定 5 分钟墙钟与 `val_bpb`；进度图 83 次实验 / 15 次保留已目视核查，含 `random seed` 入选一类噪声信号。
- [[hyra-1-0|Hyra-1.0：腾讯混元的递归自我改进科研智能体（官方博客）]] — 轻量 harness + 经验库 + 双层评估循环，覆盖 AI for AI / Science / Fun 十余场景的自评结果。
- [[hyra-2-0|Hyra 2.0：从解题到造工具（腾讯混元官方博客）]] — 多项目长周期 harness（Overall/Mission/Worker）、分层记忆与知识分级，以及 HyInv、HyPLM、HyFF、HyProver、HyQEC、旋梯手、香山七组成果。
- [[kaggle-getting-started|Kaggle 入门指南：竞赛型 MLE 的通用流程与一次完整赛题复盘（知乎专栏剪藏）]] — 人类选手的七段流程（EDA→…→Pipeline）、XGB 调参五步、5-Fold Stacking 实现、20 种子平均 +0.0004 的自认可疑结果；代码绑定 2016–2017 年 API，已附现代等价物对照。
- [[sklearn-linear-models|scikit-learn 用户指南 1.1「Linear Models」（stable 文档剪藏）]] — OLS/Ridge/Lasso/ElasticNet/LARS/OMP/贝叶斯/GLM/稳健/分位数/多项式共 13 节的估计器与求解器约定；剪藏残缺严重（折叠块全缺 29.3%、内部引用文字丢失、1.1.14 由所有者手工补入且仍有 3 处公式空缺），已附逐项外部核查表与 18 张图的目视核查。
- [[sklearn-lda-qda|scikit-learn 用户指南 1.2「Linear and Quadratic Discriminant Analysis」（stable 文档剪藏）]] — 生成式一侧：类条件高斯 + 贝叶斯、Mahalanobis 几何与 K−1 维降维、shrinkage（Ledoit-Wolf）/OAS 与三个求解器；正文完整但**三条参考文献全丢**（被 4 段重复正文冒充）、4 处引用标记被吞；版本判定到 1.9.x（QDA 公式 1/n），已附两张图的目视读数。
- [[sklearn-naive-bayes|scikit-learn 用户指南 1.9「Naive Bayes」（stable 文档剪藏，含所有者授权的 raw 回填）]] — 条件独立 + MAP 框架、五个变体各自的 $P(x_i\mid y)$、平滑 α（Laplace/Lidstone）与外存 `partial_fit`；抓取时**丢了 5 个 `<details>` 折叠块（318 词 ≈ 21.4%）**——CNB 权重公式与 Categorical 概率公式两个数学节、三个 References 块——已按所有者授权回填（原始 sha256 `5e0e5299…` → 现 `85c4050b…`，行号全库重写）；外部核查补 7 个文档未提的默认值与一处「`partial_fit` 实际五个类都有」的分歧。

## 概念

- [[code-space-search|代码空间搜索与解树（Code-Space Search）]] — 解即程序、边即改进尝试；原子改动、`Σ(T)` 摘要、五种组织方式（含单链 keep/discard）的取舍与失败模式，含 AlphaEvolve 程序数据库、AIDE 解树与 autoresearch 单链的对照。
- [[tool-centric-research|工具中心的研究（Tool-Centric Research）]] — 把产出单位从「答案」换成「工具」：符号化/参数化载体、通用性阶梯与摊销式经济性。
- [[recursive-self-improvement|递归自我改进（RSI）]] — scaffold–data–model 闭环、四层循环及其失败模式、以瓶颈分类代替能力上限叙事。
- [[research-memory-tiers|研究记忆分层与知识成熟度分级]] — 项目状态／agent 记忆／跨任务 Books 三层，观察/已验证/约定三级，以及 skill 沉淀。
- [[experience-bank|经验库（Experience Bank）]] — 结构化记录 solution/日志/评分，并合成为多样化「灵感」的机制；含与 2.0 Books 的对照。
- [[evaluator-solution-co-evolution|评估器与解的共进化]] — 无现成评估器时的双层循环，以及 2.0 把外层折成独立复核与人工审批后的新瓶颈。
- [[leaderboard-vs-cv|排行榜与本地验证（Leaderboard vs CV）]] — 本地 CV / Public LB / Private LB 三套分数的分工；「信 CV」的前提是切分复刻真实划分结构；人类竞赛经验与 AIDE / autoresearch 的同一张力对照，并新增内置 CV（LOO 与 K 折）与 AIC/BIC 作为第三种代理指标来源。
- [[regularization-and-sparsity|正则化与稀疏性（Regularization and Sparsity）]] — 五种惩罚项的目标对照、软阈值与 ARD 两条稀疏来源、$ℓ_2$ 的贝叶斯读法、**协方差收缩（shrinkage/reg_param）作为第二个正则化面**、AIC/BIC 的假设强度，以及「稀疏 ≠ 不重要」的三条边界。
- [[gaussian-discriminant-analysis|高斯判别分析（Gaussian Discriminant Analysis）]] — 生成式分类器：`GaussianNB` ⊂ LDA ⊂ QDA 的嵌套、Mahalanobis 下的线性决策面、白化 + 对类均值做 PCA 的降维、参数个数随 $d$ 的二次增长与协方差收缩的动机。
- [[naive-bayes|朴素贝叶斯（Naive Bayes）]] — 条件独立买到的两个好处（$O(d)$ 参数、可增量）与代价（相关性不可见）；五个变体各自的分布假设与 4 个易读漏的点；「分类器好、估计量坏」派生出的五条用法受影响度；平滑 α 与 `var_smoothing` 作为第三个正则化面（与 shrinkage/reg_param 的区别）。

## 实践

- [[research-agent-harness|科研智能体 Harness 设计]] — 五种形态：树搜索脚手架（AIDE）、单任务搜索循环（Hyra 1.0）、多项目长周期组织（2.0，含队列、唤醒、看板、审批门与证据要求）、进化式程序数据库（AlphaEvolve）与单文件贪心 + 人类维护指令（autoresearch）。
- [[evaluator-design-and-reward-hacking|评估器设计与 reward hacking 防护]] — 1.0 的两个作弊案例、2.0 的证据提交与独立复核、AIDE 暴露的「在固定验证划分上搜索」、AlphaEvolve 的生产指标证据、autoresearch 的「固定预算下的噪声选择」（含种子入选的可观测信号），以及 Kaggle 竞赛的「验证切分与真实划分结构不一致」。
- [[competition-ml-workflow|竞赛型 MLE 工作流（Kaggle 类任务）]] — 七段流程的分阶段判据与时效性对照（哪些仍成立、哪些已过时）、三个张力与一个坑（特征多反变差、合并 train/test 做变换）、以及各阶段对智能体的可自动化程度。
- [[linear-models-as-baselines|把线性模型当基线：预处理约束、正则强度选择与常见坑（含 LDA/QDA）]] — 尺度与共线性的四条硬约束、alpha 选择的三条路（内置 K 折 / LOO / 信息准则）及其失效条件、五个默认值陷阱、稳健回归三张示例图的误差读数与官方「拿不准用 RANSAC」的反例，以及 **LDA/QDA 生成式基线**何时先试、`shrinkage`/`OAS` 三条必跑配置与图 2 的准确率读数。

## 工具

- [[aide-scaffold|AIDE 脚手架（aideml）]] — 开源脚手架：解树、三算子、摘要算子的实现形态，作为基线的用法与版本风险。
- [[autoresearch-repo|autoresearch 仓库]] — 单卡 nanochat 简化实现：三文件契约、快速开始、小机器降规模参数表 7 条与使用边界；未固定 commit，仅对 2026-03 剪藏负责。
- [[hyra|Hyra（Hunyuan Research Agent）]] — 腾讯混元科研智能体的双版本页：1.0 单循环架构、2.0 多项目架构、两代结果表与使用边界。
- [[scikit-learn|scikit-learn（工具页）]] — 18 篇同批剪藏的模块覆盖表；`linear_model` / `discriminant_analysis` / `naive_bayes` 三个模块的估计器矩阵、22 条会沉默改变结果的默认值（class_weight×C、`RidgeCV` 的 LOO、`svd` 不能配 shrinkage、`priors` 默认类比例、`var_smoothing` 默认方差下限、`binarize=0.0` 静默二值化、`norm=False` 不归一化等）与版本边界（linear_model ≥1.8、discriminant_analysis 与 naive_bayes 1.9.x）。

## 比较

- [[hyra-1-0-vs-2-0|Hyra 1.0 与 2.0 对照：解题范式 vs 工具范式]] — 逐维度对照两版的目标、组织结构、记忆、验证方式、结果形态与证据强度，附阅读顺序建议。

## 研究

- [[automated-research-open-questions|自动化研究待验证问题]] — AIDE、AlphaEvolve、autoresearch 与 Hyra 1.0/2.0 五份来源中结果可信度、机制假设、工程风险与待补充资料清单。
