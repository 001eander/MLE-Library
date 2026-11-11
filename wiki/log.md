# 知识日志

按时间顺序仅追加记录知识操作。条目使用以下格式：

```text
## [YYYY-MM-DD] operation | 主题
- 文件：受影响的路径
- 摘要：变更内容及原因
- 来源：来源路径或页面链接
- 复核：人工确认或 dry-run 状态
```

实现提交由 Git 记录，不在此重复。只有刻意将只读查询的答案保存到 wiki 时，才记录该查询。

## [2026-11-11] ingest | Hyra：简单有效的科学发现智能体（腾讯混元，Hyra-1.0）
- 文件：`wiki/sources/hyra-1-0.md`、`wiki/tools/hyra.md`、`wiki/concepts/recursive-self-improvement.md`、`wiki/concepts/experience-bank.md`、`wiki/concepts/evaluator-solution-co-evolution.md`、`wiki/practices/research-agent-harness.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/research/automated-research-open-questions.md`、`wiki/index.md`、`wiki/overview.md`
- 摘要：首次入库来源。建立来源页（元数据、六组关键主张、局限、行号位置索引），并据此新建 1 个工具页、3 个概念页、2 个实践页与 1 个待验证问题页；索引与概览同步更新。来源为厂商自评博客，因此所有对比数字标注为来源论述，且 `hyra` 工具页与两篇实践页的 status 保持 draft、verified 2026-11-11。
- 来源：`raw/articles/Hyra.md`（sha256 `6e9db748d24d1433a87022689bf3893f78fd47c13789c2999dbcceb015a46c2a`），原文 <https://hy.tencent.com/research/hyra>
- 复核：非交互式 ingest，变更保持未提交状态，等待所有者复核后再提交 `ingest: Hyra: 简单有效的科学发现智能体`。

## [2026-11-11] ingest | Hyra 2.0：从解题到造工具（腾讯混元，Hyra 2.0）
- 文件：`wiki/sources/hyra-2-0.md`、`wiki/comparisons/hyra-1-0-vs-2-0.md`、`wiki/concepts/tool-centric-research.md`、`wiki/concepts/research-memory-tiers.md`、`wiki/tools/hyra.md`、`wiki/practices/research-agent-harness.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/concepts/recursive-self-improvement.md`、`wiki/concepts/experience-bank.md`、`wiki/concepts/evaluator-solution-co-evolution.md`、`wiki/sources/hyra-1-0.md`、`wiki/research/automated-research-open-questions.md`、`wiki/index.md`、`wiki/overview.md`
- 摘要：第二个来源入库。新建来源页（元数据、八组关键主张、成果数字一览、图表核查、局限、行号索引）与 1 个版本对照页、2 个概念页（工具中心的研究、研究记忆分层与知识成熟度分级）；把 `hyra` 工具页改写为 1.0/2.0 双版本页，扩写两篇实践页（长周期多项目 harness、2.0 验证制度），并在 RSI、经验库、评估器共进化三页补充 2.0 视角与未解点。所有成果数字保留条件口径（非 CSS/校验权重 ≤8、同预算微调、仿真、估算等），并明确标注基座模型未披露与人工审批使流程非全自动。7 张图与 1 段视频已逐张目视核查，据此补入图内数值（2,532 designs/s、131k residues/s、MAE 区间、五指长度、净旋转）并新增三项疑问（对数轴放大、Pareto 前沿按档位理解、灵巧手尺度差异）。
- 来源：`raw/articles/Hyra 2.0.md`（215 行，sha256 `248d46eaa3de80f6ed5ddea5113c329924e6a3735e0e5c45eb33017b12570ad2`），原文 <https://hy.tencent.com/research/hyra2?langVersion=zh>；剪藏 `published` 为空，故来源页不写发布日期。
- 复核：交互式 ingest，已与所有者讨论并确认页面范围；图片与视频于 2026-11-11 下载到 `/tmp/hyra2-figs/` 临时核查，未入库。变更保持未提交状态，等待复核 diff 后再提交 `ingest: Hyra 2.0 自动化科研的下一步`。

## [2026-11-11] ingest | AIDE: AI-Driven Exploration in the Space of Code（Weco AI，arXiv 2502.13138v1）
- 文件：`wiki/sources/aide.md`、`wiki/tools/aide-scaffold.md`、`wiki/concepts/code-space-search.md`、`wiki/practices/research-agent-harness.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/research/automated-research-open-questions.md`、`wiki/index.md`、`wiki/overview.md`
- 摘要：首个论文类来源。所有者指定改以论文 PDF 入库，因此先下载 arXiv:2502.13138v1 到 `raw/papers/`、用 `mle` 的 `pdftotext -layout` 提取到 `raw/extracted/2502.13138v1.md`（含 `## Page N` 分页），并按所有者指示删除原先的二手剪藏 `raw/articles/AIDE.md`（该文件为未提交的读书笔记，从未入库，故无历史丢失）。新建来源页（元数据、八组关键主张、结果数字一览、图表核查、局限、分歧、页码与行号索引）、工具页（解树/三算子/摘要算子、基线用法、使用边界、版本风险）与概念页「代码空间搜索与解树」；把 harness 实践页扩为三形态（AIDE 树搜索 / Hyra 1.0 单循环 / 2.0 多项目）并加对照表，在评估器实践页新增「在固定验证划分上搜索」这类非恶意失效模式，在待验证问题页补入 AIDE 的数字疑点、机制假设、工程风险与待补资料。第 4、7、8、9、15 页已渲染目视核查（Fig 1/2/3/4/6/7 与 Table 3、附录 C），据此补入图 4 的读数（AIDE 约 5h 后走平 ≈0.36，人类 8h ≈0.65）与图 6/7 的近似值。
- 来源：`raw/papers/2502.13138v1.pdf`（17 页，sha256 `5fc52573eab9c8cb568b0c5aab6ee75752d198504babf8278ad23fe3d542cb41`），原文 <https://arxiv.org/abs/2502.13138v1>；提取文本 `raw/extracted/2502.13138v1.md`（1033 行）。
- 分歧：① 论文 p. 9 的「4 倍于 OpenHands」是跨模型比较（16.9% vs GPT-4o+OpenHands 4.4%），p. 2 的同模型口径约 2 倍；② 正文称 Weco-Kaggle 有 63 个竞赛，附录 C 逐行统计为 80 行；③ RE-Bench 的 6h / 8h / 64h 三个时间口径未说明。三项均保留在来源页「分歧」与待验证问题页，未做取舍。
- 命名：按所有者要求统一用 AIDE 命名——`wiki/sources/aide.md`（论文/来源页，引用写作 `[[aide]], p. N`）与 `wiki/tools/aide-scaffold.md`（脚手架页）；两页不能同名，否则 `[[aide]]` 在 Obsidian 中歧义。
- 复核：交互式 ingest，未提交，等待所有者复核 diff 后再提交 `ingest: AIDE: AI-Driven Exploration in the Space of Code`。

## [2026-11-11] ingest | AlphaEvolve：由 Gemini 驱动的算法设计编码智能体（Google DeepMind 官方博客）
- 文件：`wiki/sources/alphaevolve.md`、`wiki/concepts/code-space-search.md`、`wiki/practices/research-agent-harness.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/concepts/recursive-self-improvement.md`、`wiki/research/automated-research-open-questions.md`、`wiki/sources/aide.md`、`wiki/index.md`、`wiki/overview.md`
- 摘要：第四个来源入库。新建来源页（元数据、六组关键主张、成果数字一览、图表核查、局限、待解问题、行号索引）；把 `code-space-search` 的组织方式表扩为四类并新增「程序数据库 vs 解树」对照，把 harness 实践页扩为四形态（新增进化式程序数据库一节与对照表列），在评估器实践页新增「生产环境验证」一类（0.7%/23%/1%/32.5% 与 Verilog 正确性硬门），在 RSI 页补入「改进自身训练栈的最小案例」并明确其边界（不改写 scaffold），在待验证问题页新增 7 项数字疑点、4 项机制假设与 4 项工程风险。所有数字标注为厂商自述；0.7% 分母、23%/1% 基线、75%+20% 加总、48 次乘法的最优性均列入待解。
- 来源：`raw/articles/AlphaEvolve.md`（71 行，sha256 `ebc5728e47bd8c2df839379674eb68026adb376e00e0fab8283f2c6b581069b3`），原文 <https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>，发布 2025-05-14。
- 图表核查：剪藏只有 1 张外链流程图（L29），以三种尺寸参数下载均返回 HTTP 400，本地无副本，**未目视核查**；图内信息仅据 alt 文本转述，不引用任何图内读数。
- 未做：白皮书 PDF（正文 L19 链接）未下载、未入库，作为下一个单独 ingest 的候选；未创建 `wiki/tools/alphaevolve.md`（无公开实现、无版本、仅筹划早期访问）。
- 复核：交互式 ingest，已与所有者确认页面范围与四项决策；变更保持未提交状态，等待复核 diff 后再提交 `ingest: AlphaEvolve: A Gemini-powered coding agent`。

## [2026-11-11] ingest | autoresearch：让智能体夜间自主跑单 GPU 训练实验（karpathy 仓库 README）
- 文件：`wiki/sources/autoresearch.md`、`wiki/tools/autoresearch-repo.md`、`wiki/practices/research-agent-harness.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/concepts/code-space-search.md`、`wiki/concepts/recursive-self-improvement.md`、`wiki/research/automated-research-open-questions.md`、`wiki/index.md`、`wiki/overview.md`、`raw/assets/autoresearch/progress.png`
- 摘要：第五个来源入库，也是首个「开源实现 + 无论文」的形态。新建来源页（元数据、七组关键主张、图内读数表、局限、待解问题、行号索引）与工具页（三文件契约、快速开始、小机器降规模参数表 7 条、使用边界、版本风险）；把 harness 实践页扩为五形态（新增「单文件贪心 + 人类迭代的指令文件」一节与对照表列、可迁移要点、适用边界），在评估器实践页新增「第五类：固定预算下的噪声选择」（与 AIDE 的第三类逐项对照），在代码空间搜索页新增第五种组织方式与 AIDE/autoresearch 单文件搜索对照表，在 RSI 页补「食谱级自改进」小节，在待验证问题页新增 4 项数字疑点、4 项机制假设、3 项工程风险与 5 项待补资料。
- 来源：`raw/articles/AutoResearch.md`（92 行，sha256 `71ebcc979776c1edc2d38effd18062f33d32d43edc6dd2fa2934d8ff1e12a2b9`），原文 <https://github.com/karpathy/autoresearch>；发布日 2026-03-07 仅来自剪藏 frontmatter，来源页标注为未独立核实。无 commit、无 tag、agent 侧模型无版本。
- 图表核查：README 唯一插图 `progress.png` 于 2026-11-11 从 master 抓取成功（HTTP 200，2382×1180），按所有者决定入库 `raw/assets/autoresearch/progress.png`（sha256 `a6be2ad8dece12be1907356b1254f3ba133035d264d73634dbed442887a69d8d`），**已目视核查**。据此记录图标题「83 Experiments, 15 Kept Improvements」、基线 ≈0.998 与末尾 ≈0.977 的目视读数、第 43–64 次之间约 20 次无改进的平台段，以及 15 个被保留改动的标签。
- 关键判断：图中 `random seed 42→137` 被计为一次「保留」。这提示 5 分钟预算下 `val_bpb` 的种子噪声可能与改进幅度同量级，因此在评估器实践页把它作为「固定预算下的噪声选择」的可观测证据，并在两页均标注为**本库推断（综合）**、来源未讨论。另记录一处内部张力：`prepare.py` 标注「不要修改」，但小机器降规模建议第 2–4 条恰恰落在该文件，属人类重写基线而非 agent 动作。
- 未做/未做决定：未下载父级 nanochat 仓库；未核对仓库当前默认超参与剪藏是否一致；`program.md` 的迭代实例无任何公开记录，故「研究组织代码」保留为假设。按所有者决定暂不新建「实验预算与可比性」概念页（内容先折入两篇实践页）与 AIDE/autoresearch 对照页。
- 复核：交互式 ingest，已与所有者讨论页面范围并确认四项决策（不新建概念页与对照页、图片入库 `raw/assets/`、工具页 draft + verified 2026-11-11）；变更保持未提交状态，等待复核 diff 后再提交 `ingest: karpathy/autoresearch：夜间自主的单 GPU 训练实验`。

## [2026-11-11] lint | 悬空来源引用、Hy3 归属、frontmatter 与可复核性
- 文件：`wiki/sources/aide.md`、`wiki/sources/hyra-1-0.md`、`wiki/sources/hyra-2-0.md`、`wiki/sources/autoresearch.md`、`wiki/comparisons/hyra-1-0-vs-2-0.md`、`wiki/tools/hyra.md`、`wiki/tools/aide-scaffold.md`、`wiki/concepts/code-space-search.md`、`wiki/practices/research-agent-harness.md`、`wiki/research/automated-research-open-questions.md`、`wiki/overview.md`、`README.md`、`raw/assets/hyra-2-0/`（7 图 + 1 视频）
- 摘要：① 删除对从未存在的 MLE-Bench / R&D-Agent / MLE-STAR 剪藏的引用与“交叉核对”说法，改为“本库尚未收录，无法交叉核对”；② 把“1.0 博客明确由 Hy3 驱动”改引 `raw/articles/Hyra 2.0.md:18`，并在 1.0 来源页补注；③ 删除 Hyra 2.0 发布日期的推测；④ 两个 Hyra 来源页补 `status: draft`；⑤ `code-space-search` 的 `sources` 补 `[[hyra-1-0]]`；⑥ 修正 harness 与 aide-scaffold 中残留的“三者/另外两种”，以及 overview 中 AIDE 与 autoresearch 的“唯一公开实现”矛盾；⑦ 将 Hyra 2.0 的 7 张图与 1 段视频入库 `raw/assets/hyra-2-0/` 并记录尺寸与 SHA-256，同时把图 4 的 DPA4-Pro 读数由 ≈1,100 ms/step 更正为 ≈1,000 ms/step；⑧ autoresearch 合并 `depth 9` 与 `aspect_ratio 57` 标签，并说明图标题的 15 次保留含 baseline 起点；⑨ 修正 overview 与 README 的 `skills/` 路径为 `.agents/skills/`；⑩ 待补资料补入 `Tencent-Hunyuan/hyra-results`。
- 来源：`raw/articles/Hyra 2.0.md:18`、`:73`、`:101`、`:119`、`:153`、`:155`、`:167`，抓取图与视频见 `raw/assets/hyra-2-0/`；AIDE 与 MLE-Bench 相关修复依据 `raw/extracted/2502.13138v1.md`，以及这些剪藏路径在 Git 历史中均无记录。
- 复核：dry-run lint 报告经所有者确认“全部”接受后应用；未新增或重命名 wiki 页面，`wiki/index.md` 无需改动。变更等待最终 diff 确认后再提交 `lint: 悬空来源引用、Hy3 归属、frontmatter 与可复核性`。

## [2026-11-11] ingest | Kaggle 入门指南：竞赛型 MLE 的通用流程与一次完整赛题复盘（知乎专栏剪藏）
- 文件：`wiki/sources/kaggle-getting-started.md`、`wiki/concepts/leaderboard-vs-cv.md`、`wiki/practices/competition-ml-workflow.md`、`wiki/practices/evaluator-design-and-reward-hacking.md`、`wiki/research/automated-research-open-questions.md`、`wiki/index.md`、`wiki/overview.md`、`raw/assets/kaggle-guide/`（4 图）
- 摘要：首个**人类选手**来源入库，也是本库首个非智能体、非论文/非厂商博客的实践类剪藏。新建来源页（元数据、九组关键主张、图表核查、代码与 API 时效性表、局限、待解问题、行号索引）；按所有者选择的范围方案新建概念页「排行榜与本地验证」（本地 CV / Public LB / Private LB 的分工、「信 CV」的隐含前提、与 AIDE 固定 holdout 和 autoresearch 单指标的同一张力对照）与实践页「竞赛型 MLE 工作流」（七段流程的分阶段判据与三档时效性判断、三个张力与一个坑、各阶段对智能体的可自动化程度）。在评估器实践页新增「第六类：验证切分与真实划分结构不一致」并补 4 条检查项、把 20 种子 +0.0004 作为第五类的**人类旁证**；在待验证问题页新增 5 项数字疑点（28–32）、4 项机制假设（22–25）、3 项工程风险（20–22）与 6 项待补资料。
- 来源：`raw/articles/Kaggle 入门指南.md`（375 行，sha256 `74549fbf97e8528e485cba3dc83b60e3c9d4549b5abf2de2e1d5df452f1aad7e`），原文为知乎专栏 Human Learning 长文（<https://www.zhihu.com/column/AbsTF>）。剪藏无 frontmatter：`published` 取 `2016`（正文自述 2016-04 初稿），平台最后编辑 2019-06-02 记入元数据表；作者未署名，按正文链接推断 Kaggle 用户名 dnc1994 并标注为**本库推断**。
- 图表核查：4 张外链图（`pic*.zhimg.com`）直接抓取返回 403/567，加 `Referer: https://zhuanlan.zhihu.com/` 与浏览器 UA 后全部 200，已入库 `raw/assets/kaggle-guide/`（606×892 PNG、1054×718 PNG、381×271 PNG、1297×326 JPEG，均记 SHA-256）并**逐张目视核查**。据此记录：图 1 为 Notebook 截图（`abs(Y)<100` 过滤后 y 由 ≈117 收窄到 [−0.15,0.15]）；图 2 确认 `DayOfWeek` 生成 7 个哑变量（并补注 dummy variable trap，**综合**）；图 3 的 7 档 relevance（1.0–3.0，步长 1/3）与「三位评分者取均值」交叉印证；图 4 存在**图文不一致**——图中标出测试侧 Variant A（5 折平均）/ Variant B（单模型预测）两种方案，正文与代码只实现 Variant A。
- 关键判断：① 来源的耐久部分是流程顺序、验证设计与 Stacking 实现，模型清单与全部代码片段绑定 2016–2017 年，已列 7 项 API 过时对照（`sklearn.grid_search`、`cross_validation`、`KFold(n_folds=)`、`reg:linear`、`silent` 等），并明确**不静默改写来源**；② 「特征越多越好」（L77）与「特征多反而变差且调参无效」（L290）在同一篇内并存，未调和，保留为张力与开放问题；③ 「Ensemble 同时降低 Bias 和 Variance」引用的 Springer 章节未读，按来源论述记录并列待补；④ 174→98 名换算成「前 10%→前 5%」但未给参赛队总数，列入待解。
- 未做：未下载 `dnc1994/Kaggle-Playground` 的 notebook；未读参考文献 `Beating Kaggle the Easy Way` 与 Ensemble 理论的 Springer 章节；未复核 Home Depot 榜单快照。
- 复核：交互式 ingest，已就范围（方案 2：来源页 + 1 概念 + 1 实践 + 两页扩写）、图片逐张核查、文件名与 `published`、以及不单独建「特征工程」概念页四项与所有者确认；变更保持未提交状态，等待复核 diff 后再提交 `ingest: Kaggle 入门指南`。

## [2026-11-11] ingest | scikit-learn 用户指南 1.1「Linear Models」（stable 文档剪藏）
- 文件：`wiki/sources/sklearn-linear-models.md`、`wiki/tools/scikit-learn.md`、`wiki/practices/linear-models-as-baselines.md`、`wiki/concepts/regularization-and-sparsity.md`、`wiki/concepts/leaderboard-vs-cv.md`、`wiki/practices/competition-ml-workflow.md`、`wiki/index.md`、`wiki/overview.md`、`raw/assets/sklearn-linear-models/`（18 图）
- 摘要：第七个来源入库，也是本库第一个「现行 API 参考」类来源。新建来源页（元数据、13 节结构表、九组关键主张、18 张图的逐张核查、转换损失与外部核查专节、局限、待解问题、行号索引）、工具页（18 篇同批剪藏的模块覆盖表 + `linear_model` 估计器矩阵 + 9 条会沉默改变结果的默认值 + 版本边界）、实践页（尺度/共线/截距三条硬约束、alpha 选择三条路及其失效条件、五个坑、稳健回归三图读数、成本表、对智能体的可自动化程度）与概念页（五种惩罚对照、软阈值与 ARD 两条稀疏来源、ℓ2 的贝叶斯读法、「稀疏 ≠ 不重要」三条边界）。在 `leaderboard-vs-cv` 新增「代理指标的第三种来源：内置 CV 与信息准则」小节（LOO/K 折/AIC-BIC 的假设强度差异，并把该来源加入 frontmatter），在 `competition-ml-workflow` 的模型选择段补一句现代对照与关联链接。
- 来源：`raw/articles/sklearn/1.1. Linear Models.md`（980 行，sha256 `03324a207556b360212a8de47b0bbbe74c746bc65be19144602c145c71f2fcd1`），原文 <https://scikit-learn.org/stable/modules/linear_model.html>；剪藏无版本号与发布日期，故来源页不写 `published`。
- `raw/` 修改（经所有者明确授权，**本库首例**）：所有者先手工补入缺失的 1.1.14；图书馆员随后按官方 1.8 页回填该节遗留的 3 处公式（Theil-Sen 复杂度 $\binom{n}{n_{\text{subsamples}}}$、Huber 目标、Huber 损失），并把 `[[14](url)]`/`[[15](url)]` 修正为 `[[14]](url)`/`[[15]](url)`。因此该文件**不再是抓取时的原始字节**（原 sha256 `5e142a3f…`，965 行）；L720 之后的引用行号已全部按新文件重写（1.1.15 = L784–L798、1.1.16 = L800–L875、脚注自 L877 起）。
- 版本判定：以官方 1.6.1 / 1.7.2 / 1.8.0 / 1.9.1 版本页做特征字符串比对，剪藏含 `1.1.3.1 gap safe screening`、`only solver that supports Elastic-Net`、`np.float64(...)` 输出，而 1.7.2 及更早均无 → 判定 **≥ 1.8**，1.8.0 与 1.9.1 不可区分（1.9 重写了 1.1.3.1 正文）。`applies_to` 写 `1.8.x–1.9.x（推断）`。
- 剪藏缺陷（已核查并记录在来源页，未静默修补 raw）：① 官方页 23 个折叠块（8 个 `Mathematical details`、7 个 `References`、`Solvers’ details`、`Practical considerations`、`Examples of use cases`、`Details of the algorithm`、`Theoretical considerations`）合计 19,266 字符 = 官方正文的 29.3%，剪藏**一个都没保留**（自证：L210 引用「Mathematical details」小节而下文不存在）；② 70 处 `std std-ref` 链接文字被吞，留下 L18/L118/L421/L491 等断句，按官方 1.8 原文逐条对照后列成表；③ 所有者手工补入的 1.1.14 仍有 3 处公式空缺（Theil-Sen 复杂度、Huber 目标、Huber 损失）与编号错位；④ 公式被转成 `\left(\right.` 伪 LaTeX、solver 矩阵为原始 `<table>`、Categorical 单位偏差疑为转换错误。
- 图表核查：18 张官方示例图于 2026-11-11 从 `https://scikit-learn.org/1.8/_images/` 抓取（全部 HTTP 200），入库 `raw/assets/sklearn-linear-models/`，**逐张目视核查**并记录尺寸与 SHA-256 前 12 位。据此补入三项来源未写的观察：robust_fit 三图的误差读数（y 小偏差时 RANSAC 0.328 明显差于 Theil-Sen 0.034 / Huber 0.011；y 大偏差时 RANSAC 0.002 反超）与官方「When in doubt, use RANSAC」冲突；LassoCV 0.17s vs LassoLarsCV 0.06s 的示例耗时；AIC/BIC 图横轴是 model sequence 而非 alpha（分辨率先于路径网格）。
- 决策：所有者选定方案 A（4 新页、不加比较页）、工具页按「一页伞页」组织、图片全部抓取、`leaderboard-vs-cv` 扩写。所有者同时要求忽略本机 Python 环境（故页面不写与本地安装版本的差异），并手工补全了 raw 中的 1.1.14。
- 未做：未入库其余 17 篇 sklearn 剪藏；未读来源转述的 Weisberg、Dang et al.、Huber 三处文献；未验证「跨估计器 alpha 不可比」在实现层是否被缩放补偿；未重建 `[[14]]`/`[[15]]` 与 `[^13]`/`[^20]` 的脚注编号体系（回填只修了链接语法）。
- 复核：交互式 ingest，变更保持未提交状态，等待所有者复核 diff 后再提交 `ingest: scikit-learn 1.1 Linear Models`。

## [2026-11-11] ingest | scikit-learn 用户指南 1.2「Linear and Quadratic Discriminant Analysis」（stable 文档剪藏）
- 文件：`wiki/sources/sklearn-lda-qda.md`、`wiki/concepts/gaussian-discriminant-analysis.md`、`wiki/tools/scikit-learn.md`、`wiki/concepts/regularization-and-sparsity.md`、`wiki/practices/linear-models-as-baselines.md`、`wiki/index.md`、`wiki/overview.md`、`raw/assets/sklearn-lda-qda/`（2 图）
- 摘要：第八个来源入库，也是 1.1 Linear Models 的姊妹篇——把工具页从判别式模型扩展到**生成式**模型。新建来源页（元数据、9 节结构表、七组关键主张、两张图的目视读数、转换损失与外部核查专节、局限、待解问题、行号索引）与概念页「高斯判别分析」（GaussianNB ⊂ LDA ⊂ QDA 的嵌套与每层自由度、Mahalanobis 下的线性决策面、白化 + 对类均值做 PCA 的降维、三个成员的自由参数个数推导与 QDA 的 $K\cdot d(d+1)/2$ 代价）。在 `scikit-learn` 工具页新增 `discriminant_analysis` 估计器矩阵与求解器能力表、把「会沉默改变结果的默认值」从 9 条扩到 16 条（`svd` 默认不能配 shrinkage、`shrinkage` 与 `covariance_estimator` 互斥、`n_components` 不影响预测、`priors` 默认类比例、`reg_param` 与 `shrinkage` 是两个旋钮、`store_covariance` 默认不存），并补版本边界。在 `regularization-and-sparsity` 新增「第二个正则化面：协方差收缩」（系数惩罚 vs 协方差收缩、三个端点与两个收缩终点、稀疏与收缩不可互换），在 `linear-models-as-baselines` 新增「生成式基线：LDA/QDA」（何时先试、三条必跑配置、图 2 的准确率读数、与惩罚模型相反的尺度直觉）。
- 来源：`raw/articles/sklearn/1.2. Linear and Quadratic Discriminant Analysis.md`（146 行，21,050 字节，sha256 `b2e50ad9a1b4406924a1eaaa7ed5e9980bdb147b8b78f85160d0df68e9dede55`），原文 <https://scikit-learn.org/stable/modules/lda_qda.html>；剪藏 `published`/`author` 为空，故来源页不写发布日期。
- 版本判定：以官方 1.6 / 1.7 / 1.8 / 1.9 页逐词比对，1.8 与 1.9 之间**唯一实质差异**是 QDA 的 SVD 推导由 $\frac{1}{n-1}X_k^TX_k$ 改为 $\frac{1}{n}X_k^TX_k$；剪藏写的是 $\frac{1}{n}$（L115）→ 判定为 **1.9.x**。旁证：QDA API 页说明 `covariance_` 用最大似然（有偏）估计。因此 1.2 剪藏比 1.1 剪藏（只能判到 ≥1.8）版本更精确。
- 剪藏缺陷（已核查并记录，未静默修补 raw）：① **正文完整但参考文献全丢**——官方 3 条文献（ESL §4.3 p.106–119；Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix*, JPM 30(4) 110–119, 2004；Duda/Hart/Stork §2.6.2）被替换成 4 段重复正文（L123–L135），末尾还重复了一遍 shrinkage 段与图（L137–L143）；② 4 处引用标记 `[1][2][3]` 被吞，留下 L75「written as:」、L83 与 L93「See for more details.」、L97「Ledoit and Wolf.」四处断句；③ 公式噪声化（`\left(\right.`、`C s t`）；④ `> [!note]` 为转换产物，官方页无内容折叠块（8 个 `<details>` 全是侧栏导航）。
- 外部核查（非剪藏内容，按所有者要求单独标注）：用户指南完全不覆盖 `priors`（默认＝训练集类比例）、QDA 的 `reg_param`（$S^2\leftarrow(1-\rho)S^2+\rho I$）、`store_covariance`（默认 `False`）与 `tol`（LDA 是秩阈值，QDA 只控制警告）。这些条目在工具页与来源页均标为「外部核查」，未混入来源论述。另核实实现层：`solver='svd'` 配 `shrinkage` 会抛 `NotImplementedError`（`sklearn/discriminant_analysis.py` 1.9.X），不是静默失效——初稿曾写作“静默失败”，已改。
- 图表核查：2 张官方示例图于 2026-11-11 从 `https://scikit-learn.org/1.9/_images/` 抓取（均 HTTP 200），入库 `raw/assets/sklearn-lda-qda/`（`sphx_glr_plot_lda_qda_001.png` 800×1200 `268cd8556935`、`sphx_glr_plot_lda_001.png` 640×480 `e66714782aea`，均为 SHA-256 前 12 位），**逐张目视核查**。据此记录三项来源未写的读数：① 图 1 第 1、2 行（等协方差）时 QDA 边界与 LDA 几乎重合，只有第 3 行协方差不同才出现双曲线型边界；② 图 2 无收缩 LDA 在 `n_features/n_samples`≈1.0 时约 0.77、≈1.9 触到坐标下限 0.65 后离图，而 Ledoit-Wolf 与 OAS 全程 ≈0.89–0.97；③ OAS 在比值 ≳0.6 后一直略高于 Ledoit-Wolf（比值 3.5 处 ≈0.92 vs ≈0.89），方向与 L101 一致但不可外推。
- 关键判断：① 来源开篇「have no hyperparameters to tune」（L12）与同页 `shrinkage`/`solver`/`covariance_estimator`（L97–L103）自相矛盾，来源页按「基础模型无需搜索正则强度，但协方差估计方式是结构选择」解读，并保留原话；② `shrinkage='auto'` 的「optimal」口径是**协方差估计误差**而非分类准确率（依据：L101 用 MSE 论证 OAS 更好），因此三页统一把它写成「有明确损失函数的默认值」；③ 新增一条**假设**（标为待验证）：`shrinkage`/`reg_param` 的收缩终点（对角阵/单位阵）不是仿射等变的，因此启用收缩后结果重新变成尺度依赖——文档未讨论。
- 决策：所有者确认全部提议——新增概念页、两张图入库 `raw/assets/` 并记录 SHA-256 与目视读数、图内读数写入实践页。
- 未做：未下载 `plot_lda_qda` / `plot_lda` / `plot_pca_vs_lda` 三个示例脚本；未读 ESL §4.3、Ledoit-Wolf 2004、Duda et al. §2.6.2 原文（文献条目在剪藏中已丢失）；未读同批未入库的 1.9 Naive Bayes 与 2.6 Covariance estimation 剪藏；未验证「跨模型 scale-invariance」假设。
- 复核：交互式 ingest，已与所有者讨论页面范围、图片入库与图内读数；变更保持未提交状态，等待复核 diff 后再提交 `ingest: scikit-learn 1.2 LDA/QDA`。

## [2026-11-11] ingest | scikit-learn 用户指南 1.9「Naive Bayes」（stable 文档剪藏，含 raw 回填）
- 文件：`wiki/sources/sklearn-naive-bayes.md`、`wiki/concepts/naive-bayes.md`、`wiki/tools/scikit-learn.md`、`wiki/concepts/gaussian-discriminant-analysis.md`、`wiki/concepts/regularization-and-sparsity.md`、`wiki/index.md`、`wiki/overview.md`、`raw/articles/sklearn/1.9. Naive Bayes.md`
- 摘要：第九个来源入库。新建来源页（元数据、10 行结构表、八组关键主张、版本判定、转换损失与外部核查专节、局限、七项待解问题、回填后行号索引）与概念页「朴素贝叶斯」（条件独立买到的两样东西与代价、五变体分布假设与 4 个易读漏点、参数代价对照、平滑作为第三个正则化面、「坏估计量」的五条用法受影响度、四个失效模式、四族基线定位）。工具页从两个模块扩到三个：新增 `naive_bayes` 估计器矩阵（5 个估计器 × 输入要求 × 构造参数 × 增量能力），把「会沉默改变结果的默认值」从 16 条扩到 22 条（`alpha=1.0` 即 Laplace、`fit_prior`、`var_smoothing=1e-9`、`binarize=0.0`、`norm=False`、`min_categories`），并把版本边界与使用边界同步。在 `gaussian-discriminant-analysis` 补 `GaussianNB` 的高斯似然与 `var_smoothing`（并说明它≠shrinkage），在 `regularization-and-sparsity` 新增「第三个正则化面：平滑」。
- 来源：`raw/articles/sklearn/1.9. Naive Bayes.md`，原文 <https://scikit-learn.org/stable/modules/naive_bayes.html>；剪藏 `published`/`author` 为空，故来源页不写发布日期。
- `raw/` 修改（经所有者明确授权，**本库第二例**）：抓取时共丢 5 个 `<details>` 折叠块 = 318 词 ≈ 官方正文的 21.4%，其中两个是**数学节**（1.9.3 `Weights calculation` 114 词、1.9.5 `Probability calculation` 97 词），三个是 References（主 11 词、CNB 28 词、Bernoulli 68 词）。根本原因与 1.1 节一致：转换器整块丢弃折叠块。已按官方 1.9.1 原文回填，标题层级按原结构（`### Weights calculation`、`### Probability calculation`、`## References`×3），公式改写为规范 LaTeX，**未改动任何原有句子**。文件由 115 行/11,112 字节（sha256 `5e0e5299a7d0447f039e522c7b8c4a3b3f7615bf11ab6da786ddb36e9c84bd66`）变为 171 行/13,172 字节（sha256 `85c4050b201718a7877050532715deebc2019d21f59c05a32e737244c8131fa8`）；**所有引用行号已按新文件重写**（如 `## 1.9.1` 由 L44 → L49）。
- 版本判定：**1.9.x，证据弱**。仅两个措辞级判据：剪藏写 `one-dimensional`（1.8/1.7/1.6 为 `one dimensional`）与 `through $x_n$:`（旧版为 `through \(x_n\), :`）；旁证是抓取日 `/stable/` 指向 1.9.1。1.8→1.9.1 在本节无实质改动，故默认值结论不依赖该判定。已在来源页显式标注「不应据此断言具体小版本」。
- 外部核查（非剪藏内容，均单独标注）：`GaussianNB(priors=None, var_smoothing=1e-9)`，源码 `epsilon_ = var_smoothing × max(var(X, axis=0))` 加到每个方差上；离散族 `alpha=1.0`、`fit_prior=True`、`class_prior=None`、`force_alpha=True`；`ComplementNB(norm=False)`；`BernoulliNB(binarize=0.0)`；`CategoricalNB(min_categories=None)`。七个参数用户指南全未提。核查对象：`sklearn/naive_bayes.py` @ tag `1.9.1` 与生成版 API 页。
- 分歧：1.9.6 正文（L165）称只有 `MultinomialNB`/`BernoulliNB`/`GaussianNB` 暴露 `partial_fit`，但 1.9.1 里五个估计器都有（`_BaseDiscreteNB.partial_fit` 被 MNB/CNB/BNB 继承，`CategoricalNB` 覆写），五个 API 页也都记录。保留来源原话 + 记录外部核查结果，不回写剪藏既有句子。
- 图表核查：本来源**无任何插图**，因此无 `raw/assets/` 变更、无目视读数。唯一的代码片段是 `GaussianNB` + iris 的 doctest（50/50 切分，误分类 4/75），已转录于来源页。
- 关键判断：① 「bad estimator」是全库第一条把「分类准确率」与「概率估计质量」拆开的论述，据此在概念页给出五条用法受影响度表，并明确标为**本库综合**（来源只给警告，不给量化与校准方法）；② `alpha` 按特征数缩放（分母 $N_y+\alpha n$）→ 与「alpha 不可跨估计器比较」同构，标为**综合**；③ `var_smoothing` 只是方差下限、不改变非对角项，因此**不能**读作 `GaussianNB` 的 shrinkage（标为**本库判断**）；④ `ComplementNB(norm=False)` 与文档 L103 的归一化公式不一致——按文档实现与按默认参数跑不是同一件事。
- 未做：未读 Zhang 2004（最优性）、Rennie et al. 2003（CNB）、Manning/McCallum/Metsis 三篇伯努利文献；未入库 8.2/8.3（向量化、预处理）；未读 `sklearn.calibration`；未验证「特征冗余导致过度自信」与「`var_smoothing` 对量纲差异不对称」两条推断。
- 复核：交互式 ingest，所有者选定方案 A（来源页 + 概念页 + 三页扩写，不新建实践页），并要求外部核查入库、raw 回填；变更保持未提交状态，等待复核 diff 后再提交 `ingest: scikit-learn 1.9 Naive Bayes`。
