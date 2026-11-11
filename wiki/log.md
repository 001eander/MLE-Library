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
