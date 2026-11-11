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
