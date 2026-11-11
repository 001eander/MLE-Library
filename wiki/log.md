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
