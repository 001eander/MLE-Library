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
