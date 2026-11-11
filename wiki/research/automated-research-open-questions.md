---
title: 自动化研究待验证问题（源自 Hyra-1.0）
type: research
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - open-questions
sources:
  - "[[hyra-1-0]]"
status: draft
---

# 自动化研究待验证问题（源自 Hyra-1.0）

本页集中记录在入库 [[hyra-1-0]] 时浮现、但当前证据不足以回答的问题。内容属于**假设与待验证项**，不得当作来源结论引用。

## 一、结果可信度

1. **改进幅度有多大意义？** NanoChat BPB 0.9109→0.9015、NanoGPT 77.5s→76.4s 均是对已高度优化基线的微小推进；作者自己也承认剩余空间极小（[[hyra-1-0]], section "AI for AI"）。需要多次运行与方差才能判断是否超出噪声。
2. **SOL-ExecBench 的 0.771 可比吗？** 该数值带 † 标记，评测设定为 Evaluation Stack v1.0 且为本地运行（`raw/articles/Hyra.md:48`），与 Recursive 报道口径的对齐程度未经第三方核验。
3. **算力口径缺失**：三组对照的预算是否相同、单任务耗时多少，来源均未给出；因此无法判断是「方法更好」还是「花了更多算力」。
4. **数学结果如何界定「刷新」**：55 个开放问题中 29 个被称刷新历史最好结果，但未说明「最好结果」的判定方与验证方式（`raw/articles/Hyra.md:61`）。

## 二、机制性假设

1. 收益主要来自搜索广度与经验复用，而非单次 proposal 质量——缺少消融。
2. 灵感多样性比灵感平均质量更重要（[[experience-bank]]）。
3. 双层循环的外层升级应由内层收益递减触发，而非固定轮数（[[evaluator-solution-co-evolution]]）。
4. 自生成对手池最终需要外部锚点以避免自洽退化。
5. scaffold–data–model 循环存在收益递减，瓶颈在评估器质量与验证成本（[[recursive-self-improvement]]）。

## 三、工程风险

1. **reward hacking 的自动化检测**：来源仅记录了两个个案（未来 token 泄漏、缓存 + 空 kernel），未说明是否有系统性检测器（[[evaluator-design-and-reward-hacking]]）。
2. **EB 污染**：作弊或错误记录一旦入 [[experience-bank]]，是否会作为「灵感」被反复复用？缺少淘汰与回滚机制描述。
3. **评估器与记录版本错配**：跨版本分数直接比较会制造虚假进步。
4. **主观任务的可复现性**：3D 建模用 rubrics VLM judge、配器依赖用户反馈，外部无法复算。

## 四、需要补充的资料

- Recursive「First Steps Toward Automated AI Research」的原文（Hyra 复用的公开设置来源）。
- SOL-ExecBench 与 modded-nanogpt 的评测协议原文。
- SILSO 太阳黑子数据与 $R^2=0.77$ 的实验设置。
- AdderBoard、SABRE 原始论文（用于核对 15 vs 36 参数、+44.4% 路由效率的口径）。
- `raw/articles/` 中已有的「Hyra 2.0」剪藏（尚未收录）。

## 关联页面

- [[hyra-1-0]]、[[hyra]]、[[recursive-self-improvement]]、[[experience-bank]]、[[evaluator-solution-co-evolution]]、[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
