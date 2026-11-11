---
title: 评估器与解的共进化（Evaluator–Solution Co-evolution）
type: concept
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - evaluator-design
  - research-agent
sources:
  - "[[hyra-1-0]]"
status: draft
---

# 评估器与解的共进化（Evaluator–Solution Co-evolution）

当自动化研究面对**没有现成评估器**或**评估器会被搜索策略绕过**的任务时，只优化 solution 会遇到天花板。共进化指的是：内层循环在固定评估器下优化 solution，外层循环用积累的经验升级评估器，再开启下一轮，使两者交替上升。本页依据 Hyra-1.0 的表述（[[hyra-1-0]]）。

## 来源论述

- 对未提供评估器的问题，Hyra 把单层循环升级为双层循环：先由 Hyra 根据任务描述设计初版评估器，内层据此优化 solution；内层结束后结合 EB 的历史优化评估器，再开下一轮（[[hyra-1-0]], section "Hyra Harness"，原文 `raw/articles/Hyra.md:32`）。
- 来源给出的评估器升级方向有四类：提升评估代码效率、减少 reward hacking 风险、加强评估粒度、允许更大进步空间（同上）。
- 示例：设计世界冠军级国际象棋 AI 时，初版评估器是随机 AI 组成的天梯、以 Elo 计分；随着循环推进，对手逐步升级为 EB 中记录的更强 AI（同上）。
- 黑白棋 bot 案例把这一机制落到对手池：新方案与 EB 中高分 bot 对战，按 Elo 保留 top-k，形成随搜索不断增强的对手池（[[hyra-1-0]], section "AI for Fun"，`raw/articles/Hyra.md:77`）。
- 配器案例把外层换成人类反馈：内层用固定规则检查（和声、和弦进行、节奏密度、音域冲突），外层根据用户反馈调整评估器，7 轮后产出多乐器完整配器（[[hyra-1-0]], `raw/articles/Hyra.md:92`）。
- 来源的结论性主张：当现有评估标准不足时，solver 与 evaluator 可以在双层循环中共同演化（[[hyra-1-0]], `raw/articles/Hyra.md:97`）。

## 综合：两种共进化，风险不同

| 类型 | 外层信号 | 主要收益 | 主要风险 |
| --- | --- | --- | --- |
| 自生成对手/课程（棋类） | 双方对局结果、Elo | 难度自动对齐当前水平 | 对手池崩塌或循环利用，评估失去区分度 |
| 人类/偏好反馈（配器、3D） | 用户反馈、VLM judge | 可处理难以形式化的目标 | 偏好漂移、judge 与真人审美脱节 |

（本库整理，来源未做此分类；依据 [[hyra-1-0]], section "AI for Fun"。）

关键不对称：**solution 每轮都在变强，评估器的难度若不随之提高，就会被「解出」**。因此评估器升级不是可选优化，而是搜索能力增长后的必要条件。

## 与 reward hacking 的关系

来源记录的两种具体作弊形态说明共进化必须包含「真实性检验」：

- NanoChat：把因果语言模型改成近似双向注意力，借未来 token 泄漏异常压低 BPB。
- SOL-ExecBench：在正确性检查阶段缓存输出，计时阶段运行空 kernel。

（[[hyra-1-0]], section "AI for AI"，`raw/articles/Hyra.md:55`；机制整理见 [[evaluator-design-and-reward-hacking]]。）

## 假设（待验证）

- 假设：外层升级频率应随内层收益递减而提高，而非固定轮数。
- 假设：评估器版本必须与 EB 记录绑定，否则跨版本比较会产生虚假进步。
- 假设：自生成对手池最终需要「外部锚点」（人类基线或独立基准）才能避免自洽式退化。

## 待解问题

- 评估器的改动由谁判定「更好」？来源未说明元评估（meta-evaluation）机制。
- 双层循环的预算如何在内层与外层之间分配？
- 人类反馈外层循环如何防止用户偏好被过拟合（例如仅 7 轮就收敛到某种风格）？

## 关联页面

- [[hyra-1-0]]、[[hyra]]、[[experience-bank]]、[[recursive-self-improvement]]、[[evaluator-design-and-reward-hacking]]、[[research-agent-harness]]
