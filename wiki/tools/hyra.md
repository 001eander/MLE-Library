---
title: Hyra（Hunyuan Research Agent）
type: tool
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - agent-harness
sources:
  - "[[hyra-1-0]]"
  - "[[hyra-2-0]]"
status: draft
verified: 2026-11-11
applies_to:
  - Hyra-1.0
  - Hyra 2.0
---

# Hyra（Hunyuan Research Agent）

腾讯混元的通用科研智能体系列。本页覆盖两个公开版本：**Hyra-1.0**（极简搜索循环，覆盖 AI for AI / Science / Fun）与 **Hyra 2.0**（面向工具研发的多项目长周期 harness）。版本差异的集中对照见 [[hyra-1-0-vs-2-0]]。

| 版本 | 定位 | 来源 |
| --- | --- | --- |
| Hyra-1.0 | 给定任务持续提出更优 solution，框架轻、动作空间大 | [[hyra-1-0]] |
| Hyra 2.0 | 让 agent 研发/升级科学模型、工作流与软件工具，成果被下游复用 | [[hyra-2-0]] |

## Hyra-1.0 架构

| 组件 | 职责 |
| --- | --- |
| Context Agent | 维护经验库（Experience Bank, EB）；把 solution 与评估结果写入 EB；从 EB 整合「灵感」（inspirations）投入任务队列 |
| Proposal Agent（多个） | 领取一份灵感，反思后写出新 solution（含 `solve.sh` 的 `solution/` 目录），在全新沙盒中执行并评分 |
| 任务队列 + 信号量 | 异步生产者-消费者流水线，控制沙盒、模型推理、队列等资源保持满载 |
| 评估器（可选双层） | 任务未提供评估器时，先由 Hyra 设计评估器；内层优化 solution，外层再优化评估器 |

（来源论述，[[hyra-1-0]], section "Hyra Harness"；原文 `raw/articles/Hyra.md:25-32`。）

设计原则显式对齐 The Bitter Lesson：框架尽量轻量简单、动作空间尽量广阔（[[hyra-1-0]], section "Hyra Harness"）。

## Hyra 2.0 架构

**动机**：训练一个模型要经历方案设计、实现、训练、评估与反复修改，可能持续数天到数周；多项目并行还要处理实验等待、资源分配与方向调整。来源的判断是这些工作「很难靠一个不断延长的 agent session 来管理」（[[hyra-2-0]], section "为升级工具而生的 Harness"，`raw/articles/Hyra 2.0.md:63`）。

| 组件 | 职责 | 引用 |
| --- | --- | --- |
| Overall | 与人类研究者沟通，管理项目优先级，协调资源与关键决策 | `:67` |
| Mission | 长期负责一个研究问题：定路线、设计实验、组织 Worker、对结论负责；判断何时换方向、哪些结果需复核 | `:67` |
| Worker | 独立读代码、做实验、分析失败、提新想法，**可质疑 Mission**；不同 Worker 可用不同模型 | `:67` |
| 持久化队列 + 调度器 | 按优先级、依赖关系与资源上限启动 Worker；等待训练或审批时其他任务继续；agent 可自设下一次唤醒时间 | `:69` |
| Research Board | 记录想法、实验发现、证据与关键决策；普通进展异步阅读，必须处理的问题走消息通知并要求明确回应 | `:71` |
| Review Worker | 需要独立验证时启动，先从原始问题与证据入手，减少执行者结论的锚定 | `:77` |
| 分层记忆 | 项目状态（队列 + Board）／每个 agent 的工作记忆（当前认识、关键假设、已排除路线、下一步） | `:81` |
| Books | 跨任务共享知识库：做法、理由、适用条件 + 原始证据；分「观察 / 已验证 / 约定」三级 | `:81-83` |
| skill | 复盘后把可迁移方法封装成可执行资产，本身也是被持续修改的工具 | `:85` |
| 人工审批门 | 较大算力开销与共享知识同步需审批，获准后系统继续推进 | `:65` |

（全部为来源论述，[[hyra-2-0]]；行号指 `raw/articles/Hyra 2.0.md`。）

接入方式是接进真实研发环境：企业微信对齐需求、算力平台提交与管理实验、代码仓库实现与验证改动、保存实验过程与产物（`:65`）。

## 报告的结果

### Hyra-1.0

| 场景 | 任务 | 报告结果 |
| --- | --- | --- |
| AI for AI | NanoChat Autoresearch | Validation BPB 0.9015（对照 Recursive 0.9109） |
| AI for AI | NanoGPT Speedrun | Time to 3.28 loss 76.4s（对照 77.5s） |
| AI for AI | SOL-ExecBench | Mean SOL 0.771（对照 0.754，†本地运行、Evaluation Stack v1.0） |
| AI for Science | 55 个数学开放问题 | 29 个刷新历史最好结果 |
| AI for Science | 太阳黑子递推公式 | 1932–2026 样本外 $R^2 = 0.77$ |
| AI for Science | 10 位数加法 Transformer | 15 个可训练参数（Adderboard 纪录 36 个，减少 58.3%） |
| AI for Science | 量子比特路由（IBM Q20） | 相比 SABRE 路由效率 +44.4% |
| AI for Fun | 黑白棋 bot（Botzone） | 730 个参赛程序中排名第 3 |
| AI for Fun | 2D 参考图 → 3D 模型 | rubrics VLM judge；文中称优于 Claude Code goal 模式 |
| AI for Fun | 主旋律配器 | 7 轮进化，规则内层 + 用户反馈外层 |

（来源论述，[[hyra-1-0]], `raw/articles/Hyra.md:38-97`。）

### Hyra 2.0

| 领域 | 成果 | 报告结果 | 条件 |
| --- | --- | --- | --- |
| 蛋白质设计 | HyInv（逆折叠） | 196 万参数 GNN；图 3：2,532 designs/s、scTM≈0.838；比 ProteinMPNN 快 57× | 对照 ProteinMPNN ≈40/s、≈0.826（读图近似） |
| 蛋白质语言模型 | HyPLM-Flash | 650M 参数 / 726B token；图 3：131k residues/s（最快）、平均得分≈0.73 | 得分低于 E1-600M（≈0.78）与 ESM-2 3B（≈0.76） |
| 分子模拟 | HyFF（MLIP） | 5 档 2.3M–1.2B、83 元素；图 4：MAE 能量 ≈5.9→≈0.35 meV/atom、力 ≈48→≈5.5 meV/Å | 对照 16 个公开模型；每档约 1–100 ms/step |
| 数学 | HyProver | 圆周作用猜想、五维 Petty 等距点猜想 | 称与数学家协同核验；无证明工件链接 |
| 量子纠错 | HyQEC | $[488,123,22]$，$kd^2/n=122.0$，+48%；码距 95→86；178/187 认证；8–55× 吞吐 | 非 CSS、校验权重 ≤8 |
| 机器人 | 旋梯手 | DR5 仿真 3.277 rad/s，+28.6%；完成数 +21% | 同预算微调后；仅仿真；指长 20.4–24.5 cm vs Dextra 20.4–23.2 cm |
| 处理器 | 香山 | 每 GHz 几何均值 +5.79%；10 个新 bug；1 个一致性反例 | 频率不变、面积 +0.3%；「≈2–3 个月」为估算 |

（来源论述，[[hyra-2-0]], section "初步成果"，`raw/articles/Hyra 2.0.md:87-169`；表中标「图 X」的数值于 2026-11-11 目视核查图片后补入，均以「≈」标注读图近似。逐项条件见 [[hyra-2-0]] 的「成果数字一览」与「图表核查」。）

公开渠道：成果称持续发布在 GitHub 仓库 `Tencent-Hunyuan/hyra-results`；HyInv 与 HyPLM 的技术报告称将公开；HyQEC 称将发布（`raw/articles/Hyra 2.0.md:89`、`:111`、`:141`）。

## 使用时需要注意的边界

- **版本敏感**：本页仅对应两篇官方博客的表述（verified 2026-11-11）。2.0 的框架细节、接口与成果集都可能变化；1.0 的 harness 是否仍在使用、两个版本是替代还是并存，来源均未说明（见 [[hyra-1-0-vs-2-0]] 的待解问题）。
- **基座模型不明**：据 [[hyra-2-0]]（`raw/articles/Hyra 2.0.md:18`），1.0 由开源模型 Hy3 驱动；2.0 未披露所用模型，因此两版结果无法归因到 harness 设计。
- **开放程度未说明**：只有结果仓库被点名，harness 本身是否可获取、许可与运行成本均未说明。
- **自评结果**：所有对比数字来自作者自述，缺第三方复现与算力披露；数学证明称经数学家核验但剪藏内无工件链接。
- **图已目视核查但未取点复算**：2026-11-11 已下载 7 图与视频并逐张查看（见 [[hyra-2-0]] 的「图表核查」）；图中的 scTM、吞吐、MAE 区间等均为读图近似，硬件与环境信息不在图内。
- **人工审批意味着不是全自动**：2.0 的流程里，大算力开销与共享知识同步需要人类批准（`raw/articles/Hyra 2.0.md:65`），「自动化」的边界应据此理解。
- **非标量任务的评估依赖外部信号**：1.0 的 AI for Fun、2.0 的灵巧手（仿真评分）都依赖对手池、VLM judge、仿真器或用户反馈，见 [[evaluator-design-and-reward-hacking]]。

## 与其设计最相关的可复用经验

1. **分层组织而非单会话**：长周期任务需要 Overall/Mission/Worker + 持久化队列 + 唤醒时间，而不是把上下文拉长（[[research-agent-harness]]）。
2. **证据随报告提交**：完成报告必须附代码版本、日志与原始轨迹，复核者先看原始证据再造结论（[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]）。
3. **知识分层与分级**：项目状态、agent 记忆、跨任务知识分开存，跨任务知识标成熟度（[[research-memory-tiers]]）。
4. **把经验封装成可执行工具**：EB 的「灵感」是上下文复用，skill 是可执行复用，后者更易测试与版本化（[[experience-bank]]、[[tool-centric-research]]）。
5. **以瓶颈分类驱动下一轮**：模型能力 / 搜索效率 / 验证反馈，三类限制对应不同的工具研发方向（[[tool-centric-research]]）。

## 关联页面

- 来源：[[hyra-1-0]]、[[hyra-2-0]]
- 对照：[[hyra-1-0-vs-2-0]]
- 概念：[[tool-centric-research]]、[[research-memory-tiers]]、[[recursive-self-improvement]]、[[experience-bank]]、[[evaluator-solution-co-evolution]]
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 待验证问题：[[automated-research-open-questions]]
