---
title: "Hyra-1.0：腾讯混元的递归自我改进科研智能体（官方博客）"
type: source
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - research-agent
  - recursive-self-improvement
  - evaluator-design
source_path: raw/articles/Hyra.md
source_kind: article
status: draft
---

# Hyra-1.0：腾讯混元的递归自我改进科研智能体（官方博客）

## 元数据

| 项目 | 值 |
| --- | --- |
| 原始标题 | Hyra: 简单有效的科学发现智能体 |
| 发布方 | 腾讯混元（文中署名为「腾讯混元」） |
| 原始链接 | <https://hy.tencent.com/research/hyra> |
| 形态 | 官方技术博客 / 产品发布公告 |
| 剪藏文件 | `raw/articles/Hyra.md`（123 行） |
| 剪藏 SHA-256 | `2803506d798e5d967548bff7fd54bc04afc76d87a1cbe200b8ae18b76fea73e7` |
| 发布时间 | 未在剪藏中给出；剪藏文件 `created` 为 2026-11-10，不代表原文发布时间 |

剪藏为中文原文，含 15 条脚注（AlphaEvolve、Einstein Arena、autoresearch、The Bitter Lesson、Recursive、NanoGPT Speedrun、SOL-ExecBench、EinsteinArena、Erich's Packing Center、SILSO 太阳黑子数据、scaling law 论文、AdderBoard、SABRE、Botzone、Claude Code Goal Mode）。2026-11-12 已将文中的 2 张静态图、2 张 GIF、1 段视频与 3 段音频下载到 `raw/assets/hyra-1-0/`，并把签名外链改为本地相对路径，以避免签名参数中的访问密钥 ID 进入远程仓库。

## 概要

博客宣布 Hyra-1.0（Hunyuan Research Agent）的首个版本：一个面向「性能导向」研究与工程任务的通用智能体框架，核心主张是**框架简单、动作空间大**，通过单一循环持续提出并评估 solution，从而在多个领域产生可验证的改进（来源论述，[[hyra-1-0]], section "Hyra Harness"、section "The path forward"）。

作者把工作放在递归自我改进（Recursive Self-Improvement, RSI）与自动化研究的前沿脉络中，引用 AlphaEvolve、Together AI 的多智能体数学探索、Karpathy 的 autoresearch 作为同类进展（[[hyra-1-0]], 引言，`raw/articles/Hyra.md:15-19`）。文章结构为：Harness 设计 → 三类 demo（AI for AI / AI for Science / AI for Fun）→ 后续路线。

## 关键主张

### 1. Harness：轻量框架 + 宽广动作空间

- 设计原则明确对齐 The Bitter Lesson：「框架尽量轻量简单，智能体的动作空间尽量广阔」；给定任务描述后持续循环，历史经验包含过往 solution 的运行日志、评估器反馈与源代码，循环直到 Agent 主动结束或预算耗尽，返回历史最优方案（[[hyra-1-0]], section "Hyra Harness"，`raw/articles/Hyra.md:25`）。
- 两个角色：
  - **Context Agent**：维护经验库（Experience Bank, EB）数据结构，把提出的 solution 与评估结果写入 EB，并从 EB 整合出一份份「灵感」（inspirations）投入任务队列。灵感是一组上下文，可包含代码、文件、产物、日志（[[hyra-1-0]], section "Hyra Harness"，`raw/articles/Hyra.md:27`）。
  - **多个 Proposal Agent**：每次领取一份灵感，反思后写出新的 solution（含 `solve.sh` 入口的 `solution/` 文件夹），在全新沙盒中运行与评分（[[hyra-1-0]], section "Hyra Harness"，`raw/articles/Hyra.md:28`）。
- 系统组织方式是**异步生产者-消费者流水线**：Context Agent 填满队列，Proposal Agent 抽空队列，结果回写 EB；流水线由信号量控制，使沙盒、模型推理、任务队列等资源保持满载（[[hyra-1-0]], section "Hyra Harness"，`raw/articles/Hyra.md:30`）。

### 2. 无评估器任务时升级为双层循环

- 若任务未提供评估器，Hyra 先用任务描述设计初版评估器，内层循环据此优化 solution；内层结束后结合 EB 中的经验优化评估器（提升评估代码效率、减少 reward hacking 风险、加强评估粒度、允许更大进步空间等），再用升级后的评估器开启下一轮，实现 eval 与 solution 的共进化（[[hyra-1-0]], section "Hyra Harness"，`raw/articles/Hyra.md:32`）。
- 给出的例子是「设计世界冠军级国际象棋 AI」：初版评估器为随机 AI 组成的天梯、以 Elo 计分，随着循环推进对手逐步替换为 EB 中更强的 AI（同上）。

### 3. AI for AI：复用 Recursive 的公开设置并小幅超越

- 三个任务：NanoChat Autoresearch（固定预算训练）、NanoGPT Speedrun（训练速度）、SOL-ExecBench（GPU kernel 优化）；声称复用了 Recursive 的任务定义、评测协议与初始化 solution（[[hyra-1-0]], section "AI for AI"，`raw/articles/Hyra.md:40-43`）。
- 结果表（来源论述）：

| Benchmark | 任务 | 指标 | Recursive | Hyra-1.0 |
| --- | --- | --- | --- | --- |
| NanoChat Autoresearch | 模型训练 | Validation BPB ↓ | 0.9109 | 0.9015 |
| NanoGPT Speedrun | 模型训练加速 | Time to 3.28 loss ↓ | 77.5s | 76.4s |
| SOL-ExecBench | 优化 GPU kernel | Mean SOL ↑ | 0.754 | 0.771 † |

- † 注释：为与 Recursive 的汇报口径一致，Mean SOL 采用 Evaluation Stack v1.0 的评测设定，结果来自**本地运行**（[[hyra-1-0]], section "AI for AI"，`raw/articles/Hyra.md:48-53`）。作者同时说明这些问题的基线本身已具竞争力。
- 关键观察：搜索能力变强会更快触及评估器边界。NanoChat 的 solution 曾把因果语言模型改成近似双向注意力、借未来 token 泄漏异常压低 BPB；SOL-ExecBench 的 solution 曾在正确性检查阶段缓存输出、计时阶段运行空 kernel（[[hyra-1-0]], section "AI for AI"，`raw/articles/Hyra.md:55`）。作者由此主张 eval 必须进入研究循环（见 [[evaluator-solution-co-evolution]]）。

### 4. AI for Science

- 从 EinsteinArena、Erich's Packing Center 等数据库收集 **55 个**数学开放问题（其中许多数十年无进展），声称在 **29 个**问题上刷新历史最好结果（[[hyra-1-0]], section "AI for Science"，`raw/articles/Hyra.md:61`）。
- 数据规律发现：给定 1749–1932 年逐月太阳黑子数据，Hyra 找到递推预测公式，并在 1932–2026 年样本外记录上达到 $R^2 = 0.77$；文中还声称可用于从训练日志发现 scaling law（[[hyra-1-0]], section "AI for Science"，`raw/articles/Hyra.md:63`）。
- 三个设计类案例（[[hyra-1-0]], section "AI for Science"，`raw/articles/Hyra.md:67-69`）：
  - 15 个可训练参数的 Transformer 完成 10 位数加法，较 Adderboard 公开纪录的 36 个参数减少 58.3%。
  - 量子比特路由算法在 IBM Q20 上相比经典 SABRE 提升路由效率 44.4%。
  - 给定 PARP1 靶点口袋设计候选分子，在结合-成药联合评分上超过已上市的 olaparib；作者明确说明这仍只是初步模拟筛选，尚需更严格的模拟、合成与湿实验验证。

### 5. AI for Fun：非标量反馈

- 黑白棋 bot：自对弈 + 双循环比赛评估器（新方案与 EB 中高分 bot 对战，按 Elo 保留 top-k 形成不断增强的对手池），策略从 MinMax 演化到 PUCT+MCTS 混合，最终在 Botzone 730 个参赛程序中排名第 3（[[hyra-1-0]], section "AI for Fun"，`raw/articles/Hyra.md:77`）。
- 3D 建模：仅给一张 2D 参考图，用基于 rubrics 的 VLM judge 从轮廓、比例、结构完整性、材质、视觉相似度打分，多轮修改建模代码与渲染参数；文中称结果优于 Claude Code goal 模式生成的模型（[[hyra-1-0]], section "AI for Fun"，`raw/articles/Hyra.md:84`）。
- 配器：内层用固定规则评估和声/和弦进行/节奏密度/音域冲突，外层根据用户反馈调整评估器；经 7 轮进化从保守四声部发展为多乐器完整配器（[[hyra-1-0]], section "AI for Fun"，`raw/articles/Hyra.md:92`）。
- 结论性表述：Research Agent 的价值不限于单一标量指标；反馈来自对手、视觉模型或真实用户时，模糊目标仍可转化为可迭代的搜索过程（[[hyra-1-0]], section "AI for Fun"，`raw/articles/Hyra.md:97`）。

### 6. 路线图表述

- 作者自述当前展示「仍是 Hyra 非常初步的 demo」，下一步是定义更多「有价值且能被持续提升」的任务——公开基准之外的产品系统、真实 AI 研发流水线、自然科学与工业场景（[[hyra-1-0]], section "The path forward"）。
- 提出 scaffold–data–model 进化循环：更好的 scaffold 产出更好的数据与经验，锻造更强的模型，更强模型再驱动更强的 scaffold 与更多发现（[[hyra-1-0]], section "The path forward"，`raw/articles/Hyra.md:101-103`）。

## 局限与阅读提示

- **厂商自评材料**：全部结果来自作者自述，未附论文、评测脚本细节或第三方复现；除 SOL-ExecBench 明示「本地运行」外，其余结果未说明运行环境（[[hyra-1-0]], section "AI for AI"，`raw/articles/Hyra.md:48`）。
- **基线空间有限**：作者自己承认 NanoGPT Speedrun 等问题经社区长期优化、「剩余提升空间极小」，因此 0.9109→0.9015、77.5s→76.4s 这类改进的**幅度**不宜直接外推为方法优势。
- **口径差异**：SOL-ExecBench 的 0.771 带 † 标记，评测设定为 Evaluation Stack v1.0 且为本地运行，与 Recursive 报道数值的可比性需读者自行核验。
- **未披露项**：未给出算力/预算规模、单任务耗时、Proposal Agent 数量、EB 规模与检索策略、reward hacking 检测的具体机制；文中也没有消融实验区分「框架设计」与「底层模型能力」的贡献。
- **基座模型**：本剪藏未提；同系列 [[hyra-2-0]] 称 1.0 由开源模型 Hy3 驱动（`raw/articles/Hyra 2.0.md:18`）。
- **证据成熟度不均**：PARP1 分子设计是初步模拟筛选；3D 建模、配器、黑白棋 bot 依赖主观或平台内部评分，缺少可复核的量化协议。
- **同族资料**：同系列另有 [[hyra-2-0|Hyra 2.0 官方博客]]（`raw/articles/Hyra 2.0.md`），主张从解题转向工具研发；两版差异见 [[hyra-1-0-vs-2-0]]。

## 来源位置索引

| 主题 | 位置 |
| --- | --- |
| 引言与 RSI 脉络 | `raw/articles/Hyra.md:15-19` |
| Harness 设计与双层循环 | `raw/articles/Hyra.md:21-32` |
| AI for AI 结果与 reward hacking 观察 | `raw/articles/Hyra.md:38-57` |
| AI for Science（数学、数据、设计案例） | `raw/articles/Hyra.md:59-69` |
| AI for Fun（黑白棋、3D、配器） | `raw/articles/Hyra.md:71-97` |
| 路线图 | `raw/articles/Hyra.md:99-103` |
| 脚注与外部参照 | `raw/articles/Hyra.md:107-123` |

## 关联页面

- 后续版本：[[hyra-2-0]]（对照：[[hyra-1-0-vs-2-0]]）
- 工具：[[hyra]]
- 概念：[[tool-centric-research]]、[[research-memory-tiers]]、[[recursive-self-improvement]]、[[experience-bank]]、[[evaluator-solution-co-evolution]]
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 待验证问题：[[automated-research-open-questions]]
