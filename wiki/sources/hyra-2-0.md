---
title: "Hyra 2.0：从解题到造工具（腾讯混元官方博客）"
type: source
created: 2026-11-11
updated: 2026-11-12
tags:
  - mle
  - research-agent
  - tool-use
  - recursive-self-improvement
  - verification
source_path: raw/articles/Hyra 2.0.md
source_kind: article
status: draft
---

# Hyra 2.0：从解题到造工具（腾讯混元官方博客）

## 元数据

| 项目 | 值 |
| --- | --- |
| 原始标题 | Hyra 2.0：自动化科研的下一步 |
| 发布方 | 腾讯混元（剪藏 frontmatter 署名为「腾讯混元」） |
| 原始链接 | <https://hy.tencent.com/research/hyra2?langVersion=zh> |
| 形态 | 官方技术博客 / 版本发布公告 |
| 剪藏文件 | `raw/articles/Hyra 2.0.md`（215 行） |
| 剪藏 SHA-256 | `be09ef49bd6536215def8d36c6e2bf388d184fd138a6c761e0928092c8d69ef4` |
| 发布时间 | 剪藏 `published` 字段为空；`created` 为 2026-11-10（仅为剪藏时间，不等于发布时间）。原文自称距 Hyra 1.0 发布「三个月」（`raw/articles/Hyra 2.0.md:18`），但 1.0 的发布时间同样未记录，故不写具体日期 |

剪藏为中文原文，含 23 条脚注与 7 张图（外加 1 段视频）。2026-11-11 已将 7 张图与视频抓取入库到 `raw/assets/hyra-2-0/`（`figure-1.png`–`figure-7.png`、`hand-video.mp4`），并逐张目视核查（视频抽 3 帧）；2026-11-12 将剪藏中的签名外链改为本地相对路径，以避免签名参数中的访问密钥 ID 进入远程仓库。核查结果另见「图表核查」一节，文件尺寸与 SHA-256 见该节。

## 概要

博客把 Hyra 的第二代定位从「解题」改写为「造工具」：通用 agent 不再只输出一个答案，而是参与研发和升级科学模型、研究工作流与软件工具，让一次投入的成果被后续研究反复使用（[[hyra-2-0]], section "从答案到工具"，`raw/articles/Hyra 2.0.md:38-59`）。

为支撑这件事，Hyra 2.0 把 1.0 的单循环 harness 换成面向**长周期、多项目**的研究组织：三层角色分工（Overall / Mission / Worker）、持久化任务队列与调度器、并行进展看板（Research Board）、独立复核 Worker，以及分层的记忆与知识库（Books）（[[hyra-2-0]], section "为升级工具而生的 Harness"，`raw/articles/Hyra 2.0.md:61-85`）。

随后公开了七组成果：两个从头训练的科学模型（HyInv、HyPLM-Flash）、一个原子级力场基础模型（HyFF）、一套数学证明工作流（HyProver）、一套量子码构造与验证系统（HyQEC）、一个自主设计的灵巧手结构（旋梯手），以及香山 RISC-V 处理器的性能与正确性工作（[[hyra-2-0]], section "初步成果"，`raw/articles/Hyra 2.0.md:87-169`）。全部数字来自作者自述，未附论文与评测脚本。

## 关键主张

### 1. 目标转向：从答案到工具

- 来源的核心论点：「刷题抽卡」的突破不会自动提升下一次研究的能力，不形成「智能复利」；升级一件工具则让所有下游任务共同受益（[[hyra-2-0]], section "从答案到工具"，`raw/articles/Hyra 2.0.md:40-42`）。
- 工具按能力载体分两类：**符号化工具**（代码、规则、流程：搜索算法、验证程序、数据管线、研究工作流）与**参数化工具**（把规律存进模型参数：通用语言模型、蛋白质设计模型）（`raw/articles/Hyra 2.0.md:46-49`）。
- 工具按通用性分层，越往上研发周期越长、环境越复杂：解出一道题 → 一类题的工作流或更强的推理模型 → 能自动训练这种模型的系统 → 能做任何任务的系统（AGI）（`raw/articles/Hyra 2.0.md:53-55`）。
- 作者判断「加速一个现成模型」相对容易（反馈快），更难的是端到端做出新模型（设计、实现、部署）；Hyra 2.0 的目标表述为「**成为最会创造工具的系统**」（`raw/articles/Hyra 2.0.md:57`）。
- 在 RSI 语境下，作者希望 Hyra 2.0 升级训练基础设施、数据合成管线与推理系统，帮助训练下一代 Hy 模型，而更强的模型又成为 Hyra 手里的工具（`raw/articles/Hyra 2.0.md:59`）。

### 2. Harness：为长周期、多项目的工具研发而重构

- 动机陈述：训练一个模型要经历方案设计、代码实现、训练、评估与反复修改，可能持续数天到数周；多项目并行还要处理实验等待、资源分配与方向调整，「很难靠一个不断延长的 agent session 来管理」（[[hyra-2-0]], section "为升级工具而生的 Harness"，`raw/articles/Hyra 2.0.md:63`）。
- 接入真实研发环境：在企业微信与研究者对齐需求、在算力平台提交与管理实验、在代码仓库实现与验证改动、保存实验过程与产物；**较大算力开销与共享知识同步的操作需人工审批**，获准后由系统继续推进（`raw/articles/Hyra 2.0.md:65`）。
- 三层角色（`raw/articles/Hyra 2.0.md:67`）：
  - **Overall**：与人类研究者沟通，管理项目优先级，协调资源与关键决策。
  - **Mission**：长期负责一个研究问题，决定路线、设计实验、组织 Worker 并对结论负责；判断哪些假设继续投入、哪些结果需要复核、何时换方向。
  - **Worker**：独立读代码、做实验、分析失败、提出新想法，**可以质疑 Mission**；不同 Worker 可用不同模型，提供不同分析与复核意见。
- 调度机制：Mission 可同时安排方向探索、实验执行与独立复核；任务进入**持久化队列**后按优先级、依赖关系与资源上限启动；等待训练结果或人工审批时其他任务继续；agent 可设置**下一次唤醒时间**在检查节点后接着工作（`raw/articles/Hyra 2.0.md:69`）。
- 协作记录：并行进展写入 **Research Board**（想法、实验发现、证据、关键决策）；普通进展由其他 agent 异步阅读，必须处理的问题走消息通知且接收方须明确回应（`raw/articles/Hyra 2.0.md:71`）。
- 图 2 补充了正文未写的两处机制：**共享 tmux 会话可见性**（任何 agent 可查看其他 agent 的会话，`Any agent can inspect any other agent's session`）与**直接对话通道**（Overall↔Mission↔Worker、Worker↔Worker）；队列画为 Ready / Running / Waiting 三态，并以 `Wait → wake → resume` 标注恢复路径（图 2，`raw/articles/Hyra 2.0.md:73`，已目视核查）。

### 3. 验证纪律：证据随报告提交、复核独立起步

- Worker 汇报完成时须附代码版本、实验结果、日志或原始运行轨迹；Mission 可查看实际操作记录，检查实验是否支持报告结论（[[hyra-2-0]], `raw/articles/Hyra 2.0.md:77`）。
- 需要独立验证时启动**新的 Review Worker**，让它**先接触原始问题与证据**，以减少执行者结论的锚定影响。
- 复核结果不追求「全部通过」：可能发现错误，也可能发现现有证据只支持一个**范围更窄**的结论，「这些都需要如实保留」（`raw/articles/Hyra 2.0.md:77`）。

### 4. 记忆分层与知识成熟度分级

- 三类存储分开：项目状态（任务队列 + Research Board）／每个 agent 的**分层记忆**（当前认识、关键假设、已排除路线、下一步计划）／跨任务共享知识库 **Books**（[[hyra-2-0]], `raw/articles/Hyra 2.0.md:81`）。
- Books 的记录须说明**做法、理由、适用条件**并附原始证据，按可信度分三级（`raw/articles/Hyra 2.0.md:83`）：
  - **观察**：单次实验发现的现象，供后续参考；
  - **已验证**：经过独立复现，才能作为可靠依据；
  - **约定**：影响团队行为的约定，需要人类确认。
  - 新证据出现后，已有记录也会被修正。
- 复盘时把可迁移方法整理成 **skill**（例如训练模型的经验已成套），skill 本身也是 Hyra 2.0 造出的工具，会在使用中继续修订；作者把这部分称为「持续学习」的构成（`raw/articles/Hyra 2.0.md:85`）。
- 经验留存的经济学理由：单次尝试越昂贵，越有必要留下排查方法、验证流程与自动化检查（`raw/articles/Hyra 2.0.md:79`）。

### 5. 科学模型：HyInv、HyPLM-Flash、HyFF

- **HyInv**：蛋白质逆折叠（给定骨架找序列），196 万参数的图神经网络；预训练、监督微调、强化学习与推理优化均由 Hyra 2.0 完成，并设计了「先起草、再改写」的两步采样；相比 ProteinMPNN「吻合程度更好、生成速度快了 57 倍」（[[hyra-2-0]], section "HyInv"，`raw/articles/Hyra 2.0.md:93-99`）。图 3 左给出具体基准：HyInv 吞吐标为 **2,532 designs/s**、设计质量 scTM ≈ **0.838**；ProteinMPNN 约 40 designs/s、scTM ≈ 0.826（读图近似，`raw/articles/Hyra 2.0.md:101`，已目视核查）。以 2,532 designs/s 换算，128 条序列约 50.6 ms，与小标题「50 毫秒设计 128 条」一致（本库推算）；但硬件、序列长度与批量条件仍未给出。
- **HyPLM-Flash**：6.5 亿参数、累计 7260 亿训练 token，为 ESM-2 650M 的 0.73 倍、E1-600M 与 ESM C 600M 的「五分之一到八分之一」；在比较的 12 个蛋白质语言模型中推理最快，35 个评测任务上平均排名第 3（[[hyra-2-0]], section "HyPLM"，`raw/articles/Hyra 2.0.md:109`）。图 3 右显示其吞吐标为 **131k residues/s**（全场最右），平均归一化得分 ≈ **0.73**，略低于 E1-600M（≈0.78）与 ESM-2 3B（≈0.76），落在虚线标出的 Pareto 前沿上，因此「推理最快 + 平均第 3」与图一致（`raw/articles/Hyra 2.0.md:101`，已目视核查）。研发细节称将通过技术报告公开。
- **HyFF**：面向分子模拟的原子级基础模型（输入元素种类与坐标，预测能量与每个原子的受力，作机器学习力场）。五种规模 230 万–12 亿参数，大规模量子化学数据训练，覆盖 83 种元素，支持带电与开壳层体系；**每个原子只调用与自身元素对应的子网络**以控制开销（[[hyra-2-0]], section "HyFF"，`raw/articles/Hyra 2.0.md:115-117`）。与 16 个公开模型比较，五个规模「全部位于精度–速度帕累托前沿」；已用于实际模拟：近 6 万原子的水通道蛋白膜体系跑 100 ns（观察到水分子单列逐个穿过），4.6 万原子酶体系观察到一次完整断键–成键过程（`raw/articles/Hyra 2.0.md:115-117`）。图 4 用 **Presto / Vivace / Moderato / Andante / Largo** 五个名称标注五档（按单步耗时递增排列），能量 MAE 从 ≈5.9 降到 ≈0.35 meV/atom、力 MAE 从 ≈48 降到 ≈5.5 meV/Å，单步耗时约 1–100 ms；对比曲线含 DPA4 / DPA4C(Nano) / DPA4-Pro / UMA-S-1.2 / MACE-OMol-0 / OrbMol / UBio-MolFM，图注标注 `hollow = published accuracy`（`raw/articles/Hyra 2.0.md:119`，已目视核查）。

### 6. 数学与量子码：HyProver、HyQEC

- **HyProver**：面向纯数学证明的 agent 工作流，强调 token efficiency 与探索能力；通过持续检索、转化问题、提出中间猜想推进，而非「盲目摊大饼」（[[hyra-2-0]], section "HyProver"，`raw/articles/Hyra 2.0.md:125`）。图 5（已目视核查，`raw/articles/Hyra 2.0.md:129`）把流程画成：研究问题 P⇒Q → 主 agent 带子 agent → 中间猜想 C₁/C₂/C₃ → Proof workers 产出候选证明 → 独立 LLM（fresh context）复核 → `Accept & publish` 或回到 `Refine`；上半部分存入共享研究笔记的 Accepted lemmas（L₁–L₃）又作为后续证明的前提（`Reuse as premises`）（图注同见 `raw/articles/Hyra 2.0.md:131`）。作者称已「完整解决许多开放数学问题」，并与数学家协同核验、整理了**丘成桐的圆周作用猜想**与**五维 Petty 等距点猜想**的证明（`raw/articles/Hyra 2.0.md:127`）。
- **HyQEC**：面向量子低密度奇偶校验码（qLDPC）的构造与验证系统，支持新码构造、码距搜索与精确码距证明（[[hyra-2-0]], section "HyQEC"，`raw/articles/Hyra 2.0.md:135-141`）：
  - 构造侧：设计 **Split Pauli-CPM** 算法，构造并严格验证码 $[488, 123, 22]$，编码效率 $kd^{2}/n = 122.0$；在**非 CSS、校验权重不超过 8** 的条件下，比 qLDPC Challenge 原榜首提升约 **48%**。
  - 验证侧：GPU 加速搜索 + 两套独立的精确证明工具，搜索吞吐为原有 8 线程 CPU 工具的 **8–55 倍**；把原榜首码的码距上界从 **95 收紧至 86**，并为榜单 **187 个条目中的 178 个**完成精确码距认证。
  - 工具称后续将发布。

### 7. 物理与硬件：旋梯手、香山

- **旋梯手**：Hyra 2.0 在手内物体旋转任务中自主发现五指设计——五根手指绕掌心排列、长度递增「像一段旋转楼梯」（[[hyra-2-0]], section "灵巧手"，`raw/articles/Hyra 2.0.md:145-151`）。初始测试比论文最优设计慢约 **11%**，但经**同等预算**的策略微调后反超：DR5 仿真中旋转速度 3.277 rad/s，比同样微调过的论文最优设计快 **28.6%**（3.28 vs 2.55 rad/s），目标完成数 +21%，失稳率更低。作者明确指出结果**仅限仿真、尚未真机验证**。图 6（已目视核查，`raw/articles/Hyra 2.0.md:153`）给出结构对照：Hyra Hand 五指长度 20.4 / 21.4 / 22.4 / 23.4 / 24.5 cm 绕掌心单调递增，Dextra Baseline 为 20.4 / 21.9 / 22.2 / 23.2 / 21.9 cm（非单调）；后者总长 20.4–23.2 cm，前者 20.4–24.5 cm，整体更大。配套视频（`raw/articles/Hyra 2.0.md:155`，已抽帧核查）字幕显示 3.28 vs 2.55 rad/s、净旋转 +18.3 vs +13.7 rad，并注明 `clip ratio ≈ full-eval mean`。
- **香山**：围绕这一开源 RISC-V 处理器做性能优化与正确性验证（[[hyra-2-0]], section "香山"，`raw/articles/Hyra 2.0.md:161-165`）：
  - 性能：修复 SPEC CPU2006 暴露的多项问题并结合调优，每 GHz 得分几何均值提升 **5.79%**，单核频率不变，面积 +0.3%；按官方性能看板的历史迭代速度**估算**，约等于官方团队 2–3 个月的成果。
  - 正确性：新设计的 fuzz 测试发现 7 个 bug，加 3 个独立发现共 **10 个**，均不在官方 issue 列表中；用 **TLA+** 形式化验证找到多核缓存一致性反例，相关修复已贡献给 OpenLLC。

### 8. 参与下一代 Hy 模型与对三个问题的回答

- 内部应用列举三项：预训练（优化训练基础设施、参与新训练技术验证、缩短方案到结果的周期）、后训练（自动化数据合成管线）、推理（优化推理镜像与系统以降本）（[[hyra-2-0]], section "下一代 Hy 模型"，`raw/articles/Hyra 2.0.md:173-177`）。
- 对开篇三问的回答（[[hyra-2-0]], section "回到开头的三个问题"，`raw/articles/Hyra 2.0.md:183-189`）：
  - **科研如何变化**：AI 的贡献从「解出具体问题」延伸到「提高一类问题的研究能力」；衡量成果要同时看它为后续留下了什么（证明工作流、科学模型、验证方法）。
  - **如何更经济可持续**：工具研发的前期投入由后续许多任务分摊；更快模型降低单次调用成本，更好工作流减少重复探索，skill 让后续任务少走弯路。
  - **暂时解不了怎么办**：先定位限制来源——模型能力不足、搜索效率不够，还是验证与实验反馈太慢；把该限制作为下一轮工具研发目标，同时尽量留下可复用方法与工具。

## 成果数字一览

| 领域 | 成果 | 报告数字 | 条件/口径 |
| --- | --- | --- | --- |
| 蛋白质设计 | HyInv | 196 万参数；比 ProteinMPNN 快 57× | 图 3：2,532 designs/s、scTM≈0.838；对照 ProteinMPNN ≈40/s、≈0.826（读图近似） |
| 蛋白质语言模型 | HyPLM-Flash | 650M 参数 / 726B token；12 个 PLM 中推理最快；35 任务平均第 3 | token 量为 ESM-2 650M 的 0.73×；图 3：131k residues/s、得分≈0.73（第三） |
| 分子模拟 | HyFF | 5 档 2.3M–1.2B，83 元素；全部在 Pareto 前沿 | 对照 16 个公开模型；图 4：能量 MAE ≈5.9→≈0.35 meV/atom、力 ≈48→≈5.5 meV/Å、步进 ≈1–100 ms |
| 数学 | HyProver | 圆周作用猜想、五维 Petty 等距点猜想 | 称已与数学家协同核验；剪藏内无证明工件链接 |
| 量子纠错 | HyQEC | $[488,123,22]$，$kd^2/n=122.0$，+48%；码距 95→86；178/187 认证；搜索 8–55× | 非 CSS、校验权重 ≤8；对照 qLDPC Challenge 榜单 |
| 机器人 | 旋梯手 | DR5 仿真 3.277 rad/s，+28.6%；完成数 +21% | 同预算微调后；仅仿真；图 6：五指 20.4–24.5 cm；视频：净旋转 +18.3 vs +13.7 rad |
| 处理器 | 香山 | 每 GHz 几何均值 +5.79%；面积 +0.3%；10 个 bug；1 个一致性反例 | 频率不变；「≈官方 2–3 个月」为按看板历史速度的估算 |

（全部为来源论述，[[hyra-2-0]], section "初步成果"，`raw/articles/Hyra 2.0.md:87-169`。）

## 图表核查（2026-11-11）

7 张图与 1 段视频已抓取到 `raw/assets/hyra-2-0/` 并逐张打开目视核查（视频用 `ffmpeg` 抽 2s/7s/12s 三帧）。下表仅记录**图中看得见**的内容；读图得到的近似值用「≈」标注，不替代原始数据。

本地副本与校验值（2026-11-11 抓取）：

| 文件 | 尺寸 | SHA-256 |
| --- | --- | --- |
| `figure-1.png` | 2400×1347 | `8e07e4ab4712f9de666b554d1a9306d618b306f15da6da70b7e897deba41d3f7` |
| `figure-2.png` | 2400×1680 | `cbdad0d2ee434cc1166cd52e40cfe537e352fa3dd439e30bfda3126e009cbcfa` |
| `figure-3.png` | 2400×1110 | `0f7d3d8d36649dbe06a3ca4c610b5ad5b766b6900460b53422cf859eab395b02` |
| `figure-4.png` | 2400×1110 | `98ca07a0ded42ffd90c0a3b7aee53321c1164ec1eb75967f0f5f905146509707` |
| `figure-5.png` | 2400×1313 | `922167e6c7a509dcc8ad9c648b1e8da5bd677a6d1629c8bc3550cefc5001cdbd` |
| `figure-6.png` | 1672×941 | `a05bbbc30e8f15f2c4f1b1efa48a43e18148d463cc02516f85c949cde025efc8` |
| `figure-7.png` | 2400×990 | `2c2adcecda62caa915c8956588dddd7517257f3758e41ecb9aaf074296f67eec` |
| `hand-video.mp4` | 2560×950，15.0 s | `5e8109313f6b2f507ea355e826322c6b7a7b84ad534f6f0740bad168e2e101c3` |

| 图 | 位置 | 核查到的内容 | 与正文的关系 |
| --- | --- | --- | --- |
| 图 1 | `:34` | 左侧为「研究者研发专用模型（AlphaFold）」与「agent 解研究问题」，下方为 Hyra「agent 造可复用工具」：科学模型（HyInv·HyPLM·HyFF）/ 研究工作流（HyProver）/ 软件工具箱（HyQEC）→ 复用到许多研究任务 | 补充了分类三元组；图中**未**包含旋梯手与香山 |
| 图 2 | `:73` | 三层角色 + 多 Mission；共享 tmux 可见性；直接对话通道；队列三态与 `Wait → wake → resume`；Research environment = Team chat / Repositories / Compute / Artifacts；Human approval = major compute use + shared-knowledge synchronization；下方三层记忆与 Books→Skills；右上角 `DAYS TO WEEKS` | 补充正文未写的 tmux 可见性与对话通道；与正文一致 |
| 图 3 | `:101` | 左：HyInv 2,532 designs/s、scTM≈0.838；ProteinMPNN ≈40/s、≈0.826；`57× faster` 箭头；右：HyPLM-Flash 131k residues/s、得分≈0.73，E1-600M≈0.78、ESM-2 3B≈0.76，虚线为 Pareto frontier | 为「吻合更好」「50 ms/128 条」「平均第 3」提供图中依据 |
| 图 4 | `:119` | 能量 / 力 MAE 对单步耗时（约 1–1,000 ms）；HyFF 五档 Presto/Vivace/Moderato/Andante/Largo；标注 `2.5× faster`、`11× faster`；`hollow = published accuracy` | 为「全部在前沿」提供图中依据，但需按档位理解 |
| 图 5 | `:129` | 研究问题 → 主/子 agent → 中间猜想 C₁–C₃ → Proof workers → 独立 LLM 复核（fresh context）→ Accept & publish / Refine；共享笔记分 Open conjectures 与 Accepted lemmas，后者 `Reuse as premises` | 细化正文的「提出猜想 → 证明 → 复核 → 作为前提」 |
| 图 6（+视频） | `:153`、`:155` | 指长对照见 §7；视频字幕 3.28 vs 2.55 rad/s、+28.6%、净旋转 +18.3 vs +13.7 rad、`clip ratio ≈ full-eval mean` | 数字与正文一致；尺寸差异为新发现 |
| 图 7 | `:167` | 左：HyQEC 在物理比特上做稀疏 stabilizer checks（X/Y/Z + S），认证 `d ≥ w` 与 `d ≤ w` → `d = w`；右：香山 RISC-V cores + shared cache（OpenLLC），`Tune performance` 与 `Check correctness → Refine design` 双回路 | 与正文的构造/精确码距与性能/正确性两条线一致 |

**未做的核查**：图中曲线未取点、未复算任何数值；视频只看三帧；未验证图中标注的硬件与环境（图内无此类信息）。

## 局限与阅读提示

- **厂商自评材料**：仍无技术报告、评测脚本或第三方复现；除 HyQEC 对照公开榜单、香山对照官方 issue 列表外，多数结果只有自述口径（[[hyra-2-0]], section "初步成果"）。
- **基座模型未披露**：Hyra 2.0 的博客称 Hyra 1.0 由开源模型 Hy3 驱动（`raw/articles/Hyra 2.0.md:18`）；1.0 的剪藏本身未提基座模型。本篇未说明 Hyra 2.0 使用的模型族与版本，因此无法判断成绩来自 harness、工具研发流程还是底层模型升级。
- **算力与成本缺失**：未给各项目的 token/GPU 预算、单项目时长、人工介入次数与审批频率；「更经济、更可持续」缺少可核算数据，只有机制论证（`raw/articles/Hyra 2.0.md:185`）。
- **标题与正文口径不一**：`50 毫秒设计 128 条蛋白质序列`仅见于小标题（`raw/articles/Hyra 2.0.md:91`）。
- **带条件的数字不宜外推**：+48% 限定非 CSS 与校验权重 ≤8；`8–55×` 区间极宽且未说明基准规模；+28.6% 是仿真中「同预算微调后」的对比，初始设计本身慢约 11%；`≈2–3 个月官方迭代`是按看板历史速度的估算（`raw/articles/Hyra 2.0.md:149-151`、`:163`）。
- **证明核验依赖人工**：数学结果称「协同数学家已核验并整理」，但剪藏没有证明工件或链接，脚注 16 指向 Petty 1971 年原文而非本工作的证明（`raw/articles/Hyra 2.0.md:127`、`:208`）。
- **RSI 部分为参与性表述**：「参与升级下一代 Hy 模型」是工作描述，没有可归因的改进测量（`raw/articles/Hyra 2.0.md:173-179`）。
- **对照基准偏旧**：处理器部分使用 SPEC CPU2006，属较老的基准族，与当前常用版本的可比性需核（`raw/articles/Hyra 2.0.md:163`、`:212`）。
- **图 3 的对数轴放大视觉差距**：左图横轴为对数刻度，「57×」在图上被画成两段短距离；实际是 2,532 vs 约 40 designs/s。质量差只有约 0.012 scTM，属读图近似范围（图 3，已目视核查）。
- **图 4 的「全部在前沿」需按档位理解**：五档精度差近 17 倍（最快档图中标注为 Presto，≈5.9 meV/atom、≈1 ms/step；最慢档 Largo ≈0.35 meV/atom、≈85 ms/step）；最慢档 Largo 在精度上接近 DPA4-Pro（≈1,000 ms/step，图右端），但速度仍快约一个数量级。因此「五个模型全部位于 Pareto 前沿」指的是每档在自己速度附近最优，而非整体优于所有对比模型（图 4，已目视核查）。
- **灵巧手的尺度差异未被讨论**：图 6 显示 Hyra Hand 五指 20.4/21.4/22.4/23.4/24.5 cm（绕掌心递增），Dextra Baseline 为 20.4/21.9/22.2/23.2/21.9 cm（非单调），前者整体更大；来源未讨论尺寸差异对旋转速度比较的影响（图 6，已目视核查）。

## 来源位置索引

| 主题 | 位置 |
| --- | --- |
| 剪藏元数据与摘要 | `raw/articles/Hyra 2.0.md:1-11` |
| 引言：三个问题与「工具升级」答案 | `raw/articles/Hyra 2.0.md:16-33` |
| 从答案到工具：分类、通用性阶梯、RSI 表述 | `raw/articles/Hyra 2.0.md:38-59` |
| Harness：动机、研发环境与人工审批 | `raw/articles/Hyra 2.0.md:63-65` |
| 三层角色分工 | `raw/articles/Hyra 2.0.md:67` |
| 队列、调度与唤醒时间 | `raw/articles/Hyra 2.0.md:69` |
| Research Board 协作与通知分流 | `raw/articles/Hyra 2.0.md:71` |
| 证据提交、Mission 核查、独立 Review Worker | `raw/articles/Hyra 2.0.md:77` |
| 经验留存经济学 | `raw/articles/Hyra 2.0.md:79` |
| 三层存储与 Books 三级成熟度 | `raw/articles/Hyra 2.0.md:81-83` |
| skill 沉淀与「持续学习」 | `raw/articles/Hyra 2.0.md:85` |
| HyInv | `raw/articles/Hyra 2.0.md:91-111` |
| HyPLM | `raw/articles/Hyra 2.0.md:105-111` |
| HyFF | `raw/articles/Hyra 2.0.md:113-121` |
| HyProver | `raw/articles/Hyra 2.0.md:123-131` |
| HyQEC | `raw/articles/Hyra 2.0.md:133-141` |
| 旋梯手 | `raw/articles/Hyra 2.0.md:143-157` |
| 香山 | `raw/articles/Hyra 2.0.md:159-169` |
| 下一代 Hy 模型 | `raw/articles/Hyra 2.0.md:171-179` |
| 对三个问题的回答 | `raw/articles/Hyra 2.0.md:181-189` |
| 图 1–图 7 与视频 | `raw/articles/Hyra 2.0.md:34`、`:73`、`:101`、`:119`、`:129`、`:153-157`、`:167` |
| 脚注 1–23 | `raw/articles/Hyra 2.0.md:193-215` |

## 关联页面

- 版本对照：[[hyra-1-0-vs-2-0]]
- 工具：[[hyra]]
- 概念：[[tool-centric-research]]、[[research-memory-tiers]]、[[recursive-self-improvement]]、[[experience-bank]]、[[evaluator-solution-co-evolution]]
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 前置来源：[[hyra-1-0]]
- 待验证问题：[[automated-research-open-questions]]
