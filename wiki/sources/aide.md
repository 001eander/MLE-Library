---
title: "AIDE：AI-Driven Exploration in the Space of Code（Weco AI，arXiv 2502.13138v1）"
type: source
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - code-search
  - mle-bench
  - evaluator-design
source_path: raw/papers/2502.13138v1.pdf
extracted_path: raw/extracted/2502.13138v1.md
source_kind: pdf
published: 2025-02-18
status: stable
---

# AIDE：AI-Driven Exploration in the Space of Code（Weco AI，arXiv 2502.13138v1）

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | AIDE: AI-Driven Exploration in the Space of Code |
| 作者 | Zhengyao Jiang\*、Dominik Schmidt\*†、Dhruv Srikanth、Dixing Xu、Ian Kaplan、Deniss Jacenko、Yuxiang Wu\*（\* 同等贡献，† 工作在 Weco AI 期间完成） |
| 机构 | Weco AI、Runway ML（p. 1） |
| 标识 | arXiv:2502.13138v1 [cs.AI]，2025-02-18（p. 1 边栏） |
| 形态 | 预印本 / 方法论文（含形式化定义 + 自建评测 + 第三方评测汇总）；未见同行评审版本 |
| 篇幅 | 17 页（正文 p. 1–10，参考文献 p. 10–11，附录 A–C p. 12–17） |
| 本地 PDF | `raw/papers/2502.13138v1.pdf`（648,632 字节） |
| PDF SHA-256 | `5fc52573eab9c8cb568b0c5aab6ee75752d198504babf8278ad23fe3d542cb41` |
| 提取文本 | `raw/extracted/2502.13138v1.md`（`pdftotext -layout`，含 `## Page N` 分页标记） |
| 代码 | <https://github.com/WecoAI/aideml>（p. 1；文中称 2024-04 开源，p. 2、p. 4） |

入库说明：本页依据本地 PDF 及其确定性提取文本撰写；Table 1–3 的数值已与提取文本逐项比对，第 4、7、8、9、15 页（图 1–4、图 6–7、Table 3 与附录 C 表头）已渲染并目视核查（见「图表核查」）。数字出处一律给到页码。

## 概要

论文把机器学习工程表述成一个**代码优化问题**：一个解 `s` 就是一段可独立运行的 Python 脚本，评估器 `h : S → R` 给出标量分，目标为 `s* = argmax h(s)`（式 (1)，p. 2）。与 ReAct 式 agent「把全部历史观测追加进上下文」不同，AIDE 把所有历史解组织成**解树** `T`，用硬编码策略 `π(T)` 挑一个节点作基准，用编码算子 `f(s, Σ(T))` 生成新脚本，评估后挂回树上（算法 1，p. 3）。摘要算子 `Σ(T)` 只把性能指标、超参与调试线索写回 prompt，让每个节点独立评估、prompt 不随步数膨胀（p. 4）。

评测分三层：作者自建的 Weco-Kaggle（63 个竞赛，Lite 子集 16 个表格任务，Table 1–2，p. 5–6）、第三方 MLE-Bench（75 个竞赛，AIDE+o1-preview 任意奖牌 16.9%，Table 3，p. 7）与第三方 RE-Bench（7 个 AI R&D 任务，6 小时窗口内平均分超过人类专家，Fig 4，p. 9）。作者给出两条自述局限：自建 holdout 与 Kaggle private 集不完全一致，以及模型可能见过竞赛相关数据（p. 7）。

## 关键主张

### 1. 动机：预定义搜索空间的三重代价

- 论文指出 AutoML、NAS、超参优化都依赖**预定义的配置搜索空间**，定义它本身就是重人力活；而超参搜索算法相对人类专家「偏暴力」，算力效率低，且存在过拟合验证集的风险（p. 1，`raw/extracted/2502.13138v1.md:51-58`）。
- 反命题：LLM 的编码能力让搜索可以**直接发生在代码空间**，借模型自带的领域知识把搜索范围收窄，从而提高样本效率，并因此有望处理深度学习这类算力密集任务（p. 1–2）。
- 与 ReAct 的分野被明确写进 Preliminaries：POMDP 式做法用「全部历史」当状态，缺少结构化拆解手段，且历史不断追加会让 prompt 过大、context 最终填满（p. 2）。

### 2. 形式化：解、解树、三个算子

- 解空间 `S`（如 Python 脚本）、无状态目标 `h`、根解 `s₀` 为空脚本；边 `s → s′` 表示一次改进尝试；节点存脚本与其分数（p. 3）。
- 算法 1（p. 3）：每轮 `sₙ ← f(s, Σ(Tₙ₋₁))` → `vₙ ← h(sₙ)` → `Tₙ ← Tₙ₋₁ ∪ {节点(sₙ,vₙ), 边(s→sₙ)}` → `s ← π(Tₙ)`；结束返回历史最优。
- **搜索策略 π 是硬编码规则**（p. 3）：初始解数量未达标就 draft；仍有 bug 节点且未超过调试深度就 debug；否则 improve，通常针对当前最好的非 bug 解。作者自述两条隐含启发：先铺开多样初始解再持续改进最好的；限制对同一个破损解的调试次数。

### 3. 编码算子：三个入口，improve 只改一处

- **draft**：先让模型写一份简要方案（如指定网络结构或特征工程思路），再产出实现该方案的单文件 Python 程序（p. 3）。
- **debug**：读错误日志与执行轨迹，修 import、张量维度等编码错误，**保留整体思路**（p. 3）。
- **improve**：只提出**恰好一个「原子」改动**（如换优化器、加正则），使该改动对性能的影响可直接测量（p. 4）。
- 作者给出的理由是结构性归因：「三种操作合起来让解树保持结构化，每个新节点都来自父节点的良定义修改」（p. 4）。

### 4. 摘要算子 Σ(T) 与数据预览：无状态但不失忆

- `Σ(T)` 不追加历史日志，只抽取三类信息：性能指标（accuracy、AUC-ROC、test loss）、超参设置（当解包含超参搜索时）、与调试相关的线索（如 traceback 中错位的数组形状）（p. 4）。
- 目的原文表述：摘要让「每次代码修订独立成立」（stateless），同时用先验信息指导后续提议，「获得增量推理的多数好处而不炸大 prompt」（p. 4）。
- 另有静态**数据预览**：把行数、列名、数据划分等元数据存进 workspace 并插入编码 prompt；作者明确说这不是完整 EDA，但足以支撑「选验证集划分、缩放超参」这类决策（p. 4）。

### 5. 自建评测 Weco-Kaggle：分位指标而非奖牌计数

- Weco-Kaggle 共 63 个竞赛，覆盖表格、图像分类、时间序列，部分需 GPU；从中选 16 个复杂度较低、主要 CPU 可跑的任务构成 **Weco-Kaggle Lite**（p. 5，Table 2，p. 6）。
- 协议四步（p. 5）：① 手工把训练数据切成 agent train / holdout test，比例参照 Kaggle private 集但不保证相同；② AIDE 可见 holdout 的**输入但不可见标签**，被要求产出 `submission.csv`；③ 主指标 `Exceeds % of Humans = 100(1 − q)`，`q` 是其在 Kaggle 榜上的分位，优先 private 榜、缺失时用 public 榜；另报 `Above Median`（取自 Chan et al.）；④ 跨竞赛平均。
- 选分位指标的理由：绝对分数分布不可比、奖牌计数太粗（单次单任务会退化成二值），分位既细粒度又可解释为「超过多少比例的人类选手」（p. 5）。
- Lite 子集结果（Table 1，p. 5；提交于 2024 年 2 月，排名为当时真实榜位，Table 2 图注 p. 6）：

| 智能体 | 模型 | Exceeds % of humans ↑ | Above Median (%) ↑ |
| --- | --- | --- | --- |
| AIDE | GPT-4 Turbo | **51.38** | **50.00** |
| Human with ChatGPT | GPT-4 Turbo | 41.17 | 18.75 |
| AutoML (H2O) | N/A | 35.34 | 18.75 |
| AutoGPT (Langchain) | GPT-4 Turbo | 32.34 | 0.00 |

- 单竞赛范围 12.98%–91.92%（Table 2，p. 6）。完整 Weco-Kaggle 平均 `Exceeds % of Humans` 为 48.23%，超人类中位数比例 49.21%（p. 6，Fig 2，p. 7）。

### 6. 第三方 MLE-Bench：AIDE 是当时最强脚手架

- MLE-Bench 为 75 个真实 Kaggle 竞赛的离线框架；本节结果由 Chan et al. 报告，作者只做汇总（p. 7）。
- Table 3（p. 7）关键行：AIDE+o1-preview 有效提交 82.8%、Above Median 29.4%、金 9.4%、任意奖牌 **16.9%**；AIDE+GPT-4o 为 54.9 / 14.4 / 5.0 / 8.7；AIDE+Claude 3.5 为 51.1 / 12.9 / 4.4 / 7.6；AIDE+Llama 3.1 为 27.3 / 6.7 / 1.7 / 3.0。对照：MLAB(GPT-4o) 44.3 / 1.9 / 0.8 / 0.8；OpenHands(GPT-4o) 52.0 / 7.1 / 2.7 / 4.4。除 AIDE+o1-preview（16 种子）与 AIDE+GPT-4o（36 种子）外均为 3 种子，数值为均值 ± 一个标准误。
- Lite（`complexity=low`）子集上，加 AIDE 后四项指标全部显著提升（p. 8，Fig 3）：有效提交 63.6 ± 4.5% → 92.4 ± 2.6%；Above Median 13.6 ± 0% → 59.1 ± 4.5%；金 6.1 ± 2.6% → 21.2 ± 6.9%；任意奖牌 7.6 ± 2.6% → 36.4 ± 7.9%；作者称全部 p < 0.01（双尾 t 检验）。
- 机制解释（p. 8，来源论述）：解树让每次模型调用只聚焦局部问题（例如只调试最有希望的脚本），因此能在 24 小时窗口里反复精修部分解，「系统性地处理 bug 与次优超参，而不是放弃失败的解」；并称 pass@k 增大时奖牌率接近翻倍。

### 7. 第三方 RE-Bench：快迭代赢在时间窗口，输在长程改动

- METR 的 7 个 AI R&D 任务上，AIDE+o1-preview 在**六小时时限内**平均分超过人类科学家，原因是实现与迭代速度更快（p. 9）。
- Optimize a Kernel（Triton）任务上，AIDE 找到的自定义方案「比九位人类专家在 64 小时内都快」（p. 9）。
- 失手场景（p. 9，Fig 4 图注亦明写）：需要处理更大代码库、或单次改进需多步交互的环境；Agent for Rust CodeContests 上「倾向于反复打局部补丁，而不去发现新策略」——作者把根因归给「简单的贪心策略可能陷入局部最优」。

### 8. 成本与代码膨胀（附录 B）

- 单任务 LLM 推理成本（GPT-4 Turbo，2024 年 1–2 月定价）：最高约 $2.50/任务，多数低于 $1.50（p. 13，Fig 7，p. 15，已目视核查）。作者据此主张比人类专家或传统 AutoML 服务便宜。
- 生成的代码复杂度（LOC、LLOC、Volume、N1、MI 的聚合）随迭代步数上升，第 1 步后基本单调爬升（p. 13，Fig 6，p. 15，已目视核查；x 轴到 18 步）。

## 结果数字一览

| 层面 | 设置 | 关键数字 | 口径/条件 |
| --- | --- | --- | --- |
| 自建 | Weco-Kaggle Lite（16 竞赛） | Exceeds 51.38% / Above Median 50% | GPT-4 Turbo；2024-02 手工提交、真实榜位（p. 5–6） |
| 自建 | 完整 Weco-Kaggle | Exceeds 48.23% / Above Median 49.21% | 63 个竞赛（正文口径，附录实列更多，见「分歧」）（p. 6–7） |
| 第三方 | MLE-Bench（75 竞赛，pass@1） | 任意奖牌 16.9 ± 1.1%、金 9.4 ± 0.8% | AIDE+o1-preview，16 种子（p. 7） |
| 第三方 | MLE-Bench Lite（low 复杂度） | 有效提交 63.6→92.4%；Above Median 13.6→59.1%（加 AIDE） | o1-preview ± AIDE，p<0.01（p. 8） |
| 第三方 | RE-Bench（7 任务） | 6 小时内平均分超人类；Kernel 任务快于 9 位专家（64h 预算） | 仅「时间受限」比较；人类随后反超（p. 9） |
| 成本 | 单任务 LLM 推理 | 最高 ≈$2.50，多数 <$1.50 | GPT-4 Turbo 2024-01/02 定价（p. 13） |

## 图表核查（2026-11-11）

渲染了第 4、7、8、9、15 页（`pdftoppm -r 180`，存放在 `.cache/pdf/5fc52573…/`，未入库）并逐页目视核查；表格数值另与提取文本逐项比对。读图得到的近似值以「≈」标注。

| 图表 | 位置 | 核查到的内容 | 与正文关系 |
| --- | --- | --- | --- |
| Fig 1 解树示例 | p. 4 | `s0`（空解）→ 三个 `f: draft` 分支；`s1`、`s3` 标为 bug（深色），`s2` 有效；`s1` 下两次 `f: fix`（`s4` 仍为 bug、`s5` 有效）；`s2` 经 `f: improve` 得 `s6`（红色 = optimal）；`s3` 经 `f: fix` 得 `s7` | 与算法 1、三算子一致；图例为 Empty/Bug/Valid/Optimal 四态，正文未逐一定义这些状态 |
| Fig 2 分布 | p. 7 | 柱状图按 `Exceeds % of Humans` 降序排列（未逐根计数），最高≈100%、最低≈0%，长尾平滑下降 | 支撑「平均 48.23% 但方差大」的表述 |
| Table 3 | p. 7 | 6 行数值与提取文本完全一致；图注写明 AIDE+o1-preview 用 16 种子、AIDE+GPT-4o 用 36 种子 | 与 `raw/articles/MLE-Bench.md` 剪藏记载的 16.9%/9.4% 一致（交叉核对，该剪藏尚未入库） |
| Fig 3 Lite 柱状图 | p. 8 | 黑=o1-preview、粉=o1-preview+AIDE；四组数值 63.6±4.5→92.4±2.6、13.6±0→59.1±4.5、6.1±2.6→21.2±6.9、7.6±2.6→36.4±7.9 | 与正文逐字一致 |
| Fig 4 RE-Bench 曲线 | p. 9 | 横轴 0–8 小时、纵轴 average normalized score（95% CI）；`AIDE` 为红线、`Human` 为黑线（已放大图例核对）。红线约 4h 到 ≈0.30、5h 到 ≈0.36 后基本走平到 8h（置信带变宽）；黑线 6h 后加速，8h 处 ≈0.65 | **新增读数**：正文只说「6 小时内超过人类、人类随后追平」，图上 AIDE 从 ≈5h 起近乎平台，人类在 ≈6h 后反超并继续上升到 ≈0.65 |
| Fig 6 代码复杂度 | p. 15 | 聚合复杂度从 ≈1.0（第 0 步）升到 ≈1.6（第 18 步），第 1 步有小幅下降，第 6–12 步出现平台 | 支撑「迭代使代码变复杂」；x 轴到 18 步，可作迭代预算的间接线索 |
| Fig 7 单任务成本 | p. 15 | 降序柱状：最高 ≈$2.55，少数 >$1.5，多数落在 ≈$0.5–$1.2，最低 ≈$0.15 | 与正文「最高约 $2.50、多数 <$1.50」一致 |
| 附录 C 竞赛表 | p. 15–17 | 逐行统计为 11 + 52 + 17 = **80 行**（每行都有 Data Size），其中含 Lite 的 16 个 | 与正文「63 个竞赛」不一致，见「分歧」 |

**未做的核查**：未复算任何指标；未核查 p. 14 的 Fig 5（AutoGPT 任务描述符示例）；未核对 Kaggle 榜位快照（Table 2 为 2024-02 时点）；未读参考文献原文（MLE-Bench、METR、Weco technical report 均为二手转述）；未检查开源仓库 `WecoAI/aideml` 的实现是否与论文一致。

## 局限与阅读提示

- **自建榜与第三方榜的证据强度不同**：Weco-Kaggle 的数字来自作者自己的切分与提交（单次运行、未报方差，p. 5–6）；MLE-Bench 与 RE-Bench 的数字来自第三方，但论文只是转述（p. 7、p. 9），引用时应回到 Chan et al. 与 METR。
- **对照口径不完全同模型**：Lite 上 AIDE 与「Human with ChatGPT」「AutoGPT」同为 GPT-4 Turbo，可比性较好；但 H2O AutoML 是传统流水线，预算（80/20 切分 + 600 秒搜索）与 AIDE 的迭代预算不同（p. 6、p. 12）。MLE-Bench 表里的「4 倍」比较是跨模型的（见「分歧」）。
- **harness 超参未公开**：初始解数量、调试深度上限、总迭代步数、每节点评测时限、温度等均无量值；只能从 Fig 6 的 18 步与「24 小时 timeframe」反推规模（p. 8，Fig 6）。
- **无消融**：论文没有把「树搜索 vs 上下文累积」「原子改动 vs 整段重写」「Σ(T) vs 全量历史」拆开做对照，机制优势主要靠与其它脚手架的整体比较支撑（p. 7–8）。
- **污染的处置是承认而非排除**：作者称未发现成绩与竞赛新近度相关，并指出「唯一能完全确保无污染的办法是把解提交到仍在进行的竞赛」（p. 7）。
- **适用范围窄**：作者自述为表格类 MLE 设计，泛化证据来自第三方任务（NAS、Triton kernel、GPT-2 微调等），而长程、大代码库任务上明确表现不佳（p. 9–10）。

## 分歧与内部不一致

1. **「4 倍」是跨模型比较**。p. 9（`raw/extracted/2502.13138v1.md:435-451`）写「AIDE with o1-preview earned medals in 16.9%… nearly four times that of the follow-up agent OpenHands」，而 Table 3 中 OpenHands 的数字是 **GPT-4o** 的 4.4%（16.9 / 4.4 ≈ 3.8）。同页 Table 3 给出的同模型比较是 AIDE+GPT-4o 8.7% vs OpenHands+GPT-4o 4.4%，约 2 倍；引言（p. 2）也正是这么写的：「twice the number of medals … when both use GPT-4o」。**同一篇论文里两种说法并存**，引用时应写清是否同模型。
2. **Weco-Kaggle 规模：63 还是 80？** p. 5 正文称 63 个竞赛（`raw/extracted/2502.13138v1.md:265`），但附录 C（p. 15–17）逐行统计为 80 行，且每行均带 Data Size 字段。可能是正文数字未随附录更新，也可能附录含额外任务；论文未解释。
3. **RE-Bench 的时间口径**：正文说 AIDE 在「六小时时限内」超过人类（p. 9），图 4 横轴到 8 小时，Kernel 任务又说人类专家有「64 小时」。三个时间数字各自指的是什么（agent 预算 / 作图范围 / 人类预算）论文未逐一说明，需回到 METR 原文。
4. **与 MLE-Bench 原文的口径差**：本库另有 MLE-Bench 剪藏（`raw/articles/MLE-Bench.md`，尚未按 PDF 流程入库），其中 16.9 ± 1.1% / 9.4% 与本页 Table 3 一致；但剪藏未列有效提交率，无法交叉核对 82.8% 一栏。

## 来源位置索引

| 主题 | 位置 |
| --- | --- |
| 元数据、摘要、作者与代码链接 | p. 1，`raw/extracted/2502.13138v1.md:8-68` |
| 引言：AutoML/NAS 的搜索空间问题、代码空间搜索 | p. 1–2，`:47-68` |
| 与 ReAct/POMDP 的对比 | p. 2，`:100-115` |
| 形式化：式 (1) 优化问题 | p. 2，`:104-113` |
| 算法 1 与解树定义 | p. 3，`:130-165` |
| 3.1 解空间探索：s、T、π、Σ、f | p. 3，`:145-165` |
| 3.2 搜索策略 π（draft/debug/improve） | p. 3，`:167-190` |
| 编码算子三入口（含「原子改动」） | p. 3–4，`:178-193` |
| Σ(T) 摘要算子 | p. 4，`:219-229` |
| 数据预览 | p. 4，`:231-243` |
| 图 1 解树示例 | p. 4，`:196-214` |
| Weco-Kaggle 构成与评测协议 | p. 5，`:255-317` |
| Table 1 与基线设定 | p. 5，`:255-317` |
| Table 2 逐竞赛明细 | p. 6，`:318-383` |
| 完整 Weco-Kaggle 结果 + Fig 2 | p. 6–7，`:370-410` |
| 潜在局限（切分差异、污染） | p. 7，`:427-434` |
| Table 3 MLE-Bench 结果 | p. 7，`:412-425` |
| MLE-Bench 机制解释与 pass@k | p. 8，`:484-509` |
| Fig 3 MLE-Bench Lite | p. 8，`:453-483` |
| RE-Bench 结果与失败模式 | p. 9，`:510-539` |
| Fig 4 RE-Bench 曲线 | p. 9，`:523-539` |
| 相关工作、结论 | p. 9–10，`:540-622` |
| 附录 A 基线细节（H2O/AutoGPT/人类） | p. 12–13，`:681-742` |
| 附录 B 代码复杂度与成本 | p. 13，`:743-800`、Fig 6–7 p. 15，`:878-948` |
| 附录 C 竞赛清单 | p. 13、p. 15–17，`:801-1033` |

## 关联页面

- 工具：[[aide-scaffold]]（脚手架页）
- 概念：[[code-space-search]]（代码空间搜索与解树）
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 待验证问题：[[automated-research-open-questions]]
- 对照阅读：[[hyra-1-0]]、[[hyra-2-0]]（同为「提出—评估—迭代」范式，但组织方式与验证制度不同）
- 相关但**尚未入库**的原始来源（仅存在于 `raw/articles/`）：`raw/articles/MLE-Bench.md`、`raw/articles/R&D-Agent.md`、`raw/articles/MLE-STAR.md`、`raw/articles/AlphaEvolve.md`
