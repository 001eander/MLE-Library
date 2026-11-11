---
title: autoresearch 仓库：单 GPU 上的自主训练实验基线
type: tool
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - research-agent
  - agent-harness
  - hyperparameter-search
  - nanochat
sources:
  - "[[autoresearch]]"
status: draft
verified: 2026-11-11
applies_to:
  - karpathy/autoresearch（README 剪藏 2026-03-07，未固定 commit）
  - 单块 NVIDIA GPU、Python 3.10+、uv
---

# autoresearch 仓库：单 GPU 上的自主训练实验基线

`karpathy/autoresearch` 是一个把「改训练脚本 → 跑 5 分钟 → 保留或丢弃」交给编码智能体的小型研究环境，训练代码是 nanochat 的简化单 GPU 实现（[[autoresearch]], L19）。本页只覆盖剪藏 README 所描述的仓库契约与用法；**仓库当前实现未核查**，也没有固定 commit。

## 契约（来源论述）

| 文件 | 内容 | 谁改 | 位置 |
| --- | --- | --- | --- |
| `prepare.py` | 固定常量、一次性数据准备（下载数据 + 训练 BPE 分词器）、运行时工具（数据加载器、评估） | 不改 | `raw/articles/AutoResearch.md:25`、`:66` |
| `train.py` | GPT 模型、优化器（Muon + AdamW）、训练循环 | **智能体** | `:26`、`:67` |
| `program.md` | 面向单个智能体的指令 | **人类** | `:27`、`:68` |
| `pyproject.toml` | 依赖 | — | `:69` |

- 时间预算：**每次 5 分钟墙钟**，不含启动与编译（`:29`）；吞吐约每小时 12 次实验（`:75`）。
- 指标：`val_bpb`（验证字节级比特数），越低越好（`:29`）。
- 实现依赖：除 PyTorch 与少数小包外无外部依赖，无分布式训练（`:76`）；已测试 H100（`:35`）。

## 快速开始（L37–L51）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 安装 uv（若未装）
uv sync                                          # 安装依赖
uv run prepare.py                                # 一次性：下载数据 + 训练分词器（约 2 分钟）
uv run train.py                                  # 手动跑一次实验（约 5 分钟）
```

四步都通过后再交给智能体：在仓库内启动编码工具并给出提示（`:55–L59`）：

```text
请先查看 program.md，然后开始一项新实验！先完成环境配置。
```

## 在小机器上降规模的参数表（L84–L90）

README 给想从很小的平台（含 MacBook，需用社区分支）起步的人 7 条建议。整理成「改哪里 / 改什么」：

| # | 位置 | 参数 | 建议 | 理由（来源给出的） |
| --- | --- | --- | --- | --- |
| 1 | 数据集 | — | 换熵更低的数据集（TinyStories，GPT-4 生成的短篇故事） | 数据范围窄，小模型也能出合理结果 |
| 2 | `prepare.py`（推断：分词器在此训练，具体常量位置未明） | `vocab_size` | 8192 → 4096 / 2048 / 1024，或字节级（256 种 UTF-8 字节） | 词表越小，同参数量下模型越「够用」 |
| 3 | `prepare.py` | `MAX_SEQ_LEN` | 大幅下调，性能差的机器可到 256；同时略增 `train.py` 的 `DEVICE_BATCH_SIZE` | 前向/反向的 token 数＝两者乘积 |
| 4 | `prepare.py` | `EVAL_TOKENS` | 减小 | 用更少数据评估验证损失 |
| 5 | `train.py` | `DEPTH` | 默认 8 → 约 4 | 许多变量是 `DEPTH` 的函数 |
| 6 | `train.py` | `WINDOW_PATTERN` | `"SSSL"` → `"L"` | `"SSSL"` 是交替带状注意力，在小设备上可能很低效 |
| 7 | `train.py` | `TOTAL_BATCH_SIZE` | 大幅下调但保持 2 的幂，例如 `2**14`（≈16K） | 来源称「具体数值很难确定」 |

**注意（综合）**：第 2–4 项落在 `prepare.py`（第 2 项为推断，README 只列了 3、4 项的所在文件），而该文件标注「不要修改」（`:25`）。因此降规模是**人类重写基线**，不是智能体可做的动作；重写后不同机器上的「第 N 次实验」已不在同一评估口径上。第 1 项还改变了数据分布，与第 5–7 项叠加后无法归因单项贡献。

## 作为基线怎么用（综合）

1. **当作「最小可行自主研究」的参照实现**：单文件可改、评估代码禁改、单一标量指标——比 AIDE（[[aide-scaffold]]）或 Hyra（[[hyra]]）便宜得多，也更容易读懂每一次保留/丢弃。
2. **先跑通人工基线再放 agent**：`uv run train.py` 一次得到的 `val_bpb` 是噪声参照；建议重复 3–5 次记录种子波动，再让 agent 以「超过波动范围」为保留阈值（对照图内 `random seed 42→137` 被保留，[[autoresearch]]「图表核查」）。
3. **要跨平台比较就别用固定墙钟**：若你的目标是「A 配置比 B 配置好」，把 5 分钟换成固定 token 数或固定优化步数，否则慢机器上的结论不可迁移（来源自述跨平台不可比，`:75`）。
4. **保留实验日志与 diff**：README 只说「早上会看到一份实验日志」（`:19`），未规定格式；至少要能在丢弃后复算，否则「83 次实验 15 次保留」无法审计。
5. **别在生产机器上禁用权限**：README 建议禁用所有权限并让 agent 自由改代码、反复执行（`:55`）。个人机器可接受，共享集群应加沙箱、成本上限与人工审批（对照 [[hyra-2-0]] 的审批门）。

## 使用边界

- **适用**：单卡、有明确标量指标（验证损失）、单次实验分钟级的训练 recipe 搜索；想低成本观察「agent 能否真的改出更好的训练配置」。
- **慎用**：需要跨卡/分布式训练、需要多文件重构、或指标需要长时间训练才稳定（5 分钟预算太短）的任务。
- **不可用于**：跨平台横向比较结果（来源自述）；把「本平台最优」当作通用结论。
- **安全**：默认假设 agent 有完整写权限与执行权限；不提供沙箱、成本上限或审批机制。

## 版本与时效

- `verified: 2026-11-11`；`applies_to: karpathy/autoresearch（README 剪藏 2026-03-07，未固定 commit）`。
- 本页与剪藏一致，**不代表仓库当前行为**：默认 `DEPTH`、`WINDOW_PATTERN`、词表大小等数值可能已变；引用具体超参前应核对仓库当前代码。
- README 自称仍在讨论是否支持 CPU/MPS（`:80`），因此「仅 NVIDIA」是 2026-03 快照的约束。
- 图 `progress.png` 属仓库内可变文件，本库副本抓取于 2026-11-11（`raw/assets/autoresearch/progress.png`，SHA-256 `a6be2ad8…`）。

## 待解问题

- 仓库当前的默认超参与 README 描述是否一致？`train.py` 是否真的包含完整的 Muon + AdamW 实现？
- 是否内置实验日志（格式、位置）、是否自动 git 提交每次实验？README 未说明。
- 是否支持多卡或 resume（中断续跑）？「5 分钟硬截断」如何与 checkpoint 结合？
- 社区分支（MacBook / CPU / MPS）的维护状态与质量如何？

## 关联页面

- 来源：[[autoresearch]]
- 实践：[[research-agent-harness]]、[[evaluator-design-and-reward-hacking]]
- 概念：[[code-space-search]]、[[recursive-self-improvement]]
- 其它工具：[[aide-scaffold]]、[[hyra]]
- 待验证问题：[[automated-research-open-questions]]
