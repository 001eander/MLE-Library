---
title: "AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms"
source: "https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/"
author:
  - "[[AlphaEvolve team]]"
published: 2025-05-14
created: 2026-11-10
description: "Google DeepMind 的进化式编码智能体：把 Gemini 的创造力与自动评估器结合，用于通用算法的发现与优化。"
tags:
  - "clippings"
---

# AlphaEvolve：由 Gemini 驱动的算法设计编码智能体

> 新型 AI 智能体将大语言模型的创造力与自动化评估器相结合，为数学及计算领域的实际应用演化算法。

大语言模型（LLM）具有非凡的多功能性。它们可以总结文档、生成代码，甚至头脑风暴新想法。如今，我们进一步拓展了这些能力，以应对数学和现代计算领域中基础且高度复杂的问题。

今天我们宣布推出 [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)——一个由大语言模型驱动、用于通用算法发现与优化的进化式编码智能体。AlphaEvolve 将 Gemini 模型的创造性问题解决能力与可验证答案的自动评估器相结合，并采用进化框架来改进最有前景的想法。

AlphaEvolve 提升了 Google 数据中心、芯片设计和 AI 训练流程的效率——包括训练 AlphaEvolve 自身所依托的大语言模型。它还帮助设计了更快的矩阵乘法算法，并为悬而未决的数学问题找到了新的解法，在众多领域的应用上展现出巨大前景。

## 用大语言模型设计更好的算法

2023 年，我们首次证明，大语言模型可以生成以计算机代码编写的函数，帮助在一个悬而未决的科学问题上[发现新的、可证明正确的知识](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)。AlphaEvolve 是一款能够超越单一函数发现、对整个代码库进行演化并开发复杂得多算法的智能体。

AlphaEvolve 利用一组最先进的大语言模型：速度最快、效率最高的 Gemini Flash 负责最大化所探索思路的广度，最强大的 Gemini Pro 则通过富有洞见的建议提供关键的深度。这些模型共同提出以代码形式实现算法解决方案的计算机程序。

![流程图：提示采样器先为语言模型组装提示，模型据此生成新程序；程序由评估器打分并存入程序数据库，该数据库实现了一种进化算法，决定哪些程序将用于未来的提示。](https://lh3.googleusercontent.com/mUd0dQneyWkX6ohzKGk0dE4-vJaSgvgYGPFjWC2krhVWILtIiMFkSxz8OWA3ug17ZzUd61rKPuHFWafiuVJ2j9IVFzHSlklrd5ykNk3t_AZno9gXBpU=w2880-rw-lo)

AlphaEvolve 使用自动化评估指标对所提出的程序进行验证、运行和评分。这些指标为每个解决方案的准确性和质量提供了客观、可量化的评估。这使得 AlphaEvolve 在能够清晰、系统化地衡量进展的广泛领域中尤为有用，例如数学和计算机科学。

### 优化我们的计算生态

在过去一年里，我们已将 AlphaEvolve 发现的算法部署到 Google 的计算生态系统中，包括我们的数据中心、硬件和软件。每一项改进的影响都会在我们的 AI 和计算基础设施中成倍放大，从而为所有用户构建一个更强大、更可持续的数字生态系统。

### 改进数据中心调度

AlphaEvolve 发现了一种简单却极其有效的启发式方法，帮助 [Borg](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/) 更高效地编排 Google 庞大的数据中心。该解决方案现已投入生产使用一年有余，持续平均回收 Google 全球计算资源的 0.7%。这种持续的效率提升意味着，在任何给定时刻，相同的计算资源占用下都能完成更多任务。AlphaEvolve 的解决方案不仅能带来强劲性能，还具备人类可读代码的重要运营优势：可解释性、可调试性、可预测性和易于部署。

### 协助硬件设计

AlphaEvolve 提出了一种 Verilog 重写方案，去除了用于矩阵乘法的某个关键且经过高度优化的算术电路中的不必要位。至关重要的是，该方案必须通过稳健的验证方法，以确认修改后的电路保持功能正确性。该方案已被集成到 Google 即将推出的张量处理单元（TPU）中，这是 Google 定制的 AI 加速器。通过使用芯片设计师的标准语言提出修改建议，AlphaEvolve 促进了 AI 与硬件工程师之间的协作方式，以加速未来专用芯片的设计。

### 提升 AI 训练与推理

AlphaEvolve is accelerating AI performance and research velocity. By finding smarter ways to divide a large matrix multiplication operation into more manageable subproblems, it sped up this vital kernel in Gemini's architecture by 23%, leading to a 1% reduction in Gemini's training time. Because developing generative AI models requires substantial computing resources, every efficiency gained translates to considerable savings. Beyond performance gains, AlphaEvolve significantly reduces the engineering time required for kernel optimization, from weeks of expert effort to days of automated experiments, allowing researchers to innovate faster.

AlphaEvolve 正在加速 AI 的性能与研究迭代速度。它找到了把大矩阵乘法拆分为更易处理子问题的更聪明方式，使 Gemini 架构中这一关键内核提速 23%，从而让 Gemini 的训练时间减少 1%。由于开发生成式 AI 模型需要大量算力，每一分效率提升都意味着可观的节省。除了性能收益，AlphaEvolve 还把内核优化所需的工程时间从数周的专家投入压缩到数天的自动化实验，让研究者能更快迭代。

AlphaEvolve 还可以优化底层 GPU 指令。这一极其复杂的领域通常已由编译器进行了高度优化，因此人类工程师通常不会直接修改它。在基于 Transformer 的 AI 模型中，AlphaEvolve 使 FlashAttention 内核实现最高提速 32.5%。此类优化有助于专家定位性能瓶颈，并轻松将改进纳入其代码库，从而提高生产力，并在未来节省计算资源和能源。

### 推进数学与算法发现的前沿

AlphaEvolve 还能为复杂的数学问题提出新方法。只需为其提供计算机程序的最小代码骨架，AlphaEvolve 便能设计出一种新型基于梯度的优化流程的众多组件，该流程发现了矩阵乘法的多种新算法，而矩阵乘法是计算机科学中的一个基础问题。

*AlphaEvolve 为发现更快的矩阵乘法算法而提出的一系列更改。在此示例中，它在多个组件中提出了大量更改，包括优化器与权重初始化、损失函数以及超参数扫描。这些更改极不平凡，在进化过程中需要 15 次变异。*

AlphaEvolve 的流程找到了一种使用 48 次标量乘法来相乘 4×4 复值矩阵的算法，改进了 Strassen 1969 年提出的算法——该算法此前被认为是该情形下的最佳算法。这一发现表明，相较于我们此前的工作 AlphaTensor 取得了重大进展；AlphaTensor 专门研究矩阵乘法算法，而对于 4×4 矩阵，它仅找到了针对二进制算术的改进。

为了考察 AlphaEvolve 的广度，我们将该系统应用于数学分析、几何学、组合数学和数论领域的 50 多个开放问题。该系统的灵活性使我们能够在数小时内设置大多数实验。据我们所知，在大约 75% 的情况下，它重新发现了最先进的解决方案。而在 20% 的情况下，它改进了此前已知的最佳解决方案，在相应的开放问题上取得了进展。例如，它推进了[亲吻数问题](https://en.wikipedia.org/wiki/Kissing_number)：这一几何挑战已令数学家着迷 300 多年，涉及与一个公共单位球相接触的互不重叠球体的最大数量。AlphaEvolve 发现了一种由 593 个外层球体构成的配置，并在 11 维中确立了一个新的下界。

### 未来之路

AlphaEvolve 展现了从针对特定领域发现算法，到为广泛的现实世界挑战开发更复杂算法的发展过程。我们预计，随着大语言模型能力的提升，AlphaEvolve 将持续进步，尤其是在它们更擅长编码之后。

我们与「人 + 人工智能研究」（People + AI Research）团队合作，一直在构建一个友好的用户界面，用于与 AlphaEvolve 交互。我们正在筹划一项面向特定学术用户的早期访问计划，同时也在探索让 AlphaEvolve 更广泛可用的可能性。

虽然 AlphaEvolve 目前正被应用于数学和计算领域，但其通用性意味着它可以应用于任何解决方案能够以算法形式描述并可自动验证的问题。我们相信，AlphaEvolve 有望在更多领域带来变革，例如材料科学、药物研发、可持续发展以及更广泛的技术和商业应用。

**致谢**：AlphaEvolve 由 Matej Balog、Alexander Novikov、Ngân Vũ、Marvin Eisenberger、Emilien Dupont、Po-Sen Huang、Adam Zsolt Wagner、Sergey Shirobokov、Borislav Kozlovskii、Francisco J. R. Ruiz、Abbas Mehrabian、M. Pawan Kumar、Abigail See、Swarat Chaudhuri、George Holland、Alex Davies、Sebastian Nowozin 和 Pushmeet Kohli 开发，是「用 AI 做算法发现」这一方向工作的一部分。更多细节见白皮书。
