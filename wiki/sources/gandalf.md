---
title: "GANDALF：用门控特征学习单元做表格特征选择（arXiv 2207.08548v6）"
type: source
created: 2026-11-12
updated: 2026-11-12
tags:
  - mle
  - tabular
  - deep-learning
  - gating
  - feature-selection
  - interpretability
  - benchmark
source_path: raw/papers/GANDALF.pdf
extracted_path: raw/extracted/GANDALF.md
source_kind: pdf
published: 2024
status: draft
---

# GANDALF：用门控特征学习单元做表格特征选择

## 元数据

| 项目 | 值 |
| --- | --- |
| 标题 | *GANDALF: Gated Adaptive Network for Deep Automated Learning of Features for Tabular Data* |
| 作者 | Manu Joseph（Walmart Global Tech，Bangalore）、Harsh Raj（Delhi Technological University，Dept. of Engineering Physics） |
| 发表 | 本地 PDF 为 **arXiv:2207.08548v6**，边注日期 **2024-01-10**（p. 1）；编号前缀对应 2022-07 的首次提交，本页按版本日期记 `published: 2024` |
| 本地文件 | `raw/papers/GANDALF.pdf`（25 页，679,823 B，letter 版式） |
| SHA-256 | `46fa3d94545665ca23a9dbffb20a9575999dcd2a05e6165fb8c471bb195b1ae0`（与提取文件 frontmatter 一致） |
| 抽取 | `raw/extracted/GANDALF.md`（1,354 行，`pdftotext -layout`，含 `## Page N` 共 25 段） |
| 代码 | 正文称 <https://github.com/manujosephv/pytorch_tabular>，MIT License（p. 1、p. 15）；**本库未核查该仓库的 commit、文件路径或实现细节** |
| 实验硬件 | 单卡 NVIDIA RTX 3060、20 核、16 GB RAM；batch 512，Higgs 用 4096 + 混合精度 + 子 epoch checkpoint（p. 8） |
| 主题 | 表格特征表示学习；GRU 式门控去时序化；t-softmax 稀疏特征掩码；掩码式解释性；两个公开基准上的横向比较 |

**来源定位**。这是一篇**提出单一新架构**的论文：核心贡献是 GFLU（Gated Feature Learning Unit）这个可替换 MLP 的表格处理单元，加上把 GRU 门控搬到非时序场景的适配（p. 2）。它的基准部分不是统一协议的重跑，而是「自己跑 GANDALF，其余模型沿用两个公开基准已发布的分数」，因此读数与 [[ft-transformer]] 那种同协议比较不可直接并列。

## 结构

| 页 | 内容 |
| --- | --- |
| 1 | 摘要、引言、三项贡献 |
| 2 | 贡献列表续；图 1（GRU vs GFLU） |
| 3 | 相关工作：表格 DL、特征表示学习、门控机制 |
| 4 | 记号；式 1–4（GRU 原文）；为何表格无时序；式 4 的印刷错误在此 |
| 5 | 逐阶段权重与阶段掩码的动机；式 5–6（softmax / t-softmax）；§3.1.1 特征选择 |
| 6 | 式 7–8（掩码与掩码后输入）；图 2（GFLU 详图）；§3.1.2 门控；式 9 |
| 7 | 式 10–12（更新门、候选、重置门）；单矩阵合并；§3.2 架构与初始化；式 13–14 |
| 8 | 式 15（Beta 初始化）；式 16（r-softmax 定 t）；§3.3 解释性（式 17）；§4 实验设置 |
| 9 | §4.1 协议；ADTM 定义（含回归方向的改写）；图 3（Tabular Benchmark） |
| 10 | 图 4（TabSurvey）；§4.1.1 子集四条规则；§4.2 TabSurvey |
| 11 | §4.3 结果综述 |
| 12 | §4.4 超参研究；搜索空间；图 5；13/18 的说法 |
| 13 | §4.5 解释性；§4.5.1 NDCG@K；图 6 |
| 14 | §4.5.2 消融；§5 结论（含顺序计算限制） |
| 15 | 结论续；代码地址 |
| 16 | 附录 A：表 1（18 个数据集）、表 2（TabSurvey 4 个数据集） |
| 17 | 表 3（Tabular Benchmark 测试分）、表 4（标准差） |
| 18 | 表 5（TabSurvey 21 个方法） |
| 19 | 图 7（调参收敛）；表 6（逐数据集调参结果） |
| 20 | 图 8（MoRF）；表 7（TabSurvey 调参结果） |
| 21 | 图 9（LeRF） |
| 22–25 | 参考文献 |

## 机制（照原文）

前向两段（p. 4）：输入 $X\in\mathbb R^d$ 先过一串 GFLU 得到表示 $H\in\mathbb R^{\tilde d}$，再过一个 $K$ 层 MLP 输出（式 13；回归时加训练集均值 $T_0$，式 14）。实验中 MLP 头固定为两个隐藏层 32/16、ReLU（p. 11），$\tilde d$ 保持与 $d$ 相同（p. 5）。

GFLU 内部（式 9–12，p. 6–7）：

$$
H_n=(1-z_n)\odot H_{n-1}+z_n\odot\tilde H_n,\qquad
z_n=\sigma\big(W_n^z\cdot[H_{n-1};X_n]\big),
$$
$$
\tilde H_n=\tanh\big(W_n^o\cdot[r_n\odot H_{n-1};X]\big),\qquad
r_n=\sigma\big(W_n^r\cdot[H_{n-1};X_n]\big).
$$

三处与 GRU 的差别，原文都点明了：

1. **时间步换成阶段**，$W_n^z,W_n^r,W_n^o$ 逐阶段不同（p. 4–5，图 1）；
2. **两个门吃掩码后的 $X_n$，候选吃全特征 $X$**（p. 6），理由是「avoid the induced sparsity in the feature masks from impeding gradient flow」；
3. 实践中可用单个 $W_n^i\in\mathbb R^{2d\times 2d}$ 同时算两个门（p. 7）。

掩码本身（式 7–8，p. 6）：$M_n=\text{t-softmax}(F_n,t)$，$X_n=M_n\odot X$；$F_n\in\mathbb R^d$ 是逐阶段的可学参数。稀疏激活的来由是 t-softmax 的权重 $w_t=\mathrm{ReLU}(a_i+t-\max(a))$（式 6，p. 5），它「does not suffer from computational inefficiencies like other sparse activations (α-entmax and sparsemax) which require a sorting operation」。

初始化（式 15–16，p. 7–8）：$F_n\sim\text{Beta}(\alpha_n,\beta_n)$，$\alpha_n,\beta_n\sim U(0.5,10)$；`t` 的初值用 r-softmax 反推，$t=-\text{quantile}(x,r)+\max(x)$，因为分位数操作太慢而不在训练中使用。

解释性（式 17，p. 8）：$I=\sum_{n=1}^N M_n$，可归一化为 $I^{norm}_i=I_i/\sum_i I_i$，性质上类比 GBDT 的特征重要性。

机制细节单独整理在 [[gated-feature-learning-unit]]。

## 实验协议

- **基准**：Tabular Benchmark（Grinsztajn et al., 2022）与 TabSurvey（Borisov et al., 2022）；「we refrain from re-running models already tested by benchmarks」，只把 GANDALF 放进同一套划分、预处理、折与指标（p. 9）。
- **调参**：Optuna + Tree Parzen Estimator，100 trials，前 10 trials 纯随机（p. 9、p. 12）。搜索空间：`GFLU Stages ∈ {2,…,30}`、`GFLU Dropout ~ U(0,0.5)`、`Init Sparsity ~ U(0,0.9)`、`Weight Decay ∈ {1e-3…1e-8}`、`Learning Rate ∈ {1e-3…1e-6}`（p. 12）。
- **聚合**：ADTM，$s$ 在数据集内做 min-max 缩放；TabSurvey 的回归指标是 MSE，故方向改为 $\frac{s-\max(S)}{\max(S)-\min(S)}$（p. 9）。
- **报告口径**：Tabular Benchmark 报 test split 平均分，TabSurvey 报 validation split；多折才报标准差（p. 9）。
- **成本**：用 Thop 统计参数量与 MACs，脚注注明 1 MAC ≈ 2 FLOPs（p. 9）。
- **子集**：Tabular Benchmark 的 45 个数据集中选 18 个，四条规则为仅中等规模、无 target/data 变换、特征数 > 20、同时出现在数值与类别基准时取类别版（p. 10）；TabSurvey 排除 Heloc，用 4 个数据集（p. 10）。

## 关键主张

1. **两个基准上的聚合名次不同**。Tabular Benchmark 的 ADTM 平均分：GANDALF 0.829、XGBoost 0.577、GBDT 0.476、HistGBDT 0.462、RandomForest 0.433、SAINT 0.384、FT Transformer 0.370、Resnet 0.218、MLP 0.147（图 3，p. 9）。TabSurvey 的 ADTM：XGBoost 0.957、CatBoost 0.947、**GANDALF 0.945**、LightGBM 0.925、SAINT 0.915……（图 4，p. 10）。
2. **按表 3 逐列计数，逐数据集最优为 13/18**（p. 17）。落败的 5 个是：topo_2_1（4.85，XGBoost 6.94）、MiniBooNE（92.42，XGBoost 93.72）、heloc（72.26，FT Transformer 72.67）、cpu_act（98.29，XGBoost 98.62）、yprop_4_1（8.53，RandomForest 9.39）。注意 p. 12 的「13 of 18 datasets as early as 5 trials」说的是调参收敛，不是最终逐列计数。
3. **TabSurvey 上两胜两负**：Covertype 97.7 最高（XGBoost 97.3）、Cal. Housing MSE 0.16±0.006 最低（CatBoost 0.196、XGBoost 0.206）；Adult 86.7 低于 XGBoost 87.3 与 LightGBM 87.4，HIGGS 76.9 低于 SAINT 79.8（表 5，p. 18）。
4. **参数与计算量低**：图 3 下方两张箱线图中，GANDALF 的 MACs 与参数量的量级都低于 FT Transformer/ResNet（p. 9；本页只做量级描述，原文未给逐数据集数值表）。
5. **调参很快饱和**：图 7 中 GANDALF 在第 4 个 trial 后即到 14 个数据集最优的平台（p. 19）；正文与图注写的是「13 of 18 datasets in 5 trials」（p. 12、p. 19）。
6. **超参重要度**：FANOVA 给出 GFLU Stages 影响最大，Init Sparsity 与 GFLU Dropout 次之（图 5a，p. 12）；作者推荐的搜索区间是 sparsity 0.0–0.5、dropout 0–0.05，而 stages「doesn't show any particular pattern, but is having the highest impact」（p. 12–13）。
7. **解释性相当或更好**：以 GradientSHAP/DeepLIFT 的 top 25 特征为 ground truth，GANDALF 的 NDCG@K 在 $K\in\{1,5,10,15,25,50\}$ 上不低于 XGBoost（图 6a/6b，p. 13）。目视读到的数值见下面「图表核查」。
8. **消融保真度**：把最相关特征按顺序替换成高斯噪声时分数下降更快，按最不相关顺序替换时下降更慢，因此「further shows the effectiveness of the gating mechanism」（p. 14）。
9. **自述限制**：阶段的顺序计算使其无法并行（p. 14）；优势可能更多出现在特征数较多、特征与目标关系不直接的数据上（p. 14）。

## 图表核查

渲染方式：`pdftoppm -r 130` 渲染 p. 2、4、5、6、7、9、10、12、13、19、20、21，另对 p. 4、p. 5、p. 13 做 300 dpi 裁切；渲染图只留在 `.cache/pdf/46fa3d94/`。

- **图 1（p. 2，GRU vs GFLU）**：左panel 是 GRU，$x_1..x_t$ 共享权重、输出 $o_1..o_t$；右panel 是 GFLU，$x$ 经 `Learnable Feature Mask $M_n$` 后进各阶段，阶段权重标注 `Different weights across stages`，最后接一个 Feed Forward Network 输出 $\hat y_n$。原文对「共享 vs 逐阶段」的对比就画在这一张图上。
- **图 2（p. 6，GFLU 详图）**：可见 $H_{n-1}$ 与掩码后的 $X_n$ 拼接入 $W_n^z,\sigma$ 与 $W_n^r,\sigma$，候选支路的 Concat 接的是**全特征流**（图例区分紫色 `Full Feature Flow` 与橄榄色 `Masked Feature Flow`），最后按 $1-z_n$ 与 $z_n$ 相加得到 $H_n$。这张图是「掩码只进两个门」的直接证据。
- **图 3（p. 9）**：上图为 9 个模型的 ADTM 箱线图，GANDALF 的箱体最窄且贴 0.85 附近、有低值离群点；左下 MACs、右下参数量均为对数轴，GANDALF 箱体在两者中都最低一档。
- **图 4（p. 10）**：横轴顺序为 XGBoost、CatBoost、**GANDALF**、LightGBM、SAINT、DeepFM、Net-DNF、NODE、TabNet、MLP、VIME、STG、Random Forest、Model Trees、Decision Tree，共 15 个；顶部数值 0.957/0.947/0.945/… 与标签的对应关系在提取文本里被打散，目视才能对齐。
- **图 5（p. 12）**：a 为三个超参的 FANOVA 重要度箱线图，GFLU Stages 中位数最高（约 0.45），Init Sparsity 与 GFLU Dropout 约 0.28–0.30；b 的 stages 直方图在 2–6 与 18–21 各有一簇；c 的 dropout 直方图集中在 0–0.10；d 的 sparsity 直方图绿色（高分）集中在 0.2–0.45 与 0.8–0.9 的一段，0.85 附近有红色（低分）柱。
- **图 6（p. 13）**：a、b 两组柱状图目视读数——GradientSHAP 为 ground truth：$K=1,5,10,15,25,50$ 时 GANDALF 0.88/0.81/0.82/0.82/0.79/0.92，XGBoost 0.62/0.76/0.80/0.78/0.78/0.90；DeepLIFT 为 ground truth：$K\leq15$ 目视读得 GANDALF 0.62/0.71/0.72/0.75，XGBoost 0.62/0.63/0.69/0.70，$K=25,50$ 的两组值（0.76/0.73、0.89/0.87）取自提取文本。下排两张折线图的纵轴以 1.0 起，两条曲线在 $N=14$ 附近都降到 0 附近；两个子图标题都被排成「c.」。
- **图 7（p. 19）**：GANDALF 曲线在第 4 个 trial 升到 14 后基本平在 13–14 之间（第 6–10 个 trial 为 13，第 11 个回到 14，第 18 个之后为 14）；其余模型全程 0–2。图注写「13 of 18 datasets in 5 trials」，而第 5 个 trial 的读数是 14。
- **图 8（p. 20，MoRF）**：8 个数据集面板，GANDALF 从 1.0 平滑下降至 0.1–0.2；Higgs、MiniBooNE、jannis、pol 面板可见灰色 XGBoost 曲线，Ailerons、cpu_act、heloc 面板只有红色曲线，yprop_4_1 的灰线贴在 0。图注未说明纵轴的归一化基准。
- **图 9（p. 21，LeRF）**：多数面板在 $N$ 较小时保持接近 1.0，然后在 $N=10$–14 一起掉到 0；jannis 面板的 GANDALF 曲线在 $N\approx2$ 就掉到 0.1 附近并保持，XGBoost 灰线则在中段反复上升到 0.95；yprop_4_1 的灰线不可见。

## 关键数字的来源位置

| 内容 | 位置 |
| --- | --- |
| GRU 式 1–4（含式 4 印刷错误） | p. 4 |
| t-softmax 式 5–6 | p. 5 |
| 掩码式 7–8 | p. 6 |
| 门控式 9–12、单矩阵合并 | p. 6–7 |
| 输出式 13–14、$T_0$ | p. 7 |
| Beta 初始化式 15 | p. 8 |
| r-softmax 式 16、`t` 初值 | p. 8 |
| 重要性聚合式 17 | p. 8 |
| 硬件、batch、混合精度 | p. 8 |
| ADTM 定义与回归方向改写 | p. 9 |
| 100 trials、前 10 随机、Thop | p. 9 |
| 18 数据集子集四条规则 | p. 10 |
| TabSurvey 排除 Heloc | p. 10 |
| MLP 头 32/16 | p. 11 |
| 搜索空间、13/18、FANOVA | p. 12 |
| NDCG@K 与 top 25 | p. 13 |
| MoRF/LeRF 与高斯噪声替换 | p. 14 |
| 顺序计算限制 | p. 14 |
| 表 1（18 数据集）、表 2（TabSurvey） | p. 16 |
| 表 3（测试分）、表 4（标准差） | p. 17 |
| 表 5（21 个方法） | p. 18 |
| 图 7、表 6（逐数据集调参） | p. 19 |
| 图 8、表 7 | p. 20 |
| 图 9 | p. 21 |

## 分歧与口径

以下均按原文表述记录，不替换、不推断。

1. **TabSurvey 的数据集数**：§4.2 写「We also used a smaller benchmark, with just 5 datasets」，同段又说「We excluded Heloc dataset」并「aggregate scores across the four datasets」；表 2 与表 5 都只有 4 个数据集（p. 10、p. 16、p. 18）。
2. **TabSurvey 的模型数**：§4.2 写「The benchmark covers 20 models, 12 from Deep Learning」，表 5 列 21 行（12 个深度学习方法加 GANDALF 与 8 个非深度方法），图 4 只画 15 个，原文未说明图 4 的筛选规则（p. 10、p. 18）。
3. **回归指标的尺度**：表 3 与表 4 的标题都说「for regression R2 score is reported」，但表 3 出现大于 1 的值（topo_2_1 的 4.85、yprop_4_1 的 8.53），表 4 的标准差与表 3 的分数量级也不一致（Ailerons 分数 86.46 而标准差 1.12E-03；Mercedes 的 XGBoost 标准差 8.41E-01 而分数 57.87）。原文未给单位或尺度说明（p. 17）。
4. **超参的个数与归属**：§4.4 开头写「requires only three crucial hyperparameters—# of GFLU Stages, Dropout in each GFLU, and Init Sparsity」，其后列出 5 个搜索维度（含 Learning Rate、Weight Decay）；同段又说「the only architecture specific hyperparameter that was tuned was the # of GFLU Stages and GFLU Dropout」，而 Init Sparsity 同样只存在于本架构（p. 11–12）。
5. **Init Sparsity 的推荐区间**：正文结论是「a sparsity between 0.0 and 0.5 is a good range to search」（p. 12），表 6 中被选中的值有 5 个在该区间之外（default-credit-card 0.870769、Ailerons 0.880892、topo_2_1 0.817548、eye movements 0.802989、Allstate 0.609144）（p. 19）。
6. **式 4 的印刷错误**：原文印为 $H_t=(1-z_t)\cdot h_{t-1}+z_t\cdot H_t$，左端应为 $\tilde H_t$、右端第二项应为 $H_{t-1}$，与后文式 9 的写法不一致（p. 4、p. 6）。
7. **NDCG 的 ground truth 与特征数**：§4.5.1 把「the top 25 features according to GradientSHAP and DeepLIFT」当作 ground truth，并在 $K\in\{1,5,10,15,25,50\}$ 上算 NDCG@K；但表 1 中 default-credit-card（21）、cpu_act（21）、heloc（22）、eye movements（23）、Higgs（24）的特征数小于 25，原文未说明这些数据集是否被排除（p. 13、p. 16）。
8. **消融结论与图**：正文说 MoRF 时「performance drops sharply」、LeRF 时「we should not expect huge drops in performance」（p. 14），但图 6c/6d 中两条曲线在 $N=14$ 附近都降到 0；图 9 的 jannis 面板中 GANDALF 在 LeRF 的 $N\approx2$ 就掉到 0.1 附近，而 XGBoost 灰线在中段上升到 0.95（p. 13、p. 21）。
9. **交叉引用错位**：图 6 的两个下排子图标题都排成「c.」，正文对应句写「Fig. 6b and Fig. 6c plots show the MoRF and LeRF curves」（p. 13–14）；图 7 图注写「13 of 18 datasets in 5 trials」，而该图的第 5 个 trial 读数为 14（p. 19）。
10. **权重为正的说法与式 6 不一致**：原文写「We can see that all the weights, $w_i$, are positive and at least 1 would be non-zero」（p. 5），但式 6 用 $\mathrm{ReLU}(a_i+t-\max(a))$，只有 $a_i>\max(a)-t$ 的项为正，其余恰为 0——而置零正是 t-softmax 能当稀疏掩码用的前提。

## 局限

- 基线不重跑：跨模型比较依赖两个基准原有实现的分数，GANDALF 一侧的调参预算（100 trials BO）与各基线在基准中的预算未必等同（p. 9、p. 12）。
- 无显著性检验；18 个数据集中多数只有 1 折（表 1 的 `# Folds` 列），单个数据集上的分数差没有可比较的不确定度（p. 16–17）。
- 数据集为 45 个中的 18 个子集，筛选规则偏向中等规模、特征数 > 20、无变换，结论不能外推到大规模、强变换或高基数类别特征的数据（p. 10）。
- 解释性实验的 ground truth 是两个事后归因方法的 top 25，本身不是「真实重要性」；被比较的 XGBoost 一边也未说明调参状态（p. 13）。
- 消融只扰动前 14 个特征（图 6c/6d、图 8、图 9 的横轴到 14），与 21–419 个特征的数据集不对应，原文未解释这个上限。
- 未给默认配置：只有搜索空间与逐数据集调参结果（表 6/7），没有单一「默认 GANDALF」可以直接开箱。
- 代码只给仓库地址，本库未核对其与论文式 10–12、式 15–16 的一致性。

## 待解问题

1. `pytorch_tabular` 中 GFLU 的实现是否与式 9–12 一致（掩码是否真的只进两个门、候选是否真的用全特征）？本库本轮未核查。
2. `Init Sparsity` 到 `t` 初值的映射在代码里怎么实现？是否等价于式 16 的分位数反推？
3. 掩码只进两个门这一设计，作者称「empirical evidence supports its superior performance」，但论文没有给出这组对照的数字。差异有多大？
4. `GFLU Dropout` 作用在阶段的哪个位置（掩码、隐藏表示还是门）？原文只说「applied to each GFLU stage and hence compounding the effect」。
5. ADTM 在 Tabular Benchmark 上 0.829 与 TabSurvey 上 0.945 的差别，是基准难度、数据集数量还是基线集合不同造成的？两个基准的模型集合并不相同。
6. 若把两个基准的所有模型都在同一协议下重跑并调参，GANDALF 的 13/18 与 0.829 是否仍然成立？
7. 顺序计算（$N$ 可达 30）在长特征表上的实际 wall-clock 与 GBDT 相比如何？论文只给参数量与 MACs，没给训练/推理时间。
8. 图 8/图 9 的纵轴归一化基准是什么？未知时无法把曲线斜率读成「对少数特征的依赖程度」。
9. TabSurvey 上 GANDALF 的 Cal. Housing MSE 0.16 明显好于其它方法（次优 0.196），这个数据集只有 8 个特征且被 2 折评估，是否稳定？
10. 与 [[ft-transformer]] 的结论如何对齐：后者在与 GANDALF 部分重叠的数据集（covertype、Higgs 等）上给出的是另一套数字与另一套协议。

## 关联页面

- 机制：[[gated-feature-learning-unit]]
- 概念：[[tabular-deep-learning]]、[[feature-selection]]、[[multi-layer-perceptron]]、[[regularization-and-sparsity]]
- 对照：[[gbdt-vs-tabular-dl]]、[[ft-transformer]]
- 实践：[[tabular-model-selection]]、[[hyperparameter-search]]、[[cross-validation]]
