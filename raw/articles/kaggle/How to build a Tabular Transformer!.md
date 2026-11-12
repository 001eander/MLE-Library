---
title: How to build a Tabular Transformer!
source: https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009
author:
published:
created: 2026-11-11
description: Discover what actually works in AI. Join millions of builders, researchers, and labs evaluating agents, models, and frontier technology through crowdsourced benchmarks, competitions, and hackathons.
tags:
---
Playground Series - Season 6 Episode 10

[Chris Deotte](https://www.kaggle.com/cdeotte) · 1st in this Competition · Posted a month ago

### How to build a Tabular Transformer!

## Q-FT: a Qwen transformer that reads tabular rows as embeddings

Kaggle Playground S6E10 (airline satisfaction, ROC AUC). How each row of the table is turned into vectors, fed into Qwen3-0.6B, and trained; what LoRA does; and what we found.

**At a glance**

- **0.959795**: 5-fold OOF AUC, raw features only.
- **22**: positions per row (21 feature tokens + 1 CLS).
- **10.1M**: trainable LoRA parameters on a frozen 0.6B backbone.
- **~52 min**: per fold on one A100 (3 epochs, ~920 rows/s).

## 1\. The idea in one picture

A language model normally reads *tokens* such as words. Q-FT skips text entirely:

1. It converts each **feature** of a row into one learned vector the same size as Qwen's word vectors (1,024 numbers).
2. It hands those vectors straight to Qwen.
3. It reads the answer from the last position.

It's the same trick vision-language models use for image patches.

![Q-FT pipeline](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/pipeline.png)

## 2\. How one feature becomes one token

Every one of the 21 columns gets its own small "embedder". Its output is the **sum** of up to three parts, then normalized to the scale Qwen expects:

- **Column embedding**: says *which* column this is. It's initialized from Qwen's own word vectors for the column name (the mean of the vectors of the tokens in "Flight Distance"), then trained.
- **Value embedding**: a lookup table indexed by the feature's code. Each category, each rating level, each age, and each of 3,300 Flight-Distance "routes" gets its own learned vector.
- **Numeric embedding** (Age, Flight Distance, both delays): the standardized number x goes through *periodic features* sin(2πf·x) and cos(2πf·x) at 32 learned frequencies f, then a small MLP (64 → 1,024). This is the same idea RealMLP uses so a network can resolve fine numeric differences.
- **RMSNorm × gain**: rescales the vector to RMS ≈ 0.030, the size of a typical Qwen word vector, so it enters the network like a normal token.

![How one feature becomes one token](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/feature_to_token.png)

### How each column is coded

Vocabularies come from train + test features only (label-free).

| Columns | Value embedding (code) | Vocabulary | Numeric embedding |
| --- | --- | --- | --- |
| Gender, Customer Type, Type of Travel, Class | category | 2, 2, 2, 3 | no |
| 13 service ratings (wifi … cleanliness) | rating level 0–5 | 6 each | no |
| Age | one code per distinct age | 75 | yes (standardized age) |
| Flight Distance | route id for every distance seen ≥ 3 times; rarer → "rare" code | 3,301 | yes (standardized log distance) |
| Departure / Arrival Delay | bucket {0, 1–5, 6–15, 16–30, 31–60, 61–120, >120, missing} | 8 each | yes (standardized log(1 + minutes)) |

Flight Distance gets both an ID and a number because in this dataset it behaves like a *route ID*: each distinct distance has its own mix of passengers. The ID lets the model learn per-route effects; the number lets it share information between nearby distances.

### Worked example: training row 0

| Column | Value | Code | Numeric x |
| --- | --- | --- | --- |
| Gender | Female | 0 | – |
| Customer Type | Loyal Customer | 0 | – |
| Age | 36 | 29 | −0.217 |
| Type of Travel | Personal Travel | 1 | – |
| Class | Eco | 1 | – |
| Flight Distance | 694 | 436 | −0.594 |
| Inflight wifi / Time convenient / Online booking / Gate | 4 / 4 / 4 / 3 | 4 / 4 / 4 / 3 | – |
| Food / Online boarding / Seat comfort / Entertainment | 1 / 4 / 1 / 1 | 1 / 4 / 1 / 1 | – |
| On-board / Leg room / Baggage / Check-in / Cleanliness | 3 / 4 / 4 / 3 / 1 | 3 / 4 / 4 / 3 / 1 | – |
| Departure Delay / Arrival Delay | 0 / 0.0 | 0 / 0 | −0.289 / −0.281 |
| *Label* | *satisfaction = False* | target, never an input |  |

These 21 codes and 4 numbers become 21 vectors of 1,024 numbers each, a 21 × 1,024 matrix that stands in for a "sentence".

## 3\. Feeding the vectors into Qwen

Hugging Face models accept `inputs_embeds`: you can pass vectors directly instead of token ids, which skips Qwen's word-embedding table. Q-FT builds the sequence like this:

1. **Shuffle the 21 feature tokens** into a fresh random order for every row, every training step.
2. **Append a learned CLS vector** as position 22. It's the model's "answer slot".
3. Qwen applies its usual rotary positions (0…21) and runs **causal attention**: each position sees only itself and earlier positions. So only the *last* position, CLS, sees all 21 features.
4. The final hidden state at CLS (1,024 numbers) goes through a `Linear(1024 → 1)` head, giving one logit.

![Sequence and causal attention mask](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/sequence_attention.png)

> **Why the random order matters.** A causal transformer sees features one after another. With a fixed order, the model ties itself to that one sequence. Shuffling every row teaches it that order is meaningless (like a set) and acts as strong data augmentation. In our pilots the fixed order lost **0.0069 AUC** (0.9535 vs 0.9604).

## 4\. LoRA: fine-tuning a big model by training a tiny add-on

Qwen3-0.6B has about 440M weights in its 28 layers. Instead of updating all of them, **LoRA** (Low-Rank Adaptation) freezes every original weight matrix W and learns a small correction next to it:

`output = W·x + (α / r) · B·(A·x)`

- **A** squeezes the input down to r = 16 numbers and **B** expands them back. Together B·A is a "rank-16" update to W.
- **B starts at zero**, so at step 0 the model is exactly pretrained Qwen. Training then moves only A and B. The scale α/r = 32/16 = 2.
- Adapters sit on all 7 projection matrices of every layer: attention q, k, v, o and MLP gate, up, down.
- **10.1M trainable LoRA weights** (about 2% of the backbone), plus 3.9M in the feature tokenizer and the CLS/head. The frozen 0.6B weights still run in the forward and backward passes.

![LoRA](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/lora.png)

## 5\. Training recipe

| Item | Setting |
| --- | --- |
| Data per fold | 559,708 outer-train rows (StratifiedKFold 5, seed 42); the held-out fold is only predicted |
| Backbone | Qwen/Qwen3-0.6B-Base: 28 layers, width 1,024, 16 attention heads (8 key/value heads), MLP width 3,072 |
| Fine-tuning | LoRA rank 16 (α 32) on q/k/v/o/gate/up/down; backbone frozen otherwise |
| Loss | binary cross-entropy on the CLS logit |
| Optimizer | AdamW (fused). Learning rate 2e-4 for the LoRA adapters, 1e-3 for the tokenizer, CLS and head; no weight decay |
| Schedule | 3 epochs (3,279 steps of 512 rows), 3% linear warmup, then linear decay to 0 |
| Precision | bf16 autocast for the transformer, fp32 master weights; gradient clipping at 1.0 |
| Augmentation | new random feature order for every row at every step |
| Prediction | mean logit over 5 feature orders (the table's order + 4 random) |
| Cost | ~920 rows/s, 26 GB peak memory, ~52 min per fold on one A100 (5 folds in parallel on 5 GPUs) |

## 6\. What we found

**Fold-0 pilots, 3 epochs each:**

| Variant | Fold-0 AUC | Takeaway |
| --- | --- | --- |
| **LoRA r16, pretrained Qwen** | **0.960444** | best |
| Random-init backbone (same architecture), full training | 0.960436 | pretraining adds nothing |
| Frozen backbone (only tokenizer + head train) | 0.959788 | a frozen LLM is already a decent feature mixer |
| Random column embeddings (no column names) | 0.958479 | names don't matter (compare full FT 0.958004) |
| Full fine-tune, learning rate 2e-5 | 0.958004 | learning rate too low for full FT |
| Fixed feature order | 0.953546 | shuffling is essential |

**All 5 folds:**

| Model | OOF AUC |
| --- | --- |
| TabPFN-3.5 (raw features, fine-tuned) | 0.961248 |
| Kumo-Tabular Large (raw features, fine-tuned) | 0.960378 |
| **Q-FT, LoRA (raw features only)** | **0.959795** |
| Tuned XGBoost, raw features (8 seeds) | 0.959852 |
| Tuned RealMLP, raw features (8 seeds) | 0.959311 |

Q-FT folds: 0.959602 / 0.959070 / 0.960712 / 0.960388 / 0.959283.

## 7\. Conclusion

- Our Tabular Transformer works great! On raw features it ties a tuned XGBoost and beats a tuned RealMLP. Wow!
- Next steps are to add tabular pretraining and in context learning similar to TFMs!

## UPDATE

We published a Part 2 [here](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334). Enjoy!

## 8 Comments

HotnessThis comment will be made public once posted.### [ZavodRobotov](https://www.kaggle.com/zavodrobotov)

Posted 23 days ago

Also learning\_rate warmup increase overfitting in my experiments### [ZavodRobotov](https://www.kaggle.com/zavodrobotov)

Posted 23 days ago

I think in such small dataset, there is no need for learning\_rate decay to 0. learning\_rate must be big to prevent overfitting. Also batch\_size should be small.### [DataRegressor](https://www.kaggle.com/azzamradman)

Posted 23 days ago

· 872nd in this Competition

I'm expecting a big jump in your score by the end of the competition. Maybe you're still using that leaderboard-hiding trick from your write-up last month (1 - pred) 😊. Wishing you all the best!### [yunsuxiaozi](https://www.kaggle.com/yunsuxiaozi)

Posted 23 days ago

· 1582nd in this Competition

How could anyone even think of applying a transformer to tabular data? That seems absolutely incredible.### [Hastika Cheddy](https://www.kaggle.com/hastikacheddy)

Posted 24 days ago

Really clear write-up. Since the random-init backbone matched pretrained Qwen (0.960436 vs 0.960444), it seems the gain comes from the architecture and not from language pretraining. Did you try removing the causal mask so CLS-style attention sees all features bidirectionally? That might remove the need for order shuffling, or at least reduce how much it matters.### [Chris Deotte](https://www.kaggle.com/cdeotte)

Topic Author

Posted 24 days ago

· 1st in this Competition

Hi. Great idea [@hastikacheddy](https://www.kaggle.com/hastikacheddy) I have not tried that yet, but that is a good next experiment. I think there are many ways to improve this transformer. I believe this transformer has lots of potential.### [Chovy](https://www.kaggle.com/chovyxu)

Posted 25 days ago

· 2647th in this Competition

I’m going to give this method a try too. Thanks for sharing!Posted a month ago

Really interesting approach to applying Transformers to tabular data! The feature tokenization and random shuffling strategy are especially impressive. It’s fascinating to see Q-FT achieve performance comparable to tuned XGBoost with raw features alone. Great work! 👏