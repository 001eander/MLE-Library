---
title: Part 2 - How to build a Tabular Transformer!
source: https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334
author:
published:
created: 2026-11-12
description: Playground Series - Season 6 Episode 10
tags:
---
[Kaggle](https://www.kaggle.com/organizations/kaggle) · Playground Prediction Competition · 11 days ago

more\_horiz

[Chris Deotte](https://www.kaggle.com/cdeotte) · 1st in this Competition · Posted 22 days ago

### Part 2 - How to build a Tabular Transformer!

## Q-FT v2: an order-free ModernBERT that reads a table row as a set

Kaggle Playground S6E10 (airline satisfaction, ROC AUC). This is a follow-up to our previous post, [Q-FT: a Qwen transformer that reads tabular rows as embeddings](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009). There we fed each feature of a row to Qwen3-0.6B as one "token" vector and fine-tuned it with LoRA. It tied a tuned XGBoost on raw features (5-fold OOF 0.9598).

Since then we kept improving the transformer. Three changes moved it up by about **+1e-3 AUC**, and the new model is also about **5× cheaper** to train (all 5 folds: ~50 GPU-minutes instead of ~260):

1. **Order-free encoder.** We replaced the causal Qwen decoder with a bidirectional ModernBERT encoder and put **every feature token at position 0**. The model now reads a row as a *set*, so no shuffling or order averaging is needed.
2. **Side features.** Each feature token also receives a few engineered numbers about its value, and two "summary tokens" describe the whole row.
3. **Pretraining on the table itself.** Before training on the label, the model learns to spot values that were swapped in from other rows (replaced-token detection). This uses no labels at all.

**At a glance**

- **0.960798**: 5-fold OOF AUC (Q-FT: 0.959795).
- **24**: tokens per row (CLS + 21 feature tokens + 2 summary tokens), all at position 0.
- **149M**: parameters of ModernBERT-base, all trained (no LoRA needed).
- **~6.5 min**: per fold on one A100, plus 18 min of pretraining done once for all folds (Q-FT: ~52 min per fold).

## 1\. What changed since Q-FT

|  | Q-FT (previous post) | v2 (this post) |
| --- | --- | --- |
| Backbone | Qwen3-0.6B, a decoder (causal attention) | ModernBERT-base, an encoder (bidirectional attention) |
| Feature order | positions 0…21, new random order for every row | every token at position 0: order cannot matter |
| Where the answer is read | the CLS token at the end | the mean of all output vectors |
| What is trained | LoRA adapters (rank 16) | all 149M weights |
| Inputs | 21 feature tokens | 21 feature tokens **\+ side features** + 2 summary tokens |
| Pretraining | none (only Qwen's language pretraining) | **replaced-token detection** on the table's features |
| Speed | ~920 rows/s, ~52 min per fold | ~2,930 rows/s, ~6.5 min per fold |
| 5-fold OOF AUC | 0.959795 | **0.960798** |

## 2\. The idea in one picture

The row-to-tokens trick is the same as before: every feature becomes one learned vector, and the vectors go straight into the transformer through `inputs_embeds` (no text, no word tokenizer). What changed is everything around it.

![Pipeline of the new transformer](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/v2_pipeline.png)

## 3\. Order-free: reading a row as a set

A table row has no natural order: "Age, Class, Wifi" means the same as "Wifi, Age, Class". In Q-FT the model *could* see an order, because:

- **Causal attention**: a decoder token only attends to *earlier* tokens, so the answer had to be read at the last position.
- **Positions**: every token got a position number 0, 1, 2, …, so the model could learn things like "the 3rd token is …".

So Q-FT had to **shuffle** the features for every row during training and **average 5 orders** at prediction time. That works, but it wastes capacity on learning that order doesn't matter.

The new model removes order completely:

- **ModernBERT is an encoder**: every token attends to every other token (bidirectional attention).
- ModernBERT encodes positions with rotary embeddings (RoPE), which are driven by the `position_ids` you pass in. We pass **`position_ids = 0` for every token**. Then all tokens look position-less, and the model sees an unordered set.

We checked it: shuffling the feature order changes the model's output by about 3e-6, which is floating-point error. So no shuffling during training, a single forward pass at prediction, and **more than 3× as many rows per second** as Q-FT.

![Causal attention with positions vs full attention with every token at position 0](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/v2_order_free.png)

**The output.** With full attention every token is equally informed, so we simply **average the 23 output vectors** (21 feature tokens + 2 summary tokens) and apply `Linear(768 → 1)`. A CLS head, an attention-pooling head and mean+max pooling all scored the same within noise, so we kept the simplest.

**Things we tried that did not matter:**

- ModernBERT-large scored no better than base.
- DeBERTa-v3 (base and large) was worse.
- A randomly initialized ModernBERT scored the same as the pretrained one. Just like in the Q-FT post, the language pretraining is not what makes this work.

## 4\. Side features: telling each token more about its value

Each feature token is still the sum of a **column embedding** (which column), a **value embedding** (which value, e.g. one vector per Flight-Distance "route"), and, for numeric columns, a **numeric embedding** (periodic sin/cos features → small MLP). The sum is normalized with RMSNorm and scaled to the size of ModernBERT's own word vectors.

New: every token also gets **9 engineered numbers about its value**, multiplied by a learned 9 × 768 matrix (one per column) and added to the token:

- **How common the value is**: its count in train + test, its count in the original airline-satisfaction dataset, and the ratio.
- **Route profile**: Flight Distance acts like a route ID in this data. For the row's distance, we compute the average and the spread of this column over all rows with the same distance, plus the original dataset's average.
- **How typical the value is in this row**: a label-free model predicts each column from the other 20 columns. We give the token the expected value and the residual (actual − expected).
- **The original data's opinion**: the smoothed log-odds of "satisfied" for this exact value in the original dataset.

Two **summary tokens** describe the whole row:

- **Row statistics** (28 numbers): sum, mean, spread and counts of the 13 ratings, delay differences, and a few step points.
- **Original-data block** (17 numbers): the prediction of a model trained only on the original dataset, plus statistics of the original rows that match this row on keys such as Flight Distance + Age + Class.

None of these use the competition's labels. They are either label-free or use the original dataset's labels only.

![How one feature becomes one token, with side features](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/v2_token.png)

**Effect: about +4e-4 AUC on fold 0**, the same in both seeds. That is much less than the same kind of features give a GBDT (about +1.6e-3). The reason is that the transformer's per-value embeddings already learn most of what the counts and route profiles say.

## 5\. Pretraining on the table: replaced-token detection (RTD)

This is the ELECTRA idea applied to table rows:

1. Take a row and, for about **15%** of its 21 values, **swap in the value of the same column from a random other row**.
2. Run the model and ask, for **every token**: "is this value real or replaced?" (one small `Linear(768 → 1)` head, binary cross-entropy).

To answer, the model must learn which values go together, such as which classes fly which routes and which ratings move together. That is exactly the structure the satisfaction task needs.

- **No labels are used**, so we can pretrain on the features of **train + test + the original dataset** (1.13M rows).
- Because it never sees a label, the pretraining does not depend on the CV fold. We run it **once** (3 epochs, 18 min) and start every fold's fine-tuning from the same weights.
- After pretraining, the detection head is dropped and the model is fine-tuned on satisfaction as usual.

![Replaced-token detection](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Oct-2026/v2_rtd.png)

**Effect: about +3e-4 AUC on fold 0**, in both seeds. Two details matter:

- After RTD the fine-tuning peaks earlier, so we fine-tune for **2 epochs** instead of 3 (the third epoch overfits).
- Longer pretraining (10 epochs) was worse than 3. *Supervised* pretraining on the original dataset did not help at all.

## 6\. Training recipe

| Item | Setting |
| --- | --- |
| Data per fold | 559,708 outer-train rows (StratifiedKFold 5, seed 42); the held-out fold is only predicted |
| Backbone | answerdotai/ModernBERT-base: 22 layers, width 768, 12 attention heads; bidirectional; `position_ids` = 0 for every token |
| Tokens per row | CLS + 21 feature tokens + 2 summary tokens = 24 |
| Head | mean of the 23 non-CLS output vectors → `Linear(768 → 1)` |
| Pretraining (once) | replaced-token detection: 15% of the 21 values swapped with a random row's value; per-token BCE; 3 epochs over 1.13M rows (train + test + original features); batch 512 |
| Fine-tuning | BCE on the label; all weights trained; AdamW, learning rate 5e-5 for ModernBERT and 1e-3 for the tokenizer and head; batch 512; 2 epochs; 3% warmup then linear decay; bf16 autocast; gradient clipping 1.0 |
| Prediction | one forward pass (order-free, so no order averaging) |
| Cost | ~2,930 rows/s, 11.4 GB peak memory, ~6.5 min per fold + 18 min of shared pretraining, on one A100 |

## 7\. Results

**Fold 0, step by step**:

| Variant | Fold-0 AUC |
| --- | --- |
| Q-FT (previous post): Qwen3-0.6B, causal, LoRA, shuffled orders | 0.960444 |
| Qwen3-0.6B with bidirectional attention + mean head | 0.960602 |
| ModernBERT-base, order-free, mean head | 0.960368 |
| \+ side features | 0.960655 |
| **\+ replaced-token-detection pretraining** | **0.960977** |

## 7\. Conclusions

- Transformers have many properties to experiment with including: pretraining, attention design, data representation, and more!
- Modifying transformer properties can improve transformer's performance!
- Next steps are to add more tabular pretraining and in context learning similar to TFMs!

## 3 Comments

HotnessThis comment will be made public once posted.### [Naji](https://www.kaggle.com/najiama)

Posted 20 days ago

· 307th in this Competition

Hi [@cdeotte](https://www.kaggle.com/cdeotte),

Thank you so much for sharing this brilliant Q-FT v2 architecture!

My background is actually in Electrical Engineering rather than CS, so reading your explanation of projecting scalar numbers into multi-frequency sine/cosine waves and using RMSNorm felt exactly like building a digital signal processing (DSP) pipeline. It is amazing to see how well it works on tabular data.

I wanted to share my results after implementing your architecture from scratch using Kaggle's **T4x2 GPUs**.

**Hardware & Implementation Adjustments:** To make this fit within Kaggle's 12-hour session on T4 GPUs, I used `nn.DataParallel` across both GPUs, swapped FlashAttention-2 for PyTorch's native `sdpa`.

- **RTD Pretraining (3 epochs):** ~63 minutes.
- **5-Fold Supervised Fine-Tuning (2 epochs):** ~137 minutes.
- **Total Runtime:** ~3.5 hours.

**Results:** Without even adding the external Original Dataset yet, the model hit an **OOF ROC AUC of 0.96075** (LogLoss: 0.2232, Brier: 0.0597) and **LB 0.96082**. At the end of Epoch 2, my Train BCE (0.2223) and Val BCE (0.2233) were nearly identical, proving how beautifully the RTD pretraining and your feature tokenizer regularize the 149M parameters.

Thanks again for the incredible inspiration and for pushing the boundaries of what Transformers can do on Tabular data!### [Louis Zango](https://www.kaggle.com/louiszango)

Posted 21 days ago

· 384th in this Competition

that's crazy

1Posted 22 days ago

· 102nd in this Competition

Thanks for the detailed ablations. Could you share the training code, fold assignments, OOF predictions and test predictions? That would let us compare the existing stack with and without Q-FT v2 on identical folds.