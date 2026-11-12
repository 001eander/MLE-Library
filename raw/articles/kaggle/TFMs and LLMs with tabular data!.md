---
title: "1st Place - TFMs and LLMs with tabular data!"
source: "https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data"
author:
published:
created: 2026-11-11
description: "Discover what actually works in AI. Join millions of builders, researchers, and labs evaluating agents, models, and frontier technology through crowdsourced benchmarks, competitions, and hackathons."
tags:
---
1st Place - TFMs and LLMs with tabular data! — Predicting Airline Satisfaction[Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)

![banner](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/Robot%20Reads%20the%20Airplane%20Data.png)

Kaggle Playground S6E10 (Predicting Airline Passenger Satisfaction, ROC AUC). This is the second Playground in a row that we won using a special model created by Claude Opus5.5 (first discovered in September Playground [here](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay)): **a GPT classifier that reads each table row as text, with the target written first.**

- **One model:** by itself, one such model scored public LB 0.96189, the same as stacking all 245 public OOF/test pairs (0.96186).
- **Added to the stack:** adding it to those pairs moved our public LB to 0.96223, and more GPT classifiers took us to 0.96262.

## 1\. Thank you

Thank you to Kaggle for another fun Playground, and to every Kaggler who shared notebooks, ideas and especially **OOF/test prediction pairs**. Our final stack contains **245 public pairs**. A special thanks to the people who collected and cleaned them:

- **goodpjw2008** for the 245-member pool and the route-category TabPFN work;
- **kratosyan** for stacking every public OOF library;
- **sadamtorres, kagankoral, dariushafshar, abdullahsafwan333, koumeimaki, amanatar, sachith7, megayak, busyaprime, hemingweb, samanyu1808** and everyone else whose pairs are in the pool.

To give back, we shared our own pairs during the competition: [124 public OOF/test pairs in one dataset](https://www.kaggle.com/datasets/cdeotte/s6e10-124-public-oof-and-test-predictions), [11 tabular foundation model (TFM) pairs](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions) and a [notebook that stacks them](https://www.kaggle.com/code/cdeotte/stacking-tfms-with-public-oof).

## 2\. How to make two new kinds of tabular LLMs

During this competition I shared two discussion posts showing how to build **tabular transformers**:

- [Q-FT: a Qwen transformer that reads tabular rows as embeddings](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009);
- [Q-FT v2: an order-free ModernBERT that reads a table row as a set](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334).

In both, every feature of a row becomes one learned vector that goes straight into the transformer.

**Now I would like to teach you how to make 2 new kinds of tabular LLMs.** These ones don't need a feature tokenizer at all. They read each row **as plain text**, exactly like a sentence, with an ordinary GPT. The only difference between the two is *where the target goes*:

![two kinds of tabular LLMs](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/two_directions.png)

|  | Tabular LLM #1: target LAST | Tabular LLM #2: target FIRST |
| --- | --- | --- |
| nickname | the **Detective** | the **Storyteller** |
| what it asks | "I've read every clue. What's the answer?" | "Try each ending. Which one makes this story most likely?" |
| models | P(target \| features) | P(features \| target), turned around with Bayes' rule |
| passes per row | 1 | 2 (one per target value) × several feature orders |
| this competition | ~0.961 AUC | **0.9624 AUC**, our best single model |

You can build both with any small GPT. We used **GPT-2** (124M parameters) from Hugging Face. First, turn each row into text, the same way for both models: `"column is value"` for every column, joined by `", "`.

```dart
Gender is Female, Customer Type is Loyal Customer, Age is 36, Type of Travel is Personal Travel, Class is Eco,
Flight Distance is 694, Inflight wifi service is 4, ..., Departure Delay in Minutes is 0, Arrival Delay in Minutes is 0.0
```

### 2.1 Tabular LLM #1: target last (the Detective)

![the detective](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/Robot%20Detective%20Solves%20the%20Clues.png)

The Detective reads all the clues first and names the answer at the end. Put the target **last** and let the model predict it:

```dart
Gender is Female, Customer Type is Loyal Customer, ..., Arrival Delay in Minutes is 0.0, satisfaction is True
```

**How to train it:** fine-tune GPT-2 as a normal causal language model on the training rows written this way, with the real label at the end. You can put the loss on all tokens or only on the label token. Shuffle the feature order every step: a table row has no natural order, so this teaches the model to read the row as a set, and it is great data augmentation.

**How to predict:** feed the text up to `satisfaction is` and read the next-token probabilities of `True` and `False`:

```
ids = tok("Gender is Female, ..., Arrival Delay in Minutes is 0.0, satisfaction is", return_tensors="pt").input_ids
p = model(ids).logits[0, -1].softmax(-1)                      # next-token distribution
t, f = tok(" True").input_ids[0], tok(" False").input_ids[0]
p_satisfied = (p[t] / (p[t] + p[f])).item()
```

![target last](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/target_last.png)

For train row 0 (label `False`), a model that never saw this row gives p(True) = 0.225 and p(False) = 0.775, so **P(satisfied) = 0.23**. (This model was trained with the target at random positions and read with the target last.)

This is a *discriminative* classifier: like XGBoost, it learns P(target | features) directly. It is simple, needs one forward pass per row, and scored about **0.961 AUC** here.

### 2.2 Tabular LLM #2: target first (the Storyteller)

![the storyteller](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/Robot%20Storyteller%E2%80%99s%20Believability%20Test.png)

Now flip it around. The Storyteller already knows the ending and tells the story. Put the target **first** and let the model predict the features:

```dart
satisfaction is True, Gender is Female, Customer Type is Loyal Customer, Age is 36, ..., Arrival Delay in Minutes is 0.0<eos>
satisfaction is False, Gender is Female, Customer Type is Loyal Customer, Age is 36, ..., Arrival Delay in Minutes is 0.0<eos>
```

A causal language model gives a probability to every token given the tokens before it. **Add up the log-probabilities of all the feature tokens** and you get log P(features | target). Do it once with `True` and once with `False`. Then Bayes' rule turns the two stories into a classifier:

```
log-odds(satisfied) = [ log P(True) + log P(features | True) ] − [ log P(False) + log P(features | False) ]
P(satisfied)        = sigmoid(log-odds)
```

In code, scoring one row under one feature order looks like this:

```
eos, prior = tok.eos_token, 0.4436                             # prior = training rate of satisfaction = True

def log_p(text, n_prefix):
    ids = tok(text, return_tensors="pt").input_ids
    lp = model(ids).logits[0, :-1].log_softmax(-1)
    tok_lp = lp.gather(-1, ids[0, 1:, None])[:, 0]            # log-prob of every next token
    return tok_lp[n_prefix - 1:].sum().item()                  # sum over the feature tokens only

row = " Gender is Female, Customer Type is Loyal Customer, ..., Arrival Delay in Minutes is 0.0"
lt = log_p("satisfaction is True," + row + eos, len(tok("satisfaction is True,").input_ids))
lf = log_p("satisfaction is False," + row + eos, len(tok("satisfaction is False,").input_ids))
log_odds = math.log(prior / (1 - prior)) + lt - lf
```

![recipe](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/target_first_recipe.png)

**The recipe, step by step:**

1. **Replica.** Fine-tune GPT-2 on the *original* dataset (here the 129,880-row [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) data), written as `"column is value"` text with a fresh random column order per row. The easiest way is the GReaT library ([be-great](https://github.com/kathrinse/be_great)): `GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)`.
	- The starting point matters: after the full fine-tune in step 2, a 60-epoch replica beats an 8-epoch replica by **+1.2e-3 AUC**.
		- Plain GPT-2 is far worse.
2. **Classifier.** For each fold, continue training the replica on the competition training rows with the **target first**, then the 21 features in a fresh random order every step, loss on all tokens.
	- 8 epochs, AdamW, lr 5e-5 with linear decay to 0, batch 32, bf16.
		- GPT-2 small trains at about 600 rows/s, about 2 hours per fold on one A100.
3. **Score twice.** For each held-out (or test) row, compute log P(features | True) and log P(features | False) as in the code above.
4. **Average over feature orders.** The model was trained on random orders, so ask it about several orders and combine them with log-sum-exp: `log-odds = logsumexp_o log P(x | True, o) − logsumexp_o log P(x | False, o)` (plus the prior). 30 random orders give most of the gain:

![orders curve](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/orders_curve.png)

**Worked example: train row 0 (label `False`).** Here are the per-field log-probabilities from the fold model that never saw this row:

![worked example](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/row0_evidence.png)

- **What it shows:** the model has learned which fields carry label information. Personal travel points strongly to "not satisfied", and good wifi points the other way.
- **One order:** the sum is log-odds −1.05, so **P(satisfied) = 0.26**.
- **30 orders:** log-sum-exp gives −1.07, so P(satisfied) = 0.256.
- **Both models agree:** the Detective said 0.23 for the same row.

**Details that matter:**

- **Format.** Write values exactly as the original data was written: integers as `36`, floats as `0.0`, missing values as `None`, and `True` / `False` for the target.
- **Tokenizer.** Use GPT-2's own tokenizer. In this competition models with a digit-by-digit tokenizer (e.g. Qwen) did worse.
- **Variants for diversity:**
	- GPT-2 medium and large reach the same AUC but make different errors.
		- **Target anywhere:** during training the target sits at a random position among the 22 fields. Score it by averaging the per-order log-odds over random orders.
		- Both kinds of variant stack well.

### 2.3 Which one should you use?

![two roads](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/Same%20Model%2C%20Opposite%20Directions.png)

- **Try both.** They read the same row in opposite directions, so they make different errors and stack nicely.
- **Target last** is the general-purpose choice. It works on any tabular data, like any other classifier.
- **Target first** wins when the data itself was **written by a language model**. That is the case for Kaggle's synthetic Playground data (section 5): the Storyteller then reads the row the same way it was written. Here it scored **0.9624 vs 0.961**, and it is the recipe behind both of our last two wins.

## 3\. The public OOF/test pairs are very strong

Every pair is an out-of-fold (OOF) prediction for `train.csv` plus a test prediction, made with the same `StratifiedKFold(5, shuffle=True, random_state=42)` folds. That makes them stackable:

- convert every prediction to a logit (clip at 1e-15, then at ±30; pairs that are already logits are used as they are);
- fit an **L2 logistic regression** (cuML, `C=1`) on the OOF logits of 4 folds and predict the 5th;
- average the 5 fold models for test. Averaging 5 refits also removes cuML's small run-to-run noise.

The best public pair scores about 0.9617 OOF. Stacking all 245 gives **CV 0.96218 / public LB 0.96186**. One caveat: some public members early-stop on the scored fold, so the public CV is slightly optimistic.

## 4\. Adding tabular foundation models (TFMs)

We ran 9 TFMs on the raw columns and shared their OOF/test pairs, plus tuned XGBoost and RealMLP baselines:

| model | OOF AUC |
| --- | --- |
| TabPFN-3.5 | 0.96115 |
| Kumo-Tabular large | 0.96004 |
| XGBoost (baseline) | 0.95985 |
| RealMLP (baseline) | 0.95931 |
| TabFM | 0.95887 |
| EXAONE | 0.95856 |
| Kumo-Tabular small | 0.95848 |
| LimiX-2 | 0.95847 |
| Causilo | 0.95828 |
| TabICL-2 | 0.95809 |
| Mitra-2 | 0.95707 |
| **9 TFMs stacked** | **0.96120** |

TFMs predict by reading the training rows as context (in-context learning). That gives them a different inductive bias from trees, so they stack well. The community pushed this further, for example TabPFN-3.5 with Flight Distance declared as a categorical "route ID".

In the public pool, the **26 TFM pairs** (ours plus the community's TabPFN / TabICL variants) add **+1.4e-4 CV** to the 219 other pairs: 0.96204 → 0.96218. A public TabPFN-3.5 pair is also the largest non-GPT weight in our final stack.

## 5\. Why the Storyteller wins here: this data was written by a language model

Kaggle's Playground data is synthetic. Everything we found says this dataset was generated by **GReaT** ([Borisov et al. 2023](https://arxiv.org/abs/2210.06280)), which works in two steps:

1. GPT-2 is fine-tuned on the original dataset, with each row written as `"column is value"` text in a random column order.
2. It then samples new rows token by token.

**Evidence:**

- **GPT-2 tokenizer fingerprints:** 3-digit distances are single tokens, while 4-digit distances are 2 tokens that share prefixes.
- **Generation settings:** sampling at temperature 0.7 with top-k 50 reproduces the dataset's distortions of the original.
- **Label first:** the label was the **first** thing generated. A label-first prompt reproduces the training label rate exactly.

So each row was produced as: the label first, then the 21 features one token at a time. The best possible classifier is the generator's own answer to "which label makes this row most likely?". The Storyteller is trained to ask exactly that question, so it approximates the generator far better than any model that reads the features as numbers.

## 6\. How good is one GPT classifier?

| single model | OOF AUC |
| --- | --- |
| GPT-2 small, target first (public LB **0.96189** alone) | **0.96242** |
| GPT-2 small, target anywhere, 24 epochs | **0.96245** |
| GPT-2 medium, target first | 0.96237 |
| GPT-2 large, target first | 0.96236 |
| GPT-Neo 125M, target first | 0.96216 |
| target last (same family of models) | ~0.961 |
| best public pair | ~0.9617 |
| tuned XGBoost (ours) | 0.96149 |
| TabPFN-3.5 | 0.96115 |

**One** target-first GPT-2 classifier scores about +7e-4 above the best public pair, +9e-4 above our tuned XGBoost and +1.3e-3 above TabPFN-3.5. On the public LB, by itself it ties the stack of all 245 public pairs.

## 7\. Adding the GPT classifier to the stack

![stack history](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/stack_history.png)

| stack | pairs | CV | public LB |
| --- | --- | --- | --- |
| public pairs (incl. TFMs) | 245 | 0.96218 | 0.96186 |
| \+ GPT-2 small, target first (+ our XGB, RealMLP) | 248 | 0.96266 | 0.96223 |
| \+ GPT-2 large | 247\* | 0.96277 | 0.96242 |
| \+ target-anywhere GPT, GPT-2 medium | 250 | 0.96282 | 0.96249 |
| \+ more GPT variants, collaborator members | 259 | **0.96290** | **0.96262** |

\* without our XGB and RealMLP, which added nothing once a GPT classifier was in the stack.

- **The jump.** Adding a **single** GPT classifier to 245 strong public pairs gains **+4.9e-4 CV and +3.7e-4 public LB**.
- **Why it helps this much.** The GPT classifier is very accurate and *very different* from everything else. It models P(row | label) instead of P(label | row), and it has learned the generator's own quirks.
- **How much of the stack it carries.** In the final stack, the 10 GPT-based pairs carry about **80%** of the stack's output. The 245 public pairs largely cancel each other.
- **Final picks:**
	- our public-LB best, the 259-pair stack (CV 0.96290, public 0.96262);
		- a 332-pair stack with the highest CV (0.96299, public 0.96260). It adds 73 more pairs, mostly from our collaborator, because more pairs in a logistic regression has been a reliable bet in our past competitions.

## 8\. Where tabular modeling is going

![tabular model family](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2026/tabular_model_family.png)

Four kinds of models now beat tuned GBDTs on these competitions, and each keeps getting stronger:

- **Tabular foundation models** (TabPFN, TabICL, Kumo, LimiX, Mitra, …) learn from the training rows in context, with little or no tuning.
- **LLMs that take tabular embeddings as input:** each feature becomes one learned vector fed straight into a transformer (our [Q-FT post](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009) and its [v2](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334)).
- **LLMs that read the row as text, target last:** the Detective, P(target | features).
- **LLMs that read the row as text, target first:** the Storyteller, P(features | target) turned around with Bayes' rule. On data written by a language model, it is the most accurate single model we have found, and it won us two competitions in a row.

Stacked together with the community's OOF/test pairs, these models are becoming the most accurate solution for tabular data. Thanks again to everyone who shared!

## Citation

Chris Deotte. 1st Place - TFMs and LLMs with tabular data!. https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data. 2026. Kaggle

HotnessThis comment will be made public once posted.