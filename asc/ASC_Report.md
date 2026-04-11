# Aspect Sentiment Classification (ASC) Report

**Author:** Fangyuan Zhang
**Dataset:** SemEval-2014 Task 4 — Restaurant Domain
**Date:** April 2025

---

## 1. Task Definition

Given a sentence and a specific aspect term within it, classify the sentiment polarity as **Positive**, **Negative**, or **Neutral**.

**Example:**

| Sentence | Aspect Term | Polarity |
|----------|-------------|----------|
| "The food was great but the service was terrible." | food | Positive |
| "The food was great but the service was terrible." | service | Negative |

---

## 2. Data Preparation

### 2.1 Source

Gold aspect term annotations from **SemEval-2014 Task 4** `data/Restaurants_Train_v2.xml` (train/val pool) and `data/Restaurants_Test_Gold.xml` (official test). Each `<aspectTerm>` element provides the term text and its polarity label.

### 2.2 Label Mapping

| Polarity | Label ID | Count |
|----------|----------|-------|
| Positive | 0 | 2,164 |
| Negative | 1 | 805 |
| Neutral  | 2 | 633 |

91 "conflict" examples (2.5%) were excluded as per the standard 3-class setup.

**Total usable examples:** 3,602

### 2.3 Data Split

Sentences from `Restaurants_Train_v2.xml` are split **by `sentence_id`** into train vs. validation (10% val, seed=42), shared with the ATE pipeline. The **test** set is the full official **`Restaurants_Test_Gold.xml`** (aspect-level polarity labels). Aspect-level counts:

| Split | Examples (aspect pairs) | Notes |
|-------|-------------------------|--------|
| Train | 3,220 | aspects from train split sentences |
| Validation | 382 | aspects from val split sentences |
| Test | 1,120 | all labeled aspects in Gold test |

### 2.4 Input Construction

Each example is a `(sentence, aspect_term)` pair encoded as a BERT sentence-pair:

```
[CLS] sentence_text [SEP] aspect_term [SEP]
```

This allows BERT to attend to both the full context and the target aspect via the `token_type_ids` (segment embeddings), enabling the model to learn aspect-specific sentiment.

---

## 3. Model Architecture

- **Base model:** `bert-base-uncased` (110M parameters)
- **Architecture:** `BertForSequenceClassification` with a linear classification head (768 -> 3)
- **Max sequence length:** 128 tokens

---

## 4. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Optimizer | AdamW |
| Best model selection | Macro-F1 on validation set |
| Seed | 42 |

**Hardware:** NVIDIA T4 GPU (Modal cloud)
**Training time:** ~5 minutes

---

## 5. Results

### 5.1 Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.8033** |
| **Macro-F1** | **0.7394** |
| Weighted-F1 | 0.81 |

### 5.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 0.91 | 0.89 | 0.90 | 214 |
| Negative | 0.76 | 0.68 | 0.72 | 90 |
| Neutral | 0.53 | 0.68 | 0.60 | 57 |

### 5.3 Analysis

- **Positive** sentiment is classified with high accuracy (F1=0.90), benefiting from being the majority class (59.3% of data).
- **Negative** sentiment achieves reasonable performance (F1=0.72). Some negative examples may be confused with neutral when the sentiment is implicit.
- **Neutral** is the most challenging class (F1=0.60). This is expected because:
  1. It is the smallest class (15.8% of data), leading to fewer training examples.
  2. Neutral sentiment is inherently ambiguous — it is often expressed through factual statements that lack strong sentiment cues (e.g., "Food and service was okay").
  3. The boundary between "mildly positive/negative" and "neutral" is subjective.

---

## 6. Scripts

| Script | Description |
|--------|-------------|
| `asc_prepare_data.py` | Parses XML, builds (sentence, aspect) pairs, splits data |
| `asc_train.py` | Local training script (CPU/GPU) |
| `asc_train_modal.py` | Modal cloud training script (T4 GPU) |
| `asc_evaluate.py` | Standalone evaluation with confusion matrix |

---

## 7. Reproduction

```bash
# Step 1: Prepare data
cd asc
python asc_prepare_data.py

# Step 2: Train (pick one)
python asc_train.py              # local
modal run asc_train_modal.py     # Modal cloud (T4 GPU)

# Step 3: Evaluate
python asc_evaluate.py --model_dir asc_output_restaurant_bert/final
```
