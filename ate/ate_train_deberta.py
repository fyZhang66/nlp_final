import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score

# ── Config ────────────────────────────────────────────────────
# Using deberta-base (v1) for stable compatibility with Colab transformers
MODEL_NAME = "microsoft/deberta-base"
DATA_PATH  = "ate_data"
OUTPUT_DIR = "./ate_output_deberta"
MAX_LEN    = 128

LABEL_LIST = ["O", "B-ASP", "I-ASP"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


# ── Tokenize + align labels ───────────────────────────────────
def tokenize_and_align(examples, tokenizer):
    enc = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
        padding="max_length",
    )

    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = enc.word_ids(batch_index=i)
        labels   = []
        prev_word = None

        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(LABEL2ID.get(tags[wid], 0))
            else:
                # subword continuation
                tag = tags[wid]
                if tag in ["B-ASP", "I-ASP"]:
                    labels.append(LABEL2ID["I-ASP"])
                else:
                    labels.append(-100)
            prev_word = wid

        all_labels.append(labels)

    enc["labels"] = all_labels
    return enc


# ── Metrics ───────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_tags, pred_tags = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_tags.append(ID2LABEL[l])
            pred_tags.append(ID2LABEL[p])
        y_true.append(true_tags)
        y_pred.append(pred_tags)

    return {
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"Loading data from {DATA_PATH} ...")
    ds = load_from_disk(DATA_PATH)

    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    tokenized = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    print(f"Loading model: {MODEL_NAME} ...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,

    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,   # DeBERTa-v3 is larger; reduce batch size
        per_device_eval_batch_size=16,
        learning_rate=2e-5,              # slightly lower lr works better for DeBERTa
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training ...")
    trainer.train()

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved -> {final_dir}")

    print("\nEvaluating on test set ...")
    results = trainer.evaluate(tokenized["test"])
    print(f"Test F1:        {results['eval_f1']:.4f}")
    print(f"Test Precision: {results['eval_precision']:.4f}")
    print(f"Test Recall:    {results['eval_recall']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {OUTPUT_DIR}/test_results.json")


if __name__ == "__main__":
    main()
