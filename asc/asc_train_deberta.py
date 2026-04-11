import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ────────────────────────────────────────────────────
# Changed from bert-base-uncased to DeBERTa-base
MODEL_NAME = "microsoft/deberta-base"
DATA_PATH  = "asc_data_restaurant"
OUTPUT_DIR = "./asc_output_deberta"
MAX_LEN    = 128
NUM_LABELS = 3
LABEL_NAMES = ["positive", "negative", "neutral"]


# ── Tokenize ─────────────────────────────────────────────────
def tokenize(examples, tokenizer):
    """Encode as: [CLS] sentence [SEP] aspect [SEP]"""
    return tokenizer(
        examples["sentence"],
        examples["aspect"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )


# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


# ── Main ─────────────────────────────────────────────────────
def main():
    print(f"Loading data from {DATA_PATH} ...")
    ds = load_from_disk(DATA_PATH)

    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized = ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=["sentence", "aspect"],
    )
    tokenized = tokenized.rename_column("label", "labels")

    print(f"Loading model: {MODEL_NAME} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,    # smaller batch for DeBERTa
        per_device_eval_batch_size=32,
        learning_rate=2e-5,               # slightly lower lr for DeBERTa
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("Training ...")
    trainer.train()

    # Save best model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved -> {final_dir}")

    # Evaluate on test set
    print("\nEvaluating on test set ...")
    results = trainer.evaluate(tokenized["test"])
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test Macro-F1: {results['eval_macro_f1']:.4f}")

    # Detailed per-class report
    preds  = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    print(f"\nClassification Report:\n{report}")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump({
            "accuracy":  results["eval_accuracy"],
            "macro_f1":  results["eval_macro_f1"],
            "report":    report,
        }, f, indent=2)
    print(f"Results saved -> {OUTPUT_DIR}/test_results.json")


if __name__ == "__main__":
    main()
