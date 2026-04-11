"""ASC sequence-classification training. Paths and hyperparameters from ``pipeline.config``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.config import BACKBONE_PRESETS, MAX_LEN, asc_training_run_dir, hf_asc_dataset_dir  # noqa: E402

NUM_LABELS = 3
LABEL_NAMES = ["positive", "negative", "neutral"]


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        examples["aspect"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def train_asc(domain: str, model_name: str) -> None:
    data_path = hf_asc_dataset_dir(domain)
    output_dir = asc_training_run_dir(domain, model_name)
    preset = BACKBONE_PRESETS[model_name]
    hf_model = preset["hf_model"]
    hp = preset["asc"]

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"ASC dataset not found: {data_path}\n"
            f"Run: python asc/asc_prepare_data.py --domain {domain}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Domain={domain}  backbone={model_name}")
    print(f"Loading data from {data_path} …")
    ds = load_from_disk(data_path)

    print(f"Loading tokenizer: {hf_model} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    tokenized = ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=["sentence", "aspect"],
    )
    tokenized = tokenized.rename_column("label", "labels")

    load_kw = {"num_labels": NUM_LABELS}
    if model_name == "deberta":
        load_kw["ignore_mismatched_sizes"] = True

    print(f"Loading model: {hf_model} …")
    model = AutoModelForSequenceClassification.from_pretrained(hf_model, **load_kw)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        learning_rate=hp["learning_rate"],
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

    print("Training …")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved → {final_dir}")

    print("\nEvaluating on test set …")
    results = trainer.evaluate(tokenized["test"])
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test Macro-F1: {results['eval_macro_f1']:.4f}")

    preds = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    print(f"\nClassification Report:\n{report}")

    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": results["eval_accuracy"],
                "macro_f1": results["eval_macro_f1"],
                "report": report,
            },
            f,
            indent=2,
        )


def main():
    ap = argparse.ArgumentParser(description="Train ASC (sequence classification)")
    ap.add_argument(
        "--domain",
        choices=["restaurant", "laptop"],
        default="restaurant",
        help="Training domain (must match prepared HF dataset).",
    )
    ap.add_argument(
        "--model_name",
        choices=["bert", "deberta"],
        default="bert",
        help="Backbone; paths follow pipeline.config.MODELS.",
    )
    args = ap.parse_args()
    train_asc(args.domain, args.model_name)


if __name__ == "__main__":
    main()
