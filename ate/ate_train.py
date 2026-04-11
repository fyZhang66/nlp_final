"""ATE token-classification training. Paths and hyperparameters from ``pipeline.config``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.config import (  # noqa: E402
    BACKBONE_PRESETS,
    MAX_LEN,
    ate_training_run_dir,
    hf_ate_dataset_dir,
)
from pipeline.test_results import write_test_results  # noqa: E402

LABEL_LIST = ["O", "B-ASP", "I-ASP"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


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
        labels = []
        prev_word = None

        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(LABEL2ID.get(tags[wid], 0))
            else:
                tag = tags[wid]
                if tag in ["B-ASP", "I-ASP"]:
                    labels.append(LABEL2ID["I-ASP"])
                else:
                    labels.append(-100)
            prev_word = wid

        all_labels.append(labels)

    enc["labels"] = all_labels
    return enc


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
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def train_ate(domain: str, model_name: str) -> None:
    data_path = hf_ate_dataset_dir(domain)
    output_dir = ate_training_run_dir(domain, model_name)
    preset = BACKBONE_PRESETS[model_name]
    hf_model = preset["hf_model"]
    hp = preset["ate"]

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"ATE dataset not found: {data_path}\n"
            f"Run: python ate/ate_prepare_data.py --domain {domain}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Domain={domain}  backbone={model_name}")
    print(f"Loading data from {data_path} …")
    ds = load_from_disk(data_path)

    tok_kw = {}
    if model_name == "deberta":
        tok_kw["add_prefix_space"] = True

    print(f"Loading tokenizer: {hf_model} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, **tok_kw)

    tokenized = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    print(f"Loading model: {hf_model} …")
    model = AutoModelForTokenClassification.from_pretrained(
        hf_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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

    print("Training …")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved → {final_dir}")

    print("\nEvaluating on test set …")
    results = trainer.evaluate(tokenized["test"])
    print(f"Test F1:        {results['eval_f1']:.4f}")
    print(f"Test Precision: {results['eval_precision']:.4f}")
    print(f"Test Recall:    {results['eval_recall']:.4f}")

    extras = {
        k: results[k]
        for k in ("epoch", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second")
        if k in results
    }
    write_test_results(
        output_dir,
        task="ate",
        domain=domain,
        model=model_name,
        metrics={
            "loss": results["eval_loss"],
            "f1": results["eval_f1"],
            "precision": results["eval_precision"],
            "recall": results["eval_recall"],
        },
        extras=extras or None,
    )


def main():
    ap = argparse.ArgumentParser(description="Train ATE (token classification)")
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
    train_ate(args.domain, args.model_name)


if __name__ == "__main__":
    main()
