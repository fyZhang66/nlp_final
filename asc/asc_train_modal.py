import modal
import json
import os

app = modal.App("asc-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "accelerate",
    )
)

data_vol = modal.Volume.from_name("asc-data", create_if_missing=True)
output_vol = modal.Volume.from_name("asc-output", create_if_missing=True)


@app.function(image=image, timeout=300)
def upload_data(local_files: dict[str, bytes]):
    """Upload local dataset files to the data volume."""
    pass


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/data": data_vol, "/output": output_vol},
)
def train():
    import numpy as np
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    MODEL_NAME = "bert-base-uncased"
    DATA_PATH = "/data/asc_data_restaurant"
    OUTPUT_DIR = "/output/asc_output_bert"
    MAX_LEN = 128
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
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
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
    preds = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    print(f"\nClassification Report:\n{report}")

    output = {
        "accuracy": results["eval_accuracy"],
        "macro_f1": results["eval_macro_f1"],
        "report": report,
    }

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    output_vol.commit()
    return output


@app.local_entrypoint()
def main():
    # Upload local dataset to Modal volume
    print("Uploading dataset to Modal volume ...")
    local_data_dir = os.path.join(os.path.dirname(__file__), "asc_data_restaurant")

    vol = modal.Volume.from_name("asc-data", create_if_missing=True)
    with vol.batch_upload() as batch:
        batch.put_directory(local_data_dir, "/asc_data_restaurant")
    print("Upload complete.")

    # Run training
    result = train.remote()
    print("\n" + "=" * 40)
    print("Training complete!")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(f"\n{result['report']}")
    print("Model saved to Modal volume 'asc-output'")
