import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

MAX_LEN = 128
LABEL_NAMES = ["positive", "negative", "neutral"]


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        examples["aspect"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )


def predict(model, tokenized_dataset, device, batch_size=32):
    model.eval()
    all_preds = []

    for start in range(0, len(tokenized_dataset), batch_size):
        batch = tokenized_dataset[start:start + batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        token_type_ids = torch.tensor(batch["token_type_ids"]).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="asc_data_restaurant")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    print("Loading test data ...")
    ds = load_from_disk(args.data_dir)
    test_ds = ds["test"]

    tokenized_test = test_ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
    )

    y_true = np.array(test_ds["label"])

    print("Running predictions ...")
    y_pred = predict(model, tokenized_test, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f"ASC Evaluation Results")
    print(f"{'='*40}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
