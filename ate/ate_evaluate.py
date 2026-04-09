import argparse
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

LABEL_LIST = ["O", "B-ASP", "I-ASP"]
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
MAX_LEN    = 128


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
        labels, prev_word = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word:
                labels.append(LABEL2ID.get(tags[wid], 0))
            else:
                lbl = LABEL2ID.get(tags[wid], 0)
                labels.append(lbl if lbl == LABEL2ID["I-ASP"] else -100)
            prev_word = wid
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


def predict(model, tokenizer, dataset, device, batch_size=32):
    model.eval()
    y_true, y_pred = [], []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        input_ids      = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels         = torch.tensor(batch["labels"])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        for pred_seq, label_seq in zip(preds, labels.numpy()):
            true_tags, pred_tags = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_tags.append(ID2LABEL[l])
                pred_tags.append(ID2LABEL[p])
            y_true.append(true_tags)
            y_pred.append(pred_tags)

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir",  default="ate_data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)

    print("Loading test data …")
    ds = load_from_disk(args.data_dir)
    tokenized_test = ds["test"].map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=ds["test"].column_names,
    )

    print("Running predictions …")
    y_true, y_pred = predict(model, tokenizer, tokenized_test, device)

    p      = precision_score(y_true, y_pred)
    r      = recall_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f"ATE Evaluation Results")
    print(f"{'='*40}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    results = {"precision": p, "recall": r, "f1": f1, "report": report}
    out_path = f"{args.model_dir}/ate_test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()