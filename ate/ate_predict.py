import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import argparse
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_LIST = ["O", "B-ASP", "I-ASP"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
MAX_LEN = 128


def predict_aspects(model, tokenizer, tokens, device):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)

    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    word_ids = enc.word_ids(batch_index=0)

    word_preds = {}
    prev_wid = None
    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        word_preds[wid] = ID2LABEL[preds[idx]]
        prev_wid = wid

    aspects = []
    cur_asp = []

    for wid in range(len(tokens)):
        label = word_preds.get(wid, "O")

        if label == "B-ASP":
            if cur_asp:
                aspects.append(" ".join(cur_asp))
            cur_asp = [tokens[wid]]
        elif label == "I-ASP":
            if cur_asp:
                cur_asp.append(tokens[wid])
            else:
                cur_asp = [tokens[wid]]
        else:
            if cur_asp:
                aspects.append(" ".join(cur_asp))
            cur_asp = []

    if cur_asp:
        aspects.append(" ".join(cur_asp))

    return aspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="ate_data_restaurant")
    parser.add_argument("--output", default="ate_predictions.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    print("Loading test data ...")
    ds = load_from_disk(args.data_dir)["test"]

    results = []
    for ex in ds:
        sentence = " ".join(ex["tokens"])
        aspects = predict_aspects(model, tokenizer, ex["tokens"], device)

        for asp in aspects:
            results.append({
                "sentence": sentence,
                "aspect": asp
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(results)} (sentence, aspect) pairs saved -> {args.output}")
    print("\nSample output:")
    for item in results[:3]:
        print(item)


if __name__ == "__main__":
    main()