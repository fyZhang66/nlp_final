import argparse
import os
import sys
from collections import Counter

from datasets import Dataset, DatasetDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.semeval_data import get_train_val_ids, load_sentences  # noqa: E402

POLARITY_MAP = {"positive": 0, "negative": 1, "neutral": 2}


def _triples_from_sentence(sent: dict) -> list[dict]:
    """(sentence, aspect, label) rows; skip conflict / missing polarity."""
    text = sent["text"]
    rows = []
    for a in sent["aspects"]:
        pol = a.get("polarity") or ""
        term = a.get("term", "")
        if not term or pol not in POLARITY_MAP:
            continue
        rows.append({
            "sentence": text,
            "aspect": term,
            "label": POLARITY_MAP[pol],
        })
    return rows


def build_asc_splits(domain: str):
    train_ids, val_ids, train_path, test_path = get_train_val_ids(domain)

    train_data, val_data = [], []
    for s in load_sentences(train_path):
        sid = s["sentence_id"]
        rows = _triples_from_sentence(s)
        if not rows:
            continue
        if sid in train_ids:
            train_data.extend(rows)
        elif sid in val_ids:
            val_data.extend(rows)

    test_data: list[dict] = []
    for s in load_sentences(test_path):
        test_data.extend(_triples_from_sentence(s))

    return train_data, val_data, test_data, train_path, test_path


def main():
    ap = argparse.ArgumentParser(description="ASC SemEval data → HuggingFace Dataset")
    ap.add_argument(
        "--domain",
        choices=["restaurant", "laptop"],
        default="restaurant",
        help="restaurant → asc_data_restaurant; laptop → asc_data_laptop",
    )
    args = ap.parse_args()

    domain = args.domain
    name = "asc_data_restaurant" if domain == "restaurant" else "asc_data_laptop"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    print(f"Domain: {domain}  →  {name}/")
    train_data, val_data, test_data, train_path, test_path = build_asc_splits(domain)
    print(f"Train XML: {train_path}")
    print(f"Test XML:  {test_path}")

    print(f"\nTotal aspect examples — train pool (excluding conflict): {len(train_data) + len(val_data)}")
    labels = [ex["label"] for ex in train_data + val_data]
    id2label = {v: k for k, v in POLARITY_MAP.items()}
    counts = Counter(labels)
    for lid, name in id2label.items():
        print(f"  {name}: {counts[lid]}")

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    dataset.save_to_disk(out_dir)
    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Saved to: {out_dir}/")

    print("\nSample examples:")
    for ex in train_data[:3]:
        print(f"  [{id2label[ex['label']]:>8}] \"{ex['aspect']}\" in \"{ex['sentence'][:60]}...\"")


if __name__ == "__main__":
    main()
