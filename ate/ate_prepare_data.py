import argparse
import os
import sys

from datasets import Dataset, DatasetDict

# Project root on path for ``pipeline.semeval_data``
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.semeval_data import get_train_val_ids, load_sentences  # noqa: E402


def tokenize_and_bio(sentence_text: str, aspect_terms: list[dict]) -> tuple[list, list]:
    """
    Simple whitespace tokenizer + BIO labeling.
    aspect_terms: list of {"term": str, "from": int, "to": int}
    Returns: (tokens, bio_tags)
    """
    aspect_chars: dict[int, str] = {}
    for asp in aspect_terms:
        for i in range(int(asp["from"]), int(asp["to"])):
            aspect_chars[i] = asp["term"]

    tokens, tags = [], []
    text = sentence_text
    i = 0

    while i < len(text):
        if text[i] == " ":
            i += 1
            continue
        j = i
        while j < len(text) and text[j] != " ":
            j += 1

        token = text[i:j]
        token_chars = set(range(i, j))
        overlap = token_chars & set(aspect_chars.keys())
        if overlap:
            min_char = min(overlap)
            if (min_char - 1) not in aspect_chars:
                tags.append("B-ASP")
            else:
                tags.append("I-ASP")
        else:
            tags.append("O")

        tokens.append(token)
        i = j

    return tokens, tags


def _sentence_to_ate_example(sent: dict) -> dict | None:
    aspect_terms = [
        {"term": a["term"], "from": a["from"], "to": a["to"]}
        for a in sent["aspects"]
    ]
    tokens, tags = tokenize_and_bio(sent["text"], aspect_terms)
    if not tokens:
        return None
    return {"tokens": tokens, "tags": tags}


def build_ate_splits(domain: str):
    train_ids, val_ids, train_path, test_path = get_train_val_ids(domain)
    train_sents = load_sentences(train_path)
    test_sents = load_sentences(test_path)

    train_data, val_data = [], []
    for s in train_sents:
        sid = s["sentence_id"]
        ex = _sentence_to_ate_example(s)
        if ex is None:
            continue
        if sid in train_ids:
            train_data.append(ex)
        elif sid in val_ids:
            val_data.append(ex)

    test_data = []
    for s in test_sents:
        ex = _sentence_to_ate_example(s)
        if ex is not None:
            test_data.append(ex)

    return train_data, val_data, test_data, train_path, test_path


def main():
    ap = argparse.ArgumentParser(description="ATE SemEval data → HuggingFace Dataset")
    ap.add_argument(
        "--domain",
        choices=["restaurant", "laptop"],
        default="restaurant",
        help="restaurant → ate_data_restaurant; laptop → ate_data_laptop",
    )
    args = ap.parse_args()

    domain = args.domain
    name = "ate_data_restaurant" if domain == "restaurant" else "ate_data_laptop"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    print(f"Domain: {domain}  →  {name}/")
    train_data, val_data, test_data, train_path, test_path = build_ate_splits(domain)
    print(f"Train XML: {train_path}")
    print(f"Test XML:  {test_path}")

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    dataset.save_to_disk(out_dir)
    print(dataset)
    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Saved to: {out_dir}/")

    ex = train_data[0]
    print("\nExample:")
    for tok, tag in zip(ex["tokens"], ex["tags"]):
        print(f"  {tok:20s} {tag}")


if __name__ == "__main__":
    main()
