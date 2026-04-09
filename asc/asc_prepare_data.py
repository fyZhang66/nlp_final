import xml.etree.ElementTree as ET
import random
from collections import Counter
from datasets import Dataset, DatasetDict

SEED = 42
random.seed(SEED)

TRAIN_XML = "../ate/Restaurants_Train.xml"

POLARITY_MAP = {"positive": 0, "negative": 1, "neutral": 2}


def parse_xml(xml_path: str) -> list[dict]:
    """Extract (sentence, aspect_term, polarity) triples from SemEval XML."""
    tree = ET.parse(xml_path)
    examples = []

    for sentence in tree.findall(".//sentence"):
        text_el = sentence.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        at_el = sentence.find("aspectTerms")
        if at_el is None:
            continue

        for at in at_el.findall("aspectTerm"):
            term = at.get("term", "")
            polarity = at.get("polarity", "")

            if not term or polarity not in POLARITY_MAP:
                continue  # skip "conflict" and empty labels

            examples.append({
                "sentence": text,
                "aspect": term,
                "label": POLARITY_MAP[polarity],
            })

    return examples


def main():
    print("Parsing training XML ...")
    all_examples = parse_xml(TRAIN_XML)
    print(f"Total examples (excluding 'conflict'): {len(all_examples)}")

    # Show label distribution
    labels = [ex["label"] for ex in all_examples]
    id2label = {v: k for k, v in POLARITY_MAP.items()}
    counts = Counter(labels)
    for lid, name in id2label.items():
        print(f"  {name}: {counts[lid]}")

    # Shuffle and split: 80% train / 10% val / 10% test
    random.shuffle(all_examples)
    n = len(all_examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = all_examples[:train_end]
    val_data = all_examples[train_end:val_end]
    test_data = all_examples[val_end:]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    dataset.save_to_disk("asc_data")
    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print("Saved to: asc_data/")

    # Show examples
    print("\nSample examples:")
    for ex in train_data[:3]:
        print(f"  [{id2label[ex['label']]:>8}] \"{ex['aspect']}\" in \"{ex['sentence'][:60]}...\"")


if __name__ == "__main__":
    main()
