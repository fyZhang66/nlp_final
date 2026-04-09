import xml.etree.ElementTree as ET
import random
from datasets import Dataset, DatasetDict

SEED = 42
random.seed(SEED)

TRAIN_XML = "Restaurants_Train.xml"
TEST_XML  = "Restaurants_Test.xml"


def tokenize_and_bio(sentence_text: str, aspect_terms: list[dict]) -> tuple[list, list]:
    """
    Simple whitespace tokenizer + BIO labeling.
    aspect_terms: list of {"term": str, "from": int, "to": int}
    Returns: (tokens, bio_tags)
    """
    # Build character-level aspect span set
    aspect_chars = {}
    for asp in aspect_terms:
        for i in range(int(asp["from"]), int(asp["to"])):
            aspect_chars[i] = asp["term"]

    # Tokenize by whitespace, tracking char positions
    tokens, tags = [], []
    i = 0
    text = sentence_text

    while i < len(text):
        # Skip spaces
        if text[i] == " ":
            i += 1
            continue

        # Find end of token
        j = i
        while j < len(text) and text[j] != " ":
            j += 1

        token = text[i:j]
        token_chars = set(range(i, j))

        # Check overlap with aspect spans
        overlap = token_chars & set(aspect_chars.keys())
        if overlap:
            # BIO fix:
            # If the previous character is not inside the same aspect span,
            # this token starts a new aspect -> B-ASP
            # Otherwise it continues the current aspect -> I-ASP
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


def parse_xml(xml_path: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    examples = []

    for sentence in root.findall(".//sentence"):
        text_el = sentence.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()

        aspect_terms = []
        at_el = sentence.find("aspectTerms")
        if at_el is not None:
            for at in at_el.findall("aspectTerm"):
                term = at.get("term", "")
                frm = at.get("from", "0")
                to = at.get("to", "0")
                if term:
                    aspect_terms.append({"term": term, "from": frm, "to": to})

        tokens, tags = tokenize_and_bio(text, aspect_terms)
        if tokens:
            examples.append({"tokens": tokens, "tags": tags})

    return examples


def main():
    print("Parsing training XML ...")
    train_all = parse_xml(TRAIN_XML)

    print("Parsing test XML ...")
    test_data = parse_xml(TEST_XML)

    # Split train -> 90% train / 10% val
    random.shuffle(train_all)
    split = int(0.9 * len(train_all))
    train_data = train_all[:split]
    val_data = train_all[split:]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    dataset.save_to_disk("ate_data")
    print(dataset)
    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print("Saved to: ate_data/")

    # Show one example
    ex = train_data[0]
    print("\nExample:")
    for tok, tag in zip(ex["tokens"], ex["tags"]):
        print(f"  {tok:20s} {tag}")


if __name__ == "__main__":
    main()
