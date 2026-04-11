"""Shared SemEval 2014 XML paths, sentence-level train/val split, and loaders.

ATE 与 ASC 共用同一套 *sentence_id* 划分：仅从官方 Train XML 划 train/val，
官方 Test Gold 作为 test。数据文件位于项目根目录 ``data/``。"""
from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Official v2 train + gold test (downloaded under data/)
# ---------------------------------------------------------------------------
DATA_XML: dict[str, dict[str, str]] = {
    "restaurant": {
        "train": os.path.join(PROJECT_ROOT, "data", "Restaurants_Train_v2.xml"),
        "test": os.path.join(PROJECT_ROOT, "data", "Restaurants_Test_Gold.xml"),
    },
    "laptop": {
        "train": os.path.join(PROJECT_ROOT, "data", "Laptop_Train_v2.xml"),
        "test": os.path.join(PROJECT_ROOT, "data", "Laptops_Test_Gold.xml"),
    },
}

VAL_RATIO = 0.1
SEED = 42

def train_val_split_ids(sentence_ids: list[str], val_ratio: float = VAL_RATIO, seed: int = SEED):
    """Split *sentence_ids* into disjoint train / val sets (same seed → same split)."""
    random.seed(seed)
    ids = list(sentence_ids)
    random.shuffle(ids)
    if not ids:
        return set(), set()
    n_val = max(1, int(len(ids) * val_ratio))
    if n_val >= len(ids):
        n_val = len(ids) - 1
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def _parse_sentence_el(sent_el) -> dict | None:
    sid = sent_el.get("id", "")
    text_el = sent_el.find("text")
    if text_el is None or not text_el.text:
        return None
    text = text_el.text.strip()
    aspects: list[dict] = []
    at_el = sent_el.find("aspectTerms")
    if at_el is not None:
        for at in at_el.findall("aspectTerm"):
            term = at.get("term", "")
            if not term:
                continue
            aspects.append({
                "term": term,
                "from": at.get("from", "0"),
                "to": at.get("to", "0"),
                "polarity": at.get("polarity"),
            })
    return {"sentence_id": sid, "text": text, "aspects": aspects}


def load_sentences(xml_path: str) -> list[dict]:
    """All sentences in file order: ``sentence_id``, ``text``, ``aspects``."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[dict] = []
    for sent_el in root.findall(".//sentence"):
        rec = _parse_sentence_el(sent_el)
        if rec is not None:
            out.append(rec)
    return out


def get_train_val_ids(domain: str) -> tuple[set[str], set[str], str, str]:
    """Return ``train_ids``, ``val_ids``, ``train_xml``, ``test_xml`` paths."""
    paths = DATA_XML[domain]
    train_path, test_path = paths["train"], paths["test"]
    sents = load_sentences(train_path)
    ids = [s["sentence_id"] for s in sents]
    train_ids, val_ids = train_val_split_ids(ids)
    return train_ids, val_ids, train_path, test_path
