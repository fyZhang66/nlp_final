"""Utilities for parsing SemEval XML and matching predicted / gold aspects."""

import xml.etree.ElementTree as ET


# ── XML parsing ──────────────────────────────────────────────────────────────

def whitespace_tokenize(text: str) -> list[dict]:
    """Tokenize by whitespace, returning each token with its char span."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            i += 1
            continue
        j = i
        while j < len(text) and text[j] != " ":
            j += 1
        tokens.append({"text": text[i:j], "char_start": i, "char_end": j})
        i = j
    return tokens


def _char_span_to_token_span(token_info, char_from, char_to):
    """Map a character-level [from, to) span to inclusive token indices."""
    start_tok, end_tok = None, None
    for ti, t in enumerate(token_info):
        if t["char_end"] > char_from and t["char_start"] < char_to:
            if start_tok is None:
                start_tok = ti
            end_tok = ti
    if start_tok is None:
        return 0, 0
    return start_tok, end_tok


def parse_xml_for_pipeline(xml_path: str, domain: str = "rest") -> list[dict]:
    """
    Parse SemEval XML into pipeline-ready records.

    Each record contains the sentence text, whitespace tokens, and gold
    aspect terms with token-level spans and sentiment labels.  Aspects
    with ``polarity="conflict"`` are skipped (consistent with the ASC
    module).

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``sentence_id``, ``sentence``, ``tokens``,
        ``gold_aspects`` (list of ``{term, start_token, end_token, sentiment}``).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for idx, sent_el in enumerate(root.findall(".//sentence")):
        sid = sent_el.get("id", f"{domain}_test_{idx:04d}")
        text_el = sent_el.find("text")
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()
        token_info = whitespace_tokenize(text)
        tokens = [t["text"] for t in token_info]

        gold_aspects = []
        at_el = sent_el.find("aspectTerms")
        if at_el is not None:
            for at in at_el.findall("aspectTerm"):
                term = at.get("term", "")
                polarity = at.get("polarity")      # None when attr absent
                frm = int(at.get("from", "0"))
                to = int(at.get("to", "0"))
                if not term or polarity == "conflict":
                    continue
                s_tok, e_tok = _char_span_to_token_span(token_info, frm, to)
                gold_aspects.append({
                    "term": term,
                    "start_token": s_tok,
                    "end_token": e_tok,
                    "sentiment": polarity,          # None if test XML lacks labels
                })

        records.append({
            "sentence_id": sid,
            "sentence": text,
            "tokens": tokens,
            "gold_aspects": gold_aspects,
        })

    return records


# ── Aspect matching ──────────────────────────────────────────────────────────

def match_predicted_to_gold(predicted_aspects, gold_aspects):
    """
    Two-pass matching: exact string match first, then token-overlap match.

    Returns
    -------
    matched : list[(pred_dict, gold_dict, match_type)]
        ``match_type`` is ``"exact"`` or ``"boundary"``.
    missing : list[gold_dict]
        Gold aspects with no matching prediction.
    spurious : list[pred_dict]
        Predicted aspects with no matching gold.
    """
    used_gold = set()
    used_pred = set()
    matched = []

    # Pass 1 — exact (case-insensitive)
    for pi, pred in enumerate(predicted_aspects):
        for gi, gold in enumerate(gold_aspects):
            if gi in used_gold:
                continue
            if pred["term"].lower().strip() == gold["term"].lower().strip():
                matched.append((pred, gold, "exact"))
                used_gold.add(gi)
                used_pred.add(pi)
                break

    # Pass 2 — boundary overlap on token indices
    for pi, pred in enumerate(predicted_aspects):
        if pi in used_pred:
            continue
        for gi, gold in enumerate(gold_aspects):
            if gi in used_gold:
                continue
            pred_toks = set(range(pred.get("start_token", 0),
                                  pred.get("end_token", 0) + 1))
            gold_toks = set(range(gold.get("start_token", 0),
                                  gold.get("end_token", 0) + 1))
            if pred_toks & gold_toks:
                matched.append((pred, gold, "boundary"))
                used_gold.add(gi)
                used_pred.add(pi)
                break

    missing = [gold_aspects[i] for i in range(len(gold_aspects))
               if i not in used_gold]
    spurious = [predicted_aspects[i] for i in range(len(predicted_aspects))
                if i not in used_pred]

    return matched, missing, spurious
