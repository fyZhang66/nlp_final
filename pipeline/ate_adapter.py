"""ATE model wrapper that produces enriched output (token spans + confidence).

Re-implements the core prediction logic from ``ate/ate_predict.py`` without
modifying that file, adding ``start_token``, ``end_token``, and ``confidence``
fields required by the pipeline specification.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from pipeline.config import ATE_LABELS, MAX_LEN

ID2LABEL = {i: l for i, l in enumerate(ATE_LABELS)}


def load_ate_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def predict_aspects_enriched(model, tokenizer, tokens, device):
    """
    Predict aspect terms with token-level span indices and confidence.

    Parameters
    ----------
    tokens : list[str]
        Whitespace-split word tokens of a single sentence.

    Returns
    -------
    list[dict]
        Each dict: ``{term, start_token, end_token, confidence}``.
        ``confidence`` is the mean softmax probability of the predicted BIO
        label across constituent tokens.
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits[0]          # (seq_len, num_labels)

    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = logits.argmax(dim=-1).cpu().tolist()
    word_ids = enc.word_ids(batch_index=0)

    # Map subword predictions → word-level (first subword wins)
    word_preds: dict[int, str] = {}
    word_confs: dict[int, float] = {}
    prev_wid = None
    for idx, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        label_id = preds[idx]
        word_preds[wid] = ID2LABEL[label_id]
        word_confs[wid] = float(probs[idx, label_id])
        prev_wid = wid

    # Decode BIO → aspect spans
    aspects: list[dict] = []
    cur_toks: list[str] = []
    cur_start: int | None = None
    cur_confs: list[float] = []

    def _flush():
        if cur_toks:
            aspects.append({
                "term": " ".join(cur_toks),
                "start_token": cur_start,
                "end_token": cur_start + len(cur_toks) - 1,
                "confidence": round(sum(cur_confs) / len(cur_confs), 4),
            })

    for wid in range(len(tokens)):
        label = word_preds.get(wid, "O")
        conf = word_confs.get(wid, 0.0)

        if label == "B-ASP":
            _flush()
            cur_toks = [tokens[wid]]
            cur_start = wid
            cur_confs = [conf]
        elif label == "I-ASP" and cur_toks:
            cur_toks.append(tokens[wid])
            cur_confs.append(conf)
        elif label == "I-ASP":
            cur_toks = [tokens[wid]]
            cur_start = wid
            cur_confs = [conf]
        else:
            _flush()
            cur_toks = []
            cur_start = None
            cur_confs = []

    _flush()
    return aspects
