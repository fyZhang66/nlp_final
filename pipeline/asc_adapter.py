"""ASC model wrapper providing the ``predict(sentence, aspect) → str`` interface
required by the pipeline specification.

Re-uses the same tokenization strategy as ``asc/asc_evaluate.py`` (sentence–
aspect pair encoding) without modifying that file.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline.config import SENTIMENT_LABELS, MAX_LEN


def load_asc_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def predict_sentiment(model, tokenizer, sentence: str, aspect: str, device) -> str:
    """Single-pair inference: returns ``"positive"`` / ``"negative"`` / ``"neutral"``."""
    enc = tokenizer(
        sentence, aspect,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    return SENTIMENT_LABELS[logits.argmax(dim=-1).item()]


def predict_sentiment_batch(
    model, tokenizer, pairs: list[tuple[str, str]], device, batch_size: int = 32
) -> list[str]:
    """
    Batch inference for a list of ``(sentence, aspect_term)`` pairs.

    Returns a list of sentiment label strings in the same order.
    """
    if not pairs:
        return []

    all_preds: list[str] = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        sentences = [p[0] for p in batch_pairs]
        aspects = [p[1] for p in batch_pairs]

        enc = tokenizer(
            sentences, aspects,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        ids = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(SENTIMENT_LABELS[i] for i in ids)

    return all_preds
