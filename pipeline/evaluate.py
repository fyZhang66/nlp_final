"""Metrics computation for the end-to-end ABSA pipeline.

Produces all numbers required for Table A in the requirements document:
  ATE F1, ASC Acc (gold), ASC Acc (pred), ASC Macro-F1 (gold/pred),
  and the error-propagation gap.
"""

from sklearn.metrics import accuracy_score, f1_score

from pipeline.config import SENTIMENT_LABELS
from pipeline.data_utils import match_predicted_to_gold


# ── ATE ──────────────────────────────────────────────────────────────────────

def compute_ate_f1(e2e_results, gold_records):
    """Term-level exact-match precision / recall / F1."""
    tp = fp = fn = 0
    for e2e, gold_rec in zip(e2e_results, gold_records):
        pred_terms = {a["term"].lower().strip() for a in e2e["predicted_aspects"]}
        gold_terms = {a["term"].lower().strip() for a in gold_rec["gold_aspects"]}
        tp += len(pred_terms & gold_terms)
        fp += len(pred_terms - gold_terms)
        fn += len(gold_terms - pred_terms)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ── ASC on gold aspects (upper bound) ───────────────────────────────────────

def compute_asc_on_gold(e2e_results):
    """ASC accuracy / macro-F1 when the model receives *gold* aspect terms.

    Returns zeroed metrics when gold sentiment labels are unavailable
    (e.g. competition test XML without polarity attributes).
    """
    y_true, y_pred = [], []
    for rec in e2e_results:
        for asp in rec["gold_aspects"]:
            gold_sent = asp.get("sentiment")
            pred_sent = asp.get("predicted_sentiment")
            if gold_sent is not None and pred_sent is not None:
                y_true.append(gold_sent)
                y_pred.append(pred_sent)

    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro",
                                    labels=SENTIMENT_LABELS), 4),
    }


# ── ASC on predicted aspects (real performance) ─────────────────────────────

def compute_asc_on_pred(e2e_results, gold_records):
    """ASC accuracy / macro-F1 restricted to exactly-matched ATE predictions."""
    y_true, y_pred = [], []
    for e2e, gold_rec in zip(e2e_results, gold_records):
        matched, _, _ = match_predicted_to_gold(
            e2e["predicted_aspects"], gold_rec["gold_aspects"],
        )
        for pred, gold, match_type in matched:
            if match_type == "exact":
                y_true.append(gold["sentiment"])
                y_pred.append(pred["sentiment"])

    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro",
                                    labels=SENTIMENT_LABELS), 4),
    }


# ── Aggregate ────────────────────────────────────────────────────────────────

def compute_all_metrics(e2e_results, gold_records):
    ate  = compute_ate_f1(e2e_results, gold_records)
    gold = compute_asc_on_gold(e2e_results)
    pred = compute_asc_on_pred(e2e_results, gold_records)

    return {
        "ate": ate,
        "asc_gold": gold,
        "asc_pred": pred,
        "error_propagation_gap": {
            "accuracy":  round(gold["accuracy"]  - pred["accuracy"],  4),
            "macro_f1":  round(gold["macro_f1"]  - pred["macro_f1"],  4),
        },
    }


def print_metrics_table(metrics, domain, model_name):
    has_gold_sent = (metrics["asc_gold"]["accuracy"] > 0
                     or metrics["asc_pred"]["accuracy"] > 0)

    header = (f"{'Model':<10} {'Domain':<12} {'ATE F1':<10} "
              f"{'ASC Acc(g)':<12} {'ASC Acc(p)':<12} "
              f"{'ASC F1(g)':<11} {'ASC F1(p)':<11}")
    print(f"\n{header}")
    print("-" * len(header))
    print(f"{model_name:<10} {domain:<12} "
          f"{metrics['ate']['f1']:<10.4f} "
          f"{metrics['asc_gold']['accuracy']:<12.4f} "
          f"{metrics['asc_pred']['accuracy']:<12.4f} "
          f"{metrics['asc_gold']['macro_f1']:<11.4f} "
          f"{metrics['asc_pred']['macro_f1']:<11.4f}")

    if has_gold_sent:
        print(f"\nError Propagation Gap:")
        print(f"  Accuracy : {metrics['error_propagation_gap']['accuracy']:+.4f}")
        print(f"  Macro-F1 : {metrics['error_propagation_gap']['macro_f1']:+.4f}")
    else:
        print("\n  [note] Test XML lacks gold polarity labels — "
              "ASC metrics require a gold-labelled test file "
              "(e.g. Restaurants_Test_Gold.xml).")
