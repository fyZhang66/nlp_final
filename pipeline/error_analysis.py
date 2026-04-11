"""Confusion matrix, ATE span-level error classification, and E2E error tracing.

Usage (standalone, after ``run_pipeline`` has produced E2E predictions)::

    python -m pipeline.error_analysis \\
        --e2e_predictions pipeline/outputs/e2e_predictions_restaurant_bert.jsonl \\
        --test_xml        ate/Restaurants_Test.xml \\
        --domain          restaurant \\
        --model_name      bert \\
        --output_dir      pipeline/outputs
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_utils import parse_xml_for_pipeline, match_predicted_to_gold

# Row / column order that matches the requirements document (Pos, Neu, Neg)
CM_LABELS = ["positive", "neutral", "negative"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _align_by_id(e2e_results, gold_records):
    """Return gold_records re-ordered to match e2e_results by sentence_id."""
    gold_map = {r["sentence_id"]: r for r in gold_records}
    aligned = []
    for e in e2e_results:
        g = gold_map.get(e["sentence_id"])
        if g is not None:
            aligned.append(g)
        else:
            aligned.append({
                "sentence_id": e["sentence_id"],
                "sentence": e["sentence"],
                "tokens": e["sentence"].split(),
                "gold_aspects": e.get("gold_aspects", []),
            })
    return aligned


# ── 1. ASC confusion matrix ─────────────────────────────────────────────────

def generate_asc_confusion_matrix(e2e_results, output_dir, domain, model_name):
    """3×3 confusion matrix on gold aspects (absolute + row-normalised)."""
    y_true, y_pred = [], []
    for rec in e2e_results:
        for asp in rec["gold_aspects"]:
            gold = asp.get("sentiment")
            pred = asp.get("predicted_sentiment")
            if gold is not None and pred is not None:
                y_true.append(gold)
                y_pred.append(pred)

    if not y_true:
        print("  [skip] no gold ASC predictions available")
        return None

    cm = confusion_matrix(y_true, y_pred, labels=CM_LABELS)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    os.makedirs(output_dir, exist_ok=True)

    def _write_csv(matrix, path, fmt):
        header = "," + ",".join(f"Pred_{l}" for l in CM_LABELS)
        with open(path, "w") as f:
            f.write(header + "\n")
            for i, label in enumerate(CM_LABELS):
                vals = ",".join(fmt(v) for v in matrix[i])
                f.write(f"Actual_{label},{vals}\n")

    abs_path = os.path.join(output_dir,
                            f"confusion_matrix_{domain}_{model_name}_abs.csv")
    _write_csv(cm, abs_path, str)

    norm_path = os.path.join(output_dir,
                             f"confusion_matrix_{domain}_{model_name}_norm.csv")
    _write_csv(cm_norm, norm_path, lambda v: f"{v:.4f}")

    # Pretty-print
    print(f"\n  ASC Confusion Matrix ({domain} / {model_name}):")
    col_w = 10
    print(f"  {'':>14}", end="")
    for l in CM_LABELS:
        print(f"{'P_' + l:>{col_w}}", end="")
    print()
    for i, label in enumerate(CM_LABELS):
        print(f"  {'A_' + label:>14}", end="")
        for j in range(len(CM_LABELS)):
            print(f"{cm[i][j]:>{col_w}}", end="")
        print()

    print(f"\n  → Absolute  CSV: {abs_path}")
    print(f"  → Normalised CSV: {norm_path}")

    return {"absolute": cm.tolist(), "normalized": cm_norm.tolist(),
            "labels": CM_LABELS}


# ── 2. ATE span-level error classification ──────────────────────────────────

def classify_ate_errors(e2e_results, gold_records):
    """
    Classify every gold / predicted aspect into one of:
    Correct, Missing, Spurious, Boundary Error.
    """
    stats = Counter()
    examples: dict[str, list] = {k: [] for k in
                                 ("correct", "missing", "spurious", "boundary")}

    for e2e, gold_rec in zip(e2e_results, gold_records):
        matched, missing, spurious = match_predicted_to_gold(
            e2e["predicted_aspects"], gold_rec["gold_aspects"],
        )
        for pred, gold, mtype in matched:
            key = "correct" if mtype == "exact" else "boundary"
            stats[key] += 1
            examples[key].append({
                "sentence": e2e["sentence"],
                "gold": gold["term"], "pred": pred["term"],
            })
        for g in missing:
            stats["missing"] += 1
            examples["missing"].append({
                "sentence": e2e["sentence"], "gold": g["term"],
            })
        for s in spurious:
            stats["spurious"] += 1
            examples["spurious"].append({
                "sentence": e2e["sentence"], "pred": s["term"],
            })

    total = sum(stats.values()) or 1
    rows = []
    for key in ("correct", "missing", "spurious", "boundary"):
        c = stats[key]
        rows.append({
            "error_type": key.title(),
            "count": c,
            "percentage": round(c / total * 100, 2),
            "examples": examples[key][:3],
        })
    return rows


# ── 3. E2E error tracing ────────────────────────────────────────────────────

_TRACE_CATEGORIES = [
    ("ate_miss",              "ATE Miss → Sentiment Lost"),
    ("ate_boundary_asc_wrong", "ATE Boundary Error → Sentiment Wrong"),
    ("ate_correct_asc_wrong",  "ATE Correct → ASC Wrong"),
    ("ate_spurious",           "ATE Spurious → False Positive"),
]


def trace_e2e_errors(e2e_results, gold_records):
    """Attribute every end-to-end error to a pipeline stage."""
    counts = Counter()
    examples: dict[str, list] = {k: [] for k, _ in _TRACE_CATEGORIES}

    for e2e, gold_rec in zip(e2e_results, gold_records):
        matched, missing, spurious = match_predicted_to_gold(
            e2e["predicted_aspects"], gold_rec["gold_aspects"],
        )

        for g in missing:
            counts["ate_miss"] += 1
            examples["ate_miss"].append({
                "sentence_id": e2e["sentence_id"],
                "sentence":    e2e["sentence"],
                "gold":  {"term": g["term"],
                          "sentiment": g.get("sentiment")},
                "error_type": "ATE Miss → Sentiment Lost",
            })

        for s in spurious:
            counts["ate_spurious"] += 1
            examples["ate_spurious"].append({
                "sentence_id": e2e["sentence_id"],
                "sentence":    e2e["sentence"],
                "predicted": {"term": s["term"],
                              "sentiment": s.get("sentiment", "unknown")},
                "error_type": "ATE Spurious → False Positive",
            })

        for pred, gold, mtype in matched:
            gold_sent = gold.get("sentiment")
            pred_sent = pred.get("sentiment")
            # Skip sentiment comparison when gold labels are absent
            if gold_sent is None:
                continue
            if mtype == "exact" and pred_sent != gold_sent:
                counts["ate_correct_asc_wrong"] += 1
                examples["ate_correct_asc_wrong"].append({
                    "sentence_id": e2e["sentence_id"],
                    "sentence":    e2e["sentence"],
                    "gold":      {"term": gold["term"],
                                  "sentiment": gold_sent},
                    "predicted": {"term": pred["term"],
                                  "sentiment": pred_sent},
                    "error_type": "ATE Correct → ASC Wrong",
                })
            elif mtype == "boundary" and pred_sent != gold_sent:
                counts["ate_boundary_asc_wrong"] += 1
                examples["ate_boundary_asc_wrong"].append({
                    "sentence_id": e2e["sentence_id"],
                    "sentence":    e2e["sentence"],
                    "gold":      {"term": gold["term"],
                                  "sentiment": gold_sent},
                    "predicted": {"term": pred["term"],
                                  "sentiment": pred_sent},
                    "error_type": "ATE Boundary Error → Sentiment Wrong",
                })

    total = sum(counts.values()) or 1
    rows = []
    for key, label in _TRACE_CATEGORIES:
        c = counts[key]
        rows.append({
            "category": label,
            "count": c,
            "percentage": round(c / total * 100, 2),
            "examples": examples[key][:3],
        })
    return rows


# ── 4. Error examples JSONL ─────────────────────────────────────────────────

def generate_error_examples(e2e_results, gold_records, output_dir,
                            domain, model_name):
    """Write ``error_examples_{domain}_{model}.jsonl``."""
    tracing = trace_e2e_errors(e2e_results, gold_records)
    all_ex = []
    for cat in tracing:
        all_ex.extend(cat["examples"])

    path = os.path.join(output_dir,
                        f"error_examples_{domain}_{model_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ex in all_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Error examples → {path}  ({len(all_ex)} cases)")
    return all_ex


# ── orchestrator ─────────────────────────────────────────────────────────────

def run_full_error_analysis(e2e_path, test_xml, domain, model_name, output_dir):
    """Execute all error-analysis steps on an existing E2E predictions file."""
    print(f"\n{'=' * 60}")
    print(f"Error Analysis: {domain} / {model_name}")
    print(f"{'=' * 60}")

    e2e_results  = _load_jsonl(e2e_path)
    gold_records = parse_xml_for_pipeline(test_xml, domain)
    gold_records = _align_by_id(e2e_results, gold_records)

    os.makedirs(output_dir, exist_ok=True)

    # 1 — Confusion matrix
    print("\n[1] ASC Confusion Matrix")
    cm = generate_asc_confusion_matrix(e2e_results, output_dir, domain,
                                       model_name)

    # 2 — ATE span-level errors
    print("\n[2] ATE Span-level Error Classification")
    ate_errs = classify_ate_errors(e2e_results, gold_records)
    print(f"\n  {'Type':<18} {'Count':>6} {'%':>8}  Example")
    print("  " + "-" * 64)
    for row in ate_errs:
        ex = ""
        if row["examples"]:
            e = row["examples"][0]
            parts = []
            if "gold" in e:
                parts.append(f"gold=\"{e['gold']}\"")
            if "pred" in e:
                parts.append(f"pred=\"{e['pred']}\"")
            ex = ", ".join(parts)
        print(f"  {row['error_type']:<18} {row['count']:>6} "
              f"{row['percentage']:>7.1f}%  {ex}")

    ate_path = os.path.join(output_dir,
                            f"ate_error_stats_{domain}_{model_name}.json")
    with open(ate_path, "w", encoding="utf-8") as f:
        json.dump(ate_errs, f, indent=2, ensure_ascii=False)

    # 3 — E2E error tracing
    print("\n[3] End-to-End Error Tracing")
    e2e_errs = trace_e2e_errors(e2e_results, gold_records)
    print(f"\n  {'Category':<42} {'Count':>6} {'%':>8}")
    print("  " + "-" * 56)
    for row in e2e_errs:
        print(f"  {row['category']:<42} {row['count']:>6} "
              f"{row['percentage']:>7.1f}%")

    e2e_err_path = os.path.join(output_dir,
                                f"e2e_error_tracing_{domain}_{model_name}.json")
    with open(e2e_err_path, "w", encoding="utf-8") as f:
        json.dump(e2e_errs, f, indent=2, ensure_ascii=False)

    # 4 — Error example file
    print("\n[4] Error Examples")
    generate_error_examples(e2e_results, gold_records, output_dir,
                            domain, model_name)

    return {"confusion_matrix": cm, "ate_errors": ate_errs,
            "e2e_errors": e2e_errs}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ABSA Error Analysis")
    ap.add_argument("--e2e_predictions", required=True)
    ap.add_argument("--test_xml",        required=True)
    ap.add_argument("--domain",          required=True,
                    choices=["restaurant", "laptop"])
    ap.add_argument("--model_name",      required=True,
                    choices=["bert", "deberta"])
    ap.add_argument("--output_dir",      default="pipeline/outputs")
    args = ap.parse_args()

    run_full_error_analysis(
        e2e_path=args.e2e_predictions,
        test_xml=args.test_xml,
        domain=args.domain,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
