"""End-to-end ABSA pipeline:  ATE → ASC.

Usage (run from project root)::

    python -m pipeline.run_pipeline \\
        --ate_model_dir ate/ate_output_bert/final \\
        --asc_model_dir asc/asc_output_bert/final \\
        --test_xml      data/Restaurants_Test_Gold.xml \\
        --domain        restaurant \\
        --model_name    bert \\
        --output_dir    pipeline/outputs
"""

import argparse
import json
import os
import sys

import torch

# Ensure project root is importable when invoked with ``python -m``
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ate_adapter import load_ate_model, predict_aspects_enriched
from pipeline.asc_adapter import load_asc_model, predict_sentiment_batch
from pipeline.data_utils import parse_xml_for_pipeline
from pipeline.evaluate import compute_all_metrics, print_metrics_table


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_pipeline(
    ate_model_dir: str,
    asc_model_dir: str,
    test_xml: str,
    domain: str,
    model_name: str,
    output_dir: str,
    device=None,
):
    """
    Execute the full ATE → ASC pipeline on *test_xml* and write results.

    Outputs
    -------
    ``ate_predictions_{domain}_{model}.jsonl``
        One JSON object per sentence with predicted aspect spans.
    ``e2e_predictions_{domain}_{model}.jsonl``
        Full end-to-end records including gold and predicted aspects with
        sentiments.
    ``metrics_{domain}_{model}.json``
        Aggregated metrics (Table A row).
    """
    if device is None:
        device = _get_device()
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load models ───────────────────────────────────────────────────
    print("Loading ATE model …")
    ate_model, ate_tok = load_ate_model(ate_model_dir, device)
    print("Loading ASC model …")
    asc_model, asc_tok = load_asc_model(asc_model_dir, device)

    # ── 2. Parse gold test data ──────────────────────────────────────────
    print(f"Parsing test XML: {test_xml}")
    gold_records = parse_xml_for_pipeline(test_xml, domain)
    print(f"  {len(gold_records)} sentences loaded")

    # ── 3. ATE predictions ───────────────────────────────────────────────
    print("Running ATE predictions …")
    ate_results = []
    for rec in gold_records:
        preds = predict_aspects_enriched(ate_model, ate_tok, rec["tokens"], device)
        ate_results.append({
            "sentence_id": rec["sentence_id"],
            "sentence":    rec["sentence"],
            "predicted_aspects": preds,
        })

    # ── 4. ASC on predicted aspects (batch) ──────────────────────────────
    print("Running ASC on predicted aspects …")
    pred_pairs: list[tuple[str, str]] = []
    pred_idx:   list[tuple[int, int]] = []          # (record_i, aspect_i)
    for ri, ate in enumerate(ate_results):
        for ai, asp in enumerate(ate["predicted_aspects"]):
            pred_pairs.append((ate["sentence"], asp["term"]))
            pred_idx.append((ri, ai))

    pred_sentiments = predict_sentiment_batch(asc_model, asc_tok, pred_pairs, device)
    for (ri, ai), label in zip(pred_idx, pred_sentiments):
        ate_results[ri]["predicted_aspects"][ai]["sentiment"] = label

    # ── 5. ASC on gold aspects (upper bound) ─────────────────────────────
    print("Running ASC on gold aspects …")
    gold_pairs: list[tuple[str, str]] = []
    gold_idx:   list[tuple[int, int]] = []
    for ri, rec in enumerate(gold_records):
        for ai, asp in enumerate(rec["gold_aspects"]):
            gold_pairs.append((rec["sentence"], asp["term"]))
            gold_idx.append((ri, ai))

    gold_sentiments = predict_sentiment_batch(asc_model, asc_tok, gold_pairs, device)
    gold_pred_map = {idx: label for idx, label in zip(gold_idx, gold_sentiments)}

    # ── 6. Assemble E2E records ──────────────────────────────────────────
    e2e_results = []
    for ri, rec in enumerate(gold_records):
        gold_with_pred = []
        for ai, asp in enumerate(rec["gold_aspects"]):
            gold_with_pred.append({
                "term":       asp["term"],
                "start_token": asp["start_token"],
                "end_token":   asp["end_token"],
                "sentiment":          asp["sentiment"],
                "predicted_sentiment": gold_pred_map.get((ri, ai)),
            })

        preds = []
        for asp in ate_results[ri]["predicted_aspects"]:
            preds.append({
                "term":        asp["term"],
                "start_token": asp["start_token"],
                "end_token":   asp["end_token"],
                "sentiment":       asp.get("sentiment"),
                "ate_confidence":  asp["confidence"],
            })

        e2e_results.append({
            "sentence_id":      rec["sentence_id"],
            "sentence":         rec["sentence"],
            "gold_aspects":     gold_with_pred,
            "predicted_aspects": preds,
        })

    # ── 7. Write outputs ─────────────────────────────────────────────────
    ate_path = os.path.join(output_dir,
                            f"ate_predictions_{domain}_{model_name}.jsonl")
    e2e_path = os.path.join(output_dir,
                            f"e2e_predictions_{domain}_{model_name}.jsonl")

    with open(ate_path, "w", encoding="utf-8") as f:
        for item in ate_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ATE predictions → {ate_path}")

    with open(e2e_path, "w", encoding="utf-8") as f:
        for item in e2e_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  E2E predictions → {e2e_path}")

    # ── 8. Metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    metrics = compute_all_metrics(e2e_results, gold_records)
    print_metrics_table(metrics, domain, model_name)

    met_path = os.path.join(output_dir,
                            f"metrics_{domain}_{model_name}.json")
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics → {met_path}")

    return e2e_results, metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="End-to-end ABSA Pipeline")
    ap.add_argument("--ate_model_dir", required=True)
    ap.add_argument("--asc_model_dir", required=True)
    ap.add_argument("--test_xml",      required=True)
    ap.add_argument("--domain",        required=True,
                    choices=["restaurant", "laptop"])
    ap.add_argument("--model_name",    required=True,
                    choices=["bert", "deberta"])
    ap.add_argument("--output_dir",    default="pipeline/outputs")
    args = ap.parse_args()

    run_pipeline(
        ate_model_dir=args.ate_model_dir,
        asc_model_dir=args.asc_model_dir,
        test_xml=args.test_xml,
        domain=args.domain,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
