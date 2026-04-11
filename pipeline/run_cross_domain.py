"""Cross-domain experiment matrix (8 experiments).

Usage (run from project root)::

    python -m pipeline.run_cross_domain
    python -m pipeline.run_cross_domain --output_dir pipeline/outputs

The script automatically skips experiments whose data or models are not yet
available, so it is safe to run at any stage of the project.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import RAW_DATA, MODELS, OUTPUT_DIR
from pipeline.run_pipeline import run_pipeline
from pipeline.error_analysis import run_full_error_analysis

# (id, train_domain, test_domain, model, experiment_type)
EXPERIMENTS = [
    (1, "restaurant", "restaurant", "bert",    "in-domain"),
    (2, "restaurant", "restaurant", "deberta", "in-domain"),
    (3, "restaurant", "laptop",     "bert",    "cross-domain"),
    (4, "restaurant", "laptop",     "deberta", "cross-domain"),
    (5, "laptop",     "restaurant", "bert",    "cross-domain"),
    (6, "laptop",     "restaurant", "deberta", "cross-domain"),
    (7, "laptop",     "laptop",     "bert",    "in-domain"),
    (8, "laptop",     "laptop",     "deberta", "in-domain"),
]


def _check_resources(train_dom, test_dom, model_name):
    """Return list of missing resource descriptions (empty == all OK)."""
    missing = []
    test_xml = RAW_DATA.get(test_dom, {}).get("test_xml", "")
    ate_dir  = MODELS.get(train_dom, {}).get(model_name, {}).get("ate", "")
    asc_dir  = MODELS.get(train_dom, {}).get(model_name, {}).get("asc", "")

    if not os.path.exists(test_xml):
        missing.append(f"test XML  : {test_xml}")
    if not os.path.isdir(ate_dir):
        missing.append(f"ATE model : {ate_dir}")
    if not os.path.isdir(asc_dir):
        missing.append(f"ASC model : {asc_dir}")
    return missing, test_xml, ate_dir, asc_dir


def run_all_experiments(output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    all_metrics = []
    skipped = []

    for exp_id, train_dom, test_dom, model_name, exp_type in EXPERIMENTS:
        tag = f"{train_dom[:4]}2{test_dom[:4]}_{model_name}"

        print(f"\n{'#' * 60}")
        print(f"  Experiment {exp_id}: {train_dom} → {test_dom} "
              f"({model_name})  [{exp_type}]")
        print(f"{'#' * 60}")

        missing, test_xml, ate_dir, asc_dir = _check_resources(
            train_dom, test_dom, model_name)

        if missing:
            print("  SKIPPED — missing resources:")
            for m in missing:
                print(f"    • {m}")
            skipped.append(exp_id)
            continue

        exp_out = os.path.join(output_dir, tag)
        try:
            _, metrics = run_pipeline(
                ate_model_dir=ate_dir,
                asc_model_dir=asc_dir,
                test_xml=test_xml,
                domain=test_dom,
                model_name=model_name,
                output_dir=exp_out,
            )
            metrics["experiment"] = {
                "id": exp_id,
                "train_domain": train_dom,
                "test_domain":  test_dom,
                "model":        model_name,
                "type":         exp_type,
            }
            all_metrics.append(metrics)

            # Error analysis
            e2e_path = os.path.join(
                exp_out, f"e2e_predictions_{test_dom}_{model_name}.jsonl")
            if os.path.exists(e2e_path):
                run_full_error_analysis(
                    e2e_path=e2e_path,
                    test_xml=test_xml,
                    domain=test_dom,
                    model_name=model_name,
                    output_dir=exp_out,
                )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            skipped.append(exp_id)

    # ── summary tables ───────────────────────────────────────────────────
    if all_metrics:
        _print_summary(all_metrics)

    summary_path = os.path.join(output_dir, "cross_domain_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSummary → {summary_path}")

    if skipped:
        print(f"Skipped experiments: {skipped}")

    return all_metrics


# ── pretty-print ─────────────────────────────────────────────────────────────

def _print_summary(all_metrics):
    # Table A
    print(f"\n{'=' * 90}")
    print("Table A  —  End-to-End Pipeline Results")
    print(f"{'=' * 90}")
    hdr = (f"{'ID':<4} {'Train→Test':<18} {'Model':<8} {'ATE F1':<9} "
           f"{'ASC Acc(g)':<11} {'ASC Acc(p)':<11} "
           f"{'ASC F1(g)':<11} {'ASC F1(p)':<11}")
    print(hdr)
    print("-" * len(hdr))
    for m in all_metrics:
        e = m["experiment"]
        arrow = f"{e['train_domain'][:4]}→{e['test_domain'][:4]}"
        print(f"{e['id']:<4} {arrow:<18} {e['model']:<8} "
              f"{m['ate']['f1']:<9.4f} "
              f"{m['asc_gold']['accuracy']:<11.4f} "
              f"{m['asc_pred']['accuracy']:<11.4f} "
              f"{m['asc_gold']['macro_f1']:<11.4f} "
              f"{m['asc_pred']['macro_f1']:<11.4f}")

    # Table B — Cross-Domain ATE
    in_domain_ate: dict[tuple, float] = {}
    for m in all_metrics:
        e = m["experiment"]
        if e["type"] == "in-domain":
            in_domain_ate[(e["train_domain"], e["model"])] = m["ate"]["f1"]

    print(f"\n{'=' * 65}")
    print("Table B  —  Cross-Domain ATE Comparison")
    print(f"{'=' * 65}")
    print(f"{'Train→Test':<18} {'BERT F1':<12} {'DeBERTa F1':<12} "
          f"{'Δ (vs in-domain)'}")
    print("-" * 65)

    combos = [("restaurant", "restaurant"), ("restaurant", "laptop"),
              ("laptop", "laptop"),       ("laptop", "restaurant")]
    for tr, te in combos:
        arrow = f"{tr[:4]}→{te[:4]}"
        vals: dict[str, str] = {"bert": "—", "deberta": "—"}
        delta = "baseline" if tr == te else "—"
        for m in all_metrics:
            e = m["experiment"]
            if e["train_domain"] == tr and e["test_domain"] == te:
                f1 = m["ate"]["f1"]
                vals[e["model"]] = f"{f1:.4f}"
                if tr != te:
                    base = in_domain_ate.get((tr, e["model"]))
                    if base is not None:
                        delta = f"{f1 - base:+.4f}"
        print(f"{arrow:<18} {vals['bert']:<12} {vals['deberta']:<12} {delta}")

    # Table C — Cross-Domain ASC (predicted aspects)
    in_domain_asc: dict[tuple, dict] = {}
    for m in all_metrics:
        e = m["experiment"]
        if e["type"] == "in-domain":
            in_domain_asc[(e["train_domain"], e["model"])] = m["asc_pred"]

    print(f"\n{'=' * 75}")
    print("Table C  —  Cross-Domain ASC Comparison (predicted aspects)")
    print(f"{'=' * 75}")
    print(f"{'Train→Test':<18} {'BERT Acc':<10} {'BERT F1':<10} "
          f"{'DeBERTa Acc':<12} {'DeBERTa F1':<12} {'Δ Acc'}")
    print("-" * 75)
    for tr, te in combos:
        arrow = f"{tr[:4]}→{te[:4]}"
        v: dict[str, dict[str, str]] = {
            "bert":    {"acc": "—", "f1": "—"},
            "deberta": {"acc": "—", "f1": "—"},
        }
        delta = "baseline" if tr == te else "—"
        for m in all_metrics:
            e = m["experiment"]
            if e["train_domain"] == tr and e["test_domain"] == te:
                v[e["model"]]["acc"] = f"{m['asc_pred']['accuracy']:.4f}"
                v[e["model"]]["f1"]  = f"{m['asc_pred']['macro_f1']:.4f}"
                if tr != te:
                    base = in_domain_asc.get((tr, e["model"]))
                    if base:
                        delta = f"{m['asc_pred']['accuracy'] - base['accuracy']:+.4f}"
        print(f"{arrow:<18} "
              f"{v['bert']['acc']:<10} {v['bert']['f1']:<10} "
              f"{v['deberta']['acc']:<12} {v['deberta']['f1']:<12} {delta}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cross-Domain Experiments")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    run_all_experiments(args.output_dir)


if __name__ == "__main__":
    main()
