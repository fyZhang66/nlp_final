"""Generate figures from pipeline outputs (confusion matrices, error stats, summary).

Usage::

    python -m pipeline.plot_figures
    python -m pipeline.plot_figures --outputs_dir pipeline/outputs --figures_dir pipeline/figures

Called automatically at the end of ``python -m pipeline.run_cross_domain`` when
``--skip_figures`` is not set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Non-interactive backend for headless / CI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import FIGURES_DIR, OUTPUT_DIR


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s)[:120]


def plot_confusion_matrices(
    abs_csv: Path,
    norm_csv: Path,
    out_dir: Path,
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_abs = pd.read_csv(abs_csv, index_col=0)
    df_norm = pd.read_csv(norm_csv, index_col=0).astype(float)

    def _clean_labels(idx):
        return [str(x).replace("Actual_", "").replace("_", " ") for x in idx]

    y_labels = _clean_labels(df_abs.index)
    x_labels = [str(c).replace("Pred_", "").replace("_", " ") for c in df_abs.columns]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        df_abs,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
    )
    ax.set_title(f"{title_prefix} — ASC confusion (absolute)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_abs.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        df_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
    )
    ax.set_title(f"{title_prefix} — ASC confusion (row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_norm.png", dpi=200)
    plt.close(fig)


def plot_ate_error_stats(json_path: Path, out_path: Path, title_prefix: str) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    labels = [d["error_type"] for d in data]
    pct = [d["percentage"] for d in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = sns.color_palette("husl", n_colors=len(labels))
    ax.barh(labels, pct, color=colors)
    ax.set_xlabel("Percentage")
    ax.set_title(f"{title_prefix} — ATE span-level errors")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_e2e_error_tracing(json_path: Path, out_path: Path, title_prefix: str) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    labels = [d["category"].replace(" → ", "\n→ ") for d in data]
    pct = [d["percentage"] for d in data]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, pct, color=sns.color_palette("muted", n_colors=len(labels)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Percentage of end-to-end errors")
    ax.set_title(f"{title_prefix} — E2E error tracing")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cross_domain_summary(summary_path: Path, out_dir: Path) -> None:
    if not summary_path.is_file():
        return
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [r["experiment"]["id"] for r in rows]
    labels = [
        f"{r['experiment']['train_domain'][:4]}→{r['experiment']['test_domain'][:4]}\n"
        f"{r['experiment']['model']}"
        for r in rows
    ]
    ate_f1 = [r["ate"]["f1"] for r in rows]
    asc_acc_p = [r["asc_pred"]["accuracy"] for r in rows]
    asc_f1_p = [r["asc_pred"]["macro_f1"] for r in rows]

    x = np.arange(len(ids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, ate_f1, w, label="ATE F1", color="steelblue")
    ax.bar(x + w / 2, asc_acc_p, w, label="ASC Acc (pred)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Cross-domain summary — ATE F1 vs ASC Acc (predicted aspects)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_ate_vs_asc_acc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, ate_f1, w, label="ATE F1", color="steelblue")
    ax.bar(x + w / 2, asc_f1_p, w, label="ASC Macro-F1 (pred)", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Cross-domain summary — ATE F1 vs ASC Macro-F1 (pred)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_ate_vs_asc_f1.png", dpi=200)
    plt.close(fig)


def generate_pipeline_figures(
    output_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> list[Path]:
    """
    Scan *output_dir* for experiment subfolders (each with confusion-matrix CSVs),
    write PNGs under ``figures_dir/<experiment_tag>/``. Also plot
    ``cross_domain_summary.json`` into ``figures_dir/``.

    Returns list of written PNG paths (best-effort).
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    figures_root = Path(figures_dir or FIGURES_DIR)
    written: list[Path] = []

    if not output_dir.is_dir():
        print(f"[plot_figures] skip: output_dir not found: {output_dir}")
        return written

    for sub in sorted(output_dir.iterdir()):
        if not sub.is_dir():
            continue
        cms = list(sub.glob("confusion_matrix_*_abs.csv"))
        if not cms:
            continue
        abs_csv = cms[0]
        m = re.match(
            r"confusion_matrix_(.+)_abs\.csv",
            abs_csv.name,
        )
        if not m:
            continue
        rest = m.group(1)
        norm_name = f"confusion_matrix_{rest}_norm.csv"
        norm_csv = sub / norm_name
        if not norm_csv.is_file():
            continue

        tag = sub.name
        title_prefix = tag.replace("_", " ")
        exp_fig = figures_root / _safe_name(tag)
        plot_confusion_matrices(abs_csv, norm_csv, exp_fig, title_prefix)
        written.extend([exp_fig / "confusion_matrix_abs.png", exp_fig / "confusion_matrix_norm.png"])

        ate_json = next(iter(sub.glob("ate_error_stats_*.json")), None)
        if ate_json and ate_json.is_file():
            p = exp_fig / "ate_span_errors.png"
            plot_ate_error_stats(ate_json, p, title_prefix)
            written.append(p)

        e2e_json = next(iter(sub.glob("e2e_error_tracing_*.json")), None)
        if e2e_json and e2e_json.is_file():
            p = exp_fig / "e2e_error_tracing.png"
            plot_e2e_error_tracing(e2e_json, p, title_prefix)
            written.append(p)

    summary = output_dir / "cross_domain_summary.json"
    if summary.is_file():
        plot_cross_domain_summary(summary, figures_root)
        written.extend(
            [
                figures_root / "summary_ate_vs_asc_acc.png",
                figures_root / "summary_ate_vs_asc_f1.png",
            ],
        )

    print(f"\n[plot_figures] Figures written under: {figures_root}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Plot pipeline figures from CSV/JSON outputs")
    ap.add_argument("--outputs_dir", default=None, help="Pipeline outputs (default: config OUTPUT_DIR)")
    ap.add_argument("--figures_dir", default=None, help="Figure output root (default: config FIGURES_DIR)")
    args = ap.parse_args()
    generate_pipeline_figures(output_dir=args.outputs_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
