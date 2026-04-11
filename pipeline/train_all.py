"""Prepare SemEval HF datasets and train all ATE+ASC checkpoints (2×2×2 jobs).

Usage (from project root)::

    python -m pipeline.train_all
    python -m pipeline.train_all --train-only
    python -m pipeline.train_all --prepare-only
    python -m pipeline.train_all --domain laptop --model_name bert
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline.config import PROJECT_ROOT

DOMAINS = ("restaurant", "laptop")
MODEL_NAMES = ("bert", "deberta")


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n{'─' * 60}\n$ {' '.join(cmd)}\n{'─' * 60}")
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_data() -> None:
    root = Path(PROJECT_ROOT)
    py = sys.executable
    for dom in DOMAINS:
        _run([py, str(root / "ate" / "ate_prepare_data.py"), "--domain", dom], root)
    for dom in DOMAINS:
        _run([py, str(root / "asc" / "asc_prepare_data.py"), "--domain", dom], root)


def train_all(
    domains: tuple[str, ...] = DOMAINS,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> None:
    root = Path(PROJECT_ROOT)
    py = sys.executable
    ate_train = root / "ate" / "ate_train.py"
    asc_train = root / "asc" / "asc_train.py"
    for dom in domains:
        for mn in model_names:
            _run([py, str(ate_train), "--domain", dom, "--model_name", mn], root)
            _run([py, str(asc_train), "--domain", dom, "--model_name", mn], root)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare data + train all ATE/ASC models per pipeline.config.MODELS",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run ate_prepare_data / asc_prepare_data for both domains.",
    )
    ap.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training (expects HF datasets already on disk).",
    )
    ap.add_argument(
        "--domain",
        choices=[*DOMAINS, "all"],
        default="all",
        help="Limit training to one domain (default: all).",
    )
    ap.add_argument(
        "--model_name",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Limit training to one backbone (default: all).",
    )
    args = ap.parse_args()

    if args.prepare_only and args.train_only:
        ap.error("Use at most one of --prepare-only / --train-only")

    domains = DOMAINS if args.domain == "all" else (args.domain,)
    model_names = MODEL_NAMES if args.model_name == "all" else (args.model_name,)

    if args.prepare_only:
        prepare_data()
        return

    if args.train_only:
        train_all(domains=domains, model_names=model_names)
        return

    prepare_data()
    train_all(domains=domains, model_names=model_names)


if __name__ == "__main__":
    main()
