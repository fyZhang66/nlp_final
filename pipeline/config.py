"""Centralized path and configuration constants for the ABSA pipeline."""

import os

from pipeline.semeval_data import DATA_XML

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Raw XML data (SemEval 2014 Task 4) — same files as ``ate_prepare`` / ``asc_prepare``
# ---------------------------------------------------------------------------
RAW_DATA = {
    "restaurant": {
        "train_xml": DATA_XML["restaurant"]["train"],
        "test_xml":  DATA_XML["restaurant"]["test"],
    },
    "laptop": {
        "train_xml": DATA_XML["laptop"]["train"],
        "test_xml":  DATA_XML["laptop"]["test"],
    },
}

# ---------------------------------------------------------------------------
# Trained model directories
# Models are keyed by *training domain*, then model architecture.
# ---------------------------------------------------------------------------
MODELS = {
    "restaurant": {
        "bert": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output", "final"),
        },
        "deberta": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_deberta", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_deberta", "final"),
        },
    },
    "laptop": {
        "bert": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_laptop", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_laptop", "final"),
        },
        "deberta": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_laptop_deberta", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_laptop_deberta", "final"),
        },
    },
}

# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pipeline", "outputs")

# ---------------------------------------------------------------------------
# Label constants (kept consistent with upstream modules)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
POLARITY_MAP = {"positive": 0, "negative": 1, "neutral": 2}
ATE_LABELS = ["O", "B-ASP", "I-ASP"]
MAX_LEN = 128
