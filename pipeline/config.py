"""Centralized path and configuration constants for the ABSA pipeline."""

import os
from typing import Literal

from pipeline.semeval_data import DATA_XML

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Domain = Literal["restaurant", "laptop"]
ModelName = Literal["bert", "deberta"]

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
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_restaurant_bert", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_restaurant_bert", "final"),
        },
        "deberta": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_restaurant_deberta", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_restaurant_deberta", "final"),
        },
    },
    "laptop": {
        "bert": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_laptop_bert", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_laptop_bert", "final"),
        },
        "deberta": {
            "ate": os.path.join(PROJECT_ROOT, "ate", "ate_output_laptop_deberta", "final"),
            "asc": os.path.join(PROJECT_ROOT, "asc", "asc_output_laptop_deberta", "final"),
        },
    },
}

# ---------------------------------------------------------------------------
# HuggingFace datasets on disk (from ``ate_prepare_data`` / ``asc_prepare_data``)
# ---------------------------------------------------------------------------


def hf_ate_dataset_dir(domain: Domain) -> str:
    sub = "ate_data_restaurant" if domain == "restaurant" else "ate_data_laptop"
    return os.path.join(PROJECT_ROOT, "ate", sub)


def hf_asc_dataset_dir(domain: Domain) -> str:
    sub = "asc_data_restaurant" if domain == "restaurant" else "asc_data_laptop"
    return os.path.join(PROJECT_ROOT, "asc", sub)


def ate_training_run_dir(domain: Domain, model_name: ModelName) -> str:
    """Checkpoint root for ATE ``TrainingArguments``; best weights → ``.../final`` (see MODELS)."""
    return os.path.dirname(MODELS[domain][model_name]["ate"])


def asc_training_run_dir(domain: Domain, model_name: ModelName) -> str:
    """Checkpoint root for ASC ``TrainingArguments``; best weights → ``.../final``."""
    return os.path.dirname(MODELS[domain][model_name]["asc"])


# ---------------------------------------------------------------------------
# Backbone presets (HF id + training hyperparameters)
# ---------------------------------------------------------------------------
BACKBONE_PRESETS: dict[str, dict] = {
    "bert": {
        "hf_model": "bert-base-uncased",
        "ate": {
            "num_train_epochs": 5,
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
        },
        "asc": {
            "num_train_epochs": 5,
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
        },
    },
    "deberta": {
        "hf_model": "microsoft/deberta-base",
        "ate": {
            "num_train_epochs": 5,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
        },
        "asc": {
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
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
