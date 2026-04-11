"""Unified ``test_results.json`` envelope for ASC / ATE training scripts."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

Task = Literal["asc", "ate"]

SCHEMA_VERSION = 1


def write_test_results(
    output_dir: str,
    *,
    task: Task,
    domain: str,
    model: str,
    metrics: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> None:
    """Write ``{output_dir}/test_results.json`` with a stable top-level shape."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "domain": domain,
        "model": model,
        "split": "test",
        "metrics": metrics,
    }
    if extras:
        payload["extras"] = extras
    path = os.path.join(output_dir, "test_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
