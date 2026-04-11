"""Backward-compatible entry: ASC DeBERTa on *restaurant* (old default).

Prefer: ``python asc/asc_train.py --model_name deberta`` (any domain).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asc_train import train_asc

if __name__ == "__main__":
    train_asc("restaurant", "deberta")
