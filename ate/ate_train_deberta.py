"""Backward-compatible entry: ATE DeBERTa on *restaurant* (old default).

Prefer: ``python ate/ate_train.py --model_name deberta`` (any domain).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ate_train import train_ate

if __name__ == "__main__":
    train_ate("restaurant", "deberta")
