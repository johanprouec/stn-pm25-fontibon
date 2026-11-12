"""Shared data contracts used by every preprocessing pipeline (Beijing R1,
Beijing R2, Fontibon STN) and consumed by ``src/training/trainer.py``.

Moved out of ``src/data/beijing.py`` into its own module so that
``src/data/fontibon.py`` can use the same ``PreparedSplit`` contract
without creating a circular import (``beijing.py`` already imports
``build_windows`` from ``fontibon.py``). ``PreparedSplit`` is re-exported
from ``src.data.beijing`` unchanged, so existing imports
(``from src.data.beijing import PreparedSplit``) keep working.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedSplit:
    """One partition's model-ready arrays, plus enough metadata
    (`y_original`, `target_timestamps`) to audit or evaluate without
    guessing what an array contains. This is the exact contract
    ``src/training/trainer.py`` (``Trainer.fit`` / ``evaluate_split_batched``)
    expects from any dataset — Beijing R1/R2 and Fontibon STN all produce
    this same shape of object."""
    encoder_input: np.ndarray        # (N, ventana, n_features)
    decoder_input: np.ndarray        # (N, decoder_label_len + prediction_length, 1)
    y_scaled: np.ndarray             # (N, 1)
    y_original: np.ndarray           # (N, 1), original units (ug/m3)
    target_timestamps: pd.DatetimeIndex

    @property
    def n_windows(self) -> int:
        return int(self.encoder_input.shape[0])
