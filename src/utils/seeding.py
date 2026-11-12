"""Centralized experiment seeding.

Must be called BEFORE constructing a model, its optimizer, or any shuffled
tf.data.Dataset — Keras layers draw their initial weights from the global
random state at construction time, so seeding after the model already
exists cannot make weight initialization reproducible. `Trainer` also
calls this internally (see src/training/trainer.py) as a second line of
defense for dataset shuffling, but that call is too late to affect weight
init if the model was already built beforehand; scripts must call this
function themselves first. See docs/IMPLEMENTATION_ASSUMPTIONS.md.
"""
from __future__ import annotations

import tensorflow as tf


def set_experiment_seed(seed: int) -> None:
    """Order of operations in any experiment script:

        set_experiment_seed(seed)
        model = STNModel(..., seed=seed)
        trainer = Trainer(model, config)
        trainer.fit(...)
    """
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"set_experiment_seed: enable_op_determinism unavailable ({e})")
