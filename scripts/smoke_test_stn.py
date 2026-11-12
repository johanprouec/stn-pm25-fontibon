"""Architecture smoke test for the STN (CANDIDATE_V1 hyperparameters).

Synthetic data only — no Beijing, no Fontibon, no real training epochs.
Verifies: forward pass, finite MSE loss, gradients, and that a single
optimizer step actually changes the weights. This is NOT a training run;
it exists to catch wiring bugs before any real experiment is attempted.

Usage: python scripts/smoke_test_stn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.models.stn import STNModel, build_decoder_input

SEED = 42
BATCH = 4
LOOKBACK = 24
N_FEATURES = 3
LABEL_LEN = 12
PRED_LEN = 1

CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    model = STNModel(**CANDIDATE_V1, seed=SEED)

    # Synthetic data standing in for real windows: random encoder input,
    # a random "observed" PM2.5 history used to build a leakage-safe
    # decoder input, and a random normalized-scale target for the smoke
    # loss (not a real label — this script never touches real data).
    encoder_input = tf.random.normal((BATCH, LOOKBACK, N_FEATURES), seed=1)
    pm25_history = tf.random.normal((BATCH, LOOKBACK), seed=2)
    decoder_input = build_decoder_input(pm25_history, label_len=LABEL_LEN, prediction_length=PRED_LEN)
    y_dummy = tf.random.normal((BATCH, 1), seed=3)

    print("encoder_input shape:", encoder_input.shape)
    print("decoder_input shape:", decoder_input.shape)

    # Forward pass (eval mode: deterministic SparseAttention sampling)
    out_eval = model((encoder_input, decoder_input), training=False)
    print("Forward (training=False) output shape:", out_eval.shape)
    assert out_eval.shape == (BATCH, 1)
    assert np.isfinite(out_eval.numpy()).all(), "non-finite values in forward output"

    # Loss + gradients (training mode: stochastic SparseAttention sampling)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # [reported by paper], irrelevant to the smoke test itself
    weights_before = [w.numpy().copy() for w in model.trainable_variables]

    with tf.GradientTape() as tape:
        out_train = model((encoder_input, decoder_input), training=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(y_dummy, out_train))

    print("MSE loss (single synthetic batch):", float(loss.numpy()))
    assert np.isfinite(loss.numpy()), "loss is not finite"

    grads = tape.gradient(loss, model.trainable_variables)
    n_missing = sum(1 for g in grads if g is None)
    n_nonfinite = sum(
        1 for g in grads if g is not None and (tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)))
    )
    print(f"Trainable variables: {len(model.trainable_variables)} | missing grads: {n_missing} | non-finite grads: {n_nonfinite}")
    assert n_missing == 0, "some trainable variables received no gradient"
    assert n_nonfinite == 0, "some gradients are NaN/Inf"

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    weights_after = [w.numpy() for w in model.trainable_variables]
    n_changed = sum(
        1 for before, after in zip(weights_before, weights_after) if not np.allclose(before, after, atol=0)
    )
    print(f"Weights changed after one optimizer step: {n_changed}/{len(weights_before)}")
    assert n_changed == len(weights_before), "some weights did not change after the optimizer step"

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"\nCANDIDATE_V1 total trainable parameters: {total_params}")
    if total_params > 2_000_000:
        print(f"WARNING: parameter count {total_params} exceeds 2,000,000 (advisory only)")

    print("\nSmoke test passed: forward, loss, gradients and one optimizer step all behaved as expected.")
    print("No real dataset (Beijing/Fontibon) was loaded. No training epochs were run.")


if __name__ == "__main__":
    main()
