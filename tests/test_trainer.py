"""Tests for src/training/trainer.py, src/utils/seeding.py, and
src/data/beijing.py::persistence_prediction — all on small synthetic data,
never real Beijing/Fontibon."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

tf.keras.utils.set_random_seed(999)
tf.random.set_global_generator(tf.random.Generator.from_seed(999))
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from src.data.beijing import PreparedSplit, persistence_prediction
from src.models.stn import STNModel
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig, evaluate_split_batched
from src.utils.seeding import set_experiment_seed

D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT = 16, 2, 32, 3, 0.0
N_FEATURES = 3
LOOKBACK, LABEL_LEN, PRED_LEN = 24, 12, 1


def make_synthetic_split(n: int, seed: int) -> PreparedSplit:
    rng = np.random.default_rng(seed)
    encoder_input = rng.normal(size=(n, LOOKBACK, N_FEATURES)).astype(np.float32)
    pm25_history = encoder_input[:, :, 0]  # channel 0 = "pm25" by convention here
    label = pm25_history[:, -LABEL_LEN:]
    placeholder = np.zeros((n, PRED_LEN), dtype=np.float32)
    decoder_input = np.concatenate([label, placeholder], axis=1)[..., np.newaxis]
    y_scaled = rng.normal(size=(n, 1)).astype(np.float32)
    y_original = y_scaled * 50.0 + 20.0  # arbitrary affine stand-in for inverse_transform
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    return PreparedSplit(
        encoder_input=encoder_input, decoder_input=decoder_input,
        y_scaled=y_scaled, y_original=y_original, target_timestamps=ts,
    )


def make_model(seed=None):
    return STNModel(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, factor=FACTOR, dropout=DROPOUT,
        n_encoder_units=1, n_decoder_layers=1, non_selected_fill="zero", seed=seed,
    )


# --- optimization_loss vs deterministic train/val loss consistency ----------

def test_train_rmse_scaled_squared_equals_train_loss():
    set_experiment_seed(11)
    model = make_model(seed=11)
    train_split = make_synthetic_split(64, seed=1)
    val_split = make_synthetic_split(32, seed=2)

    config = TrainerConfig(
        experiment_name="unit_test_metrics", seed=11, batch_size=16, max_epochs=2,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(train_split, val_split)

    for row in history:
        np.testing.assert_allclose(row["train_rmse_scaled"] ** 2, row["train_loss"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(row["val_rmse_scaled"] ** 2, row["val_loss"], rtol=1e-5, atol=1e-8)


def tmp_root(name):
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / name


# --- batched vs direct evaluation --------------------------------------------

def test_batched_predictions_match_direct_predictions():
    model = make_model(seed=5)
    split = make_synthetic_split(37, seed=3)  # not a multiple of batch_size, on purpose

    direct = model((split.encoder_input, split.decoder_input), training=False).numpy()
    batched = evaluate_split_batched(model, split, batch_size=8)

    np.testing.assert_allclose(direct, batched, atol=1e-6)
    assert batched.shape == direct.shape


# --- seed before model construction ------------------------------------------

def test_same_seed_before_construction_gives_same_initial_weights():
    set_experiment_seed(123)
    model_a = make_model(seed=123)

    set_experiment_seed(123)
    model_b = make_model(seed=123)

    split = make_synthetic_split(4, seed=1)
    model_a((split.encoder_input, split.decoder_input), training=False)  # build
    model_b((split.encoder_input, split.decoder_input), training=False)

    for wa, wb in zip(model_a.trainable_variables, model_b.trainable_variables):
        np.testing.assert_allclose(wa.numpy(), wb.numpy(), atol=1e-6)


def test_different_seed_before_construction_gives_different_weights():
    set_experiment_seed(1)
    model_a = make_model(seed=1)

    set_experiment_seed(2)
    model_b = make_model(seed=2)

    split = make_synthetic_split(4, seed=1)
    model_a((split.encoder_input, split.decoder_input), training=False)
    model_b((split.encoder_input, split.decoder_input), training=False)

    diffs = [
        not np.allclose(wa.numpy(), wb.numpy(), atol=1e-6)
        for wa, wb in zip(model_a.trainable_variables, model_b.trainable_variables)
    ]
    assert any(diffs), "expected at least some weights to differ between seeds"


# --- persistence baseline -----------------------------------------------------

def test_persistence_uses_exactly_last_historical_pm25():
    split = make_synthetic_split(10, seed=7)
    feature_names = ["pm25", "dewp", "temp"]
    pred = persistence_prediction(split, feature_names)

    expected = split.encoder_input[:, -1, 0].reshape(-1, 1)  # pm25 is channel 0
    np.testing.assert_array_equal(pred, expected)
    assert pred.shape == (10, 1)


# --- trainer never touches "test" ---------------------------------------------

def test_trainer_fit_signature_has_no_test_parameter():
    params = inspect.signature(Trainer.fit).parameters
    assert "test" not in params
    assert "test_split" not in params
    assert set(params) - {"self"} == {"train_split", "val_split"}


# --- best checkpoint matches minimum val_loss ----------------------------------

def test_best_epoch_matches_minimum_val_loss_in_history():
    set_experiment_seed(21)
    model = make_model(seed=21)
    train_split = make_synthetic_split(48, seed=1)
    val_split = make_synthetic_split(16, seed=2)

    config = TrainerConfig(
        experiment_name="unit_test_best", seed=21, batch_size=16, max_epochs=4,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(train_split, val_split)

    val_losses = [row["val_loss"] for row in history]
    min_epoch = history[int(np.argmin(val_losses))]["epoch"]
    assert trainer.best_epoch == min_epoch
    np.testing.assert_allclose(trainer.best_val_loss, min(val_losses), rtol=1e-6)

    best_so_far_epochs = [row["epoch"] for row in history if row["best_so_far"]]
    assert trainer.best_epoch == max(best_so_far_epochs)


# --- history contains all required fields --------------------------------------

def test_history_contains_all_required_fields():
    set_experiment_seed(31)
    model = make_model(seed=31)
    train_split = make_synthetic_split(32, seed=1)
    val_split = make_synthetic_split(16, seed=2)

    config = TrainerConfig(
        experiment_name="unit_test_fields", seed=31, batch_size=16, max_epochs=1,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(train_split, val_split)

    expected_fields = {
        "epoch", "optimization_loss_mean", "train_loss", "val_loss",
        "train_rmse_scaled", "val_rmse_scaled", "learning_rate",
        "gradient_global_norm_mean", "gradient_global_norm_max", "gradient_global_norm_std",
        "epoch_time_seconds", "best_so_far",
    }
    assert set(history[0].keys()) == expected_fields


# --- incremental per-epoch persistence (not just at the very end) --------------

def test_history_file_is_updated_after_every_epoch_not_just_at_the_end():
    """Spies on Trainer._save_history to record, at each call, how many
    rows were on disk-bound at that point — must be called once per epoch,
    with a strictly growing row count (1, 2, 3, ...), not once at the end
    with the full count."""
    set_experiment_seed(41)
    model = make_model(seed=41)
    train_split = make_synthetic_split(32, seed=1)
    val_split = make_synthetic_split(16, seed=2)

    config = TrainerConfig(
        experiment_name="unit_test_incremental_save", seed=41, batch_size=16, max_epochs=3,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    trainer = Trainer(model, config)

    row_counts_at_save = []
    original_save = trainer._save_history

    def spy_save_history():
        row_counts_at_save.append(len(trainer.history))
        original_save()
        # the file on disk must match what was in memory at this exact call
        on_disk = json.loads(trainer.metrics_path.with_suffix(".json").read_text(encoding="utf-8"))
        assert len(on_disk) == len(trainer.history)

    trainer._save_history = spy_save_history
    trainer.fit(train_split, val_split)

    assert row_counts_at_save == [1, 2, 3], f"expected one save per epoch with growing row counts, got {row_counts_at_save}"


def test_incremental_saving_does_not_change_training_math():
    """Same synthetic data, same seed: a model trained with the
    incremental-save trainer must produce numerically identical losses to
    what the same math would without it — this change is I/O-only."""
    set_experiment_seed(51)
    model_a = make_model(seed=51)
    train_split = make_synthetic_split(32, seed=1)
    val_split = make_synthetic_split(16, seed=2)
    config_a = TrainerConfig(
        experiment_name="unit_test_math_a", seed=51, batch_size=16, max_epochs=2,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    history_a = Trainer(model_a, config_a).fit(train_split, val_split)

    set_experiment_seed(51)
    model_b = make_model(seed=51)
    config_b = TrainerConfig(
        experiment_name="unit_test_math_b", seed=51, batch_size=16, max_epochs=2,
        early_stopping=EarlyStoppingConfig(patience=100),
        checkpoints_root=tmp_root("ckpt"), tensorboard_root=tmp_root("tb"), metrics_root=tmp_root("metrics"),
    )
    history_b = Trainer(model_b, config_b).fit(train_split, val_split)

    for ra, rb in zip(history_a, history_b):
        np.testing.assert_allclose(ra["train_loss"], rb["train_loss"], atol=1e-7)
        np.testing.assert_allclose(ra["val_loss"], rb["val_loss"], atol=1e-7)
