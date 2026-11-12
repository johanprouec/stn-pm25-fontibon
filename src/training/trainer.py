"""Reusable training loop for STNModel — built generically enough to reuse
for R2 and the A/B/C/D transfer experiments later (none of which are
implemented yet).

Three distinct loss-like numbers are tracked per epoch, and they are NOT
interchangeable (see docs/IMPLEMENTATION_ASSUMPTIONS.md, "optimization
loss vs. deterministic train/val loss"):

- `optimization_loss_mean`: the mean of the per-batch MSE values observed
  WHILE weights were still changing during that epoch's optimizer steps
  (training=True, stochastic SparseAttention sampling). Diagnostic only —
  never compare it directly to val_loss, since it was measured against a
  moving target (the weights mid-epoch), not the epoch's final weights.
- `train_loss`: MSE computed AFTER the epoch's weights are final, with
  training=False (deterministic sampling), over the WHOLE train split.
- `val_loss`: same, over the whole validation split.
`train_loss` and `val_loss` are the only two that are directly comparable
to each other — same weights, same deterministic evaluation mode, just a
different split. `train_rmse_scaled = sqrt(train_loss)` and
`val_rmse_scaled = sqrt(val_loss)` follow from that.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.utils.seeding import set_experiment_seed


@dataclass
class EarlyStoppingConfig:
    """Candidate values, NOT reported by Zhang & Zhang — see
    docs/IMPLEMENTATION_ASSUMPTIONS.md. The paper only fixes a maximum of
    200 epochs; early stopping itself, and these specific numbers, are our
    own decision to avoid overfitting and wasted compute."""
    monitor: str = "val_loss"
    patience: int = 15
    min_delta: float = 0.0
    restore_best_weights: bool = True


@dataclass
class TrainerConfig:
    experiment_name: str
    seed: int
    learning_rate: float = 1e-4       # [reported by paper]
    batch_size: int = 32               # [reported by paper]
    max_epochs: int = 200               # [reported by paper]
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoints_root: Path = Path("outputs/checkpoints")
    tensorboard_root: Path = Path("outputs/tensorboard")
    metrics_root: Path = Path("outputs/metrics")

    @property
    def run_id(self) -> str:
        return f"{self.experiment_name}_seed{self.seed}"


def evaluate_split_batched(model: tf.keras.Model, split, batch_size: int) -> np.ndarray:
    """Predictions for an entire PreparedSplit, computed in fixed-size
    batches (shuffle=False) rather than one single call over the whole
    split — avoids holding the full activation graph for potentially tens
    of thousands of windows in memory at once (matters for the full R1/R2/
    Fontibon splits, not just the pilot's small subsample). Always run
    with training=False (deterministic SparseAttention sampling).
    Predictions come back in the SAME order as the input split.
    """
    ds = tf.data.Dataset.from_tensor_slices((split.encoder_input, split.decoder_input)).batch(batch_size)
    preds = [model((enc, dec), training=False).numpy() for enc, dec in ds]
    return np.concatenate(preds, axis=0)


class Trainer:
    """Custom loop: STNModel + Adam + MSE, with per-epoch/per-batch
    instrumentation. `fit` never touches any data the caller doesn't pass
    it — in particular it has no notion of "test"; whether Beijing test
    2014 (or Fontibon test) is ever seen is entirely the caller's
    responsibility. Callers should call
    ``src.utils.seeding.set_experiment_seed(seed)`` BEFORE constructing
    the model passed here — `Trainer.__init__`/`fit` also call it, as a
    second line of defense for dataset-shuffling determinism, but by then
    it is too late to affect the model's weight initialization.
    """

    def __init__(self, model: tf.keras.Model, config: TrainerConfig):
        self.model = model
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.history: List[Dict] = []
        self.best_epoch: Optional[int] = None
        self.stopping_epoch: Optional[int] = None
        self.best_val_loss: float = float("inf")

        self.checkpoint_dir = config.checkpoints_root / config.run_id
        self.tensorboard_dir = config.tensorboard_root / config.run_id
        self.metrics_path = config.metrics_root / f"{config.run_id}_history"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.summary_writer = tf.summary.create_file_writer(str(self.tensorboard_dir))

    def _dataset(self, split, shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            ((split.encoder_input, split.decoder_input), split.y_scaled)
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=split.n_windows, seed=self.config.seed, reshuffle_each_iteration=True)
        return ds.batch(self.config.batch_size)

    @tf.function
    def _train_step(self, encoder_input, decoder_input, y):
        with tf.GradientTape() as tape:
            pred = self.model((encoder_input, decoder_input), training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))
        grads = tape.gradient(loss, self.model.trainable_variables)
        finite_pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        n_missing = len(grads) - len(finite_pairs)
        grad_norm = tf.linalg.global_norm([g for g, _ in finite_pairs])
        self.optimizer.apply_gradients(finite_pairs)
        return loss, grad_norm, n_missing

    def fit(self, train_split, val_split) -> List[Dict]:
        set_experiment_seed(self.config.seed)  # second line of defense — see class docstring
        train_ds = self._dataset(train_split, shuffle=True)

        es = self.config.early_stopping
        epochs_without_improvement = 0
        best_weights = None
        unexpected_missing_grads = 0

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()

            opt_losses, batch_grad_norms = [], []
            for (enc, dec), y in train_ds:
                loss, gnorm, n_missing = self._train_step(enc, dec, y)
                opt_losses.append(float(loss.numpy()))
                batch_grad_norms.append(float(gnorm.numpy()))
                unexpected_missing_grads += int(n_missing.numpy()) if hasattr(n_missing, "numpy") else int(n_missing)

            # Deterministic, post-weight-update, full-split evaluation —
            # this is what train_loss/val_loss/*_rmse_scaled are computed
            # from, NOT the optimization-time batch losses above.
            train_pred = evaluate_split_batched(self.model, train_split, self.config.batch_size)
            val_pred = evaluate_split_batched(self.model, val_split, self.config.batch_size)
            train_loss = float(np.mean((train_pred - train_split.y_scaled) ** 2))
            val_loss = float(np.mean((val_pred - val_split.y_scaled) ** 2))
            train_rmse_scaled = float(np.sqrt(train_loss))
            val_rmse_scaled = float(np.sqrt(val_loss))

            lr_value = self.optimizer.learning_rate
            lr_value = float(lr_value.numpy()) if hasattr(lr_value, "numpy") else float(lr_value)

            is_best = val_loss < self.best_val_loss - es.min_delta

            row = {
                "epoch": epoch,
                "optimization_loss_mean": float(np.mean(opt_losses)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse_scaled": train_rmse_scaled,
                "val_rmse_scaled": val_rmse_scaled,
                "learning_rate": lr_value,
                "gradient_global_norm_mean": float(np.mean(batch_grad_norms)),
                "gradient_global_norm_max": float(np.max(batch_grad_norms)),
                "gradient_global_norm_std": float(np.std(batch_grad_norms)),
                "epoch_time_seconds": time.time() - t0,
                "best_so_far": bool(is_best),
            }
            self.history.append(row)
            self._log_tensorboard(row)
            self._save_history()  # persist after EVERY epoch, not just at the end — I/O only, no effect on loss/optimizer/weights
            print(
                f"[{self.config.run_id}] epoch {epoch:3d} | opt_loss={row['optimization_loss_mean']:.5f} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"train_rmse={train_rmse_scaled:.5f} val_rmse={val_rmse_scaled:.5f} "
                f"grad_norm_mean={row['gradient_global_norm_mean']:.4f} "
                f"{'[BEST]' if is_best else ''} time={row['epoch_time_seconds']:.1f}s"
            )

            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                if es.restore_best_weights:
                    best_weights = [w.numpy() for w in self.model.trainable_variables]
                self.model.save_weights(str(self.checkpoint_dir / "best.weights.h5"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= es.patience:
                    self.stopping_epoch = epoch
                    print(f"[{self.config.run_id}] early stopping at epoch {epoch} "
                          f"(best epoch {self.best_epoch}, val_loss={self.best_val_loss:.5f})")
                    break

        if self.stopping_epoch is None:
            self.stopping_epoch = len(self.history)

        if es.restore_best_weights and best_weights is not None:
            for w, value in zip(self.model.trainable_variables, best_weights):
                w.assign(value)

        if unexpected_missing_grads > 0:
            print(f"WARNING: {unexpected_missing_grads} (variable, batch) pairs had a None gradient during training")

        # No extra _save_history() call here: the in-loop call above already
        # wrote the final epoch's row (including on an early-stopping break),
        # so the file on disk is already complete at this point.
        return self.history

    def _log_tensorboard(self, row: Dict) -> None:
        with self.summary_writer.as_default():
            for key in (
                "optimization_loss_mean", "train_loss", "val_loss",
                "train_rmse_scaled", "val_rmse_scaled", "learning_rate",
                "gradient_global_norm_mean", "gradient_global_norm_max",
            ):
                tf.summary.scalar(key, row[key], step=row["epoch"])
        self.summary_writer.flush()

    def _save_history(self) -> None:
        self.metrics_path.with_suffix(".json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        if self.history:
            with open(self.metrics_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                writer.writeheader()
                writer.writerows(self.history)
