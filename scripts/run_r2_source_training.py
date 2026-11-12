"""R2 SOURCE TRAINING — full training on Beijing TRAIN + VALIDATION (3
channels: pm25, temp, dew), for the three approved seeds (42, 123, 2026),
run SEQUENTIALLY in this one process (never concurrently).

Produces the definitive R2 SOURCE checkpoints that will later initialize
transfer Configs B/C/D on Fontibon (not implemented yet). This is NOT a new
R1 replication and is never compared against Zhang & Zhang's RMSE=19.04 as
an experimental target — see docs/IMPLEMENTATION_ASSUMPTIONS.md, section
"R2 -- preentrenamiento source para transferencia".

Forbidden here: Beijing test 2014, Fontibon (training or evaluation),
A/B/C/D, any architecture/hyperparameter tuning, reusing R1's 11-channel
checkpoints. Each seed starts from an independent initialization
(set_experiment_seed(seed) BEFORE building the model); no weights are ever
reused across seeds.

If a seed's summary already exists, that seed is skipped (never silently
overwritten) -- mirrors scripts/run_r1_development.py. If a seed fails, the
exception propagates and the script stops; already-written checkpoints/
history for completed epochs and completed seeds are preserved untouched;
this script performs no automatic retry.

Usage: python scripts/run_r2_source_training.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.data.beijing import prepare_beijing_r2
from src.evaluation.metrics import (
    evaluate_scaled_predictions,
    residual_diagnostics,
    save_predictions_with_residuals,
)
from src.evaluation.plots import generate_all_r1_development_figures
from src.models.stn import STNModel
from src.models.transfer_compat import compare_transfer_weight_shapes
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig, evaluate_split_batched
from src.utils.paths import BEIJING_CSV, OUTPUTS_DIR
from src.utils.seeding import set_experiment_seed

SEEDS = [42, 123, 2026]
CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)
ARCHITECTURE_METADATA = {
    **CANDIDATE_V1,
    "causal_mode": "probsparse",
    "lookback": 24,
    "decoder_label_len": 12,
    "prediction_length": 1,
}
EXPECTED_PARAMS = 99425
EXPERIMENT_NAME = "R2_source"


def _fontibon_shaped_target_model(seed: int) -> tf.keras.Model:
    """Built ONLY to exercise the (24,3) encoder / (13,1) decoder input
    shape that a Fontibon-target model will have -- never trained, never
    given a real target/label. Uses SYNTHETIC random tensors of exactly
    that shape rather than the real Fontibon CSV: this is a pure
    structural-compatibility check (does `target_model.encoder`/`.decoder`
    have the same weight shapes as the R2 source model?), which depends
    only on the shapes `(1, 24, 3)` / `(1, 13, 1)`, never on the actual
    values -- so it does not need, and should not require, local access to
    Raw_Consolidated_ECA_Complete_2022.csv. Does not affect R2's own
    architecture, weights, or results in any way.
    """
    rng = np.random.RandomState(seed)
    sample_window = rng.uniform(size=(1, 24, 3)).astype(np.float32)
    sample_decoder = rng.uniform(size=(1, 13, 1)).astype(np.float32)
    target_model = STNModel(**CANDIDATE_V1, seed=seed)
    target_model((tf.constant(sample_window), tf.constant(sample_decoder)), training=False)
    return target_model


def run_one_seed(seed: int, data) -> None:
    t_start = time.time()
    run_name = f"{EXPERIMENT_NAME}_seed{seed}"

    summary_path = OUTPUTS_DIR / "metrics" / f"{run_name}_summary.json"
    if summary_path.exists():
        print(f"SKIPPING seed={seed}: {summary_path} already exists (not overwriting).")
        return

    print("=" * 70)
    print(f"R2 SOURCE TRAINING -- seed={seed} -- full Beijing TRAIN + VALIDATION (3 channels)")
    print("NOT a new R1 replication. Beijing test 2014 and Fontibon are NEVER touched.")
    print("=" * 70)

    # --- Model: independent init, seed set BEFORE construction ---------------
    print(f"\n[1/5] Building CANDIDATE_V1 STN from scratch (seed={seed}, 3-channel input)...")
    set_experiment_seed(seed)
    model = STNModel(**CANDIDATE_V1, seed=seed)

    # --- Training: full train/validation, up to 200 epochs, patience=15 ------
    print(f"\n[2/5] Training: max_epochs=200, batch_size=32, lr=1e-4, patience=15, seed={seed}")
    config = TrainerConfig(
        experiment_name=EXPERIMENT_NAME,
        seed=seed,
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=200,
        early_stopping=EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(data.train, data.validation)  # best weights restored internally; history saved every epoch

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    if total_params != EXPECTED_PARAMS:
        raise RuntimeError(f"seed={seed}: expected {EXPECTED_PARAMS} trainable parameters, got {total_params}")
    epoch_times = [row["epoch_time_seconds"] for row in history]

    # --- Metrics: TRAIN + VALIDATION only, in ug/m3 ---------------------------
    print("\n[3/5] Evaluating BEST checkpoint on TRAIN and VALIDATION (ug/m3)...")
    train_pred_scaled = evaluate_split_batched(model, data.train, config.batch_size)
    val_pred_scaled = evaluate_split_batched(model, data.validation, config.batch_size)

    train_metrics_ugm3 = evaluate_scaled_predictions(data.train.y_scaled, train_pred_scaled, data.scalers.inverse_transform_pm25)
    val_metrics_ugm3 = evaluate_scaled_predictions(data.validation.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25)

    val_pred_ugm3 = data.scalers.inverse_transform_pm25(val_pred_scaled)
    val_real_ugm3 = data.validation.y_original.ravel()
    residuals = residual_diagnostics(val_real_ugm3, val_pred_ugm3)

    print(f"  train (ug/m3): {train_metrics_ugm3}")
    print(f"  validation (ug/m3): {val_metrics_ugm3}")
    print(f"  residual diagnostics (validation, ug/m3): {residuals}")

    predictions_path = OUTPUTS_DIR / "metrics" / f"{run_name}_validation_predictions.csv"
    save_predictions_with_residuals(data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, predictions_path)
    print(f"  row-level validation predictions saved: {predictions_path}")

    # --- Post-checkpoint compatibility re-verification ------------------------
    # Reload from the SAVED checkpoint into a fresh model instance (not the
    # in-memory trained one) to prove the checkpoint itself, on disk, yields
    # an encoder/decoder usable for future transfer.
    print("\n[4/5] Re-verifying compatibility from the SAVED checkpoint (fresh model instance)...")
    reloaded_model = STNModel(**CANDIDATE_V1, seed=seed)
    reloaded_model((data.train.encoder_input[:1], data.train.decoder_input[:1]), training=False)
    reloaded_model.load_weights(str(trainer.checkpoint_dir / "best.weights.h5"))

    target_model = _fontibon_shaped_target_model(seed=seed + 1)  # different seed on purpose: independent init
    compatibility_report = compare_transfer_weight_shapes(reloaded_model, target_model)
    print(f"  compatible (from checkpoint): {compatibility_report['compatible']}")
    if not compatibility_report["compatible"]:
        raise RuntimeError(f"seed={seed}: checkpoint encoder/decoder NOT compatible with Fontibon-shaped target: {compatibility_report}")

    # --- Figures ----------------------------------------------------------------
    print("\n[5/5] Generating figures...")
    figures_dir = OUTPUTS_DIR / "figures" / run_name
    generate_all_r1_development_figures(
        history, data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, figures_dir
    )
    print(f"  saved to: {figures_dir}")

    total_time = time.time() - t_start

    metadata = {
        "seed": seed,
        "feature_names": data.feature_names,
        "feature_order": data.feature_names,
        "architecture": ARCHITECTURE_METADATA,
        "best_epoch": trainer.best_epoch,
        "stopping_epoch": trainer.stopping_epoch,
        "best_val_loss": trainer.best_val_loss,
        "checkpoint_path": str(trainer.checkpoint_dir / "best.weights.h5"),
        "n_parameters": total_params,
        "source_domain": "Beijing",
        "target_domain_intended": "Fontibon",
        "encoder_transferable": True,
        "decoder_transferable": True,
        "output_head_transfer_policy": "reinitialize",
        "beijing_test_used": False,
        "fontibon_used_for_training": False,
    }
    metadata_path = OUTPUTS_DIR / "metrics" / f"{run_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metadata saved: {metadata_path}")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "source_pretraining_train_validation_only",
        "seed": seed,
        "config": {**CANDIDATE_V1, "learning_rate": config.learning_rate, "batch_size": config.batch_size,
                   "max_epochs": config.max_epochs, "early_stopping": {
                       "monitor": config.early_stopping.monitor, "patience": config.early_stopping.patience,
                       "min_delta": config.early_stopping.min_delta,
                       "restore_best_weights": config.early_stopping.restore_best_weights}},
        "best_epoch": trainer.best_epoch,
        "stopping_epoch": trainer.stopping_epoch,
        "total_epochs": len(history),
        "best_val_loss_scaled": trainer.best_val_loss,
        "train_metrics_ug_m3": train_metrics_ugm3,
        "validation_metrics_ug_m3": val_metrics_ugm3,
        "validation_residual_diagnostics_ug_m3": residuals,
        "total_trainable_parameters": total_params,
        "total_training_time_seconds": total_time,
        "sum_epoch_time_seconds": float(np.sum(epoch_times)),
        "mean_epoch_time_seconds": float(np.mean(epoch_times)),
        "median_epoch_time_seconds": float(np.median(epoch_times)),
        "gradient_norm_initial_epoch": history[0]["gradient_global_norm_mean"],
        "gradient_norm_final_epoch": history[-1]["gradient_global_norm_mean"],
        "gradient_norm_max_over_training": max(row["gradient_global_norm_max"] for row in history),
        "history_path": str((OUTPUTS_DIR / "metrics" / f"{config.run_id}_history.json")),
        "predictions_path": str(predictions_path),
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "figures_dir": str(figures_dir),
        "metadata_path": str(metadata_path),
        "compatibility_report_from_checkpoint": compatibility_report,
        "beijing_test_2014_used": False,
        "fontibon_trained_or_evaluated": False,
        "reused_r1_checkpoints": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"R2 SOURCE TRAINING (seed {seed}) -- train/validation only, Beijing test 2014 NOT evaluated")
    print(f"Total epochs: {len(history)} | best epoch: {trainer.best_epoch} | stopping epoch: {trainer.stopping_epoch}")
    print(f"Validation RMSE (ug/m3): {val_metrics_ugm3['RMSE']:.3f} | R2: {val_metrics_ugm3['R2']:.4f}")
    print(f"Compatible with Fontibon-shaped target (from checkpoint): {compatibility_report['compatible']}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min) | median epoch: {np.median(epoch_times):.1f}s")
    print(f"Summary saved: {summary_path}")
    print("=" * 70)


def main() -> None:
    print("\nPreparing Beijing R2 data ONCE for all three seeds (include_test=False)...")
    data = prepare_beijing_r2(BEIJING_CSV, include_test=False)
    print(f"  feature_names: {data.feature_names}")
    print(f"  train windows: {data.train.n_windows} | validation windows: {data.validation.n_windows}")
    assert data.test is None
    assert data.feature_names == ["pm25", "temp", "dew"]

    for seed in SEEDS:
        run_one_seed(seed, data)

    print("\nAll requested R2 source seeds finished (or were already present).")


if __name__ == "__main__":
    main()
