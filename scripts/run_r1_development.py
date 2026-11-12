"""R1 DEVELOPMENT — full training run on Beijing TRAIN + VALIDATION, for a
single given seed.

NOT "final R1": no hyperparameter tuning has happened, and Beijing test
2014 is never built, referenced, or evaluated anywhere in this script.
Config: configs/beijing_r1_development.yaml (CANDIDATE_V1 architecture) —
identical across every seed; only the seed changes between runs.

Forbidden here (see the reviews that authorized these runs):
Beijing test 2014, Fontibon, R2, A/B/C/D, hyperparameter tuning, any
comparison against the paper's RMSE=19.04, and any automatic architecture
change based on a run's result. Seeds are run ONE AT A TIME (not
concurrently), each starting from an independent random initialization —
never reusing another seed's weights.

Refuses to run if that seed's summary already exists, so a previously
completed seed (e.g. 42) can never be silently overwritten by re-running
this script.

Usage: python scripts/run_r1_development.py [seed]   (default seed: 42)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.data.beijing import prepare_beijing_r1, persistence_prediction
from src.evaluation.metrics import (
    classify_model_vs_baseline,
    evaluate_scaled_predictions,
    residual_diagnostics,
    save_predictions_with_residuals,
    target_distribution,
)
from src.evaluation.plots import generate_all_r1_development_figures
from src.models.stn import STNModel
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig, evaluate_split_batched
from src.utils.paths import BEIJING_CSV, OUTPUTS_DIR
from src.utils.seeding import set_experiment_seed

CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)
EXPERIMENT_NAME = "R1_development"


def main(seed: int) -> None:
    t_start = time.time()
    run_name = f"{EXPERIMENT_NAME}_seed{seed}"

    summary_path = OUTPUTS_DIR / "metrics" / f"{run_name}_summary.json"
    if summary_path.exists():
        print(f"REFUSING TO RUN: {summary_path} already exists. "
              f"This seed's results are already complete — not overwriting them. "
              f"Delete it manually first if you really intend to redo this seed.")
        sys.exit(1)

    print("=" * 70)
    print(f"R1 DEVELOPMENT -- seed={seed} -- full TRAIN + VALIDATION, Beijing test 2014 NEVER built")
    print("NOT final R1. No tuning. No test. No Fontibon.")
    print("=" * 70)

    # --- 1. Data: test is never even constructed in this script ---------------
    print("\n[1/6] Preparing Beijing R1 data (include_test=False)...")
    data = prepare_beijing_r1(BEIJING_CSV, include_test=False)
    print(f"  train windows: {data.train.n_windows}")
    print(f"  validation windows: {data.validation.n_windows}")
    assert data.test is None

    # --- 2. Persistence baseline on validation — not seed-dependent, so
    # compute it once and reuse across seeds rather than "overwriting" it
    # with identical numbers every run. ---
    baselines_path = OUTPUTS_DIR / "metrics" / "R1_beijing_validation_baselines.json"
    if baselines_path.exists():
        print(f"\n[2/6] Persistence baseline already computed, reusing: {baselines_path}")
        persistence_metrics = json.loads(baselines_path.read_text(encoding="utf-8"))["persistence"]
    else:
        print("\n[2/6] Computing persistence baseline on VALIDATION...")
        persistence_pred_scaled = persistence_prediction(data.validation, data.feature_names)
        persistence_metrics = evaluate_scaled_predictions(
            data.validation.y_scaled, persistence_pred_scaled, data.scalers.inverse_transform_pm25
        )
        baselines_payload = {
            "note": "Sanity check for Beijing R1 only. NOT the Primera Entrega's selected baseline (that was Ridge, on Fontibon). Not seed-dependent; computed once and reused.",
            "split": "validation",
            "persistence": persistence_metrics,
        }
        baselines_path.write_text(json.dumps(baselines_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  saved: {baselines_path}")
    print(f"  persistence (ug/m3): {persistence_metrics}")

    val_target_dist = target_distribution(data.validation.y_original)
    print(f"  validation target distribution (ug/m3): {val_target_dist}")

    # --- 3. Model (seed BEFORE construction; independent init, no reused weights) ---
    print(f"\n[3/6] Building CANDIDATE_V1 STN (seed={seed}, set before construction)...")
    set_experiment_seed(seed)
    model = STNModel(**CANDIDATE_V1, seed=seed)

    # --- 4. Training: full train/validation, up to 200 epochs, patience=15 -----
    print(f"\n[4/6] Training: max_epochs=200, batch_size=32, lr=1e-4, patience=15, seed={seed}")
    config = TrainerConfig(
        experiment_name=EXPERIMENT_NAME,
        seed=seed,
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=200,
        early_stopping=EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(data.train, data.validation)  # best weights restored internally at the end; history saved incrementally every epoch

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    epoch_times = [row["epoch_time_seconds"] for row in history]

    # --- 5. Best-checkpoint metrics in ug/m3, train + validation ONLY ----------
    print("\n[5/6] Evaluating BEST checkpoint on TRAIN and VALIDATION (ug/m3)...")
    train_pred_scaled = evaluate_split_batched(model, data.train, config.batch_size)
    val_pred_scaled = evaluate_split_batched(model, data.validation, config.batch_size)

    train_metrics_ugm3 = evaluate_scaled_predictions(data.train.y_scaled, train_pred_scaled, data.scalers.inverse_transform_pm25)
    val_metrics_ugm3 = evaluate_scaled_predictions(data.validation.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25)

    val_pred_ugm3 = data.scalers.inverse_transform_pm25(val_pred_scaled)
    val_real_ugm3 = data.validation.y_original.ravel()
    residuals = residual_diagnostics(val_real_ugm3, val_pred_ugm3)

    classification = classify_model_vs_baseline(val_metrics_ugm3["RMSE"], persistence_metrics["RMSE"])

    print(f"  train (ug/m3): {train_metrics_ugm3}")
    print(f"  validation (ug/m3): {val_metrics_ugm3}")
    print(f"  residual diagnostics (validation, ug/m3): {residuals}")
    print(f"  descriptive classification vs persistence: {classification}")

    predictions_path = OUTPUTS_DIR / "metrics" / f"{run_name}_validation_predictions.csv"
    save_predictions_with_residuals(data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, predictions_path)
    print(f"  row-level validation predictions saved: {predictions_path}")

    # --- 6. Figures --------------------------------------------------------------
    print("\n[6/6] Generating figures...")
    figures_dir = OUTPUTS_DIR / "figures" / run_name
    generate_all_r1_development_figures(
        history, data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, figures_dir
    )
    print(f"  saved to: {figures_dir}")

    total_time = time.time() - t_start

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "development_validation_only",
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
        "persistence_validation_metrics_ug_m3": persistence_metrics,
        "descriptive_classification_vs_persistence": classification,
        "validation_target_distribution_ug_m3": val_target_dist,
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
        "beijing_test_2014_evaluated": False,
        "fontibon_used": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"R1 DEVELOPMENT (seed {seed}) -- train/validation only, test 2014 NOT evaluated")
    print(f"Total epochs: {len(history)} | best epoch: {trainer.best_epoch} | stopping epoch: {trainer.stopping_epoch}")
    print(f"Validation RMSE (ug/m3): {val_metrics_ugm3['RMSE']:.3f} | Persistence RMSE: {persistence_metrics['RMSE']:.3f}")
    print(f"Classification: {classification}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min) | median epoch: {np.median(epoch_times):.1f}s")
    print(f"Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed_arg)
