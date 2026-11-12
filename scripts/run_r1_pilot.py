"""R1 PILOT — integration check only, NOT a real training run.

Verifies the full path CSV -> preprocessing -> dataset -> STN -> training
-> validation -> inverse transform -> metrics -> checkpoint -> TensorBoard
-> gradient norms, end to end, on a short subsample of Beijing
train/validation.

Explicitly forbidden here (see docs/PHASE2_PLAN.md and the review that
authorized this script): training 200 real epochs, hyperparameter tuning,
architecture comparison, and — most importantly — touching Beijing test
2014 for anything. This script builds the test split only long enough to
count its windows for the preprocessing audit (outputs/metrics/
beijing_preprocessing_audit.json); it is never passed to the model.

Usage: python scripts/run_r1_pilot.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf

from src.data.beijing import build_preprocessing_audit, load_beijing_raw, prepare_beijing_r1, subsample_chronological
from src.evaluation.metrics import evaluate_scaled_predictions
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig, evaluate_split_batched
from src.models.stn import STNModel
from src.utils.paths import BEIJING_CSV, OUTPUTS_DIR
from src.utils.seeding import set_experiment_seed

SEED = 42
CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)
MAX_PILOT_EPOCHS = 5
MAX_TRAIN_WINDOWS = 4096
MAX_VAL_WINDOWS = 1024


def main() -> None:
    t_start = time.time()
    set_experiment_seed(SEED)  # must run before STNModel(...) below — see src/utils/seeding.py

    print("=" * 70)
    print("R1 PILOT -- INTEGRATION CHECK ONLY, NOT A FINAL RESULT")
    print("Beijing test 2014 is prepared ONLY to count its windows for the")
    print("preprocessing audit. It is never passed to the model below.")
    print("=" * 70)

    # --- 1. Preprocessing (test included ONLY for the audit's window count) ---
    print("\n[1/5] Preparing Beijing R1 data (chronological split, train-only-fit scalers)...")
    data = prepare_beijing_r1(BEIJING_CSV, include_test=True)
    raw_df = load_beijing_raw(BEIJING_CSV)

    audit = build_preprocessing_audit(data, raw_df)
    audit_path = OUTPUTS_DIR / "metrics" / "beijing_preprocessing_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved preprocessing audit: {audit_path}")
    print(f"  train windows: {audit['n_windows']['train']}")
    print(f"  validation windows: {audit['n_windows']['validation']}")
    print(f"  test windows (prepared for audit only, NOT used below): {audit['n_windows']['test_prepared_not_evaluated']}")
    print(f"  channels ({audit['n_channels']}): {audit['feature_names']}")
    print(f"  pm25 train min/max (ug/m3): {audit['pm25_train_min_max_ug_m3']}")
    print(f"  cbwd categories learned from train: {audit['cbwd_categories_learned_from_train']}")

    # From here on, `data.test` is never referenced again.
    train_split = subsample_chronological(data.train, MAX_TRAIN_WINDOWS)
    val_split = subsample_chronological(data.validation, MAX_VAL_WINDOWS)
    print(f"\n[2/5] Pilot subsample (chronological, not random): "
          f"train={train_split.n_windows} val={val_split.n_windows}")

    # --- 3. Model ---
    print("\n[3/5] Building CANDIDATE_V1 STN...")
    model = STNModel(**CANDIDATE_V1, seed=SEED)
    total_params = None  # reported after first call builds the model

    # --- 4. Training ---
    print(f"\n[4/5] Training pilot: max_epochs={MAX_PILOT_EPOCHS}, batch_size=32, lr=1e-4, seed={SEED}")
    config = TrainerConfig(
        experiment_name="R1_pilot",
        seed=SEED,
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=MAX_PILOT_EPOCHS,
        early_stopping=EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(train_split, val_split)

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)

    # --- 5. Validation evaluation in ug/m3 (never test) ---
    print("\n[5/5] Evaluating VALIDATION in ug/m3 (test 2014 not touched)...")
    val_pred_scaled = evaluate_split_batched(model, val_split, config.batch_size)
    val_metrics_ugm3 = evaluate_scaled_predictions(
        val_split.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25
    )
    print(f"Validation metrics (ug/m3): {val_metrics_ugm3}")

    total_time = time.time() - t_start

    summary = {
        "label": "PILOT DE INTEGRACION -- NO RESULTADO FINAL",
        "seed": SEED,
        "candidate": "CANDIDATE_V1",
        "total_trainable_parameters": total_params,
        "train_windows_used": train_split.n_windows,
        "validation_windows_used": val_split.n_windows,
        "epochs_run": len(history),
        "history": history,
        "validation_metrics_ug_m3": val_metrics_ugm3,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "tensorboard_dir": str(trainer.tensorboard_dir),
        "total_pilot_time_seconds": total_time,
        "beijing_test_2014_evaluated": False,
        "fontibon_used": False,
    }
    summary_path = OUTPUTS_DIR / "metrics" / "R1_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PILOT DE INTEGRACION -- NO RESULTADO FINAL")
    print(f"Total trainable parameters: {total_params}")
    print(f"Epochs run: {len(history)} (max allowed: {MAX_PILOT_EPOCHS})")
    print(f"Validation RMSE (ug/m3): {val_metrics_ugm3['RMSE']:.3f}")
    print(f"Validation MAE (ug/m3): {val_metrics_ugm3['MAE']:.3f}")
    print(f"Validation R2: {val_metrics_ugm3['R2']:.4f}")
    print(f"Checkpoint: {trainer.checkpoint_dir}")
    print(f"Total pilot time: {total_time:.1f}s")
    print(f"Summary saved: {summary_path}")
    print("Beijing test 2014: NOT evaluated. Fontibon: NOT used.")
    print("=" * 70)


if __name__ == "__main__":
    main()
