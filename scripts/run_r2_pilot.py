"""R2 PILOT — integration check only, NOT a real training run.

Verifies: 3-channel Beijing preprocessing -> decoder -> STN -> gradients ->
checkpoint -> inverse transform -> validation metrics, end to end, on a
short subsample of Beijing R2 train/validation. Also audits structural
compatibility between an R2-shaped (source) and a Fontibon-shaped (target)
STNModel, BEFORE any real transfer is attempted.

Forbidden here (see the review that authorized this script): Beijing test
2014, training or evaluating on Fontibon (validation or test), A/B/C/D,
the full 3-seed R2 run, and reusing R1's checkpoints as a starting point
(R2 trains its 3-channel input projection from scratch).

Usage: python scripts/run_r2_pilot.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.data.beijing import build_r2_preprocessing_audit, prepare_beijing_r2, subsample_chronological
from src.data.fontibon import load_fontibon_features
from src.evaluation.metrics import evaluate_scaled_predictions
from src.models.stn import STNModel
from src.models.transfer_compat import compare_transfer_weight_shapes
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig, evaluate_split_batched
from src.utils.paths import BEIJING_CSV, FONTIBON_CSV, OUTPUTS_DIR
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

    print("=" * 70)
    print("R2 PILOT -- INTEGRATION CHECK ONLY, NOT A FINAL RESULT")
    print("3-channel Beijing transfer-source preprocessing. Beijing test 2014")
    print("and Fontibon training/validation/test are NEVER touched below.")
    print("=" * 70)

    # --- 1. Data: Beijing R2, test never built ---------------------------------
    print("\n[1/6] Preparing Beijing R2 data (3 channels: pm25, temp, dew; include_test=False)...")
    data = prepare_beijing_r2(BEIJING_CSV, include_test=False)
    print(f"  feature_names: {data.feature_names}")
    print(f"  train windows: {data.train.n_windows} | validation windows: {data.validation.n_windows}")
    assert data.test is None

    # --- 2. Structural compatibility audit: R2 source vs Fontibon-shaped target ---
    print("\n[2/6] Auditing source/target weight-shape compatibility (architecture only, no training)...")
    set_experiment_seed(SEED)
    source_model = STNModel(**CANDIDATE_V1, seed=SEED)
    source_model((data.train.encoder_input[:1], data.train.decoder_input[:1]), training=False)  # build

    # Target: Fontibon-shaped model, built with real Fontibon INPUT features
    # only (never a target/label, never train/val/test-specific splitting)
    # purely to exercise a realistic (24,3) input shape.
    fontibon_features = load_fontibon_features(FONTIBON_CSV)
    sample_window = fontibon_features.iloc[:24].to_numpy(dtype=np.float32)[np.newaxis, :, :]
    sample_decoder = np.concatenate(
        [sample_window[:, -12:, 0], np.zeros((1, 1), dtype=np.float32)], axis=1
    )[..., np.newaxis]
    target_model = STNModel(**CANDIDATE_V1, seed=SEED + 1)  # different seed on purpose: independent init, not a copy of source
    target_model((tf.constant(sample_window), tf.constant(sample_decoder)), training=False)

    compatibility_report = compare_transfer_weight_shapes(source_model, target_model)
    print(f"  compatible: {compatibility_report['compatible']}")
    if not compatibility_report["compatible"]:
        print(f"  MISMATCHES: {compatibility_report}")

    audit = build_r2_preprocessing_audit(data, compatibility_report)
    audit_path = OUTPUTS_DIR / "metrics" / "R2_preprocessing_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved: {audit_path}")
    print(f"  encoder_input_shape: {audit['encoder_input_shape']} | decoder_input_shape: {audit['decoder_input_shape']}")
    print(f"  pm25 train min/max: {audit['pm25_train_min_max_ug_m3']}")
    print(f"  temp train min/max: {audit['temp_train_min_max']}")
    print(f"  dew train min/max: {audit['dew_train_min_max']}")

    # --- 3. Pilot subsample (chronological, not random) -------------------------
    train_split = subsample_chronological(data.train, MAX_TRAIN_WINDOWS)
    val_split = subsample_chronological(data.validation, MAX_VAL_WINDOWS)
    print(f"\n[3/6] Pilot subsample: train={train_split.n_windows} val={val_split.n_windows}")

    # --- 4. Model: trained FROM SCRATCH, not from R1's checkpoints ---------------
    print("\n[4/6] Building CANDIDATE_V1 STN from scratch (R2 does NOT reuse R1's 11-channel weights)...")
    set_experiment_seed(SEED)
    model = STNModel(**CANDIDATE_V1, seed=SEED)
    total_params = None

    # --- 5. Pilot training --------------------------------------------------------
    print(f"\n[5/6] Training pilot: max_epochs={MAX_PILOT_EPOCHS}, batch_size=32, lr=1e-4, seed={SEED}")
    config = TrainerConfig(
        experiment_name="R2_pilot",
        seed=SEED,
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=MAX_PILOT_EPOCHS,
        early_stopping=EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True),
    )
    trainer = Trainer(model, config)
    history = trainer.fit(train_split, val_split)

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)

    # --- 6. Validation metrics in ug/m3 -------------------------------------------
    print("\n[6/6] Evaluating VALIDATION in ug/m3 (Beijing test 2014 and Fontibon not touched)...")
    val_pred_scaled = evaluate_split_batched(model, val_split, config.batch_size)
    val_metrics_ugm3 = evaluate_scaled_predictions(
        val_split.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25
    )
    print(f"Validation metrics (ug/m3): {val_metrics_ugm3}")

    total_time = time.time() - t_start

    summary = {
        "label": "R2 PILOT DE INTEGRACION -- NO RESULTADO FINAL",
        "seed": SEED,
        "candidate": "CANDIDATE_V1",
        "feature_names": data.feature_names,
        "total_trainable_parameters": total_params,
        "train_windows_used": train_split.n_windows,
        "validation_windows_used": val_split.n_windows,
        "epochs_run": len(history),
        "history": history,
        "validation_metrics_ug_m3": val_metrics_ugm3,
        "compatibility_report": compatibility_report,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "tensorboard_dir": str(trainer.tensorboard_dir),
        "total_pilot_time_seconds": total_time,
        "beijing_test_2014_used": False,
        "fontibon_trained_or_evaluated": False,
        "transfer_ab_cd_executed": False,
        "reused_r1_checkpoints": False,
    }
    summary_path = OUTPUTS_DIR / "metrics" / "R2_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("R2 PILOT DE INTEGRACION -- NO RESULTADO FINAL")
    print(f"Total trainable parameters: {total_params}")
    print(f"Epochs run: {len(history)} (max allowed: {MAX_PILOT_EPOCHS})")
    print(f"Validation RMSE (ug/m3): {val_metrics_ugm3['RMSE']:.3f}")
    print(f"Validation MAE (ug/m3): {val_metrics_ugm3['MAE']:.3f}")
    print(f"Validation R2: {val_metrics_ugm3['R2']:.4f}")
    print(f"Source/target weight-shape compatible: {compatibility_report['compatible']}")
    print(f"Checkpoint: {trainer.checkpoint_dir}")
    print(f"Total pilot time: {total_time:.1f}s")
    print(f"Summary saved: {summary_path}")
    print("Beijing test 2014: NOT used. Fontibon: NOT trained/evaluated. A/B/C/D: NOT executed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
