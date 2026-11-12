"""Full transfer training (Config A/B/C/D) on Fontibon TRAIN+VALIDATION,
one seed at a time. Reuses the same infrastructure already verified by
scripts/run_transfer_pilot.py (src/training/transfer.py, prepare_fontibon_stn)
with NO subsampling (full 10802/2564 windows) and the approved stopping
criteria (max_epochs=200, patience=15, min_delta=0.0, restore_best_weights=True).

Fontibon test is NEVER built. R1/R2 are never touched (B/C/D only READ the
R2 source checkpoint for the SAME seed; A never touches R2 at all).
Refuses to run if that (config, seed)'s summary already exists.

Usage: python scripts/run_transfer_full_training.py --config A --seed 42
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.data.fontibon import prepare_fontibon_stn
from src.evaluation.metrics import (
    evaluate_scaled_predictions,
    residual_diagnostics,
    save_predictions_with_residuals,
)
from src.evaluation.plots import generate_all_r1_development_figures
from src.training.trainer import EarlyStoppingConfig, TrainerConfig, evaluate_split_batched
from src.training.transfer import (
    CANDIDATE_V1,
    TRANSFER_CONFIGS,
    build_trainer_for_config,
    build_transfer_model,
    r2_source_checkpoint_path,
)
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(config_name: str, seed: int) -> None:
    t0 = time.time()
    run_name = f"transfer_{config_name}_seed{seed}"
    summary_path = OUTPUTS_DIR / "metrics" / f"{run_name}_summary.json"
    if summary_path.exists():
        print(f"REFUSING TO RUN: {summary_path} already exists. Not overwriting.")
        sys.exit(1)

    if config_name not in TRANSFER_CONFIGS:
        raise ValueError(f"unknown config {config_name!r}")
    cfg = TRANSFER_CONFIGS[config_name]

    ckpt_path = r2_source_checkpoint_path(seed)
    hash_before = sha256_of(ckpt_path) if cfg.load_source else None

    print(f"[1/6] prepare_fontibon_stn(include_test=False) -- config={config_name} seed={seed}")
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=False)
    assert data.test is None, "Fontibon test must never be built"
    print(f"  train={data.train.n_windows} validation={data.validation.n_windows}")
    assert data.train.n_windows == 10802 and data.validation.n_windows == 2564

    sample_enc = tf.convert_to_tensor(data.train.encoder_input[:1], dtype=tf.float32)
    sample_dec = tf.convert_to_tensor(data.train.decoder_input[:1], dtype=tf.float32)

    print(f"[2/6] building TransferSTNModel config={config_name} seed={seed}...")
    model, build_report = build_transfer_model(config_name, seed, sample_enc, sample_dec, architecture=CANDIDATE_V1)

    encoder_before = [w.copy() for w in model.encoder.get_weights()]
    decoder_before = [w.copy() for w in model.decoder.get_weights()]

    es = EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True)
    trainer_config = TrainerConfig(
        experiment_name=f"transfer_{config_name}", seed=seed,
        learning_rate=cfg.decoder_head_lr, batch_size=32, max_epochs=200,
        early_stopping=es,
    )
    trainer = build_trainer_for_config(config_name, model, trainer_config)

    print(f"[3/6] training: max_epochs=200, batch_size=32, patience=15, seed={seed}")
    history = trainer.fit(data.train, data.validation)  # incremental history save every epoch; checkpoint on each improvement

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    for row in history:
        assert np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"])
        assert np.isfinite(row["gradient_global_norm_mean"])

    encoder_after = model.encoder.get_weights()
    decoder_after = model.decoder.get_weights()
    encoder_frozen_ok = (all(np.array_equal(b, a) for b, a in zip(encoder_before, encoder_after)) if cfg.freeze_encoder else None)
    decoder_frozen_ok = (all(np.array_equal(b, a) for b, a in zip(decoder_before, decoder_after)) if cfg.freeze_decoder else None)
    if cfg.freeze_encoder:
        assert encoder_frozen_ok, "encoder changed despite being frozen!"
    if cfg.freeze_decoder:
        assert decoder_frozen_ok, "decoder changed despite being frozen!"

    print("[4/6] evaluating TRAIN and VALIDATION (ug/m3)...")
    train_pred_scaled = evaluate_split_batched(model, data.train, trainer_config.batch_size)
    val_pred_scaled = evaluate_split_batched(model, data.validation, trainer_config.batch_size)
    train_metrics = evaluate_scaled_predictions(data.train.y_scaled, train_pred_scaled, data.scalers.inverse_transform_pm25)
    val_metrics = evaluate_scaled_predictions(data.validation.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25)

    val_pred_ugm3 = data.scalers.inverse_transform_pm25(val_pred_scaled)
    val_real_ugm3 = data.validation.y_original.ravel()
    residuals = residual_diagnostics(val_real_ugm3, val_pred_ugm3)

    predictions_path = OUTPUTS_DIR / "metrics" / f"{run_name}_validation_predictions.csv"
    save_predictions_with_residuals(data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, predictions_path)

    print("[5/6] generating figures...")
    figures_dir = OUTPUTS_DIR / "figures" / run_name
    generate_all_r1_development_figures(history, data.validation.target_timestamps, val_real_ugm3, val_pred_ugm3, figures_dir)

    hash_after = sha256_of(ckpt_path) if cfg.load_source else None
    r2_checkpoint_intact = (hash_before == hash_after) if cfg.load_source else None
    if cfg.load_source:
        assert r2_checkpoint_intact, f"R2 source checkpoint {ckpt_path} CHANGED during training!"

    total_time = time.time() - t0
    epoch_times = [row["epoch_time_seconds"] for row in history]

    summary = {
        "experiment": f"transfer_{config_name}", "config": config_name, "seed": seed,
        "status": "train_validation_only", "best_epoch": trainer.best_epoch, "stopping_epoch": trainer.stopping_epoch,
        "total_epochs": len(history), "best_val_loss_scaled": trainer.best_val_loss,
        "train_metrics_ug_m3": train_metrics, "validation_metrics_ug_m3": val_metrics,
        "validation_residual_diagnostics_ug_m3": residuals,
        "total_trainable_parameters": total_params,
        "total_training_time_seconds": total_time,
        "mean_epoch_time_seconds": float(np.mean(epoch_times)),
        "gradient_norm_initial_epoch": history[0]["gradient_global_norm_mean"],
        "gradient_norm_final_epoch": history[-1]["gradient_global_norm_mean"],
        "gradient_norm_max_over_training": max(row["gradient_global_norm_max"] for row in history),
        "build_report": {k: v for k, v in build_report.items() if k != "compatibility_report"},
        "compatibility_report": build_report.get("compatibility_report"),
        "r2_source_checkpoint_sha256_before": hash_before, "r2_source_checkpoint_sha256_after": hash_after,
        "r2_source_checkpoint_intact": r2_checkpoint_intact,
        "encoder_frozen_confirmed_unchanged": encoder_frozen_ok, "decoder_frozen_confirmed_unchanged": decoder_frozen_ok,
        "predictions_path": str(predictions_path), "figures_dir": str(figures_dir),
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "beijing_test_2014_used": False, "fontibon_test_used": False, "r1_or_r2_modified": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"[6/6] saved {summary_path}")
    print(f"RESULT config={config_name} seed={seed} best_epoch={trainer.best_epoch} stopping_epoch={trainer.stopping_epoch} "
          f"val_RMSE={val_metrics['RMSE']:.3f} val_MAE={val_metrics['MAE']:.3f} val_R2={val_metrics['R2']:.4f} "
          f"time={total_time:.1f}s status=OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(TRANSFER_CONFIGS))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.seed)
