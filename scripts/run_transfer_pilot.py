"""Transfer pilot (Config A/B/C/D) -- INTEGRATION CHECK ONLY, NOT a final
scientific result. Verifies the full chain (Fontibon pipeline -> transfer
model build/freeze/load -> training -> evaluation) works end to end on a
small, deterministic, chronological subsample shared by all four configs
(subsample_chronological truncates to the first N windows -- no
randomness, so every config sees IDENTICAL train/validation observations).

Never builds/uses Fontibon test. Never touches Beijing test 2014. Never
retrains R1/R2. Only seed 42 is authorized for this pilot round.

Usage:
    python scripts/run_transfer_pilot.py --config A --seed 42 --max-epochs 2 \
        --max-train-windows 512 --max-validation-windows 128
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

from src.data.beijing import subsample_chronological
from src.data.fontibon import prepare_fontibon_stn
from src.evaluation.metrics import evaluate_scaled_predictions
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


def main(config_name: str, seed: int, max_epochs: int, max_train_windows: int, max_validation_windows: int, batch_size: int) -> None:
    t0 = time.time()
    run_name = f"transfer_pilot_{config_name}_seed{seed}"
    summary_path = OUTPUTS_DIR / "metrics" / f"{run_name}_summary.json"
    if summary_path.exists():
        print(f"REFUSING TO RUN: {summary_path} already exists. Not overwriting.")
        sys.exit(1)

    if config_name not in TRANSFER_CONFIGS:
        raise ValueError(f"unknown config {config_name!r}")
    cfg = TRANSFER_CONFIGS[config_name]

    ckpt_path = r2_source_checkpoint_path(seed)
    hash_before = sha256_of(ckpt_path) if cfg.load_source else None

    print(f"[1/5] prepare_fontibon_stn(include_test=False) -- config={config_name} seed={seed}")
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=False)
    assert data.test is None, "Fontibon test must never be built in this pilot"

    train_split = subsample_chronological(data.train, max_train_windows)
    val_split = subsample_chronological(data.validation, max_validation_windows)
    print(f"  train={train_split.n_windows} val={val_split.n_windows} (deterministic, chronological, shared across A/B/C/D)")

    sample_enc = tf.convert_to_tensor(train_split.encoder_input[:1], dtype=tf.float32)
    sample_dec = tf.convert_to_tensor(train_split.decoder_input[:1], dtype=tf.float32)

    print(f"[2/5] building TransferSTNModel config={config_name}...")
    model, build_report = build_transfer_model(config_name, seed, sample_enc, sample_dec, architecture=CANDIDATE_V1)

    encoder_before = [w.copy() for w in model.encoder.get_weights()]
    decoder_before = [w.copy() for w in model.decoder.get_weights()]
    head_before = [w.copy() for w in model.output_head.get_weights()]

    es = EarlyStoppingConfig(monitor="val_loss", patience=15, min_delta=0.0, restore_best_weights=True)
    trainer_config = TrainerConfig(
        experiment_name=f"transfer_pilot_{config_name}", seed=seed,
        learning_rate=cfg.decoder_head_lr, batch_size=batch_size, max_epochs=max_epochs,
        early_stopping=es,
    )
    trainer = build_trainer_for_config(config_name, model, trainer_config)

    print(f"[3/5] training max_epochs={max_epochs} batch_size={batch_size}...")
    history = trainer.fit(train_split, val_split)

    for row in history:
        assert np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"])
        assert np.isfinite(row["gradient_global_norm_mean"])
    all_finite = True

    encoder_after = model.encoder.get_weights()
    decoder_after = model.decoder.get_weights()
    head_after = model.output_head.get_weights()

    encoder_frozen_ok = (all(np.array_equal(b, a) for b, a in zip(encoder_before, encoder_after)) if cfg.freeze_encoder else None)
    decoder_frozen_ok = (all(np.array_equal(b, a) for b, a in zip(decoder_before, decoder_after)) if cfg.freeze_decoder else None)
    head_updated = not all(np.allclose(b, a) for b, a in zip(head_before, head_after))

    if cfg.freeze_encoder:
        assert encoder_frozen_ok, "encoder weights changed despite being frozen!"
    else:
        assert not all(np.array_equal(b, a) for b, a in zip(encoder_before, encoder_after)), "encoder should have updated but did not"
    if cfg.freeze_decoder:
        assert decoder_frozen_ok, "decoder weights changed despite being frozen!"
    assert head_updated, "output head did not update during training!"

    print("[4/5] evaluating validation (ug/m3)...")
    val_pred_scaled = evaluate_split_batched(model, val_split, batch_size)
    val_metrics = evaluate_scaled_predictions(val_split.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25)

    hash_after = sha256_of(ckpt_path) if cfg.load_source else None
    r2_checkpoint_intact = (hash_before == hash_after) if cfg.load_source else None
    if cfg.load_source:
        assert r2_checkpoint_intact, f"R2 source checkpoint {ckpt_path} CHANGED during the pilot!"

    total_time = time.time() - t0

    summary = {
        "label": "TRANSFER PILOT -- INTEGRATION CHECK ONLY, NO RESULTADO CIENTIFICO DEFINITIVO",
        "config": config_name, "seed": seed,
        "train_windows_used": train_split.n_windows, "validation_windows_used": val_split.n_windows,
        "epochs_run": len(history),
        "final_train_loss_scaled": history[-1]["train_loss"], "final_val_loss_scaled": history[-1]["val_loss"],
        "validation_metrics_ug_m3": val_metrics,
        "history": history,
        "build_report": {k: v for k, v in build_report.items() if k != "compatibility_report"},
        "compatibility_report": build_report.get("compatibility_report"),
        "r2_source_checkpoint_path": str(ckpt_path) if cfg.load_source else None,
        "r2_source_checkpoint_sha256_before": hash_before,
        "r2_source_checkpoint_sha256_after": hash_after,
        "r2_source_checkpoint_intact": r2_checkpoint_intact,
        "output_head_loaded_from_source": build_report.get("output_head_loaded", False),
        "encoder_frozen_confirmed_unchanged": encoder_frozen_ok,
        "decoder_frozen_confirmed_unchanged": decoder_frozen_ok,
        "head_confirmed_updated": head_updated,
        "all_losses_and_gradients_finite": all_finite,
        "total_pilot_time_seconds": total_time,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "status": "OK",
        "beijing_test_2014_used": False,
        "fontibon_test_used": False,
        "r1_or_r2_modified": False,
        "other_seeds_used": False,
        "full_training_executed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"[5/5] saved {summary_path}")
    print(f"RESULT config={config_name} seed={seed} val_RMSE={val_metrics['RMSE']:.3f} "
          f"val_loss={history[-1]['val_loss']:.5f} time={total_time:.1f}s status=OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(TRANSFER_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--max-validation-windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.config, args.seed, args.max_epochs, args.max_train_windows, args.max_validation_windows, args.batch_size)
