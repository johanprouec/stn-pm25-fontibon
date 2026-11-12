"""R1 — FINAL, ONE-TIME evaluation on Beijing test 2014.

Runs ONLY after R1 Development was approved and frozen (see
outputs/metrics/R1_frozen_protocol.json, written before this script ever
touches test data). For each of the three frozen seeds (42, 123, 2026):
rebuild the exact frozen architecture, load its existing best checkpoint,
run inference ONLY (training=False, no optimizer, no gradient step) over
the 8,487 test windows, inverse-transform, and compute metrics in ug/m3.

No training happens anywhere in this script. No architecture or
hyperparameter is chosen or adjusted based on what test shows. Fontibon is
never referenced.

Usage: python scripts/run_r1_test_evaluation.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.beijing import prepare_beijing_r1, persistence_prediction
from src.evaluation.metrics import (
    evaluate_scaled_predictions,
    residual_diagnostics,
    save_predictions_with_residuals,
    target_distribution,
)
from src.evaluation.plots import plot_metrics_across_seeds, plot_predictions_vs_real, plot_scatter
from src.models.stn import STNModel
from src.training.trainer import evaluate_split_batched
from src.utils.paths import BEIJING_CSV, OUTPUTS_DIR

SEEDS = [42, 123, 2026]
CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)
PAPER_REFERENCE = {"RMSE": 19.04, "MAE": 11.13, "R2": 0.937}  # Zhang & Zhang (2023), Beijing t+1


def main() -> None:
    t_start = time.time()

    frozen_path = OUTPUTS_DIR / "metrics" / "R1_frozen_protocol.json"
    if not frozen_path.exists():
        print(f"REFUSING TO RUN: {frozen_path} does not exist. The protocol must be frozen "
              f"and documented before Beijing test 2014 is ever touched.")
        sys.exit(1)
    print("Frozen protocol found:", frozen_path)

    print("=" * 70)
    print("R1 -- FINAL EVALUATION ON BEIJING TEST 2014 (ONE TIME ONLY)")
    print("Architecture, hyperparameters, preprocessing, splits and checkpoints are FROZEN.")
    print("No training. No tuning. No seed selection. Fontibon not used.")
    print("=" * 70)

    # --- 1. Data, WITH test this time -------------------------------------------
    print("\n[1/5] Preparing Beijing R1 data (include_test=True)...")
    data = prepare_beijing_r1(BEIJING_CSV, include_test=True)
    print(f"  train windows: {data.train.n_windows} | validation windows: {data.validation.n_windows} | "
          f"test windows: {data.test.n_windows}")
    assert data.test.n_windows == 8487, (
        f"test window count {data.test.n_windows} != audited 8487 — the frozen pipeline must "
        f"reproduce exactly the audited count; stopping rather than silently proceeding."
    )

    test_target_dist = target_distribution(data.test.y_original)
    print(f"  test target distribution (ug/m3): {test_target_dist}")

    # --- 2. Persistence sanity check on test -------------------------------------
    print("\n[2/5] Persistence baseline on TEST (sanity check only)...")
    persistence_pred_scaled = persistence_prediction(data.test, data.feature_names)
    persistence_metrics = evaluate_scaled_predictions(
        data.test.y_scaled, persistence_pred_scaled, data.scalers.inverse_transform_pm25
    )
    print(f"  persistence (ug/m3): {persistence_metrics}")

    # --- 3. Per-seed inference-only evaluation -----------------------------------
    print("\n[3/5] Per-seed inference on TEST (training=False only, no optimizer)...")
    figures_dir = OUTPUTS_DIR / "figures" / "R1_test_final"
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics = {}
    per_seed_results = {}
    for seed in SEEDS:
        checkpoint_path = OUTPUTS_DIR / "checkpoints" / f"R1_development_seed{seed}" / "best.weights.h5"
        if not checkpoint_path.exists():
            print(f"REFUSING TO CONTINUE: checkpoint missing for seed {seed}: {checkpoint_path}")
            sys.exit(1)

        model = STNModel(**CANDIDATE_V1, seed=seed)
        model((data.test.encoder_input[:2], data.test.decoder_input[:2]), training=False)  # build
        model.load_weights(str(checkpoint_path))

        test_pred_scaled = evaluate_split_batched(model, data.test, batch_size=32)
        test_pred_ugm3 = data.scalers.inverse_transform_pm25(test_pred_scaled)
        test_real_ugm3 = data.test.y_original.ravel()

        metrics = evaluate_scaled_predictions(data.test.y_scaled, test_pred_scaled, data.scalers.inverse_transform_pm25)
        residuals = residual_diagnostics(test_real_ugm3, test_pred_ugm3)

        print(f"  seed={seed}: RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.4f} "
              f"bias={residuals['bias_mean']:.3f} std={residuals['std']:.3f}")

        predictions_path = OUTPUTS_DIR / "metrics" / f"R1_test_seed{seed}_predictions.csv"
        save_predictions_with_residuals(data.test.target_timestamps, test_real_ugm3, test_pred_ugm3, predictions_path)

        seed_summary = {
            "seed": seed,
            "split": "test_2014",
            "checkpoint_used": str(checkpoint_path),
            "inference_only": True,
            "training_performed": False,
            "n_test_windows": data.test.n_windows,
            "test_metrics_ug_m3": metrics,
            "test_residual_diagnostics_ug_m3": residuals,
            "predictions_path": str(predictions_path),
            "fontibon_used": False,
        }
        summary_path = OUTPUTS_DIR / "metrics" / f"R1_test_seed{seed}_summary.json"
        summary_path.write_text(json.dumps(seed_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        per_seed_metrics[seed] = metrics
        per_seed_results[seed] = {
            "metrics": metrics, "residuals": residuals,
            "pred_ugm3": test_pred_ugm3, "real_ugm3": test_real_ugm3,
        }

        plot_predictions_vs_real(
            data.test.target_timestamps, test_real_ugm3, test_pred_ugm3,
            figures_dir / f"test_predictions_vs_real_seed{seed}.png", split_label="Test 2014",
        )
        plot_scatter(test_real_ugm3, test_pred_ugm3, figures_dir / f"test_scatter_seed{seed}.png", split_label="Test 2014")

    plot_metrics_across_seeds(per_seed_metrics, figures_dir / "test_metrics_across_seeds.png")

    # --- 4. Aggregate across the three seeds -------------------------------------
    print("\n[4/5] Aggregating across the three seeds...")
    rmse_arr = np.array([per_seed_metrics[s]["RMSE"] for s in SEEDS])
    mae_arr = np.array([per_seed_metrics[s]["MAE"] for s in SEEDS])
    r2_arr = np.array([per_seed_metrics[s]["R2"] for s in SEEDS])

    agg = {
        "RMSE_mean": float(rmse_arr.mean()), "RMSE_std": float(rmse_arr.std(ddof=1)),
        "MAE_mean": float(mae_arr.mean()), "MAE_std": float(mae_arr.std(ddof=1)),
        "R2_mean": float(r2_arr.mean()), "R2_std": float(r2_arr.std(ddof=1)),
    }
    print(f"  Test RMSE: {agg['RMSE_mean']:.3f} +/- {agg['RMSE_std']:.3f}")
    print(f"  Test MAE:  {agg['MAE_mean']:.3f} +/- {agg['MAE_std']:.3f}")
    print(f"  Test R2:   {agg['R2_mean']:.4f} +/- {agg['R2_std']:.4f}")

    # Descriptive-only difference vs the paper's published reference — NOT a
    # claim of exact replication (see docs/IMPLEMENTATION_ASSUMPTIONS.md for
    # the documented methodological differences).
    paper_diff = {
        "RMSE_diff_ours_minus_paper": agg["RMSE_mean"] - PAPER_REFERENCE["RMSE"],
        "MAE_diff_ours_minus_paper": agg["MAE_mean"] - PAPER_REFERENCE["MAE"],
        "R2_diff_ours_minus_paper": agg["R2_mean"] - PAPER_REFERENCE["R2"],
    }

    total_time = time.time() - t_start

    final_summary = {
        "label": "R1 FINAL TEST EVALUATION -- Beijing test 2014, evaluated exactly once",
        "seeds": SEEDS,
        "n_test_windows": data.test.n_windows,
        "per_seed_test_metrics_ug_m3": per_seed_metrics,
        "aggregate_test_metrics_ug_m3": agg,
        "persistence_test_metrics_ug_m3": persistence_metrics,
        "test_target_distribution_ug_m3": test_target_dist,
        "paper_reference_zhang_zhang_2023": PAPER_REFERENCE,
        "descriptive_difference_vs_paper_reference": paper_diff,
        "comparability_caveat": (
            "This is a comparison against a published reference result, NOT a claim of exact "
            "replication. Documented methodological differences: reduced STN (2 encoder / 1 "
            "decoder vs. the paper's 3/2); chronological validation split vs. the paper's random "
            "validation split; an explicit anti-leakage protocol for missing-data interpolation "
            "not described in the paper; MinMax scaling fit exclusively on TRAIN, whereas the "
            "paper does not specify its fitting scope; and our own choices for d_model, heads, "
            "d_ff, dropout, factor and decoder label length, none of which the paper reports. "
            "See docs/IMPLEMENTATION_ASSUMPTIONS.md."
        ),
        "total_evaluation_time_seconds": total_time,
        "training_performed": False,
        "architecture_modified_after_seeing_test": False,
        "seed_selected_based_on_test": False,
        "fontibon_used": False,
    }
    final_path = OUTPUTS_DIR / "metrics" / "R1_test_final_summary.json"
    final_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("R1 FINAL TEST EVALUATION COMPLETE (Beijing test 2014, evaluated once)")
    print(f"Test RMSE: {agg['RMSE_mean']:.3f} +/- {agg['RMSE_std']:.3f} ug/m3  "
          f"(paper reference: {PAPER_REFERENCE['RMSE']})")
    print(f"Test MAE:  {agg['MAE_mean']:.3f} +/- {agg['MAE_std']:.3f} ug/m3  "
          f"(paper reference: {PAPER_REFERENCE['MAE']})")
    print(f"Test R2:   {agg['R2_mean']:.4f} +/- {agg['R2_std']:.4f}  "
          f"(paper reference: {PAPER_REFERENCE['R2']})")
    print(f"Persistence (test, sanity check): RMSE={persistence_metrics['RMSE']:.3f} "
          f"MAE={persistence_metrics['MAE']:.3f} R2={persistence_metrics['R2']:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary saved: {final_path}")
    print("No training was performed. Architecture/hyperparameters NOT changed after seeing test.")
    print("=" * 70)


if __name__ == "__main__":
    main()
