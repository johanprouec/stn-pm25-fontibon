"""R2 source training -- cross-seed stability comparison (42, 123, 2026).

Reads the three seeds' summary JSONs (must all exist already -- this script
trains nothing) and builds the comparison table + mean/std + a descriptive
stability diagnostic. Never touches Beijing test 2014 or Fontibon; only
reads files already produced by scripts/run_r2_source_training.py.

No seed is presented as "the" R2 source checkpoint -- all three are kept
for the future seed-to-seed pairing with local Fontibon transfer runs (see
docs/IMPLEMENTATION_ASSUMPTIONS.md).

Usage: python scripts/compare_r2_source_seeds.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluation.plots import plot_metrics_across_seeds
from src.utils.paths import OUTPUTS_DIR

SEEDS = [42, 123, 2026]


def main() -> None:
    summaries = {}
    for seed in SEEDS:
        path = OUTPUTS_DIR / "metrics" / f"R2_source_seed{seed}_summary.json"
        if not path.exists():
            print(f"MISSING: {path} -- seed {seed} has not finished (or hasn't started). Stopping.")
            sys.exit(1)
        summaries[seed] = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for seed in SEEDS:
        s = summaries[seed]
        rows.append({
            "seed": seed,
            "best_epoch": s["best_epoch"],
            "stopping_epoch": s["stopping_epoch"],
            "total_epochs": s["total_epochs"],
            "train_RMSE": s["train_metrics_ug_m3"]["RMSE"],
            "val_RMSE": s["validation_metrics_ug_m3"]["RMSE"],
            "val_MAE": s["validation_metrics_ug_m3"]["MAE"],
            "val_R2": s["validation_metrics_ug_m3"]["R2"],
            "bias": s["validation_residual_diagnostics_ug_m3"]["bias_mean"],
            "training_time_seconds": s["total_training_time_seconds"],
            "gradient_norm_initial": s["gradient_norm_initial_epoch"],
            "gradient_norm_final": s["gradient_norm_final_epoch"],
            "gradient_norm_max": s["gradient_norm_max_over_training"],
            "compatible_from_checkpoint": s["compatibility_report_from_checkpoint"]["compatible"],
        })

    print("Seed | best_ep | stop_ep | total_ep | train_RMSE | val_RMSE | val_MAE | val_R2  | bias    | time(s) | grad_init | grad_final | grad_max | compatible")
    for r in rows:
        print(f"{r['seed']:5d}| {r['best_epoch']:7d} | {r['stopping_epoch']:7d} | {r['total_epochs']:8d} | "
              f"{r['train_RMSE']:10.3f} | {r['val_RMSE']:8.3f} | {r['val_MAE']:7.3f} | {r['val_R2']:.4f} | "
              f"{r['bias']:7.3f} | {r['training_time_seconds']:7.1f} | "
              f"{r['gradient_norm_initial']:9.4f} | {r['gradient_norm_final']:10.4f} | {r['gradient_norm_max']:8.3f} | "
              f"{r['compatible_from_checkpoint']}")

    val_rmse = np.array([r["val_RMSE"] for r in rows])
    val_mae = np.array([r["val_MAE"] for r in rows])
    val_r2 = np.array([r["val_R2"] for r in rows])

    stats = {
        "val_RMSE_mean": float(val_rmse.mean()), "val_RMSE_std": float(val_rmse.std(ddof=1)),
        "val_MAE_mean": float(val_mae.mean()), "val_MAE_std": float(val_mae.std(ddof=1)),
        "val_R2_mean": float(val_r2.mean()), "val_R2_std": float(val_r2.std(ddof=1)),
    }

    print(f"\nValidation RMSE: {stats['val_RMSE_mean']:.3f} ± {stats['val_RMSE_std']:.3f}")
    print(f"Validation MAE:  {stats['val_MAE_mean']:.3f} ± {stats['val_MAE_std']:.3f}")
    print(f"Validation R2:   {stats['val_R2_mean']:.4f} ± {stats['val_R2_std']:.4f}")

    cv_rmse = stats["val_RMSE_std"] / stats["val_RMSE_mean"] if stats["val_RMSE_mean"] else float("nan")
    best_epoch_spread = max(r["best_epoch"] for r in rows) - min(r["best_epoch"] for r in rows)
    all_compatible = all(r["compatible_from_checkpoint"] for r in rows)

    diagnostics = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "val_RMSE_coefficient_of_variation": cv_rmse,
        "best_epoch_spread": best_epoch_spread,
        "all_checkpoints_compatible_with_fontibon_shaped_target": all_compatible,
        "note": (
            "Descriptive only. No seed is presented as 'the' R2 source checkpoint -- all three "
            "are kept, paired by seed with the corresponding local Fontibon transfer run "
            "(local seed 42 <- R2 source seed 42, etc; see docs/IMPLEMENTATION_ASSUMPTIONS.md). "
            "R2 is source pretraining, not a new R1 replication -- not compared against "
            "Zhang & Zhang's published Beijing RMSE=19.04. Beijing test 2014 not evaluated for "
            "any seed. Fontibon not trained or evaluated."
        ),
    }

    payload = {
        "rows": rows,
        "validation_stats": stats,
        "stability_diagnostics": diagnostics,
        "beijing_test_2014_evaluated": False,
        "fontibon_trained_or_evaluated": False,
    }
    out_path = OUTPUTS_DIR / "metrics" / "R2_source_seed_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    figures_dir = OUTPUTS_DIR / "figures" / "R2_source_comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = {seed: summaries[seed]["validation_metrics_ug_m3"] for seed in SEEDS}
    fig_path = figures_dir / "R2_validation_metrics_across_seeds.png"
    plot_metrics_across_seeds(
        seed_metrics, fig_path,
        title="R2 source — VALIDATION metrics por semilla (referencia descriptiva, sin selección de ganador)",
    )
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
