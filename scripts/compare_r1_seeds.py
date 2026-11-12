"""R1 Development — cross-seed stability comparison (42, 123, 2026).

Reads the three seeds' summary JSONs (must all exist already — this script
trains nothing) and builds the comparison table + mean/std + a descriptive
stability diagnostic. Never touches Beijing test 2014 or Fontibon; only
reads files already produced by scripts/run_r1_development.py.

Usage: python scripts/compare_r1_seeds.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.paths import OUTPUTS_DIR

SEEDS = [42, 123, 2026]


def main() -> None:
    summaries = {}
    for seed in SEEDS:
        path = OUTPUTS_DIR / "metrics" / f"R1_development_seed{seed}_summary.json"
        if not path.exists():
            print(f"MISSING: {path} — seed {seed} has not finished (or hasn't started). Stopping.")
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
            "training_time_seconds": s["total_training_time_seconds"],
            "sum_epoch_time_seconds": s.get("sum_epoch_time_seconds"),
            "gradient_norm_initial": s["gradient_norm_initial_epoch"],
            "gradient_norm_final": s["gradient_norm_final_epoch"],
            "gradient_norm_max": s["gradient_norm_max_over_training"],
            "classification_vs_persistence": s["descriptive_classification_vs_persistence"],
        })

    print("Seed | best_ep | stop_ep | total_ep | train_RMSE | val_RMSE | val_MAE | val_R2  | grad_init | grad_final | grad_max")
    for r in rows:
        print(f"{r['seed']:5d}| {r['best_epoch']:7d} | {r['stopping_epoch']:7d} | {r['total_epochs']:8d} | "
              f"{r['train_RMSE']:10.3f} | {r['val_RMSE']:8.3f} | {r['val_MAE']:7.3f} | {r['val_R2']:.4f} | "
              f"{r['gradient_norm_initial']:9.4f} | {r['gradient_norm_final']:10.4f} | {r['gradient_norm_max']:8.3f}")

    val_rmse = np.array([r["val_RMSE"] for r in rows])
    val_mae = np.array([r["val_MAE"] for r in rows])
    val_r2 = np.array([r["val_R2"] for r in rows])

    stats = {
        "val_RMSE_mean": float(val_rmse.mean()), "val_RMSE_std": float(val_rmse.std(ddof=1)),
        "val_MAE_mean": float(val_mae.mean()), "val_MAE_std": float(val_mae.std(ddof=1)),
        "val_R2_mean": float(val_r2.mean()), "val_R2_std": float(val_r2.std(ddof=1)),
    }

    persistence = summaries[SEEDS[0]]["persistence_validation_metrics_ug_m3"]

    print(f"\nValidation RMSE: {stats['val_RMSE_mean']:.3f} ± {stats['val_RMSE_std']:.3f}")
    print(f"Validation MAE:  {stats['val_MAE_mean']:.3f} ± {stats['val_MAE_std']:.3f}")
    print(f"Validation R2:   {stats['val_R2_mean']:.4f} ± {stats['val_R2_std']:.4f}")
    print(f"\nPersistence (reference, not seed-dependent): RMSE={persistence['RMSE']:.3f} "
          f"MAE={persistence['MAE']:.3f} R2={persistence['R2']:.4f}")

    # Descriptive stability diagnostics — no automatic model changes, no
    # "winner" selection. Coefficient of variation as a simple, transparent
    # dispersion indicator.
    cv_rmse = stats["val_RMSE_std"] / stats["val_RMSE_mean"] if stats["val_RMSE_mean"] else float("nan")
    best_epoch_spread = max(r["best_epoch"] for r in rows) - min(r["best_epoch"] for r in rows)

    diagnostics = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "val_RMSE_coefficient_of_variation": cv_rmse,
        "best_epoch_spread": best_epoch_spread,
        "all_seeds_beat_persistence": bool(np.all(val_rmse < persistence["RMSE"])),
        "note": (
            "Descriptive only. No seed is presented as 'the' result; all three form the "
            "stability report. Beijing test 2014 not evaluated for any seed. Fontibon not used."
        ),
    }

    payload = {
        "rows": rows,
        "validation_stats": stats,
        "persistence_validation_metrics_ug_m3": persistence,
        "stability_diagnostics": diagnostics,
        "beijing_test_2014_evaluated": False,
        "fontibon_used": False,
    }
    out_path = OUTPUTS_DIR / "metrics" / "R1_development_seed_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
