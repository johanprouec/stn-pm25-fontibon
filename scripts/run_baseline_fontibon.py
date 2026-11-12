"""Reproduce the Primera Entrega Ridge baseline on Fontibon using the
refactored src/ modules, and verify the numbers match the notebook exactly.

Usage: python scripts/run_baseline_fontibon.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.fontibon import build_all_partitions, flatten, load_fontibon_features
from src.evaluation.metrics import high_episode_metrics, regression_metrics
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR

SEED = 42
VENTANA = 24
LIMITE_IMPUTA = 6

# Notebook report text says 40; notebook's executed cell used 37. See
# docs/PHASE1_AUDIT.md. We report both explicitly rather than picking one
# silently.
HIGH_THRESHOLDS = [40.0, 37.0]


def main() -> None:
    np.random.seed(SEED)

    X = load_fontibon_features(FONTIBON_CSV)
    datos = build_all_partitions(X, ventana=VENTANA, limite=LIMITE_IMPUTA)

    Xtr, ytr, _ = datos["train"]
    Xva, yva, _ = datos["val"]
    Xte, yte, _ = datos["test"]

    n_windows = {k: int(len(v[1])) for k, v in datos.items()}
    print("Windows per partition:", n_windows)

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(flatten(Xtr), ytr)

    pred_val = model.predict(flatten(Xva))
    pred_test = model.predict(flatten(Xte))

    val_metrics = regression_metrics(yva, pred_val)
    test_metrics = regression_metrics(yte, pred_test)

    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    high_val = {str(t): high_episode_metrics(yva, pred_val, threshold=t) for t in HIGH_THRESHOLDS}
    high_test = {str(t): high_episode_metrics(yte, pred_test, threshold=t) for t in HIGH_THRESHOLDS}
    high_train_counts = {str(t): int((ytr >= t).sum()) for t in HIGH_THRESHOLDS}

    expected = {
        "n_windows": {"train": 10802, "val": 2564, "test": 2574},
        "val_rmse": 6.545,
        "test_rmse": 7.063,
    }
    checks = {
        "n_windows_match": n_windows == expected["n_windows"],
        "val_rmse_match": abs(val_metrics["RMSE"] - expected["val_rmse"]) < 0.001,
        "test_rmse_match": abs(test_metrics["RMSE"] - expected["test_rmse"]) < 0.001,
    }
    print("Reproducibility checks vs Primera Entrega report:", checks)
    assert all(checks.values()), f"Refactor does not reproduce report numbers: {checks}"

    out_dir = OUTPUTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "ventana": VENTANA,
        "limite_imputacion_h": LIMITE_IMPUTA,
        "n_windows": n_windows,
        "validacion": val_metrics,
        "test": test_metrics,
        "high_episode_train_counts": high_train_counts,
        "high_episode_val": high_val,
        "high_episode_test": high_test,
        "reproducibility_checks": checks,
    }
    out_path = out_dir / "baseline_fontibon_ridge.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
