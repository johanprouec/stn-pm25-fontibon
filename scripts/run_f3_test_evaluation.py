"""Fase 3 — evaluación FINAL y ÚNICA sobre Fontibón test.

Solo corre si existe outputs/metrics/F3_frozen_protocol.json (escrito antes
de tocar test). Para cada corrida congelada (A: modelo final; B y C:
contraste descriptivo de la transferencia):
  1. verifica el SHA-256 del checkpoint,
  2. reconstruye la arquitectura y carga los pesos,
  3. comprueba que reproduce su RMSE de validación de la Fase 2,
  4. solo entonces predice sobre test (training=False, sin optimizador).
Además reajusta Ridge exactamente como en la Primera Entrega (debe
reproducir test RMSE 7,063) y calcula la persistencia. No entrena la STN,
no ajusta nada con base en test y no reemplaza el protocolo.

Uso: python scripts/run_f3_test_evaluation.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.fontibon import build_all_partitions, flatten, load_fontibon_features, prepare_fontibon_stn
from src.evaluation.metrics import (
    evaluate_scaled_predictions,
    high_episode_metrics,
    regression_metrics,
    residual_diagnostics,
    save_predictions_with_residuals,
    target_distribution,
)
from src.evaluation.plots import plot_predictions_vs_real, plot_scatter
from src.training.trainer import evaluate_split_batched
from src.training.transfer import CANDIDATE_V1, build_transfer_model
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR

PROTOCOL_PATH = OUTPUTS_DIR / "metrics" / "F3_frozen_protocol.json"
FINAL_SUMMARY_PATH = OUTPUTS_DIR / "metrics" / "F3_test_final_summary.json"
THRESHOLDS = [40.0, 37.0]
VAL_TOLERANCE = 1e-3
BOOTSTRAP_REPS = 2000
BLOCK_HOURS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}


def block_bootstrap_rmse_diff(y, pred_a, pred_b, rng):
    """IC 95 % de RMSE(a) - RMSE(b) con bootstrap por bloques móviles de 24 h
    (respeta la autocorrelación de la serie). Descriptivo, no es una prueba formal."""
    n = len(y)
    n_blocks = int(np.ceil(n / BLOCK_HOURS))
    starts_max = n - BLOCK_HOURS
    diffs = np.empty(BOOTSTRAP_REPS)
    for r in range(BOOTSTRAP_REPS):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK_HOURS)[None, :]).ravel()[:n]
        ea, eb = pred_a[idx] - y[idx], pred_b[idx] - y[idx]
        diffs[r] = np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2))
    return {"ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5))}


def conformal_coverage(val_abs_resid, test_resid, y_test, nominal):
    n = len(val_abs_resid)
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    q = float(np.quantile(val_abs_resid, level, method="higher"))
    inside = np.abs(test_resid) <= q
    high = y_test >= THRESHOLDS[0]
    return {
        "nominal": nominal,
        "half_width_ug_m3": q,
        "coverage_test": float(inside.mean()),
        "coverage_test_high_episodes_ge40": float(inside[high].mean()),
        "n_high": int(high.sum()),
    }


def main() -> None:
    t_start = time.time()
    if not PROTOCOL_PATH.exists():
        print(f"REFUSING TO RUN: {PROTOCOL_PATH} no existe. El protocolo debe congelarse antes de tocar test.")
        sys.exit(1)
    if FINAL_SUMMARY_PATH.exists():
        print(f"REFUSING TO RUN: {FINAL_SUMMARY_PATH} ya existe. Fontibón test se evalúa una sola vez.")
        sys.exit(1)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    runs = protocol["checkpoint_integrity"]["runs"]

    print("=" * 70)
    print("FASE 3 -- EVALUACIÓN FINAL SOBRE FONTIBÓN TEST (UNA SOLA VEZ)")
    print("Protocolo congelado:", protocol["frozen_at"], "| modelo final:", protocol["final_model"]["config"])
    print("=" * 70)

    print("\n[1/5] Datos de Fontibón (include_test=True)...")
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    assert data.validation.n_windows == 2564, data.validation.n_windows
    assert data.test.n_windows == protocol["data_protocol"]["expected_test_windows"], data.test.n_windows
    y_test = data.test.y_original.ravel()
    print(f"  ventanas -> train {data.train.n_windows} | val {data.validation.n_windows} | test {data.test.n_windows}")

    print("\n[2/5] Integridad de checkpoints y reproducción de validación (antes de test)...")
    sample_enc = data.train.encoder_input[:2]
    sample_dec = data.train.decoder_input[:2]
    models = {}
    for name, info in runs.items():
        ckpt = OUTPUTS_DIR.parent / info["checkpoint"]
        assert sha256(ckpt) == info["checkpoint_sha256"], f"SHA-256 distinto en {name}"
        model, _ = build_transfer_model(info["config"], info["seed"], sample_enc, sample_dec, architecture=CANDIDATE_V1)
        model.load_weights(str(ckpt))
        val_pred_scaled = evaluate_split_batched(model, data.validation, batch_size=32)
        val_metrics = evaluate_scaled_predictions(data.validation.y_scaled, val_pred_scaled, data.scalers.inverse_transform_pm25)
        expected = info["validation_metrics_ug_m3"]["RMSE"]
        diff = abs(val_metrics["RMSE"] - expected)
        print(f"  {name}: val RMSE {val_metrics['RMSE']:.4f} (Fase 2: {expected:.4f}, dif {diff:.2e})")
        if diff > VAL_TOLERANCE:
            print(f"REFUSING TO CONTINUE: {name} no reproduce su validación. Test NO se evalúa.")
            sys.exit(1)
        models[name] = (info, model, data.scalers.inverse_transform_pm25(val_pred_scaled).ravel())

    print("\n[3/5] Referencias: Ridge (reajustado como en la Fase 1) y persistencia...")
    np.random.seed(42)
    X = load_fontibon_features(FONTIBON_CSV)
    parts = build_all_partitions(X, ventana=24, limite=6)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ts_te) = parts["train"], parts["val"], parts["test"]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(flatten(Xtr), ytr)
    t0 = time.perf_counter()
    ridge_test = ridge.predict(flatten(Xte))
    ridge_infer_s = time.perf_counter() - t0
    ridge_metrics = regression_metrics(yte, ridge_test)
    assert abs(ridge_metrics["RMSE"] - protocol["reference_models_on_test"]["ridge"]["must_reproduce_test_rmse"]) < 1e-6
    assert np.array_equal(np.asarray(ts_te), np.asarray(data.test.target_timestamps)), "ventanas de Ridge y STN no coinciden"
    assert np.allclose(np.asarray(yte).ravel(), y_test), "objetivos de Ridge y STN no coinciden"
    pm25_idx = data.feature_names.index("pm25")
    persistence = data.scalers.inverse_transform_pm25(data.test.encoder_input[:, -1, pm25_idx].reshape(-1, 1)).ravel()
    persistence_metrics = regression_metrics(y_test, persistence)
    print(f"  Ridge test RMSE {ridge_metrics['RMSE']:.3f} (reproduce Fase 1) | persistencia {persistence_metrics['RMSE']:.3f}")
    save_predictions_with_residuals(data.test.target_timestamps, y_test, ridge_test,
                                    OUTPUTS_DIR / "metrics" / "F3_ridge_test_predictions.csv")

    print("\n[4/5] Inferencia sobre test (training=False)...")
    figures_dir = OUTPUTS_DIR / "figures" / "F3_test_final"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    per_run = {}
    for name, (info, model, val_pred) in models.items():
        t0 = time.perf_counter()
        test_pred_scaled = evaluate_split_batched(model, data.test, batch_size=32)
        infer_s = time.perf_counter() - t0
        pred = data.scalers.inverse_transform_pm25(test_pred_scaled).ravel()
        metrics = regression_metrics(y_test, pred)
        result = {
            "config": info["config"], "seed": info["seed"],
            "test_metrics_ug_m3": metrics,
            "test_residual_diagnostics_ug_m3": residual_diagnostics(y_test, pred),
            "high_episodes_test": {str(t): high_episode_metrics(y_test, pred, threshold=t) for t in THRESHOLDS},
            "rmse_minus_ridge": metrics["RMSE"] - ridge_metrics["RMSE"],
            "rmse_minus_ridge_block_bootstrap": block_bootstrap_rmse_diff(y_test, pred, ridge_test, rng),
            "fraction_windows_abs_error_below_ridge": float(np.mean(np.abs(pred - y_test) < np.abs(ridge_test - y_test))),
            "inference_seconds_test": infer_s,
        }
        if info["config"] == "A":
            val_abs = np.abs(val_pred - data.validation.y_original.ravel())
            result["conformal"] = [conformal_coverage(val_abs, pred - y_test, y_test, a)
                                   for a in protocol["calibration_protocol"]["nominal_coverage"]]
            plot_predictions_vs_real(data.test.target_timestamps, y_test, pred,
                                     figures_dir / f"test_predictions_vs_real_{name}.png", split_label="Fontibón test")
            plot_scatter(y_test, pred, figures_dir / f"test_scatter_{name}.png", split_label="Fontibón test")
        save_predictions_with_residuals(data.test.target_timestamps, y_test, pred,
                                        OUTPUTS_DIR / "metrics" / f"F3_{name}_test_predictions.csv")
        per_run[name] = result
        print(f"  {name}: RMSE {metrics['RMSE']:.3f} MAE {metrics['MAE']:.3f} R2 {metrics['R2']:.4f} | "
              f"RMSE_high40 {result['high_episodes_test']['40.0']['RMSE_high']:.3f}")

    print("\n[5/5] Agregados y criterios de la Primera Entrega...")
    aggregate = {}
    for cfg in ("A", "B", "C"):
        rs = [r for r in per_run.values() if r["config"] == cfg]
        aggregate[cfg] = {
            "n_seeds": len(rs),
            "RMSE": mean_std([r["test_metrics_ug_m3"]["RMSE"] for r in rs]),
            "MAE": mean_std([r["test_metrics_ug_m3"]["MAE"] for r in rs]),
            "R2": mean_std([r["test_metrics_ug_m3"]["R2"] for r in rs]),
            "RMSE_high_40": mean_std([r["high_episodes_test"]["40.0"]["RMSE_high"] for r in rs]),
            "Bias_high_40": mean_std([r["high_episodes_test"]["40.0"]["Bias_high"] for r in rs]),
            "RMSE_high_37": mean_std([r["high_episodes_test"]["37.0"]["RMSE_high"] for r in rs]),
        }
    ridge_high = {str(t): high_episode_metrics(y_test, ridge_test, threshold=t) for t in THRESHOLDS}
    a_runs = [r for r in per_run.values() if r["config"] == "A"]
    rmse_consistent = all(r["test_metrics_ug_m3"]["RMSE"] < ridge_metrics["RMSE"] for r in a_runs)
    high_improved = aggregate["A"]["RMSE_high_40"]["mean"] < ridge_high["40.0"]["RMSE_high"]

    def transfer_beats_a(cfg):
        pairs = [(per_run[f"transfer_{cfg}_seed{s}"], per_run[f"transfer_A_seed{s}"])
                 for s in protocol["final_model"]["seeds"] if f"transfer_{cfg}_seed{s}" in per_run]
        return all(t["test_metrics_ug_m3"]["RMSE"] < a["test_metrics_ug_m3"]["RMSE"] for t, a in pairs) \
            and aggregate[cfg]["RMSE"]["mean"] < aggregate["A"]["RMSE"]["mean"]

    criteria = {
        "depth_rmse_consistent_vs_ridge": rmse_consistent,
        "depth_high_episodes_improved_vs_ridge": high_improved,
        "depth_justified_by_primera_entrega_criteria": bool(rmse_consistent and high_improved),
        "transfer_B_useful": transfer_beats_a("B"),
        "transfer_C_useful": transfer_beats_a("C"),
    }

    summary = {
        "label": "FASE 3 -- evaluación final sobre Fontibón test, realizada una sola vez",
        "protocol": str(PROTOCOL_PATH.relative_to(OUTPUTS_DIR.parent)).replace("\\", "/"),
        "n_test_windows": data.test.n_windows,
        "test_period": [str(data.test.target_timestamps.min()), str(data.test.target_timestamps.max())],
        "test_target_distribution_ug_m3": target_distribution(data.test.y_original),
        "train_pm25_max_ug_m3": float(data.train.y_original.max()),
        "n_test_targets_above_train_max": int((y_test > data.train.y_original.max()).sum()),
        "per_run": per_run,
        "aggregate_by_config": aggregate,
        "ridge_test": {"metrics_ug_m3": ridge_metrics, "high_episodes_test": ridge_high,
                       "inference_seconds_test": ridge_infer_s},
        "persistence_test": {"metrics_ug_m3": persistence_metrics,
                             "high_episodes_test": {str(t): high_episode_metrics(y_test, persistence, threshold=t) for t in THRESHOLDS}},
        "criteria_primera_entrega": criteria,
        "bootstrap": {"method": "bloques móviles", "block_hours": BLOCK_HOURS, "reps": BOOTSTRAP_REPS, "rng_seed": 2026},
        "training_performed": False,
        "model_or_seed_changed_after_seeing_test": False,
        "total_seconds": time.time() - t_start,
    }
    FINAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    a = aggregate["A"]
    print("\n" + "=" * 70)
    print(f"A (final)  test RMSE {a['RMSE']['mean']:.3f} ± {a['RMSE']['std']:.3f} | "
          f"MAE {a['MAE']['mean']:.3f} | R2 {a['R2']['mean']:.4f}")
    for cfg in ("B", "C"):
        print(f"{cfg}          test RMSE {aggregate[cfg]['RMSE']['mean']:.3f} (n={aggregate[cfg]['n_seeds']})")
    print(f"Ridge      test RMSE {ridge_metrics['RMSE']:.3f} | persistencia {persistence_metrics['RMSE']:.3f}")
    print(f"RMSE_high40: A {a['RMSE_high_40']['mean']:.3f} vs Ridge {ridge_high['40.0']['RMSE_high']:.3f}")
    print("Criterios:", criteria)
    print("Resumen:", FINAL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
