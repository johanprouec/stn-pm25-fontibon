"""Fase 3 — análisis de fallas del modelo final A sobre Fontibón test.

Parte de las predicciones ya guardadas por run_f3_test_evaluation.py (no
reentrena nada). Caracteriza los errores por nivel de concentración, hora
del día, mes, tipo de cambio y presencia de datos imputados; revisa la
calibración (intervalos conformales y dispersión entre semillas) y el
comportamiento fuera de dominio (novedad de la ventana frente a train y
niveles por encima del máximo de entrenamiento).

Uso: python scripts/run_f3_failure_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.fontibon import load_fontibon_features, prepare_fontibon_stn
from src.models.stn import build_decoder_input
from src.training.transfer import CANDIDATE_V1, build_transfer_model
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR

SEEDS = [42, 123, 2026]
METRICS = OUTPUTS_DIR / "metrics"
FIG_DIR = OUTPUTS_DIR / "figures" / "F3_failure_analysis"
OUT_JSON = METRICS / "F3_failure_analysis_summary.json"
BINS = [0, 10, 20, 30, 40, np.inf]
BIN_LABELS = ["<10", "10–20", "20–30", "30–40", "≥40"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def err_stats(y, p):
    e = p - y
    return {"n": int(len(y)), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e))}


def grouped(df, key, models):
    out = {}
    for g, sub in df.groupby(key, observed=True):
        out[str(g)] = {m: err_stats(sub["y"].to_numpy(), sub[m].to_numpy()) for m in models}
        if "A_seed42" in df:
            out[str(g)]["A_per_seed_RMSE"] = [err_stats(sub["y"].to_numpy(), sub[f"A_seed{s}"].to_numpy())["RMSE"] for s in SEEDS]
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    final = json.loads((METRICS / "F3_test_final_summary.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ datos
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    test, sc, fn = data.test, data.scalers, data.feature_names
    pm_idx = fn.index("pm25")
    ts = pd.DatetimeIndex(test.target_timestamps)
    df = pd.DataFrame({"t": ts, "y": test.y_original.ravel()})
    for s in SEEDS:
        p = pd.read_csv(METRICS / f"F3_transfer_A_seed{s}_test_predictions.csv", parse_dates=["target_timestamp"])
        assert np.array_equal(p["target_timestamp"].to_numpy(), ts.to_numpy())
        assert np.allclose(p["y_real"].to_numpy(), df["y"].to_numpy())
        df[f"A_seed{s}"] = p["y_pred"].to_numpy()
    df["A"] = df[[f"A_seed{s}" for s in SEEDS]].mean(axis=1)  # promedio de las 3 semillas (para lectura por grupos)
    df["A_spread"] = df[[f"A_seed{s}" for s in SEEDS]].std(axis=1, ddof=1)
    ridge = pd.read_csv(METRICS / "F3_ridge_test_predictions.csv")
    df["Ridge"] = ridge["y_pred"].to_numpy()
    df["last_obs"] = sc.inverse_transform_pm25(test.encoder_input[:, -1, pm_idx])
    df["Persistencia"] = df["last_obs"]
    df["delta"] = df["y"] - df["last_obs"]
    df["nivel"] = pd.cut(df["y"], BINS, labels=BIN_LABELS, right=False)
    df["hora"] = df["t"].dt.hour
    df["mes"] = df["t"].dt.strftime("%Y-%m")
    df["cambio"] = np.select([df["delta"] >= 5, df["delta"] <= -5], ["subida ≥ 5", "bajada ≥ 5"], "estable (|Δ| < 5)")

    # Ventanas con valores imputados (huecos ≤ 6 h interpolados en la historia de 24 h)
    raw = load_fontibon_features(FONTIBON_CSV)
    nan_any = raw[fn].isna().any(axis=1)
    pos = raw.index.get_indexer(ts)
    assert (pos >= 24).all()
    df["horas_imputadas"] = [int(nan_any.iloc[i - 24:i].sum()) for i in pos]
    df["imputada"] = np.where(df["horas_imputadas"] > 0, "con imputación", "sin imputación")

    models = ["A", "Ridge", "Persistencia"]
    summary = {
        "note": "Errores = predicción - real. 'A' es el promedio de las predicciones de las 3 semillas; "
                "A_per_seed_RMSE da el RMSE de cada semilla en el mismo grupo.",
        "overall": {m: err_stats(df["y"].to_numpy(), df[m].to_numpy()) for m in models + [f"A_seed{s}" for s in SEEDS]},
        "by_level": grouped(df, "nivel", models),
        "by_hour": grouped(df, "hora", models),
        "by_month": grouped(df, "mes", models),
        "by_change": grouped(df, "cambio", models),
        "by_imputation": grouped(df, "imputada", models),
    }

    # Peores errores del modelo A
    df["absA"] = (df["A"] - df["y"]).abs()
    worst = df.nlargest(20, "absA")
    summary["worst_20"] = {
        "share_underestimation": float((worst["A"] < worst["y"]).mean()),
        "share_rises_ge5": float((worst["delta"] >= 5).mean()),
        "share_y_ge40": float((worst["y"] >= 40).mean()),
        "hours": worst["hora"].value_counts().sort_index().to_dict(),
        "rows": [{"t": str(r.t), "y": float(r.y), "A": float(r.A), "Ridge": float(r.Ridge), "ultima_hora": float(r.last_obs),
                  "delta": float(r.delta)} for r in worst.itertuples()],
    }
    top_decile = df["absA"] >= df["absA"].quantile(0.9)
    summary["top_decile_errors"] = {
        "share_rises_ge5": float((df.loc[top_decile, "delta"] >= 5).mean()),
        "share_rises_ge5_overall": float((df["delta"] >= 5).mean()),
        "share_y_ge40": float((df.loc[top_decile, "y"] >= 40).mean()),
        "share_y_ge40_overall": float((df["y"] >= 40).mean()),
    }

    # ---------------------------------------------------------- calibración
    conformal = {s: final["per_run"][f"transfer_A_seed{s}"]["conformal"] for s in SEEDS}
    cov_by_level = {}
    for nominal_i, nominal in enumerate([0.8, 0.9]):
        cov_by_level[str(nominal)] = {}
        for lvl, sub in df.groupby("nivel", observed=True):
            covs = [float((np.abs(sub[f"A_seed{s}"] - sub["y"]) <= conformal[s][nominal_i]["half_width_ug_m3"]).mean()) for s in SEEDS]
            cov_by_level[str(nominal)][str(lvl)] = {"coverage_mean": float(np.mean(covs)), "n": int(len(sub))}
    spread_q = pd.qcut(df["A_spread"], 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (mayor)"])
    summary["calibration"] = {
        "conformal_per_seed": conformal,
        "conformal_coverage_by_level": cov_by_level,
        "seed_spread": {
            "mean_ug_m3": float(df["A_spread"].mean()),
            "corr_spread_vs_abs_error": float(np.corrcoef(df["A_spread"], df["absA"])[0, 1]),
            "mean_abs_error_by_spread_quartile": {str(k): float(v) for k, v in df.groupby(spread_q, observed=True)["absA"].mean().items()},
        },
    }

    # ------------------------------------------------------- fuera de dominio
    tr = data.train.encoder_input.reshape(len(data.train.encoder_input), -1).astype(np.float32)
    te = test.encoder_input.reshape(len(test.encoder_input), -1).astype(np.float32)
    tr_sq = (tr ** 2).sum(1)
    nn = np.empty(len(te), dtype=np.float32)
    for s in range(0, len(te), 256):
        d2 = (te[s:s + 256] ** 2).sum(1)[:, None] + tr_sq[None, :] - 2 * te[s:s + 256] @ tr.T
        nn[s:s + 256] = np.sqrt(np.maximum(d2.min(1), 0))
    df["novedad"] = nn
    nov_q = pd.qcut(df["novedad"], 4, labels=["Q1 (más típica)", "Q2", "Q3", "Q4 (más novedosa)"])
    outside = ((test.encoder_input < -1e-6) | (test.encoder_input > 1 + 1e-6)).any(axis=(1, 2))
    summary["out_of_domain"] = {
        "novelty_definition": "distancia euclidiana (escala del modelo) de cada ventana de test a la ventana de train más parecida",
        "corr_novelty_vs_abs_error": float(np.corrcoef(df["novedad"], df["absA"])[0, 1]),
        "error_by_novelty_quartile": {str(k): {m: err_stats(sub["y"].to_numpy(), sub[m].to_numpy()) for m in models}
                                      for k, sub in df.groupby(nov_q, observed=True)},
        "test_windows_with_any_input_outside_train_range": int(outside.sum()),
        "test_target_max": float(df["y"].max()),
        "train_target_max": float(data.train.y_original.max()),
    }

    # Niveles constantes por encima del rango típico (fuera de dominio sintético)
    pm_levels = [20.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    train_mean = data.train.encoder_input.mean(axis=(0, 1))
    enc = np.broadcast_to(train_mean, (len(pm_levels), 24, len(fn))).astype(np.float32).copy()
    enc[:, :, pm_idx] = sc.pm25_scaler.transform(np.array(pm_levels).reshape(-1, 1)).reshape(-1, 1)
    dec = build_decoder_input(tf.constant(enc[:, :, pm_idx]), label_len=12).numpy()
    protocol = json.loads((METRICS / "F3_frozen_protocol.json").read_text(encoding="utf-8"))
    ood_pred = {}
    for s in SEEDS:
        info = protocol["checkpoint_integrity"]["runs"][f"transfer_A_seed{s}"]
        model, _ = build_transfer_model("A", s, data.train.encoder_input[:2], data.train.decoder_input[:2], architecture=CANDIDATE_V1)
        model.load_weights(str(OUTPUTS_DIR.parent / info["checkpoint"]))
        ood_pred[s] = sc.inverse_transform_pm25(model((tf.constant(enc), tf.constant(dec)), training=False).numpy())
    summary["out_of_domain"]["constant_level_inputs"] = {
        "note": "24 h constantes de PM2,5 con meteorología en la media de train; el máximo de train es 100,8 µg/m3",
        "levels_ug_m3": pm_levels,
        "pred_mean_ug_m3": np.mean([ood_pred[s] for s in SEEDS], axis=0).astype(float).tolist(),
        "pred_per_seed": {str(s): ood_pred[s].astype(float).tolist() for s in SEEDS},
    }
    summary["training_performed"] = False
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ figuras
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(BIN_LABELS))
    w = 0.26
    for i, (m, c) in enumerate(zip(models, (BLUE, ORANGE, AQUA))):
        rm = [summary["by_level"][b][m]["RMSE"] for b in BIN_LABELS]
        bi = [summary["by_level"][b][m]["bias"] for b in BIN_LABELS]
        axes[0].bar(x + (i - 1) * w, rm, width=w - 0.02, color=c, label=m)
        axes[1].bar(x + (i - 1) * w, bi, width=w - 0.02, color=c, label=m)
    counts = [summary["by_level"][b]["A"]["n"] for b in BIN_LABELS]
    for ax, title in zip(axes, ("RMSE por nivel de PM2,5 real", "Sesgo medio (pred − real)")):
        ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(BIN_LABELS, counts)], fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style(ax)
    axes[0].set_ylabel("µg/m³", color=INK2)
    axes[1].axhline(0, color=INK2, linewidth=1)
    axes[1].legend(frameon=False, fontsize=8)
    fig.supxlabel("PM2,5 real en la hora a predecir (µg/m³)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_level.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    hours = sorted(int(h) for h in summary["by_hour"])
    for m, c in zip(models, (BLUE, ORANGE, AQUA)):
        ax.plot(hours, [summary["by_hour"][str(h)][m]["RMSE"] for h in hours], color=c, linewidth=2, marker="o", markersize=4, label=m)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día de la predicción", color=INK2)
    ax.set_ylabel("RMSE (µg/m³)", color=INK2)
    ax.set_title("Error por hora del día en test", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_hour.png", dpi=200)
    plt.close(fig)

    # resumen en consola
    print("Por nivel (RMSE / sesgo):")
    for b in BIN_LABELS:
        r = summary["by_level"][b]
        print(f"  {b:>6} n={r['A']['n']:4d} | A {r['A']['RMSE']:.2f}/{r['A']['bias']:+.2f} | Ridge {r['Ridge']['RMSE']:.2f}/{r['Ridge']['bias']:+.2f} | Pers {r['Persistencia']['RMSE']:.2f}/{r['Persistencia']['bias']:+.2f}")
    for key in ("by_change", "by_imputation", "by_month"):
        print(key, {g: (v["A"]["n"], round(v["A"]["RMSE"], 2), round(v["Ridge"]["RMSE"], 2), round(v["A"]["bias"], 2)) for g, v in summary[key].items()})
    print("peores 20:", {k: v for k, v in summary["worst_20"].items() if k != "rows"})
    print("decil superior:", summary["top_decile_errors"])
    print("calibración:", json.dumps(summary["calibration"]["conformal_coverage_by_level"]), summary["calibration"]["seed_spread"])
    print("OOD:", {k: v for k, v in summary["out_of_domain"].items() if k not in ("error_by_novelty_quartile",)})
    print("OOD por novedad (RMSE A / Ridge):", {k: (round(v["A"]["RMSE"], 2), round(v["Ridge"]["RMSE"], 2)) for k, v in summary["out_of_domain"]["error_by_novelty_quartile"].items()})
    print("hora pico de RMSE A:", max(hours, key=lambda h: summary["by_hour"][str(h)]["A"]["RMSE"]))


if __name__ == "__main__":
    main()
