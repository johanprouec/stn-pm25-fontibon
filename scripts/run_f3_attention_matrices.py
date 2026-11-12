"""Fase 3 — matrices de atención consulta × clave y figuras de cierre.

Auditoría pura sobre el modelo final congelado (A, F3_frozen_protocol.json):
no entrena, no cambia modelo ni umbral y no toma decisiones con test.
Reutiliza forward_with_attention() de run_f3_explainability.py, que ya se
verificó contra la predicción del modelo, y lee los pesos reales devueltos
por SparseAttention.call(..., return_attention=True).

Produce:
  - attention_matrix_normal_seed42.png / attention_matrix_high_seed42.png:
    matriz consulta × clave de enc0_long1 para las 4 cabezas (sin promediar).
  - attention_heads_seed42.png: comparación de cabezas sobre todo el test.
  - attention_synthetic_seed42.png: matrices y masa temporal en 3 casos sintéticos.
  - attention_normal_vs_high.png: comparación descriptiva normal (< 40) / alto (>= 40), 3 semillas.
  - test_residual_distribution.png y test_model_comparison.png (solo resultados ya existentes).
  - outputs/metrics/F3_attention_matrices_summary.json (+ arrays .npz).

Uso: python scripts/run_f3_attention_matrices.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.fontibon import prepare_fontibon_stn
from src.training.transfer import CANDIDATE_V1, build_transfer_model
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR

_spec = importlib.util.spec_from_file_location("f3x", ROOT / "scripts" / "run_f3_explainability.py")
f3x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f3x)

METRICS = OUTPUTS_DIR / "metrics"
FIG_DIR = OUTPUTS_DIR / "figures" / "F3_attention"
FIG_FINAL = OUTPUTS_DIR / "figures" / "F3_test_final"
OUT_JSON = METRICS / "F3_attention_matrices_summary.json"
SEEDS = [42, 123, 2026]
EXAMPLE_SEED = 42
HIGH = 40.0  # umbral congelado en el protocolo
HOURS = np.arange(-23, 1)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, NA_GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#e9e8e4"
SEQ = f3x.SEQ.copy()
SEQ.set_bad(NA_GRAY)


def style(ax):
    f3x.style(ax)


def load_model(protocol, data, seed):
    info = protocol["checkpoint_integrity"]["runs"][f"transfer_A_seed{seed}"]
    model, _ = build_transfer_model("A", seed, data.train.encoder_input[:2], data.train.decoder_input[:2],
                                    architecture=CANDIDATE_V1)
    model.load_weights(str(ROOT / info["checkpoint"]))
    return model


def capture_long1(model, enc, dec, batch=256):
    """Pesos reales de enc0_long1: weights (N,H,n_top,24) y consultas seleccionadas (N,H,n_top)."""
    ws, idx, preds, diffs = [], [], [], []
    for s in range(0, len(enc), batch):
        e, d = tf.constant(enc[s:s + batch], tf.float32), tf.constant(dec[s:s + batch], tf.float32)
        out, caps = f3x.forward_with_attention(model, e, d)
        ref = model((e, d), training=False)
        diffs.append(float(tf.reduce_max(tf.abs(out - ref))))
        ws.append(caps["enc0_long1"]["attention_weights"].numpy())
        idx.append(caps["enc0_long1"]["selected_indices"].numpy())
        preds.append(out.numpy())
    return np.concatenate(ws), np.concatenate(idx), np.concatenate(preds), max(diffs)


def query_key_matrix(w, sel):
    """Matriz consulta × clave (24×24) de UNA ventana y UNA cabeza; filas no seleccionadas = NaN."""
    M = np.full((24, 24), np.nan)
    for j, q in enumerate(sel):
        M[q, :] = w[j, :]
    return M


def per_window_stats(W):
    """W: (N,H,n_top,24). Masa en últimas 6 h, posición media de la clave (horas atrás) y entropía normalizada."""
    last6 = W[..., -6:].sum(-1).mean(-1)  # (N,H)
    pos = (W * (23 - np.arange(24))).sum(-1).mean(-1)  # horas atrás promedio
    ent = (-(W * np.log(np.clip(W, 1e-12, 1))).sum(-1) / np.log(24)).mean(-1)
    return last6, pos, ent


def selection_freq(S, mask):
    return np.array([(S[mask] == p).any(axis=2).mean() for p in range(24)])


def plot_matrix_panels(W1, S1, title, path, y_real, pred):
    H = W1.shape[0]
    mats = [query_key_matrix(W1[h], S1[h]) for h in range(H)]
    vmax = np.nanmax(mats)
    fig, axes = plt.subplots(1, H, figsize=(15, 4.4), sharey=True)
    for h, ax in enumerate(axes):
        im = ax.imshow(mats[h], cmap=SEQ, vmin=0, vmax=vmax, extent=[-23.5, 0.5, 0.5, -23.5], aspect="equal")
        ax.set_title(f"Cabeza {h + 1}", color=INK, fontsize=10)
        ax.set_xlabel("Clave (hora observada)", color=INK2, fontsize=9)
        ax.set_xticks([-20, -15, -10, -5, 0])
        ax.set_yticks([-20, -15, -10, -5, 0])
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Consulta (hora)", color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("Peso de atención", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.suptitle(f"{title} — real {y_real:.1f} µg/m³, predicción {pred:.1f} µg/m³ "
                 "(filas grises: consultas no seleccionadas por la atención dispersa)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    protocol = json.loads((METRICS / "F3_frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["final_model"]["config"] == "A"
    assert protocol["metrics"]["high_episodes"]["primary_threshold_ug_m3"] == HIGH
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    test, sc, fn = data.test, data.scalers, data.feature_names
    pm_idx = fn.index("pm25")
    y = test.y_original.ravel()
    high = y >= HIGH
    last_obs = sc.inverse_transform_pm25(test.encoder_input[:, -1, pm_idx])
    delta = y - last_obs

    # Reglas fijas para elegir las ventanas de ejemplo (no dependen del error del modelo)
    normal_pool = np.where((~high) & (np.abs(delta) < 5))[0]
    normal_idx = int(normal_pool[np.argmin(np.abs(y[normal_pool] - np.median(y[~high])))])
    high_pool = np.where(high)[0]
    high_idx = int(high_pool[np.argmin(np.abs(y[high_pool] - np.median(y[high])))])

    summary = {"note": "Auditoría: pesos reales de SparseAttention (enc0_long1) del modelo A congelado; sin entrenamiento.",
               "threshold_high_ug_m3": HIGH,
               "example_selection_rules": {
                   "normal": "ventana con PM2,5 real < 40 y |cambio| < 5, cuyo valor real es el más cercano a la mediana del grupo < 40",
                   "high": "ventana con PM2,5 real >= 40 cuyo valor real es el más cercano a la mediana del grupo >= 40"},
               "seeds": {}}
    arrays = {}
    W42 = S42 = P42 = None
    for seed in SEEDS:
        model = load_model(protocol, data, seed)
        W, S, P, maxdiff = capture_long1(model, test.encoder_input, test.decoder_input)
        assert maxdiff < 1e-5 and np.isfinite(W).all()
        pred = sc.inverse_transform_pm25(P)
        last6, pos, ent = per_window_stats(W)
        mass = W.mean(axis=2)  # (N,H,24) masa por clave, promediando consultas seleccionadas de cada cabeza
        grp = {}
        for name, m in (("normal", ~high), ("high", high)):
            grp[name] = {
                "n": int(m.sum()),
                "mass_last6h": float(last6[m].mean()),
                "mean_key_hours_back": float(pos[m].mean()),
                "normalized_entropy": float(ent[m].mean()),
                "key_mass_profile": mass[m].mean(axis=(0, 1)).tolist(),
                "query_selection_freq": selection_freq(S, m).tolist(),
                "per_head_mass_last6h": last6[m].mean(0).tolist(),
                "per_head_mean_key_hours_back": pos[m].mean(0).tolist(),
                "per_head_entropy": ent[m].mean(0).tolist(),
            }
        diff = {k: grp["high"][k] - grp["normal"][k] for k in ("mass_last6h", "mean_key_hours_back", "normalized_entropy")}
        summary["seeds"][str(seed)] = {"forward_check_max_abs_diff": maxdiff, "attention_shape_enc0_long1": list(W.shape),
                                       "groups": grp, "high_minus_normal": diff,
                                       "pred_check_vs_saved_csv_max_abs": None}
        saved = pd.read_csv(METRICS / f"F3_transfer_A_seed{seed}_test_predictions.csv")
        summary["seeds"][str(seed)]["pred_check_vs_saved_csv_max_abs"] = float(np.max(np.abs(saved["y_pred"].to_numpy() - pred)))
        arrays[f"weights_seed{seed}_normal_example"] = W[normal_idx]
        arrays[f"weights_seed{seed}_high_example"] = W[high_idx]
        arrays[f"selected_seed{seed}_normal_example"] = S[normal_idx]
        arrays[f"selected_seed{seed}_high_example"] = S[high_idx]
        if seed == EXAMPLE_SEED:
            W42, S42, P42, model42 = W, S, pred, model
        print(f"seed {seed}: forward ok ({maxdiff:.1e}), shape {W.shape}, diff alto-normal {({k: round(v, 4) for k, v in diff.items()})}")

    # consistencia de la diferencia alto - normal entre semillas
    cons = {}
    for k in ("mass_last6h", "mean_key_hours_back", "normalized_entropy"):
        vals = [summary["seeds"][str(s)]["high_minus_normal"][k] for s in SEEDS]
        cons[k] = {"per_seed": vals, "same_sign_all_seeds": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals))}
    summary["high_minus_normal_consistency"] = cons

    # ejemplos seed42
    for tag, i in (("normal", normal_idx), ("high", high_idx)):
        summary[f"example_{tag}"] = {"timestamp": str(test.target_timestamps[i]), "y_real": float(y[i]),
                                     "pred_seed42": float(P42[i]), "last_observed": float(last_obs[i]),
                                     "selected_queries_per_head_hours": [[int(q) - 23 for q in sorted(S42[i, h])] for h in range(S42.shape[1])],
                                     "mass_last6h_per_head": W42[i][..., -6:].sum(-1).mean(-1).tolist()}
        plot_matrix_panels(W42[i], S42[i],
                           f"enc0_long1 · semilla 42 · {'episodio normal' if tag == 'normal' else 'episodio alto (≥ 40)'} · {str(test.target_timestamps[i])[:16]}",
                           FIG_DIR / f"attention_matrix_{tag}_seed42.png", y[i], P42[i])

    # comparación de cabezas (seed42, todo el test)
    g42 = summary["seeds"][str(EXAMPLE_SEED)]["groups"]
    H = W42.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for h, c in zip(range(H), (BLUE, ORANGE, AQUA, "#6250d6")):
        axes[0].plot(HOURS, W42[~high][:, h].mean(axis=(0, 1)), color=c, linewidth=2, label=f"Cabeza {h + 1}")
    axes[0].axhline(1 / 24, color=INK2, linestyle="--", linewidth=1, label="Uniforme (1/24)")
    axes[0].set_xlabel("Clave (hora observada)", color=INK2)
    axes[0].set_ylabel("Masa media de atención", color=INK2)
    axes[0].set_title("Masa por clave y cabeza (episodios normales)", color=INK, fontsize=10, loc="left")
    style(axes[0])
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    x = np.arange(H)
    for k, (grp_name, c) in enumerate((("normal", BLUE), ("high", ORANGE))):
        axes[1].bar(x + (k - 0.5) * 0.38, g42[grp_name]["per_head_mass_last6h"], width=0.36, color=c,
                    label="Normal (< 40)" if grp_name == "normal" else "Alto (≥ 40)")
    axes[1].set_xticks(x, [f"Cabeza {h + 1}" for h in range(H)])
    axes[1].set_ylabel("Masa en las últimas 6 h", color=INK2)
    axes[1].set_title("Masa en las últimas 6 h por cabeza", color=INK, fontsize=10, loc="left")
    style(axes[1])
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attention_heads_seed42.png", dpi=200)
    plt.close(fig)

    # normal vs alto (3 semillas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for s, c in zip(SEEDS, (BLUE, ORANGE, AQUA)):
        gs = summary["seeds"][str(s)]["groups"]
        axes[0].plot(HOURS, gs["normal"]["key_mass_profile"], color=c, linewidth=2, label=f"Semilla {s} · normal")
        axes[0].plot(HOURS, gs["high"]["key_mass_profile"], color=c, linewidth=2, linestyle=":", label=f"Semilla {s} · alto")
        axes[1].plot(HOURS, gs["normal"]["query_selection_freq"], color=c, linewidth=2)
        axes[1].plot(HOURS, gs["high"]["query_selection_freq"], color=c, linewidth=2, linestyle=":")
    axes[0].set_title("Masa de atención por clave", color=INK, fontsize=10, loc="left")
    axes[1].set_title("Frecuencia con que cada hora es consulta seleccionada", color=INK, fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("Hora relativa (0 = última observada)", color=INK2)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Normal (< 40, línea continua) frente a alto (≥ 40, punteada) en test", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attention_normal_vs_high.png", dpi=200)
    plt.close(fig)

    # casos sintéticos con atención (seed42)
    train_mean = data.train.encoder_input.mean(axis=(0, 1))
    tm = {c: float(v) for c, v in zip(sc.other_cols, sc.other_scaler.inverse_transform(
        train_mean[[fn.index(c) for c in sc.other_cols]].reshape(1, -1)).ravel())}
    cases = {"constante_20": (np.full(24, 20.0), "≈ 20 (serie estable)"),
             "rampa_15_a_45_ultimas_6h": (np.r_[np.full(18, 15.0), np.linspace(20, 45, 6)], "≥ 45 si continúa la tendencia"),
             "pico_en_la_ultima_hora": (np.r_[np.full(23, 15.0), [60.0]], "entre 15 y 60 (sin respuesta única)")}
    enc = np.stack([f3x.scale_window(v, np.full(24, tm["temp"]), np.full(24, tm["dew"]), sc, fn) for v, _ in cases.values()]).astype(np.float32)
    dec = f3x.decoder_from_encoder(enc, pm_idx)
    Wc, Sc, Pc, md = capture_long1(model42, enc, dec)
    predc = sc.inverse_transform_pm25(Pc)
    expl_prev = json.loads((METRICS / "F3_explainability_summary.json").read_text(encoding="utf-8"))
    summary["synthetic_cases_seed42"] = {}
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3.0 * len(cases)), gridspec_kw={"width_ratios": [1.5, 1]})
    for r, (name, (series, expected)) in enumerate(cases.items()):
        mats = np.stack([query_key_matrix(Wc[r, h], Sc[r, h]) for h in range(Wc.shape[1])])
        Mavg = np.nanmean(np.where(np.isnan(mats), np.nan, mats), axis=0) if not np.isnan(mats).all() else mats[0]
        km = Wc[r].mean(axis=(0, 1))
        summary["synthetic_cases_seed42"][name] = {
            "pred_seed42": float(predc[r]),
            "pred_mean_3_seeds_existing": expl_prev["aggregate"][f"synthetic_{name}"]["mean"],
            "reasonableness_criterion": expected, "ground_truth": None,
            "key_mass_profile": km.tolist(), "mass_last6h": float(km[-6:].sum()), "mass_last_hour": float(km[-1])}
        ax = axes[r, 0]
        ax.plot(HOURS, series, color=INK2, linewidth=2, marker="o", markersize=3, label="PM2,5 de entrada")
        ax.scatter([1], [predc[r]], color=BLUE, s=60, zorder=3, label=f"Predicción t+1: {predc[r]:.1f}")
        ax2 = ax.twinx()
        ax2.bar(HOURS, km, color=ORANGE, alpha=0.35, width=0.8)
        ax2.set_ylim(0, max(0.2, km.max() * 1.2))
        ax2.set_ylabel("Masa de atención", color=ORANGE, fontsize=8)
        ax2.tick_params(colors=ORANGE, labelsize=7)
        pretty = {"constante_20": "Constante 20", "rampa_15_a_45_ultimas_6h": "Rampa 15→45 (últimas 6 h)",
                  "pico_en_la_ultima_hora": "Pico de 60 en la última hora"}[name]
        ax.set_title(f"{pretty} · criterio: {expected}", color=INK, fontsize=9, loc="left")
        lo, hi = min(series.min(), predc[r]), max(series.max(), predc[r])
        ax.set_ylim(lo - max(5, 0.1 * (hi - lo)), hi + max(5, 0.1 * (hi - lo)))
        ax.set_ylabel("µg/m³", color=INK2)
        style(ax)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        im = axes[r, 1].imshow(Mavg, cmap=SEQ, vmin=0, extent=[-23.5, 0.5, 0.5, -23.5])
        axes[r, 1].set_title("Consulta × clave (media de cabezas)", color=INK, fontsize=9, loc="left")
        axes[r, 1].set_ylabel("Consulta (hora)", color=INK2, fontsize=8)
        axes[r, 1].tick_params(colors=INK2, labelsize=7)
        fig.colorbar(im, ax=axes[r, 1], fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("Hora relativa (0 = última observada; t+1 a la derecha)", color=INK2)
    axes[-1, 1].set_xlabel("Clave (hora observada)", color=INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attention_synthetic_seed42.png", dpi=200)
    plt.close(fig)

    # residuos y comparación final (solo resultados ya existentes)
    final = json.loads((METRICS / "F3_test_final_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.arange(-50, 31, 1.5)
    for s, c in zip(SEEDS, (BLUE, ORANGE, AQUA)):
        r = pd.read_csv(METRICS / f"F3_transfer_A_seed{s}_test_predictions.csv")["residual"].to_numpy()
        ax.hist(r, bins=bins, histtype="step", linewidth=1.8, color=c, label=f"Semilla {s} (sesgo {r.mean():+.2f})")
    ax.axvline(0, color=INK, linewidth=1)
    bias = np.mean([final["per_run"][f"transfer_A_seed{s}"]["test_residual_diagnostics_ug_m3"]["bias_mean"] for s in SEEDS])
    ax.axvline(bias, color=INK2, linewidth=1, linestyle="--", label=f"Sesgo medio {bias:+.2f}")
    ax.set_xlabel("Residuo = predicción − real (µg/m³)", color=INK2)
    ax.set_ylabel("Ventanas", color=INK2)
    ax.set_title("Distribución de residuos del modelo A en test", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_FINAL / "test_residual_distribution.png", dpi=200)
    plt.close(fig)

    agg = final["aggregate_by_config"]
    labels = ["A (3 semillas)", "B (2 semillas)", "C (3 semillas)", "Ridge", "Persistencia"]
    vals = [agg["A"]["RMSE"]["mean"], agg["B"]["RMSE"]["mean"], agg["C"]["RMSE"]["mean"],
            final["ridge_test"]["metrics_ug_m3"]["RMSE"], final["persistence_test"]["metrics_ug_m3"]["RMSE"]]
    errs = [agg["A"]["RMSE"]["std"], agg["B"]["RMSE"]["std"], agg["C"]["RMSE"]["std"], 0, 0]
    highs = [agg["A"]["RMSE_high_40"]["mean"], agg["B"]["RMSE_high_40"]["mean"], agg["C"]["RMSE_high_40"]["mean"],
             final["ridge_test"]["high_episodes_test"]["40.0"]["RMSE_high"], final["persistence_test"]["high_episodes_test"]["40.0"]["RMSE_high"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = [BLUE, "#9ec5f4", "#5598e7", ORANGE, AQUA]
    for ax, v, e, title in ((axes[0], vals, errs, "RMSE en test (todas las ventanas)"),
                            (axes[1], highs, None, "RMSE en episodios ≥ 40 µg/m³ (n = 118)")):
        bars = ax.bar(labels, v, color=colors, yerr=e, capsize=4)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.15, f"{val:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("µg/m³", color=INK2)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        style(ax)
    fig.text(0.01, -0.02, "D no se evaluó en test (solo piloto de integración); B2026 excluida (entrenamiento interrumpido).",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG_FINAL / "test_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary["training_performed"] = False
    np.savez(METRICS / "F3_attention_matrices_arrays.npz", **arrays)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ejemplos:", {k: summary[f"example_{k}"]["timestamp"] for k in ("normal", "high")})
    print("consistencia alto-normal:", {k: v["same_sign_all_seeds"] for k, v in cons.items()})
    print("Resumen:", OUT_JSON)


if __name__ == "__main__":
    main()
