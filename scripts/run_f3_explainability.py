"""Fase 3 — auditoría de explicabilidad del modelo final A (3 semillas).

Todo es inferencia sobre los checkpoints congelados (F3_frozen_protocol.json);
no se entrena ni se modifica nada. Análisis:
  1. Mapas de atención: forward manual capa por capa con return_attention=True,
     verificado contra la predicción del modelo (debe coincidir).
  2. Gradientes integrados (IG) por hora y variable, con prueba de completitud.
  3. Prueba de atajo: ¿el modelo copia la última hora (persistencia)?
  4. Casos sintéticos de respuesta conocida.
  5. Ablaciones sobre ventanas reales de test.

Uso: python scripts/run_f3_explainability.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

from src.data.fontibon import prepare_fontibon_stn
from src.models.stn import build_decoder_input
from src.training.trainer import evaluate_split_batched
from src.training.transfer import CANDIDATE_V1, build_transfer_model
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR

PROTOCOL_PATH = OUTPUTS_DIR / "metrics" / "F3_frozen_protocol.json"
OUT_JSON = OUTPUTS_DIR / "metrics" / "F3_explainability_summary.json"
FIG_DIR = OUTPUTS_DIR / "figures" / "F3_explainability"
SEEDS = [42, 123, 2026]
HIGH = 40.0
IG_STEPS = 32
IG_N_NORMAL = 400
LABEL_LEN = CANDIDATE_V1.get("decoder_label_len", 12)

# Paleta de referencia (skill dataviz): categórica fija y rampa azul secuencial.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0f6fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
FEATURE_LABELS = {"pm25": "PM2,5", "temp": "Temperatura", "dew": "Punto de rocío"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# 1. Forward manual con captura de atención
# ----------------------------------------------------------------------------
def run_block(block, x):
    attn_out, info = block.attn(x, causal=block.causal, causal_mode=block.causal_mode,
                                training=False, return_attention=True)
    x = block.norm1(x + block.dropout1(attn_out, training=False))
    x = block.norm2(x + block.dropout2(block.ffn2(block.ffn1(x)), training=False))
    return x, info


def forward_with_attention(model, enc, dec):
    E = model.encoder
    x = E.positional_encoding(E.input_projection(enc))
    caps = {}
    for ui, unit in enumerate(E.units):
        half = x.shape[1] // 2
        long_b, caps[f"enc{ui}_long1"] = run_block(unit.long_block1, x)
        long_b = unit.pool(unit.conv(long_b))
        long_b, caps[f"enc{ui}_long2"] = run_block(unit.long_block2, long_b)
        short_b, caps[f"enc{ui}_short"] = run_block(unit.short_block, x[:, -half:, :])
        x = tf.concat([long_b, short_b], axis=1)
    enc_out = x
    D = model.decoder
    y = D.positional_encoding(D.input_projection(dec))
    for li, layer in enumerate(D.layers_list):
        sa, caps[f"dec{li}_self"] = layer.self_attn(y, causal=True, causal_mode="probsparse",
                                                    training=False, return_attention=True)
        y = layer.norm1(y + layer.dropout1(sa, training=False))
        cross, scores = layer.cross_attn(query=y, key=enc_out, value=enc_out, training=False,
                                         return_attention_scores=True)
        caps[f"dec{li}_cross"] = {"attention_weights": scores}
        y = layer.norm2(y + layer.dropout2(cross, training=False))
        y = layer.norm3(y + layer.dropout3(layer.ffn2(layer.ffn1(y)), training=False))
    return model.output_head(y), caps


def key_mass(info, L_keys):
    """Masa de atención promedio que recibe cada posición clave, promediando
    sobre consultas seleccionadas y cabezas. Devuelve (B, L_keys)."""
    w = info["attention_weights"].numpy()  # (B,H,n_top,Lk)
    return w.mean(axis=(1, 2))[:, :L_keys]


def selection_freq(info, L):
    idx = info["selected_indices"].numpy()  # (B,H,n_top)
    counts = np.zeros(L)
    for p in range(L):
        counts[p] = (idx == p).any(axis=2).mean()  # fracción (ventana, cabeza) que selecciona la posición p
    return counts


def attention_analysis(model, split, high_mask):
    enc, dec = split.encoder_input.astype(np.float32), split.decoder_input.astype(np.float32)
    acc = {k: [] for k in ("long1_mass", "short_mass", "cross_last", "dec_last_selected", "dec_last_mass")}
    sel_long1 = []
    max_diff = 0.0
    for s in range(0, len(enc), 256):
        e, d = tf.constant(enc[s:s + 256]), tf.constant(dec[s:s + 256])
        out, caps = forward_with_attention(model, e, d)
        ref = model((e, d), training=False)
        max_diff = max(max_diff, float(tf.reduce_max(tf.abs(out - ref))))
        acc["long1_mass"].append(key_mass(caps["enc0_long1"], 24))
        acc["short_mass"].append(key_mass(caps["enc0_short"], 12))
        sel_long1.append(caps["enc0_long1"]["selected_indices"].numpy())
        cross = caps["dec0_cross"]["attention_weights"].numpy()  # (B,H,13,24)
        acc["cross_last"].append(cross[:, :, -1, :].mean(axis=1))
        info = caps["dec0_self"]
        idx = info["selected_indices"].numpy()  # (B,H,n_top)
        last = d.shape[1] - 1
        is_sel = idx == last
        acc["dec_last_selected"].append(is_sel.any(axis=2).mean(axis=1))
        w = info["attention_weights"].numpy()  # (B,H,n_top,13)
        # atención de la posición a predecir sobre las 13 claves, cuando fue seleccionada
        m = np.where(is_sel[..., None], w, np.nan)
        acc["dec_last_mass"].append(np.nanmean(np.nanmean(m, axis=2), axis=1))
    A = {k: np.concatenate(v) for k, v in acc.items()}
    sel = np.concatenate(sel_long1)
    sel_freq = np.array([(sel == p).any(axis=2).mean() for p in range(24)])

    def by_group(arr):
        return {"all": np.nanmean(arr, axis=0), "high": np.nanmean(arr[high_mask], axis=0),
                "normal": np.nanmean(arr[~high_mask], axis=0)}

    return {
        "forward_check_max_abs_diff": max_diff,
        "enc_long1_key_mass": by_group(A["long1_mass"]),
        "enc_long1_selection_freq": sel_freq,
        "enc_short_key_mass": by_group(A["short_mass"]),
        "cross_attention_pred_position": by_group(A["cross_last"]),
        "dec_pred_position_selected_frac": float(np.mean(A["dec_last_selected"])),
        "dec_pred_position_self_mass": by_group(A["dec_last_mass"]),
    }


# ----------------------------------------------------------------------------
# 2. Gradientes integrados
# ----------------------------------------------------------------------------
def integrated_gradients(model, enc, dec, enc_base, dec_base, steps=IG_STEPS, batch=64):
    alphas = (np.arange(steps) + 0.5) / steps  # regla del punto medio
    att_e = np.zeros_like(enc)
    att_d = np.zeros_like(dec)
    for s in range(0, len(enc), batch):
        e, d = enc[s:s + batch], dec[s:s + batch]
        ge = np.zeros_like(e)
        gd = np.zeros_like(d)
        for a in alphas:
            ei = tf.constant(enc_base + a * (e - enc_base), dtype=tf.float32)
            di = tf.constant(dec_base + a * (d - dec_base), dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch([ei, di])
                out = tf.reduce_sum(model((ei, di), training=False))
            g = tape.gradient(out, [ei, di])
            ge += g[0].numpy()
            gd += g[1].numpy()
        att_e[s:s + batch] = (e - enc_base) * ge / steps
        att_d[s:s + batch] = (d - dec_base) * gd / steps
    f_x = model((tf.constant(enc), tf.constant(dec)), training=False).numpy().ravel()
    f_b = model((tf.constant(enc_base[None].repeat(1, 0)), tf.constant(dec_base[None])), training=False).numpy().ravel()[0]
    completeness_err = np.abs(att_e.sum(axis=(1, 2)) + att_d.sum(axis=(1, 2)) - (f_x - f_b))
    return att_e, att_d, completeness_err, float(np.mean(np.abs(f_x - f_b)))


# ----------------------------------------------------------------------------
# 3-5. Atajo, casos sintéticos, ablaciones
# ----------------------------------------------------------------------------
def predict_ugm3(model, enc, dec, scalers):
    out = []
    for s in range(0, len(enc), 256):
        out.append(model((tf.constant(enc[s:s + 256], tf.float32), tf.constant(dec[s:s + 256], tf.float32)),
                         training=False).numpy())
    return scalers.inverse_transform_pm25(np.concatenate(out))


def decoder_from_encoder(enc, pm_idx):
    return build_decoder_input(tf.constant(enc[:, :, pm_idx], tf.float32), label_len=LABEL_LEN).numpy()


def scale_window(pm25, temp, dew, scalers, feature_names):
    """Ventana sintética en unidades físicas -> escala del modelo."""
    cols = {"pm25": scalers.pm25_scaler.transform(np.asarray(pm25, float).reshape(-1, 1)).ravel()}
    other = scalers.other_scaler.transform(np.column_stack([temp, dew]) if scalers.other_cols == ["temp", "dew"]
                                           else np.column_stack([dew, temp]))
    for i, c in enumerate(scalers.other_cols):
        cols[c] = other[:, i]
    return np.stack([cols[f] for f in feature_names], axis=-1)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    t0 = time.time()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["final_model"]["config"] == "A"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    test, fn, sc = data.test, data.feature_names, data.scalers
    pm_idx = fn.index("pm25")
    assert np.allclose(test.decoder_input[:, :LABEL_LEN, 0], test.encoder_input[:, -LABEL_LEN:, pm_idx])
    y = test.y_original.ravel()
    high = y >= HIGH
    last_obs = sc.inverse_transform_pm25(test.encoder_input[:, -1, pm_idx])
    ts = test.target_timestamps

    # Línea base de IG: ventana constante con la media de TRAIN por variable (escala del modelo)
    train_mean = data.train.encoder_input.mean(axis=(0, 1))
    enc_base = np.broadcast_to(train_mean, test.encoder_input.shape[1:]).astype(np.float32).copy()
    dec_base = decoder_from_encoder(enc_base[None], pm_idx)[0]
    train_mean_phys = {
        "pm25": float(sc.inverse_transform_pm25(train_mean[pm_idx])[0]),
        **{c: float(v) for c, v in zip(sc.other_cols, sc.other_scaler.inverse_transform(
            train_mean[[fn.index(c) for c in sc.other_cols]].reshape(1, -1)).ravel())},
    }

    rng = np.random.default_rng(2026)
    ig_idx = np.concatenate([np.where(high)[0], rng.choice(np.where(~high)[0], IG_N_NORMAL, replace=False)])
    ig_high = high[ig_idx]

    synth_cases = {
        "constante_10": np.full(24, 10.0), "constante_20": np.full(24, 20.0),
        "constante_40": np.full(24, 40.0), "constante_60": np.full(24, 60.0),
        "escalon_15_a_35_ultimas_4h": np.r_[np.full(20, 15.0), np.full(4, 35.0)],
        "rampa_15_a_45_ultimas_6h": np.r_[np.full(18, 15.0), np.linspace(20, 45, 6)],
        "descenso_45_a_15_ultimas_6h": np.r_[np.full(18, 45.0), np.linspace(40, 15, 6)],
        "pico_aislado_hace_12h": np.r_[np.full(11, 15.0), [60.0], np.full(12, 15.0)],
        "pico_en_la_ultima_hora": np.r_[np.full(23, 15.0), [60.0]],
    }
    synth_enc = np.stack([scale_window(v, np.full(24, train_mean_phys["temp"]), np.full(24, train_mean_phys["dew"]), sc, fn)
                          for v in synth_cases.values()]).astype(np.float32)
    synth_dec = decoder_from_encoder(synth_enc, pm_idx)

    # Ablaciones sobre test real
    abl = {}
    enc_meteo = test.encoder_input.copy()
    for c in ("temp", "dew"):
        enc_meteo[:, :, fn.index(c)] = train_mean[fn.index(c)]
    abl["meteo_neutralizada"] = (enc_meteo, test.decoder_input)
    perm = np.random.default_rng(17).permutation(24)
    enc_perm = test.encoder_input[:, perm, :]
    abl["orden_temporal_permutado"] = (enc_perm, decoder_from_encoder(enc_perm, pm_idx))
    enc_flat = test.encoder_input.copy()
    enc_flat[:, :, pm_idx] = enc_flat[:, -1:, pm_idx]
    abl["historia_pm25_plana_ultimo_valor"] = (enc_flat, decoder_from_encoder(enc_flat, pm_idx))

    results = {"seeds": {}}
    att_store, ig_store = {}, {}
    for seed in SEEDS:
        name = f"transfer_A_seed{seed}"
        info = protocol["checkpoint_integrity"]["runs"][name]
        model, _ = build_transfer_model("A", seed, data.train.encoder_input[:2], data.train.decoder_input[:2],
                                        architecture=CANDIDATE_V1)
        model.load_weights(str(OUTPUTS_DIR.parent / info["checkpoint"]))
        print(f"\n== {name}")

        att = attention_analysis(model, test, high)
        assert att["forward_check_max_abs_diff"] < 1e-5, att["forward_check_max_abs_diff"]
        att_store[seed] = att
        print(f"  atención: forward verificado (dif máx {att['forward_check_max_abs_diff']:.1e})")

        att_e, att_d, comp_err, mean_delta = integrated_gradients(
            model, test.encoder_input[ig_idx].astype(np.float32), test.decoder_input[ig_idx].astype(np.float32),
            enc_base, dec_base)
        # PM2,5 entra por encoder y decoder: se suman ambas rutas por hora (decoder = últimas 12 h)
        hour_feat = np.abs(att_e).copy()  # (N,24,3)
        hour_feat[:, -LABEL_LEN:, pm_idx] += np.abs(att_d[:, :LABEL_LEN, 0])
        share_feat = hour_feat.sum(axis=1) / hour_feat.sum(axis=(1, 2))[:, None]  # (N,3)
        ig_store[seed] = {"hour_feat_all": hour_feat.mean(0), "hour_feat_high": hour_feat[ig_high].mean(0),
                          "hour_feat_normal": hour_feat[~ig_high].mean(0)}
        print(f"  IG: error de completitud medio {comp_err.mean():.2e} (|f(x)-f(base)| medio {mean_delta:.3f})")

        pred = predict_ugm3(model, test.encoder_input, test.decoder_input, sc)
        ts_index = {t: i for i, t in enumerate(ts)}
        lag = {}
        for k in (0, 1, 2):
            pairs = [(pred[i], y[ts_index[t - np.timedelta64(k, "h")]]) for i, t in enumerate(ts)
                     if (t - np.timedelta64(k, "h")) in ts_index]
            p, r = np.array(pairs).T
            lag[f"corr_pred_vs_real_shift_{k}h"] = float(np.corrcoef(p, r)[0, 1])
        rises = (y - last_obs) >= 5.0
        falls = (y - last_obs) <= -5.0
        shortcut = {
            "corr_pred_last_observed": float(np.corrcoef(pred, last_obs)[0, 1]),
            "corr_pred_real": float(np.corrcoef(pred, y)[0, 1]),
            "rmse_pred_vs_persistence": rmse(pred, last_obs),
            "slope_pred_on_last_observed": float(np.polyfit(last_obs, pred, 1)[0]),
            **lag,
            "n_rises_ge5": int(rises.sum()),
            "fraction_of_rise_captured_mean": float(np.mean((pred[rises] - last_obs[rises]) / (y[rises] - last_obs[rises]))),
            "n_falls_le_minus5": int(falls.sum()),
            "fraction_of_fall_captured_mean": float(np.mean((pred[falls] - last_obs[falls]) / (y[falls] - last_obs[falls]))),
        }
        synth_pred = predict_ugm3(model, synth_enc, synth_dec, sc)
        base_rmse = rmse(pred, y)
        ablations = {}
        for k, (e, d) in abl.items():
            p = predict_ugm3(model, e, d, sc)
            ablations[k] = {"rmse": rmse(p, y), "rmse_change_vs_original": rmse(p, y) - base_rmse,
                            "rmse_high_ge40": rmse(p[high], y[high]),
                            "mean_abs_change_in_prediction": float(np.mean(np.abs(p - pred)))}
        results["seeds"][str(seed)] = {
            "forward_check_max_abs_diff": att["forward_check_max_abs_diff"],
            "ig_completeness_error_mean": float(comp_err.mean()),
            "ig_mean_abs_output_change_vs_baseline": mean_delta,
            "ig_feature_share_mean": dict(zip(fn, share_feat.mean(0).astype(float).tolist())),
            "ig_feature_share_high": dict(zip(fn, share_feat[ig_high].mean(0).astype(float).tolist())),
            "ig_share_last_hour_pm25": float((hour_feat[:, -1, pm_idx] / hour_feat.sum(axis=(1, 2))).mean()),
            "ig_share_last_6h_all_features": float((hour_feat[:, -6:, :].sum(axis=(1, 2)) / hour_feat.sum(axis=(1, 2))).mean()),
            "attention_enc_long1_mass_last_hour": float(att["enc_long1_key_mass"]["all"][-1]),
            "attention_enc_long1_mass_last_6h": float(att["enc_long1_key_mass"]["all"][-6:].sum()),
            "attention_enc_long1_mass_uniform_reference_last_6h": 6 / 24,
            "attention_cross_mass_short_half": float(att["cross_attention_pred_position"]["all"][12:].sum()),
            "dec_pred_position_selected_frac": att["dec_pred_position_selected_frac"],
            "shortcut_persistence": shortcut,
            "synthetic_cases_pred_ug_m3": dict(zip(synth_cases, synth_pred.astype(float).tolist())),
            "ablations_test": {"original_rmse": base_rmse, **ablations},
        }
        print(f"  atajo: corr(pred, última hora) {shortcut['corr_pred_last_observed']:.3f} | "
              f"corr(pred, real) {shortcut['corr_pred_real']:.3f} | subida captada {shortcut['fraction_of_rise_captured_mean']:.2f}")

    # Agregados entre semillas
    def agg(path):
        vals = []
        for s in SEEDS:
            v = results["seeds"][str(s)]
            for p in path:
                v = v[p]
            vals.append(v)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "per_seed": vals}

    results["aggregate"] = {
        "ig_share_pm25": agg(["ig_feature_share_mean", "pm25"]),
        "ig_share_temp": agg(["ig_feature_share_mean", "temp"]),
        "ig_share_dew": agg(["ig_feature_share_mean", "dew"]),
        "ig_share_last_hour_pm25": agg(["ig_share_last_hour_pm25"]),
        "ig_share_last_6h_all_features": agg(["ig_share_last_6h_all_features"]),
        "attention_enc_long1_mass_last_6h": agg(["attention_enc_long1_mass_last_6h"]),
        "corr_pred_last_observed": agg(["shortcut_persistence", "corr_pred_last_observed"]),
        "corr_pred_real": agg(["shortcut_persistence", "corr_pred_real"]),
        "fraction_of_rise_captured": agg(["shortcut_persistence", "fraction_of_rise_captured_mean"]),
        "fraction_of_fall_captured": agg(["shortcut_persistence", "fraction_of_fall_captured_mean"]),
        **{f"ablation_{k}_rmse_change": agg(["ablations_test", k, "rmse_change_vs_original"]) for k in abl},
        **{f"synthetic_{k}": agg(["synthetic_cases_pred_ug_m3", k]) for k in synth_cases},
    }
    results["setup"] = {
        "ig": {"steps": IG_STEPS, "baseline": "ventana constante con la media de train por variable",
               "baseline_physical_units": train_mean_phys, "n_windows": int(len(ig_idx)),
               "n_high_ge40": int(ig_high.sum()), "n_normal_random": IG_N_NORMAL, "rng_seed": 2026},
        "synthetic_meteo": "temperatura y punto de rocío constantes en la media de train",
        "ablation_permutation_seed": 17,
        "attention_note": ("enc0_long1 opera sobre las 24 horas reales (posición 23 = hora t); después de la "
                           "primera unidad las posiciones mezclan horas (pooling/concatenación), por eso solo "
                           "las capas de la unidad 0 se leen como horas."),
        "training_performed": False,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # Arreglos crudos para poder rehacer figuras sin recalcular
    np.savez(OUTPUTS_DIR / "metrics" / "F3_explainability_arrays.npz",
             **{f"att_long1_{g}_seed{s}": att_store[s]["enc_long1_key_mass"][g] for s in SEEDS for g in ("all", "high", "normal")},
             **{f"ig_hour_feat_{g}_seed{s}": ig_store[s][f"hour_feat_{g}"] for s in SEEDS for g in ("all", "high", "normal")},
             feature_names=np.array(fn))

    # ------------------------------------------------------------------ figuras
    hours = np.arange(-23, 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for s, color in zip(SEEDS, (BLUE, ORANGE, AQUA)):
        ax.plot(hours, att_store[s]["enc_long1_key_mass"]["all"], color=color, linewidth=2,
                marker="o", markersize=4, label=f"Semilla {s}")
    ax.axhline(1 / 24, color=INK2, linewidth=1, linestyle="--", label="Atención uniforme (1/24)")
    ax.set_xlabel("Hora relativa en la ventana (0 = última hora observada)", color=INK2)
    ax.set_ylabel("Masa media de atención", color=INK2)
    ax.set_title("Atención del encoder sobre las 24 horas (modelo A, test)", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attention_encoder_by_hour.png", dpi=200)
    plt.close(fig)

    hf = np.mean([ig_store[s]["hour_feat_all"] for s in SEEDS], axis=0)  # (24,3)
    hf = hf / hf.sum()
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    im = ax.imshow(hf.T, aspect="auto", cmap=SEQ, extent=[-23.5, 0.5, len(fn) - 0.5, -0.5])
    ax.set_yticks(range(len(fn)), [FEATURE_LABELS[f] for f in fn])
    ax.set_xlabel("Hora relativa en la ventana (0 = última hora observada)", color=INK2)
    ax.set_title("Importancia por hora y variable (gradientes integrados)", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "integrated_gradients_hour_feature.png", dpi=200)
    plt.close(fig)

    names = list(synth_cases)
    pretty = {
        "constante_10": "Constante 10", "constante_20": "Constante 20", "constante_40": "Constante 40",
        "constante_60": "Constante 60", "escalon_15_a_35_ultimas_4h": "Escalón 15→35 (últimas 4 h)",
        "rampa_15_a_45_ultimas_6h": "Rampa 15→45 (últimas 6 h)", "descenso_45_a_15_ultimas_6h": "Descenso 45→15 (últimas 6 h)",
        "pico_aislado_hace_12h": "Pico de 60 hace 12 h", "pico_en_la_ultima_hora": "Pico de 60 en la última hora",
    }
    means = [results["aggregate"][f"synthetic_{k}"]["mean"] for k in names]
    lasts = [synth_cases[k][-1] for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    yy = np.arange(len(names))
    ax.barh(yy - 0.2, lasts, height=0.38, color="#c3c2b7", label="Última hora observada")
    ax.barh(yy + 0.2, means, height=0.38, color=BLUE, label="Predicción del modelo A (media de 3 semillas)")
    for yi, m in zip(yy, means):
        ax.text(m + 0.8, yi + 0.2, f"{m:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yy, [pretty[n] for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("PM2,5 (µg/m³)", color=INK2)
    ax.set_title("Casos sintéticos de respuesta conocida", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.13))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_cases.png", dpi=200)
    plt.close(fig)

    print(f"\nResumen: {OUT_JSON}\nFiguras: {FIG_DIR}\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
