"""Domain-distance analysis: Beijing vs. Fontibon, TRAIN partitions ONLY,
on the 3 variables shared by the transfer experiments (pm25, temp, dew).

Purely descriptive — computes statistics about how different the two
domains are, it does NOT select, tune, or even look at any hyperparameter
or architecture choice, and it never touches Beijing test 2014 or any
Fontibon split other than train. Its only purpose is to document, BEFORE
A/B/C/D are run, how large the raw-level and standardized-level gap
between domains is, so that whatever A/B/C/D eventually find can be
interpreted against this baseline instead of in a vacuum.

Usage: python scripts/run_domain_distance_analysis.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.data.beijing import BEIJING_TRAIN_END, BEIJING_TRAIN_START, load_beijing_raw
from src.data.fontibon import chronological_cuts, load_fontibon_features
from src.utils.paths import BEIJING_CSV, FONTIBON_CSV, OUTPUTS_DIR

VARIABLES = ["pm25", "temp", "dew"]


def _position_range(index: pd.DatetimeIndex, start: str, end: str) -> Tuple[int, int]:
    mask = (index >= start) & (index <= end)
    positions = np.where(mask)[0]
    return int(positions.min()), int(positions.max()) + 1


def beijing_train_raw() -> pd.DataFrame:
    df = load_beijing_raw(BEIJING_CSV)
    raw = pd.DataFrame(
        {"pm25": df["pm2.5"].astype(float), "temp": df["TEMP"].astype(float), "dew": df["DEWP"].astype(float)},
        index=df.index,
    )
    start, end = _position_range(raw.index, BEIJING_TRAIN_START, BEIJING_TRAIN_END)
    return raw.iloc[start:end]


def fontibon_train_raw() -> pd.DataFrame:
    X = load_fontibon_features(FONTIBON_CSV)  # already pm25, temp, dew
    cortes = chronological_cuts(len(X))
    start, end = cortes["train"]
    return X.iloc[start:end]


def descriptive_stats(series: pd.Series) -> Dict:
    s = series.dropna()
    return {
        "n": int(len(s)), "n_missing": int(series.isna().sum()),
        "mean": float(s.mean()), "std": float(s.std()),
        "min": float(s.min()), "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)), "p95": float(s.quantile(0.95)), "max": float(s.max()),
        "range": float(s.max() - s.min()),
    }


def acf_at_lags(series: pd.Series, lags=(1, 24, 168)) -> Dict:
    return {f"lag_{l}h": (float(series.autocorr(lag=l)) if series.dropna().shape[0] > l else None) for l in lags}


def hourly_profile(series: pd.Series) -> Dict[int, float]:
    return {int(h): float(v) for h, v in series.groupby(series.index.hour).mean().items()}


def main() -> None:
    print("=" * 70)
    print("DOMAIN DISTANCE ANALYSIS -- Beijing vs Fontibon, TRAIN only, descriptive")
    print("Beijing test 2014 and Fontibon validation/test are NEVER touched below.")
    print("=" * 70)

    beijing = beijing_train_raw()
    fontibon = fontibon_train_raw()
    print(f"\nBeijing train rows: {len(beijing)} | Fontibon train rows: {len(fontibon)}")

    stats = {"beijing": {v: descriptive_stats(beijing[v]) for v in VARIABLES},
              "fontibon": {v: descriptive_stats(fontibon[v]) for v in VARIABLES}}

    acf = {"beijing": acf_at_lags(beijing["pm25"]), "fontibon": acf_at_lags(fontibon["pm25"])}
    hourly = {"beijing": hourly_profile(beijing["pm25"]), "fontibon": hourly_profile(fontibon["pm25"])}

    wasserstein_raw, wasserstein_std = {}, {}
    for var in VARIABLES:
        b = beijing[var].dropna().to_numpy()
        f = fontibon[var].dropna().to_numpy()
        wasserstein_raw[var] = float(wasserstein_distance(b, f))
        b_std = (b - b.mean()) / b.std()
        f_std = (f - f.mean()) / f.std()
        wasserstein_std[var] = float(wasserstein_distance(b_std, f_std))

    print("\nDescriptive stats (pm25, train):")
    print(f"  Beijing:  mean={stats['beijing']['pm25']['mean']:.2f}  std={stats['beijing']['pm25']['std']:.2f}  max={stats['beijing']['pm25']['max']:.1f}")
    print(f"  Fontibon: mean={stats['fontibon']['pm25']['mean']:.2f}  std={stats['fontibon']['pm25']['std']:.2f}  max={stats['fontibon']['pm25']['max']:.1f}")
    print(f"\nWasserstein distance (raw units): {wasserstein_raw}")
    print(f"Wasserstein distance (standardized): {wasserstein_std}")

    # --- Figures ---
    figures_dir = OUTPUTS_DIR / "figures" / "domain_distance"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for var in VARIABLES:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(beijing[var].dropna(), bins=60, alpha=0.5, density=True, label="Beijing (train)")
        ax.hist(fontibon[var].dropna(), bins=60, alpha=0.5, density=True, label="Fontibon (train)")
        ax.set_xlabel(var)
        ax.set_ylabel("densidad")
        ax.set_title(f"Distribucion {var} -- Beijing vs Fontibon (train)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"distribution_{var}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = dict(sorted(hourly["beijing"].items()))
    hf = dict(sorted(hourly["fontibon"].items()))
    ax.plot(list(hb.keys()), list(hb.values()), marker="o", label="Beijing")
    ax.plot(list(hf.keys()), list(hf.values()), marker="o", label="Fontibon")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("PM2.5 medio (ug/m3)")
    ax.set_title("Perfil horario promedio de PM2.5 -- train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "hourly_profile_pm25.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lags = [1, 24, 168]
    ax.bar(np.arange(len(lags)) - 0.15, [acf["beijing"][f"lag_{l}h"] for l in lags], width=0.3, label="Beijing")
    ax.bar(np.arange(len(lags)) + 0.15, [acf["fontibon"][f"lag_{l}h"] for l in lags], width=0.3, label="Fontibon")
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"{l}h" for l in lags])
    ax.set_ylabel("Autocorrelacion PM2.5")
    ax.set_title("Autocorrelacion de PM2.5 en lag 1/24/168h -- train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "acf_pm25.png", dpi=150)
    plt.close(fig)

    result = {
        "label": "DOMAIN DISTANCE ANALYSIS -- descriptive only, no hyperparameter selection",
        "variables": VARIABLES,
        "descriptive_stats": stats,
        "autocorrelation_pm25": acf,
        "hourly_profile_pm25_mean_ug_m3": hourly,
        "wasserstein_distance_raw_units": wasserstein_raw,
        "wasserstein_distance_standardized": wasserstein_std,
        "scale_difference_note": (
            f"Beijing train PM2.5 mean {stats['beijing']['pm25']['mean']:.1f} ug/m3 "
            f"(std {stats['beijing']['pm25']['std']:.1f}, max {stats['beijing']['pm25']['max']:.0f}) vs. "
            f"Fontibon train PM2.5 mean {stats['fontibon']['pm25']['mean']:.1f} ug/m3 "
            f"(std {stats['fontibon']['pm25']['std']:.1f}, max {stats['fontibon']['pm25']['max']:.0f}). "
            "The absolute PM2.5 level and spread differ by roughly a factor of 5 between domains "
            "(this is why each domain keeps its OWN MinMax scaler — see "
            "docs/IMPLEMENTATION_ASSUMPTIONS.md). The transfer hypothesis for Config B/C/D is NOT "
            "that absolute PM2.5 levels transfer, but that TEMPORAL structure (autocorrelation, "
            "hourly profile shape) captured by the encoder/decoder may partially transfer despite "
            "the level/scale difference -- the standardized Wasserstein distance above is the "
            "quantitative counterpart of that hypothesis (comparing SHAPE of the distributions, "
            "with the level/scale difference removed by standardization)."
        ),
        "figures_dir": str(figures_dir),
        "beijing_test_2014_used": False,
        "fontibon_validation_or_test_used": False,
        "used_to_select_ABCD_hyperparameters": False,
    }
    out_path = OUTPUTS_DIR / "metrics" / "domain_distance_analysis.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {out_path}")
    print(f"Figures: {figures_dir}")
    print("Beijing test 2014: NOT used. Fontibon validation/test: NOT used.")
    print("=" * 70)


if __name__ == "__main__":
    main()
