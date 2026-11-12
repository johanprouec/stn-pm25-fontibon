"""Fontibon (Bogota) preprocessing pipeline.

This module is a direct refactor of the pipeline developed in
``PrimeraEntrega_DL.ipynb`` (cells 1-2-4). The logic — Magnus dew-point
formula, the 70/15/15 chronological split by row position, the <=6h causal
interpolation of input variables, the causal-safety check that rejects any
window whose interpolation used information at or after the target hour, and
the rule that the PM2.5 target is never imputed — is reproduced unchanged.
See docs/PHASE1_AUDIT.md for the verification that this refactor reproduces
the notebook's original numbers exactly (train=10802, val=2564, test=2574
windows; Ridge val RMSE=6.545, test RMSE=7.063).

Do not alter the interpolation/window-validity logic without re-verifying
against the notebook — it encodes the leak-prevention rules the project
depends on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.data.common import PreparedSplit
from src.data.transfer_features import TRANSFER_FEATURE_NAMES
from src.models.stn import build_decoder_input

STATION = "FONTIBON"
DEW_POINT_A = 17.625
DEW_POINT_B = 243.04

PM25_COL = "PM2.5_µg/m3"
TEMP_COL = "Temperature_°C"
RH_COL = "RH_%"


def _to_numeric(series: pd.Series) -> pd.Series:
    """Comma-decimal-safe numeric coercion, as used in the notebook."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def magnus_dew_point(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Dew point (deg C) via the Magnus formula, parameters a=17.625, b=243.04.

    Ported unchanged from the notebook. Source of the constants: standard
    Magnus-Tetens approximation (as cited implicitly in the notebook markdown);
    not re-derived here.
    """
    gamma = np.log(rh_pct / 100) + (temp_c * DEW_POINT_A) / (DEW_POINT_B + temp_c)
    return (DEW_POINT_B * gamma) / (DEW_POINT_A - gamma)


def load_fontibon_features(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV, filter to the FONTIBON station, and build the
    3-column feature frame (pm25, temp, dew) indexed by hourly timestamp.

    Mirrors notebook cell 2 exactly, including the BOM-stripping of column
    names and the comma-decimal parsing of numeric columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]

    df = df[df["STATION"] == STATION].copy()

    fecha = pd.to_datetime(df["DATE"], format="%d/%m/%Y")
    hora = df["HOUR"].astype(str).str.slice(0, 2).astype(int)
    df["ts"] = fecha + pd.to_timedelta(hora, unit="h")
    df = df.sort_values("ts").set_index("ts")

    X = pd.DataFrame({
        "pm25": _to_numeric(df[PM25_COL]),
        "temp": _to_numeric(df[TEMP_COL]),
        "hr": _to_numeric(df[RH_COL]),
    })
    X["dew"] = magnus_dew_point(X["temp"], X["hr"])

    return X[TRANSFER_FEATURE_NAMES]  # ["pm25", "temp", "dew"] — must match Beijing R2 exactly, see src/data/transfer_features.py


def chronological_cuts(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Dict[str, Tuple[int, int]]:
    """Row-position cut points for train/val/test, matching the notebook's
    ``int(n * 0.70)`` / ``int(n * 0.85)`` positional split (70/15/15)."""
    return {
        "train": (0, int(n * train_frac)),
        "val": (int(n * train_frac), int(n * (train_frac + val_frac))),
        "test": (int(n * (train_frac + val_frac)), n),
    }


def interpolate_short_gaps(part: pd.DataFrame, limite: int = 6):
    """Linearly interpolate only internal gaps of length <= ``limite``,
    requiring observed values on both sides. Returns the filled frame plus,
    per column, the index of the right endpoint used for each interpolated
    position (-1 where no interpolation occurred) — needed by
    :func:`build_windows` to reject windows that would leak future
    information. Ported unchanged from the notebook.
    """
    lleno = part.copy()
    extremo_derecho = {col: np.full(len(part), -1, dtype=int) for col in part.columns}

    for col in part.columns:
        valores = part[col].to_numpy(dtype=float)
        faltante = np.isnan(valores)
        i = 0
        while i < len(valores):
            if not faltante[i]:
                i += 1
                continue
            j = i
            while j < len(valores) and faltante[j]:
                j += 1
            longitud = j - i
            if (
                longitud <= limite
                and i > 0
                and j < len(valores)
                and not np.isnan(valores[i - 1])
                and not np.isnan(valores[j])
            ):
                izquierda = valores[i - 1]
                derecha = valores[j]
                for k in range(i, j):
                    proporcion = (k - (i - 1)) / (j - (i - 1))
                    lleno.iloc[k, lleno.columns.get_loc(col)] = izquierda + proporcion * (derecha - izquierda)
                    extremo_derecho[col][k] = j
            i = j

    return lleno, extremo_derecho


def build_windows(
    X_original: pd.DataFrame,
    inicio: int,
    fin: int,
    ventana: int = 24,
    limite: int = 6,
    target_col: str = "pm25",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build causal, leakage-safe (ventana, n_features) windows within a
    single partition slice ``X_original.iloc[inicio:fin]``.

    A window at position k is kept only if: (1) the target at k was
    actually observed (never imputed), (2) the ventana-hour history is fully
    populated after short-gap interpolation, and (3) none of the
    interpolations used inside that history relied on information at or
    after k (checked via the ``extremo_derecho`` bookkeeping from
    :func:`interpolate_short_gaps`). Because ``X_original`` is sliced to
    ``[inicio:fin]`` before any interpolation happens, windows can never
    reach across a train/val/test boundary. Ported unchanged from the
    notebook, with ``target_col`` generalized (default ``"pm25"`` preserves
    the original Fontibon behavior exactly) so the same function can be
    reused for Beijing's R1 preprocessing (``src/data/beijing.py``), which
    also names its target column ``"pm25"`` but has additional feature
    columns the notebook never had.
    """
    parte = X_original.iloc[inicio:fin].copy()
    parte_interp, extremo = interpolate_short_gaps(parte, limite)

    Xs, ys, fechas = [], [], []

    for k in range(ventana, len(parte)):
        if pd.isna(parte.iloc[k][target_col]):
            continue

        historial = parte_interp.iloc[k - ventana:k]
        if historial.isna().any().any():
            continue

        valido = True
        for col in parte_interp.columns:
            utilizados = extremo[col][k - ventana:k]
            if np.any((utilizados >= 0) & (utilizados >= k)):
                valido = False
                break
        if not valido:
            continue

        Xs.append(historial.to_numpy())
        ys.append(parte.iloc[k][target_col])
        fechas.append(parte.index[k])

    return np.array(Xs), np.array(ys), pd.DatetimeIndex(fechas)


def build_all_partitions(
    X: pd.DataFrame,
    ventana: int = 24,
    limite: int = 6,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Convenience wrapper: chronological cuts + windows for all three
    partitions, in one call."""
    cortes = chronological_cuts(len(X), train_frac, val_frac)
    return {
        nombre: build_windows(X, inicio, fin, ventana=ventana, limite=limite)
        for nombre, (inicio, fin) in cortes.items()
    }


def flatten(A: np.ndarray) -> np.ndarray:
    """(n, ventana, n_features) -> (n, ventana * n_features), as used to
    feed the Ridge/Random Forest baselines."""
    return A.reshape(len(A), -1)


# ============================================================================
# Fontibon STN preparation — the TARGET side of the transfer experiments
# (Config A/B/C/D, not implemented/executed in this module). Builds the
# exact same (encoder_input, decoder_input, y_scaled, y_original,
# target_timestamps) contract as src/data/beijing.py's R1/R2 preparation
# (``src.data.common.PreparedSplit``), on top of the ALREADY-AUDITED
# Fontibon windowing (``load_fontibon_features``, ``chronological_cuts``,
# ``build_windows`` — same functions the Primera Entrega's Ridge baseline
# uses, unchanged). Window counts are therefore guaranteed, by
# construction, to match the already-verified 10802/2564/2574
# (tests/test_partitions.py) — scaling an already-NaN-aware column with
# sklearn's MinMaxScaler does not change which rows are missing (NaN stays
# NaN under both fit, which uses nanmin/nanmax, and transform), so it
# cannot change which windows pass the causal-validity check in
# ``build_windows``.
#
# Deliberately NOT reusing Beijing's R2 scaler (BeijingR2Scalers): transfer
# compatibility between Beijing and Fontibon is about matching the SHAPE
# and SEMANTIC ORDER of the 3 input channels (both already guaranteed by
# TRANSFER_FEATURE_NAMES), not about sharing numeric scaling statistics —
# Beijing PM2.5 (train mean ~98.6 ug/m3) and Fontibon PM2.5 (train mean
# ~19.1 ug/m3, see tests/test_scaler_leakage.py) have very different
# distributions, and fitting one scaler on both (or reusing Beijing's)
# would distort Fontibon's own [0, 1] normalization range. Each domain
# keeps an independently-fit scaler, exactly as already decided for
# Beijing R1 vs R2 (see docs/IMPLEMENTATION_ASSUMPTIONS.md).
# ============================================================================
@dataclass
class FontibonSTNScalers:
    """Fit exclusively on Fontibon TRAIN (the same 70% chronological slice
    already used for the Ridge baseline — ``chronological_cuts``), never
    on Beijing. `pm25_scaler` is kept separate from `other_scaler` so
    PM2.5 predictions can be inverse-transformed without the other 2
    variables' scaling parameters."""
    pm25_scaler: MinMaxScaler
    other_scaler: MinMaxScaler  # fit on [temp, dew], in that order
    other_cols: List[str]

    def inverse_transform_pm25(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        return self.pm25_scaler.inverse_transform(y_scaled).ravel()


@dataclass
class FontibonPreparedData:
    train: PreparedSplit
    validation: PreparedSplit
    test: Optional[PreparedSplit]
    scalers: FontibonSTNScalers
    feature_names: List[str]


def prepare_fontibon_stn(
    csv_path: Path,
    ventana: int = 24,
    limite: int = 6,
    decoder_label_len: int = 12,
    prediction_length: int = 1,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    include_test: bool = False,
) -> FontibonPreparedData:
    """Build leakage-safe STN-ready windows for Fontibon, matching the
    already-approved 70/15/15 chronological split (same split used for the
    Primera Entrega's Ridge baseline — ``chronological_cuts``,
    ``baseline_fontibon_ridge.json``: val RMSE=6.545, test RMSE=7.063).

    `include_test=False` by default, same convention as
    `prepare_beijing_r1`/`prepare_beijing_r2`: this function CAN build the
    test split (needed eventually for the Fase 3 final evaluation), but no
    Fase 2 transfer experiment (A/B/C/D) may ever call this with
    `include_test=True`, let alone evaluate it — see
    docs/PHASE2_PLAN.md, "Regla de oro".
    """
    X = load_fontibon_features(csv_path)  # already in TRANSFER_FEATURE_NAMES order: pm25, temp, dew

    cortes = chronological_cuts(len(X), train_frac, val_frac)
    train_pos, val_pos, test_pos = cortes["train"], cortes["val"], cortes["test"]

    # --- MinMax scalers, fit on TRAIN only (own scaler, see module note
    # above — never Beijing's). NaN passes through sklearn's
    # MinMaxScaler.transform unchanged (fit uses nanmin/nanmax, disregarding
    # missing values), so this does not interact with the interpolation/
    # validity logic inside build_windows below. ---
    train_raw_slice = X.iloc[train_pos[0]:train_pos[1]]
    pm25_scaler = MinMaxScaler().fit(train_raw_slice[["pm25"]].to_numpy())
    other_cols = ["temp", "dew"]
    other_scaler = MinMaxScaler().fit(train_raw_slice[other_cols].to_numpy())

    scaled = pd.DataFrame(index=X.index)
    scaled["pm25"] = pm25_scaler.transform(X[["pm25"]].to_numpy()).ravel()
    scaled[other_cols] = other_scaler.transform(X[other_cols].to_numpy())
    scaled = scaled[TRANSFER_FEATURE_NAMES]

    cuts = {"train": train_pos, "validation": val_pos}
    if include_test:
        cuts["test"] = test_pos

    splits: Dict[str, PreparedSplit] = {}
    for name, (start, end) in cuts.items():
        X_win, y_win, t_win = build_windows(scaled, start, end, ventana=ventana, limite=limite, target_col="pm25")

        y_original = pm25_scaler.inverse_transform(y_win.reshape(-1, 1)).ravel()

        pm25_channel = TRANSFER_FEATURE_NAMES.index("pm25")
        pm25_history_scaled = X_win[:, :, pm25_channel]
        decoder_in = build_decoder_input(
            tf.convert_to_tensor(pm25_history_scaled, dtype=tf.float32),
            label_len=decoder_label_len, prediction_length=prediction_length,
        ).numpy()

        splits[name] = PreparedSplit(
            encoder_input=X_win,
            decoder_input=decoder_in,
            y_scaled=y_win.reshape(-1, 1),
            y_original=y_original.reshape(-1, 1),
            target_timestamps=t_win,
        )

    scalers = FontibonSTNScalers(pm25_scaler=pm25_scaler, other_scaler=other_scaler, other_cols=other_cols)

    return FontibonPreparedData(
        train=splits["train"],
        validation=splits["validation"],
        test=splits.get("test"),
        scalers=scalers,
        feature_names=list(TRANSFER_FEATURE_NAMES),
    )


def build_fontibon_stn_preprocessing_audit(data: FontibonPreparedData) -> Dict:
    """Pre-training audit payload — preprocessing facts only, deliberately
    no predictive/validation/test metric of any kind (mirrors
    src.data.beijing.build_r2_preprocessing_audit)."""
    return {
        "feature_names": data.feature_names,
        "feature_order_source_of_truth": "src.data.transfer_features.TRANSFER_FEATURE_NAMES",
        "n_windows": {
            "train": data.train.n_windows,
            "validation": data.validation.n_windows,
            "test": data.test.n_windows if data.test is not None else None,
        },
        "n_windows_expected_from_ridge_baseline": {"train": 10802, "validation": 2564, "test": 2574},
        "pm25_train_min_max_ug_m3": [
            float(data.scalers.pm25_scaler.data_min_[0]), float(data.scalers.pm25_scaler.data_max_[0]),
        ],
        "temp_train_min_max": [
            float(data.scalers.other_scaler.data_min_[0]), float(data.scalers.other_scaler.data_max_[0]),
        ],
        "dew_train_min_max": [
            float(data.scalers.other_scaler.data_min_[1]), float(data.scalers.other_scaler.data_max_[1]),
        ],
        "encoder_input_shape": list(data.train.encoder_input.shape[1:]),
        "decoder_input_shape": list(data.train.decoder_input.shape[1:]),
        "scaler_is_fontibons_own_not_beijings": True,
        "beijing_test_2014_used": False,
        "fontibon_test_used": data.test is not None,
        "note": (
            "No predictive/validation/test metric of any kind is included here — "
            "preprocessing facts only. `fontibon_test_used` reflects whether the test "
            "split was BUILT (include_test=True), not whether it was ever evaluated — "
            "no Phase 2 script may evaluate it regardless."
        ),
    }
