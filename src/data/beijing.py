"""Beijing PM2.5 (UCI / Chen 2015) loading, audit, and R1 preparation.

R1 preparation (``prepare_beijing_r1``) builds leakage-safe, chronologically
partitioned windows for the 8 conceptual variables the paper reports
(PM2.5, dew point, temperature, pressure, wind direction, accumulated wind
speed, accumulated snow, accumulated rain), with wind direction one-hot
encoded (never ordinal). It deliberately reuses the exact windowing/
short-gap-interpolation logic already audited for Fontibon
(``src/data/fontibon.py: interpolate_short_gaps``, ``build_windows``) rather
than re-implementing it — see docs/PHASE1_AUDIT.md for that logic's
verification and docs/IMPLEMENTATION_ASSUMPTIONS.md, "Beijing preprocessing
(R1)", for what is a deliberate adaptation versus what the paper actually
specifies (it does not describe how it avoids leakage during interpolation
at all).

Wind direction (``cbwd``) is categorical (SE/NW/cv/NE in this file) and
must be one-hot encoded, never mapped to an arbitrary integer — an ordinal
mapping like NE=0, NW=1, SE=2, cv=3 would fabricate a distance/ordering
between categories that does not exist in the source variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.data.common import PreparedSplit  # re-exported below for backward compatibility
from src.data.fontibon import build_windows
from src.data.transfer_features import TRANSFER_FEATURE_NAMES
from src.models.stn import build_decoder_input

WIND_DIRECTION_CATEGORIES = ["SE", "NW", "cv", "NE"]

# Approved chronological split (docs/PHASE2_PLAN.md §2). Train/validation
# are contiguous with no gap; test is the paper's own held-out year and is
# NOT consulted anywhere in this module's pilot-preparation path unless the
# caller explicitly asks for it (see `include_test` on `prepare_beijing_r1`).
BEIJING_TRAIN_START = "2010-01-01 00:00"
BEIJING_TRAIN_END = "2013-08-31 23:00"
BEIJING_VAL_START = "2013-09-01 00:00"
BEIJING_VAL_END = "2013-12-31 23:00"
BEIJING_TEST_START = "2014-01-01 00:00"
BEIJING_TEST_END = "2014-12-31 23:00"

# R1's 7 continuous conceptual variables (an 8th, wind direction, is
# categorical and handled separately via one-hot). "pm25" matches
# Fontibon's column-naming convention so build_windows' default
# target_col="pm25" needs no override.
CONTINUOUS_COLS = ["pm25", "dewp", "temp", "pres", "iws", "is_", "ir"]
OTHER_CONTINUOUS_COLS = ["dewp", "temp", "pres", "iws", "is_", "ir"]  # everything except the target


def load_beijing_raw(csv_path: Path) -> pd.DataFrame:
    """Load the PRSA CSV and attach an hourly timestamp index.

    Column meanings (as published): pm2.5 (ug/m3), DEWP (dew point, deg C),
    TEMP (deg C), PRES (hPa), cbwd (categorical wind direction), Iws
    (accumulated wind speed, m/s), Is (accumulated snow hours), Ir
    (accumulated rain hours).
    """
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("ts").set_index("ts")
    return df


def one_hot_wind_direction(cbwd: pd.Series) -> pd.DataFrame:
    """One-hot encode ``cbwd`` (no ordinal/integer encoding)."""
    return pd.get_dummies(cbwd, prefix="cbwd")


def missing_blocks(is_na: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True-runs in a boolean missingness array."""
    blocks = []
    i = 0
    n = len(is_na)
    while i < n:
        if is_na[i]:
            j = i
            while j < n and is_na[j]:
                j += 1
            blocks.append(j - i)
            i = j
        else:
            i += 1
    return np.array(blocks, dtype=int)


def audit_summary(df: pd.DataFrame) -> Dict:
    """Numeric summary used to cross-check against the paper and
    docs/PHASE1_AUDIT.md. Returns plain Python types (JSON-serializable)."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_ts = full_range.difference(df.index)
    pm25_na = df["pm2.5"].isna().to_numpy()
    blocks = missing_blocks(pm25_na)

    return {
        "n_rows": int(len(df)),
        "date_min": str(df.index.min()),
        "date_max": str(df.index.max()),
        "n_duplicated_timestamps": int(df.index.duplicated().sum()),
        "n_missing_timestamps": int(len(missing_ts)),
        "pm25_missing_count": int(pm25_na.sum()),
        "pm25_missing_pct": float(pm25_na.mean() * 100),
        "pm25_missing_n_blocks": int(len(blocks)),
        "pm25_missing_max_block": int(blocks.max()) if len(blocks) else 0,
        "pm25_missing_blocks_gt_24h": int((blocks > 24).sum()) if len(blocks) else 0,
        "cbwd_value_counts": df["cbwd"].value_counts().to_dict(),
    }


# ============================================================================
# R1 preparation: chronological split -> train-only-fit one-hot(cbwd) and
# MinMax scaling -> leakage-safe windows -> decoder input, per variable.
# ============================================================================
@dataclass
class BeijingScalers:
    """Fit exclusively on TRAIN. `pm25_scaler` is kept separate from
    `other_scaler` specifically so PM2.5 predictions can be inverse-
    transformed without touching (or needing) the other 6 continuous
    variables' scaling parameters."""
    pm25_scaler: MinMaxScaler
    other_scaler: MinMaxScaler
    cbwd_encoder: OneHotEncoder
    other_cols: List[str]
    cbwd_categories: List[str]

    def inverse_transform_pm25(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        return self.pm25_scaler.inverse_transform(y_scaled).ravel()


@dataclass
class BeijingPreparedData:
    train: PreparedSplit
    validation: PreparedSplit
    test: Optional[PreparedSplit]
    scalers: BeijingScalers
    feature_names: List[str]


def _position_range(index: pd.DatetimeIndex, start: str, end: str) -> Tuple[int, int]:
    """[start_pos, end_pos) — a half-open position slice covering every row
    with timestamp in [start, end] inclusive."""
    mask = (index >= start) & (index <= end)
    positions = np.where(mask)[0]
    if len(positions) == 0:
        raise ValueError(f"no rows found in range [{start}, {end}]")
    return int(positions.min()), int(positions.max()) + 1


def prepare_beijing_r1(
    csv_path: Path,
    ventana: int = 24,
    limite: int = 6,
    decoder_label_len: int = 12,
    prediction_length: int = 1,
    include_test: bool = False,
) -> BeijingPreparedData:
    """Build leakage-safe R1 windows for Beijing, per the approved
    chronological split.

    Reuses `src/data/fontibon.py`'s already-audited `interpolate_short_gaps`
    / `build_windows` (short-gap-only interpolation for input variables,
    never for the PM2.5 target, with the causal "no interpolation may use
    information at or after the target hour" guarantee — see
    docs/PHASE1_AUDIT.md). Only PM2.5 has missing values in the raw file
    (verified in docs/PHASE1_AUDIT.md §6); the other 6 continuous columns
    pass through interpolation unchanged since they have none.

    `include_test=False` by default: this function CAN build the test 2014
    split (needed eventually for the R1 replication comparison against the
    paper), but the pilot-preparation path in this Phase 2 stage must never
    request it — see docs/PHASE2_PLAN.md, "Regla de oro".
    """
    df = load_beijing_raw(csv_path)

    raw = pd.DataFrame(
        {
            "pm25": df["pm2.5"].astype(float),
            "dewp": df["DEWP"].astype(float),
            "temp": df["TEMP"].astype(float),
            "pres": df["PRES"].astype(float),
            "iws": df["Iws"].astype(float),
            "is_": df["Is"].astype(float),
            "ir": df["Ir"].astype(float),
        },
        index=df.index,
    )
    cbwd = df["cbwd"].astype(str)

    train_pos = _position_range(raw.index, BEIJING_TRAIN_START, BEIJING_TRAIN_END)
    val_pos = _position_range(raw.index, BEIJING_VAL_START, BEIJING_VAL_END)
    test_pos = _position_range(raw.index, BEIJING_TEST_START, BEIJING_TEST_END)
    if train_pos[1] != val_pos[0] or val_pos[1] != test_pos[0]:
        raise AssertionError("train/validation/test position ranges are not contiguous — check the date boundaries")

    # --- one-hot(cbwd), fit on TRAIN only; handle_unknown="ignore" means any
    # category never seen in train (none, in practice — SE/NW/cv/NE all
    # appear in train) would map to an all-zero row rather than raising. ---
    cbwd_train = cbwd.iloc[train_pos[0]:train_pos[1]]
    cbwd_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cbwd_encoder.fit(cbwd_train.to_numpy().reshape(-1, 1))
    cbwd_categories = [str(c) for c in cbwd_encoder.categories_[0]]
    onehot_cols = [f"cbwd_{c}" for c in cbwd_categories]
    onehot_all = pd.DataFrame(
        cbwd_encoder.transform(cbwd.to_numpy().reshape(-1, 1)),
        columns=onehot_cols, index=raw.index,
    )

    feature_names = CONTINUOUS_COLS + onehot_cols  # fixed column order, documented

    # --- MinMax scalers, fit on TRAIN only (raw units, before windowing;
    # scaling is a per-column affine map, so it commutes with the later
    # linear interpolation of short gaps — filling with scaled interpolated
    # values equals scaling the raw interpolated values). ---
    train_raw_slice = raw.iloc[train_pos[0]:train_pos[1]]
    pm25_scaler = MinMaxScaler().fit(train_raw_slice[["pm25"]].to_numpy())
    other_scaler = MinMaxScaler().fit(train_raw_slice[OTHER_CONTINUOUS_COLS].to_numpy())

    scaled = pd.DataFrame(index=raw.index)
    scaled["pm25"] = pm25_scaler.transform(raw[["pm25"]].to_numpy()).ravel()
    scaled[OTHER_CONTINUOUS_COLS] = other_scaler.transform(raw[OTHER_CONTINUOUS_COLS].to_numpy())
    for c in onehot_cols:
        scaled[c] = onehot_all[c].to_numpy()  # already 0/1, never rescaled
    scaled = scaled[feature_names]

    cuts = {"train": train_pos, "validation": val_pos}
    if include_test:
        cuts["test"] = test_pos

    splits: Dict[str, PreparedSplit] = {}
    for name, (start, end) in cuts.items():
        X_win, y_win, t_win = build_windows(scaled, start, end, ventana=ventana, limite=limite, target_col="pm25")

        y_original = pm25_scaler.inverse_transform(y_win.reshape(-1, 1)).ravel()

        pm25_channel = feature_names.index("pm25")
        pm25_history_scaled = X_win[:, :, pm25_channel]  # (N, ventana), scaled
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

    scalers = BeijingScalers(
        pm25_scaler=pm25_scaler, other_scaler=other_scaler, cbwd_encoder=cbwd_encoder,
        other_cols=OTHER_CONTINUOUS_COLS, cbwd_categories=cbwd_categories,
    )

    return BeijingPreparedData(
        train=splits["train"],
        validation=splits["validation"],
        test=splits.get("test"),
        scalers=scalers,
        feature_names=feature_names,
    )


def subsample_chronological(split: PreparedSplit, max_windows: int) -> PreparedSplit:
    """First `max_windows` windows of `split`, in chronological order (NOT
    a random subsample) — used only to keep the R1 pilot short; never used
    for the full R1 training run."""
    if split.n_windows <= max_windows:
        return split
    return PreparedSplit(
        encoder_input=split.encoder_input[:max_windows],
        decoder_input=split.decoder_input[:max_windows],
        y_scaled=split.y_scaled[:max_windows],
        y_original=split.y_original[:max_windows],
        target_timestamps=split.target_timestamps[:max_windows],
    )


def build_preprocessing_audit(data: BeijingPreparedData, raw_df: pd.DataFrame) -> Dict:
    """Pre-training audit payload for
    outputs/metrics/beijing_preprocessing_audit.json. Contains only
    preprocessing/preparation facts — deliberately no predictive metric of
    any kind (test 2014 is never scored anywhere in this project yet)."""
    pm25_na = raw_df["pm2.5"].isna().to_numpy()
    blocks = missing_blocks(pm25_na)

    train_hours = len(pd.date_range(BEIJING_TRAIN_START, BEIJING_TRAIN_END, freq="h"))
    max_possible_train_windows = train_hours - 24
    train_discard_pct = 100.0 * (1 - data.train.n_windows / max_possible_train_windows)

    return {
        "raw_rows": int(len(raw_df)),
        "date_min": str(raw_df.index.min()),
        "date_max": str(raw_df.index.max()),
        "missing_per_column": {
            c: int(raw_df[c].isna().sum())
            for c in ["pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
        },
        "pm25_missing_blocks_total": int(len(blocks)),
        "pm25_missing_blocks_gt_6h": int((blocks > 6).sum()) if len(blocks) else 0,
        "pm25_missing_blocks_gt_24h": int((blocks > 24).sum()) if len(blocks) else 0,
        "n_windows": {
            "train": data.train.n_windows,
            "validation": data.validation.n_windows,
            "test_prepared_not_evaluated": data.test.n_windows if data.test is not None else None,
        },
        "train_max_possible_windows": max_possible_train_windows,
        "train_discard_pct": round(train_discard_pct, 3),
        "n_channels": len(data.feature_names),
        "feature_names": data.feature_names,
        "pm25_train_min_max_ug_m3": [
            float(data.scalers.pm25_scaler.data_min_[0]), float(data.scalers.pm25_scaler.data_max_[0]),
        ],
        "other_continuous_train_min_max": {
            col: [float(mn), float(mx)]
            for col, mn, mx in zip(
                data.scalers.other_cols, data.scalers.other_scaler.data_min_, data.scalers.other_scaler.data_max_
            )
        },
        "cbwd_categories_learned_from_train": data.scalers.cbwd_categories,
        "note": "No predictive/test-set result of any kind is included here — preprocessing facts only.",
    }


def persistence_prediction(split: PreparedSplit, feature_names: List[str]) -> np.ndarray:
    """Naive baseline: predicted PM2.5(t+1) = observed PM2.5(t), i.e. the
    LAST historical value already present in each window's own encoder
    input (scaled) — not a second computation, just reading the value
    that's already there. Returns (N, 1) in the same scaled units as
    `split.y_scaled`; use `BeijingScalers.inverse_transform_pm25` to get
    ug/m3 for reporting. This is a sanity check for Beijing R1 specifically
    — NOT the Primera Entrega's selected baseline (that was Ridge, on
    Fontibon; see docs/PHASE1_AUDIT.md) and not meant to replace it."""
    pm25_channel = feature_names.index("pm25")
    last_observed = split.encoder_input[:, -1, pm25_channel]
    return last_observed.reshape(-1, 1)


# ============================================================================
# R2: transfer-source preprocessing. NOT the R1 pipeline reused with fewer
# columns — a deliberately separate function, because R1's own column list
# (CONTINUOUS_COLS above) uses a different name ("dewp") and a different
# column ORDER (dew point before temperature) than what Fontibon expects
# (TRANSFER_FEATURE_NAMES = ["pm25", "temp", "dew"]). Reusing R1's ordering
# here would have silently fed the wrong physical variable into the wrong
# input channel once transferred. See
# docs/IMPLEMENTATION_ASSUMPTIONS.md, "R2 — transfer source pretraining".
# ============================================================================
@dataclass
class BeijingR2Scalers:
    """Fit exclusively on Beijing R2 TRAIN — a separate scaler from R1's
    and from Fontibon's own (see module docstring: transfer compatibility
    is semantic/dimensional, not a shared physical scaler)."""
    pm25_scaler: MinMaxScaler
    other_scaler: MinMaxScaler  # fit on [temp, dew], in that order
    other_cols: List[str]

    def inverse_transform_pm25(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        return self.pm25_scaler.inverse_transform(y_scaled).ravel()


@dataclass
class BeijingR2PreparedData:
    train: PreparedSplit
    validation: PreparedSplit
    test: Optional[PreparedSplit]
    scalers: BeijingR2Scalers
    feature_names: List[str]


def prepare_beijing_r2(
    csv_path: Path,
    ventana: int = 24,
    limite: int = 6,
    decoder_label_len: int = 12,
    prediction_length: int = 1,
    include_test: bool = False,
) -> BeijingR2PreparedData:
    """R2: transfer-source pretraining pipeline. R1 already answered the
    paper-replication requirement with the 8 conceptual variables and is
    now closed (see docs/PHASE2_PLAN.md). R2's purpose is different:
    produce a Beijing-pretrained STNModel whose 3-channel input
    (TRANSFER_FEATURE_NAMES: pm25, temp, dew) and encoder-projection shape
    are compatible with the Fontibon target model that will later receive
    transfer (Config B/C/D — not implemented yet).

    Uses the SAME development split as R1 (train 2010-01-01..2013-08-31,
    val 2013-09-01..2013-12-31). Beijing test 2014 already served its
    purpose in R1's final evaluation and is not built here by default
    (include_test=False; do not pass True in this round — see
    docs/PHASE2_PLAN.md).
    """
    df = load_beijing_raw(csv_path)

    raw = pd.DataFrame(
        {
            "pm25": df["pm2.5"].astype(float),
            "temp": df["TEMP"].astype(float),
            "dew": df["DEWP"].astype(float),
        },
        index=df.index,
    )
    raw = raw[TRANSFER_FEATURE_NAMES]  # enforce the canonical order explicitly, not just by construction order above
    assert list(raw.columns) == TRANSFER_FEATURE_NAMES

    train_pos = _position_range(raw.index, BEIJING_TRAIN_START, BEIJING_TRAIN_END)
    val_pos = _position_range(raw.index, BEIJING_VAL_START, BEIJING_VAL_END)
    test_pos = _position_range(raw.index, BEIJING_TEST_START, BEIJING_TEST_END)
    if train_pos[1] != val_pos[0] or val_pos[1] != test_pos[0]:
        raise AssertionError("train/validation/test position ranges are not contiguous — check the date boundaries")

    train_slice = raw.iloc[train_pos[0]:train_pos[1]]
    pm25_scaler = MinMaxScaler().fit(train_slice[["pm25"]].to_numpy())
    other_cols = ["temp", "dew"]
    other_scaler = MinMaxScaler().fit(train_slice[other_cols].to_numpy())

    scaled = pd.DataFrame(index=raw.index)
    scaled["pm25"] = pm25_scaler.transform(raw[["pm25"]].to_numpy()).ravel()
    scaled[other_cols] = other_scaler.transform(raw[other_cols].to_numpy())
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

    scalers = BeijingR2Scalers(pm25_scaler=pm25_scaler, other_scaler=other_scaler, other_cols=other_cols)

    return BeijingR2PreparedData(
        train=splits["train"],
        validation=splits["validation"],
        test=splits.get("test"),
        scalers=scalers,
        feature_names=list(TRANSFER_FEATURE_NAMES),
    )


def build_r2_preprocessing_audit(data: BeijingR2PreparedData, compatibility_report: Optional[Dict] = None) -> Dict:
    """Pre-training audit payload for outputs/metrics/R2_preprocessing_audit.json.
    Deliberately no predictive/test-set result of any kind."""
    return {
        "feature_names": data.feature_names,
        "feature_order_source_of_truth": "src.data.transfer_features.TRANSFER_FEATURE_NAMES",
        "n_windows": {
            "train": data.train.n_windows,
            "validation": data.validation.n_windows,
            "test": data.test.n_windows if data.test is not None else None,
        },
        "pm25_train_min_max_ug_m3": [
            float(data.scalers.pm25_scaler.data_min_[0]), float(data.scalers.pm25_scaler.data_max_[0]),
        ],
        "temp_train_min_max": [
            float(data.scalers.other_scaler.data_min_[0]), float(data.scalers.other_scaler.data_max_[0]),
        ],
        "dew_train_min_max": [
            float(data.scalers.other_scaler.data_min_[1]), float(data.scalers.other_scaler.data_max_[1]),
        ],
        "encoder_input_shape": list(data.train.encoder_input.shape[1:]),  # (ventana, n_features)
        "decoder_input_shape": list(data.train.decoder_input.shape[1:]),  # (label_len+pred_len, 1)
        "compatibility_report": compatibility_report,
        "beijing_test_2014_used": data.test is not None,
        "fontibon_test_used": False,
        "note": "No predictive/validation/test metric of any kind is included here — preprocessing and compatibility facts only.",
    }
