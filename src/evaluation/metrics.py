"""Evaluation metrics, in original PM2.5 units (ug/m3).

``regression_metrics`` is a direct port of the notebook's ``metricas``
function. ``high_episode_metrics`` is new for Phase 2 (not present in the
Primera Entrega notebook): RMSE/MAE/Bias restricted to observations at or
above a threshold, to diagnose behavior during elevated-PM2.5 episodes.

Threshold note: the Primera Entrega *report* text states 40 ug/m3, but the
notebook's actual executed cell used 37 ug/m3 (citing Colombian Resolucion
2254 de 2017). This is a documented inconsistency (see
docs/PHASE1_AUDIT.md) — the report's own numbers (118 obs, mean real 49.87,
mean pred 34.91, bias -14.96) were verified to correspond exactly to
threshold=40, not 37. Per explicit instruction, 40 is kept as the primary
threshold going forward for methodological continuity with the report;
callers needing the notebook's as-run value must pass threshold=37
explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

DEFAULT_HIGH_THRESHOLD = 40.0


def regression_metrics(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, R2 in original units. Ported unchanged from the notebook."""
    error = y_real - y_pred
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    r2 = 1 - (np.sum(error ** 2) / np.sum((y_real - y_real.mean()) ** 2))
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


def high_episode_metrics(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> Dict[str, float]:
    """RMSE_high, MAE_high, Bias_high on the subset y_real >= threshold.

    Bias_high = mean(prediction - real): negative means underestimation,
    positive means overestimation (matches the sign convention specified for
    Phase 2, which is the opposite sign of the notebook's own
    "subestimacion = real - pred" diagnostic — do not confuse the two when
    comparing against the Primera Entrega figures).

    Returns n_high=0 and NaN metrics if no observations meet the threshold;
    callers must check n_high before treating this as a selection criterion
    (see docs/PHASE2_PLAN.md on validation having few high episodes).
    """
    mask = y_real >= threshold
    n_high = int(mask.sum())
    if n_high == 0:
        return {"RMSE_high": float("nan"), "MAE_high": float("nan"), "Bias_high": float("nan"), "n_high": 0}

    error = y_pred[mask] - y_real[mask]
    rmse_high = np.sqrt(np.mean(error ** 2))
    mae_high = np.mean(np.abs(error))
    bias_high = np.mean(error)
    return {
        "RMSE_high": float(rmse_high),
        "MAE_high": float(mae_high),
        "Bias_high": float(bias_high),
        "n_high": n_high,
    }


def count_high_episodes(y: np.ndarray, threshold: float = DEFAULT_HIGH_THRESHOLD) -> int:
    return int((y >= threshold).sum())


def evaluate_scaled_predictions(
    y_scaled_true: np.ndarray,
    y_scaled_pred: np.ndarray,
    inverse_transform: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, float]:
    """The only sanctioned path from a model's normalized-scale output to a
    reportable metric: inverse-transform BOTH true and predicted values
    back to ug/m3, then compute RMSE/MAE/R2 in that original unit.
    Normalized-scale RMSE/MSE (e.g. a trainer's `val_loss`) is legitimate
    as an internal selection/monitoring signal — the target's
    transformation is a fixed, linear, per-partition scaler, so minimizing
    normalized MSE preserves the same model ranking as minimizing ug/m3
    RMSE within one split — but must never be reported as the headline
    result. See docs/IMPLEMENTATION_ASSUMPTIONS.md.
    """
    y_true = np.asarray(inverse_transform(y_scaled_true)).ravel()
    y_pred = np.asarray(inverse_transform(y_scaled_pred)).ravel()
    return regression_metrics(y_true, y_pred)


def residual_diagnostics(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Pre-test diagnostic (not meant to be a large analysis on its own —
    see docs/PHASE2_PLAN.md). residual = prediction - real, same sign
    convention as Bias_high in high_episode_metrics: negative = STN
    underestimates, positive = STN overestimates."""
    residual = np.asarray(y_pred) - np.asarray(y_real)
    return {
        "bias_mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "p05": float(np.percentile(residual, 5)),
        "p25": float(np.percentile(residual, 25)),
        "p50": float(np.percentile(residual, 50)),
        "p75": float(np.percentile(residual, 75)),
        "p95": float(np.percentile(residual, 95)),
    }


def target_distribution(y: np.ndarray) -> Dict[str, float]:
    """Descriptive distribution of the target, in original units — used to
    look at Beijing's tail behavior WITHOUT picking a new high-episode
    threshold post-hoc (Fontibon's 40 ug/m3 does not transfer to Beijing's
    very different concentration levels; see
    docs/IMPLEMENTATION_ASSUMPTIONS.md)."""
    y = np.asarray(y)
    return {
        "p50": float(np.percentile(y, 50)),
        "p75": float(np.percentile(y, 75)),
        "p90": float(np.percentile(y, 90)),
        "p95": float(np.percentile(y, 95)),
        "p99": float(np.percentile(y, 99)),
        "max": float(np.max(y)),
    }


def classify_model_vs_baseline(model_rmse: float, baseline_rmse: float, margin: float = 0.05) -> str:
    """Purely descriptive classification (never an automatic "success"
    declaration — see docs/PHASE2_PLAN.md, "Criterio de sanity-check").
    `margin` (default 5%) is the relative-RMSE-improvement band used to
    call two RMSEs "similar" rather than one clearly better/worse; it is a
    reporting convenience, not a statistical test."""
    relative_improvement = (baseline_rmse - model_rmse) / baseline_rmse
    if relative_improvement > margin:
        return "A: el modelo converge y mejora claramente a la referencia"
    elif relative_improvement > -margin:
        return "B: el modelo converge pero queda similar a la referencia"
    else:
        return "C: el modelo no converge o queda claramente peor que la referencia"


def save_predictions_with_residuals(
    timestamps, y_real: np.ndarray, y_pred: np.ndarray, path: Union[str, Path]
) -> None:
    """Row-level predictions (not just aggregated metrics): target_timestamp,
    y_real, y_pred, residual (= y_pred - y_real), in original units.
    Needed for later error analysis / time-series-appropriate interval
    construction — an aggregate RMSE alone cannot support that. Never used
    to define a post-hoc high-episode threshold from validation (see
    docs/PHASE2_PLAN.md, "Episodios altos Beijing")."""
    df = pd.DataFrame({
        "target_timestamp": pd.DatetimeIndex(timestamps),
        "y_real": np.asarray(y_real).ravel(),
        "y_pred": np.asarray(y_pred).ravel(),
    })
    df["residual"] = df["y_pred"] - df["y_real"]
    df.to_csv(path, index=False)
