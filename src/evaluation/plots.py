"""Training-dynamics and prediction figures, regenerable from saved files
(a JSON history + saved prediction arrays), never dependent on objects
still alive in memory — matches the project's reproducibility rule."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _epochs(history: List[Dict]) -> List[int]:
    return [row["epoch"] for row in history]


def plot_loss_train_vs_validation(history: List[Dict], save_path: Path) -> None:
    epochs = _epochs(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_loss"] for r in history], label="train_loss (deterministic)")
    ax.plot(epochs, [r["val_loss"] for r in history], label="val_loss")
    best = [r["epoch"] for r in history if r.get("best_so_far")]
    if best:
        ax.axvline(best[-1], color="gray", ls="--", lw=1, label=f"best epoch ({best[-1]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (escala normalizada)")
    ax.set_title("Pérdida — train vs validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rmse_scaled_train_vs_validation(history: List[Dict], save_path: Path) -> None:
    epochs = _epochs(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_rmse_scaled"] for r in history], label="train_rmse_scaled")
    ax.plot(epochs, [r["val_rmse_scaled"] for r in history], label="val_rmse_scaled")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (escala normalizada)")
    ax.set_title("RMSE escalado — train vs validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gradient_norm_mean(history: List[Dict], save_path: Path) -> None:
    epochs = _epochs(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["gradient_global_norm_mean"] for r in history], color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norma global del gradiente (media por batch)")
    ax.set_title("Norma de gradiente — media por época")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gradient_norm_max(history: List[Dict], save_path: Path) -> None:
    epochs = _epochs(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["gradient_global_norm_max"] for r in history], color="firebrick")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norma global del gradiente (máxima por batch)")
    ax.set_title("Norma de gradiente — máximo por época")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_learning_rate(history: List[Dict], save_path: Path) -> None:
    epochs = _epochs(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["learning_rate"] for r in history], color="seagreen")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate por época")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_validation_predictions_vs_real(
    timestamps: pd.DatetimeIndex, y_real: np.ndarray, y_pred: np.ndarray, save_path: Path, n_hours: int = 24 * 10
) -> None:
    """First `n_hours` of validation, in chronological order (validation
    windows are already time-ordered) — not the full split, which would be
    illegible at 2,841+ points."""
    order = np.argsort(timestamps.values)
    ts = timestamps[order][:n_hours]
    real = np.asarray(y_real).ravel()[order][:n_hours]
    pred = np.asarray(y_pred).ravel()[order][:n_hours]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts, real, label="Observado", lw=1.3)
    ax.plot(ts, pred, label="STN (predicho)", lw=1.1)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title(f"Validación — primeras {n_hours} horas: observado vs predicho")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_validation_scatter(y_real: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    real = np.asarray(y_real).ravel()
    pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real, pred, s=6, alpha=0.3)
    lim = [0, max(real.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "--", color="gray", lw=1, label="y = x (ideal)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observado (ug/m3)")
    ax.set_ylabel("Predicho (ug/m3)")
    ax.set_title("Validación — observado vs predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_r1_development_figures(
    history: List[Dict],
    val_timestamps: pd.DatetimeIndex,
    val_y_real: np.ndarray,
    val_y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_train_vs_validation(history, out_dir / "loss_train_vs_validation.png")
    plot_rmse_scaled_train_vs_validation(history, out_dir / "rmse_scaled_train_vs_validation.png")
    plot_gradient_norm_mean(history, out_dir / "gradient_norm_mean.png")
    plot_gradient_norm_max(history, out_dir / "gradient_norm_max.png")
    plot_learning_rate(history, out_dir / "learning_rate.png")
    plot_validation_predictions_vs_real(val_timestamps, val_y_real, val_y_pred, out_dir / "validation_predictions_vs_real.png")
    plot_validation_scatter(val_y_real, val_y_pred, out_dir / "validation_scatter_real_vs_pred.png")


# ============================================================================
# R1 test-evaluation figures (Beijing test 2014, one-time final evaluation).
# Thin, correctly-titled variants of the validation plots above — same
# math, reused rather than duplicated logic where the function is already
# split-agnostic.
# ============================================================================
def plot_predictions_vs_real(
    timestamps: pd.DatetimeIndex, y_real: np.ndarray, y_pred: np.ndarray, save_path: Path,
    split_label: str, n_hours: int = 24 * 14,
) -> None:
    """Same as plot_validation_predictions_vs_real but with a caller-given
    split label in the title (e.g. 'Test 2014') and a default 14-day
    window (fixed, reproducible — not hand-picked to "look good")."""
    order = np.argsort(timestamps.values)
    ts = timestamps[order][:n_hours]
    real = np.asarray(y_real).ravel()[order][:n_hours]
    pred = np.asarray(y_pred).ravel()[order][:n_hours]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts, real, label="Observado", lw=1.3)
    ax.plot(ts, pred, label="STN (predicho)", lw=1.1)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title(f"{split_label} — primeras {n_hours // 24} jornadas: observado vs predicho")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter(y_real: np.ndarray, y_pred: np.ndarray, save_path: Path, split_label: str) -> None:
    real = np.asarray(y_real).ravel()
    pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real, pred, s=6, alpha=0.3)
    lim = [0, max(real.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "--", color="gray", lw=1, label="y = x (ideal)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observado (ug/m3)")
    ax.set_ylabel("Predicho (ug/m3)")
    ax.set_title(f"{split_label} — observado vs predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics_across_seeds(
    seed_metrics: Dict[int, Dict[str, float]],
    save_path: Path,
    title: str = "Test 2014 — métricas por semilla (referencia descriptiva, sin selección de ganador)",
) -> None:
    """Bar chart of RMSE/MAE across seeds, with R2 annotated — purely
    descriptive comparison, no seed is highlighted as "the winner"."""
    seeds = sorted(seed_metrics.keys())
    rmses = [seed_metrics[s]["RMSE"] for s in seeds]
    maes = [seed_metrics[s]["MAE"] for s in seeds]
    r2s = [seed_metrics[s]["R2"] for s in seeds]

    x = np.arange(len(seeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, rmses, width, label="RMSE (ug/m3)")
    ax.bar(x + width / 2, maes, width, label="MAE (ug/m3)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}\nR2={r2s[i]:.4f}" for i, s in enumerate(seeds)])
    ax.set_ylabel("ug/m3")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
