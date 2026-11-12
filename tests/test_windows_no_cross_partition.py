"""Windows must never cross a train/val/test boundary, and must never use
information at or after the hour they are trying to predict."""
import numpy as np
import pandas as pd

from src.data.fontibon import build_windows


def test_real_partitions_windows_stay_within_bounds(fontibon_features, fontibon_partitions):
    from src.data.fontibon import chronological_cuts

    cortes = chronological_cuts(len(fontibon_features))
    for name, (inicio, fin) in cortes.items():
        _, _, fechas = fontibon_partitions[name]
        lower = fontibon_features.index[inicio]
        upper = fontibon_features.index[fin - 1]
        assert (fechas >= lower).all()
        assert (fechas <= upper).all()


def test_synthetic_window_rejects_interpolation_touching_target_hour():
    """Construct a small frame where a short gap's right anchor lands
    exactly on a later target hour; the window ending there must be
    rejected, and the next one (whose history no longer touches that
    anchor) must be accepted."""
    idx = pd.date_range("2022-01-01", periods=10, freq="h")
    pm25 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # gap at positions 1..3, anchored at 0 (left) and 4 (right)
    x = [1, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9, 10]
    df = pd.DataFrame({"pm25": pm25, "x": x}, index=idx)

    Xs, ys, fechas = build_windows(df, 0, len(df), ventana=3, limite=6)

    # k=4: history is positions [1,2,3], all interpolated using right
    # anchor j=4 == k -> must be rejected (leakage into the target hour).
    assert idx[4] not in fechas

    # k=5: history is positions [2,3,4]; anchor j=4 < k=5 -> must be kept.
    assert idx[5] in fechas


def test_synthetic_window_drops_when_target_missing():
    idx = pd.date_range("2022-01-01", periods=8, freq="h")
    pm25 = [1, 2, 3, np.nan, 5, 6, 7, 8]
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    df = pd.DataFrame({"pm25": pm25, "x": x}, index=idx)

    Xs, ys, fechas = build_windows(df, 0, len(df), ventana=3, limite=6)

    assert idx[3] not in fechas  # target itself missing -> never predicted


def test_synthetic_window_drops_when_history_has_long_gap():
    idx = pd.date_range("2022-01-01", periods=12, freq="h")
    pm25 = list(range(1, 13))
    x = [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10, 11, 12]  # 7-hour gap > limite=6
    df = pd.DataFrame({"pm25": pm25, "x": x}, index=idx)

    Xs, ys, fechas = build_windows(df, 0, len(df), ventana=3, limite=6)

    # Any window whose history overlaps positions 2..8 must be dropped
    # because that gap (length 7) exceeds the interpolation limit and is
    # left as NaN.
    for k in range(3, 12):
        history_positions = range(k - 3, k)
        touches_gap = any(2 <= p <= 8 for p in history_positions)
        if touches_gap:
            assert idx[k] not in fechas
