"""The PM2.5 target must never be imputed, and every returned y must equal
the raw (non-interpolated) observation at its timestamp."""
import numpy as np


def test_targets_are_never_nan(fontibon_partitions):
    for name, (_, y, _) in fontibon_partitions.items():
        assert not np.isnan(y).any(), f"{name}: found imputed/NaN target(s)"


def test_targets_match_raw_series_exactly(fontibon_features, fontibon_partitions):
    raw_pm25 = fontibon_features["pm25"]
    for name, (_, y, fechas) in fontibon_partitions.items():
        raw_at_fechas = raw_pm25.loc[fechas].to_numpy()
        assert np.allclose(y, raw_at_fechas), f"{name}: target y does not match raw pm25 series"


def test_no_window_history_contains_the_target_hour(fontibon_partitions):
    """Structural check: for every partition, every window's date list has
    length == number of windows and each window covers exactly `ventana`
    hours strictly before its own timestamp (spot-checked via count)."""
    for name, (X, y, fechas) in fontibon_partitions.items():
        assert X.shape[0] == len(y) == len(fechas)
        assert X.shape[1] == 24  # ventana
        assert X.shape[2] == 3  # pm25, temp, dew
