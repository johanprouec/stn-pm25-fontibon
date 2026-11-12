"""Shape contracts for features, windows and metrics."""
import numpy as np

from src.data.fontibon import flatten
from src.evaluation.metrics import high_episode_metrics, regression_metrics


def test_feature_frame_shape(fontibon_features):
    assert fontibon_features.shape[1] == 3
    assert list(fontibon_features.columns) == ["pm25", "temp", "dew"]


def test_window_tensor_shapes(fontibon_partitions):
    for name, (X, y, fechas) in fontibon_partitions.items():
        assert X.ndim == 3
        assert X.shape[1:] == (24, 3)
        assert y.ndim == 1
        assert len(fechas) == X.shape[0] == y.shape[0]


def test_flatten_shape(fontibon_partitions):
    X, _, _ = fontibon_partitions["train"]
    flat = flatten(X)
    assert flat.shape == (X.shape[0], 24 * 3)


def test_regression_metrics_shape_and_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = regression_metrics(y, pred)
    assert set(m.keys()) == {"RMSE", "MAE", "R2"}
    assert all(isinstance(v, float) for v in m.values())


def test_high_episode_metrics_keys_and_empty_case():
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    m = high_episode_metrics(y, pred, threshold=100.0)
    assert m["n_high"] == 0
    assert np.isnan(m["RMSE_high"])

    m2 = high_episode_metrics(y, pred, threshold=15.0)
    assert m2["n_high"] == 2
    assert set(m2.keys()) == {"RMSE_high", "MAE_high", "Bias_high", "n_high"}


def test_high_episode_bias_sign_convention():
    # prediction below real -> underestimation -> negative bias
    y = np.array([50.0, 50.0])
    pred = np.array([40.0, 40.0])
    m = high_episode_metrics(y, pred, threshold=40.0)
    assert m["Bias_high"] < 0
