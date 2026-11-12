"""Tests for the Beijing R2 (transfer-source) pipeline: src/data/beijing.py
::prepare_beijing_r2, src/data/transfer_features.py, and
src/models/transfer_compat.py::compare_transfer_weight_shapes.

Beijing test 2014 is NEVER built here (include_test defaults False and no
test in this file passes True) — R1 already used it for its own purpose.
Fontibon is never trained, validated, or tested here; its pipeline is only
imported to check feature-name/order agreement (structural check only)."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.data.beijing import prepare_beijing_r2
from src.data.fontibon import load_fontibon_features
from src.data.transfer_features import TRANSFER_FEATURE_NAMES
from src.models.stn import STNModel
from src.models.transfer_compat import compare_transfer_weight_shapes
from src.utils.paths import BEIJING_CSV, FONTIBON_CSV

CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1, non_selected_fill="zero",
)


@pytest.fixture(scope="module")
def r2_data():
    return prepare_beijing_r2(BEIJING_CSV, include_test=False)


# 1-4. FEATURE NAMES / ORDER, AND AGREEMENT WITH FONTIBON --------------------------

def test_exactly_three_features(r2_data):
    assert len(r2_data.feature_names) == 3
    assert r2_data.train.encoder_input.shape[-1] == 3


def test_feature_names_correct(r2_data):
    assert r2_data.feature_names == ["pm25", "temp", "dew"]


def test_feature_order_matches_transfer_source_of_truth(r2_data):
    assert r2_data.feature_names == TRANSFER_FEATURE_NAMES


@pytest.mark.skipif(
    not FONTIBON_CSV.exists(),
    reason="Raw_Consolidated_ECA_Complete_2022.csv not available locally (institutional "
           "dataset, not redistributed) -- see test_fontibon_feature_order_contract_synthetic "
           "below for the CSV-independent version of this same check.",
)
def test_beijing_r2_order_matches_fontibon_order():
    """The actual, audited check the review demanded: build Fontibon's real
    feature frame and confirm its column order is IDENTICAL to Beijing
    R2's — not assumed, verified against the live Fontibon pipeline. Skipped
    (not failed) when the real CSV isn't present, e.g. in the delivery
    package, which doesn't redistribute it."""
    fontibon_columns = list(load_fontibon_features(FONTIBON_CSV).columns)
    assert fontibon_columns == TRANSFER_FEATURE_NAMES


def test_fontibon_feature_order_contract_synthetic(tmp_path):
    """Same contract as the test above (load_fontibon_features' output
    columns must equal TRANSFER_FEATURE_NAMES), but exercised against a
    tiny SYNTHETIC CSV with the real schema -- runs unconditionally, with
    no dependency on the institutional Fontibon dataset. Calls the exact
    same production function (src.data.fontibon.load_fontibon_features),
    not a reimplementation, so it still catches a real regression in the
    column-order contract."""
    synthetic = pd.DataFrame({
        "STATION": ["FONTIBON", "FONTIBON", "MOVIL FONTIBON", "FONTIBON"],
        "DATE": ["01/01/2022", "01/01/2022", "01/01/2022", "01/01/2022"],
        "HOUR": ["00:00", "01:00", "00:00", "02:00"],
        "PM2.5_µg/m3": [10.5, 12.0, 999.0, 11.2],
        "Temperature_°C": [18.0, 17.5, 20.0, 17.8],
        "RH_%": [60.0, 62.0, 50.0, 61.0],
    })
    csv_path = tmp_path / "synthetic_fontibon.csv"
    synthetic.to_csv(csv_path, index=False, encoding="utf-8")

    result = load_fontibon_features(csv_path)
    assert list(result.columns) == TRANSFER_FEATURE_NAMES
    assert len(result) == 3  # the MOVIL FONTIBON row must be excluded, per src/data/fontibon.py


# 5-6. SCALERS ---------------------------------------------------------------------

def test_pm25_scaler_fit_only_on_beijing_r2_train(r2_data):
    # cross-check against R1's own audited Beijing train min/max for pm25 —
    # same variable, same train period, must match regardless of which
    # pipeline (R1 or R2) computed it.
    np.testing.assert_allclose(r2_data.scalers.pm25_scaler.data_min_[0], 0.0)
    np.testing.assert_allclose(r2_data.scalers.pm25_scaler.data_max_[0], 994.0)


def test_pm25_scaler_is_independent_and_invertible(r2_data):
    recovered = r2_data.scalers.inverse_transform_pm25(r2_data.train.y_scaled)
    np.testing.assert_allclose(recovered.reshape(-1, 1), r2_data.train.y_original, atol=1e-6)


# 7-9. VENTANAS ---------------------------------------------------------------------

def test_train_validation_window_counts_positive_and_disjoint_in_time(r2_data):
    assert r2_data.train.n_windows > 0
    assert r2_data.validation.n_windows > 0
    train_ts = set(r2_data.train.target_timestamps)
    val_ts = set(r2_data.validation.target_timestamps)
    assert train_ts.isdisjoint(val_ts)
    assert r2_data.train.target_timestamps.max() < r2_data.validation.target_timestamps.min()


def test_no_window_crosses_partitions(r2_data):
    import pandas as pd
    from src.data.beijing import BEIJING_TRAIN_END, BEIJING_VAL_START, BEIJING_VAL_END
    assert r2_data.train.target_timestamps.max() <= pd.Timestamp(BEIJING_TRAIN_END)
    assert r2_data.validation.target_timestamps.min() >= pd.Timestamp(BEIJING_VAL_START)
    assert r2_data.validation.target_timestamps.max() <= pd.Timestamp(BEIJING_VAL_END)


def test_target_never_missing(r2_data):
    assert not np.isnan(r2_data.train.y_scaled).any()
    assert not np.isnan(r2_data.validation.y_scaled).any()


# 10-12. DECODER ---------------------------------------------------------------------

def test_decoder_shape(r2_data):
    N = r2_data.train.n_windows
    assert r2_data.train.decoder_input.shape == (N, 13, 1)


def test_decoder_history_matches_encoder_pm25_channel(r2_data):
    pm25_channel = r2_data.feature_names.index("pm25")
    encoder_pm25_last12 = r2_data.train.encoder_input[:5, -12:, pm25_channel]
    decoder_label = r2_data.train.decoder_input[:5, :12, 0]
    np.testing.assert_allclose(encoder_pm25_last12, decoder_label, atol=1e-6)


def test_decoder_placeholder_is_zero(r2_data):
    last = r2_data.train.decoder_input[:, -1, :]
    np.testing.assert_array_equal(last, np.zeros_like(last))


# 13. FINITE ---------------------------------------------------------------------------

def test_all_arrays_finite(r2_data):
    for split in (r2_data.train, r2_data.validation):
        assert np.isfinite(split.encoder_input).all()
        assert np.isfinite(split.decoder_input).all()
        assert np.isfinite(split.y_scaled).all()
        assert np.isfinite(split.y_original).all()


# 14. NO BEIJING TEST ------------------------------------------------------------------

def test_beijing_test_not_built_by_default(r2_data):
    assert r2_data.test is None


# 15-16. SOURCE/TARGET WEIGHT SHAPE COMPATIBILITY ------------------------------------

def test_source_target_encoder_decoder_weight_shapes_compatible():
    source_model = STNModel(**CANDIDATE_V1, seed=42)
    target_model = STNModel(**CANDIDATE_V1, seed=999)  # different seed/init, same architecture

    dummy_enc = tf.zeros((1, 24, 3))
    dummy_dec = tf.zeros((1, 13, 1))
    source_model((dummy_enc, dummy_dec), training=False)
    target_model((dummy_enc, dummy_dec), training=False)

    report = compare_transfer_weight_shapes(source_model, target_model)
    assert report["compatible"] is True
    assert report["encoder_compatible"] is True
    assert report["decoder_compatible"] is True
    assert report["encoder_mismatches"] == []
    assert report["decoder_mismatches"] == []


def test_weight_shape_comparison_detects_real_mismatch():
    """Sanity check the comparator itself: a target with a DIFFERENT
    n_features (11, like R1) must be reported incompatible."""
    source_model = STNModel(**CANDIDATE_V1, seed=42)
    target_model = STNModel(**CANDIDATE_V1, seed=42)

    source_model((tf.zeros((1, 24, 3)), tf.zeros((1, 13, 1))), training=False)
    target_model((tf.zeros((1, 24, 11)), tf.zeros((1, 13, 1))), training=False)  # 11 channels, like R1

    report = compare_transfer_weight_shapes(source_model, target_model)
    assert report["compatible"] is False
    assert report["encoder_compatible"] is False
    assert len(report["encoder_mismatches"]) > 0


# 17-18. ARCHITECTURE CONFIGURABLE / STRUCTURAL NAMES ACCESSIBLE ---------------------

def test_architecture_configurable_for_future_freeze():
    model_reduced = STNModel(**CANDIDATE_V1, seed=1)
    model_paper_depth = STNModel(**{**CANDIDATE_V1, "n_encoder_units": 3, "n_decoder_layers": 2}, seed=1)
    assert len(model_reduced.encoder.units) == 2
    assert len(model_paper_depth.encoder.units) == 3


def test_model_encoder_decoder_output_head_accessible():
    model = STNModel(**CANDIDATE_V1, seed=42)
    model((tf.zeros((1, 24, 3)), tf.zeros((1, 13, 1))), training=False)
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.output_head is not None
