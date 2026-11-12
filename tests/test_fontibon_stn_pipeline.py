"""Tests for the Fontibon STN pipeline (src/data/fontibon.py::prepare_fontibon_stn),
the TARGET side of the transfer experiments (Config A/B/C/D — not run
here). Mirrors tests/test_beijing_r2_pipeline.py's structure for the
analogous Beijing R2 (SOURCE) pipeline.

Fontibon test is built in exactly ONE test below
(test_window_counts_match_ridge_baseline, include_test=True), purely to
verify the pipeline reproduces the already-known/approved window count —
it is never evaluated against any model anywhere in this file or in
src/training/transfer.py. No other test in this file requests
include_test=True."""
import numpy as np
import pytest

from src.data.beijing import prepare_beijing_r2
from src.data.fontibon import prepare_fontibon_stn
from src.data.transfer_features import TRANSFER_FEATURE_NAMES
from src.utils.paths import BEIJING_CSV, FONTIBON_CSV


FONTIBON_SKIP_REASON = "Skipped: Fontibón institutional dataset is not distributed with the reproducible package."
requires_fontibon_csv = pytest.mark.skipif(not FONTIBON_CSV.exists(), reason=FONTIBON_SKIP_REASON)


@pytest.fixture(scope="module")
def fontibon_stn_data():
    if not FONTIBON_CSV.exists():
        pytest.skip(FONTIBON_SKIP_REASON)
    return prepare_fontibon_stn(FONTIBON_CSV, include_test=False)


# 1-2. FEATURE NAMES / ORDER -----------------------------------------------------

def test_exactly_three_features(fontibon_stn_data):
    assert len(fontibon_stn_data.feature_names) == 3
    assert fontibon_stn_data.train.encoder_input.shape[-1] == 3


def test_feature_names_match_transfer_source_of_truth(fontibon_stn_data):
    assert fontibon_stn_data.feature_names == TRANSFER_FEATURE_NAMES == ["pm25", "temp", "dew"]


# 3. WINDOW COUNTS — the one place include_test=True is used in this file ------

@requires_fontibon_csv
def test_window_counts_match_ridge_baseline():
    """Reported in Primer Entrega DL.pdf / outputs/metrics/baseline_fontibon_ridge.json
    and already locked in by tests/test_partitions.py at the raw-partition
    level. This test checks the SAME counts survive prepare_fontibon_stn's
    extra steps (scaling, decoder construction) unchanged — building the
    test split here is a one-time pipeline-correctness check, not an
    experiment; it is never evaluated against any model."""
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    assert data.train.n_windows == 10802
    assert data.validation.n_windows == 2564
    assert data.test.n_windows == 2574


# 4-5. SCALERS --------------------------------------------------------------------

def test_scaler_fit_only_on_train(fontibon_stn_data):
    # Cross-check against the already-reported train target mean (19.10
    # ug/m3, tests/test_scaler_leakage.py) via round-trip inverse transform.
    recovered_train_mean = fontibon_stn_data.scalers.inverse_transform_pm25(
        fontibon_stn_data.train.y_scaled
    ).mean()
    assert abs(recovered_train_mean - 19.10) < 0.05


def test_scaler_is_independent_and_invertible(fontibon_stn_data):
    recovered = fontibon_stn_data.scalers.inverse_transform_pm25(fontibon_stn_data.train.y_scaled)
    np.testing.assert_allclose(recovered.reshape(-1, 1), fontibon_stn_data.train.y_original, atol=1e-6)


def test_scaler_is_fontibons_own_not_beijings(fontibon_stn_data):
    """Guards against the exact mistake Task 2 explicitly forbids: silently
    reusing Beijing's R2 scaler instead of fitting one on Fontibon. Beijing
    R2's pm25 train range is [0.0, 994.0] (outputs/metrics/R2_preprocessing_audit.json);
    Fontibon's is drastically different (much lower peak concentrations) —
    if these ever matched, it would mean Fontibon's scaler was never
    actually fit on Fontibon data."""
    beijing_r2_data = prepare_beijing_r2(BEIJING_CSV, include_test=False)
    fontibon_pm25_min = fontibon_stn_data.scalers.pm25_scaler.data_min_[0]
    fontibon_pm25_max = fontibon_stn_data.scalers.pm25_scaler.data_max_[0]
    beijing_pm25_min = beijing_r2_data.scalers.pm25_scaler.data_min_[0]
    beijing_pm25_max = beijing_r2_data.scalers.pm25_scaler.data_max_[0]
    assert (fontibon_pm25_min, fontibon_pm25_max) != (beijing_pm25_min, beijing_pm25_max)
    assert fontibon_pm25_max < beijing_pm25_max  # Fontibon's levels are much lower than Beijing's


# 6-7. NO TEMPORAL LEAKAGE ---------------------------------------------------------

def test_train_validation_chronologically_ordered_no_overlap(fontibon_stn_data):
    train_ts = set(fontibon_stn_data.train.target_timestamps)
    val_ts = set(fontibon_stn_data.validation.target_timestamps)
    assert train_ts.isdisjoint(val_ts)
    assert fontibon_stn_data.train.target_timestamps.max() < fontibon_stn_data.validation.target_timestamps.min()


def test_target_never_missing(fontibon_stn_data):
    assert not np.isnan(fontibon_stn_data.train.y_scaled).any()
    assert not np.isnan(fontibon_stn_data.validation.y_scaled).any()


# 8-10. CAUSAL DECODER CONSTRUCTION -------------------------------------------------

def test_decoder_shape(fontibon_stn_data):
    N = fontibon_stn_data.train.n_windows
    assert fontibon_stn_data.train.decoder_input.shape == (N, 13, 1)


def test_decoder_history_matches_encoder_pm25_channel(fontibon_stn_data):
    pm25_channel = fontibon_stn_data.feature_names.index("pm25")
    encoder_pm25_last12 = fontibon_stn_data.train.encoder_input[:5, -12:, pm25_channel]
    decoder_label = fontibon_stn_data.train.decoder_input[:5, :12, 0]
    np.testing.assert_allclose(encoder_pm25_last12, decoder_label, atol=1e-6)


def test_decoder_placeholder_is_zero(fontibon_stn_data):
    last = fontibon_stn_data.train.decoder_input[:, -1, :]
    np.testing.assert_array_equal(last, np.zeros_like(last))


# 11. FINITE ------------------------------------------------------------------------

def test_all_arrays_finite(fontibon_stn_data):
    for split in (fontibon_stn_data.train, fontibon_stn_data.validation):
        assert np.isfinite(split.encoder_input).all()
        assert np.isfinite(split.decoder_input).all()
        assert np.isfinite(split.y_scaled).all()
        assert np.isfinite(split.y_original).all()


# 12. NO FONTIBON TEST BY DEFAULT ----------------------------------------------------

def test_fontibon_test_not_built_by_default(fontibon_stn_data):
    assert fontibon_stn_data.test is None
