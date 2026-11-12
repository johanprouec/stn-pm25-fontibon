"""Tests for the Beijing R1 preparation pipeline (src/data/beijing.py).

Beijing test 2014 IS built here (include_test=True) purely to test that
the split machinery is correct — no metric is ever computed against it in
this test file, and the R1 pilot script never requests it. See
docs/PHASE2_PLAN.md, "Regla de oro"."""
import numpy as np
import pandas as pd
import pytest

from src.data.beijing import (
    BEIJING_TEST_END,
    BEIJING_TEST_START,
    BEIJING_TRAIN_END,
    BEIJING_TRAIN_START,
    BEIJING_VAL_END,
    BEIJING_VAL_START,
    load_beijing_raw,
    prepare_beijing_r1,
)
from src.utils.paths import BEIJING_CSV

EXPECTED_FEATURE_NAMES = ["pm25", "dewp", "temp", "pres", "iws", "is_", "ir", "cbwd_NE", "cbwd_NW", "cbwd_SE", "cbwd_cv"]

# Audited via a real run of prepare_beijing_r1 (see docs/PHASE1_AUDIT.md /
# outputs/metrics/beijing_preprocessing_audit.json) — NOT invented. Frozen
# here so a silent regression in the pipeline is caught.
EXPECTED_N_WINDOWS = {"train": 29237, "validation": 2841, "test": 8487}


@pytest.fixture(scope="module")
def beijing_data():
    return prepare_beijing_r1(BEIJING_CSV, include_test=True)


@pytest.fixture(scope="module")
def beijing_raw():
    return load_beijing_raw(BEIJING_CSV)


# 1-4. SPLIT TEMPORAL -----------------------------------------------------------

def test_split_temporal_exact_bounds(beijing_data):
    d = beijing_data
    assert d.train.target_timestamps.min() >= pd.Timestamp(BEIJING_TRAIN_START)
    assert d.train.target_timestamps.max() <= pd.Timestamp(BEIJING_TRAIN_END)
    assert d.validation.target_timestamps.min() >= pd.Timestamp(BEIJING_VAL_START)
    assert d.validation.target_timestamps.max() <= pd.Timestamp(BEIJING_VAL_END)
    assert d.test.target_timestamps.min() >= pd.Timestamp(BEIJING_TEST_START)
    assert d.test.target_timestamps.max() <= pd.Timestamp(BEIJING_TEST_END)


def test_zero_overlap_between_partitions(beijing_data):
    d = beijing_data
    train_ts = set(d.train.target_timestamps)
    val_ts = set(d.validation.target_timestamps)
    test_ts = set(d.test.target_timestamps)
    assert train_ts.isdisjoint(val_ts)
    assert val_ts.isdisjoint(test_ts)
    assert train_ts.isdisjoint(test_ts)


def test_train_ends_before_validation(beijing_data):
    assert beijing_data.train.target_timestamps.max() < beijing_data.validation.target_timestamps.min()


def test_validation_ends_before_test(beijing_data):
    assert beijing_data.validation.target_timestamps.max() < beijing_data.test.target_timestamps.min()


# 5-7. SCALERS / ENCODER AJUSTADOS SOLO CON TRAIN --------------------------------

def test_cbwd_encoder_fit_only_on_train(beijing_data, beijing_raw):
    train_slice = beijing_raw.loc[BEIJING_TRAIN_START:BEIJING_TRAIN_END]
    expected_categories = sorted(train_slice["cbwd"].astype(str).unique())
    assert sorted(beijing_data.scalers.cbwd_categories) == expected_categories


def test_pm25_scaler_fit_only_on_train(beijing_data, beijing_raw):
    train_slice = beijing_raw.loc[BEIJING_TRAIN_START:BEIJING_TRAIN_END]
    expected_min = float(train_slice["pm2.5"].min())
    expected_max = float(train_slice["pm2.5"].max())
    np.testing.assert_allclose(beijing_data.scalers.pm25_scaler.data_min_[0], expected_min)
    np.testing.assert_allclose(beijing_data.scalers.pm25_scaler.data_max_[0], expected_max)


def test_other_scalers_fit_only_on_train(beijing_data, beijing_raw):
    train_slice = beijing_raw.loc[BEIJING_TRAIN_START:BEIJING_TRAIN_END]
    col_map = {"dewp": "DEWP", "temp": "TEMP", "pres": "PRES", "iws": "Iws", "is_": "Is", "ir": "Ir"}
    for i, our_col in enumerate(beijing_data.scalers.other_cols):
        raw_col = col_map[our_col]
        expected_min = float(train_slice[raw_col].min())
        expected_max = float(train_slice[raw_col].max())
        np.testing.assert_allclose(beijing_data.scalers.other_scaler.data_min_[i], expected_min)
        np.testing.assert_allclose(beijing_data.scalers.other_scaler.data_max_[i], expected_max)


# 8-9. ONE-HOT NO ORDINAL ----------------------------------------------------------

def test_one_hot_not_ordinal(beijing_data):
    onehot_cols = [c for c in beijing_data.feature_names if c.startswith("cbwd_")]
    assert len(onehot_cols) == 4  # one binary channel per category, not a single ordinal column
    channel_idx = [beijing_data.feature_names.index(c) for c in onehot_cols]
    values = beijing_data.train.encoder_input[:, :, channel_idx]
    assert set(np.unique(values).tolist()) <= {0.0, 1.0}


def test_four_cbwd_categories_present_in_train():
    assert set(prepare_beijing_r1(BEIJING_CSV).scalers.cbwd_categories) == {"NE", "NW", "SE", "cv"}


# 10. FEATURE NAMES -----------------------------------------------------------------

def test_feature_names_correct(beijing_data):
    assert beijing_data.feature_names == EXPECTED_FEATURE_NAMES


# 11-14. SHAPES -----------------------------------------------------------------------

def test_encoder_shape(beijing_data):
    N = beijing_data.train.n_windows
    assert beijing_data.train.encoder_input.shape == (N, 24, len(EXPECTED_FEATURE_NAMES))


def test_encoder_feature_dim_matches_real_dataset(beijing_data):
    assert beijing_data.train.encoder_input.shape[-1] == 11


def test_decoder_shape(beijing_data):
    N = beijing_data.train.n_windows
    assert beijing_data.train.decoder_input.shape == (N, 13, 1)


def test_target_shape(beijing_data):
    N = beijing_data.train.n_windows
    assert beijing_data.train.y_scaled.shape == (N, 1)
    assert beijing_data.train.y_original.shape == (N, 1)


# 15. TARGET NUNCA MISSING ------------------------------------------------------------

def test_target_never_missing(beijing_data):
    for split in (beijing_data.train, beijing_data.validation, beijing_data.test):
        assert not np.isnan(split.y_scaled).any()
        assert not np.isnan(split.y_original).any()


# 16. NINGUNA VENTANA CRUZA PARTICIONES ------------------------------------------------

def test_no_window_crosses_partition_boundaries(beijing_data):
    d = beijing_data
    assert d.train.target_timestamps.max() <= pd.Timestamp(BEIJING_TRAIN_END)
    assert d.validation.target_timestamps.min() >= pd.Timestamp(BEIJING_VAL_START)
    assert d.validation.target_timestamps.max() <= pd.Timestamp(BEIJING_VAL_END)
    assert d.test.target_timestamps.min() >= pd.Timestamp(BEIJING_TEST_START)


# 17. NO INTERPOLACIÓN FUTURA ---------------------------------------------------------
# The causal-safety guarantee itself (no interpolation may use information
# at or after the target hour) is exhaustively tested against synthetic
# worst cases in tests/test_windows_no_cross_partition.py, on the exact
# same build_windows/interpolate_short_gaps code this module reuses. Here
# we only confirm real Beijing windows come out fully finite, i.e. that
# reuse didn't silently disable that guarantee (e.g. by leaving NaNs).

def test_real_beijing_windows_have_no_leftover_nan(beijing_data):
    assert np.isfinite(beijing_data.train.encoder_input).all()
    assert np.isfinite(beijing_data.validation.encoder_input).all()
    assert np.isfinite(beijing_data.test.encoder_input).all()


# 18. GAPS > 6H PROVOCAN DESCARTE -------------------------------------------------------

def test_gaps_over_6h_cause_window_discard(beijing_data):
    """Some windows must be discarded (missing PM2.5 blocks >6h exist in
    the raw data, per docs/PHASE1_AUDIT.md §6: 20 blocks >24h). Total
    windows must be strictly less than the theoretical maximum."""
    d = beijing_data
    train_hours = len(pd.date_range(BEIJING_TRAIN_START, BEIJING_TRAIN_END, freq="h"))
    max_possible = train_hours - 24
    assert d.train.n_windows < max_possible
    discard_pct = 100 * (1 - d.train.n_windows / max_possible)
    assert 0 < discard_pct < 20  # sanity bound: some but not excessive discard


# 19. ROUNDTRIP -----------------------------------------------------------------------

def test_roundtrip_inverse_transform_recovers_y_original(beijing_data):
    for split in (beijing_data.train, beijing_data.validation, beijing_data.test):
        recovered = beijing_data.scalers.inverse_transform_pm25(split.y_scaled).reshape(-1, 1)
        np.testing.assert_allclose(recovered, split.y_original, atol=1e-6)


# 20-21. DECODER INPUT ------------------------------------------------------------------

def test_decoder_placeholder_is_zero(beijing_data):
    last = beijing_data.train.decoder_input[:, -1, :]
    np.testing.assert_array_equal(last, np.zeros_like(last))


def test_decoder_history_matches_same_pm25_scaling_as_encoder(beijing_data):
    """The decoder's PM2.5 history must come from the exact same MinMax
    transform as the encoder's own PM2.5 channel — not a second,
    independently-fit transform of the same variable."""
    pm25_channel = beijing_data.feature_names.index("pm25")
    encoder_pm25_last12 = beijing_data.train.encoder_input[:5, -12:, pm25_channel]
    decoder_label = beijing_data.train.decoder_input[:5, :12, 0]
    np.testing.assert_allclose(encoder_pm25_last12, decoder_label, atol=1e-6)


# 22. ENTRADAS FINALES FINITAS ----------------------------------------------------------

def test_all_final_arrays_are_finite(beijing_data):
    for split in (beijing_data.train, beijing_data.validation, beijing_data.test):
        assert np.isfinite(split.encoder_input).all()
        assert np.isfinite(split.decoder_input).all()
        assert np.isfinite(split.y_scaled).all()
        assert np.isfinite(split.y_original).all()


# 23. NÚMERO DE VENTANAS POR PARTICIÓN (auditado, no inventado) --------------------------

def test_window_counts_match_audited_values(beijing_data):
    assert beijing_data.train.n_windows == EXPECTED_N_WINDOWS["train"]
    assert beijing_data.validation.n_windows == EXPECTED_N_WINDOWS["validation"]
    assert beijing_data.test.n_windows == EXPECTED_N_WINDOWS["test"]


# include_test=False path (used by the pilot) --------------------------------------------

def test_include_test_false_leaves_test_none():
    data = prepare_beijing_r1(BEIJING_CSV, include_test=False)
    assert data.test is None
    assert data.train.n_windows == EXPECTED_N_WINDOWS["train"]
    assert data.validation.n_windows == EXPECTED_N_WINDOWS["validation"]
