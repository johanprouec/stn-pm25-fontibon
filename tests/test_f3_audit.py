"""Fase 3 — integridad de la evaluación final y de la extracción de atención.

Los tests sobre el modelo usan pesos aleatorios y entradas sintéticas (no
necesitan datos ni checkpoints). Los tests sobre resultados leen los
archivos F3_* ya generados y se omiten si no existen; el de ventanas de
Fontibón test se omite si falta el CSV institucional.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.training.transfer import CANDIDATE_V1, build_transfer_model
from src.utils.paths import FONTIBON_CSV, OUTPUTS_DIR, PROJECT_ROOT

METRICS = OUTPUTS_DIR / "metrics"
PROTOCOL = METRICS / "F3_frozen_protocol.json"
FINAL = METRICS / "F3_test_final_summary.json"
ATTENTION = METRICS / "F3_attention_matrices_summary.json"
SEEDS = [42, 123, 2026]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _f3x():
    spec = importlib.util.spec_from_file_location("f3x", PROJECT_ROOT / "scripts" / "run_f3_explainability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def random_model_and_inputs():
    rng = np.random.default_rng(0)
    enc = rng.uniform(0, 1, size=(4, 24, 3)).astype(np.float32)
    dec = np.concatenate([enc[:, -12:, :1], np.zeros((4, 1, 1), np.float32)], axis=1)
    model, _ = build_transfer_model("A", 42, enc[:2], dec[:2], architecture=CANDIDATE_V1)
    return model, tf.constant(enc), tf.constant(dec)


def test_forward_with_attention_reproduces_model_forward(random_model_and_inputs):
    model, enc, dec = random_model_and_inputs
    out, _ = _f3x().forward_with_attention(model, enc, dec)
    ref = model((enc, dec), training=False)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5)


def test_attention_shapes_finite_and_normalized(random_model_and_inputs):
    model, enc, dec = random_model_and_inputs
    _, caps = _f3x().forward_with_attention(model, enc, dec)
    w = caps["enc0_long1"]["attention_weights"].numpy()
    sel = caps["enc0_long1"]["selected_indices"].numpy()
    assert w.shape == (4, CANDIDATE_V1["num_heads"], 12, 24)
    assert sel.shape == (4, CANDIDATE_V1["num_heads"], 12)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)
    assert ((sel >= 0) & (sel < 24)).all()
    cross = caps["dec0_cross"]["attention_weights"].numpy()
    assert cross.shape == (4, CANDIDATE_V1["num_heads"], 13, 24) and np.isfinite(cross).all()


@pytest.mark.skipif(not PROTOCOL.exists(), reason="F3_frozen_protocol.json no disponible")
def test_frozen_protocol_final_model_and_threshold():
    p = _load(PROTOCOL)
    assert p["fontibon_test_touched_before_freeze"] is False
    assert p["final_model"]["config"] == "A"
    assert p["final_model"]["seeds"] == SEEDS
    assert p["metrics"]["high_episodes"]["primary_threshold_ug_m3"] == 40.0
    assert p["training_performed"] is False


@pytest.mark.skipif(not FINAL.exists(), reason="F3_test_final_summary.json no disponible")
def test_final_evaluation_was_inference_only():
    f = _load(FINAL)
    assert f["n_test_windows"] == 2574
    assert f["training_performed"] is False
    assert f["model_or_seed_changed_after_seeing_test"] is False
    assert f["aggregate_by_config"]["A"]["n_seeds"] == 3
    assert abs(f["ridge_test"]["metrics_ug_m3"]["RMSE"] - 7.063201147985151) < 1e-6


@pytest.mark.skipif(not (METRICS / "F3_ridge_test_predictions.csv").exists(), reason="predicciones F3 no disponibles")
def test_predictions_timestamps_and_targets_match_ridge():
    ridge = pd.read_csv(METRICS / "F3_ridge_test_predictions.csv")
    assert len(ridge) == 2574
    for s in SEEDS:
        a = pd.read_csv(METRICS / f"F3_transfer_A_seed{s}_test_predictions.csv")
        assert (a["target_timestamp"] == ridge["target_timestamp"]).all()
        np.testing.assert_allclose(a["y_real"], ridge["y_real"])


@pytest.mark.skipif(not FINAL.exists(), reason="F3_test_final_summary.json no disponible")
def test_high_episode_count_uses_threshold_40():
    ridge = pd.read_csv(METRICS / "F3_ridge_test_predictions.csv")
    n_high = int((ridge["y_real"] >= 40.0).sum())
    f = _load(FINAL)
    assert n_high == f["ridge_test"]["high_episodes_test"]["40.0"]["n_high"] == 118
    for s in SEEDS:
        assert f["per_run"][f"transfer_A_seed{s}"]["high_episodes_test"]["40.0"]["n_high"] == n_high


@pytest.mark.skipif(not ATTENTION.exists(), reason="F3_attention_matrices_summary.json no disponible")
def test_attention_audit_used_real_weights_of_final_model():
    a = _load(ATTENTION)
    assert a["training_performed"] is False and a["threshold_high_ug_m3"] == 40.0
    for s in SEEDS:
        v = a["seeds"][str(s)]
        assert v["attention_shape_enc0_long1"] == [2574, 4, 12, 24]
        assert v["forward_check_max_abs_diff"] < 1e-5
        assert v["pred_check_vs_saved_csv_max_abs"] < 1e-3


@pytest.mark.skipif(not FONTIBON_CSV.exists(), reason="Skipped: Fontibón institutional dataset is not distributed with the reproducible package.")
def test_fontibon_test_has_2574_windows():
    from src.data.fontibon import prepare_fontibon_stn
    data = prepare_fontibon_stn(FONTIBON_CSV, include_test=True)
    assert data.test.n_windows == 2574
