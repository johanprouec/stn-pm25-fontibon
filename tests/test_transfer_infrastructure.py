"""Tests for src/training/transfer.py — the reusable Config A/B/C/D
infrastructure. INFRASTRUCTURE ONLY: every test here uses a tiny synthetic
architecture and synthetic data, never Beijing/Fontibon real data, and
none of them run a real/long training loop (at most a single manual
gradient step, to verify freezing/learning-rate wiring). A/B/C/D are not
executed against real data anywhere in this file.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml

from src.models.stn import STNModel
from src.training.trainer import TrainerConfig
from src.training.transfer import (
    CANDIDATE_V1,
    TRANSFER_CONFIGS,
    DifferentialLRTrainer,
    TransferSTNModel,
    build_trainer_for_config,
    build_transfer_model,
    load_r2_source_into_target,
    r2_source_checkpoint_path,
)
from src.utils.paths import CONFIGS_DIR, OUTPUTS_DIR
from src.utils.seeding import set_experiment_seed

# Tiny architecture, NOT CANDIDATE_V1 — only used to keep these
# infrastructure tests fast; the actual transfer experiments must use
# CANDIDATE_V1 (build_transfer_model's default), which is cross-checked
# against the frozen config/metadata separately below.
TINY_ARCH = dict(d_model=8, num_heads=2, d_ff=16, factor=2, dropout=0.0, n_encoder_units=1, n_decoder_layers=1, non_selected_fill="zero")


def _sample_inputs(n_features=3, batch=4):
    enc = tf.random.uniform((batch, 24, n_features))
    dec = tf.random.uniform((batch, 13, 1))
    return enc, dec


def _save_tiny_source_checkpoint(tmp_path, seed=1) -> Path:
    enc, dec = _sample_inputs()
    model = STNModel(**TINY_ARCH, seed=seed)
    model((enc, dec), training=False)
    ckpt_path = tmp_path / "tiny_source.weights.h5"
    model.save_weights(str(ckpt_path))
    return ckpt_path, model


# 1. ARCHITECTURE CONSISTENCY WITH THE FROZEN CONFIG ---------------------------------

def test_candidate_v1_matches_frozen_config_yaml():
    yaml_arch = yaml.safe_load((CONFIGS_DIR / "stn_candidate_v1.yaml").read_text(encoding="utf-8"))["architecture"]
    for key in ("d_model", "num_heads", "d_ff", "factor", "dropout", "n_encoder_units", "n_decoder_layers", "non_selected_fill"):
        assert CANDIDATE_V1[key] == yaml_arch[key], f"transfer.CANDIDATE_V1[{key}] drifted from configs/stn_candidate_v1.yaml"


# 2. CONFIG REGISTRY MATCHES THE 4 YAML FILES ----------------------------------------

@pytest.mark.parametrize("name,yaml_file", [
    ("A", "transfer_A_from_scratch.yaml"),
    ("B", "transfer_B_feature_extractor.yaml"),
    ("C", "transfer_C_partial_finetune.yaml"),
    ("D", "transfer_D_full_finetune.yaml"),
])
def test_transfer_config_matches_yaml_freezing_and_lr(name, yaml_file):
    spec = yaml.safe_load((CONFIGS_DIR / yaml_file).read_text(encoding="utf-8"))
    cfg = TRANSFER_CONFIGS[name]
    assert cfg.freeze_encoder == spec["freezing"]["encoder"]
    assert cfg.freeze_decoder == spec["freezing"]["decoder"]
    assert spec["freezing"]["output_head"] is False  # output_head is never frozen in any config — always trainable (fresh)
    es = spec["early_stopping"]
    assert es["patience"] == 15
    assert es["min_delta"] == 0.0
    assert es["restore_best_weights"] is True
    assert spec["training"]["max_epochs"] == 200


# 3. SOURCE CHECKPOINT PATH FOLLOWS THE CORRECTED, SEED-KEYED PATTERN --------------

def test_r2_source_checkpoint_path_is_seed_keyed():
    p42 = r2_source_checkpoint_path(42)
    p123 = r2_source_checkpoint_path(123)
    assert p42 != p123
    assert "R2_source_seed42" in str(p42)
    assert str(p42) == str(OUTPUTS_DIR / "checkpoints" / "R2_source_seed42" / "best.weights.h5")


# 4-5. LOADING SOURCE WEIGHTS: encoder/decoder copied, output_head NEVER touched ------

def test_load_r2_source_into_target_copies_encoder_decoder_not_head(tmp_path):
    ckpt_path, source_model = _save_tiny_source_checkpoint(tmp_path, seed=1)
    enc, dec = _sample_inputs()

    target_model = TransferSTNModel(**TINY_ARCH, seed=999, freeze_encoder=True, freeze_decoder=True)
    target_model((enc, dec), training=False)
    target_head_before = [w.numpy().copy() for w in target_model.output_head.trainable_variables]

    report = load_r2_source_into_target(target_model, ckpt_path, enc, dec, architecture=TINY_ARCH)

    assert report["encoder_loaded"] is True
    assert report["decoder_loaded"] is True
    assert report["output_head_loaded"] is False
    assert report["compatibility_report"]["compatible"] is True

    # encoder/decoder now match the SOURCE model exactly
    for tw, sw in zip(target_model.encoder.get_weights(), source_model.encoder.get_weights()):
        np.testing.assert_allclose(tw, sw)
    for tw, sw in zip(target_model.decoder.get_weights(), source_model.decoder.get_weights()):
        np.testing.assert_allclose(tw, sw)

    # output_head UNCHANGED by the load (still whatever it was before, not source's)
    for before, after in zip(target_head_before, target_model.output_head.trainable_variables):
        np.testing.assert_allclose(before, after.numpy())
    source_head = source_model.output_head.get_weights()
    target_head_after = target_model.output_head.get_weights()
    assert not all(np.allclose(a, b) for a, b in zip(source_head, target_head_after))


def test_load_r2_source_into_target_raises_on_incompatible_shapes(tmp_path):
    ckpt_path, _ = _save_tiny_source_checkpoint(tmp_path, seed=1)
    enc_wrong_channels, dec = _sample_inputs(n_features=11)  # R1-shaped, incompatible with a 3-channel source
    target_model = TransferSTNModel(**TINY_ARCH, seed=2)
    target_model((enc_wrong_channels, dec), training=False)

    with pytest.raises(ValueError):
        load_r2_source_into_target(target_model, ckpt_path, enc_wrong_channels, dec, architecture=TINY_ARCH)


def test_load_r2_source_into_target_missing_checkpoint_raises(tmp_path):
    enc, dec = _sample_inputs()
    target_model = TransferSTNModel(**TINY_ARCH, seed=2)
    target_model((enc, dec), training=False)
    with pytest.raises(FileNotFoundError):
        load_r2_source_into_target(target_model, tmp_path / "does_not_exist.weights.h5", enc, dec, architecture=TINY_ARCH)


# 6. OUTPUT_HEAD REINITIALIZATION IS INDEPENDENT ACROSS SEEDS -------------------------

def test_output_head_differs_across_seeds_even_with_same_loaded_source(tmp_path):
    ckpt_path, _ = _save_tiny_source_checkpoint(tmp_path, seed=1)
    enc, dec = _sample_inputs()

    model_a = TransferSTNModel(**TINY_ARCH, seed=10, freeze_encoder=True, freeze_decoder=True)
    model_a((enc, dec), training=False)
    load_r2_source_into_target(model_a, ckpt_path, enc, dec, architecture=TINY_ARCH)

    model_b = TransferSTNModel(**TINY_ARCH, seed=20, freeze_encoder=True, freeze_decoder=True)
    model_b((enc, dec), training=False)
    load_r2_source_into_target(model_b, ckpt_path, enc, dec, architecture=TINY_ARCH)

    # same source encoder/decoder in both, but DIFFERENT output_head init
    head_a = model_a.output_head.get_weights()
    head_b = model_b.output_head.get_weights()
    assert not all(np.allclose(a, b) for a, b in zip(head_a, head_b))


# 7. FROZEN WEIGHTS UNCHANGED AFTER A TRAINING STEP (Config B/C) ----------------------

def _one_training_step(model, trainer_config_kwargs, tmp_path):
    config = TrainerConfig(
        experiment_name="test_transfer_infra", seed=1, learning_rate=1e-2, batch_size=4, max_epochs=1,
        checkpoints_root=tmp_path / "checkpoints", tensorboard_root=tmp_path / "tb", metrics_root=tmp_path / "metrics",
        **trainer_config_kwargs,
    )
    from src.training.trainer import Trainer
    trainer = Trainer(model, config)
    enc, dec = _sample_inputs()
    y = tf.random.uniform((4, 1))
    trainer._train_step(enc, dec, y)
    return trainer


def test_config_B_style_frozen_encoder_and_decoder_unchanged_after_training_step(tmp_path):
    model = TransferSTNModel(**TINY_ARCH, seed=1, freeze_encoder=True, freeze_decoder=True)
    enc, dec = _sample_inputs()
    model((enc, dec), training=False)

    # Real get_weights() snapshots (not trainable_variables/.numpy() aliases) —
    # the canonical, order-stable way to compare "did this layer's weights
    # change at all", independent of how training tracks variables.
    encoder_before = [w.copy() for w in model.encoder.get_weights()]
    decoder_before = [w.copy() for w in model.decoder.get_weights()]
    head_before = [w.copy() for w in model.output_head.get_weights()]

    _one_training_step(model, {}, tmp_path)

    for before, after in zip(encoder_before, model.encoder.get_weights()):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(decoder_before, model.decoder.get_weights()):
        np.testing.assert_array_equal(before, after)
    # the only trainable part (head) SHOULD have changed
    assert not all(np.allclose(b, a) for b, a in zip(head_before, model.output_head.get_weights()))


def test_config_C_style_frozen_encoder_only_decoder_and_head_change(tmp_path):
    model = TransferSTNModel(**TINY_ARCH, seed=1, freeze_encoder=True, freeze_decoder=False)
    enc, dec = _sample_inputs()
    model((enc, dec), training=False)

    encoder_before = [w.copy() for w in model.encoder.get_weights()]
    decoder_before = [w.copy() for w in model.decoder.get_weights()]

    _one_training_step(model, {}, tmp_path)

    for before, after in zip(encoder_before, model.encoder.get_weights()):
        np.testing.assert_array_equal(before, after)
    assert not all(np.allclose(b, a) for b, a in zip(decoder_before, model.decoder.get_weights()))


def test_config_A_style_nothing_frozen():
    enc, dec = _sample_inputs()
    model = TransferSTNModel(**TINY_ARCH, seed=1, freeze_encoder=False, freeze_decoder=False)
    model((enc, dec), training=False)
    assert model.encoder.trainable is True
    assert model.decoder.trainable is True
    assert len(model.encoder.trainable_variables) > 0
    assert len(model.decoder.trainable_variables) > 0


# 8. DIFFERENTIATED LEARNING RATES FOR CONFIG D --------------------------------------

def test_differential_lr_trainer_has_two_distinct_learning_rates(tmp_path):
    model = TransferSTNModel(**TINY_ARCH, seed=1, freeze_encoder=False, freeze_decoder=False)
    enc, dec = _sample_inputs()
    model((enc, dec), training=False)

    config = TrainerConfig(
        experiment_name="test_transfer_infra_D", seed=1, learning_rate=1e-4, batch_size=4, max_epochs=1,
        checkpoints_root=tmp_path / "checkpoints", tensorboard_root=tmp_path / "tb", metrics_root=tmp_path / "metrics",
    )
    trainer = build_trainer_for_config("D", model, config)
    assert isinstance(trainer, DifferentialLRTrainer)
    assert float(trainer.optimizer_encoder.learning_rate.numpy()) == pytest.approx(1e-5)
    assert float(trainer.optimizer.learning_rate.numpy()) == pytest.approx(1e-4)

    encoder_before = [w.numpy().copy() for w in model.encoder.trainable_variables]
    y = tf.random.uniform((4, 1))
    trainer._train_step(enc, dec, y)
    # Config D freezes nothing — encoder variables must actually change too
    assert not all(np.allclose(b, v.numpy()) for b, v in zip(encoder_before, model.encoder.trainable_variables))


def test_build_trainer_for_config_ABC_use_plain_trainer_not_differential(tmp_path):
    from src.training.trainer import Trainer
    for name in ("A", "B", "C"):
        model = TransferSTNModel(**TINY_ARCH, seed=1,
                                  freeze_encoder=TRANSFER_CONFIGS[name].freeze_encoder,
                                  freeze_decoder=TRANSFER_CONFIGS[name].freeze_decoder)
        enc, dec = _sample_inputs()
        model((enc, dec), training=False)
        config = TrainerConfig(
            experiment_name=f"test_transfer_infra_{name}", seed=1, learning_rate=1e-4, batch_size=4, max_epochs=1,
            checkpoints_root=tmp_path / "checkpoints", tensorboard_root=tmp_path / "tb", metrics_root=tmp_path / "metrics",
        )
        trainer = build_trainer_for_config(name, model, config)
        assert type(trainer) is Trainer, f"config {name} should use the unmodified base Trainer"


# 9. REPRODUCIBILITY BY SEED ----------------------------------------------------------

def test_build_transfer_model_same_seed_reproduces_initial_weights():
    enc, dec = _sample_inputs()
    set_experiment_seed(123)
    model_1 = TransferSTNModel(**TINY_ARCH, seed=123)
    model_1((enc, dec), training=False)

    set_experiment_seed(123)
    model_2 = TransferSTNModel(**TINY_ARCH, seed=123)
    model_2((enc, dec), training=False)

    for w1, w2 in zip(model_1.trainable_variables, model_2.trainable_variables):
        np.testing.assert_allclose(w1.numpy(), w2.numpy())


def test_build_transfer_model_different_seed_gives_different_weights():
    enc, dec = _sample_inputs()
    set_experiment_seed(1)
    model_1 = TransferSTNModel(**TINY_ARCH, seed=1)
    model_1((enc, dec), training=False)

    set_experiment_seed(2)
    model_2 = TransferSTNModel(**TINY_ARCH, seed=2)
    model_2((enc, dec), training=False)

    assert not all(
        np.allclose(w1.numpy(), w2.numpy()) for w1, w2 in zip(model_1.trainable_variables, model_2.trainable_variables)
    )


# 10. NO ACCESS TO FONTIBON TEST FROM THE TRANSFER MODULE -----------------------------

def test_transfer_module_never_references_fontibon_test_or_include_test_true():
    source = Path("src/training/transfer.py").read_text(encoding="utf-8")
    assert "include_test=True" not in source
    assert "prepare_fontibon_stn" not in source  # transfer.py is model/weight infra only; it does not load any data itself
    assert ".test" not in source


# 11. R1 / R2 RESULTS PRESERVED (regression guard against accidental overwrite) -------

def test_r1_test_final_summary_unchanged():
    summary = json.loads((OUTPUTS_DIR / "metrics" / "R1_test_final_summary.json").read_text(encoding="utf-8"))
    agg = summary["aggregate_test_metrics_ug_m3"]
    assert agg["RMSE_mean"] == pytest.approx(21.1267, abs=0.01)
    assert agg["MAE_mean"] == pytest.approx(11.8616, abs=0.01)
    assert agg["R2_mean"] == pytest.approx(0.94865, abs=0.001)
    assert summary["n_test_windows"] == 8487


def test_r2_source_seed_comparison_unchanged():
    comparison = json.loads((OUTPUTS_DIR / "metrics" / "R2_source_seed_comparison.json").read_text(encoding="utf-8"))
    rows = {r["seed"]: r for r in comparison["rows"]}
    assert rows[42]["val_RMSE"] == pytest.approx(21.141, abs=0.01)
    assert rows[123]["val_RMSE"] == pytest.approx(20.971, abs=0.01)
    assert rows[2026]["val_RMSE"] == pytest.approx(21.068, abs=0.01)
    assert comparison["validation_stats"]["val_RMSE_mean"] == pytest.approx(21.060, abs=0.01)


@pytest.mark.skipif(
    # Solo se omite si NO hay ningún checkpoint R2 (paquete de entrega). Si existen algunos y falta
    # otro, el test se ejecuta y falla, porque eso sí sería un problema real.
    not any(r2_source_checkpoint_path(s).exists() for s in (42, 123, 2026)),
    reason="Skipped: training checkpoint is intentionally excluded from the submission package.",
)
def test_r2_source_checkpoints_still_on_disk_and_unmodified_count():
    for seed in (42, 123, 2026):
        ckpt = r2_source_checkpoint_path(seed)
        assert ckpt.exists(), f"R2 source checkpoint for seed {seed} is missing: {ckpt}"
