"""Reusable transfer-learning infrastructure for Config A/B/C/D
(Beijing R2 -> Fontibon). INFRASTRUCTURE ONLY — nothing in this module
trains on real data or is executed against Fontibon/Beijing by itself; it
is exercised only by tests (synthetic data, tiny architectures) until the
Fase 2 transfer experiments are explicitly authorized (see
docs/PHASE2_PLAN.md §5, §7).

Four configs, matching configs/transfer_{A,B,C,D}_*.yaml exactly:
    A — from scratch: random init, nothing frozen, single LR (1e-4).
    B — feature extractor: load R2 encoder+decoder, freeze BOTH, train
        only a fresh output_head.
    C — partial fine-tune: load R2 encoder+decoder, freeze ONLY encoder,
        train decoder + head.
    D — full fine-tune: load R2 encoder+decoder, freeze nothing, but use
        a SLOWER learning rate for the encoder (1e-5) than for
        decoder+head (1e-4).

The output_head is NEVER loaded from the R2 source checkpoint in any
config — it is always a fresh, independently-initialized layer (see
`TransferSTNModel`/`load_r2_source_into_target` below). Checkpoints are
paired strictly by seed (local Fontibon seed N <-> R2 source seed N) per
docs/IMPLEMENTATION_ASSUMPTIONS.md — never "pick the lowest-RMSE source
checkpoint for every run".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf

from src.models.stn import STNModel
from src.models.transfer_compat import compare_transfer_weight_shapes
from src.training.trainer import EarlyStoppingConfig, Trainer, TrainerConfig
from src.utils.paths import OUTPUTS_DIR
from src.utils.seeding import set_experiment_seed

# Must match configs/stn_candidate_v1.yaml and the exact architecture used
# to train the R2 source checkpoints (scripts/run_r2_source_training.py) —
# a frozen checkpoint's weights are only loadable into a model with this
# EXACT architecture. Cross-checked by
# tests/test_transfer_infrastructure.py::test_candidate_v1_matches_frozen_config_yaml.
CANDIDATE_V1 = dict(
    d_model=32, num_heads=4, d_ff=128, factor=3, dropout=0.10,
    n_encoder_units=2, n_decoder_layers=1,
    non_selected_fill="zero",
)


def r2_source_checkpoint_path(seed: int) -> Path:
    """Path to the R2 SOURCE checkpoint for a given seed — the only
    allowed way to locate a source checkpoint, so every caller agrees on
    the same path and the seed-to-seed pairing rule cannot be bypassed by
    accident (e.g. by hardcoding a single shared path, which is exactly
    the bug this corrects in configs/transfer_{B,C,D}_*.yaml — see
    docs/IMPLEMENTATION_ASSUMPTIONS.md)."""
    return OUTPUTS_DIR / "checkpoints" / f"R2_source_seed{seed}" / "best.weights.h5"


# ============================================================================
# TransferSTNModel — identical to STNModel, except frozen submodules
# (encoder and/or decoder) are forced into inference mode (training=False)
# internally, REGARDLESS of the outer `training` flag the caller passes.
# This is the only behavioral difference from STNModel — it does not
# change STNModel itself (the frozen, approved architecture file is never
# touched), it just controls how `training` propagates into submodules
# that `.trainable=False` has already excluded from the optimizer's
# gradient updates. Needed because `.trainable=False` only stops gradient
# updates; it does NOT, by itself, change Dropout's or SparseAttention's
# stochastic behavior during a forward call with training=True — without
# this override, a "frozen" encoder would still apply live Dropout masking
# and stochastic key-sampling noise to its (unchanging) weights on every
# training step, which is not the intended "frozen feature extractor"
# behavior for Config B/C.
# ============================================================================
class TransferSTNModel(STNModel):
    def __init__(self, *args, freeze_encoder: bool = False, freeze_decoder: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        if freeze_encoder:
            self.encoder.trainable = False
        if freeze_decoder:
            self.decoder.trainable = False

    def call(self, inputs, training: Optional[bool] = None) -> tf.Tensor:
        encoder_input, decoder_input = inputs
        encoder_training = False if self.freeze_encoder else training
        decoder_training = False if self.freeze_decoder else training
        encoder_output = self.encoder(encoder_input, training=encoder_training)
        decoder_output = self.decoder(decoder_input, encoder_output, training=decoder_training)
        return self.output_head(decoder_output)

    @property
    def trainable_variables(self):
        """Keras's own `tf.keras.Model.trainable_variables` (the base
        class property) was found, empirically, to NOT reliably respect
        `self.decoder.trainable = False` for `tf.keras.layers.MultiHeadAttention`'s
        internal EinsumDense kernels specifically (its own `.trainable_variables`,
        queried directly on that layer, correctly returns empty — the leak
        is only in how the top-level Model aggregates across the whole
        object graph). This is consistent with a pre-existing, documented
        quirk of this TF/Keras version's `MultiHeadAttention` handling (see
        src/models/stn.py, "MultiHeadAttention needed explicit seeding").
        Confirmed with a reproduction: freezing both encoder and decoder
        still left 8 of `decoder`'s (cross-attention) variables in
        `super().trainable_variables`.

        Fixed here by identity-filtering the base property's output against
        `self.encoder.variables` / `self.decoder.variables` (the UNGATED
        variable listing, which reliably includes every variable regardless
        of the aggregation bug) for whichever submodules are frozen. This
        property is what both the base `Trainer` (`self.model.trainable_variables`)
        and gradient computation rely on, so the fix applies uniformly to
        every config (A/B/C/D) without needing per-trainer special-casing.
        """
        frozen_var_ids = set()
        if self.freeze_encoder:
            frozen_var_ids |= {id(v) for v in self.encoder.variables}
        if self.freeze_decoder:
            frozen_var_ids |= {id(v) for v in self.decoder.variables}
        if not frozen_var_ids:
            return super().trainable_variables
        return [v for v in super().trainable_variables if id(v) not in frozen_var_ids]


@dataclass
class TransferConfig:
    name: str
    load_source: bool
    freeze_encoder: bool
    freeze_decoder: bool
    encoder_lr: Optional[float]   # None when encoder is frozen (not trained at all) or equal to decoder_head_lr
    decoder_head_lr: float


# Matches configs/transfer_{A,B,C,D}_*.yaml exactly — see that file's
# `learning_rates`/`freezing` blocks. early_stopping.patience/min_delta are
# deliberately NOT encoded here: they are still an open methodological
# decision (null/TBD in all four yaml files) that must be set identically
# across A/B/C/D before any of them runs — see docs/PHASE2_PLAN.md and the
# final report for this round.
TRANSFER_CONFIGS: Dict[str, TransferConfig] = {
    "A": TransferConfig(name="A", load_source=False, freeze_encoder=False, freeze_decoder=False, encoder_lr=None, decoder_head_lr=1e-4),
    "B": TransferConfig(name="B", load_source=True, freeze_encoder=True, freeze_decoder=True, encoder_lr=None, decoder_head_lr=1e-4),
    "C": TransferConfig(name="C", load_source=True, freeze_encoder=True, freeze_decoder=False, encoder_lr=None, decoder_head_lr=1e-4),
    "D": TransferConfig(name="D", load_source=True, freeze_encoder=False, freeze_decoder=False, encoder_lr=1e-5, decoder_head_lr=1e-4),
}


def load_r2_source_into_target(
    target_model: TransferSTNModel,
    source_checkpoint_path: Path,
    sample_encoder_input: tf.Tensor,
    sample_decoder_input: tf.Tensor,
    architecture: Dict = CANDIDATE_V1,
) -> Dict:
    """Load ONLY encoder+decoder weights from an R2 source checkpoint into
    `target_model` (already built). `target_model.output_head` is never
    touched — it keeps whatever independent initialization it already had.

    Mechanism: build a temporary, full-architecture "source-shaped" model,
    load the ENTIRE checkpoint into it (this is the only object
    `model.save_weights(...).h5` can restore into, since it was saved from
    a full STNModel), verify structural compatibility against `target_model`
    via `compare_transfer_weight_shapes` (raises if not compatible — this
    is the project's one existing, already-tested compatibility check, not
    a new one), then copy ONLY `.encoder` and `.decoder` weights across via
    `get_weights()`/`set_weights()`. The temporary source model's own
    `output_head` is discarded; never copied anywhere.
    """
    source_checkpoint_path = Path(source_checkpoint_path)
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"R2 source checkpoint not found: {source_checkpoint_path}")

    source_model = STNModel(**architecture, seed=0)  # seed is irrelevant — overwritten by load_weights below
    source_model((sample_encoder_input, sample_decoder_input), training=False)
    source_model.load_weights(str(source_checkpoint_path))

    compat = compare_transfer_weight_shapes(source_model, target_model)
    if not compat["compatible"]:
        raise ValueError(
            f"R2 source checkpoint {source_checkpoint_path} is NOT structurally compatible "
            f"with the target model — refusing to transfer. Report: {compat}"
        )

    target_model.encoder.set_weights(source_model.encoder.get_weights())
    target_model.decoder.set_weights(source_model.decoder.get_weights())

    return {
        "loaded_from": str(source_checkpoint_path),
        "encoder_loaded": True,
        "decoder_loaded": True,
        "output_head_loaded": False,  # always False — see module docstring
        "compatibility_report": compat,
    }


def build_transfer_model(
    config_name: str,
    seed: int,
    sample_encoder_input: tf.Tensor,
    sample_decoder_input: tf.Tensor,
    architecture: Dict = CANDIDATE_V1,
) -> "tuple[TransferSTNModel, Dict]":
    """Build a `TransferSTNModel` for the named config (A/B/C/D), seeded
    and built (called once with sample inputs), with R2 source weights
    loaded into encoder/decoder if the config calls for it (B/C/D) —
    always using the SAME seed's R2 checkpoint (seed-to-seed pairing,
    never the lowest-RMSE source checkpoint for every run).

    Does not train anything. Returns (model, report) where `report`
    documents exactly what was loaded/frozen, for the eventual experiment
    log.
    """
    if config_name not in TRANSFER_CONFIGS:
        raise ValueError(f"unknown transfer config {config_name!r}; must be one of {list(TRANSFER_CONFIGS)}")
    cfg = TRANSFER_CONFIGS[config_name]

    set_experiment_seed(seed)  # BEFORE construction — see src/utils/seeding.py
    model = TransferSTNModel(
        **architecture, seed=seed, freeze_encoder=cfg.freeze_encoder, freeze_decoder=cfg.freeze_decoder,
    )
    model((sample_encoder_input, sample_decoder_input), training=False)  # build

    report: Dict = {
        "config": cfg.name, "seed": seed,
        "freeze_encoder": cfg.freeze_encoder, "freeze_decoder": cfg.freeze_decoder,
        "encoder_lr": cfg.encoder_lr, "decoder_head_lr": cfg.decoder_head_lr,
        "loaded_source": False,
    }
    if cfg.load_source:
        ckpt_path = r2_source_checkpoint_path(seed)
        load_report = load_r2_source_into_target(
            model, ckpt_path, sample_encoder_input, sample_decoder_input, architecture=architecture,
        )
        report["loaded_source"] = True
        report.update(load_report)

    return model, report


# ============================================================================
# DifferentialLRTrainer — ONLY needed for Config D (encoder_lr !=
# decoder_head_lr). Config A/B/C use the base `Trainer` completely
# unmodified: freezing is a model-level property (`.trainable = False`),
# already transparent to `Trainer._train_step`, which computes gradients
# against `self.model.trainable_variables` — Keras itself excludes frozen
# submodules' variables from that list, no trainer change needed.
#
# Known, documented limitation: `Trainer.fit`'s per-epoch history row logs
# a single `"learning_rate"` field, read from `self.optimizer.learning_rate`
# — for this subclass that is the decoder/output_head group's LR (matching
# `config.learning_rate`, i.e. `decoder_head_lr`), NOT the encoder's. The
# encoder's LR is fixed for the whole run and already recorded once in the
# experiment's config/metadata, so this is a deliberate simplification, not
# a silent inaccuracy — re-overriding all of `fit()` just to add a second
# per-epoch LR column was judged not worth duplicating ~100 lines of the
# base trainer for a column that never changes epoch to epoch.
# ============================================================================
class DifferentialLRTrainer(Trainer):
    def __init__(self, model: tf.keras.Model, config: TrainerConfig, encoder_learning_rate: float):
        super().__init__(model, config)  # self.optimizer = Adam(config.learning_rate) -> decoder+head group
        self.encoder_learning_rate = encoder_learning_rate
        self.optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=encoder_learning_rate)

    @tf.function
    def _train_step(self, encoder_input, decoder_input, y):
        with tf.GradientTape() as tape:
            pred = self.model((encoder_input, decoder_input), training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))

        encoder_vars = list(self.model.encoder.trainable_variables)
        other_vars = list(self.model.decoder.trainable_variables) + list(self.model.output_head.trainable_variables)
        all_vars = encoder_vars + other_vars

        grads = tape.gradient(loss, all_vars)
        encoder_grads = grads[: len(encoder_vars)]
        other_grads = grads[len(encoder_vars):]

        encoder_pairs = [(g, v) for g, v in zip(encoder_grads, encoder_vars) if g is not None]
        other_pairs = [(g, v) for g, v in zip(other_grads, other_vars) if g is not None]
        n_missing = (len(encoder_grads) - len(encoder_pairs)) + (len(other_grads) - len(other_pairs))

        all_pairs_grads = [g for g, _ in encoder_pairs] + [g for g, _ in other_pairs]
        grad_norm = tf.linalg.global_norm(all_pairs_grads)

        if encoder_pairs:
            self.optimizer_encoder.apply_gradients(encoder_pairs)
        if other_pairs:
            self.optimizer.apply_gradients(other_pairs)

        return loss, grad_norm, n_missing


def build_trainer_for_config(config_name: str, model: tf.keras.Model, trainer_config: TrainerConfig) -> Trainer:
    """`trainer_config.learning_rate` must already be set to
    `TRANSFER_CONFIGS[config_name].decoder_head_lr` by the caller (it is
    the LR the base `Trainer.__init__` uses for its single optimizer, and
    — for config D — also the LR logged in the per-epoch history, see
    `DifferentialLRTrainer`'s docstring)."""
    cfg = TRANSFER_CONFIGS[config_name]
    if cfg.encoder_lr is not None and cfg.encoder_lr != cfg.decoder_head_lr:
        return DifferentialLRTrainer(model, trainer_config, encoder_learning_rate=cfg.encoder_lr)
    return Trainer(model, trainer_config)
