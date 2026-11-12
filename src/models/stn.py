"""Reduced Sparse Attention-based Transformer Network (STN), per Zhang &
Zhang (2023) as adapted by this project.

Architecture-level decisions here — the two-branch encoder unit, the
decoder input construction, the reduced 2-encoder/1-decoder depth for
Candidate V1 — are all documented in docs/IMPLEMENTATION_ASSUMPTIONS.md,
section "STN architecture implementation decisions", with the same
reported/promised/candidate separation used throughout this project. Do
not treat any default value in this module as an approved experiment
hyperparameter; Candidate V1 (configs/stn_candidate_v1.yaml) is explicitly
not final — see docs/PHASE2_PLAN.md §7.

No training happens in this module. It only builds the computation graph.
"""
from __future__ import annotations

from typing import List, Optional

import tensorflow as tf

from src.models.sparse_attention import SparseAttention


# ============================================================================
# Child-seed derivation — every SparseAttention instance inside the STN
# used to receive the SAME base `seed`, which meant two blocks with equal
# sequence lengths could draw identical key-sampling sequences (their
# tf.random.Generator would be seeded identically and called the same
# number of times). Each block now gets its own seed, deterministically
# derived from the model's base seed plus a stable integer offset — never
# Python's hash() (salted per-process, not reproducible run to run).
# ============================================================================
def derive_child_seed(base_seed: Optional[int], offset: int) -> Optional[int]:
    """Deterministic, process-stable derivation of a per-block seed.

    base_seed=None -> child seed=None (propagates "no reproducibility
    requested" — every block falls back to TF's global generator, as
    SparseAttention already does on its own).
    base_seed=int -> a different, but fully reproducible, child seed per
    distinct offset. Uses fixed multiplicative constants (Knuth-style),
    not Python's hash(), so the mapping is identical across processes and
    runs.
    """
    if base_seed is None:
        return None
    return (base_seed * 2654435761 + offset * 40503) % (2**31 - 1)


# Stable offset scheme (documented, not enforced by any other module):
#   encoder_unit[i]:  long_block_1 -> i*10 + 1
#                     long_block_2 -> i*10 + 2
#                     short_block  -> i*10 + 3
#   decoder_layer[i]: self_attn    -> 1000 + i*10 + 1
# The 1000 gap between encoder and decoder ranges tolerates up to ~99
# encoder units before any risk of offset collision — far beyond any
# realistic n_encoder_units for this project.
_ENCODER_UNIT_STRIDE = 10
_ENCODER_LONG_BLOCK_1_SUBOFFSET = 1
_ENCODER_LONG_BLOCK_2_SUBOFFSET = 2
_ENCODER_SHORT_BLOCK_SUBOFFSET = 3
_DECODER_OFFSET_BASE = 1000
_DECODER_LAYER_STRIDE = 10
_DECODER_SELF_ATTN_SUBOFFSET = 1
_DECODER_CROSS_ATTN_SUBOFFSET = 2


# ============================================================================
# Positional encoding — reported by the paper (sinusoidal, as in the
# original Transformer). No trainable parameters.
# ============================================================================
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)); PE(pos, 2i+1) = cos(...).

    Adds the encoding to its input (standard usage), rather than returning
    it standalone. Stateless and deterministic: the same (L, d_model) call
    always produces the same encoding, computed fresh each call from the
    input's static sequence length — no cached table, so it works
    unmodified for both the encoder's L=24 and the decoder's L=13 (or any
    other static length) without reconstruction.
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x: tf.Tensor) -> tf.Tensor:
        L = x.shape[1]
        if L is None:
            raise ValueError("SinusoidalPositionalEncoding requires a statically known sequence length")
        pe = self._build_pe(L, self.d_model, dtype=x.dtype)
        return x + pe[tf.newaxis, :, :]

    @staticmethod
    def _build_pe(seq_len: int, d_model: int, dtype=tf.float32) -> tf.Tensor:
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]        # (L, 1)
        dim_idx = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]         # (1, d_model)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(dim_idx / 2.0)) / tf.cast(d_model, tf.float32))
        angle_rads = position * angle_rates                                   # (L, d_model)
        is_odd = tf.cast(tf.range(d_model) % 2, tf.float32)                   # 0 -> sin column, 1 -> cos column
        pe = tf.sin(angle_rads) * (1.0 - is_odd) + tf.cos(angle_rads) * is_odd
        return tf.cast(pe, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


# ============================================================================
# SparseAttentionBlock — post-norm Transformer block built around
# SparseAttention. Structure reported by the paper; the ReLU activation and
# the exact feed-forward dims are our own choice (see
# docs/IMPLEMENTATION_ASSUMPTIONS.md).
# ============================================================================
class SparseAttentionBlock(tf.keras.layers.Layer):
    """Input -> SparseAttention -> Dropout -> Residual -> LayerNorm
    -> FeedForward(Dense(d_ff, relu) -> Dense(d_model)) -> Dropout
    -> Residual -> LayerNorm. Shape-preserving: (B, L, d_model) -> (B, L, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        causal: bool = False,
        causal_mode: str = "probsparse",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.causal = causal
        self.causal_mode = causal_mode

        self.attn = SparseAttention(
            d_model=d_model, num_heads=num_heads, factor=factor,
            non_selected_fill=non_selected_fill, seed=seed, name="sparse_attention",
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout, name="dropout_attn")
        self.norm1 = tf.keras.layers.LayerNormalization(name="norm_attn")

        # Feed-forward: Dense(d_ff, relu) -> Dense(d_model). ReLU is our
        # choice (conventional Transformer default), not paper-reported —
        # see docs/IMPLEMENTATION_ASSUMPTIONS.md.
        self.ffn1 = tf.keras.layers.Dense(d_ff, activation="relu", name="ffn_expand")
        self.ffn2 = tf.keras.layers.Dense(d_model, name="ffn_project")
        self.dropout2 = tf.keras.layers.Dropout(dropout, name="dropout_ffn")
        self.norm2 = tf.keras.layers.LayerNormalization(name="norm_ffn")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        attn_out = self.attn(x, causal=self.causal, causal_mode=self.causal_mode, training=training)
        x = self.norm1(x + self.dropout1(attn_out, training=training))

        ffn_out = self.ffn2(self.ffn1(x))
        x = self.norm2(x + self.dropout2(ffn_out, training=training))
        return x


# ============================================================================
# EncoderUnit — our interpretation of Fig. 3's two-branch encoder unit.
# See docs/IMPLEMENTATION_ASSUMPTIONS.md for exactly what is ambiguous in
# the paper here and how we resolved it (NOT presented as literal fact from
# the paper): the short branch takes the LAST L/2 positions of the raw
# unit input, and the two branches are concatenated along the TEMPORAL
# axis (long branch first, then short branch).
# ============================================================================
class EncoderUnit(tf.keras.layers.Layer):
    """(B, L, d_model) -> (B, L, d_model), L even.

    Long branch:  L -> SparseAttentionBlock -> Conv1D(k=3, same) ->
                  MaxPool1D(pool=2, stride=2) -> SparseAttentionBlock -> L/2
    Short branch: last L/2 positions of the unit's raw input ->
                  SparseAttentionBlock -> L/2
    Output: concat([long, short], axis=1) -> L
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        unit_index: int = 0,
        conv_kernel_size: int = 3,   # [reported by paper]
        pool_size: int = 2,           # stride [reported by paper]; pool_size == stride [our choice, operationally consistent]
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unit_index = unit_index
        base = unit_index * _ENCODER_UNIT_STRIDE

        def make_block(name, suboffset):
            return SparseAttentionBlock(
                d_model, num_heads, d_ff, factor, dropout,
                non_selected_fill=non_selected_fill,
                seed=derive_child_seed(seed, base + suboffset),
                causal=False, name=name,
            )

        self.long_block1 = make_block("long_branch_block_1", _ENCODER_LONG_BLOCK_1_SUBOFFSET)
        self.conv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=conv_kernel_size, padding="same", activation=None, name="distilling_conv1d"
        )
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_size, name="distilling_maxpool")
        self.long_block2 = make_block("long_branch_block_2", _ENCODER_LONG_BLOCK_2_SUBOFFSET)

        self.short_block = make_block("short_branch_block", _ENCODER_SHORT_BLOCK_SUBOFFSET)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        L = x.shape[1]
        if L is None or L % 2 != 0:
            raise ValueError(f"EncoderUnit requires a statically known, even sequence length, got {L}")
        half = L // 2

        long_branch = self.long_block1(x, training=training)
        long_branch = self.conv(long_branch)
        long_branch = self.pool(long_branch)         # length L/2
        long_branch = self.long_block2(long_branch, training=training)

        short_branch = self.short_block(x[:, -half:, :], training=training)  # length L/2

        return tf.concat([long_branch, short_branch], axis=1)  # length L


# ============================================================================
# STNEncoder — input projection + positional encoding + N EncoderUnits.
# Sub-layer names are stable (encoder_input_projection, encoder_unit_0, ...)
# for later transfer/freezing (Config B/C/D).
# ============================================================================
class STNEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        n_units: int,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name="encoder", **kwargs)
        self.input_projection = tf.keras.layers.Dense(d_model, name="encoder_input_projection")
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, name="encoder_positional_encoding")
        self.units: List[EncoderUnit] = [
            EncoderUnit(
                d_model, num_heads, d_ff, factor, dropout,
                non_selected_fill=non_selected_fill, seed=seed, unit_index=i, name=f"encoder_unit_{i}",
            )
            for i in range(n_units)
        ]

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for unit in self.units:
            x = unit(x, training=training)
        return x


# ============================================================================
# Decoder input construction — never sees the real future target. See
# docs/IMPLEMENTATION_ASSUMPTIONS.md, "Decoder input (anti-leakage design)".
# ============================================================================
def build_decoder_input(pm25_history_normalized: tf.Tensor, label_len: int, prediction_length: int = 1) -> tf.Tensor:
    """Build the decoder's input sequence from OBSERVED history only.

    Args:
        pm25_history_normalized: (B, lookback) or (B, lookback, 1) —
            normalized PM2.5 values for the lookback window ending at the
            current hour t (i.e. positions t-lookback+1 .. t). This
            function has no parameter for the future target y[t+1]; it
            cannot leak it because it is never given it.
        label_len: number of trailing historical values to use as "start
            tokens" (Informer-style label length).
        prediction_length: number of future placeholder positions
            (zero-filled) to append — 1 for this project's t+1 horizon.

    Returns:
        (B, label_len + prediction_length, 1): the last `label_len`
        historical values, followed by `prediction_length` exact zeros.
    """
    if pm25_history_normalized.shape.rank == 2:
        pm25_history_normalized = pm25_history_normalized[..., tf.newaxis]

    label = pm25_history_normalized[:, -label_len:, :]  # (B, label_len, 1) — history only, up to t
    batch_size = tf.shape(pm25_history_normalized)[0]
    placeholder = tf.zeros((batch_size, prediction_length, 1), dtype=pm25_history_normalized.dtype)
    return tf.concat([label, placeholder], axis=1)  # (B, label_len + prediction_length, 1)


# ============================================================================
# STNDecoderLayer — masked SparseAttention self-attention + STANDARD
# (dense) multi-head cross-attention + feed-forward. The paper specifies a
# conventional (non-sparse) attention layer here — see module docstring
# for the citation.
# ============================================================================
class STNDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        layer_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_index = layer_index

        self.self_attn = SparseAttention(
            d_model=d_model, num_heads=num_heads, factor=factor,
            non_selected_fill=non_selected_fill,
            seed=derive_child_seed(seed, _DECODER_OFFSET_BASE + layer_index * _DECODER_LAYER_STRIDE + _DECODER_SELF_ATTN_SUBOFFSET),
            name="masked_sparse_self_attention",
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout, name="dropout_self_attn")
        self.norm1 = tf.keras.layers.LayerNormalization(name="norm_self_attn")

        # Standard (dense) multi-head attention for cross-attention, per
        # the paper's own wording — NOT SparseAttention here.
        #
        # kernel_initializer is given an EXPLICIT derived seed rather than
        # relying on TF's global random state: empirically,
        # tf.keras.layers.MultiHeadAttention's internal EinsumDense
        # projection kernels do NOT reliably reinitialize the same way
        # after tf.keras.utils.set_random_seed() if a prior
        # MultiHeadAttention has already been built+used earlier in the
        # process (verified while debugging
        # tests/test_trainer.py::test_same_seed_before_construction_gives_same_initial_weights
        # — Dense/Conv1D/SparseAttention all reproduced correctly, only
        # this layer's kernels didn't). Explicit seeding sidesteps that
        # entirely and is consistent with how every other block in this
        # model is seeded (derive_child_seed).
        cross_attn_seed = derive_child_seed(seed, _DECODER_OFFSET_BASE + layer_index * _DECODER_LAYER_STRIDE + _DECODER_CROSS_ATTN_SUBOFFSET)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=cross_attn_seed),
            name="cross_attention",
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout, name="dropout_cross_attn")
        self.norm2 = tf.keras.layers.LayerNormalization(name="norm_cross_attn")

        self.ffn1 = tf.keras.layers.Dense(d_ff, activation="relu", name="ffn_expand")
        self.ffn2 = tf.keras.layers.Dense(d_model, name="ffn_project")
        self.dropout3 = tf.keras.layers.Dropout(dropout, name="dropout_ffn")
        self.norm3 = tf.keras.layers.LayerNormalization(name="norm_ffn")

    def call(self, x: tf.Tensor, encoder_output: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        self_attn_out = self.self_attn(x, causal=True, causal_mode="probsparse", training=training)
        x = self.norm1(x + self.dropout1(self_attn_out, training=training))

        cross_out = self.cross_attn(query=x, key=encoder_output, value=encoder_output, training=training)
        x = self.norm2(x + self.dropout2(cross_out, training=training))

        ffn_out = self.ffn2(self.ffn1(x))
        x = self.norm3(x + self.dropout3(ffn_out, training=training))
        return x


class STNDecoder(tf.keras.layers.Layer):
    """Input projection (1 -> d_model) + positional encoding + N STNDecoderLayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        n_layers: int,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name="decoder", **kwargs)
        self.input_projection = tf.keras.layers.Dense(d_model, name="decoder_input_projection")
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, name="decoder_positional_encoding")
        self.layers_list: List[STNDecoderLayer] = [
            STNDecoderLayer(
                d_model, num_heads, d_ff, factor, dropout,
                non_selected_fill=non_selected_fill, seed=seed, layer_index=i, name=f"decoder_layer_{i}",
            )
            for i in range(n_layers)
        ]

    def call(self, x: tf.Tensor, encoder_output: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.layers_list:
            x = layer(x, encoder_output, training=training)
        return x


# ============================================================================
# Output head — linear, last decoder position only, no bounding
# nonlinearity (output stays in normalized scale; inverse_transform to
# ug/m3 happens outside the model). Named `output_head` so Config B/C/D can
# find and (re)initialize it independently of encoder/decoder.
# ============================================================================
class OutputHead(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name="output_head", **kwargs)
        self.dense = tf.keras.layers.Dense(1, name="output_linear")

    def call(self, decoder_output: tf.Tensor) -> tf.Tensor:
        last_position = decoder_output[:, -1, :]  # (B, d_model)
        return self.dense(last_position)  # (B, 1)


# ============================================================================
# STNModel — full assembly. `n_features` is NOT hardcoded: the encoder's
# input projection infers it from whatever encoder_input is passed
# (3 for R2/Fontibon, more for R1 Beijing with one-hot wind direction).
# ============================================================================
class STNModel(tf.keras.Model):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: float,
        dropout: float,
        n_encoder_units: int,
        n_decoder_layers: int,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = STNEncoder(
            d_model, num_heads, d_ff, factor, dropout, n_encoder_units,
            non_selected_fill=non_selected_fill, seed=seed,
        )
        self.decoder = STNDecoder(
            d_model, num_heads, d_ff, factor, dropout, n_decoder_layers,
            non_selected_fill=non_selected_fill, seed=seed,
        )
        self.output_head = OutputHead()

    def call(self, inputs, training: Optional[bool] = None) -> tf.Tensor:
        """`inputs` is a (encoder_input, decoder_input) tuple:
        encoder_input (B, lookback, n_features), decoder_input
        (B, decoder_label_len + prediction_length, 1). Returns (B, 1),
        normalized-scale PM2.5 at t+1.
        """
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder(encoder_input, training=training)
        decoder_output = self.decoder(decoder_input, encoder_output, training=training)
        return self.output_head(decoder_output)
