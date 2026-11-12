"""Tests for src/models/stn.py — architecture-only, no real training, no
Beijing/Fontibon data (see docs/PHASE2_PLAN.md §7 and the review that
authorized this file). Values used here (d_model=32 etc.) mirror
CANDIDATE_V1 (configs/stn_candidate_v1.yaml) but are still not approved
experiment hyperparameters."""
import numpy as np
import pytest
import tensorflow as tf

tf.keras.utils.set_random_seed(4321)
tf.random.set_global_generator(tf.random.Generator.from_seed(4321))
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from src.models.sparse_attention import SparseAttention
from src.models.stn import (
    EncoderUnit,
    OutputHead,
    SinusoidalPositionalEncoding,
    SparseAttentionBlock,
    STNDecoder,
    STNDecoderLayer,
    STNEncoder,
    derive_child_seed,
    STNModel,
    build_decoder_input,
)

D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT = 32, 4, 128, 3, 0.10
N_ENCODER_UNITS, N_DECODER_LAYERS = 2, 1
LOOKBACK, LABEL_LEN, PRED_LEN = 24, 12, 1
N_FEATURES = 3


def make_model(n_features=N_FEATURES, n_encoder_units=N_ENCODER_UNITS, n_decoder_layers=N_DECODER_LAYERS, seed=None):
    return STNModel(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, factor=FACTOR, dropout=DROPOUT,
        n_encoder_units=n_encoder_units, n_decoder_layers=n_decoder_layers,
        non_selected_fill="zero", seed=seed,
    )


def make_inputs(batch=4, n_features=N_FEATURES):
    encoder_input = tf.random.normal((batch, LOOKBACK, n_features))
    pm25_history = tf.random.normal((batch, LOOKBACK))
    decoder_input = build_decoder_input(pm25_history, label_len=LABEL_LEN, prediction_length=PRED_LEN)
    return encoder_input, decoder_input


# 1. POSITIONAL ENCODING ------------------------------------------------------

@pytest.mark.parametrize("L", [24, 13])
def test_positional_encoding_shape(L):
    pe_layer = SinusoidalPositionalEncoding(D_MODEL)
    x = tf.zeros((2, L, D_MODEL))
    out = pe_layer(x)
    assert out.shape == (2, L, D_MODEL)


def test_positional_encoding_is_deterministic():
    pe_layer = SinusoidalPositionalEncoding(D_MODEL)
    x = tf.random.normal((2, 24, D_MODEL), seed=1)
    out1 = pe_layer(x)
    out2 = pe_layer(x)
    np.testing.assert_array_equal(out1.numpy(), out2.numpy())


def test_positional_encoding_has_no_trainable_params():
    pe_layer = SinusoidalPositionalEncoding(D_MODEL)
    x = tf.zeros((1, 24, D_MODEL))
    pe_layer(x)
    assert len(pe_layer.trainable_variables) == 0


def test_positional_encoding_serializable_config():
    pe_layer = SinusoidalPositionalEncoding(D_MODEL, name="pe")
    config = pe_layer.get_config()
    rebuilt = SinusoidalPositionalEncoding.from_config(config)
    assert rebuilt.d_model == D_MODEL


# 2. SPARSEATTENTIONBLOCK -----------------------------------------------------

def test_sparse_attention_block_shape():
    block = SparseAttentionBlock(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))
    out = block(x, training=False)
    assert out.shape == (2, 24, D_MODEL)


def test_sparse_attention_block_residual_matters():
    """Sanity check the block is not degenerately ignoring its input (a
    residual connection means output correlates with input, not that it's
    a trivial passthrough)."""
    block = SparseAttentionBlock(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x1 = tf.random.normal((2, 24, D_MODEL), seed=1)
    x2 = tf.random.normal((2, 24, D_MODEL), seed=2)
    out1 = block(x1, training=False)
    out2 = block(x2, training=False)
    assert not np.allclose(out1.numpy(), out2.numpy())


def test_sparse_attention_block_gradients():
    block = SparseAttentionBlock(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))
    with tf.GradientTape() as tape:
        out = block(x, training=True)
        loss = tf.reduce_sum(out ** 2)
    grads = tape.gradient(loss, block.trainable_variables)
    assert len(grads) == len(block.trainable_variables) > 0
    for var, g in zip(block.trainable_variables, grads):
        assert g is not None, f"no gradient for {var.name}"
        assert not tf.reduce_any(tf.math.is_nan(g)).numpy()


def test_sparse_attention_block_no_nan():
    block = SparseAttentionBlock(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))
    out = block(x, training=False)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


def test_sparse_attention_block_deterministic_when_not_training():
    block = SparseAttentionBlock(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))
    out1 = block(x, training=False)
    out2 = block(x, training=False)
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


# 3. ENCODER BRANCH LENGTHS ----------------------------------------------------

def test_encoder_unit_branch_lengths():
    unit = EncoderUnit(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))

    long_branch = unit.long_block1(x, training=False)
    long_branch = unit.conv(long_branch)
    long_branch = unit.pool(long_branch)
    long_branch = unit.long_block2(long_branch, training=False)
    assert long_branch.shape[1] == 12

    short_branch = unit.short_block(x[:, -12:, :], training=False)
    assert short_branch.shape[1] == 12

    concatenated = tf.concat([long_branch, short_branch], axis=1)
    assert concatenated.shape[1] == 24


# 4. ENCODER UNIT --------------------------------------------------------------

def test_encoder_unit_shape():
    unit = EncoderUnit(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 24, D_MODEL))
    out = unit(x, training=False)
    assert out.shape == (2, 24, D_MODEL)


def test_encoder_unit_requires_even_static_length():
    unit = EncoderUnit(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 23, D_MODEL))
    with pytest.raises(ValueError):
        unit(x, training=False)


# 5. ENCODER COMPLETO -----------------------------------------------------------

def test_full_encoder_two_units_shape():
    encoder = STNEncoder(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT, n_units=2)
    x = tf.random.normal((2, 24, N_FEATURES))
    out = encoder(x, training=False)
    assert out.shape == (2, 24, D_MODEL)


def test_encoder_paper_compatible_three_units_shape():
    encoder = STNEncoder(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT, n_units=3)
    x = tf.random.normal((2, 24, N_FEATURES))
    out = encoder(x, training=False)
    assert out.shape == (2, 24, D_MODEL)


def test_encoder_sublayer_names_are_stable():
    encoder = STNEncoder(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT, n_units=2)
    assert encoder.input_projection.name == "encoder_input_projection"
    assert encoder.units[0].name == "encoder_unit_0"
    assert encoder.units[1].name == "encoder_unit_1"


# 6-8. DECODER INPUT ------------------------------------------------------------

def test_decoder_input_shape():
    history = tf.random.normal((4, LOOKBACK))
    dec_in = build_decoder_input(history, label_len=LABEL_LEN, prediction_length=PRED_LEN)
    assert dec_in.shape == (4, LABEL_LEN + PRED_LEN, 1)


def test_decoder_input_last_element_is_zero():
    history = tf.random.normal((4, LOOKBACK))
    dec_in = build_decoder_input(history, label_len=LABEL_LEN, prediction_length=PRED_LEN)
    last = dec_in[:, -PRED_LEN:, :].numpy()
    np.testing.assert_array_equal(last, np.zeros_like(last))


def test_decoder_input_never_uses_future_target():
    """Construct history where every value is its own time index (0..23),
    so the last 12 (indices 12..23, i.e. t-11..t) are known values, and a
    separate 'future target' (value 999, standing in for y[t+1]) is never
    passed to the function at all — anti-leakage by construction, not by
    filtering. Verify the decoder input contains exactly [12..23, 0]."""
    history = tf.constant([[float(i) for i in range(LOOKBACK)]])  # (1, 24)
    dec_in = build_decoder_input(history, label_len=LABEL_LEN, prediction_length=PRED_LEN).numpy()[0, :, 0]

    expected = np.array(list(range(12, 24)) + [0.0])
    np.testing.assert_array_equal(dec_in, expected)

    future_target_sentinel = 999.0
    assert future_target_sentinel not in dec_in

    import inspect
    params = inspect.signature(build_decoder_input).parameters
    assert "y" not in params and "target" not in params and "future" not in params


def test_decoder_input_accepts_pre_expanded_3d_history():
    history = tf.random.normal((4, LOOKBACK, 1))
    dec_in = build_decoder_input(history, label_len=LABEL_LEN, prediction_length=PRED_LEN)
    assert dec_in.shape == (4, LABEL_LEN + PRED_LEN, 1)


# 9. DECODER LAYER ---------------------------------------------------------------

def test_decoder_layer_shape():
    layer = STNDecoderLayer(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT)
    x = tf.random.normal((2, 13, D_MODEL))
    encoder_output = tf.random.normal((2, 24, D_MODEL))
    out = layer(x, encoder_output, training=False)
    assert out.shape == (2, 13, D_MODEL)


def test_decoder_stack_shape_and_names():
    decoder = STNDecoder(D_MODEL, NUM_HEADS, D_FF, FACTOR, DROPOUT, n_layers=2)
    x = tf.random.normal((2, 13, 1))
    encoder_output = tf.random.normal((2, 24, D_MODEL))
    out = decoder(x, encoder_output, training=False)
    assert out.shape == (2, 13, D_MODEL)
    assert decoder.layers_list[0].name == "decoder_layer_0"
    assert decoder.layers_list[1].name == "decoder_layer_1"


# 10-11. STN COMPLETO -------------------------------------------------------------

def test_stn_full_forward_three_variables():
    model = make_model(n_features=3)
    encoder_input, decoder_input = make_inputs(batch=4, n_features=3)
    out = model((encoder_input, decoder_input), training=False)
    assert out.shape == (4, 1)


def test_stn_full_forward_beijing_like_channel_count():
    """R1 Beijing: 8 conceptual variables, wind direction one-hot expands
    the channel count — pick 11 as a representative post-one-hot width."""
    model = make_model(n_features=11)
    encoder_input, decoder_input = make_inputs(batch=2, n_features=11)
    out = model((encoder_input, decoder_input), training=False)
    assert out.shape == (2, 1)


# 12. NO NaN/Inf -------------------------------------------------------------------

@pytest.mark.parametrize("training", [False, True])
def test_stn_forward_no_nan_or_inf(training):
    model = make_model()
    encoder_input, decoder_input = make_inputs()
    out = model((encoder_input, decoder_input), training=training)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()
    assert not tf.reduce_any(tf.math.is_inf(out)).numpy()


# 13. BACKWARD / GRADIENTS ----------------------------------------------------------

def test_stn_all_trainable_variables_receive_finite_gradients():
    model = make_model()
    encoder_input, decoder_input = make_inputs()

    with tf.GradientTape() as tape:
        out = model((encoder_input, decoder_input), training=True)
        loss = tf.reduce_mean(tf.square(out))

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0

    missing, non_finite = [], []
    for var, g in zip(model.trainable_variables, grads):
        if g is None:
            missing.append(var.name)
        elif tf.reduce_any(tf.math.is_nan(g)).numpy() or tf.reduce_any(tf.math.is_inf(g)).numpy():
            non_finite.append(var.name)

    assert not missing, f"variables with no gradient: {missing}"
    assert not non_finite, f"variables with non-finite gradient: {non_finite}"


# 14. training=False DETERMINISTIC ----------------------------------------------------

def test_stn_deterministic_when_not_training():
    model = make_model()
    encoder_input, decoder_input = make_inputs()

    out1 = model((encoder_input, decoder_input), training=False)
    out2 = model((encoder_input, decoder_input), training=False)
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


# 15. training=True EXECUTABLE ----------------------------------------------------

def test_stn_runs_with_training_true():
    model = make_model()
    encoder_input, decoder_input = make_inputs()
    out = model((encoder_input, decoder_input), training=True)
    assert out.shape == (4, 1)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


# 16. PARAMETER COUNT -----------------------------------------------------------------

def test_stn_parameter_count_reported_and_warns_above_2m(capsys):
    model = make_model()
    encoder_input, decoder_input = make_inputs()
    model((encoder_input, decoder_input), training=False)  # build

    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"CANDIDATE_V1 total trainable parameters: {total_params}")
    if total_params > 2_000_000:
        print(f"WARNING: parameter count {total_params} exceeds 2,000,000 (advisory only, not a hard limit)")

    assert total_params > 0
    captured = capsys.readouterr()
    assert "total trainable parameters" in captured.out


# 17. SAVE / LOAD WEIGHTS -----------------------------------------------------------

def test_save_and_load_weights_reproduce_prediction(tmp_path):
    model = make_model()
    encoder_input, decoder_input = make_inputs()
    model((encoder_input, decoder_input), training=False)  # build

    out_before = model((encoder_input, decoder_input), training=False).numpy()

    weights_path = tmp_path / "stn_weights.weights.h5"
    model.save_weights(str(weights_path))

    model2 = make_model()
    model2((encoder_input, decoder_input), training=False)  # build with matching shapes
    model2.load_weights(str(weights_path))

    out_after = model2((encoder_input, decoder_input), training=False).numpy()
    np.testing.assert_allclose(out_before, out_after, atol=1e-6)


# 18. STRUCTURAL NAMES ----------------------------------------------------------------

def test_stn_has_named_encoder_decoder_output_head():
    model = make_model()
    encoder_input, decoder_input = make_inputs()
    model((encoder_input, decoder_input), training=False)  # build

    assert isinstance(model.encoder, STNEncoder)
    assert isinstance(model.decoder, STNDecoder)
    assert isinstance(model.output_head, OutputHead)
    assert model.encoder.name == "encoder"
    assert model.decoder.name == "decoder"
    assert model.output_head.name == "output_head"


# 19. CHILD-SEED STRATEGY (per-block unique, reproducible seeds) ------------------------

def _collect_sparse_attention_seeds(model: STNModel) -> dict:
    seeds = {}
    for unit in model.encoder.units:
        seeds[f"{unit.name}/long_block1"] = unit.long_block1.attn.seed
        seeds[f"{unit.name}/long_block2"] = unit.long_block2.attn.seed
        seeds[f"{unit.name}/short_block"] = unit.short_block.attn.seed
    for layer in model.decoder.layers_list:
        seeds[f"{layer.name}/self_attn"] = layer.self_attn.seed
    return seeds


def test_derive_child_seed_none_base_gives_none():
    assert derive_child_seed(None, 0) is None
    assert derive_child_seed(None, 42) is None


def test_derive_child_seed_deterministic_given_base_and_offset():
    assert derive_child_seed(42, 5) == derive_child_seed(42, 5)


def test_distinct_blocks_have_distinct_seeds():
    model = make_model(seed=42)
    encoder_input, decoder_input = make_inputs()
    model((encoder_input, decoder_input), training=False)  # build

    seeds = _collect_sparse_attention_seeds(model)
    values = list(seeds.values())
    assert all(v is not None for v in values)
    assert len(set(values)) == len(values), f"duplicate child seeds found: {seeds}"


def test_same_base_seed_reproduces_identical_child_seed_map():
    model_a = make_model(seed=123)
    model_b = make_model(seed=123)
    ei, di = make_inputs()
    model_a((ei, di), training=False)
    model_b((ei, di), training=False)

    assert _collect_sparse_attention_seeds(model_a) == _collect_sparse_attention_seeds(model_b)


def test_different_base_seed_changes_child_seeds():
    model_a = make_model(seed=123)
    model_b = make_model(seed=456)
    ei, di = make_inputs()
    model_a((ei, di), training=False)
    model_b((ei, di), training=False)

    seeds_a = _collect_sparse_attention_seeds(model_a)
    seeds_b = _collect_sparse_attention_seeds(model_b)
    assert seeds_a != seeds_b
    assert all(seeds_a[k] != seeds_b[k] for k in seeds_a)


def test_stn_deterministic_when_not_training_with_child_seeds():
    """training=False determinism (test 14) still holds once every block
    has its own distinct seed, not just with a single shared one."""
    model = make_model(seed=42)
    encoder_input, decoder_input = make_inputs()

    out1 = model((encoder_input, decoder_input), training=False)
    out2 = model((encoder_input, decoder_input), training=False)
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


def test_child_seed_scheme_does_not_change_parameter_count():
    model_seeded = make_model(seed=42)
    model_unseeded = make_model(seed=None)
    ei, di = make_inputs()
    model_seeded((ei, di), training=False)
    model_unseeded((ei, di), training=False)

    count = lambda m: sum(int(tf.size(v)) for v in m.trainable_variables)
    assert count(model_seeded) == count(model_unseeded)
