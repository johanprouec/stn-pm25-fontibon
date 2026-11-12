"""Tests for src/models/sparse_attention.py, run in isolation from any full
STN (which does not exist yet). d_model=32, heads=4, factor=3 here are
software test values, not approved experiment hyperparameters — see
docs/IMPLEMENTATION_ASSUMPTIONS.md and docs/PHASE2_PLAN.md §7."""
import numpy as np
import pytest
import tensorflow as tf

# Without this, TF's multi-threaded (oneDNN) matmul/reduction kernels can
# return run-to-run bit-level differences small enough to flip a near-tied
# top-k selection between two otherwise-identical calls, which then looks
# like a large output difference even though nothing about the model or
# the sampled keys actually changed. Must be set before the layer runs any
# ops (module import time is early enough here since this is the first
# test module pytest executes that touches TensorFlow).
tf.keras.utils.set_random_seed(1234)
# The global generator (used by SparseAttention instances constructed with
# seed=None) must exist BEFORE determinism is turned on, or TF refuses to
# lazily create it under enable_op_determinism().
tf.random.set_global_generator(tf.random.Generator.from_seed(1234))
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from src.models.sparse_attention import SparseAttention


def make_layer(d_model=32, heads=4, factor=3, seed=None, non_selected_fill="zero"):
    return SparseAttention(d_model=d_model, num_heads=heads, factor=factor, seed=seed, non_selected_fill=non_selected_fill)


# 1. SHAPE ---------------------------------------------------------------

def test_output_shape_matches_spec():
    layer = make_layer(d_model=32, heads=4, factor=3)
    x = tf.random.normal((4, 24, 32))
    out = layer(x)
    assert out.shape == (4, 24, 32)


@pytest.mark.parametrize("causal_mode", ["probsparse", "strict"])
def test_output_shape_causal(causal_mode):
    layer = make_layer(d_model=32, heads=4, factor=3)
    x = tf.random.normal((4, 24, 32))
    out = layer(x, causal=True, causal_mode=causal_mode)
    assert out.shape == (4, 24, 32)


# 2. BATCH > 1 -------------------------------------------------------------

def test_batch_independence():
    """Processing example i within a batch must give the same result as
    processing example i alone. Since the layer's RNG now advances on
    every call (item 5 of the review — no more identical-every-time
    sampling), this needs three SEPARATE, identically-seeded-and-weighted
    instances, each called exactly once, so each is compared at the same
    "first call" generator state rather than reusing one instance across
    three sequential (and therefore differently-sampled) calls."""
    d_model, heads, factor, seed = 16, 2, 3, 7
    x1 = tf.random.normal((1, 12, d_model), seed=1)
    x2 = tf.random.normal((1, 12, d_model), seed=2)
    batched = tf.concat([x1, x2], axis=0)

    tf.keras.utils.set_random_seed(321)
    layer_batched = make_layer(d_model=d_model, heads=heads, factor=factor, seed=seed)
    out_batched = layer_batched(batched)

    tf.keras.utils.set_random_seed(321)
    layer_single1 = make_layer(d_model=d_model, heads=heads, factor=factor, seed=seed)
    out1 = layer_single1(x1)

    tf.keras.utils.set_random_seed(321)
    layer_single2 = make_layer(d_model=d_model, heads=heads, factor=factor, seed=seed)
    out2 = layer_single2(x2)

    np.testing.assert_allclose(out_batched[0].numpy(), out1[0].numpy(), atol=1e-4)
    np.testing.assert_allclose(out_batched[1].numpy(), out2[0].numpy(), atol=1e-4)


# 3. MULTI-HEAD ------------------------------------------------------------

@pytest.mark.parametrize("d_model,heads", [(32, 4), (24, 6), (16, 1)])
def test_multi_head_shapes(d_model, heads):
    layer = make_layer(d_model=d_model, heads=heads, factor=3)
    x = tf.random.normal((2, 20, d_model))
    out = layer(x)
    assert out.shape == (2, 20, d_model)


def test_non_divisible_d_model_raises():
    with pytest.raises(ValueError):
        SparseAttention(d_model=30, num_heads=4, factor=3)


# 4. NO NaN / NO Inf ---------------------------------------------------------

@pytest.mark.parametrize("causal,causal_mode", [(False, "probsparse"), (True, "probsparse"), (True, "strict")])
def test_no_nan_or_inf_in_output(causal, causal_mode):
    layer = make_layer(d_model=32, heads=4, factor=3)
    x = tf.random.normal((4, 24, 32))
    out = layer(x, causal=causal, causal_mode=causal_mode)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()
    assert not tf.reduce_any(tf.math.is_inf(out)).numpy()


# 5. GRADIENTS ---------------------------------------------------------------

@pytest.mark.parametrize("causal,causal_mode", [(False, "probsparse"), (True, "probsparse"), (True, "strict")])
def test_gradients_exist_and_are_finite(causal, causal_mode):
    layer = make_layer(d_model=32, heads=4, factor=3)
    x = tf.random.normal((4, 24, 32))

    with tf.GradientTape() as tape:
        out = layer(x, causal=causal, causal_mode=causal_mode, training=True)
        loss = tf.reduce_sum(out ** 2)

    grads = tape.gradient(loss, layer.trainable_variables)

    assert len(grads) == len(layer.trainable_variables) > 0
    for var, g in zip(layer.trainable_variables, grads):
        assert g is not None, f"no gradient for {var.name}"
        assert not tf.reduce_any(tf.math.is_nan(g)).numpy(), f"NaN gradient for {var.name}"
        assert not tf.reduce_any(tf.math.is_inf(g)).numpy(), f"Inf gradient for {var.name}"


# 6. MÁSCARA CAUSAL -----------------------------------------------------------
#
# Two distinct properties, tested separately (see
# docs/IMPLEMENTATION_ASSUMPTIONS.md for why they are NOT the same thing):
#   A. The actual attention computation for a selected/used query never
#      assigns weight to a future key ("masked sparse attention" itself
#      works). True for BOTH causal_mode values.
#   B. Whether a query's OUTPUT depends on future input at all. True only
#      for causal_mode="strict" — causal_mode="probsparse" can have a
#      query's SELECTION (not its attention weights) shift due to later
#      queries' scores, a known characteristic inherited from Informer,
#      not a target-data leak (see module docstring).

def test_probsparse_causal_attention_weights_never_attend_future_keys():
    """Property A for the default causal mode: whichever queries get
    selected, their attention weight on any future key position must be
    ~0 after the causal mask + softmax."""
    d_model, heads, factor, L = 16, 2, 3, 12
    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=11)
    x = tf.random.normal((2, L, d_model), seed=42)

    _, info = layer(x, causal=True, causal_mode="probsparse", return_attention=True)
    selected_indices = info["selected_indices"].numpy()  # (B,H,n_top)
    attn = info["attention_weights"].numpy()  # (B,H,n_top,Lk)

    B, H, n_top = selected_indices.shape
    for b in range(B):
        for h in range(H):
            for n in range(n_top):
                query_pos = selected_indices[b, h, n]
                future_weight = attn[b, h, n, query_pos + 1:]
                assert np.all(future_weight < 1e-6), (
                    f"query at position {query_pos} attends to a future key "
                    f"(max weight {future_weight.max() if future_weight.size else 0})"
                )


def test_strict_causal_mode_output_independent_of_future_input():
    """Property B, for causal_mode='strict' only: a position's OUTPUT must
    not change at all when only future input values change — the property
    the plain 'probsparse' mode does not fully guarantee (see
    test_probsparse_causal_selection_can_depend_on_future_scores below)."""
    d_model, heads, factor, L = 16, 2, 3, 12
    t = L // 2

    base = tf.random.normal((1, L, d_model), seed=123)
    base_np = base.numpy().copy()
    modified_np = base_np.copy()
    modified_np[:, t + 1:, :] += 50.0
    modified = tf.constant(modified_np)

    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=99)

    out_base = layer(base, causal=True, causal_mode="strict")
    out_modified = layer(modified, causal=True, causal_mode="strict")

    np.testing.assert_allclose(
        out_base[:, : t + 1, :].numpy(),
        out_modified[:, : t + 1, :].numpy(),
        atol=1e-4,
        err_msg="causal_mode='strict' leaked future information into past positions",
    )

    # Sanity check the test itself is sensitive.
    out_base_noncausal = layer(base, causal=False)
    out_modified_noncausal = layer(modified, causal=False)
    assert not np.allclose(
        out_base_noncausal.numpy(), out_modified_noncausal.numpy(), atol=1e-4
    ), "test is not actually sensitive to the perturbation without causal masking"


def test_probsparse_causal_selection_can_depend_on_future_scores_known_limitation():
    """Documents the accepted limitation of causal_mode='probsparse'
    (matches Informer's own masked ProbSparse): because top-u selection is
    a global competition across all Lq positions, whether an early query
    is selected can shift when later input changes, even though that
    early query's own attention weights (tested above) never touch a
    future key. This is expected, not a bug — see
    docs/IMPLEMENTATION_ASSUMPTIONS.md. This test is a smoke test that the
    scenario is reachable, not a correctness requirement; it does not fail
    the suite if no flip happens to occur for this particular seed/input."""
    d_model, heads, factor, L = 16, 2, 3, 12
    t = L // 2

    base = tf.random.normal((1, L, d_model), seed=123)
    modified_np = base.numpy().copy()
    modified_np[:, t + 1:, :] += 50.0
    modified = tf.constant(modified_np)

    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=99)

    _, info_base = layer(base, causal=True, causal_mode="probsparse", return_attention=True)
    _, info_modified = layer(modified, causal=True, causal_mode="probsparse", return_attention=True)

    # No assertion on equality/inequality of selection — this is
    # documentation-by-test of the algorithm's known characteristic, not a
    # pass/fail correctness check (see docstring).
    assert info_base["selected_indices"].shape == info_modified["selected_indices"].shape


def test_causal_requires_self_attention():
    layer = make_layer(d_model=16, heads=2, factor=3)
    q = tf.random.normal((1, 10, 16))
    k = tf.random.normal((1, 14, 16))
    with pytest.raises(ValueError):
        layer((q, k, k), causal=True)


def test_invalid_causal_mode_raises():
    layer = make_layer(d_model=16, heads=2, factor=3)
    x = tf.random.normal((1, 10, 16))
    with pytest.raises(ValueError):
        layer(x, causal=True, causal_mode="not_a_real_mode")


# 7. REPRODUCIBILIDAD / training vs evaluation sampling -----------------------
#
# training=True: stochastic (stateful generator, advances between calls).
#   - Requirement C: two independently-constructed instances, seeded
#     identically, with identical weights, called with the identical
#     SEQUENCE of training=True inputs, must produce the identical
#     SEQUENCE of outputs.
#   - The SAME instance is allowed — expected — to sample DIFFERENT keys
#     across successive training=True calls (normal stochastic behavior).
# training=False (or training=None, which is treated the same):
#   - Requirement A: the same instance, same input, called repeatedly,
#     must produce EXACTLY the same output every time (no sampling noise
#     in validation/test metrics).

def test_same_seed_same_call_sequence_reproduces_across_instances_during_training():
    d_model, heads, factor, seed = 16, 2, 3, 2026
    inputs = [tf.random.normal((2, 12, d_model), seed=s) for s in (5, 6, 7)]

    tf.keras.utils.set_random_seed(555)
    layer_a = make_layer(d_model=d_model, heads=heads, factor=factor, seed=seed)
    outputs_a = [layer_a(x, training=True).numpy() for x in inputs]

    tf.keras.utils.set_random_seed(555)  # re-seed globals so Dense init matches layer_a's
    layer_b = make_layer(d_model=d_model, heads=heads, factor=factor, seed=seed)
    outputs_b = [layer_b(x, training=True).numpy() for x in inputs]

    for wa, wb in zip(layer_a.trainable_variables, layer_b.trainable_variables):
        np.testing.assert_allclose(wa.numpy(), wb.numpy(), atol=1e-6, err_msg="weights diverged between instances")

    for i, (oa, ob) in enumerate(zip(outputs_a, outputs_b)):
        np.testing.assert_allclose(oa, ob, atol=1e-5, err_msg=f"output {i} in the call sequence diverged")


def test_sampling_can_differ_between_successive_training_calls_on_same_instance():
    """When sample_k < L, repeated training=True calls on one seeded
    instance are expected to (very likely) sample different keys."""
    d_model, heads, factor, L = 16, 2, 3, 24  # sample_k = n_top = 12 < L = 24
    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=2026)
    x = tf.random.normal((1, L, d_model), seed=1)

    out1 = layer(x, training=True).numpy()
    out2 = layer(x, training=True).numpy()

    assert not np.allclose(out1, out2, atol=1e-6), (
        "consecutive training=True calls on the same instance produced "
        "identical output; expected different sampled keys (sample_k < L)"
    )


@pytest.mark.parametrize("training_value", [False, None])
def test_evaluation_sampling_is_deterministic_across_repeated_calls(training_value):
    """Requirement A: training=False (and training=None, treated the same
    — see class docstring) must give bit-for-bit-reproducible output
    across repeated calls on the same instance, unlike training=True."""
    d_model, heads, factor, L = 16, 2, 3, 24
    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=2026)
    x = tf.random.normal((1, L, d_model), seed=1)

    outputs = [layer(x, training=training_value).numpy() for _ in range(4)]
    for i, out in enumerate(outputs[1:], start=1):
        np.testing.assert_allclose(
            outputs[0], out, atol=1e-6,
            err_msg=f"evaluation call {i} differs from call 0 (training={training_value!r})",
        )


def test_training_none_matches_training_false():
    """training=None must be treated as evaluation (deterministic), not as
    an implicit True."""
    d_model, heads, factor, L = 16, 2, 3, 24
    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=2026)
    x = tf.random.normal((1, L, d_model), seed=1)

    out_none = layer(x, training=None).numpy()
    out_false = layer(x, training=False).numpy()
    np.testing.assert_allclose(out_none, out_false, atol=1e-6)


# 8. LONGITUD PEQUEÑA -----------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 2, 3])
def test_small_sequence_lengths_do_not_crash(seq_len):
    layer = make_layer(d_model=8, heads=2, factor=5)
    x = tf.random.normal((2, seq_len, 8))
    out = layer(x)
    assert out.shape == (2, seq_len, 8)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


@pytest.mark.parametrize("seq_len", [1, 2, 3])
def test_small_sequence_lengths_do_not_crash_causal(seq_len):
    layer = make_layer(d_model=8, heads=2, factor=5)
    x = tf.random.normal((2, seq_len, 8))
    out = layer(x, causal=True, causal_mode="strict")
    assert out.shape == (2, seq_len, 8)
    assert not tf.reduce_any(tf.math.is_nan(out)).numpy()


@pytest.mark.parametrize(
    "L,factor,expected",
    [(24, 5.0, 20), (24, 3.0, 12)],
)
def test_sample_k_and_n_top_exact_known_values(L, factor, expected):
    assert SparseAttention.compute_sample_k(L, factor) == expected
    assert SparseAttention.compute_n_top(L, factor) == expected


@pytest.mark.parametrize("L,factor", [(1, 5.0), (2, 10.0), (5, 8.0), (24, 3.0), (200, 5.0)])
def test_sample_k_and_n_top_never_exceed_L(L, factor):
    sample_k = SparseAttention.compute_sample_k(L, factor)
    n_top = SparseAttention.compute_n_top(L, factor)
    assert 1 <= sample_k <= L
    assert 1 <= n_top <= L


def test_sample_k_clips_when_formula_exceeds_L():
    # ceil(ln(5))=2, factor=100 -> ceil(200)=200, clipped down to L=5.
    assert SparseAttention.compute_sample_k(5, 100.0) == 5
    assert SparseAttention.compute_n_top(5, 100.0) == 5


# 9. FACTOR CONFIGURABLE ----------------------------------------------------------

@pytest.mark.parametrize("factor", [1.0, 3.0, 8.0])
def test_factor_is_configurable(factor):
    layer = make_layer(d_model=16, heads=2, factor=factor)
    x = tf.random.normal((2, 24, 16))
    out = layer(x)
    assert out.shape == (2, 24, 16)


def test_larger_factor_selects_more_top_queries():
    n_top_small = SparseAttention.compute_n_top(100, factor=1.0)
    n_top_large = SparseAttention.compute_n_top(100, factor=8.0)
    assert n_top_large > n_top_small


# Sampling structure: independent per query, shared across batch/heads -----

@pytest.mark.parametrize("stochastic", [True, False])
def test_key_sampling_shape_is_per_query_not_shared(stochastic):
    layer = make_layer(d_model=16, heads=2, factor=3, seed=3)
    Lq, Lk, sample_k = 20, 20, layer.compute_sample_k(20, 3)
    idx = layer._sample_key_indices_per_query(Lq, Lk, sample_k, restrict_causal=False, stochastic=stochastic)
    assert idx.shape == (Lq, sample_k)


@pytest.mark.parametrize("stochastic", [True, False])
def test_key_sampling_varies_across_query_rows(stochastic):
    layer = make_layer(d_model=16, heads=2, factor=3, seed=3)
    Lq, Lk, sample_k = 20, 20, layer.compute_sample_k(20, 3)
    idx = layer._sample_key_indices_per_query(Lq, Lk, sample_k, restrict_causal=False, stochastic=stochastic).numpy()
    # Not every row should be identical to row 0 (would indicate accidental
    # sharing of the sample set across query positions).
    assert not all(np.array_equal(idx[i], idx[0]) for i in range(1, Lq))


@pytest.mark.parametrize("stochastic", [True, False])
def test_strict_mode_key_sampling_respects_causal_prefix(stochastic):
    layer = make_layer(d_model=16, heads=2, factor=5, seed=3)
    Lq = Lk = 10
    sample_k = layer.compute_sample_k(Lk, 5)
    idx = layer._sample_key_indices_per_query(Lq, Lk, sample_k, restrict_causal=True, stochastic=stochastic).numpy()
    for i in range(Lq):
        assert idx[i].max() <= i, f"query {i} sampled a key beyond its own position"


# ------------------------------------------------------------------------
# Output projection bias (item 2 of the review)
# ------------------------------------------------------------------------

def test_output_projection_has_no_bias():
    layer = make_layer(d_model=16, heads=2, factor=3)
    x = tf.random.normal((1, 8, 16))
    layer(x)  # build the sublayers
    assert layer.dense.use_bias is False


def test_zero_fill_positions_produce_exact_zero_output():
    """With non_selected_fill='zero' and use_bias=False on the output
    projection, a query position that is non-selected in EVERY head must
    have an EXACTLY zero final output vector — a bias would turn the
    documented zero-fill into a non-zero value for every such position."""
    d_model, heads, factor, L = 16, 2, 3, 24  # n_top < L guarantees some non-selected positions
    layer = make_layer(d_model=d_model, heads=heads, factor=factor, seed=4, non_selected_fill="zero")
    x = tf.random.normal((1, L, d_model), seed=8)

    out, info = layer(x, training=False, return_attention=True)
    out_np = out.numpy()[0]  # (L, d_model)
    selected_in_any_head = set(info["selected_indices"].numpy()[0].flatten().tolist())  # union over heads
    never_selected = [i for i in range(L) if i not in selected_in_any_head]

    assert len(never_selected) > 0, "expected at least one position unselected in every head for this L/factor"
    for i in never_selected:
        np.testing.assert_allclose(
            out_np[i], np.zeros(d_model), atol=1e-6,
            err_msg=f"position {i} (unselected in all heads) is not exactly zero",
        )


# Default non_selected_fill ----------------------------------------------------

def test_default_non_selected_fill_is_zero():
    layer = SparseAttention(d_model=16, num_heads=2, factor=3)
    assert layer.non_selected_fill == "zero"


def test_mean_fill_is_selectable_and_differs_from_zero_fill():
    d_model, heads, factor, L = 16, 2, 3, 24  # n_top < L so some positions use fallback
    x = tf.random.normal((2, L, d_model), seed=9)

    tf.keras.utils.set_random_seed(77)
    layer_zero = SparseAttention(d_model=d_model, num_heads=heads, factor=factor, non_selected_fill="zero", seed=1)
    out_zero = layer_zero(x)

    tf.keras.utils.set_random_seed(77)
    layer_mean = SparseAttention(d_model=d_model, num_heads=heads, factor=factor, non_selected_fill="mean", seed=1)
    out_mean = layer_mean(x)

    assert not np.allclose(out_zero.numpy(), out_mean.numpy(), atol=1e-6)


# Diagnostics: return_attention=True -----------------------------------------

@pytest.mark.parametrize("causal,causal_mode", [(False, "probsparse"), (True, "probsparse")])
def test_return_attention_mode_has_selected_indices(causal, causal_mode):
    layer = make_layer(d_model=16, heads=2, factor=3)
    x = tf.random.normal((2, 20, 16))
    out, info = layer(x, causal=causal, causal_mode=causal_mode, return_attention=True)

    n_top = SparseAttention.compute_n_top(20, 3)
    assert out.shape == (2, 20, 16)
    assert info["selected_indices"].shape == (2, 2, n_top)
    assert info["attention_weights"].shape == (2, 2, n_top, 20)


def test_return_attention_mode_strict_causal_has_selected_mask():
    layer = make_layer(d_model=16, heads=2, factor=3)
    x = tf.random.normal((2, 20, 16))
    out, info = layer(x, causal=True, causal_mode="strict", return_attention=True)

    assert out.shape == (2, 20, 16)
    assert info["selected_mask"].shape == (2, 2, 20)
    assert info["attention_weights"].shape == (2, 2, 20, 20)
