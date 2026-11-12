"""Sparse (ProbSparse-style) multi-head attention, isolated from the rest of
the STN.

This implements the mechanism described in Zhang & Zhang (2023), section
"Self-attention mechanism and sparse analysis", eqs. (1)-(7): for each
query, approximate how "peaked" (non-uniform) its attention distribution
would be, using only a random sample of keys; keep only the top-u queries
by that sparsity score; compute full attention for those; leave the rest
filled in with a placeholder context value.

The paper gives the *shape* of the mechanism but not every numeric detail
(see docs/IMPLEMENTATION_ASSUMPTIONS.md, "SparseAttention implementation
decision" for the full, explicitly-sourced breakdown of what is Zhang &
Zhang, what is borrowed from Informer (Zhou et al. 2021 — cited by the
paper itself, and the clear architectural ancestor of this mechanism), and
what is our own engineering decision). This layer is deliberately generic
(no fixed d_model/heads/factor): those are architecture decisions for
src/models/stn.py, not yet made (see docs/PHASE2_PLAN.md §7).

Two causal modes are offered:
- "probsparse" (default): global top-u selection across all query
  positions, matching Zhang & Zhang / Informer's own masked self-attention
  as closely as practical. The final attention logits for selected queries
  are causally masked (no future keys), but — exactly as in Informer's own
  decoder ProbSparse — WHICH queries get selected can be influenced by
  sparsity scores of later queries, since the sampling and the top-u
  competition are not themselves restricted to a causal prefix. This is a
  characteristic of the algorithm, not a target-data leak: see
  docs/IMPLEMENTATION_ASSUMPTIONS.md for the precise distinction and why
  actual data leakage is prevented at the decoder-input level (no real
  future PM2.5, no teacher forcing with unknown future values), not by
  this layer's internal ranking.
- "strict": an experimental, non-default variant that also restricts key
  sampling and top-u competition to each query's own causal prefix,
  eliminating even the algorithmic dependency above, at extra compute
  cost. Kept for comparison; not what R1 uses.
"""
from __future__ import annotations

import math
from typing import Optional

import tensorflow as tf

NEG_INF = -1e9


class SparseAttention(tf.keras.layers.Layer):
    """Multi-head ProbSparse-style attention.

    Args:
        d_model: total model/embedding dimension (must be divisible by
            num_heads).
        num_heads: number of attention heads.
        factor: sampling/selection constant `c` used by
            :func:`compute_sample_k` and :func:`compute_n_top`. Both are
            computed as ``ceil(c * ceil(ln(L)))``, clipped to ``[1, L]`` —
            see those functions' docstrings for the exact rounding
            convention (`factor` may be a float; the two-stage ceil is
            what makes that well-defined). This is a software knob, not a
            value reported by Zhang & Zhang — see module docstring.
        non_selected_fill: what output value non-selected (non-top-u)
            query positions get. ``"zero"`` (default) matches Zhang &
            Zhang's literal wording ("filling other pairs with zero") —
            used for the R1 replication. ``"mean"`` uses the mean of V
            over all key positions (or, when ``causal=True``, a causal
            cumulative mean up to each query's own position, to avoid
            leaking future information into the fallback value) — an
            Informer-inspired alternative, kept as a configurable
            experimental option, not the default. See
            docs/IMPLEMENTATION_ASSUMPTIONS.md. Note: zero-filling most
            positions does not necessarily starve them of gradient in the
            full STN block, since the surrounding residual connections
            (paper: "a residual connection is used around each of two
            sub-layers") give every position a gradient path independent
            of this layer's own fill choice; that block does not exist yet
            here, so this layer alone cannot rely on it.
        seed: optional int. Governs both sampling modes below. When None,
            training-mode sampling uses TF's global random generator (not
            reproducible) and evaluation-mode sampling falls back to a
            fixed internal seed (still deterministic call-to-call, just
            not tied to a value the caller chose).

    Training vs. evaluation sampling (``training`` argument of ``call``):
        ``training=True``: stochastic sampling via a per-instance
            ``tf.random.Generator`` seeded from `seed`. Its state advances
            on every call, so repeated calls on the SAME instance can
            sample different keys (normal stochastic regularization
            behavior) — while two independently-constructed instances
            seeded identically and called with the same sequence of
            inputs reproduce the same sequence of outputs.
        ``training=False`` or ``training=None``: deterministic sampling
            via ``tf.random.stateless_uniform`` with a seed pair fixed at
            construction time (derived from `seed`, or a fixed fallback
            if `seed` is None). This never advances any state, so the
            same input, weights and instance always produce exactly the
            same output no matter how many times ``call`` is invoked —
            required so validation/test metrics aren't noisy from this
            layer's own internal sampling. ``training=None`` is treated
            as evaluation (deterministic), matching Keras convention that
            an unset training flag defaults to inference behavior;
            stochastic sampling only happens when a caller (or a
            surrounding ``training=True`` context propagated by Keras)
            explicitly asks for it.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        factor: float = 5.0,
        non_selected_fill: str = "zero",
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if non_selected_fill not in ("mean", "zero"):
            raise ValueError("non_selected_fill must be 'mean' or 'zero'")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.factor = factor
        self.non_selected_fill = non_selected_fill
        self.seed = seed
        # Training: stateful, advances between calls (see class docstring).
        self._rng = tf.random.Generator.from_seed(seed) if seed is not None else tf.random.get_global_generator()
        # Evaluation: fixed stateless seed pair, never advances — same
        # call always samples the same keys, so val/test metrics aren't
        # noisy from this layer's own randomness.
        self._eval_seed_pair = tf.constant([seed if seed is not None else 0, 0], dtype=tf.int32)

        self.wq = tf.keras.layers.Dense(d_model, name="query_proj")
        self.wk = tf.keras.layers.Dense(d_model, name="key_proj")
        self.wv = tf.keras.layers.Dense(d_model, name="value_proj")
        # use_bias=False: with non_selected_fill="zero", most query
        # positions carry an exact-zero context vector out of the
        # attention mechanism. A learned bias here would turn that
        # documented "zero" into a non-zero value for every position,
        # silently breaking the zero-fill semantics the paper describes.
        # Wq/Wk/Wv keep their default bias — only the layer's own output
        # projection needs this constraint.
        self.dense = tf.keras.layers.Dense(d_model, use_bias=False, name="output_proj")

    # ------------------------------------------------------------------
    # Sizing formulas — NOT specified by Zhang & Zhang (2023). Borrowed
    # from Informer's u = c * ceil(ln(L)) convention (c = `factor` here).
    #
    # Rounding convention (needed because `factor` may be a float, unlike
    # Informer's integer `c`): first take ceil(ln(L)) as an integer
    # `base`, then multiply by `factor` and ceil the product back to an
    # integer. Example: L=24, factor=5 -> ceil(ln(24))=4 -> ceil(5*4)=20.
    # L=24, factor=3 -> ceil(ln(24))=4 -> ceil(3*4)=12.
    # ------------------------------------------------------------------
    @staticmethod
    def compute_sample_k(seq_len_keys: int, factor: float) -> int:
        """Number of keys sampled per query to estimate its sparsity score."""
        base = math.ceil(math.log(seq_len_keys))
        val = math.ceil(factor * base)
        return max(1, min(val, seq_len_keys))

    @staticmethod
    def compute_n_top(seq_len_queries: int, factor: float) -> int:
        """Number of top queries kept for full attention computation."""
        base = math.ceil(math.log(seq_len_queries))
        val = math.ceil(factor * base)
        return max(1, min(val, seq_len_queries))

    def _sample_key_indices_per_query(
        self, Lq: int, Lk: int, sample_k: int, restrict_causal: bool, stochastic: bool
    ) -> tf.Tensor:
        """Per-query random key indices, shape (Lq, sample_k) — shared
        across batch and heads (as Informer's own `_prob_QK` does), but
        NOT shared across query positions: each query row gets its own
        independent sample. See docs/IMPLEMENTATION_ASSUMPTIONS.md §3.

        `restrict_causal=True` draws query i's sample only from key
        positions [0, i] (used by causal_mode="strict"); `False` draws
        from the full [0, Lk) range regardless of position (used by the
        default "probsparse" causal mode and always for non-causal,
        matching Informer's own unrestricted sampling).

        `stochastic=True` (training) draws from the stateful `self._rng`,
        which advances every call. `stochastic=False` (evaluation) draws
        from `tf.random.stateless_uniform` with the fixed `self._eval_seed_pair`,
        which never advances — see class docstring.
        """
        if restrict_causal:
            query_positions = tf.range(Lq, dtype=tf.float32)
            n_valid = query_positions + 1.0  # query i may sample from keys [0, i]
            if stochastic:
                u = self._rng.uniform([Lq, sample_k], minval=0.0, maxval=1.0)
            else:
                u = tf.random.stateless_uniform([Lq, sample_k], seed=self._eval_seed_pair, minval=0.0, maxval=1.0)
            idx = tf.cast(tf.floor(u * n_valid[:, tf.newaxis]), tf.int32)
            idx = tf.clip_by_value(idx, 0, Lk - 1)
        else:
            if stochastic:
                idx = self._rng.uniform([Lq, sample_k], minval=0, maxval=Lk, dtype=tf.int32)
            else:
                idx = tf.random.stateless_uniform(
                    [Lq, sample_k], seed=self._eval_seed_pair, minval=0, maxval=Lk, dtype=tf.int32
                )
        return idx  # (Lq, sample_k)

    def _split_heads(self, x: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        # (B, L, d_model) -> (B, H, L, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor, batch_size: tf.Tensor, seq_len: int) -> tf.Tensor:
        # (B, H, L, depth) -> (B, L, d_model)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_len, self.d_model))

    def call(
        self,
        inputs,
        mask: Optional[tf.Tensor] = None,
        causal: bool = False,
        causal_mode: str = "probsparse",
        training: Optional[bool] = None,
        return_attention: bool = False,
    ):
        """Self-attention by default (single tensor input); pass a
        ``(query, key, value)`` tuple for cross-attention.

        Args:
            inputs: a tensor of shape (B, L, d_model) for self-attention,
                or a 3-tuple (query, key, value) of tensors shaped
                (B, Lq, d_model)/(B, Lk, d_model)/(B, Lk, d_model). L must
                be statically known (fixed window sizes, as used
                throughout this project) — dynamic/None sequence-length
                dimensions are not supported, since sample_k/n_top are
                computed once per call from the static shape.
            mask: optional additive mask broadcastable to
                (B, num_heads, n_top, Lk), added to attention logits before
                softmax (e.g. for padding). Independent of `causal`.
            causal: if True, prevents a query at position i from attending
                to key positions > i. Requires self-attention (Lq == Lk).
            causal_mode: "probsparse" (default) or "strict" — see module
                docstring for the difference. Ignored when causal=False.
            return_attention: if True, also return a dict with the
                selected query indices and their attention weights
                (diagnostic use — e.g. Phase 3 explainability).

        Returns:
            Tensor of shape (B, Lq, d_model), or (output, info_dict) when
            return_attention=True.
        """
        if isinstance(inputs, (tuple, list)):
            query_in, key_in, value_in = inputs
        else:
            query_in = key_in = value_in = inputs

        batch_size = tf.shape(query_in)[0]
        Lq_static = query_in.shape[1]
        Lk_static = key_in.shape[1]
        if Lq_static is None or Lk_static is None:
            raise ValueError(
                "SparseAttention requires statically known sequence lengths "
                "(got a dynamic/None shape) because sample_k/n_top are "
                "computed once at call time from the static shape."
            )
        Lq = int(Lq_static)
        Lk = int(Lk_static)
        if causal and Lq != Lk:
            raise ValueError("causal=True requires self-attention (Lq == Lk)")
        if causal_mode not in ("probsparse", "strict"):
            raise ValueError("causal_mode must be 'probsparse' or 'strict'")

        Q = self._split_heads(self.wq(query_in), batch_size)   # (B,H,Lq,depth)
        K = self._split_heads(self.wk(key_in), batch_size)     # (B,H,Lk,depth)
        V = self._split_heads(self.wv(value_in), batch_size)   # (B,H,Lk,depth)

        scale = tf.cast(self.depth, tf.float32) ** 0.5
        stochastic = training is True  # None or False -> deterministic evaluation sampling

        if causal and causal_mode == "strict":
            context, info = self._strict_causal_branch(Q, K, V, Lq, Lk, batch_size, scale, mask, stochastic)
        else:
            context, info = self._sparse_core(Q, K, V, Lq, Lk, batch_size, scale, mask, causal=causal, stochastic=stochastic)

        output = self._merge_heads(context, batch_size, Lq)  # (B,Lq,d_model)
        output = self.dense(output)

        if return_attention:
            return output, info
        return output

    # ------------------------------------------------------------------
    # Shared core: matches Zhang & Zhang / Informer's own mechanism.
    # 1. per-query key sampling (unrestricted — see module docstring on
    #    why this is fine even when causal=True: it matches Informer's
    #    own masked/decoder ProbSparse, and actual data leakage is
    #    prevented at the decoder-input level, not here).
    # 2. sparsity score M(q_i) = max_j(score) - mean_j(score).
    # 3. GLOBAL top-u selection across all Lq queries.
    # 4. full attention for the selected queries only, causally masked
    #    when causal=True.
    # 5. fallback fill for non-selected queries (zero by default).
    # ------------------------------------------------------------------
    def _sparse_core(self, Q, K, V, Lq, Lk, batch_size, scale, mask, causal: bool, stochastic: bool):
        sample_k = self.compute_sample_k(Lk, self.factor)
        n_top = self.compute_n_top(Lq, self.factor)

        idx_per_query = self._sample_key_indices_per_query(
            Lq, Lk, sample_k, restrict_causal=False, stochastic=stochastic
        )  # (Lq, sample_k)
        K_sample = tf.gather(K, idx_per_query, axis=2)  # (B,H,Lq,sample_k,depth)
        scores_sample = tf.reduce_sum(Q[..., tf.newaxis, :] * K_sample, axis=-1) / scale  # (B,H,Lq,sample_k)
        sparsity_M = tf.reduce_max(scores_sample, axis=-1) - tf.reduce_mean(scores_sample, axis=-1)  # (B,H,Lq)

        # top-u query selection (no gradient flows through this discrete
        # choice, only through the values gathered afterwards — same as
        # Informer's ProbSparse).
        _, top_idx = tf.math.top_k(sparsity_M, k=n_top, sorted=False)  # (B,H,n_top) int32
        Q_reduced = tf.gather(Q, top_idx, axis=2, batch_dims=2)  # (B,H,n_top,depth)

        logits = tf.matmul(Q_reduced, K, transpose_b=True) / scale  # (B,H,n_top,Lk)
        if causal:
            query_pos = tf.cast(top_idx, tf.int32)[..., tf.newaxis]  # (B,H,n_top,1)
            key_pos = tf.range(Lk, dtype=tf.int32)[tf.newaxis, tf.newaxis, tf.newaxis, :]  # (1,1,1,Lk)
            causal_bool = key_pos > query_pos  # True where key is in the future -> mask out
            logits = tf.where(causal_bool, tf.fill(tf.shape(logits), tf.constant(NEG_INF, tf.float32)), logits)
        if mask is not None:
            logits = logits + mask
        attn_weights = tf.nn.softmax(logits, axis=-1)  # (B,H,n_top,Lk)
        context_selected = tf.matmul(attn_weights, V)  # (B,H,n_top,depth)

        if self.non_selected_fill == "zero":
            fill = tf.zeros_like(Q)  # (B,H,Lq,depth)
        elif causal:
            # Causal cumulative mean, not Informer's raw (undivided)
            # cumsum: normalizing keeps the fallback's magnitude
            # comparable to the plain mean(V) used non-causally, instead
            # of growing unboundedly with position. See
            # docs/IMPLEMENTATION_ASSUMPTIONS.md.
            cum_sum = tf.cumsum(V, axis=2)  # (B,H,Lk,depth), Lk == Lq here
            counts = tf.range(1, Lk + 1, dtype=tf.float32)[tf.newaxis, tf.newaxis, :, tf.newaxis]
            fill = cum_sum / counts
        else:
            fill = tf.reduce_mean(V, axis=2, keepdims=True)  # (B,H,1,depth)
            fill = tf.broadcast_to(fill, tf.shape(Q))  # (B,H,Lq,depth)

        batch_idx = tf.broadcast_to(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (batch_size, self.num_heads, n_top))
        head_idx = tf.broadcast_to(tf.reshape(tf.range(self.num_heads), (1, -1, 1)), (batch_size, self.num_heads, n_top))
        scatter_idx = tf.reshape(tf.stack([batch_idx, head_idx, top_idx], axis=-1), (-1, 3))
        updates = tf.reshape(context_selected, (-1, self.depth))
        context = tf.tensor_scatter_nd_update(fill, scatter_idx, updates)  # (B,H,Lq,depth)

        return context, {"selected_indices": top_idx, "attention_weights": attn_weights}

    # ------------------------------------------------------------------
    # EXPERIMENTAL "strict" causal mode: additionally restricts key
    # sampling and top-u competition to each query's own causal prefix,
    # so that a query's selection into (or out of) the top-u set cannot
    # be influenced by any later query's data — not just its own
    # attention output. Not the default; kept for comparison. See
    # docs/IMPLEMENTATION_ASSUMPTIONS.md, "VARIANTES EXPERIMENTALES".
    # ------------------------------------------------------------------
    def _strict_causal_branch(self, Q, K, V, Lq, Lk, batch_size, scale, mask, stochastic: bool):
        sample_k = self.compute_sample_k(Lk, self.factor)

        idx_per_query = self._sample_key_indices_per_query(
            Lq, Lk, sample_k, restrict_causal=True, stochastic=stochastic
        )  # (Lq, sample_k)
        K_sample = tf.gather(K, idx_per_query, axis=2)  # (B,H,Lq,sample_k,depth)
        scores_sample = tf.reduce_sum(Q[..., tf.newaxis, :] * K_sample, axis=-1) / scale  # (B,H,Lq,sample_k)
        sparsity_M = tf.reduce_max(scores_sample, axis=-1) - tf.reduce_mean(scores_sample, axis=-1)  # (B,H,Lq)

        # Rank query i only against queries 0..i (its own causal prefix).
        M_row = sparsity_M[..., :, tf.newaxis]      # (B,H,Lq,1)
        M_col = sparsity_M[..., tf.newaxis, :]      # (B,H,1,Lq)
        beats = M_col > M_row                        # (B,H,Lq,Lq)
        causal_valid = tf.linalg.band_part(tf.ones((Lq, Lq), dtype=tf.bool), -1, 0)  # j <= i
        rank = tf.reduce_sum(tf.cast(tf.logical_and(beats, causal_valid), tf.int32), axis=-1)  # (B,H,Lq)

        n_top_per_prefix = tf.constant(
            [self.compute_n_top(i + 1, self.factor) for i in range(Lq)], dtype=tf.int32
        )  # (Lq,) — static, safe to precompute in Python since Lq is known at call time
        selected_mask = rank < n_top_per_prefix  # (B,H,Lq) bool, causally valid by construction

        logits = tf.matmul(Q, K, transpose_b=True) / scale  # (B,H,Lq,Lk)
        key_pos = tf.range(Lk, dtype=tf.int32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
        query_pos = tf.range(Lq, dtype=tf.int32)[tf.newaxis, tf.newaxis, :, tf.newaxis]
        causal_bool = key_pos > query_pos
        logits = tf.where(causal_bool, tf.fill(tf.shape(logits), tf.constant(NEG_INF, tf.float32)), logits)
        if mask is not None:
            logits = logits + mask
        attn_weights = tf.nn.softmax(logits, axis=-1)  # (B,H,Lq,Lk)
        context_all = tf.matmul(attn_weights, V)  # (B,H,Lq,depth)

        if self.non_selected_fill == "zero":
            fill = tf.zeros_like(Q)
        else:
            cum_sum = tf.cumsum(V, axis=2)  # (B,H,Lk,depth), Lk == Lq here
            counts = tf.range(1, Lk + 1, dtype=tf.float32)[tf.newaxis, tf.newaxis, :, tf.newaxis]
            fill = cum_sum / counts

        selected_mask_f = tf.cast(selected_mask, tf.float32)[..., tf.newaxis]  # (B,H,Lq,1)
        context = selected_mask_f * context_all + (1.0 - selected_mask_f) * fill

        return context, {"selected_mask": selected_mask, "attention_weights": attn_weights}

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "factor": self.factor,
            "non_selected_fill": self.non_selected_fill,
            "seed": self.seed,
        })
        return config
