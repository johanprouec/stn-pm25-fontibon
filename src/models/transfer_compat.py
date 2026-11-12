"""Structural compatibility auditing between a transfer SOURCE model (R2,
Beijing, 3 channels) and a transfer TARGET model (Fontibon, 3 channels) —
verifying that ``target.encoder.set_weights(source.encoder.get_weights())``
(and the same for ``decoder``) would not raise a shape mismatch, BEFORE any
actual transfer experiment (Config A/B/C/D) is implemented.

This module only compares shapes. It never trains, transfers, or loads
weights across models — that is explicitly out of scope for this round
(see docs/PHASE2_PLAN.md).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import tensorflow as tf


def _weight_shapes(layer: tf.keras.layers.Layer) -> List[Tuple[int, ...]]:
    return [tuple(w.shape) for w in layer.get_weights()]


def compare_transfer_weight_shapes(source_model: tf.keras.Model, target_model: tf.keras.Model) -> Dict:
    """Compare `source_model.encoder`/`.decoder` weight shapes (in order)
    against `target_model`'s. Both models must already be built (called at
    least once) so their weights exist.

    Returns a dict with a top-level `compatible` bool and, for each of
    encoder/decoder, the per-index shape lists and any mismatches found.
    `output_head` is reported for reference only and does NOT affect
    `compatible` — Config B/C/D's whole point is that the output head is
    reinitialized fresh for the target domain, never transferred.
    """
    source_encoder_shapes = _weight_shapes(source_model.encoder)
    target_encoder_shapes = _weight_shapes(target_model.encoder)
    source_decoder_shapes = _weight_shapes(source_model.decoder)
    target_decoder_shapes = _weight_shapes(target_model.decoder)
    source_head_shapes = _weight_shapes(source_model.output_head)
    target_head_shapes = _weight_shapes(target_model.output_head)

    def mismatches(source_shapes, target_shapes):
        n = max(len(source_shapes), len(target_shapes))
        out = []
        for i in range(n):
            s = source_shapes[i] if i < len(source_shapes) else None
            t = target_shapes[i] if i < len(target_shapes) else None
            if s != t:
                out.append({"index": i, "source_shape": s, "target_shape": t})
        return out

    encoder_mismatches = mismatches(source_encoder_shapes, target_encoder_shapes)
    decoder_mismatches = mismatches(source_decoder_shapes, target_decoder_shapes)

    encoder_compatible = len(encoder_mismatches) == 0
    decoder_compatible = len(decoder_mismatches) == 0

    return {
        "compatible": encoder_compatible and decoder_compatible,
        "encoder_compatible": encoder_compatible,
        "decoder_compatible": decoder_compatible,
        "encoder_n_weights_source": len(source_encoder_shapes),
        "encoder_n_weights_target": len(target_encoder_shapes),
        "decoder_n_weights_source": len(source_decoder_shapes),
        "decoder_n_weights_target": len(target_decoder_shapes),
        "encoder_mismatches": encoder_mismatches,
        "decoder_mismatches": decoder_mismatches,
        "output_head_shapes_source": [list(s) for s in source_head_shapes],
        "output_head_shapes_target": [list(s) for s in target_head_shapes],
        "note": (
            "output_head shapes are informational only and are NOT part of "
            "`compatible` — the output head is always reinitialized fresh "
            "for the target domain in every transfer config, never copied "
            "from source."
        ),
    }
