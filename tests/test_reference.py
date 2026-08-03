"""Sanity checks on the fp32 references.

The references are ground truth for every Pallas kernel, so they get their own
tests -- a bug here would silently bless a broken kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma3_pallas.reference import attention, attention_mask, gated_mlp, gelu_tanh
from gemma3_pallas.shapes import GEMMA3_1B


def _qkv(q_len, kv_len, seed=0):
    keys = jax.random.split(jax.random.key(seed), 3)
    cfg = GEMMA3_1B
    q = jax.random.normal(keys[0], (q_len, cfg.num_heads, cfg.head_dim), jnp.float32)
    k = jax.random.normal(keys[1], (kv_len, cfg.num_kv_heads, cfg.head_dim), jnp.float32)
    v = jax.random.normal(keys[2], (kv_len, cfg.num_kv_heads, cfg.head_dim), jnp.float32)
    return q, k, v


def test_causal_mask_is_lower_triangular():
    mask = np.asarray(attention_mask(4, 4))
    assert (mask == np.tril(np.ones((4, 4), bool))).all()


def test_window_mask_is_a_band():
    mask = np.asarray(attention_mask(5, 5, window=2))
    # Query 4 may see keys 3 and 4 only.
    assert mask[4].tolist() == [False, False, False, True, True]
    # Query 0 always sees itself.
    assert mask[0].tolist() == [True, False, False, False, False]


def test_every_query_attends_to_at_least_itself():
    """A window mask that starves a row would produce NaNs in softmax."""
    mask = np.asarray(attention_mask(64, 64, window=8))
    assert mask.sum(axis=-1).min() >= 1


def test_attention_output_is_a_convex_combination_of_values():
    """Softmax rows sum to one, so constant V must pass through unchanged."""
    q, k, _ = _qkv(16, 16)
    v = jnp.ones_like(k) * 3.0
    out = attention(q, k, v)
    assert np.allclose(np.asarray(out), 3.0, atol=1e-5)


def test_window_wider_than_sequence_equals_dense_causal():
    q, k, v = _qkv(32, 32)
    dense = attention(q, k, v, window=None)
    wide = attention(q, k, v, window=1024)
    assert np.allclose(np.asarray(dense), np.asarray(wide), atol=1e-6)


def test_window_actually_changes_the_result():
    """Guards against a window argument that is silently ignored."""
    q, k, v = _qkv(64, 64)
    dense = attention(q, k, v, window=None)
    windowed = attention(q, k, v, window=8)
    assert not np.allclose(np.asarray(dense), np.asarray(windowed), atol=1e-3)


def test_mqa_broadcasts_the_single_kv_head_to_all_query_heads():
    """With 4 query heads and 1 KV head, identical queries must give identical output."""
    cfg = GEMMA3_1B
    assert cfg.num_kv_heads == 1
    q, k, v = _qkv(8, 8)
    q = jnp.broadcast_to(q[:, :1, :], q.shape)  # all heads identical
    out = np.asarray(attention(q, k, v))
    for h in range(1, cfg.num_heads):
        assert np.allclose(out[:, 0, :], out[:, h, :], atol=1e-6)


def test_attention_shape():
    cfg = GEMMA3_1B
    q, k, v = _qkv(12, 20)
    assert attention(q, k, v).shape == (12, cfg.num_heads, cfg.head_dim)


def test_gelu_tanh_matches_known_values():
    x = jnp.array([-1.0, 0.0, 1.0], jnp.float32)
    got = np.asarray(gelu_tanh(x))
    assert got[1] == pytest.approx(0.0, abs=1e-7)
    assert got[2] == pytest.approx(0.8411, abs=1e-3)
    assert got[0] == pytest.approx(-0.1588, abs=1e-3)


def test_gated_mlp_shape_and_zero_input():
    cfg = GEMMA3_1B
    t = 4
    keys = jax.random.split(jax.random.key(1), 3)
    w_gate = jax.random.normal(keys[0], (cfg.embed_dim, cfg.hidden_dim), jnp.float32) * 0.02
    w_up = jax.random.normal(keys[1], (cfg.embed_dim, cfg.hidden_dim), jnp.float32) * 0.02
    w_down = jax.random.normal(keys[2], (cfg.hidden_dim, cfg.embed_dim), jnp.float32) * 0.02

    x = jnp.zeros((t, cfg.embed_dim), jnp.float32)
    out = gated_mlp(x, w_gate, w_up, w_down)
    assert out.shape == (t, cfg.embed_dim)
    # gelu(0) == 0, so the gate closes and the block emits zeros.
    assert np.allclose(np.asarray(out), 0.0, atol=1e-6)
